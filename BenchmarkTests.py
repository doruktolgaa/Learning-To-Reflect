import os
import re
import ast
import time
import json
import random
import logging
from typing import List, Dict, Tuple, Any, Optional

import pandas as pd
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

SEED = 42
random.seed(SEED)

BIG_M = 350000
MAX_CONTEXT_CHARS = 10000

MAIN_MODEL = "openai/gpt-5.4-nano"
JUDGE_MODEL = "openai/gpt-5.4-nano"
SINGLE_PASS_MODEL_Claude = "anthropic/claude-sonnet-4.6"
SINGLE_PASS_MODEL_Gpt = "openai/gpt-5.4"
SINGLE_PASS_MODEL_Gemini = "google/gemini-3.1-flash-lite-preview"
SINGLE_PASS_MODEL_Gptnano = MAIN_MODEL

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY bulunamadı.")

SEMEVAL_ROOT = os.path.expanduser("~/Downloads/SemEval2017")
TRAIN_RATIO = 0.8
SMALL_RUN = False
SMALL_TEST_SAMPLES = 10

DESKTOP_DIR = os.path.expanduser("~/Desktop")
BENCHMARK_DIR = os.path.join(DESKTOP_DIR, "nerd_benchmark_reports")
os.makedirs(BENCHMARK_DIR, exist_ok=True)

BENCHMARK_LOG_FILE = os.path.join(BENCHMARK_DIR, "benchmark.log")
BENCHMARK_SUMMARY_CSV = os.path.join(BENCHMARK_DIR, "benchmark_summary.csv")
BENCHMARK_SUMMARY_JSON = os.path.join(BENCHMARK_DIR, "benchmark_summary.json")
BENCHMARK_COMBINED_SAMPLE_CSV = os.path.join(BENCHMARK_DIR, "all_methods_sample_summary.csv")

ACTIONS = [
    "STOP",
    "LIGHT_EDIT",
    "ADD_MISSING",
    "REMOVE_UNSUPPORTED",
    "REPLACE_WEAKEST",
    "REGENERATE",
]

ACTION_MIN_CHANGES = {
    "LIGHT_EDIT": 1,
    "ADD_MISSING": 1,
    "REMOVE_UNSUPPORTED": 1,
    "REPLACE_WEAKEST": 1,
    "REGENERATE": 1,
}

_EMPTY_CHANGED = {
    "changed_term_count": 0,
    "added_terms": [],
    "removed_terms": [],
    "added_words": [],
    "removed_words": [],
    "replacements": [],
}


# ============================================================
# LOGGER
# ============================================================

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("keyword_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.FileHandler(BENCHMARK_LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


logger = setup_logger()


# ============================================================
# GENERAL HELPERS
# ============================================================

def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"REPORT SAVED: {path}")

def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON SAVED: {path}")

def log_dataframe(title: str, df: pd.DataFrame, max_rows: int = 100) -> None:
    logger.info("=" * 120)
    logger.info(title)
    logger.info("=" * 120)
    if df.empty:
        logger.info("EMPTY DATAFRAME")
    else:
        logger.info("\n" + df.head(max_rows).to_string(index=False))

def to_serializable_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: (v if isinstance(v, (float, int, str, bool)) or v is None else str(v))
        for k, v in d.items()
    }

def get_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[:max_chars]


# ============================================================
# TEXT / KEYWORD HELPERS
# ============================================================

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def normalize_keyword(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[_/]", " ", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokenize_for_overlap(text: str) -> List[str]:
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []

def parse_keywords(value) -> List[str]:
    if isinstance(value, list):
        return [normalize_text(x) for x in value if normalize_text(x)]

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    s = str(value).strip()

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [normalize_text(x) for x in parsed if normalize_text(x)]
    except Exception:
        pass

    for sep in ["\n", ";", ",", "|"]:
        if sep in s:
            return [normalize_text(x) for x in s.split(sep) if normalize_text(x)]

    return [normalize_text(s)] if s else []

def deduplicate_keywords(keywords: List[str]) -> List[str]:
    seen = set()
    out = []
    for kw in keywords:
        nk = normalize_keyword(kw)
        if nk and nk not in seen:
            seen.add(nk)
            out.append(normalize_text(kw))
    return out

def keywords_to_text(keywords: List[str]) -> str:
    return ", ".join(keywords)

def parse_model_keywords(output: str, target_count: int) -> List[str]:
    text = normalize_text(output)

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return deduplicate_keywords(
                    [normalize_text(x) for x in parsed if normalize_text(x)]
                )[:target_count]
        except Exception:
            pass

    text = re.sub(r"^(keywords?|keyphrases?)\s*:\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(r"\n|;|,|\|", text)
    cleaned = [
        normalize_text(re.sub(r"^\s*[-*•\d\.\)]*\s*", "", p))
        for p in parts
    ]
    return deduplicate_keywords([x for x in cleaned if x])[:target_count]

def keyword_in_text(keyword: str, text: str) -> bool:
    return normalize_keyword(keyword) in normalize_text(text).lower()

def keyword_token_overlap(pred_kw: str, gold_kw: str) -> float:
    p = set(tokenize_for_overlap(pred_kw))
    g = set(tokenize_for_overlap(gold_kw))
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec = inter / len(p)
    rec = inter / len(g)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

def match_keywords(pred_keywords: List[str], gold_keywords: List[str], threshold: float = 0.8):
    pred_norm = [normalize_keyword(x) for x in pred_keywords]
    gold_norm = [normalize_keyword(x) for x in gold_keywords]

    matched_pred = set()
    matched_gold = set()

    for i, pk in enumerate(pred_norm):
        if not pk:
            continue
        for j, gk in enumerate(gold_norm):
            if j in matched_gold:
                continue
            if pk == gk:
                matched_pred.add(i)
                matched_gold.add(j)
                break

    for i, pk in enumerate(pred_norm):
        if i in matched_pred or not pk:
            continue
        for j, gk in enumerate(gold_norm):
            if j in matched_gold or not gk:
                continue
            if keyword_token_overlap(pk, gk) >= threshold:
                matched_pred.add(i)
                matched_gold.add(j)
                break

    return matched_pred, matched_gold

def precision_recall_f1(pred_keywords: List[str], gold_keywords: List[str]) -> Dict[str, float]:
    pred_keywords = deduplicate_keywords(pred_keywords)
    gold_keywords = deduplicate_keywords(gold_keywords)

    if not pred_keywords and not gold_keywords:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0}
    if not pred_keywords or not gold_keywords:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0}

    matched_pred, matched_gold = match_keywords(pred_keywords, gold_keywords)
    tp = len(matched_pred)
    prec = tp / len(pred_keywords)
    rec = tp / len(gold_keywords)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "matched": tp,
    }

def present_recall(text: str, pred_keywords: List[str], gold_keywords: List[str]) -> float:
    present_gold = [g for g in gold_keywords if keyword_in_text(g, text)]
    if not present_gold:
        return 1.0
    _, matched_gold = match_keywords(pred_keywords, present_gold)
    return len(matched_gold) / len(present_gold)

def avg_keyword_token_overlap(pred_keywords: List[str], gold_keywords: List[str]) -> float:
    pred_keywords = deduplicate_keywords(pred_keywords)
    gold_keywords = deduplicate_keywords(gold_keywords)
    if not pred_keywords or not gold_keywords:
        return 0.0
    return sum(
        max(keyword_token_overlap(pk, gk) for gk in gold_keywords)
        for pk in pred_keywords
    ) / len(pred_keywords)

def duplicate_penalty(pred_keywords: List[str]) -> float:
    if not pred_keywords:
        return 0.0
    normed = [normalize_keyword(x) for x in pred_keywords if normalize_keyword(x)]
    return min((len(normed) - len(set(normed))) * 0.10, 0.30)

def unsupported_penalty(text: str, pred_keywords: List[str]) -> float:
    if not pred_keywords:
        return 0.0
    unsup = sum(1 for kw in pred_keywords if not keyword_in_text(kw, text))
    return min((unsup / max(len(pred_keywords), 1)) * 0.25, 0.25)

def count_penalty(pred_keywords: List[str], target_count: int) -> float:
    return min(abs(len(pred_keywords) - target_count) * 0.04, 0.20)

def compute_duplicate_ratio(pred_keywords: List[str]) -> float:
    if not pred_keywords:
        return 0.0
    normed = [normalize_keyword(x) for x in pred_keywords if normalize_keyword(x)]
    if not normed:
        return 0.0
    return min((len(normed) - len(set(normed))) / len(normed), 1.0)

def compute_unsupported_ratio(text: str, pred_keywords: List[str]) -> float:
    if not pred_keywords:
        return 0.0
    return sum(1 for kw in pred_keywords if not keyword_in_text(kw, text)) / len(pred_keywords)

def compute_quality(text: str, pred_keywords: List[str], gold_keywords: List[str]) -> Dict[str, float]:
    gold_count = len(deduplicate_keywords(gold_keywords))
    pred_keywords = deduplicate_keywords(pred_keywords)
    gold_keywords = deduplicate_keywords(gold_keywords)

    base = precision_recall_f1(pred_keywords, gold_keywords)
    p = base["precision"]
    r = base["recall"]
    f1 = base["f1"]
    pres_r = present_recall(text, pred_keywords, gold_keywords)
    overlap = avg_keyword_token_overlap(pred_keywords, gold_keywords)
    dup_pen = duplicate_penalty(pred_keywords)
    unsup_pen = unsupported_penalty(text, pred_keywords)
    cnt_pen = count_penalty(pred_keywords, gold_count)

    quality = (
        0.42 * f1 +
        0.20 * p +
        0.20 * r +
        0.10 * pres_r +
        0.08 * overlap -
        0.02 * dup_pen -
        0.01 * unsup_pen -
        0.01 * cnt_pen
    )

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "present_recall": pres_r,
        "avg_overlap": overlap,
        "duplicate_penalty": dup_pen,
        "unsupported_penalty": unsup_pen,
        "count_penalty": cnt_pen,
        "quality": quality,
        "pred_count": len(pred_keywords),
        "gold_count": gold_count,
        "matched": base["matched"],
    }

def metrics_delta(old: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    return {
        f"delta_{k}": new[k] - old.get(k, 0.0)
        for k in new
        if isinstance(new[k], (int, float))
    }

def keyword_change_count(old_keywords: List[str], new_keywords: List[str]) -> int:
    old_norm = {normalize_keyword(x) for x in old_keywords if normalize_keyword(x)}
    new_norm = {normalize_keyword(x) for x in new_keywords if normalize_keyword(x)}
    return max(len(old_norm - new_norm), len(new_norm - old_norm))

def diff_changed_terms(old_keywords: List[str], new_keywords: List[str]) -> Dict[str, Any]:
    old_map = {normalize_keyword(x): x for x in old_keywords if normalize_keyword(x)}
    new_map = {normalize_keyword(x): x for x in new_keywords if normalize_keyword(x)}

    removed = [old_map[n] for n in set(old_map) - set(new_map)]
    added = [new_map[n] for n in set(new_map) - set(old_map)]
    replacements = [
        {"position": i, "old": r, "new": a}
        for i, (r, a) in enumerate(zip(removed, added))
    ]

    old_words = {w for kw in old_keywords for w in tokenize_for_overlap(kw)}
    new_words = {w for kw in new_keywords for w in tokenize_for_overlap(kw)}

    return {
        "added_terms": added,
        "removed_terms": removed,
        "replacements": replacements,
        "added_words": sorted(new_words - old_words),
        "removed_words": sorted(old_words - new_words),
        "changed_term_count": keyword_change_count(old_keywords, new_keywords),
    }


# ============================================================
# DATASET
# ============================================================

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return normalize_text(f.read())

def read_key_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    lines = [normalize_text(x) for x in raw.splitlines() if normalize_text(x)]
    return deduplicate_keywords(lines if lines else parse_keywords(raw))

def load_semeval2010_examples(root_dir: str) -> List[dict]:
    docs_dir = os.path.join(root_dir, "docsutf8")
    keys_dir = os.path.join(root_dir, "keys")

    if not os.path.isdir(docs_dir):
        raise ValueError(f"docsutf8 not found: {docs_dir}")
    if not os.path.isdir(keys_dir):
        raise ValueError(f"keys not found: {keys_dir}")

    examples = []
    for doc_file in sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".txt")):
        doc_id = os.path.splitext(doc_file)[0]

        key_path = next(
            (
                p for p in [
                    os.path.join(keys_dir, doc_id + ".txt"),
                    os.path.join(keys_dir, doc_id + ".key"),
                ]
                if os.path.exists(p)
            ),
            None,
        )

        if key_path is None:
            continue

        text = read_text_file(os.path.join(docs_dir, doc_file))
        gold = deduplicate_keywords(read_key_file(key_path))

        if len(text.split()) < 40 or len(gold) < 3:
            continue

        examples.append({
            "id": doc_id,
            "text": text,
            "gold_keywords": gold,
        })

    if not examples:
        raise ValueError("No examples found.")

    return examples

def build_train_test_split(
    examples: List[dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> Tuple[List[dict], List[dict]]:
    examples = list(examples)
    rng = random.Random(seed)
    rng.shuffle(examples)

    split = max(1, min(int(len(examples) * train_ratio), len(examples) - 1))
    return examples[:split], examples[split:]


# ============================================================
# OPENROUTER MODEL
# ============================================================

class OpenRouterLLM:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = BIG_M,
        retries: int = 3,
        system_prompt: Optional[str] = None,
    ) -> str:
        sys_prompt = system_prompt or "You are a careful assistant."
        last_err = None

        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                last_err = e
                logger.info(f"MODEL CALL FAILED | attempt={attempt+1}/{retries} | error={e}")
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"OpenRouter call failed: {last_err}")


# ============================================================
# JUDGE
# ============================================================

JUDGE_SYSTEM_PROMPT = """
You are a strict and calibrated keyword extraction judge.
Your scores are used as evaluation signals, so calibration matters.

CRITICAL SCORING RULES:
- Scores must be precise floats with 2 decimal places
- Avoid 0.00 and 1.00 unless evidence is overwhelming
- Most real keyword lists should score between 0.30 and 0.75

Evaluate these dimensions:
- groundedness
- coverage
- specificity
- conciseness
- uniqueness
- count_fit
- duplicate_risk
- unsupported_risk

Return ONLY valid JSON.
""".strip()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def normalize_guideline_list(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    out = []
    seen = set()
    for x in items:
        s = normalize_text(x)
        nk = normalize_keyword(s)
        if s and nk and nk not in seen:
            seen.add(nk)
            out.append(s)
    return out[:8]

def extract_json_block(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip()).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:idx + 1])
                    except Exception:
                        break

        fragment = text[start:]
        try:
            fragment = re.sub(r',\s*"[^"]*$', "", fragment)
            fragment = re.sub(r':\s*"[^"]*$', ': ""', fragment)
            open_arrays = fragment.count("[") - fragment.count("]")
            fragment += "]" * max(open_arrays, 0)
            open_braces = fragment.count("{") - fragment.count("}")
            fragment += "}" * max(open_braces, 0)
            return json.loads(fragment)
        except Exception:
            pass

    raise ValueError(f"Judge JSON parse failed: {text[:500]}")

class Judge:
    def __init__(self, llm: OpenRouterLLM):
        self.llm = llm

    def evaluate(self, text: str, keywords: List[str], target_count: int) -> Dict[str, Any]:
        context = get_context(text)

        dup_ratio = compute_duplicate_ratio(keywords)
        unsup_ratio = compute_unsupported_ratio(text, keywords)
        count_diff = abs(len(keywords) - target_count)

        prompt = f"""
Evaluate the keyword list below for the given document.

DOCUMENT:
{context}

KEYWORD LIST ({len(keywords)} keywords, target={target_count}):
{keywords_to_text(keywords)}

OBJECTIVE MEASUREMENTS:
- Duplicate ratio: {dup_ratio:.3f}
- Unsupported ratio: {unsup_ratio:.3f}
- Count difference from target: {count_diff}

Return ONLY this JSON schema:
{{
  "overall_score": 0.00,
  "groundedness": 0.00,
  "coverage": 0.00,
  "specificity": 0.00,
  "conciseness": 0.00,
  "uniqueness": 0.00,
  "count_fit": 0.00,
  "duplicate_risk": 0.00,
  "unsupported_risk": 0.00,
  "salvageable_with_small_edits": true,
  "edit_pressure": "medium",
  "major_failures": ["..."],
  "weak_keywords": ["..."],
  "missing_concepts": ["..."],
  "editing_guidelines": [
    "Prefer short canonical keyphrases",
    "Prefer phrases explicitly supported by the document"
  ],
  "reason_short": "..."
}}
""".strip()

        raw = self.llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=BIG_M,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )

        try:
            parsed = extract_json_block(raw)
        except ValueError:
            logger.warning(f"Judge JSON parse failed, fallback used. raw tail: {raw[-300:]}")
            parsed = {}

        judge = {
            "overall_score": min(max(safe_float(parsed.get("overall_score", 0.0)), 0.0), 1.0),
            "groundedness": min(max(safe_float(parsed.get("groundedness", 0.0)), 0.0), 1.0),
            "coverage": min(max(safe_float(parsed.get("coverage", 0.0)), 0.0), 1.0),
            "specificity": min(max(safe_float(parsed.get("specificity", 0.0)), 0.0), 1.0),
            "conciseness": min(max(safe_float(parsed.get("conciseness", 0.0)), 0.0), 1.0),
            "uniqueness": min(max(safe_float(parsed.get("uniqueness", 0.0)), 0.0), 1.0),
            "count_fit": min(max(safe_float(parsed.get("count_fit", 0.0)), 0.0), 1.0),
            "duplicate_risk": min(max(safe_float(parsed.get("duplicate_risk", 0.0)), 0.0), 1.0),
            "unsupported_risk": min(max(safe_float(parsed.get("unsupported_risk", 0.0)), 0.0), 1.0),
            "salvageable_with_small_edits": bool(parsed.get("salvageable_with_small_edits", True)),
            "edit_pressure": normalize_text(parsed.get("edit_pressure", "medium")).lower() or "medium",
            "major_failures": [normalize_text(x) for x in parsed.get("major_failures", []) if normalize_text(x)],
            "weak_keywords": [normalize_text(x) for x in parsed.get("weak_keywords", []) if normalize_text(x)],
            "missing_concepts": [normalize_text(x) for x in parsed.get("missing_concepts", []) if normalize_text(x)],
            "editing_guidelines": normalize_guideline_list(parsed.get("editing_guidelines", [])),
            "reason_short": normalize_text(parsed.get("reason_short", "")),
        }

        if judge["edit_pressure"] not in {"low", "medium", "high"}:
            judge["edit_pressure"] = "medium"

        if not judge["editing_guidelines"]:
            judge["editing_guidelines"] = [
                "Prefer short document-grounded keyphrases",
                "Remove unsupported or weak terms",
                "Avoid duplicates and near-duplicates",
            ]

        return judge


# ============================================================
# PROMPTS
# ============================================================

EDITOR_SYSTEM_PROMPT = """
You are a careful keyword extraction editor.
Output only a comma-separated keyword list.
Do not explain. Do not add numbering or bullets.
Follow the action constraints strictly.
""".strip()

def initial_prompt(text: str, target_count: int) -> str:
    context = get_context(text)
    return f"""
Extract {target_count} keyphrases from the document.
Output only a comma-separated list.

Requirements:
- Use concrete, document-grounded keyphrases
- Avoid generic one-word filler terms
- Prefer canonical noun phrases
- Avoid duplicates and near-duplicates

Document:
{context}
""".strip()

def format_editing_guidelines(guidelines: List[str]) -> str:
    if not guidelines:
        return (
            "- Prefer short document-grounded keyphrases\n"
            "- Remove unsupported or weak terms\n"
            "- Avoid duplicates and near-duplicates"
        )
    return "\n".join(f"- {g}" for g in guidelines)

def format_reflection_history_brief(reflection_history: List[Dict[str, Any]], last_k: int = 2) -> str:
    if not reflection_history:
        return "No previous refinement attempts."

    lines = []
    for idx, item in enumerate(reflection_history[-last_k:], 1):
        lines.append(
            f"{idx}. action={item.get('action','')}, "
            f"changed={item.get('changed_terms', {}).get('changed_term_count', 0)}"
        )
    return "\n".join(lines)

def get_required_min_changes(action: str) -> int:
    return ACTION_MIN_CHANGES.get(action, 1)

def build_edit_prompt(
    action: str,
    text: str,
    keywords: List[str],
    judge_metrics: Dict[str, Any],
    target_count: int,
    reflection_history: List[Dict[str, Any]],
) -> str:
    context = get_context(text)
    min_changes = get_required_min_changes(action)
    salvageable = judge_metrics.get("salvageable_with_small_edits", True)
    editing_guidelines = judge_metrics.get("editing_guidelines", [])

    action_instr = {
        "LIGHT_EDIT": "Make a small but real improvement to the list.",
        "ADD_MISSING": "Improve coverage by adding missing core concepts.",
        "REMOVE_UNSUPPORTED": "Remove weak, generic, or unsupported terms.",
        "REPLACE_WEAKEST": "Replace the weakest current keywords with stronger grounded ones.",
        "REGENERATE": "Rebuild the whole list from scratch.",
    }.get(action, "Improve the list.")

    regen_line = (
        "Start fresh from the document instead of preserving the current list."
        if (action == "REGENERATE" or not salvageable)
        else "Preserve useful keywords and only revise what is needed."
    )

    return f"""
Task: {action}

Current keywords:
{keywords_to_text(keywords)}

Judge summary:
- reason: {judge_metrics.get("reason_short", "")}
- major_failures: {judge_metrics.get("major_failures", [])}
- weak_keywords: {judge_metrics.get("weak_keywords", [])}
- missing_concepts: {judge_metrics.get("missing_concepts", [])}
- edit_pressure: {judge_metrics.get("edit_pressure", "medium")}
- salvageable_with_small_edits: {salvageable}

Action instruction:
{action_instr}

Judge-provided editing guidelines:
{format_editing_guidelines(editing_guidelines)}

Recent refinement history:
{format_reflection_history_brief(reflection_history, last_k=2)}

Document:
{context}

*** STRICT RULES — MUST FOLLOW EXACTLY ***
- OUTPUT EXACTLY {target_count} KEYWORDS TOTAL
- CHANGE AT LEAST {min_changes} KEYWORD(S)
- Do not return the identical list
- Do NOT output any explanation, numbering, or bullets
- OUTPUT ONLY A COMMA-SEPARATED KEYWORD LIST
- {regen_line}
""".strip()


# ============================================================
# ENV
# ============================================================

class Env:
    def __init__(self, main: OpenRouterLLM, judge: Judge):
        self.main = main
        self.judge = judge

    def initial_answer(self, text: str, target_count: int) -> List[str]:
        context = get_context(text)
        logger.info(f"[CTX][INITIAL] original_len={len(text)} used_len={len(context)}")

        raw = self.main.generate(
            initial_prompt(text, target_count),
            temperature=0.25,
            max_tokens=BIG_M,
            system_prompt=EDITOR_SYSTEM_PROMPT,
        )
        return parse_model_keywords(raw, target_count)

    def judge_keywords(self, text: str, keywords: List[str], target_count: int) -> Dict[str, Any]:
        context = get_context(text)
        logger.info(f"[CTX][JUDGE] original_len={len(text)} used_len={len(context)}")
        return self.judge.evaluate(text=context, keywords=keywords, target_count=target_count)

    def step(
        self,
        action: str,
        text: str,
        keywords: List[str],
        target_count: int,
        reflection_history: List[Dict[str, Any]],
        judge_metrics: Dict[str, Any],
    ) -> List[str]:
        if action == "STOP":
            return keywords

        context = get_context(text)
        logger.info(f"[CTX][EDIT] original_len={len(text)} used_len={len(context)}")

        prompt = build_edit_prompt(
            action=action,
            text=text,
            keywords=keywords,
            judge_metrics=judge_metrics,
            target_count=target_count,
            reflection_history=reflection_history,
        )

        raw = self.main.generate(
            prompt=prompt,
            temperature=0.10 if action != "REGENERATE" else 0.20,
            max_tokens=BIG_M,
            system_prompt=EDITOR_SYSTEM_PROMPT,
        )
        return parse_model_keywords(raw, target_count)


# ============================================================
# BENCHMARK ROW HELPERS
# ============================================================

def make_benchmark_step_row(
    method_name: str,
    sample_index: int,
    sample_name: str,
    step: int,
    action: str,
    keywords_before: List[str],
    candidate_keywords: List[str],
    active_keywords_after: List[str],
    eval_before: Optional[Dict[str, float]],
    eval_after: Dict[str, float],
    judge_before: Optional[Dict[str, Any]],
    judge_after: Optional[Dict[str, Any]],
    changed_terms: Dict[str, Any],
    phase: str = "refine",
    done: bool = False,
    stop_reason: str = "",
) -> Dict[str, Any]:
    return {
        "method": method_name,
        "sample_index": sample_index,
        "sample_id": sample_name,
        "step": step,
        "phase": phase,
        "action": action,
        "done": done,
        "stop_reason": stop_reason,
        "keywords_before": keywords_to_text(keywords_before),
        "candidate_keywords": keywords_to_text(candidate_keywords),
        "active_keywords_after": keywords_to_text(active_keywords_after),
        "eval_quality_before": None if eval_before is None else eval_before.get("quality"),
        "eval_quality_after": eval_after.get("quality"),
        "precision_before": None if eval_before is None else eval_before.get("precision"),
        "precision_after": eval_after.get("precision"),
        "recall_before": None if eval_before is None else eval_before.get("recall"),
        "recall_after": eval_after.get("recall"),
        "f1_before": None if eval_before is None else eval_before.get("f1"),
        "f1_after": eval_after.get("f1"),
        "judge_overall_before": None if judge_before is None else judge_before.get("overall_score"),
        "judge_overall_after": None if judge_after is None else judge_after.get("overall_score"),
        "judge_groundedness_after": None if judge_after is None else judge_after.get("groundedness"),
        "judge_coverage_after": None if judge_after is None else judge_after.get("coverage"),
        "judge_specificity_after": None if judge_after is None else judge_after.get("specificity"),
        "judge_conciseness_after": None if judge_after is None else judge_after.get("conciseness"),
        "judge_uniqueness_after": None if judge_after is None else judge_after.get("uniqueness"),
        "judge_count_fit_after": None if judge_after is None else judge_after.get("count_fit"),
        "judge_duplicate_risk_after": None if judge_after is None else judge_after.get("duplicate_risk"),
        "judge_unsupported_risk_after": None if judge_after is None else judge_after.get("unsupported_risk"),
        "judge_reason_after": None if judge_after is None else judge_after.get("reason_short"),
        "changed_term_count": changed_terms.get("changed_term_count", 0),
        "added_terms": json.dumps(changed_terms.get("added_terms", []), ensure_ascii=False),
        "removed_terms": json.dumps(changed_terms.get("removed_terms", []), ensure_ascii=False),
        "added_words": json.dumps(changed_terms.get("added_words", []), ensure_ascii=False),
        "removed_words": json.dumps(changed_terms.get("removed_words", []), ensure_ascii=False),
        "replacements": json.dumps(changed_terms.get("replacements", []), ensure_ascii=False),
    }

def summarize_method_results(method_name: str, sample_df: pd.DataFrame) -> Dict[str, Any]:
    if sample_df.empty:
        return {"method": method_name}

    improved_final = (sample_df["final_eval_quality"] > sample_df["initial_eval_quality"]).mean()
    improved_best = (sample_df["best_eval_quality"] > sample_df["initial_eval_quality"]).mean()
    worsened_final = (sample_df["final_eval_quality"] < sample_df["initial_eval_quality"]).mean()
    worsened_best = (sample_df["best_eval_quality"] < sample_df["initial_eval_quality"]).mean()

    return {
        "method": method_name,
        "num_samples": len(sample_df),
        "avg_initial_eval_quality": sample_df["initial_eval_quality"].mean(),
        "avg_final_eval_quality": sample_df["final_eval_quality"].mean(),
        "avg_best_eval_quality": sample_df["best_eval_quality"].mean(),
        "avg_initial_judge_overall": sample_df["initial_judge_overall"].mean(),
        "avg_final_judge_overall": sample_df["final_judge_overall"].mean(),
        "avg_best_judge_overall": sample_df["best_judge_overall"].mean(),
        "avg_initial_f1": sample_df["initial_f1"].mean(),
        "avg_final_f1": sample_df["final_f1"].mean(),
        "avg_best_f1": sample_df["best_f1"].mean(),
        "avg_initial_recall": sample_df["initial_recall"].mean(),
        "avg_final_recall": sample_df["final_recall"].mean(),
        "avg_best_recall": sample_df["best_recall"].mean(),
        "avg_eval_gain_final_vs_initial": sample_df["eval_quality_gain_final_vs_initial"].mean(),
        "avg_eval_gain_best_vs_initial": sample_df["eval_quality_gain_best_vs_initial"].mean(),
        "avg_judge_gain_final_vs_initial": sample_df["judge_gain_final_vs_initial"].mean(),
        "avg_judge_gain_best_vs_initial": sample_df["judge_gain_best_vs_initial"].mean(),
        "avg_steps_taken": sample_df["steps_taken"].mean(),
        "improved_final_rate": improved_final,
        "improved_best_rate": improved_best,
        "worsened_final_rate": worsened_final,
        "worsened_best_rate": worsened_best,
    }


# ============================================================
# BENCHMARK METHODS
# ============================================================
def evaluate_single_pass(test_examples: List[dict], env_single_pass: Env, count: int) -> Dict[str, Any]:
    if(count == 1):
        method_name = "single_pass_claude"
    if(count == 2):
        method_name = "single_pass_gpt"
    if(count == 3):
        method_name = "single_pass_gemini"
    if(count == 4):
        method_name = "single_pass_gptnano"
    logger.info(method_name)
    step_rows = []
    sample_rows = []
    progress = {"method": method_name, "samples": []}

    logger.info("=" * 100)
    logger.info("BENCHMARK START | SINGLE-PASS")
    logger.info("=" * 100)

    for i, sample in enumerate(test_examples):
        text = sample["text"]
        gold = sample["gold_keywords"]
        target_count = len(gold)
        sample_name = sample.get("id", f"test_{i+1}")

        logger.info(f"[{method_name}] SAMPLE {i+1}/{len(test_examples)} | id={sample_name}")

        pred_kw = env_single_pass.initial_answer(text, target_count)
        eval_m = compute_quality(text, pred_kw, gold)
        logger.info(f"Quality: {eval_m['quality']}")

        step_rows.append(make_benchmark_step_row(
            method_name=method_name,
            sample_index=i + 1,
            sample_name=sample_name,
            step=0,
            action="INITIAL_ONLY",
            keywords_before=[],
            candidate_keywords=pred_kw,
            active_keywords_after=pred_kw,
            eval_before=None,
            eval_after=eval_m,
            judge_before=None,
            judge_after=None,
            changed_terms=_EMPTY_CHANGED,
            phase="initial",
            done=True,
            stop_reason="single_pass_no_refinement",
        ))

        rec = {
            "method": method_name,
            "sample_index": i + 1,
            "sample_id": sample_name,
            "gold_count": target_count,
            "initial_eval_quality": eval_m["quality"],
            "final_eval_quality": eval_m["quality"],
            "best_eval_quality": eval_m["quality"],
            "initial_judge_overall": None,
            "final_judge_overall": None,
            "best_judge_overall": None,
            "initial_f1": eval_m["f1"],
            "final_f1": eval_m["f1"],
            "best_f1": eval_m["f1"],
            "initial_recall": eval_m["recall"],
            "final_recall": eval_m["recall"],
            "best_recall": eval_m["recall"],
            "eval_quality_gain_final_vs_initial": 0.0,
            "eval_quality_gain_best_vs_initial": 0.0,
            "judge_gain_final_vs_initial": None,
            "judge_gain_best_vs_initial": None,
            "steps_taken": 0,
            "action_history": "",
            "final_action": "",
            "initial_keywords": keywords_to_text(pred_kw),
            "final_keywords": keywords_to_text(pred_kw),
            "best_keywords": keywords_to_text(pred_kw),
            "gold_keywords": keywords_to_text(gold),
            "stop_reason": "single_pass_no_refinement",
        }
        sample_rows.append(rec)

        progress["samples"].append({
            "sample_index": i + 1,
            "sample_id": sample_name,
            "gold_count": target_count,
            "initial_keywords": list(pred_kw),
            "final_keywords": list(pred_kw),
            "best_keywords": list(pred_kw),
            "initial_eval_metrics": to_serializable_metrics(eval_m),
            "final_eval_metrics": to_serializable_metrics(eval_m),
            "best_eval_metrics": to_serializable_metrics(eval_m),
            "initial_judge_metrics": None,
            "final_judge_metrics": None,
            "best_judge_metrics": None,
            "steps": [],
            "stop_reason": "single_pass_no_refinement",
        })

    step_df = pd.DataFrame(step_rows)
    sample_df = pd.DataFrame(sample_rows)
    summary = summarize_method_results(method_name, sample_df)

    return {
        "method": method_name,
        "step_df": step_df,
        "sample_df": sample_df,
        "progress": progress,
        "summary": summary,
    }

def evaluate_always_REGENERATE(test_examples: List[dict], env: Env, steps: int = 10) -> Dict[str, Any]:
    method_name = "always_REGENERATE"
    step_rows = []
    sample_rows = []
    progress = {"method": method_name, "steps": steps, "samples": []}

    logger.info("=" * 100)
    logger.info(f"BENCHMARK START | ALWAYS-REGENERATE  | steps={steps}")
    logger.info("=" * 100)

    for i, sample in enumerate(test_examples):
        text = sample["text"]
        gold = sample["gold_keywords"]
        target_count = len(gold)
        sample_name = sample.get("id", f"test_{i+1}")

        logger.info(f"[{method_name}] SAMPLE {i+1}/{len(test_examples)} | id={sample_name}")

        pred_kw = env.initial_answer(text, target_count)
        initial_keywords = list(pred_kw)

        eval_m = compute_quality(text, pred_kw, gold)
        judge_m = env.judge_keywords(text, pred_kw, target_count)

        init_eval = dict(eval_m)
        init_judge = dict(judge_m)

        best_kw = list(pred_kw)
        best_eval = dict(eval_m)
        best_judge = dict(judge_m)
        reflection_history = []

        step_rows.append(make_benchmark_step_row(
            method_name=method_name,
            sample_index=i + 1,
            sample_name=sample_name,
            step=0,
            action="INITIAL",
            keywords_before=[],
            candidate_keywords=pred_kw,
            active_keywords_after=pred_kw,
            eval_before=None,
            eval_after=eval_m,
            judge_before=None,
            judge_after=judge_m,
            changed_terms=_EMPTY_CHANGED,
            phase="initial",
            done=False,
            stop_reason="",
        ))

        sample_progress = {
            "sample_index": i + 1,
            "sample_id": sample_name,
            "gold_count": target_count,
            "initial_keywords": list(initial_keywords),
            "initial_eval_metrics": to_serializable_metrics(init_eval),
            "initial_judge_metrics": to_serializable_metrics(init_judge),
            "steps": [],
        }

        for step in range(steps):
            old_kw = list(pred_kw)
            old_eval = dict(eval_m)
            old_judge = dict(judge_m)

            cand_kw = env.step("REGENERATE", text, pred_kw, target_count, reflection_history, judge_m)
            cand_eval = compute_quality(text, cand_kw, gold)
            cand_judge = env.judge_keywords(text, cand_kw, target_count)

            changed_terms = diff_changed_terms(old_kw, cand_kw)

            reflection_history.append({
                "step": step + 1,
                "action": "REGENERATE",
                "candidate_keywords": list(cand_kw),
                "gold_deltas": metrics_delta(old_eval, cand_eval),
                "changed_terms": changed_terms,
            })

            pred_kw = cand_kw
            eval_m = cand_eval
            judge_m = cand_judge

            if eval_m["quality"] > best_eval["quality"]:
                best_kw = list(pred_kw)
                best_eval = dict(eval_m)
                best_judge = dict(judge_m)

            done = step == (steps - 1)
            stop_reason = "fixed_REGENERATE_steps_completed" if done else ""

            step_rows.append(make_benchmark_step_row(
                method_name=method_name,
                sample_index=i + 1,
                sample_name=sample_name,
                step=step + 1,
                action="REGENERATE",
                keywords_before=old_kw,
                candidate_keywords=cand_kw,
                active_keywords_after=pred_kw,
                eval_before=old_eval,
                eval_after=eval_m,
                judge_before=old_judge,
                judge_after=judge_m,
                changed_terms=changed_terms,
                phase="refine",
                done=done,
                stop_reason=stop_reason,
            ))

            sample_progress["steps"].append({
                "step": step + 1,
                "action": "REGENERATE",
                "candidate_keywords": list(cand_kw),
                "eval_metrics_after": to_serializable_metrics(eval_m),
                "judge_metrics_after": to_serializable_metrics(judge_m),
                "changed_terms": changed_terms,
            })

        rec = {
            "method": method_name,
            "sample_index": i + 1,
            "sample_id": sample_name,
            "gold_count": target_count,
            "initial_eval_quality": init_eval["quality"],
            "final_eval_quality": eval_m["quality"],
            "best_eval_quality": best_eval["quality"],
            "initial_judge_overall": init_judge["overall_score"],
            "final_judge_overall": judge_m["overall_score"],
            "best_judge_overall": best_judge["overall_score"],
            "initial_f1": init_eval["f1"],
            "final_f1": eval_m["f1"],
            "best_f1": best_eval["f1"],
            "initial_recall": init_eval["recall"],
            "final_recall": eval_m["recall"],
            "best_recall": best_eval["recall"],
            "eval_quality_gain_final_vs_initial": eval_m["quality"] - init_eval["quality"],
            "eval_quality_gain_best_vs_initial": best_eval["quality"] - init_eval["quality"],
            "judge_gain_final_vs_initial": judge_m["overall_score"] - init_judge["overall_score"],
            "judge_gain_best_vs_initial": best_judge["overall_score"] - init_judge["overall_score"],
            "steps_taken": steps,
            "action_history": " -> ".join(["REGENERATE"] * steps),
            "final_action": "REGENERATE",
            "initial_keywords": keywords_to_text(initial_keywords),
            "final_keywords": keywords_to_text(pred_kw),
            "best_keywords": keywords_to_text(best_kw),
            "gold_keywords": keywords_to_text(gold),
            "stop_reason": "fixed_REGENERATE_steps_completed",
        }
        sample_rows.append(rec)

        sample_progress.update({
            "final_keywords": list(pred_kw),
            "best_keywords": list(best_kw),
            "stop_reason": "fixed_REGENERATE_steps_completed",
        })
        progress["samples"].append(sample_progress)

    step_df = pd.DataFrame(step_rows)
    sample_df = pd.DataFrame(sample_rows)
    summary = summarize_method_results(method_name, sample_df)

    return {
        "method": method_name,
        "step_df": step_df,
        "sample_df": sample_df,
        "progress": progress,
        "summary": summary,
    }

def evaluate_fixed_10_step_iterative(test_examples: List[dict], env: Env) -> Dict[str, Any]:
    method_name = "random_10_step_iterative"
    random_actions_pool = [
        "LIGHT_EDIT",
        "ADD_MISSING",
        "REMOVE_UNSUPPORTED",
        "REPLACE_WEAKEST",
        "REGENERATE",
    ]

    step_rows = []
    sample_rows = []
    progress = {
        "method": method_name,
        "action_pool": random_actions_pool,
        "samples": [],
    }

    logger.info("=" * 100)
    logger.info(f"BENCHMARK START | RANDOM-10-STEP-ITERATIVE | pool={random_actions_pool}")
    logger.info("=" * 100)

    for i, sample in enumerate(test_examples):
        text = sample["text"]
        gold = sample["gold_keywords"]
        target_count = len(gold)
        sample_name = sample.get("id", f"test_{i+1}")

        logger.info(f"[{method_name}] SAMPLE {i+1}/{len(test_examples)} | id={sample_name}")

        pred_kw = env.initial_answer(text, target_count)
        initial_keywords = list(pred_kw)

        eval_m = compute_quality(text, pred_kw, gold)
        judge_m = env.judge_keywords(text, pred_kw, target_count)

        init_eval = dict(eval_m)
        init_judge = dict(judge_m)

        best_kw = list(pred_kw)
        best_eval = dict(eval_m)
        best_judge = dict(judge_m)
        reflection_history = []

        step_rows.append(make_benchmark_step_row(
            method_name=method_name,
            sample_index=i + 1,
            sample_name=sample_name,
            step=0,
            action="INITIAL",
            keywords_before=[],
            candidate_keywords=pred_kw,
            active_keywords_after=pred_kw,
            eval_before=None,
            eval_after=eval_m,
            judge_before=None,
            judge_after=judge_m,
            changed_terms=_EMPTY_CHANGED,
            phase="initial",
            done=False,
            stop_reason="",
        ))

        sample_progress = {
            "sample_index": i + 1,
            "sample_id": sample_name,
            "gold_count": target_count,
            "initial_keywords": list(initial_keywords),
            "initial_eval_metrics": to_serializable_metrics(init_eval),
            "initial_judge_metrics": to_serializable_metrics(init_judge),
            "steps": [],
        }

        action_history = []

        for step in range(10):
            action = random.choice(random_actions_pool)
            action_history.append(action)

            old_kw = list(pred_kw)
            old_eval = dict(eval_m)
            old_judge = dict(judge_m)

            cand_kw = env.step(action, text, pred_kw, target_count, reflection_history, judge_m)
            cand_eval = compute_quality(text, cand_kw, gold)
            cand_judge = env.judge_keywords(text, cand_kw, target_count)

            changed_terms = diff_changed_terms(old_kw, cand_kw)

            reflection_history.append({
                "step": step + 1,
                "action": action,
                "candidate_keywords": list(cand_kw),
                "gold_deltas": metrics_delta(old_eval, cand_eval),
                "changed_terms": changed_terms,
            })

            pred_kw = cand_kw
            eval_m = cand_eval
            judge_m = cand_judge

            if eval_m["quality"] > best_eval["quality"]:
                best_kw = list(pred_kw)
                best_eval = dict(eval_m)
                best_judge = dict(judge_m)

            done = step == 9
            stop_reason = "random_10_step_sequence_completed" if done else ""

            step_rows.append(make_benchmark_step_row(
                method_name=method_name,
                sample_index=i + 1,
                sample_name=sample_name,
                step=step + 1,
                action=action,
                keywords_before=old_kw,
                candidate_keywords=cand_kw,
                active_keywords_after=pred_kw,
                eval_before=old_eval,
                eval_after=eval_m,
                judge_before=old_judge,
                judge_after=judge_m,
                changed_terms=changed_terms,
                phase="refine",
                done=done,
                stop_reason=stop_reason,
            ))

            sample_progress["steps"].append({
                "step": step + 1,
                "action": action,
                "candidate_keywords": list(cand_kw),
                "eval_metrics_after": to_serializable_metrics(eval_m),
                "judge_metrics_after": to_serializable_metrics(judge_m),
                "changed_terms": changed_terms,
            })

        rec = {
            "method": method_name,
            "sample_index": i + 1,
            "sample_id": sample_name,
            "gold_count": target_count,
            "initial_eval_quality": init_eval["quality"],
            "final_eval_quality": eval_m["quality"],
            "best_eval_quality": best_eval["quality"],
            "initial_judge_overall": init_judge["overall_score"],
            "final_judge_overall": judge_m["overall_score"],
            "best_judge_overall": best_judge["overall_score"],
            "initial_f1": init_eval["f1"],
            "final_f1": eval_m["f1"],
            "best_f1": best_eval["f1"],
            "initial_recall": init_eval["recall"],
            "final_recall": eval_m["recall"],
            "best_recall": best_eval["recall"],
            "eval_quality_gain_final_vs_initial": eval_m["quality"] - init_eval["quality"],
            "eval_quality_gain_best_vs_initial": best_eval["quality"] - init_eval["quality"],
            "judge_gain_final_vs_initial": judge_m["overall_score"] - init_judge["overall_score"],
            "judge_gain_best_vs_initial": best_judge["overall_score"] - init_judge["overall_score"],
            "steps_taken": 10,
            "action_history": " -> ".join(action_history),
            "final_action": action_history[-1] if action_history else "",
            "initial_keywords": keywords_to_text(initial_keywords),
            "final_keywords": keywords_to_text(pred_kw),
            "best_keywords": keywords_to_text(best_kw),
            "gold_keywords": keywords_to_text(gold),
            "stop_reason": "random_10_step_sequence_completed",
        }
        sample_rows.append(rec)

        sample_progress.update({
            "final_keywords": list(pred_kw),
            "best_keywords": list(best_kw),
            "action_history": action_history,
            "stop_reason": "random_10_step_sequence_completed",
        })
        progress["samples"].append(sample_progress)

    step_df = pd.DataFrame(step_rows)
    sample_df = pd.DataFrame(sample_rows)
    summary = summarize_method_results(method_name, sample_df)

    return {
        "method": method_name,
        "step_df": step_df,
        "sample_df": sample_df,
        "progress": progress,
        "summary": summary,
    }

# ============================================================
# SAVE BENCHMARK OUTPUTS
# ============================================================

def save_benchmark_method_outputs(result: Dict[str, Any], output_dir: str) -> None:
    method = result["method"]

    step_csv = os.path.join(output_dir, f"{method}_step_summary.csv")
    sample_csv = os.path.join(output_dir, f"{method}_sample_summary.csv")
    progress_json = os.path.join(output_dir, f"{method}_progress.json")

    save_dataframe(result["step_df"], step_csv)
    save_dataframe(result["sample_df"], sample_csv)
    save_json(result["progress"], progress_json)

    logger.info(f"[SAVED] {method} | step={step_csv}")
    logger.info(f"[SAVED] {method} | sample={sample_csv}")
    logger.info(f"[SAVED] {method} | progress={progress_json}")


# ============================================================
# MAIN
# ============================================================

def run_test_only_benchmarks() -> None:
    logger.info("=" * 120)
    logger.info("START — TEST-ONLY KEYWORD BENCHMARKS")
    logger.info("=" * 120)
    logger.info(f"SEED: {SEED}")
    logger.info(f"BENCHMARK_DIR: {BENCHMARK_DIR}")
    logger.info(f"MAIN_MODEL: {MAIN_MODEL}")
    logger.info(f"JUDGE_MODEL: {JUDGE_MODEL}")
    logger.info(f"MAX_CONTEXT_CHARS: {MAX_CONTEXT_CHARS}")
    logger.info(f"SEMEVAL_ROOT: {SEMEVAL_ROOT}")

    all_examples = load_semeval2010_examples(SEMEVAL_ROOT)
    train_examples, test_examples = build_train_test_split(
        all_examples,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )

    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Train examples (unused here): {len(train_examples)}")
    logger.info(f"Test examples: {len(test_examples)}")

    if SMALL_RUN:
        test_examples = test_examples[:SMALL_TEST_SAMPLES]
        logger.info(f"SMALL_RUN active | using first {len(test_examples)} test samples")

    main_llm = OpenRouterLLM(MAIN_MODEL)
    single_pass_llm = OpenRouterLLM(SINGLE_PASS_MODEL_Claude)
    single_pass_llm_gpt = OpenRouterLLM(SINGLE_PASS_MODEL_Gpt)
    single_pass_llm_gemini = OpenRouterLLM(SINGLE_PASS_MODEL_Gemini)
    single_pass_llm_gptnano = OpenRouterLLM(SINGLE_PASS_MODEL_Gptnano)
    judge = Judge(OpenRouterLLM(JUDGE_MODEL))

    env = Env(main_llm, judge)
    env_single_pass = Env(single_pass_llm, judge)
    env_single_pass_gpt = Env(single_pass_llm_gpt, judge)
    env_single_pass_gemini = Env(single_pass_llm_gemini, judge)
    env_single_pass_gpt_nano = Env(single_pass_llm_gptnano, judge)
    results = []

    r4 = evaluate_single_pass(test_examples, env_single_pass_gpt_nano, 4)
    save_benchmark_method_outputs(r4, BENCHMARK_DIR)
    results.append(r4)

    r3 = evaluate_single_pass(test_examples, env_single_pass_gemini, 3)
    save_benchmark_method_outputs(r3, BENCHMARK_DIR)
    results.append(r3)

    r1 = evaluate_single_pass(test_examples, env_single_pass, 1)
    save_benchmark_method_outputs(r1, BENCHMARK_DIR)
    results.append(r1)

    r2 = evaluate_single_pass(test_examples, env_single_pass_gpt, 2)
    save_benchmark_method_outputs(r2, BENCHMARK_DIR)
    results.append(r2)

    r5 = evaluate_always_REGENERATE(test_examples, env, steps=10)
    save_benchmark_method_outputs(r5, BENCHMARK_DIR)
    results.append(r5)

    r6 = evaluate_fixed_10_step_iterative(test_examples, env)
    save_benchmark_method_outputs(r6, BENCHMARK_DIR)
    results.append(r6)

    summary_df = pd.DataFrame([r["summary"] for r in results])
    combined_sample_df = pd.concat([r["sample_df"] for r in results], ignore_index=True)

    save_dataframe(summary_df, BENCHMARK_SUMMARY_CSV)
    save_dataframe(combined_sample_df, BENCHMARK_COMBINED_SAMPLE_CSV)

    combined_json = {
        "seed": SEED,
        "benchmark_dir": BENCHMARK_DIR,
        "num_test_samples": len(test_examples),
        "methods": [r["method"] for r in results],
        "summaries": [r["summary"] for r in results],
        "artifacts": {
            "benchmark_summary_csv": BENCHMARK_SUMMARY_CSV,
            "combined_sample_csv": BENCHMARK_COMBINED_SAMPLE_CSV,
        },
    }
    save_json(combined_json, BENCHMARK_SUMMARY_JSON)

    log_dataframe("BENCHMARK SUMMARY", summary_df)
    log_dataframe("BENCHMARK ALL-METHOD SAMPLE SUMMARY", combined_sample_df)

    logger.info("=" * 120)
    logger.info("BENCHMARK DONE")
    logger.info(f"All outputs saved under: {BENCHMARK_DIR}")
    logger.info("=" * 120)


if __name__ == "__main__":
    run_test_only_benchmarks()
