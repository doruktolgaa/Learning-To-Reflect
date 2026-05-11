import os
import re
import ast
import time
import json
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from openai import OpenAI
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 5
SMALL_RUN_EPOCHS = 2
MAX_STEPS = 10
LR = 1e-4
GAMMA = 0.99
BIG_M = 350000

MAIN_MODEL = "openai/gpt-5.4-nano"
JUDGE_MODEL = "openai/gpt-5.4-nano"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

SEMEVAL_ROOT = os.path.expanduser("~/Downloads/SemEval2017")
DOCS_DIR = os.path.join(SEMEVAL_ROOT, "docsutf8")
KEYS_DIR = os.path.join(SEMEVAL_ROOT, "keys")

TRAIN_RATIO = 0.8
SMALL_RUN = False
SMALL_TRAIN_SAMPLES = 10
SMALL_TEST_SAMPLES = 10

DESKTOP_DIR = os.path.expanduser("~/Desktop")
REPORT_DIR = os.path.join(DESKTOP_DIR, "rl_semeval2017_keyword_v7_reports_sameLLM")
os.makedirs(REPORT_DIR, exist_ok=True)

LOG_FILE = os.path.join(REPORT_DIR, "training_semeval2017_keywords_v7.log")
MODEL_PATH = os.path.join(REPORT_DIR, "rl_keyword_controller_semeval2017_v7.pt")

TRAIN_STEP_CSV = os.path.join(REPORT_DIR, "train_step_metrics.csv")
TRAIN_SAMPLE_CSV = os.path.join(REPORT_DIR, "train_sample_summary.csv")
TRAIN_EPOCH_CSV = os.path.join(REPORT_DIR, "train_epoch_summary.csv")
TEST_STEP_CSV = os.path.join(REPORT_DIR, "test_step_metrics.csv")
TEST_SAMPLE_CSV = os.path.join(REPORT_DIR, "test_sample_summary.csv")
TEST_FINAL_CSV = os.path.join(REPORT_DIR, "test_final_overview.csv")
TRAIN_PROGRESS_JSON = os.path.join(REPORT_DIR, "train_progress.json")
TEST_PROGRESS_JSON = os.path.join(REPORT_DIR, "test_progress.json")
COMBINED_PROGRESS_JSON = os.path.join(REPORT_DIR, "combined_progress.json")

FULL_LOG_TEXT = True
ACCEPT_EPS = 1e-6
ENTROPY_BETA = 0.005
EARLY_STOP_NO_REWARD_IMPROVEMENT_PATIENCE = 3

MAX_CONTEXT_CHARS = 10000

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY bulunamadı.")

ACTIONS = ["STOP", "LIGHT_EDIT", "ADD_MISSING", "REMOVE_UNSUPPORTED", "REPLACE_WEAKEST", "REGENERATE"]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for a, i in ACTION2IDX.items()}

WRONG_EARLY_STOP_PENALTY = 0.12
SECOND_STEP_STOP_PENALTY = 0.05
ACTION_COST = 0.01

GOLD_DELTA_WEIGHTS = {
    "quality": 0.35,
    "f1": 0.40,
    "recall": 0.15,
    "precision": 0.10,
}

ACTION_MIN_CHANGES = {
    "LIGHT_EDIT": 1,
    "ADD_MISSING": 1,
    "REMOVE_UNSUPPORTED": 1,
    "REPLACE_WEAKEST": 1,
    "REGENERATE": 1,
}

# ============================================================
# LOGGER
# ============================================================

def setup_logger():
    logger = logging.getLogger("rl_keyword_v7")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = setup_logger()

# ============================================================
# CONTEXT CONTROL
# ============================================================

def get_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# ============================================================
# TEXT / KEYWORD HELPERS
# ============================================================

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def truncate_text(s: str, n: int = 300) -> str:
    s = normalize_text(s)
    return s if FULL_LOG_TEXT else (s[:n] + ("..." if len(s) > n else ""))

def normalize_keyword(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[_/]", " ", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokenize_for_overlap(text: str) -> List[str]:
    text = re.sub(r"[^a-z0-9\s\-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip().split() if text.strip() else []

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
    for sep in ["\n", ";", ","]:
        if sep in s:
            return [normalize_text(x) for x in s.split(sep) if normalize_text(x)]
    return [normalize_text(s)] if s else []

def deduplicate_keywords(keywords: List[str]) -> List[str]:
    seen, out = set(), []
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
                return deduplicate_keywords([normalize_text(x) for x in parsed if normalize_text(x)])[:target_count]
        except Exception:
            pass
    text = re.sub(r"^(keywords?|keyphrases?)\s*:\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(r"\n|;|,|\|", text)
    cleaned = [normalize_text(re.sub(r"^\s*[-*•\d\.\)]*\s*", "", p)) for p in parts]
    return deduplicate_keywords([x for x in cleaned if x])[:target_count]

def keyword_in_text(keyword: str, text: str) -> bool:
    return normalize_keyword(keyword) in normalize_text(text).lower()

def keyword_token_overlap(pred_kw: str, gold_kw: str) -> float:
    p, g = set(tokenize_for_overlap(pred_kw)), set(tokenize_for_overlap(gold_kw))
    if not p or not g:
        return 0.0
    inter = len(p & g)
    prec, rec = inter / len(p), inter / len(g)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

def match_keywords(pred_keywords: List[str], gold_keywords: List[str], threshold: float = 0.8):
    pred_norm = [normalize_keyword(x) for x in pred_keywords]
    gold_norm = [normalize_keyword(x) for x in gold_keywords]
    matched_pred, matched_gold = set(), set()
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
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1, "matched": tp}

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
    return sum(max(keyword_token_overlap(pk, gk) for gk in gold_keywords) for pk in pred_keywords) / len(pred_keywords)

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

def compute_avg_keyword_length(pred_keywords: List[str]) -> float:
    if not pred_keywords:
        return 0.0
    return min(sum(len(kw.split()) for kw in pred_keywords) / len(pred_keywords) / 6.0, 1.0)

def compute_quality(text: str, pred_keywords: List[str], gold_keywords: List[str]) -> Dict[str, float]:
    gold_count = len(deduplicate_keywords(gold_keywords))
    pred_keywords = deduplicate_keywords(pred_keywords)
    gold_keywords = deduplicate_keywords(gold_keywords)

    base = precision_recall_f1(pred_keywords, gold_keywords)
    p, r, f1 = base["precision"], base["recall"], base["f1"]
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
    return {f"delta_{k}": v - old.get(k, 0.0) for k, v in new.items() if isinstance(v, (int, float))}

def keyword_change_count(old_keywords: List[str], new_keywords: List[str]) -> int:
    old_norm = {normalize_keyword(x) for x in old_keywords if normalize_keyword(x)}
    new_norm = {normalize_keyword(x) for x in new_keywords if normalize_keyword(x)}
    return max(len(old_norm - new_norm), len(new_norm - old_norm))

def diff_changed_terms(old_keywords: List[str], new_keywords: List[str]) -> Dict[str, Any]:
    old_map = {normalize_keyword(x): x for x in old_keywords if normalize_keyword(x)}
    new_map = {normalize_keyword(x): x for x in new_keywords if normalize_keyword(x)}
    removed = [old_map[n] for n in set(old_map) - set(new_map)]
    added = [new_map[n] for n in set(new_map) - set(old_map)]
    replacements = [{"position": i, "old": r, "new": a} for i, (r, a) in enumerate(zip(removed, added))]
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

_EMPTY_CHANGED = {
    "changed_term_count": 0,
    "added_terms": [],
    "removed_terms": [],
    "added_words": [],
    "removed_words": [],
    "replacements": [],
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
        examples.append({"id": doc_id, "text": text, "gold_keywords": gold})

    if not examples:
        raise ValueError("No examples found.")
    return examples

def build_train_test_split(examples: List[dict], train_ratio: float = TRAIN_RATIO):
    examples = list(examples)
    random.shuffle(examples)
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
            api_key=OPENROUTER_API_KEY
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
Your scores are used as input to a reinforcement learning system — so calibration matters enormously.

CRITICAL SCORING RULES:
- Scores must be precise floats with 2 decimal places (e.g. 0.43, 0.71, 0.58)
- NEVER round to 0 or 1 unless the evidence is overwhelming
- The average overall_score across many evaluations should be around 0.45-0.55, not 0.8+
- A score of 1.0 means literally perfect — almost never appropriate
- A score of 0.0 means completely useless — rare
- Most real keyword lists score between 0.30 and 0.75

DIMENSION SCORING:
- groundedness
- coverage
- specificity
- conciseness
- uniqueness
- count_fit
- duplicate_risk
- unsupported_risk

IMPORTANT:
- In addition to evaluation, produce concise editing guidelines for the editor.

You are NOT the editor.
Return ONLY valid JSON.
""".strip()

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
    

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

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
                        return json.loads(text[start:idx+1])
                    except Exception:
                        break

        fragment = text[start:]
        try:
            fragment = re.sub(r',\s*"[^"]*$', '', fragment)
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
    "Use shorter canonical keyphrases",
    "Prefer phrases explicitly supported by the document"
  ],
  "reason_short": "..."
}}

Guideline writing rules:
- Write some editing_guidelines
- Examples of good style:
    "Prefer short canonical keyphrases, ideally 1 to 4 words",
    "Prefer topic/index terms, not claims, findings, or result statements",
    "Do not paraphrase a good keyword into a longer descriptive phrase.",
    "Prefer phrases explicitly present in the document when possible",
    "Avoid generic filler words - Prefer document-grounded terms",
    "Do NOT create duplicates or near-duplicates", 
    "Do NOT invent concepts not in the document"
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

def judge_delta(old_judge: Dict[str, Any], new_judge: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "overall_score", "groundedness", "coverage", "specificity",
        "conciseness", "uniqueness", "count_fit",
        "duplicate_risk", "unsupported_risk"
    ]
    return {f"delta_{k}": new_judge.get(k, 0.0) - old_judge.get(k, 0.0) for k in keys}

# ============================================================
# PROMPTS
# ============================================================

def format_editing_guidelines(guidelines: List[str]) -> str:
    if not guidelines:
        return "- Prefer short document-grounded keyphrases\n- Remove unsupported or weak terms\n- Avoid duplicates and near-duplicates"
    return "\n".join(f"- {g}" for g in guidelines)


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


def format_reflection_history_brief(reflection_history: List[Dict[str, Any]], last_k: int = 2) -> str:
    if not reflection_history:
        return "No previous refinement attempts."
    lines = []
    for idx, item in enumerate(reflection_history[-last_k:], 1):
        jd = item.get("judge_deltas", {})
        lines.append(
            f"{idx}. action={item.get('action','')}, "
            f"delta_overall={jd.get('delta_overall_score',0.0):+.4f}, "
            f"delta_coverage={jd.get('delta_coverage',0.0):+.4f}"
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
# STATE
# ============================================================

@dataclass
class State:
    step: int
    judge_overall: float
    delta_judge_overall: float
    judge_groundedness: float
    judge_coverage: float
    judge_specificity: float
    judge_conciseness: float
    judge_uniqueness: float
    judge_count_fit: float
    judge_duplicate_risk: float
    judge_unsupported_risk: float
    pred_count_norm: float
    reflection_count_norm: float
    last_delta_judge_overall: float
    last_change_count_norm: float
    remaining_steps_norm: float
    duplicate_ratio: float
    unsupported_ratio: float
    avg_keyword_length: float
    judge_momentum: float
    salvageable_flag: float
    last_action_idx: int

def state_tensor(s: State) -> torch.Tensor:
    numeric = torch.tensor([
        s.step / max(MAX_STEPS, 1),
        s.judge_overall,
        (s.delta_judge_overall + 1.0) / 2.0,
        s.judge_groundedness,
        s.judge_coverage,
        s.judge_specificity,
        s.judge_conciseness,
        s.judge_uniqueness,
        s.judge_count_fit,
        s.judge_duplicate_risk,
        s.judge_unsupported_risk,
        min(s.pred_count_norm / 2.0, 1.0),
        s.reflection_count_norm,
        (s.last_delta_judge_overall + 1.0) / 2.0,
        s.last_change_count_norm,
        s.remaining_steps_norm,
        s.duplicate_ratio,
        s.unsupported_ratio,
        s.avg_keyword_length,
        (s.judge_momentum + 1.0) / 2.0,
        s.salvageable_flag,
    ], dtype=torch.float32)

    action_one_hot = torch.zeros(len(ACTIONS), dtype=torch.float32)
    if 0 <= s.last_action_idx < len(ACTIONS):
        action_one_hot[s.last_action_idx] = 1.0

    return torch.cat([numeric, action_one_hot], dim=0).to(DEVICE)

STATE_DIM = 21 + len(ACTIONS)

# ============================================================
# POLICY
# ============================================================

class Policy(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, len(ACTIONS)),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

# ============================================================
# ENV
# ============================================================

EDITOR_SYSTEM_PROMPT = """
You are a careful keyword extraction editor.
Output only a comma-separated keyword list.
Do not explain. Do not add numbering or bullets.
Follow the action constraints strictly.
""".strip()

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
        prompt = build_edit_prompt(action, text, keywords, judge_metrics, target_count, reflection_history)
        logger.info(f"EDIT PROMPT ({action}): {truncate_text(prompt, 5000000)}")
        raw = self.main.generate(
            prompt=prompt,
            temperature=0.10 if action != "REGENERATE" else 0.20,
            max_tokens=BIG_M,
            system_prompt=EDITOR_SYSTEM_PROMPT,
        )
        return parse_model_keywords(raw, target_count)

# ============================================================
# RL HELPERS
# ============================================================

def compute_returns(rewards: List[float], gamma: float = GAMMA) -> List[float]:
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns

def build_state(
    step,
    judge_metrics,
    prev_judge_overall,
    reflection_history,
    pred_count,
    gold_count,
    text,
    pred_keywords,
    last_action_idx=0,
) -> State:
    n = len(reflection_history)
    last_delta_j, last_chg_norm = 0.0, 0.0
    if reflection_history:
        last = reflection_history[-1]
        last_delta_j = last.get("judge_deltas", {}).get("delta_overall_score", 0.0)
        last_chg_norm = min(last.get("changed_terms", {}).get("changed_term_count", 0) / max(gold_count, 1), 1.0)

    j_overall = judge_metrics["overall_score"]
    recent_deltas = [x.get("judge_deltas", {}).get("delta_overall_score", 0.0) for x in reflection_history[-3:]]

    return State(
        step=step,
        judge_overall=j_overall,
        delta_judge_overall=j_overall - prev_judge_overall,
        judge_groundedness=judge_metrics["groundedness"],
        judge_coverage=judge_metrics["coverage"],
        judge_specificity=judge_metrics["specificity"],
        judge_conciseness=judge_metrics["conciseness"],
        judge_uniqueness=judge_metrics["uniqueness"],
        judge_count_fit=judge_metrics["count_fit"],
        judge_duplicate_risk=judge_metrics["duplicate_risk"],
        judge_unsupported_risk=judge_metrics["unsupported_risk"],
        pred_count_norm=min(pred_count / max(gold_count, 1), 2.0),
        reflection_count_norm=min(n / max(MAX_STEPS, 1), 1.0),
        last_delta_judge_overall=last_delta_j,
        last_change_count_norm=last_chg_norm,
        remaining_steps_norm=(MAX_STEPS - step) / max(MAX_STEPS, 1),
        duplicate_ratio=compute_duplicate_ratio(pred_keywords),
        unsupported_ratio=compute_unsupported_ratio(text, pred_keywords),
        avg_keyword_length=compute_avg_keyword_length(pred_keywords),
        judge_momentum=sum(recent_deltas) / len(recent_deltas) if recent_deltas else 0.0,
        salvageable_flag=1.0 if judge_metrics.get("salvageable_with_small_edits", True) else 0.0,
        last_action_idx=last_action_idx,
    )

def should_early_stop_due_to_reward_plateau(rewards: List[float], patience: int = EARLY_STOP_NO_REWARD_IMPROVEMENT_PATIENCE) -> bool:
    if len(rewards) < patience + 1:
        return False
    recent = rewards[-(patience + 1):]
    for i in range(1, len(recent)):
        if recent[i] > 0:
            return False
    return True

# ============================================================
# REWARD
# ============================================================

def compute_gold_delta_component(old_eval: Dict[str, float], new_eval: Dict[str, float]) -> float:
    return sum(w * (new_eval.get(m, 0.0) - old_eval.get(m, 0.0)) for m, w in GOLD_DELTA_WEIGHTS.items())

def compute_reward(action: str, old_eval: Dict[str, float], new_eval: Dict[str, float], step: int) -> Tuple[float, Dict[str, float]]:
    if action == "STOP":
        reward = 0.0
        return reward, {"gold_component": 0.0, "action_cost": 0.0, "final_reward": reward}

    gold = compute_gold_delta_component(old_eval, new_eval)
    reward = gold - ACTION_COST
    return reward, {"gold_component": gold, "action_cost": ACTION_COST, "final_reward": reward}

# ============================================================
# UTILS
# ============================================================

def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"REPORT SAVED: {path}")

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON SAVED: {path}")

def log_dataframe(title: str, df: pd.DataFrame, max_rows: int = 200):
    logger.info("=" * 120)
    logger.info(title)
    logger.info("=" * 120)
    logger.info("EMPTY DATAFRAME" if df.empty else "\n" + df.head(max_rows).to_string(index=False))

def to_serializable_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v if isinstance(v, (float, int, str, bool)) or v is None else str(v) for k, v in d.items()}

def _q(d, k):
    return None if d is None else d.get(k)

def make_step_row_common(
    split_name, sample_index, sample_name, step, action, reward, done, gold_count,
    keywords_before, candidate_keywords, active_keywords_after, best_keywords_after,
    eval_before, eval_candidate, eval_after, best_eval,
    judge_before, judge_candidate, judge_after, best_judge,
    changed_terms, reward_breakdown=None, epoch=None, phase="refine", stop_reason="",
):
    return {
        "split": split_name,
        "epoch": epoch,
        "sample_index": sample_index,
        "sample_id": sample_name,
        "step": step,
        "phase": phase,
        "action": action,
        "reward": reward,
        "reward_gold_component": None if reward_breakdown is None else reward_breakdown.get("gold_component"),
        "done": done,
        "stop_reason": stop_reason,
        "gold_count": gold_count,
        "keywords_before": keywords_to_text(keywords_before),
        "candidate_keywords": keywords_to_text(candidate_keywords),
        "active_keywords_after": keywords_to_text(active_keywords_after),
        "best_keywords_after": keywords_to_text(best_keywords_after),
        "eval_quality_before": _q(eval_before, "quality"),
        "candidate_eval_quality": _q(eval_candidate, "quality"),
        "active_eval_quality_after": _q(eval_after, "quality"),
        "best_eval_quality_after": _q(best_eval, "quality"),
        "precision_before": _q(eval_before, "precision"),
        "precision_after": _q(eval_after, "precision"),
        "recall_before": _q(eval_before, "recall"),
        "recall_after": _q(eval_after, "recall"),
        "f1_before": _q(eval_before, "f1"),
        "f1_after": _q(eval_after, "f1"),
        "judge_overall_before": _q(judge_before, "overall_score"),
        "judge_overall_candidate": _q(judge_candidate, "overall_score"),
        "judge_overall_after": _q(judge_after, "overall_score"),
        "judge_best_after": _q(best_judge, "overall_score"),
        "judge_grounded_before": _q(judge_before, "groundedness"),
        "judge_grounded_after": _q(judge_after, "groundedness"),
        "judge_coverage_before": _q(judge_before, "coverage"),
        "judge_coverage_after": _q(judge_after, "coverage"),
        "judge_specificity_before": _q(judge_before, "specificity"),
        "judge_specificity_after": _q(judge_after, "specificity"),
        "judge_conciseness_before": _q(judge_before, "conciseness"),
        "judge_conciseness_after": _q(judge_after, "conciseness"),
        "judge_uniqueness_before": _q(judge_before, "uniqueness"),
        "judge_uniqueness_after": _q(judge_after, "uniqueness"),
        "judge_count_fit_before": _q(judge_before, "count_fit"),
        "judge_count_fit_after": _q(judge_after, "count_fit"),
        "judge_duplicate_risk_before": _q(judge_before, "duplicate_risk"),
        "judge_duplicate_risk_after": _q(judge_after, "duplicate_risk"),
        "judge_unsupported_risk_before": _q(judge_before, "unsupported_risk"),
        "judge_unsupported_risk_after": _q(judge_after, "unsupported_risk"),
        "judge_edit_pressure_before": _q(judge_before, "edit_pressure"),
        "judge_reason_before": _q(judge_before, "reason_short"),
        "judge_reason_candidate": _q(judge_candidate, "reason_short"),
        "judge_major_failures_before": None if judge_before is None else json.dumps(judge_before.get("major_failures", []), ensure_ascii=False),
        "judge_major_failures_candidate": None if judge_candidate is None else json.dumps(judge_candidate.get("major_failures", []), ensure_ascii=False),
        "judge_weak_keywords_before": None if judge_before is None else json.dumps(judge_before.get("weak_keywords", []), ensure_ascii=False),
        "judge_missing_concepts_before": None if judge_before is None else json.dumps(judge_before.get("missing_concepts", []), ensure_ascii=False),
        "changed_term_count": changed_terms.get("changed_term_count", 0),
        "added_terms": json.dumps(changed_terms.get("added_terms", []), ensure_ascii=False),
        "removed_terms": json.dumps(changed_terms.get("removed_terms", []), ensure_ascii=False),
        "added_words": json.dumps(changed_terms.get("added_words", []), ensure_ascii=False),
        "removed_words": json.dumps(changed_terms.get("removed_words", []), ensure_ascii=False),
        "replacements": json.dumps(changed_terms.get("replacements", []), ensure_ascii=False),
    }

# ============================================================
# REFINE
# ============================================================

def refine_once(
    env: Env,
    action: str,
    text: str,
    current_keywords: List[str],
    current_judge_metrics: Dict[str, Any],
    current_eval_metrics: Dict[str, float],
    gold_keywords: List[str],
    target_count: int,
    reflection_history: List[Dict[str, Any]],
):
    cand_kw = env.step(action, text, current_keywords, target_count, reflection_history, current_judge_metrics)
    cand_eval = compute_quality(text, cand_kw, gold_keywords)
    cand_judge = env.judge_keywords(text, cand_kw, target_count)
    return cand_kw, cand_eval, cand_judge

# ============================================================
# TRAIN
# ============================================================

def train(train_examples, policy: Policy, env: Env, epochs: int = EPOCHS):
    opt = optim.Adam(policy.parameters(), lr=LR)
    train_step_rows, train_sample_rows, train_epoch_rows = [], [], []
    train_progress = {"phase": "train", "epochs": []}

    for epoch in range(epochs):
        logger.info("=" * 100)
        logger.info(f"TRAIN EPOCH START | epoch={epoch+1}/{epochs}")
        logger.info("=" * 100)
        random.shuffle(train_examples)
        epoch_records = []

        for i, sample in enumerate(train_examples):
            text, gold = sample["text"], sample["gold_keywords"]
            target_count = len(gold)
            sample_name = sample.get("id", f"train_{i+1}")
            logger.info(f"TRAIN SAMPLE START | epoch={epoch+1} | sample={i+1}/{len(train_examples)} | id={sample_name}")
            logger.info(f"GOLD KEYWORDS ({target_count}): {keywords_to_text(gold)}")

            pred_kw = env.initial_answer(text, target_count)
            eval_m = compute_quality(text, pred_kw, gold)
            judge_m = env.judge_keywords(text, pred_kw, target_count)
            init_eval = dict(eval_m)
            init_judge = dict(judge_m)
            best_kw = list(pred_kw)
            best_eval = dict(eval_m)
            best_judge = dict(judge_m)
            best_reward_sum = 0.0
            running_reward_sum = 0.0
            prev_j_over = 0.0
            last_a_idx = ACTION2IDX["STOP"]

            logger.info(f"INITIAL KEYWORDS: {keywords_to_text(pred_kw)}")
            logger.info(
                "INITIAL | eval_quality=%.4f | f1=%.4f | recall=%.4f | judge_overall=%.4f",
                eval_m["quality"], eval_m["f1"], eval_m["recall"], judge_m["overall_score"]
            )

            refl_hist, log_probs, rewards, entropies, act_hist = [], [], [], [], []
            sample_progress = {
                "epoch": epoch + 1,
                "sample_index": i + 1,
                "sample_id": sample_name,
                "gold_count": target_count,
                "gold_keywords": gold,
                "initial_keywords": list(pred_kw),
                "initial_eval_metrics": to_serializable_metrics(init_eval),
                "initial_judge_metrics": to_serializable_metrics(init_judge),
                "steps": [],
            }

            train_step_rows.append(make_step_row_common(
                "train", i + 1, sample_name, 0, "INITIAL", None, False, target_count,
                [], [], pred_kw, best_kw,
                None, eval_m, eval_m, best_eval,
                None, judge_m, judge_m, best_judge,
                _EMPTY_CHANGED, epoch=epoch + 1, phase="initial",
            ))

            stop_reason = ""
            for step in range(MAX_STEPS):
                s = build_state(step, judge_m, prev_j_over, refl_hist, len(pred_kw), target_count, text, pred_kw, last_a_idx)
                policy.train()
                logits = policy(state_tensor(s))
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                dist = Categorical(logits=logits)
                a_idx = dist.sample()
                lp, ent = dist.log_prob(a_idx), dist.entropy()
                entropies.append(ent)
                action = IDX2ACTION[a_idx.item()]
                act_hist.append(action)
                last_a_idx = a_idx.item()

                logger.info(
                    "TRAIN STEP %d | judge=%.4f | coverage=%.4f | specificity=%.4f | grounded=%.4f | ACTION=%s | PROBS={%s}",
                    step + 1,
                    s.judge_overall,
                    s.judge_coverage,
                    s.judge_specificity,
                    s.judge_groundedness,
                    action,
                    ", ".join(f"{IDX2ACTION[j]}:{probs[j]:.3f}" for j in range(len(ACTIONS))),
                )

                old_kw, old_eval, old_judge = list(pred_kw), dict(eval_m), dict(judge_m)

                if action == "STOP":
                    reward, rb = compute_reward(action, old_eval, old_eval, step)
                    log_probs.append(lp)
                    rewards.append(reward)
                    running_reward_sum += reward
                    if running_reward_sum > best_reward_sum + ACCEPT_EPS:
                        best_kw, best_eval, best_judge = list(pred_kw), dict(eval_m), dict(judge_m)
                        best_reward_sum = running_reward_sum

                    stop_reason = "policy_stop"
                    train_step_rows.append(make_step_row_common(
                        "train", i + 1, sample_name, step + 1, action, reward, True, target_count,
                        old_kw, old_kw, old_kw, best_kw,
                        old_eval, old_eval, old_eval, best_eval,
                        old_judge, old_judge, old_judge, best_judge,
                        _EMPTY_CHANGED, reward_breakdown=rb, epoch=epoch + 1, stop_reason=stop_reason,
                    ))
                    sample_progress["steps"].append({
                        "step": step + 1,
                        "action": action,
                        "reward": reward,
                        "done": True,
                        "stop_reason": stop_reason,
                    })
                    break

                cand_kw, cand_eval, cand_judge = refine_once(
                    env, action, text, pred_kw, judge_m, eval_m,
                    gold, target_count, refl_hist,
                )

                reward, rb = compute_reward(action, old_eval, cand_eval, step)
                done = (step == MAX_STEPS - 1)
                gold_deltas = metrics_delta(old_eval, cand_eval)
                j_deltas = judge_delta(old_judge, cand_judge)
                chg_terms = diff_changed_terms(old_kw, cand_kw)

                logger.info(
                    "TRAIN STEP %d | cand_judge=%.4f | reward=%.4f | f1 Δ=%+.4f | changed=%d",
                    step + 1,
                    cand_judge["overall_score"],
                    reward,
                    gold_deltas.get("delta_f1", 0.0),
                    chg_terms["changed_term_count"],
                )

                refl_hist.append({
                    "step": step + 1,
                    "action": action,
                    "candidate_keywords": list(cand_kw),
                    "gold_deltas": gold_deltas,
                    "judge_deltas": j_deltas,
                    "changed_terms": chg_terms,
                })

                log_probs.append(lp)
                rewards.append(reward)
                running_reward_sum += reward
                prev_j_over = judge_m["overall_score"]
                pred_kw, eval_m, judge_m = cand_kw, cand_eval, cand_judge

                if running_reward_sum > best_reward_sum + ACCEPT_EPS:
                    best_kw, best_eval, best_judge = list(pred_kw), dict(eval_m), dict(judge_m)
                    best_reward_sum = running_reward_sum

                train_step_rows.append(make_step_row_common(
                    "train", i + 1, sample_name, step + 1, action, reward, done, target_count,
                    old_kw, cand_kw, pred_kw, best_kw,
                    old_eval, cand_eval, eval_m, best_eval,
                    old_judge, cand_judge, judge_m, best_judge,
                    chg_terms, reward_breakdown=rb, epoch=epoch + 1,
                ))

                sample_progress["steps"].append({
                    "step": step + 1,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "candidate_keywords": cand_kw,
                    "active_keywords_after": list(pred_kw),
                    "judge_deltas": to_serializable_metrics(j_deltas),
                    "gold_deltas": to_serializable_metrics(gold_deltas),
                })

                if should_early_stop_due_to_reward_plateau(rewards, EARLY_STOP_NO_REWARD_IMPROVEMENT_PATIENCE):
                    stop_reason = "early_stop_no_reward_improvement_3"
                    logger.info(f"EARLY STOP TRIGGERED | sample={sample_name} | step={step+1} | reason={stop_reason}")
                    policy.train()
                    logits = policy(state_tensor(s))
                    dist = Categorical(logits=logits)
                    stop_idx = torch.tensor(ACTION2IDX["STOP"], device=DEVICE)
                    lp = dist.log_prob(stop_idx)
                    ent = dist.entropy()
                    log_probs.append(lp)
                    entropies.append(ent)
                    rewards.append(0.0)
                    running_reward_sum += 0.0
                    if running_reward_sum > best_reward_sum + ACCEPT_EPS:
                        best_kw, best_eval, best_judge = list(pred_kw), dict(eval_m), dict(judge_m)
                        best_reward_sum = running_reward_sum
                    break

                if done:
                    stop_reason = "max_steps_reached"
                    break

            returns = compute_returns(rewards, GAMMA)
            returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
            if len(returns_t) > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            policy_loss = sum(-lp * G for lp, G in zip(log_probs, returns_t)) if log_probs else torch.tensor(0.0, device=DEVICE)
            entropy_bonus = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=DEVICE)
            loss = policy_loss - ENTROPY_BETA * entropy_bonus

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            opt.step()

            rec = {
                "epoch": epoch + 1,
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
                "steps_taken": len(act_hist),
                "action_history": " -> ".join(act_hist),
                "final_action": act_hist[-1] if act_hist else "",
                "final_keywords": keywords_to_text(pred_kw),
                "best_keywords": keywords_to_text(best_kw),
                "gold_keywords": keywords_to_text(gold),
                "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
                "sum_reward": sum(rewards) if rewards else 0.0,
                "best_reward_sum": best_reward_sum,
                "policy_loss": float(policy_loss.detach().cpu()),
                "entropy_bonus": float(entropy_bonus.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "stop_reason": stop_reason,
            }

            train_sample_rows.append(rec)
            epoch_records.append(rec)
            sample_progress.update({
                "final_keywords": list(pred_kw),
                "best_keywords": list(best_kw),
                "action_history": act_hist,
                "mean_reward": rec["mean_reward"],
                "best_reward_sum": best_reward_sum,
                "stop_reason": stop_reason,
            })
            train_progress["epochs"].append(sample_progress)

            logger.info(f"ACTION HISTORY: {' -> '.join(act_hist)}")
            logger.info(
                "TRAIN SAMPLE END | initial_q=%.4f | final_q=%.4f | best_q=%.4f | initial_j=%.4f | final_j=%.4f | best_j=%.4f | best_reward_sum=%.4f | stop_reason=%s",
                init_eval["quality"], eval_m["quality"], best_eval["quality"],
                init_judge["overall_score"], judge_m["overall_score"], best_judge["overall_score"],
                best_reward_sum,
                stop_reason
            )
            logger.info("-" * 90)

        ckpt = MODEL_PATH.replace(".pt", f".epoch{epoch+1}.pt")
        torch.save(policy.state_dict(), ckpt)
        logger.info(f"CHECKPOINT SAVED: {ckpt}")

        edf = pd.DataFrame(epoch_records)
        train_epoch_rows.append({
            "epoch": epoch + 1,
            "num_samples": len(epoch_records),
            **{
                f"avg_{k}": edf[k].mean() for k in [
                    "initial_eval_quality", "final_eval_quality", "best_eval_quality",
                    "initial_judge_overall", "final_judge_overall", "best_judge_overall",
                    "initial_f1", "final_f1", "best_f1",
                    "initial_recall", "final_recall", "best_recall",
                    "steps_taken", "sum_reward", "mean_reward", "best_reward_sum", "policy_loss", "entropy_bonus", "loss",
                ]
            },
        })

    step_df = pd.DataFrame(train_step_rows)
    sample_df = pd.DataFrame(train_sample_rows)
    epoch_df = pd.DataFrame(train_epoch_rows)

    save_dataframe(step_df, TRAIN_STEP_CSV)
    save_dataframe(sample_df, TRAIN_SAMPLE_CSV)
    save_dataframe(epoch_df, TRAIN_EPOCH_CSV)
    save_json(train_progress, TRAIN_PROGRESS_JSON)

    log_dataframe("TRAIN STEP SUMMARY", step_df)
    log_dataframe("TRAIN SAMPLE SUMMARY", sample_df)
    log_dataframe("TRAIN EPOCH SUMMARY", epoch_df)

    return {
        "train_step_df": step_df,
        "train_sample_df": sample_df,
        "train_epoch_df": epoch_df,
        "train_progress": train_progress,
    }

# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(test_examples, policy: Policy, env: Env):
    test_step_rows, test_sample_rows = [], []
    test_progress = {"phase": "test", "samples": []}

    for i, sample in enumerate(test_examples):
        text, gold = sample["text"], sample["gold_keywords"]
        target_count = len(gold)
        sample_name = sample.get("id", f"test_{i+1}")

        logger.info("=" * 100)
        logger.info(f"TEST SAMPLE START | sample={i+1}/{len(test_examples)} | id={sample_name}")
        logger.info("=" * 100)

        pred_kw = env.initial_answer(text, target_count)
        eval_m = compute_quality(text, pred_kw, gold)
        judge_m = env.judge_keywords(text, pred_kw, target_count)

        init_eval = dict(eval_m)
        init_judge = dict(judge_m)
        best_kw = list(pred_kw)
        best_eval = dict(eval_m)
        best_judge = dict(judge_m)
        best_reward_sum = 0.0
        running_reward_sum = 0.0

        prev_j_over = 0.0
        last_a_idx = ACTION2IDX["STOP"]
        rewards = []
        act_hist = []
        refl_hist = []
        stop_reason = ""

        sample_progress = {
            "sample_index": i + 1,
            "sample_id": sample_name,
            "gold_count": target_count,
            "gold_keywords": gold,
            "initial_keywords": list(pred_kw),
            "initial_eval_metrics": to_serializable_metrics(init_eval),
            "initial_judge_metrics": to_serializable_metrics(init_judge),
            "steps": [],
        }

        test_step_rows.append(make_step_row_common(
            "test", i + 1, sample_name, 0, "INITIAL", None, False, target_count,
            [], [], pred_kw, best_kw,
            None, eval_m, eval_m, best_eval,
            None, judge_m, judge_m, best_judge,
            _EMPTY_CHANGED, phase="initial",
        ))

        for step in range(MAX_STEPS):
            s = build_state(step, judge_m, prev_j_over, refl_hist, len(pred_kw), target_count, text, pred_kw, last_a_idx)
            policy.eval()
            logits = policy(state_tensor(s))
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            dist = Categorical(logits=logits)
            a_idx = dist.sample().item()
            action = IDX2ACTION[a_idx]
            act_hist.append(action)
            last_a_idx = a_idx

            logger.info(
                "TEST STEP %d | judge=%.4f | coverage=%.4f | specificity=%.4f | grounded=%.4f | ACTION=%s | PROBS={%s}",
                step + 1,
                s.judge_overall,
                s.judge_coverage,
                s.judge_specificity,
                s.judge_groundedness,
                action,
                ", ".join(f"{IDX2ACTION[j]}:{probs[j]:.3f}" for j in range(len(ACTIONS))),
            )

            old_kw, old_eval, old_judge = list(pred_kw), dict(eval_m), dict(judge_m)

            if action == "STOP":
                reward, rb = compute_reward(action, old_eval, old_eval, step)
                rewards.append(reward)
                running_reward_sum += reward
                if running_reward_sum > best_reward_sum + ACCEPT_EPS:
                    best_kw, best_eval, best_judge = list(pred_kw), dict(eval_m), dict(judge_m)
                    best_reward_sum = running_reward_sum

                stop_reason = "policy_stop"
                test_step_rows.append(make_step_row_common(
                    "test", i + 1, sample_name, step + 1, action, reward, True, target_count,
                    old_kw, old_kw, old_kw, best_kw,
                    old_eval, old_eval, old_eval, best_eval,
                    old_judge, old_judge, old_judge, best_judge,
                    _EMPTY_CHANGED, reward_breakdown=rb, stop_reason=stop_reason,
                ))
                sample_progress["steps"].append({
                    "step": step + 1,
                    "action": action,
                    "reward": reward,
                    "done": True,
                    "stop_reason": stop_reason,
                })
                break

            cand_kw, cand_eval, cand_judge = refine_once(
                env, action, text, pred_kw, judge_m, eval_m,
                gold, target_count, refl_hist,
            )

            reward, rb = compute_reward(action, old_eval, cand_eval, step)
            rewards.append(reward)
            running_reward_sum += reward
            gold_deltas = metrics_delta(old_eval, cand_eval)
            j_deltas = judge_delta(old_judge, cand_judge)
            chg_terms = diff_changed_terms(old_kw, cand_kw)

            pred_kw, eval_m, judge_m = cand_kw, cand_eval, cand_judge
            prev_j_over = old_judge["overall_score"]

            if running_reward_sum > best_reward_sum + ACCEPT_EPS:
                best_kw, best_eval, best_judge = list(pred_kw), dict(eval_m), dict(judge_m)
                best_reward_sum = running_reward_sum

            refl_hist.append({
                "step": step + 1,
                "action": action,
                "candidate_keywords": list(cand_kw),
                "gold_deltas": gold_deltas,
                "judge_deltas": j_deltas,
                "changed_terms": chg_terms,
            })

            done = (step == MAX_STEPS - 1)
            test_step_rows.append(make_step_row_common(
                "test", i + 1, sample_name, step + 1, action, reward, done, target_count,
                old_kw, cand_kw, pred_kw, best_kw,
                old_eval, cand_eval, eval_m, best_eval,
                old_judge, cand_judge, judge_m, best_judge,
                chg_terms, reward_breakdown=rb,
            ))

            sample_progress["steps"].append({
                "step": step + 1,
                "action": action,
                "reward": reward,
                "done": done,
                "candidate_keywords": cand_kw,
                "active_keywords_after": list(pred_kw),
                "judge_deltas": to_serializable_metrics(j_deltas),
                "gold_deltas": to_serializable_metrics(gold_deltas),
            })

            if done:
                stop_reason = "max_steps_reached"
                break

        rec = {
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
            "steps_taken": len(act_hist),
            "action_history": " -> ".join(act_hist),
            "final_action": act_hist[-1] if act_hist else "",
            "final_keywords": keywords_to_text(pred_kw),
            "best_keywords": keywords_to_text(best_kw),
            "gold_keywords": keywords_to_text(gold),
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
            "sum_reward": sum(rewards) if rewards else 0.0,
            "best_reward_sum": best_reward_sum,
            "stop_reason": stop_reason,
        }

        test_sample_rows.append(rec)
        sample_progress.update({
            "final_keywords": list(pred_kw),
            "best_keywords": list(best_kw),
            "action_history": act_hist,
            "mean_reward": rec["mean_reward"],
            "best_reward_sum": best_reward_sum,
            "stop_reason": stop_reason,
        })
        test_progress["samples"].append(sample_progress)

        logger.info(
            "TEST SAMPLE END | initial_q=%.4f | final_q=%.4f | best_q=%.4f | initial_j=%.4f | final_j=%.4f | best_j=%.4f | best_reward_sum=%.4f | stop_reason=%s",
            init_eval["quality"], eval_m["quality"], best_eval["quality"],
            init_judge["overall_score"], judge_m["overall_score"], best_judge["overall_score"],
            best_reward_sum,
            stop_reason
        )
        logger.info("-" * 90)

    step_df = pd.DataFrame(test_step_rows)
    sample_df = pd.DataFrame(test_sample_rows)
    final_df = sample_df.copy()

    save_dataframe(step_df, TEST_STEP_CSV)
    save_dataframe(sample_df, TEST_SAMPLE_CSV)
    save_dataframe(final_df, TEST_FINAL_CSV)
    save_json(test_progress, TEST_PROGRESS_JSON)

    log_dataframe("TEST STEP SUMMARY", step_df)
    log_dataframe("TEST SAMPLE SUMMARY", sample_df)

    return {
        "test_step_df": step_df,
        "test_sample_df": sample_df,
        "test_final_df": final_df,
        "test_progress": test_progress,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("START — RL Keyword Extraction v7")
    logger.info(f"DEVICE: {DEVICE} | STATE_DIM: {STATE_DIM}")
    logger.info(f"EPOCHS: {EPOCHS} | SMALL_RUN_EPOCHS: {SMALL_RUN_EPOCHS} | MAX_STEPS: {MAX_STEPS}")
    logger.info(f"LR: {LR} | GAMMA: {GAMMA} | ENTROPY_BETA: {ENTROPY_BETA}")
    logger.info(f"BIG_M: {BIG_M}")
    logger.info(f"MAIN_MODEL: {MAIN_MODEL} | JUDGE_MODEL: {JUDGE_MODEL}")
    logger.info(f"ACTIONS ({len(ACTIONS)}): {ACTIONS}")
    logger.info(f"SMALL_RUN: {SMALL_RUN} | TRAIN: {SMALL_TRAIN_SAMPLES} | TEST: {SMALL_TEST_SAMPLES}")
    logger.info(f"MAX_CONTEXT_CHARS: {MAX_CONTEXT_CHARS}")
    logger.info(f"EARLY_STOP_NO_REWARD_IMPROVEMENT_PATIENCE: {EARLY_STOP_NO_REWARD_IMPROVEMENT_PATIENCE}")
    logger.info("DESIGN v7 unified context:")
    logger.info("  initial prompt  — get_context(text)")
    logger.info("  edit prompt     — get_context(text)")
    logger.info("  judge prompt    — get_context(text)")
    logger.info("  all stages see the same first 10000 chars")

    all_examples = load_semeval2010_examples(SEMEVAL_ROOT)
    train_examples, test_examples = build_train_test_split(all_examples, TRAIN_RATIO)
    logger.info(f"Total: {len(all_examples)} | Train: {len(train_examples)} | Test: {len(test_examples)}")

    if SMALL_RUN:
        train_examples = train_examples[:SMALL_TRAIN_SAMPLES]
        test_examples = test_examples[:SMALL_TEST_SAMPLES]
        logger.info("SMALL RUN ACTIVE")

    logger.info(f"Train used: {len(train_examples)} | Test used: {len(test_examples)}")

    main_llm = OpenRouterLLM(MAIN_MODEL)
    judge = Judge(OpenRouterLLM(JUDGE_MODEL))
    env = Env(main_llm, judge)
    policy = Policy(STATE_DIM).to(DEVICE)

    logger.info(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    effective_epochs = SMALL_RUN_EPOCHS if SMALL_RUN else EPOCHS
    logger.info(f"Effective epochs: {effective_epochs}")

    train_out = train(train_examples, policy, env, epochs=effective_epochs)
    test_out = evaluate(test_examples, policy, env)

    save_json({
        "train": train_out["train_progress"],
        "test": test_out["test_progress"],
        "artifacts": {
            "train_step_csv": TRAIN_STEP_CSV,
            "train_sample_csv": TRAIN_SAMPLE_CSV,
            "train_epoch_csv": TRAIN_EPOCH_CSV,
            "test_step_csv": TEST_STEP_CSV,
            "test_sample_csv": TEST_SAMPLE_CSV,
            "test_final_csv": TEST_FINAL_CSV,
        },
    }, COMBINED_PROGRESS_JSON)

    torch.save(policy.state_dict(), MODEL_PATH)
    logger.info(f"MODEL SAVED: {MODEL_PATH} | LOG: {LOG_FILE} | REPORTS: {REPORT_DIR}")
    logger.info("DONE")

if __name__ == "__main__":
    main()

