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
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 5
SMALL_RUN_EPOCHS = 1
MAX_STEPS = 10
LR = 1e-4
GAMMA = 0.99

MAIN_MODEL = "openai/gpt-5.4-nano"
JUDGE_MODEL = "openai/gpt-5.4-nano"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

DATASET_NAME = "Babelscape/multinerd"

# None -> tüm diller
TARGET_LANGS = ["zh", "pl", "it", "pt", "fr", "es", "nl", "en", "de", "ru"]
TRAIN_RATIO = 0.8
USE_HF_SPLITS = True

SMALL_RUN = True
SMALL_TRAIN_SAMPLES = 10
SMALL_TEST_SAMPLES = 10

DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
REPORT_DIR = os.path.join(DESKTOP_DIR, "rl_multinerd_ner_v1_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

LOG_FILE = os.path.join(REPORT_DIR, "training_multinerd_ner_v1.log")
MODEL_PATH = os.path.join(REPORT_DIR, "rl_ner_controller_multinerd_v1.pt")

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

BIG_M = 10_000_000
MAX_CONTEXT_CHARS = BIG_M
MAX_TOKENS_PER_SAMPLE = BIG_M
MIN_TOKENS_PER_SAMPLE = 1

STOP_QUALITY_THRESHOLD = 0.90
STOP_REWARD = 1.0
BAD_STOP_PENALTY = -1
EARLY_BAD_STOP_EXTRA_PENALTY = -0.10
EARLY_STOP_STEP_THRESHOLD = 2

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY bulunamadı.")

# ============================================================
# LABEL MAP
# ============================================================

LABEL2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-ANIM": 7,
    "I-ANIM": 8,
    "B-BIO": 9,
    "I-BIO": 10,
    "B-CEL": 11,
    "I-CEL": 12,
    "B-DIS": 13,
    "I-DIS": 14,
    "B-EVE": 15,
    "I-EVE": 16,
    "B-FOOD": 17,
    "I-FOOD": 18,
    "B-INST": 19,
    "I-INST": 20,
    "B-MEDIA": 21,
    "I-MEDIA": 22,
    "B-MYTH": 23,
    "I-MYTH": 24,
    "B-PLANT": 25,
    "I-PLANT": 26,
    "B-TIME": 27,
    "I-TIME": 28,
    "B-VEHI": 29,
    "I-VEHI": 30,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

ENTITY_TYPES = sorted(
    list(
        {
            lab.split("-", 1)[1]
            for lab in LABEL2ID
            if lab != "O" and "-" in lab
        }
    )
)

ACTIONS = [
    "STOP",
    "FIX_BOUNDARIES",
    "ADD_MISSING",
    "REMOVE_SPURIOUS",
    "TYPE_CORRECT",
    "REPLACE_WEAKEST_SPAN",
    "REGENERATE",
]
ACTION2IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX2ACTION = {i: a for a, i in ACTION2IDX.items()}

ACTION_COST = 0.01

GOLD_DELTA_WEIGHTS = {
    "quality": 0.35,
    "entity_f1": 0.35,
    "entity_recall": 0.15,
    "entity_precision": 0.10,
    "token_accuracy": 0.05,
}

# ============================================================
# LOGGER
# ============================================================

def setup_logger():
    logger = logging.getLogger("rl_multinerd_ner_v1")
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

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())

def truncate_text(s: str, n: int = 300) -> str:
    s = normalize_text(s)
    return s if FULL_LOG_TEXT else (s[:n] + ("..." if len(s) > n else ""))

# ============================================================
# NER HELPERS
# ============================================================

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def clamp_label_id(x: int) -> int:
    return x if x in ID2LABEL else 0

def label_name(x: int) -> str:
    return ID2LABEL.get(clamp_label_id(x), "O")

def labels_to_text(tag_ids: List[int]) -> str:
    return "[" + ", ".join(str(clamp_label_id(x)) for x in tag_ids) + "]"

def labels_to_names(tag_ids: List[int]) -> str:
    return "[" + ", ".join(label_name(x) for x in tag_ids) + "]"

def tokens_to_text(tokens: List[str]) -> str:
    return " ".join(tokens)

def compact_token_table(tokens: List[str], tag_ids: List[int], max_items: int = BIG_M) -> str:
    rows = []
    for i, (tok, tg) in enumerate(zip(tokens[:max_items], tag_ids[:max_items])):
        rows.append(f"{i}: {tok} -> {tg} ({label_name(tg)})")
    return "\n".join(rows)

def align_tag_length(tag_ids: List[int], target_len: int) -> List[int]:
    tag_ids = [clamp_label_id(safe_int(x, 0)) for x in tag_ids]
    if len(tag_ids) == target_len:
        return tag_ids
    if len(tag_ids) > target_len:
        return tag_ids[:target_len]
    return tag_ids + [0] * (target_len - len(tag_ids))

def parse_model_tag_array(output: str, target_len: int) -> List[int]:
    text = output.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()

    if text.startswith("[") and "]" in text:
        try:
            arr = ast.literal_eval(text[text.find("["): text.rfind("]") + 1])
            if isinstance(arr, list):
                parsed = []
                for x in arr:
                    if isinstance(x, int):
                        parsed.append(clamp_label_id(x))
                    elif isinstance(x, str):
                        xs = x.strip()
                        if xs.isdigit():
                            parsed.append(clamp_label_id(int(xs)))
                        else:
                            parsed.append(clamp_label_id(LABEL2ID.get(xs, 0)))
                    else:
                        parsed.append(0)
                return align_tag_length(parsed, target_len)
        except Exception:
            pass

    nums = re.findall(r"-?\d+", text)
    if nums:
        parsed = [clamp_label_id(int(x)) for x in nums]
        return align_tag_length(parsed, target_len)

    toks = re.findall(r"[BIO]-[A-Z]+|O", text)
    if toks:
        parsed = [clamp_label_id(LABEL2ID.get(x, 0)) for x in toks]
        return align_tag_length(parsed, target_len)

    return [0] * target_len

def bio_prefix_and_type(label_id: int) -> Tuple[str, Optional[str]]:
    lab = label_name(label_id)
    if lab == "O":
        return "O", None
    if "-" not in lab:
        return "O", None
    prefix, ent_type = lab.split("-", 1)
    return prefix, ent_type

def fix_invalid_bio(tags: List[int]) -> List[int]:
    out = []
    prev_type = None
    prev_open = False

    for t in tags:
        prefix, ent_type = bio_prefix_and_type(t)
        if prefix == "O":
            out.append(0)
            prev_type = None
            prev_open = False
            continue

        if prefix == "B":
            out.append(t)
            prev_type = ent_type
            prev_open = True
            continue

        if prev_open and prev_type == ent_type:
            out.append(t)
        else:
            out.append(LABEL2ID.get(f"B-{ent_type}", 0))
        prev_type = ent_type
        prev_open = True

    return out

def extract_entities_from_tags(tokens: List[str], tags: List[int]) -> List[Dict[str, Any]]:
    tags = align_tag_length(tags, len(tokens))
    tags = fix_invalid_bio(tags)

    ents = []
    i = 0
    n = len(tokens)

    while i < n:
        prefix, ent_type = bio_prefix_and_type(tags[i])
        if prefix == "B" and ent_type is not None:
            start = i
            j = i + 1
            while j < n:
                p2, t2 = bio_prefix_and_type(tags[j])
                if p2 == "I" and t2 == ent_type:
                    j += 1
                else:
                    break
            ents.append({
                "start": start,
                "end": j - 1,
                "type": ent_type,
                "text": " ".join(tokens[start:j]),
            })
            i = j
        else:
            i += 1
    return ents

def entity_to_tuple(ent: Dict[str, Any]) -> Tuple[int, int, str]:
    return (ent["start"], ent["end"], ent["type"])

def entity_metrics(tokens: List[str], pred_tags: List[int], gold_tags: List[int]) -> Dict[str, float]:
    pred_ents = extract_entities_from_tags(tokens, pred_tags)
    gold_ents = extract_entities_from_tags(tokens, gold_tags)

    pred_set = {entity_to_tuple(e) for e in pred_ents}
    gold_set = {entity_to_tuple(e) for e in gold_ents}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return {
        "entity_precision": prec,
        "entity_recall": rec,
        "entity_f1": f1,
        "pred_entity_count": len(pred_ents),
        "gold_entity_count": len(gold_ents),
        "matched_entities": tp,
    }

def token_accuracy(pred_tags: List[int], gold_tags: List[int]) -> float:
    pred_tags = align_tag_length(pred_tags, len(gold_tags))
    if not gold_tags:
        return 1.0
    correct = sum(1 for p, g in zip(pred_tags, gold_tags) if p == g)
    return correct / len(gold_tags)

def non_o_precision(pred_tags: List[int], gold_tags: List[int]) -> float:
    pred_non_o = sum(1 for p in pred_tags if p != 0)
    if pred_non_o == 0:
        return 1.0 if all(g == 0 for g in gold_tags) else 0.0
    correct_non_o = sum(1 for p, g in zip(pred_tags, gold_tags) if p != 0 and p == g)
    return correct_non_o / pred_non_o

def non_o_recall(pred_tags: List[int], gold_tags: List[int]) -> float:
    gold_non_o = sum(1 for g in gold_tags if g != 0)
    if gold_non_o == 0:
        return 1.0
    correct_non_o = sum(1 for p, g in zip(pred_tags, gold_tags) if g != 0 and p == g)
    return correct_non_o / gold_non_o

def invalid_i_ratio(tags: List[int]) -> float:
    if not tags:
        return 0.0
    bad = 0
    fixed = fix_invalid_bio(tags)
    for a, b in zip(tags, fixed):
        if a != b:
            bad += 1
    return bad / len(tags)

def o_ratio(tags: List[int]) -> float:
    if not tags:
        return 0.0
    return sum(1 for t in tags if t == 0) / len(tags)

def changed_tag_count(old_tags: List[int], new_tags: List[int]) -> int:
    L = min(len(old_tags), len(new_tags))
    return sum(1 for i in range(L) if old_tags[i] != new_tags[i]) + abs(len(old_tags) - len(new_tags))

def tag_diff_summary(tokens: List[str], old_tags: List[int], new_tags: List[int], max_items: int = BIG_M) -> Dict[str, Any]:
    L = min(len(tokens), len(old_tags), len(new_tags))
    changed = []
    for i in range(L):
        if old_tags[i] != new_tags[i]:
            changed.append({
                "index": i,
                "token": tokens[i],
                "old": old_tags[i],
                "old_label": label_name(old_tags[i]),
                "new": new_tags[i],
                "new_label": label_name(new_tags[i]),
            })
    return {
        "changed_tag_count": len(changed),
        "changed_positions": [x["index"] for x in changed[:max_items]],
        "changed_items": changed[:max_items],
    }

_EMPTY_CHANGED = {
    "changed_tag_count": 0,
    "changed_positions": [],
    "changed_items": [],
}

def compute_quality(tokens: List[str], pred_tags: List[int], gold_tags: List[int]) -> Dict[str, float]:
    pred_tags = fix_invalid_bio(align_tag_length(pred_tags, len(tokens)))
    gold_tags = fix_invalid_bio(align_tag_length(gold_tags, len(tokens)))

    ent_m = entity_metrics(tokens, pred_tags, gold_tags)
    tok_acc = token_accuracy(pred_tags, gold_tags)
    nz_prec = non_o_precision(pred_tags, gold_tags)
    nz_rec = non_o_recall(pred_tags, gold_tags)
    inv_i = invalid_i_ratio(pred_tags)
    o_r = o_ratio(pred_tags)

    quality = (
        0.45 * ent_m["entity_f1"]
        + 0.20 * ent_m["entity_precision"]
        + 0.20 * ent_m["entity_recall"]
        + 0.10 * tok_acc
        + 0.03 * nz_prec
        + 0.02 * nz_rec
        - 0.03 * inv_i
    )

    return {
        **ent_m,
        "token_accuracy": tok_acc,
        "non_o_precision": nz_prec,
        "non_o_recall": nz_rec,
        "invalid_i_ratio": inv_i,
        "o_ratio": o_r,
        "quality": quality,
        "pred_non_o_count": sum(1 for x in pred_tags if x != 0),
        "gold_non_o_count": sum(1 for x in gold_tags if x != 0),
    }

def metrics_delta(old: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    return {f"delta_{k}": v - old.get(k, 0.0) for k, v in new.items() if isinstance(v, (int, float))}

# ============================================================
# DATASET
# ============================================================

def example_id_from_row(split_name: str, idx: int, lang: str) -> str:
    return f"{split_name}_{lang}_{idx}"

def prepare_example(row: Dict[str, Any], split_name: str, idx: int) -> Optional[Dict[str, Any]]:
    tokens = row.get("tokens", [])
    tags = row.get("ner_tags", [])
    lang = row.get("lang", "unknown")

    if not isinstance(tokens, list) or not isinstance(tags, list):
        return None
    if len(tokens) != len(tags):
        return None
    if len(tokens) < MIN_TOKENS_PER_SAMPLE:
        return None

    tokens = [str(x) for x in tokens[:MAX_TOKENS_PER_SAMPLE]]
    tags = [clamp_label_id(safe_int(x, 0)) for x in tags[:MAX_TOKENS_PER_SAMPLE]]

    return {
        "id": example_id_from_row(split_name, idx, lang),
        "split": split_name,
        "lang": lang,
        "tokens": tokens,
        "text": tokens_to_text(tokens),
        "gold_tags": tags,
    }

def filter_lang(example: Dict[str, Any]) -> bool:
    if TARGET_LANGS is None:
        return True
    return example.get("lang") in TARGET_LANGS

def load_multinerd_examples() -> Tuple[List[dict], List[dict]]:
    logger.info(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    if USE_HF_SPLITS:
        train_src = ds["train"]
        test_src = ds["validation"] if "validation" in ds else ds["test"]

        train_examples = []
        for i, row in enumerate(train_src):
            ex = prepare_example(row, "train", i)
            if ex is None:
                continue
            if not filter_lang(ex):
                continue
            train_examples.append(ex)

        test_examples = []
        for i, row in enumerate(test_src):
            ex = prepare_example(row, "test", i)
            if ex is None:
                continue
            if not filter_lang(ex):
                continue
            test_examples.append(ex)
    else:
        src = ds["train"]
        all_examples = []
        for i, row in enumerate(src):
            ex = prepare_example(row, "all", i)
            if ex is None:
                continue
            if not filter_lang(ex):
                continue
            all_examples.append(ex)
        random.shuffle(all_examples)
        split = max(1, min(int(len(all_examples) * TRAIN_RATIO), len(all_examples) - 1))
        train_examples, test_examples = all_examples[:split], all_examples[split:]

    if not train_examples or not test_examples:
        raise ValueError("Multinerd'den yeterli örnek yüklenemedi. TARGET_LANGS / split ayarlarını kontrol et.")

    return train_examples, test_examples

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
        temperature: float = 0.4,
        max_tokens: int = 30000,
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
You are a strict named entity recognition evaluator.

You are evaluating a token-level NER tag array without gold labels.
Your role is to assess the current prediction, not to instruct or control the editor.

IMPORTANT:
- Return ONLY valid JSON.
- Scores must be floats between 0.00 and 1.00.
- Avoid extreme 0.00 or 1.00 unless strongly justified.
- Keep text fields concise.
- Do not provide step-by-step editing instructions.
- Do not recommend specific actions.
- Focus on observable issues in the current prediction.

The labels are:
0 O
1 B-PER
2 I-PER
3 B-ORG
4 I-ORG
5 B-LOC
6 I-LOC
7 B-ANIM
8 I-ANIM
9 B-BIO
10 I-BIO
11 B-CEL
12 I-CEL
13 B-DIS
14 I-DIS
15 B-EVE
16 I-EVE
17 B-FOOD
18 I-FOOD
19 B-INST
20 I-INST
21 B-MEDIA
22 I-MEDIA
23 B-MYTH
24 I-MYTH
25 B-PLANT
26 I-PLANT
27 B-TIME
28 I-TIME
29 B-VEHI
30 I-VEHI
""".strip()

def normalize_short_list(items: Any, limit: int = 6) -> List[str]:
    if not isinstance(items, list):
        return []
    out = []
    seen = set()
    for x in items:
        s = normalize_text(x)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:limit]

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

    raise ValueError(f"Judge JSON parse failed: {text[:500]}")

class Judge:
    def __init__(self, llm: OpenRouterLLM):
        self.llm = llm

    def evaluate(self, tokens: List[str], pred_tags: List[int]) -> Dict[str, Any]:
        pred_tags = fix_invalid_bio(align_tag_length(pred_tags, len(tokens)))
        pred_ents = extract_entities_from_tags(tokens, pred_tags)
        inv_ratio = invalid_i_ratio(pred_tags)
        non_o = sum(1 for x in pred_tags if x != 0)

        token_lines = compact_token_table(tokens, pred_tags)
        entity_preview = json.dumps(pred_ents[:50], ensure_ascii=False)

        prompt = f"""
Evaluate this token-level NER prediction.

TOKENS COUNT: {len(tokens)}
PREDICTED NON-O TAG COUNT: {non_o}
INVALID-I RATIO: {inv_ratio:.3f}

TOKEN -> TAG TABLE:
{token_lines}

PREDICTED ENTITY PREVIEW:
{entity_preview}

Return ONLY this JSON schema:
{{
  "overall_score": 0.00,
  "groundedness": 0.00,
  "coverage": 0.00,
  "boundary_quality": 0.00,
  "type_correctness": 0.00,
  "schema_compliance": 0.00,
  "consistency": 0.00,
  "duplicate_risk": 0.00,
  "unsupported_risk": 0.00,
  "edit_pressure": "medium",
  "major_failures": ["..."],
  "weak_spans": ["..."],
  "protected_spans": ["..."],
  "missing_concepts": ["..."],
  "reason_short": "..."
}}

Scoring guidance:
- groundedness: tagged entities should be directly supported by the tokens
- coverage: obvious entities should not be missed
- boundary_quality: B/I boundaries should fit the span exactly
- type_correctness: entity type should be the most plausible one for the span
- schema_compliance: BIO and label ids should be valid
- consistency: repeated patterns should be tagged similarly when context matches
- duplicate_risk: over-tagging or repetitive spurious tagging
- unsupported_risk: hallucinated or weakly supported entity spans

Field guidance:
- major_failures: major visible problems in the current tagging
- weak_spans: suspicious spans or regions
- protected_spans: spans that appear plausible and stable
- missing_concepts: likely missing entities or regions, described neutrally
- reason_short: short overall assessment
""".strip()

        raw = self.llm.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1800,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )

        try:
            parsed = extract_json_block(raw)
        except Exception:
            logger.warning(f"Judge JSON parse failed, fallback used. raw tail: {raw[-300:]}")
            parsed = {}

        judge = {
            "overall_score": min(max(safe_float(parsed.get("overall_score", 0.0)), 0.0), 1.0),
            "groundedness": min(max(safe_float(parsed.get("groundedness", 0.0)), 0.0), 1.0),
            "coverage": min(max(safe_float(parsed.get("coverage", 0.0)), 0.0), 1.0),
            "boundary_quality": min(max(safe_float(parsed.get("boundary_quality", 0.0)), 0.0), 1.0),
            "type_correctness": min(max(safe_float(parsed.get("type_correctness", 0.0)), 0.0), 1.0),
            "schema_compliance": min(max(safe_float(parsed.get("schema_compliance", 0.0)), 0.0), 1.0),
            "consistency": min(max(safe_float(parsed.get("consistency", 0.0)), 0.0), 1.0),
            "duplicate_risk": min(max(safe_float(parsed.get("duplicate_risk", 0.0)), 0.0), 1.0),
            "unsupported_risk": min(max(safe_float(parsed.get("unsupported_risk", 0.0)), 0.0), 1.0),
            "edit_pressure": normalize_text(parsed.get("edit_pressure", "medium")).lower() or "medium",
            "major_failures": normalize_short_list(parsed.get("major_failures", []), limit=6),
            "weak_spans": normalize_short_list(parsed.get("weak_spans", []), limit=8),
            "protected_spans": normalize_short_list(parsed.get("protected_spans", []), limit=8),
            "missing_concepts": normalize_short_list(parsed.get("missing_concepts", []), limit=6),
            "reason_short": normalize_text(parsed.get("reason_short", "")),
        }

        if judge["edit_pressure"] not in {"low", "medium", "high"}:
            judge["edit_pressure"] = "medium"

        return judge

def judge_delta(old_judge: Dict[str, Any], new_judge: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "overall_score",
        "groundedness",
        "coverage",
        "boundary_quality",
        "type_correctness",
        "schema_compliance",
        "consistency",
        "duplicate_risk",
        "unsupported_risk",
    ]
    return {f"delta_{k}": new_judge.get(k, 0.0) - old_judge.get(k, 0.0) for k in keys}

# ============================================================
# PROMPTS
# ============================================================

def token_index_block(tokens: List[str], max_items: int = BIG_M) -> str:
    return "\n".join(f"{i}: {tok}" for i, tok in enumerate(tokens[:max_items]))

def initial_prompt(tokens: List[str]) -> str:
    token_lines = token_index_block(tokens)
    return f"""
You are performing token-level NER.

Return a JSON array of EXACTLY {len(tokens)} integers.

Label schema:
0 O
1 B-PER
2 I-PER
3 B-ORG
4 I-ORG
5 B-LOC
6 I-LOC
7 B-ANIM
8 I-ANIM
9 B-BIO
10 I-BIO
11 B-CEL
12 I-CEL
13 B-DIS
14 I-DIS
15 B-EVE
16 I-EVE
17 B-FOOD
18 I-FOOD
19 B-INST
20 I-INST
21 B-MEDIA
22 I-MEDIA
23 B-MYTH
24 I-MYTH
25 B-PLANT
26 I-PLANT
27 B-TIME
28 I-TIME
29 B-VEHI
30 I-VEHI

STRICT RULES:
- Output ONLY the JSON integer array
- Length must be exactly {len(tokens)}

TOKENS:
{token_lines}
""".strip()

def format_reflection_history_brief(reflection_history: List[Dict[str, Any]], last_k: int = 2) -> str:
    if not reflection_history:
        return "No previous refinement attempts."
    lines = []
    for idx, item in enumerate(reflection_history[-last_k:], 1):
        jd = item.get("judge_deltas", {})
        changed = item.get("changed_tags", {}).get("changed_tag_count", 0)
        lines.append(
            f"{idx}. action={item.get('action','')}, "
            f"delta_overall={jd.get('delta_overall_score',0.0):+.4f}, "
            f"delta_coverage={jd.get('delta_coverage',0.0):+.4f}, "
            f"changed={changed}"
        )
    return "\n".join(lines)

def build_edit_prompt(
    action: str,
    tokens: List[str],
    current_tags: List[int],
    judge_metrics: Dict[str, Any],
    reflection_history: List[Dict[str, Any]],
) -> str:
    action_instr = {
        "FIX_BOUNDARIES": (
            "Fix BIO boundary mistakes and adjust span start/end positions where the evidence supports it. "
            "Do not change entity type unless boundary repair necessarily forces a BIO correction."
        ),
        "ADD_MISSING": (
            "Add missing entities that are supported by the tokens and context. "
            "Keep unsupported additions out of the sequence."
        ),
        "REMOVE_SPURIOUS": (
            "Remove unsupported, weak, or implausible entity tags and convert those positions to O where appropriate."
        ),
        "TYPE_CORRECT": (
            "Correct entity types where the span appears plausible but the assigned entity type is likely wrong."
        ),
        "REPLACE_WEAKEST_SPAN": (
            "Replace the most suspicious span or region with the most plausible tagging pattern for that region."
        ),
        "REGENERATE": (
            "Recompute the entire tag sequence from the tokens and produce the most plausible full tagging."
        ),
    }.get(action, "Improve the tag sequence.")

    token_lines = token_index_block(tokens)
    tag_lines = compact_token_table(tokens, current_tags)

    return f"""
Task: {action}

Current tag array:
{labels_to_text(current_tags)}

Current token->tag view:
{tag_lines}

Judge observations:
- reason: {judge_metrics.get("reason_short", "")}
- major_failures: {judge_metrics.get("major_failures", [])}
- weak_spans: {judge_metrics.get("weak_spans", [])}
- protected_spans: {judge_metrics.get("protected_spans", [])}
- missing_concepts: {judge_metrics.get("missing_concepts", [])}
- edit_pressure: {judge_metrics.get("edit_pressure", "medium")}

Recent refinement history:
{format_reflection_history_brief(reflection_history, last_k=2)}

Action instruction:
{action_instr}

Label schema:
0 O
1 B-PER
2 I-PER
3 B-ORG
4 I-ORG
5 B-LOC
6 I-LOC
7 B-ANIM
8 I-ANIM
9 B-BIO
10 I-BIO
11 B-CEL
12 I-CEL
13 B-DIS
14 I-DIS
15 B-EVE
16 I-EVE
17 B-FOOD
18 I-FOOD
19 B-INST
20 I-INST
21 B-MEDIA
22 I-MEDIA
23 B-MYTH
24 I-MYTH
25 B-PLANT
26 I-PLANT
27 B-TIME
28 I-TIME
29 B-VEHI
30 I-VEHI

TOKENS:
{token_lines}

*** STRICT RULES — MUST FOLLOW EXACTLY ***
- OUTPUT EXACTLY ONE JSON ARRAY OF LENGTH {len(tokens)}
- EVERY ITEM MUST BE AN INTEGER BETWEEN 0 AND 30
- Use BIO tagging correctly
- Output ONLY the JSON array
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
    judge_boundary_quality: float
    judge_type_correctness: float
    judge_schema_compliance: float
    judge_consistency: float
    judge_duplicate_risk: float
    judge_unsupported_risk: float
    pred_non_o_norm: float
    reflection_count_norm: float
    last_delta_judge_overall: float
    last_change_count_norm: float
    remaining_steps_norm: float
    invalid_i_ratio: float
    o_ratio_value: float
    entity_density: float
    judge_momentum: float
    last_action_idx: int

def state_tensor(s: State) -> torch.Tensor:
    numeric = torch.tensor([
        s.step / max(MAX_STEPS, 1),
        s.judge_overall,
        (s.delta_judge_overall + 1.0) / 2.0,
        s.judge_groundedness,
        s.judge_coverage,
        s.judge_boundary_quality,
        s.judge_type_correctness,
        s.judge_schema_compliance,
        s.judge_consistency,
        s.judge_duplicate_risk,
        s.judge_unsupported_risk,
        min(s.pred_non_o_norm / 2.0, 1.0),
        s.reflection_count_norm,
        (s.last_delta_judge_overall + 1.0) / 2.0,
        s.last_change_count_norm,
        s.remaining_steps_norm,
        s.invalid_i_ratio,
        s.o_ratio_value,
        s.entity_density,
        (s.judge_momentum + 1.0) / 2.0,
    ], dtype=torch.float32)

    action_one_hot = torch.zeros(len(ACTIONS), dtype=torch.float32)
    if 0 <= s.last_action_idx < len(ACTIONS):
        action_one_hot[s.last_action_idx] = 1.0

    return torch.cat([numeric, action_one_hot], dim=0).to(DEVICE)

STATE_DIM = 20 + len(ACTIONS)

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
You are a careful NER tag editor.
Return ONLY a JSON array of integer tag ids.
No explanation. No markdown. No bullets.
""".strip()

class Env:
    def __init__(self, main: OpenRouterLLM, judge: Judge):
        self.main = main
        self.judge = judge

    def initial_answer(self, tokens: List[str]) -> List[int]:
        raw = self.main.generate(
            initial_prompt(tokens),
            temperature=0.40,
            max_tokens=max(2200, len(tokens) * 6),
            system_prompt=EDITOR_SYSTEM_PROMPT,
        )
        return fix_invalid_bio(parse_model_tag_array(raw, len(tokens)))

    def judge_tags(self, tokens: List[str], pred_tags: List[int]) -> Dict[str, Any]:
        return self.judge.evaluate(tokens=tokens, pred_tags=pred_tags)

    def step(
        self,
        action: str,
        tokens: List[str],
        current_tags: List[int],
        reflection_history: List[Dict[str, Any]],
        judge_metrics: Dict[str, Any],
    ) -> List[int]:
        if action == "STOP":
            return current_tags

        prompt = build_edit_prompt(action, tokens, current_tags, judge_metrics, reflection_history)
        logger.info(f"EDIT PROMPT ({action}): {truncate_text(prompt, 5000000)}")

        raw = self.main.generate(
            prompt=prompt,
            temperature=0.40 if action != "REGENERATE" else 0.20,
            max_tokens=max(220, len(tokens) * 6),
            system_prompt=EDITOR_SYSTEM_PROMPT,
        )
        return fix_invalid_bio(parse_model_tag_array(raw, len(tokens)))

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
    pred_tags,
    last_action_idx=0,
) -> State:
    n = len(reflection_history)
    last_delta_j, last_chg_norm = 0.0, 0.0
    if reflection_history:
        last = reflection_history[-1]
        last_delta_j = last.get("judge_deltas", {}).get("delta_overall_score", 0.0)
        last_chg_norm = min(
            last.get("changed_tags", {}).get("changed_tag_count", 0) / max(len(pred_tags), 1),
            1.0
        )

    j_overall = judge_metrics["overall_score"]
    recent_deltas = [x.get("judge_deltas", {}).get("delta_overall_score", 0.0) for x in reflection_history[-3:]]

    pred_non_o = sum(1 for x in pred_tags if x != 0)

    return State(
        step=step,
        judge_overall=j_overall,
        delta_judge_overall=j_overall - prev_judge_overall,
        judge_groundedness=judge_metrics["groundedness"],
        judge_coverage=judge_metrics["coverage"],
        judge_boundary_quality=judge_metrics["boundary_quality"],
        judge_type_correctness=judge_metrics["type_correctness"],
        judge_schema_compliance=judge_metrics["schema_compliance"],
        judge_consistency=judge_metrics["consistency"],
        judge_duplicate_risk=judge_metrics["duplicate_risk"],
        judge_unsupported_risk=judge_metrics["unsupported_risk"],
        pred_non_o_norm=min(pred_non_o / max(len(pred_tags), 1), 1.0),
        reflection_count_norm=min(n / max(MAX_STEPS, 1), 1.0),
        last_delta_judge_overall=last_delta_j,
        last_change_count_norm=last_chg_norm,
        remaining_steps_norm=(MAX_STEPS - step) / max(MAX_STEPS, 1),
        invalid_i_ratio=invalid_i_ratio(pred_tags),
        o_ratio_value=o_ratio(pred_tags),
        entity_density=min(sum(1 for x in pred_tags if x != 0) / max(len(pred_tags), 1), 1.0),
        judge_momentum=sum(recent_deltas) / len(recent_deltas) if recent_deltas else 0.0,
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
    return sum(
        w * (new_eval.get(m, 0.0) - old_eval.get(m, 0.0))
        for m, w in GOLD_DELTA_WEIGHTS.items()
    )

def compute_reward(action: str, old_eval: Dict[str, float], new_eval: Dict[str, float], step: int) -> Tuple[float, Dict[str, float]]:
    if action == "STOP":
        q = old_eval.get("quality", 0.0)

        if q > STOP_QUALITY_THRESHOLD:
            reward = STOP_REWARD
        elif q < 0.75 and step <= EARLY_STOP_STEP_THRESHOLD:
            reward = -1.5
        elif q < 0.5:
            reward = BAD_STOP_PENALTY
            if step <= EARLY_STOP_STEP_THRESHOLD:
                reward += EARLY_BAD_STOP_EXTRA_PENALTY
        else:
            reward = 0.0

        return reward, {
            "gold_component": 0.0,
            "stop_quality": q,
            "bad_stop_penalty": BAD_STOP_PENALTY if q <= STOP_QUALITY_THRESHOLD else 0.0,
            "early_bad_stop_extra_penalty": EARLY_BAD_STOP_EXTRA_PENALTY if (q <= STOP_QUALITY_THRESHOLD and step <= EARLY_STOP_STEP_THRESHOLD) else 0.0,
            "action_cost": 0.0,
            "final_reward": reward,
        }

    gold = compute_gold_delta_component(old_eval, new_eval)
    reward = gold - ACTION_COST
    return reward, {
        "gold_component": gold,
        "action_cost": ACTION_COST,
        "final_reward": reward,
    }

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
    split_name, sample_index, sample_name, step, action, reward, done,
    tokens, tags_before, candidate_tags, active_tags_after, best_tags_after,
    eval_before, eval_candidate, eval_after, best_eval,
    judge_before, judge_candidate, judge_after, best_judge,
    changed_tags, lang="", reward_breakdown=None, epoch=None, phase="refine", stop_reason="",
):
    return {
        "split": split_name,
        "epoch": epoch,
        "sample_index": sample_index,
        "sample_id": sample_name,
        "lang": lang,
        "step": step,
        "phase": phase,
        "action": action,
        "reward": reward,
        "reward_gold_component": None if reward_breakdown is None else reward_breakdown.get("gold_component"),
        "done": done,
        "stop_reason": stop_reason,
        "token_count": len(tokens),
        "tokens_text": tokens_to_text(tokens),
        "tags_before": labels_to_text(tags_before),
        "candidate_tags": labels_to_text(candidate_tags),
        "active_tags_after": labels_to_text(active_tags_after),
        "best_tags_after": labels_to_text(best_tags_after),
        "labels_before": labels_to_names(tags_before),
        "labels_after": labels_to_names(active_tags_after),
        "eval_quality_before": _q(eval_before, "quality"),
        "candidate_eval_quality": _q(eval_candidate, "quality"),
        "active_eval_quality_after": _q(eval_after, "quality"),
        "best_eval_quality_after": _q(best_eval, "quality"),
        "entity_precision_before": _q(eval_before, "entity_precision"),
        "entity_precision_after": _q(eval_after, "entity_precision"),
        "entity_recall_before": _q(eval_before, "entity_recall"),
        "entity_recall_after": _q(eval_after, "entity_recall"),
        "entity_f1_before": _q(eval_before, "entity_f1"),
        "entity_f1_after": _q(eval_after, "entity_f1"),
        "token_accuracy_before": _q(eval_before, "token_accuracy"),
        "token_accuracy_after": _q(eval_after, "token_accuracy"),
        "judge_overall_before": _q(judge_before, "overall_score"),
        "judge_overall_candidate": _q(judge_candidate, "overall_score"),
        "judge_overall_after": _q(judge_after, "overall_score"),
        "judge_best_after": _q(best_judge, "overall_score"),
        "judge_grounded_before": _q(judge_before, "groundedness"),
        "judge_grounded_after": _q(judge_after, "groundedness"),
        "judge_coverage_before": _q(judge_before, "coverage"),
        "judge_coverage_after": _q(judge_after, "coverage"),
        "judge_boundary_before": _q(judge_before, "boundary_quality"),
        "judge_boundary_after": _q(judge_after, "boundary_quality"),
        "judge_type_before": _q(judge_before, "type_correctness"),
        "judge_type_after": _q(judge_after, "type_correctness"),
        "judge_schema_before": _q(judge_before, "schema_compliance"),
        "judge_schema_after": _q(judge_after, "schema_compliance"),
        "judge_consistency_before": _q(judge_before, "consistency"),
        "judge_consistency_after": _q(judge_after, "consistency"),
        "judge_duplicate_risk_before": _q(judge_before, "duplicate_risk"),
        "judge_duplicate_risk_after": _q(judge_after, "duplicate_risk"),
        "judge_unsupported_risk_before": _q(judge_before, "unsupported_risk"),
        "judge_unsupported_risk_after": _q(judge_after, "unsupported_risk"),
        "judge_edit_pressure_before": _q(judge_before, "edit_pressure"),
        "judge_reason_before": _q(judge_before, "reason_short"),
        "judge_reason_candidate": _q(judge_candidate, "reason_short"),
        "judge_major_failures_before": None if judge_before is None else json.dumps(judge_before.get("major_failures", []), ensure_ascii=False),
        "judge_major_failures_candidate": None if judge_candidate is None else json.dumps(judge_candidate.get("major_failures", []), ensure_ascii=False),
        "judge_weak_spans_before": None if judge_before is None else json.dumps(judge_before.get("weak_spans", []), ensure_ascii=False),
        "judge_missing_concepts_before": None if judge_before is None else json.dumps(judge_before.get("missing_concepts", []), ensure_ascii=False),
        "changed_tag_count": changed_tags.get("changed_tag_count", 0),
        "changed_positions": json.dumps(changed_tags.get("changed_positions", []), ensure_ascii=False),
        "changed_items": json.dumps(changed_tags.get("changed_items", []), ensure_ascii=False),
    }

# ============================================================
# REFINE
# ============================================================

def refine_once(
    env: Env,
    action: str,
    tokens: List[str],
    current_tags: List[int],
    current_judge_metrics: Dict[str, Any],
    current_eval_metrics: Dict[str, float],
    gold_tags: List[int],
    reflection_history: List[Dict[str, Any]],
):
    cand_tags = env.step(action, tokens, current_tags, reflection_history, current_judge_metrics)
    cand_eval = compute_quality(tokens, cand_tags, gold_tags)
    cand_judge = env.judge_tags(tokens, cand_tags)
    return cand_tags, cand_eval, cand_judge

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
            tokens = sample["tokens"]
            gold = sample["gold_tags"]
            lang = sample.get("lang", "")
            sample_name = sample.get("id", f"train_{i+1}")

            logger.info(f"TRAIN SAMPLE START | epoch={epoch+1} | sample={i+1}/{len(train_examples)} | id={sample_name} | lang={lang}")
            logger.info(f"TOKENS ({len(tokens)}): {tokens_to_text(tokens)}")
            logger.info(f"GOLD TAGS: {labels_to_text(gold)}")

            pred_tags = env.initial_answer(tokens)
            eval_m = compute_quality(tokens, pred_tags, gold)
            judge_m = env.judge_tags(tokens, pred_tags)

            init_eval = dict(eval_m)
            init_judge = dict(judge_m)
            best_tags = list(pred_tags)
            best_eval = dict(eval_m)
            best_judge = dict(judge_m)
            best_reward = float("-inf")
            prev_j_over = 0.0
            last_a_idx = ACTION2IDX["STOP"]

            logger.info(f"INITIAL TAGS: {labels_to_text(pred_tags)}")
            logger.info(
                "INITIAL | eval_quality=%.4f | entity_f1=%.4f | entity_recall=%.4f | token_acc=%.4f | judge_overall=%.4f",
                eval_m["quality"], eval_m["entity_f1"], eval_m["entity_recall"], eval_m["token_accuracy"], judge_m["overall_score"]
            )

            refl_hist, log_probs, rewards, entropies, act_hist = [], [], [], [], []
            sample_progress = {
                "epoch": epoch + 1,
                "sample_index": i + 1,
                "sample_id": sample_name,
                "lang": lang,
                "tokens": tokens,
                "gold_tags": gold,
                "initial_tags": list(pred_tags),
                "initial_eval_metrics": to_serializable_metrics(init_eval),
                "initial_judge_metrics": to_serializable_metrics(init_judge),
                "steps": [],
            }

            train_step_rows.append(make_step_row_common(
                "train", i + 1, sample_name, 0, "INITIAL", None, False,
                tokens, [], [], pred_tags, best_tags,
                None, eval_m, eval_m, best_eval,
                None, judge_m, judge_m, best_judge,
                _EMPTY_CHANGED, lang=lang, epoch=epoch + 1, phase="initial",
            ))

            stop_reason = ""
            for step in range(MAX_STEPS):
                s = build_state(step, judge_m, prev_j_over, refl_hist, pred_tags, last_a_idx)
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
                    "TRAIN STEP %d | judge=%.4f | coverage=%.4f | boundary=%.4f | type=%.4f | ACTION=%s | PROBS={%s}",
                    step + 1,
                    s.judge_overall,
                    s.judge_coverage,
                    s.judge_boundary_quality,
                    s.judge_type_correctness,
                    action,
                    ", ".join(f"{IDX2ACTION[j]}:{probs[j]:.3f}" for j in range(len(ACTIONS))),
                )

                old_tags, old_eval, old_judge = list(pred_tags), dict(eval_m), dict(judge_m)

                if action == "STOP":
                    reward, rb = compute_reward(action, old_eval, old_eval, step)
                    log_probs.append(lp)
                    rewards.append(reward)
                    stop_reason = "policy_stop"

                    train_step_rows.append(make_step_row_common(
                        "train", i + 1, sample_name, step + 1, action, reward, True,
                        tokens, old_tags, old_tags, old_tags, best_tags,
                        old_eval, old_eval, old_eval, best_eval,
                        old_judge, old_judge, old_judge, best_judge,
                        _EMPTY_CHANGED, lang=lang, reward_breakdown=rb, epoch=epoch + 1, stop_reason=stop_reason,
                    ))

                    sample_progress["steps"].append({
                        "step": step + 1,
                        "action": action,
                        "reward": reward,
                        "done": True,
                        "stop_reason": stop_reason,
                    })
                    break

                cand_tags, cand_eval, cand_judge = refine_once(
                    env, action, tokens, pred_tags, judge_m, eval_m, gold, refl_hist
                )

                reward, rb = compute_reward(action, old_eval, cand_eval, step)
                done = (step == MAX_STEPS - 1)
                gold_deltas = metrics_delta(old_eval, cand_eval)
                j_deltas = judge_delta(old_judge, cand_judge)
                chg_tags = tag_diff_summary(tokens, old_tags, cand_tags)

                logger.info(
                    "TRAIN STEP %d | cand_judge=%.4f | reward=%.4f | entity_f1 Δ=%+.4f | changed=%d",
                    step + 1,
                    cand_judge["overall_score"],
                    reward,
                    gold_deltas.get("delta_entity_f1", 0.0),
                    chg_tags["changed_tag_count"],
                )

                refl_hist.append({
                    "step": step + 1,
                    "action": action,
                    "candidate_tags": list(cand_tags),
                    "gold_deltas": gold_deltas,
                    "judge_deltas": j_deltas,
                    "changed_tags": chg_tags,
                })

                log_probs.append(lp)
                rewards.append(reward)

                prev_j_over = judge_m["overall_score"]
                pred_tags, eval_m, judge_m = cand_tags, cand_eval, cand_judge

                if reward > best_reward + ACCEPT_EPS:
                    best_reward = reward
                    best_tags, best_eval, best_judge = list(pred_tags), dict(eval_m), dict(judge_m)

                train_step_rows.append(make_step_row_common(
                    "train", i + 1, sample_name, step + 1, action, reward, done,
                    tokens, old_tags, cand_tags, pred_tags, best_tags,
                    old_eval, cand_eval, eval_m, best_eval,
                    old_judge, cand_judge, judge_m, best_judge,
                    chg_tags, lang=lang, reward_breakdown=rb, epoch=epoch + 1,
                ))

                sample_progress["steps"].append({
                    "step": step + 1,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "candidate_tags": cand_tags,
                    "active_tags_after": list(pred_tags),
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
                "lang": lang,
                "token_count": len(tokens),
                "initial_eval_quality": init_eval["quality"],
                "final_eval_quality": eval_m["quality"],
                "best_eval_quality": best_eval["quality"],
                "initial_judge_overall": init_judge["overall_score"],
                "final_judge_overall": judge_m["overall_score"],
                "best_judge_overall": best_judge["overall_score"],
                "initial_entity_f1": init_eval["entity_f1"],
                "final_entity_f1": eval_m["entity_f1"],
                "best_entity_f1": best_eval["entity_f1"],
                "initial_entity_recall": init_eval["entity_recall"],
                "final_entity_recall": eval_m["entity_recall"],
                "best_entity_recall": best_eval["entity_recall"],
                "initial_token_accuracy": init_eval["token_accuracy"],
                "final_token_accuracy": eval_m["token_accuracy"],
                "best_token_accuracy": best_eval["token_accuracy"],
                "eval_quality_gain_final_vs_initial": eval_m["quality"] - init_eval["quality"],
                "eval_quality_gain_best_vs_initial": best_eval["quality"] - init_eval["quality"],
                "judge_gain_final_vs_initial": judge_m["overall_score"] - init_judge["overall_score"],
                "judge_gain_best_vs_initial": best_judge["overall_score"] - init_judge["overall_score"],
                "steps_taken": len(act_hist),
                "action_history": " -> ".join(act_hist),
                "final_action": act_hist[-1] if act_hist else "",
                "final_tags": labels_to_text(pred_tags),
                "best_tags": labels_to_text(best_tags),
                "gold_tags": labels_to_text(gold),
                "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
                "sum_reward": sum(rewards) if rewards else 0.0,
                "policy_loss": float(policy_loss.detach().cpu()),
                "entropy_bonus": float(entropy_bonus.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "stop_reason": stop_reason,
            }

            train_sample_rows.append(rec)
            epoch_records.append(rec)

            sample_progress.update({
                "final_tags": list(pred_tags),
                "best_tags": list(best_tags),
                "action_history": act_hist,
                "mean_reward": rec["mean_reward"],
                "stop_reason": stop_reason,
            })
            train_progress["epochs"].append(sample_progress)

            logger.info(f"ACTION HISTORY: {' -> '.join(act_hist)}")
            logger.info(
                "TRAIN SAMPLE END | initial_q=%.4f | final_q=%.4f | best_q=%.4f | initial_j=%.4f | final_j=%.4f | best_j=%.4f | stop_reason=%s",
                init_eval["quality"], eval_m["quality"], best_eval["quality"],
                init_judge["overall_score"], judge_m["overall_score"], best_judge["overall_score"],
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
                    "initial_entity_f1", "final_entity_f1", "best_entity_f1",
                    "initial_entity_recall", "final_entity_recall", "best_entity_recall",
                    "initial_token_accuracy", "final_token_accuracy", "best_token_accuracy",
                    "steps_taken", "sum_reward", "mean_reward", "policy_loss", "entropy_bonus", "loss",
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
        tokens = sample["tokens"]
        gold = sample["gold_tags"]
        lang = sample.get("lang", "")
        sample_name = sample.get("id", f"test_{i+1}")

        logger.info("=" * 100)
        logger.info(f"TEST SAMPLE START | sample={i+1}/{len(test_examples)} | id={sample_name} | lang={lang}")
        logger.info("=" * 100)

        pred_tags = env.initial_answer(tokens)
        eval_m = compute_quality(tokens, pred_tags, gold)
        judge_m = env.judge_tags(tokens, pred_tags)

        init_eval = dict(eval_m)
        init_judge = dict(judge_m)
        best_tags = list(pred_tags)
        best_eval = dict(eval_m)
        best_judge = dict(judge_m)
        best_reward = float("-inf")

        prev_j_over = 0.0
        last_a_idx = ACTION2IDX["STOP"]
        rewards = []
        act_hist = []
        refl_hist = []
        stop_reason = ""

        sample_progress = {
            "sample_index": i + 1,
            "sample_id": sample_name,
            "lang": lang,
            "tokens": tokens,
            "gold_tags": gold,
            "initial_tags": list(pred_tags),
            "initial_eval_metrics": to_serializable_metrics(init_eval),
            "initial_judge_metrics": to_serializable_metrics(init_judge),
            "steps": [],
        }

        test_step_rows.append(make_step_row_common(
            "test", i + 1, sample_name, 0, "INITIAL", None, False,
            tokens, [], [], pred_tags, best_tags,
            None, eval_m, eval_m, best_eval,
            None, judge_m, judge_m, best_judge,
            _EMPTY_CHANGED, lang=lang, phase="initial",
        ))

        for step in range(MAX_STEPS):
            s = build_state(step, judge_m, prev_j_over, refl_hist, pred_tags, last_a_idx)
            policy.eval()
            logits = policy(state_tensor(s))
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            dist = Categorical(logits=logits)
            a_idx = dist.sample().item()
            action = IDX2ACTION[a_idx]
            act_hist.append(action)
            last_a_idx = a_idx

            logger.info(
                "TEST STEP %d | judge=%.4f | coverage=%.4f | boundary=%.4f | type=%.4f | ACTION=%s | PROBS={%s}",
                step + 1,
                s.judge_overall,
                s.judge_coverage,
                s.judge_boundary_quality,
                s.judge_type_correctness,
                action,
                ", ".join(f"{IDX2ACTION[j]}:{probs[j]:.3f}" for j in range(len(ACTIONS))),
            )

            old_tags, old_eval, old_judge = list(pred_tags), dict(eval_m), dict(judge_m)

            if action == "STOP":
                reward, rb = compute_reward(action, old_eval, old_eval, step)
                rewards.append(reward)
                stop_reason = "policy_stop"

                test_step_rows.append(make_step_row_common(
                    "test", i + 1, sample_name, step + 1, action, reward, True,
                    tokens, old_tags, old_tags, old_tags, best_tags,
                    old_eval, old_eval, old_eval, best_eval,
                    old_judge, old_judge, old_judge, best_judge,
                    _EMPTY_CHANGED, lang=lang, reward_breakdown=rb, stop_reason=stop_reason,
                ))

                sample_progress["steps"].append({
                    "step": step + 1,
                    "action": action,
                    "reward": reward,
                    "done": True,
                    "stop_reason": stop_reason,
                })
                break

            cand_tags, cand_eval, cand_judge = refine_once(
                env, action, tokens, pred_tags, judge_m, eval_m, gold, refl_hist
            )

            reward, rb = compute_reward(action, old_eval, cand_eval, step)
            rewards.append(reward)
            gold_deltas = metrics_delta(old_eval, cand_eval)
            j_deltas = judge_delta(old_judge, cand_judge)
            chg_tags = tag_diff_summary(tokens, old_tags, cand_tags)

            logger.info(
                "TEST STEP %d | cand_judge=%.4f | reward=%.4f | entity_f1 Δ=%+.4f | changed=%d",
                step + 1,
                cand_judge["overall_score"],
                reward,
                gold_deltas.get("delta_entity_f1", 0.0),
                chg_tags["changed_tag_count"],
            )

            refl_hist.append({
                "step": step + 1,
                "action": action,
                "candidate_tags": list(cand_tags),
                "gold_deltas": gold_deltas,
                "judge_deltas": j_deltas,
                "changed_tags": chg_tags,
                "reward": reward,
                "reward_gold_component": rb.get("gold_component", 0.0),
            })

            prev_j_over = judge_m["overall_score"]
            pred_tags, eval_m, judge_m = cand_tags, cand_eval, cand_judge

            if reward > best_reward + ACCEPT_EPS:
                best_reward = reward
                best_tags, best_eval, best_judge = list(pred_tags), dict(eval_m), dict(judge_m)

            done = (step == MAX_STEPS - 1)

            test_step_rows.append(make_step_row_common(
                "test", i + 1, sample_name, step + 1, action, reward, done,
                tokens, old_tags, cand_tags, pred_tags, best_tags,
                old_eval, cand_eval, eval_m, best_eval,
                old_judge, cand_judge, judge_m, best_judge,
                chg_tags, lang=lang, reward_breakdown=rb,
            ))

            sample_progress["steps"].append({
                "step": step + 1,
                "action": action,
                "reward": reward,
                "done": done,
                "candidate_tags": cand_tags,
                "active_tags_after": list(pred_tags),
                "judge_deltas": to_serializable_metrics(j_deltas),
                "gold_deltas": to_serializable_metrics(gold_deltas),
            })

            if done:
                stop_reason = "max_steps_reached"
                break

        rec = {
            "sample_index": i + 1,
            "sample_id": sample_name,
            "lang": lang,
            "token_count": len(tokens),
            "initial_eval_quality": init_eval["quality"],
            "final_eval_quality": eval_m["quality"],
            "best_eval_quality": best_eval["quality"],
            "initial_judge_overall": init_judge["overall_score"],
            "final_judge_overall": judge_m["overall_score"],
            "best_judge_overall": best_judge["overall_score"],
            "initial_entity_f1": init_eval["entity_f1"],
            "final_entity_f1": eval_m["entity_f1"],
            "best_entity_f1": best_eval["entity_f1"],
            "initial_entity_recall": init_eval["entity_recall"],
            "final_entity_recall": eval_m["entity_recall"],
            "best_entity_recall": best_eval["entity_recall"],
            "initial_token_accuracy": init_eval["token_accuracy"],
            "final_token_accuracy": eval_m["token_accuracy"],
            "best_token_accuracy": best_eval["token_accuracy"],
            "eval_quality_gain_final_vs_initial": eval_m["quality"] - init_eval["quality"],
            "eval_quality_gain_best_vs_initial": best_eval["quality"] - init_eval["quality"],
            "judge_gain_final_vs_initial": judge_m["overall_score"] - init_judge["overall_score"],
            "judge_gain_best_vs_initial": best_judge["overall_score"] - init_judge["overall_score"],
            "steps_taken": len(act_hist),
            "action_history": " -> ".join(act_hist),
            "final_action": act_hist[-1] if act_hist else "",
            "final_tags": labels_to_text(pred_tags),
            "best_tags": labels_to_text(best_tags),
            "gold_tags": labels_to_text(gold),
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
            "sum_reward": sum(rewards) if rewards else 0.0,
            "stop_reason": stop_reason,
        }

        test_sample_rows.append(rec)
        sample_progress.update({
            "final_tags": list(pred_tags),
            "best_tags": list(best_tags),
            "action_history": act_hist,
            "mean_reward": rec["mean_reward"],
            "stop_reason": stop_reason,
        })
        test_progress["samples"].append(sample_progress)

        logger.info(
            "TEST SAMPLE END | initial_q=%.4f | final_q=%.4f | best_q=%.4f | initial_j=%.4f | final_j=%.4f | best_j=%.4f | stop_reason=%s",
            init_eval["quality"], eval_m["quality"], best_eval["quality"],
            init_judge["overall_score"], judge_m["overall_score"], best_judge["overall_score"],
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
    logger.info("START — RL MultiNERD NER Refinement v1")
    logger.info(f"DEVICE: {DEVICE} | STATE_DIM: {STATE_DIM}")
    logger.info(f"EPOCHS: {EPOCHS} | SMALL_RUN_EPOCHS: {SMALL_RUN_EPOCHS} | MAX_STEPS: {MAX_STEPS}")
    logger.info(f"LR: {LR} | GAMMA: {GAMMA} | ENTROPY_BETA: {ENTROPY_BETA}")
    logger.info(f"MAIN_MODEL: {MAIN_MODEL} | JUDGE_MODEL: {JUDGE_MODEL}")
    logger.info(f"DATASET: {DATASET_NAME}")
    logger.info(f"TARGET_LANGS: {TARGET_LANGS}")
    logger.info(f"ACTIONS ({len(ACTIONS)}): {ACTIONS}")
    logger.info(f"SMALL_RUN: {SMALL_RUN} | TRAIN: {SMALL_TRAIN_SAMPLES} | TEST: {SMALL_TEST_SAMPLES}")
    logger.info(f"MAX_TOKENS_PER_SAMPLE: {MAX_TOKENS_PER_SAMPLE}")
    logger.info(f"MAX_CONTEXT_CHARS: {MAX_CONTEXT_CHARS}")
    logger.info(f"EARLY_STOP_NO_REWARD_IMPROVEMENT_PATIENCE: {EARLY_STOP_NO_REWARD_IMPROVEMENT_PATIENCE}")

    train_examples, test_examples = load_multinerd_examples()
    logger.info(f"Loaded raw examples | Train: {len(train_examples)} | Test: {len(test_examples)}")

    if SMALL_RUN:
        random.shuffle(train_examples)
        random.shuffle(test_examples)
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
        "config": {
            "device": DEVICE,
            "epochs": effective_epochs,
            "max_steps": MAX_STEPS,
            "lr": LR,
            "gamma": GAMMA,
            "entropy_beta": ENTROPY_BETA,
            "main_model": MAIN_MODEL,
            "judge_model": JUDGE_MODEL,
            "dataset_name": DATASET_NAME,
            "target_langs": TARGET_LANGS,
            "actions": ACTIONS,
            "small_run": SMALL_RUN,
            "small_train_samples": SMALL_TRAIN_SAMPLES,
            "small_test_samples": SMALL_TEST_SAMPLES,
            "max_tokens_per_sample": MAX_TOKENS_PER_SAMPLE,
            "max_context_chars": MAX_CONTEXT_CHARS,
        },
    }, COMBINED_PROGRESS_JSON)

    torch.save(policy.state_dict(), MODEL_PATH)
    logger.info(f"MODEL SAVED: {MODEL_PATH}")
    logger.info("DONE")

if __name__ == "__main__":
    main()

