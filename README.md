# RL-Based Refinement of LLM Outputs for NER and Keyword Extraction

Two reinforcement learning experiments that train a small policy to refine outputs from a large language model (LLM), step by step, over a discrete action space.


## Repo layout

```
.
├── README.md
├── keyword_LTR.py      # Keyphrase experiment 
├── NER_LTR.ipynb    # NER experiment
├── BenchmarkTests.py    # Benchmarks
```

The notebooks write all metrics and checkpoints to a `reports/` directory, including:

| File                          | Content                                    |
|-------------------------------|--------------------------------------------|
| `train_step_metrics.csv`      | Per-step action, reward, value             |
| `train_sample_summary.csv`    | Per-sample episode summary                 |
| `train_epoch_summary.csv`     | Per-epoch aggregates                       |
| `test_step_metrics.csv`       | Same, for the test split                   |
| `test_sample_summary.csv`     | Same, for the test split                   |
| `test_final_overview.csv`     | Final scores                               |
| `combined_progress.json`      | Full run history + config                  |
| `*.pt`                        | Trained policy weights                     |
| `training_*.log`              | Run log                                    |


## Setup

Set your OpenRouter API key as an environment variable. The notebooks read it via `os.getenv("OPENROUTER_API_KEY", "")` and will not work without it.

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

The model used for both the main and judge calls is set near the top of each notebook:

```python
MAIN_MODEL  = "openai/gpt-5.4-nano"
JUDGE_MODEL = "openai/gpt-5.4-nano"
```

Swap these for any OpenRouter-compatible model identifier.

### Dataset-specific notes

**NER notebook.** MultiNERD is pulled directly from Hugging Face — nothing to download manually. The target languages are configured in `TARGET_LANGS` (defaults to `zh, pl, it, pt, fr, es, nl, en, de, ru`); set it to `None` to use all languages.

**Keyword notebook.** Expects SemEval-2017 Task 10 to live at `~/Downloads/SemEval2017` with the standard `docsutf8/` and `keys/` subfolders. Edit `SEMEVAL_ROOT` at the top of the notebook if you store it elsewhere.

---

## Running

Both notebooks are single-cell scripts — run the cell top to bottom. Reports and the trained policy are written to the directory configured in `REPORT_DIR` (defaults to `~/Desktop/...` in the originals; **change this to a relative path like `./reports` before running in your own setup**).

### Quick (sanity-check) run

`SMALL_RUN = True` is the default. It trims the run to:

- NER: 800 training samples, 200 test samples, 5 epochs
- Keyword: subset of the dataset, 2 epochs

### Full run

Set `SMALL_RUN = False` and increase `EPOCHS` as desired. Expect the full run to take significantly longer because every action issues an LLM call.

---

## Config (top of each notebook)

| Variable                 | Default | Notes                              |
|--------------------------|---------|------------------------------------|
| `SEED`                   | 42      | Reproducibility                    |
| `EPOCHS`                 | 5       | Used when `SMALL_RUN = False`      |
| `MAX_STEPS`              | 10      | Max refinement steps per sample    |
| `LR`                     | 1e-4    | Adam                               |
| `GAMMA`                  | 0.99    | Discount                           |
| `ENTROPY_BETA`           | 0.005   | Entropy bonus                      |
| `STOP_QUALITY_THRESHOLD` | 0.90    | Auto-stop above this score         |

---

## Reproducibility

- Fixed seed (`SEED = 42`) for `random` and `torch`.
- LLM calls are non-deterministic; expect small variance across runs even with the same seed.
- Exact model versions on OpenRouter may change over time — pin the model identifier if exact reproduction matters.

---
