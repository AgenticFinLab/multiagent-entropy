# Entropy Judger

## Overview

The Entropy Judger is a solution-selection algorithm that uses token-level entropy features to pick the most likely correct answer from a set of MAS repeated runs. Instead of treating all candidate answers equally (random selection) or relying on majority vote, the Entropy Judger scores each candidate using a trained XGBoost + LightGBM ensemble and selects the one with the highest predicted probability of being correct.

This validates the paper's claim that entropy dynamics — already collected during normal inference — carry enough signal to consistently improve selection accuracy across all MAS configurations and tasks.

## Directory Structure

```
entropy_judger/
├── train_judger.py          # Step 1: train and serialize the judger from existing data
├── run_single.py            # Wrapper: runs one experiment with overridden save_folder
├── run.sh                   # Step 2: run K=20 repeated inferences via bash
├── evaluate.py              # Step 3: score runs and compute comparison tables
├── models/                  # Frozen model pkl files (auto-created by train_judger.py)
│   ├── xgboost_judger.pkl
│   ├── lightgbm_judger.pkl
│   └── feature_columns.pkl
└── results/
    ├── all_runs_scored.csv  # Cached scored runs (auto-created by evaluate.py)
    └── tables/
        ├── best_of_k.csv    # Best-of-K comparison table
        ├── best_of_k.txt    # Human-readable version for paper
        ├── early_stop.csv   # Early-Stop accuracy vs cost table
        └── early_stop.txt
```

## Experiment Design

### Data Flow

```
existing merged_datasets.csv
        │
        ▼
  train_judger.py  ──►  frozen XGBoost + LightGBM pkl
        
  run.sh  ──►  K=20 repeated inferences
               (experiments/results/raw/.../entropy_judger_run{k}_*)
        │
        ▼
  evaluate.py  ──►  evaluation pipeline extracts entropy features per run
                ──►  frozen judger scores each (sample, run) pair
                ──►  comparison tables
```

### Three-Step Pipeline

**Step 1 — Train Judger (`train_judger.py`)**

Trains on `data_mining/data/merged_datasets.csv` (all existing single-run experiments across 6 datasets × 5 models × 5 architectures). The judger is trained once and then frozen — it is never retrained on the repeated-run data, ensuring no data leakage.

- Features: all numeric entropy features from the aggregated CSV, plus one-hot encoded `dataset`, `model_name`, `architecture`
- Target: `is_finally_correct`
- Ensemble: average of XGBoost and LightGBM predicted probabilities

**Step 2 — Run K=20 Repeats (`run.sh` + `run_single.py`)**

Calls `entropy_judger/run_single.py` K times per architecture per dataset. `run_single.py` is a thin wrapper around `experiments/scripts/run_experiment.py` that overrides `save_folder` so results land in `experiments/results_entropy_judger/raw/` instead of the default `experiments/results/`. Each run gets a distinct experiment name `run{k}_{dataset}_{model}_{arch}`, making the run index parseable by `evaluate.py`.

**Step 3 — Evaluate (`evaluate.py`)**

For each (dataset, model, architecture, sample) group of 20 runs:

| Strategy | Description |
|----------|-------------|
| **Random@K** | Mean accuracy when picking one of the first K runs uniformly at random |
| **MajVote@K** | Majority vote over the first K runs |
| **Judger Best-of-K** | Pick the run with the highest `prob_correct` among the first K |
| **Oracle@K** | 1 if any of the first K runs is correct (upper bound) |
| **Early-Stop(θ)** | Scan runs in order; stop at the first run where `prob_correct ≥ θ` |

## Usage

### Prerequisites

```bash
pip install xgboost lightgbm
```

### Step 1 — Train the Judger

Run from the repository root:

```bash
python entropy_judger/train_judger.py \
  --data-path data_mining/data/merged_datasets.csv \
  --output-dir entropy_judger/models/
```

Output:
- `entropy_judger/models/xgboost_judger.pkl`
- `entropy_judger/models/lightgbm_judger.pkl`
- `entropy_judger/models/feature_columns.pkl`

### Step 2 — Run Repeated Inferences

```bash
bash entropy_judger/run.sh
```

**Controlling scope via environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS` | all 5 models | Space-separated model config names (without `.yml`) |
| `DATASETS` | all 6 datasets | Space-separated dataset config names |
| `ARCHITECTURES` | all 5 architectures | Space-separated architecture names |
| `K` | `20` | Number of repeated runs per configuration |
| `DATA_NUM` | _(use dataset config)_ | Per-dataset sample counts, format `dataset:N,dataset:N` |
| `DATA_NUM_DEFAULT` | _(use dataset config)_ | Fallback sample count for datasets not listed in `DATA_NUM` |

Examples:

```bash
# Quick validation: 1 model, 1 dataset, 3 runs, 50 samples each
MODELS="qwen3-4b" DATASETS="gsm8k" K=3 DATA_NUM_DEFAULT=50 bash entropy_judger/run.sh

# Different sample counts per dataset
DATA_NUM="gsm8k:100,math500:50,aime2024:30" bash entropy_judger/run.sh

# Full run with default sample counts from each dataset config
bash entropy_judger/run.sh
```

Results are written to `experiments/results_entropy_judger/raw/{dataset}/{model}/run{k}_{dataset}_{model}_{arch}_{timestamp}/`.

### Step 3 — Evaluate

```bash
python entropy_judger/evaluate.py \
  --runs-dir experiments/results_entropy_judger/raw/ \
  --models-dir entropy_judger/models/ \
  --output-dir entropy_judger/results/tables/
```

On the first run, all experiment directories are processed and results are cached to `entropy_judger/results/all_runs_scored.csv`. Subsequent evaluations with different `--k-values` or `--thresholds` load from the cache directly.

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--k-values` | `1 3 5 10 20` | K values for Best-of-K evaluation |
| `--thresholds` | `0.5 0.6 0.7 0.8 0.9` | θ values for Early-Stop evaluation |
| `--scored-cache` | `entropy_judger/results/all_runs_scored.csv` | Cache path; delete to force re-scoring |

## Output Tables

### Best-of-K Table (`best_of_k.txt`)

Rows are (dataset, architecture) pairs; columns compare Random, Majority Vote, Judger, and Oracle at each K value. The table is split into LLaMA and Qwen model family sub-tables.

```
======================================================================
Model Family: Qwen
======================================================================

  K = 5
  Dataset          Architecture   Random  MajVote   Judger   Oracle       Δ
  -------------------------------------------------------------------------
  gsm8k            single          0.712    0.751    0.784    0.891  +0.072
  gsm8k            centralized     0.748    0.779    0.801    0.903  +0.053
  ...
```

### Early-Stop Table (`early_stop.txt`)

Shows the accuracy–cost tradeoff: higher θ means the judger only stops when confident, yielding higher accuracy but more runs on average.

```
  Architecture   θ   Accuracy   Avg Runs
  ----------------------------------------
  centralized  0.50      0.801       3.21
  centralized  0.70      0.834       7.45
  centralized  0.90      0.861      14.02
```

## Relation to Data Mining

The Entropy Judger reuses the same feature engineering and model infrastructure as `data_mining/`:

| Component | Source |
|-----------|--------|
| Feature column list and exclusions | `data_mining/code/features.py: DEFAULT_EXCLUDE_COLUMNS` |
| XGBoost / LightGBM constructors | `data_mining/code/base/model_factory.py: ModelFactory.classifier()` |
| Feature encoding convention | mirrors `data_mining/code/utils.py: encode_categorical_features()` |
| Training data | `data_mining/data/merged_datasets.csv` |

The key difference from the standard data mining classification analysis is:
1. Models are **serialized** so the same trained weights can be applied to new runs
2. Evaluation is at the **pass@K selection** level rather than individual sample classification accuracy
