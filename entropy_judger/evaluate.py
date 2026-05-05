"""
Evaluate the Entropy Judger on K=20 repeated inference runs.

This script:
1. Discovers all entropy_judger_run* experiments under experiments/results/raw/
2. Runs the evaluation pipeline (ExperimentAnalyzer → EntropyStatistic → Aggregator)
   on each run to extract entropy features in the same format as merged_datasets.csv
3. Applies the frozen judger models to score each (sample, run) pair
4. Computes and saves comparison tables:
   - Best-of-K: Random@K vs Majority-Vote@K vs Judger-Best-of-K vs Oracle@K
   - Early-Stop: accuracy vs avg runs used as a function of threshold θ

Usage:
    python entropy_judger/evaluate.py \
        --runs-dir experiments/results/raw/ \
        --models-dir entropy_judger/models/ \
        --output-dir entropy_judger/results/tables/
"""

import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "data_mining" / "code"))

from features import DEFAULT_EXCLUDE_COLUMNS

# Categorical columns that were one-hot encoded during training
_CAT_COLS = ["dataset", "model_name", "architecture"]
_ID_COLS = ["sample_id"]

# Prefix that identifies judger experiment folders (exp_name = run{k}_...)
EXP_PREFIX = "run"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _run_evaluation_pipeline(exp_dir: Path, dataset: str, model: str) -> pd.DataFrame | None:
    """
    Run ExperimentAnalyzer → EntropyStatistic → Aggregator on a single
    experiment directory and return the aggregated exclude_agent DataFrame.

    Returns None if the pipeline fails or produces no data.
    """
    try:
        from evaluation.experiment_analyzer import ExperimentAnalyzer
        from evaluation.entropy_statistic import EntropyStatistic
        from evaluation.aggregator import Aggregator
        from evaluation.base.constants import DATASET_TASK_TYPE_MAP

        task_type = DATASET_TASK_TYPE_MAP.get(dataset, "math")

        analyzer = ExperimentAnalyzer(
            experiment_dir=str(exp_dir),
            task_type=task_type,
        )
        metrics = analyzer.analyze()

        entropy_stat = EntropyStatistic(experiment_dir=str(exp_dir))
        entropy_results = entropy_stat.compute()

        aggregator = Aggregator(
            metrics=metrics,
            entropy_results=entropy_results,
            experiment_dir=str(exp_dir),
        )
        df = aggregator.aggregate(exclude_agent=True)
        return df
    except Exception as exc:
        print(f"    [WARN] Pipeline failed for {exp_dir.name}: {exc}")
        return None


def collect_scored_runs(
    runs_dir: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """
    Walk runs_dir, find all entropy_judger_run* experiments, extract features,
    score with the frozen judger, return a flat DataFrame with columns:
      run_k, dataset, model_name, sample_id, architecture, is_correct, prob_correct
    """
    # Load frozen models + feature column list
    feat_path = models_dir / "feature_columns.pkl"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {feat_path}. Run train_judger.py first.")

    with open(feat_path, "rb") as f:
        feature_cols = pickle.load(f)

    models = {}
    for name in ("xgboost", "lightgbm"):
        mp = models_dir / f"{name}_judger.pkl"
        if mp.exists():
            with open(mp, "rb") as f:
                models[name] = pickle.load(f)

    if not models:
        raise FileNotFoundError(f"No judger pkl files found in {models_dir}")

    print(f"Loaded judger: {list(models.keys())}, {len(feature_cols)} features")

    records = []

    # Discover all experiment dirs matching the prefix
    # Layout: runs_dir/{dataset}/{model_folder}/{exp_name}_{timestamp}_{ms}_{pid}/
    # If a run_k was retried, multiple dirs share the same logical key — keep only the latest.
    for dataset_dir in sorted(runs_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            # Group dirs by logical run key (run_k, arch); keep only the newest dir per key
            from collections import defaultdict
            key_to_dirs: dict = defaultdict(list)
            for exp_dir in model_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                exp_name = exp_dir.name
                if not exp_name.startswith(EXP_PREFIX):
                    continue
                try:
                    run_k = int(exp_name.split("_")[0][len(EXP_PREFIX):])
                except (IndexError, ValueError):
                    continue
                arch = None
                for a in ("single", "sequential", "centralized", "debate", "hybrid"):
                    if f"_{a}_" in exp_name or exp_name.split("_")[3] == a:
                        arch = a
                        break
                if arch is None:
                    continue
                key_to_dirs[(run_k, arch)].append(exp_dir)

            for (run_k, arch), dirs in sorted(key_to_dirs.items()):
                # Pick the directory with the latest modification time
                exp_dir = max(dirs, key=lambda d: d.stat().st_mtime)
                if len(dirs) > 1:
                    print(f"  [INFO] run_k={run_k} arch={arch}: {len(dirs)} dirs found, using newest: {exp_dir.name}")

                print(f"  Processing run={run_k} dataset={dataset} model={model_name} arch={arch}")
                df_run = _run_evaluation_pipeline(exp_dir, dataset, model_name)
                if df_run is None or df_run.empty:
                    continue

                # Add identifiers if missing
                if "dataset" not in df_run.columns:
                    df_run["dataset"] = dataset
                if "model_name" not in df_run.columns:
                    df_run["model_name"] = model_name
                if "architecture" not in df_run.columns:
                    df_run["architecture"] = arch

                # One-hot encode categoricals to match training encoding
                df_enc = pd.get_dummies(df_run, columns=_CAT_COLS, drop_first=False)

                # Align to training feature columns (fill missing OHE columns with 0)
                for col in feature_cols:
                    if col not in df_enc.columns:
                        df_enc[col] = 0
                X = df_enc[feature_cols].fillna(df_enc[feature_cols].median()).values

                # Ensemble prediction: average prob across available models
                probs = np.array([m.predict_proba(X)[:, 1] for m in models.values()])
                prob_correct = probs.mean(axis=0)

                for i, row in enumerate(df_run.itertuples(index=False)):
                    records.append({
                        "run_k": run_k,
                        "dataset": dataset,
                        "model_name": model_name,
                        "architecture": arch,
                        "sample_id": row.sample_id if hasattr(row, "sample_id") else i,
                        "is_correct": bool(row.is_finally_correct),
                        "prob_correct": float(prob_correct[i]),
                    })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _majority_vote(group: pd.DataFrame, k: int) -> bool:
    """True if the majority of the first k answers are correct."""
    subset = group.head(k)
    return subset["is_correct"].sum() > k / 2


def compute_best_of_k(df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """
    For each (model_family, dataset, sample_id) group, compute:
      Random@K, MajVote@K, Judger-Best-of-K, Oracle@K
    aggregated across questions.
    """
    rows = []
    group_keys = ["dataset", "model_name", "architecture"]

    for (dataset, model, arch), grp in df.groupby(group_keys):
        model_family = "Qwen" if "qwen" in model else "LLaMA"
        # Sub-group by sample_id
        sample_groups = {sid: g for sid, g in grp.groupby("sample_id")}

        for k in k_values:
            randoms, majvotes, judgers, oracles = [], [], [], []
            for sid, sg in sample_groups.items():
                # Take first k runs (sorted by run_k for reproducibility)
                sg_k = sg.sort_values("run_k").head(k)
                if len(sg_k) == 0:
                    continue
                randoms.append(sg_k["is_correct"].mean())
                majvotes.append(float(_majority_vote(sg_k, k)))
                best_idx = sg_k["prob_correct"].idxmax()
                judgers.append(float(sg_k.loc[best_idx, "is_correct"]))
                oracles.append(float(sg_k["is_correct"].any()))

            if not randoms:
                continue

            rows.append({
                "model_family": model_family,
                "model_name": model,
                "dataset": dataset,
                "architecture": arch,
                "K": k,
                "Random@K": np.mean(randoms),
                "MajVote@K": np.mean(majvotes),
                "Judger-Best-of-K": np.mean(judgers),
                "Oracle@K": np.mean(oracles),
                "Delta_vs_Random": np.mean(judgers) - np.mean(randoms),
            })

    return pd.DataFrame(rows)


def compute_early_stop(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    """
    For each sample, scan runs in order. Stop at first run where
    prob_correct >= θ. Record whether that run was correct and
    how many runs were used. If no run meets θ, use the last run.
    """
    rows = []
    group_keys = ["dataset", "model_name", "architecture"]

    for (dataset, model, arch), grp in df.groupby(group_keys):
        model_family = "Qwen" if "qwen" in model else "LLaMA"
        sample_groups = {sid: g for sid, g in grp.groupby("sample_id")}
        K_max = grp["run_k"].max()

        for theta in thresholds:
            corrects, runs_used = [], []
            for sid, sg in sample_groups.items():
                sg_sorted = sg.sort_values("run_k")
                stopped = False
                for _, r in sg_sorted.iterrows():
                    runs_used_so_far = int(r["run_k"])
                    if r["prob_correct"] >= theta:
                        corrects.append(float(r["is_correct"]))
                        runs_used.append(runs_used_so_far)
                        stopped = True
                        break
                if not stopped:
                    # Never met threshold: use last run
                    last = sg_sorted.iloc[-1]
                    corrects.append(float(last["is_correct"]))
                    runs_used.append(K_max)

            if not corrects:
                continue

            rows.append({
                "model_family": model_family,
                "model_name": model,
                "dataset": dataset,
                "architecture": arch,
                "theta": theta,
                "Accuracy": np.mean(corrects),
                "Avg_Runs_Used": np.mean(runs_used),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _format_best_of_k_table(df_bestk: pd.DataFrame, output_dir: Path) -> None:
    """Save Best-of-K results as CSV and human-readable text tables."""
    df_bestk.to_csv(output_dir / "best_of_k.csv", index=False)

    lines = []
    for family in sorted(df_bestk["model_family"].unique()):
        lines.append(f"\n{'='*70}")
        lines.append(f"Model Family: {family}")
        lines.append(f"{'='*70}")
        sub = df_bestk[df_bestk["model_family"] == family]
        for k in sorted(sub["K"].unique()):
            lines.append(f"\n  K = {k}")
            lines.append(f"  {'Dataset':<16} {'Architecture':<14} {'Random':>8} {'MajVote':>8} {'Judger':>8} {'Oracle':>8} {'Δ':>7}")
            lines.append(f"  {'-'*73}")
            for _, r in sub[sub["K"] == k].sort_values(["dataset", "architecture"]).iterrows():
                lines.append(
                    f"  {r['dataset']:<16} {r['architecture']:<14}"
                    f" {r['Random@K']:>8.3f} {r['MajVote@K']:>8.3f}"
                    f" {r['Judger-Best-of-K']:>8.3f} {r['Oracle@K']:>8.3f}"
                    f" {r['Delta_vs_Random']:>+7.3f}"
                )

    txt_path = output_dir / "best_of_k.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved Best-of-K table → {output_dir / 'best_of_k.csv'} and {txt_path}")


def _format_early_stop_table(df_early: pd.DataFrame, output_dir: Path) -> None:
    """Save Early-Stop results as CSV and human-readable text tables."""
    df_early.to_csv(output_dir / "early_stop.csv", index=False)

    lines = []
    for family in sorted(df_early["model_family"].unique()):
        lines.append(f"\n{'='*70}")
        lines.append(f"Model Family: {family}  —  Early-Stop accuracy vs cost")
        lines.append(f"{'='*70}")
        sub = df_early[df_early["model_family"] == family]
        for dataset in sorted(sub["dataset"].unique()):
            lines.append(f"\n  Dataset: {dataset}")
            lines.append(f"  {'Architecture':<14} {'θ':>6} {'Accuracy':>10} {'Avg Runs':>10}")
            lines.append(f"  {'-'*44}")
            for _, r in sub[sub["dataset"] == dataset].sort_values(["architecture", "theta"]).iterrows():
                lines.append(
                    f"  {r['architecture']:<14} {r['theta']:>6.2f}"
                    f" {r['Accuracy']:>10.3f} {r['Avg_Runs_Used']:>10.2f}"
                )

    txt_path = output_dir / "early_stop.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Saved Early-Stop table → {output_dir / 'early_stop.csv'} and {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Entropy Judger on repeated runs")
    parser.add_argument(
        "--runs-dir",
        default="experiments/results_entropy_judger/raw/",
        help="Root directory containing experiment result folders",
    )
    parser.add_argument(
        "--models-dir",
        default="entropy_judger/models/",
        help="Directory with frozen judger pkl files from train_judger.py",
    )
    parser.add_argument(
        "--output-dir",
        default="entropy_judger/results/tables/",
        help="Directory to write output tables",
    )
    parser.add_argument(
        "--scored-cache",
        default="entropy_judger/results/all_runs_scored.csv",
        help="Path to cache/load the scored runs CSV (skips re-scoring if exists)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 20],
        help="K values for Best-of-K evaluation",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="θ thresholds for Early-Stop evaluation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.scored_cache)

    # Step 1: collect & score all runs (or load from cache)
    if cache_path.exists():
        print(f"Loading cached scored runs from {cache_path}")
        df_scored = pd.read_csv(cache_path)
    else:
        print("Collecting and scoring runs ...")
        df_scored = collect_scored_runs(
            runs_dir=Path(args.runs_dir),
            models_dir=Path(args.models_dir),
        )
        if df_scored.empty:
            print("ERROR: No scored runs found. Check --runs-dir and that run.sh has been executed.")
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_scored.to_csv(cache_path, index=False)
        print(f"Saved scored runs → {cache_path} ({len(df_scored)} rows)")

    print(f"\nScored runs: {len(df_scored)} rows across "
          f"{df_scored['dataset'].nunique()} datasets, "
          f"{df_scored['model_name'].nunique()} models, "
          f"{df_scored['run_k'].max()} max repeats")

    # Step 2: Best-of-K comparison
    print("\nComputing Best-of-K metrics ...")
    df_bestk = compute_best_of_k(df_scored, args.k_values)
    _format_best_of_k_table(df_bestk, output_dir)

    # Step 3: Early-Stop accuracy vs cost
    print("Computing Early-Stop metrics ...")
    df_early = compute_early_stop(df_scored, args.thresholds)
    _format_early_stop_table(df_early, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
