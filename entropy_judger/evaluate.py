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

# Categorical columns that were one-hot encoded during training
_CAT_COLS = ["dataset", "model_name", "architecture"]
_ID_COLS = ["sample_id"]

# Prefix that identifies judger experiment folders (exp_name = run{k}_...)
EXP_PREFIX = "run"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _run_evaluation_pipeline(
    exp_dir: Path, dataset: str, model: str
) -> pd.DataFrame | None:
    """
    Run ExperimentAnalyzer → EntropyStatistic → Aggregator on a single
    experiment directory and return the exclude_agent aggregated DataFrame.

    Returns None if the pipeline fails or produces no data.
    """
    import json as _json
    try:
        from evaluation.experiment_analyzer import ExperimentAnalyzer
        from evaluation.entropy_statistic import EntropyStatistic
        from evaluation.aggregator import Aggregator
        from evaluation.base.constants import DATASET_TASK_MAP
        from evaluation.base.data_loader import BaseDataLoader

        task_type = DATASET_TASK_MAP.get(dataset.lower(), "math")

        # exp_dir layout: .../results_entropy_judger/raw/{dataset}/{model}/{exp_name}/
        # DataLoader._get_traces_path resolves:
        #   results_path / dataset / model / exp_name / "traces"
        # So we point results_path at the raw root.
        raw_root = exp_dir.parent.parent.parent  # …/results_entropy_judger/raw

        # Model name → config file stem mapping
        _MODEL_CFG = {
            "llama_3_1_8b_instruct": "llama-3.1-8b-instruct",
            "llama_3_2_3b_instruct": "llama-3.2-3b-instruct",
            "qwen3_0_6b": "qwen3-0.6b",
            "qwen3_4b": "qwen3-4b",
            "qwen3_8b": "qwen3-8b",
            "qwen3_14b": "qwen3-14b",
        }
        # Dataset name → config file stem mapping (data_name may be uppercase)
        _DATASET_CFG = {
            "gsm8k": "gsm8k",
            "math500": "math500",
            "humaneval": "humaneval",
            "mmlu": "mmlu",
            "aime2024_16384": "aime2024",
            "aime2025_16384": "aime2025",
            "aime2024": "aime2024",
            "aime2025": "aime2025",
        }

        def _build_fallback_config(exp_name_: str, dataset_: str, model_: str) -> dict:
            """Reconstruct config from original configs/ when configs_exp is missing."""
            import yaml as _yaml
            configs_dir = _REPO_ROOT / "experiments" / "configs"

            base_cfg = {}
            base_path = configs_dir / "base_config.yml"
            if base_path.exists():
                with open(base_path, encoding="utf-8") as f:
                    base_cfg = _yaml.safe_load(f) or {}

            model_stem = _MODEL_CFG.get(model_, model_.replace("_", "-"))
            model_path = configs_dir / "model_specific" / f"{model_stem}.yml"
            model_cfg = {}
            if model_path.exists():
                with open(model_path, encoding="utf-8") as f:
                    model_cfg = _yaml.safe_load(f) or {}

            ds_stem = _DATASET_CFG.get(dataset_.lower(), dataset_.lower())
            ds_path = configs_dir / "dataset_specific" / f"{ds_stem}.yml"
            ds_cfg = {}
            if ds_path.exists():
                with open(ds_path, encoding="utf-8") as f:
                    ds_cfg = _yaml.safe_load(f) or {}

            # Parse agent_type from exp_name: run{k}_{dataset}_{model}_{arch}_{timestamp}
            agent_type = "single"
            for a in ("sequential", "centralized", "debate", "hybrid", "single"):
                if f"_{a}_" in exp_name_ or exp_name_.split("_")[3] == a:
                    agent_type = a
                    break

            merged = {**base_cfg, **model_cfg, **ds_cfg}
            merged["agent_type"] = agent_type
            # Flatten nested data block fields needed by analyzer
            if "task_type" not in merged and "task_type" in ds_cfg:
                merged["task_type"] = ds_cfg["task_type"]
            return merged

        class _JudgerDataLoader(BaseDataLoader):
            def _init_paths(self_):
                self_.results_path = raw_root
                self_.configs_path = _REPO_ROOT / "experiments" / "configs_exp"
                self_.data_path = _REPO_ROOT / "experiments" / "data"
                self_.results_finagent_path = raw_root
                self_.results_gaia_path = raw_root

            def load_experiment_config(self_, dataset_: str, experiment_name_: str, model_name_: str = None):
                try:
                    return super().load_experiment_config(dataset_, experiment_name_, model_name_)
                except (FileNotFoundError, Exception):
                    return _build_fallback_config(experiment_name_, dataset_, model_name_ or model)

            def load_ground_truth(self_, dataset_: str):
                # Patch dataset_map to cover short names used in our exp dirs
                _extra = {
                    "aime2024": "AIME2024",
                    "aime2025": "AIME2025",
                    "gsm8k": "GSM8K",
                    "humaneval": "HumanEval",
                    "mmlu": "MMLU",
                    "math500": "Math500",
                }
                # Try parent implementation first; fall back with extended map
                try:
                    return super().load_ground_truth(dataset_)
                except FileNotFoundError:
                    folder = _extra.get(dataset_.lower(), dataset_)
                    dataset_path = self_.data_path / folder
                    import glob as _glob
                    files = list(dataset_path.glob("*-all-samples.json"))
                    if not files:
                        raise FileNotFoundError(f"Ground truth file not found in: {dataset_path}")
                    import json as _json2
                    with open(files[0], "r", encoding="utf-8") as f:
                        data = _json2.load(f)
                    if isinstance(data, list):
                        return {str(item["main_id"]): item for item in data}
                    # dict-of-lists format: {"main_id": [...], "groundtruth": [...], ...}
                    if isinstance(data, dict) and "main_id" in data:
                        n = len(data["main_id"])
                        return {
                            str(data["main_id"][i]): {k: data[k][i] for k in data if isinstance(data[k], list) and i < len(data[k])}
                            for i in range(n)
                        }
                    return data

        data_loader = _JudgerDataLoader(str(_REPO_ROOT))
        exp_name = exp_dir.name

        # Step 1: correctness metrics
        analyzer = ExperimentAnalyzer(base_path=str(_REPO_ROOT), data_loader=data_loader)
        metrics = analyzer.analyze_experiment(
            dataset=dataset.lower(),
            model_name=model,
            experiment_name=exp_name,
            task_type=task_type,
        )

        # Step 2: entropy features
        entropy_stat = EntropyStatistic(base_path=str(_REPO_ROOT), data_loader=data_loader)
        entropy_results = entropy_stat.analyze_experiment_entropy(
            dataset=dataset.lower(),
            model_name=model,
            experiment_name=exp_name,
        )

        # Step 3: aggregate into a flat CSV, then read it back as DataFrame
        # Aggregator expects nested structure:
        # metrics: {"dataset":..., "models": {model: {"experiments": {exp_name: data}}}}
        # entropy: {"dataset":..., "models": {model: {"experiments": {exp_name: data}}}}
        tmp_dir = exp_dir / "_eval_tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_metrics = tmp_dir / "all_metrics.json"
        tmp_entropy = tmp_dir / "all_entropy_results.json"

        metrics_wrapped = {
            "dataset": dataset.lower(),
            "task_type": task_type,
            "models": {model: {"experiments": {exp_name: metrics}}},
        }
        entropy_wrapped = {
            "dataset": dataset.lower(),
            "models": {model: {"experiments": {exp_name: entropy_results}}},
            "architectures": [metrics.get("agent_architecture", "unknown")],
        }

        with open(tmp_metrics, "w", encoding="utf-8") as f:
            _json.dump(metrics_wrapped, f)
        with open(tmp_entropy, "w", encoding="utf-8") as f:
            _json.dump(entropy_wrapped, f)

        aggregator = Aggregator(
            entropy_file=str(tmp_entropy),
            metrics_file=str(tmp_metrics),
            output_dir=str(tmp_dir),
        )
        aggregator.generate_aggregated_csvs()

        csv_path = tmp_dir / "all_aggregated_data_exclude_agent.csv"
        if not csv_path.exists():
            # Fall back to the full CSV if exclude_agent variant wasn't generated
            csv_path = tmp_dir / "all_aggregated_data.csv"
        if not csv_path.exists():
            print(f"    [WARN] No aggregated CSV produced for {exp_dir.name}")
            return None

        return pd.read_csv(csv_path)

    except Exception as exc:
        import traceback
        print(f"    [WARN] Pipeline failed for {exp_dir.name}: {exc}")
        print(f"    {traceback.format_exc().splitlines()[-2]}")
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
        raise FileNotFoundError(
            f"Feature columns not found: {feat_path}. Run train_judger.py first."
        )

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
                    run_k = int(exp_name.split("_")[0][len(EXP_PREFIX) :])
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
                    print(
                        f"  [INFO] run_k={run_k} arch={arch}: {len(dirs)} dirs found, using newest: {exp_dir.name}"
                    )

                print(
                    f"  Processing run={run_k} dataset={dataset} model={model_name} arch={arch}"
                )
                df_run = _run_evaluation_pipeline(exp_dir, dataset, model_name)
                if df_run is None or df_run.empty:
                    continue

                # Add identifiers if missing, then defragment
                extra = {}
                if "dataset" not in df_run.columns:
                    extra["dataset"] = dataset
                if "model_name" not in df_run.columns:
                    extra["model_name"] = model_name
                if "architecture" not in df_run.columns:
                    extra["architecture"] = arch
                if extra:
                    df_run = pd.concat(
                        [df_run, pd.DataFrame(extra, index=df_run.index)], axis=1
                    ).copy()

                # One-hot encode categoricals to match training encoding
                df_enc = pd.get_dummies(df_run, columns=_CAT_COLS, drop_first=False)

                # Align to training feature columns (fill missing OHE columns with 0)
                missing = [col for col in feature_cols if col not in df_enc.columns]
                if missing:
                    df_enc = pd.concat(
                        [df_enc, pd.DataFrame(0, index=df_enc.index, columns=missing)],
                        axis=1,
                    )
                X = df_enc[feature_cols].fillna(df_enc[feature_cols].median())

                # Ensemble prediction: average prob across available models
                probs = np.array([m.predict_proba(X)[:, 1] for m in models.values()])
                prob_correct = probs.mean(axis=0)

                for i, row in enumerate(df_run.itertuples(index=False)):
                    records.append(
                        {
                            "run_k": run_k,
                            "dataset": dataset,
                            "model_name": model_name,
                            "architecture": arch,
                            "sample_id": (
                                row.sample_id if hasattr(row, "sample_id") else i
                            ),
                            "is_correct": bool(row.is_finally_correct),
                            "prob_correct": float(prob_correct[i]),
                        }
                    )

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

            rows.append(
                {
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
                }
            )

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

            rows.append(
                {
                    "model_family": model_family,
                    "model_name": model,
                    "dataset": dataset,
                    "architecture": arch,
                    "theta": theta,
                    "Accuracy": np.mean(corrects),
                    "Avg_Runs_Used": np.mean(runs_used),
                }
            )

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
            lines.append(
                f"  {'Dataset':<16} {'Architecture':<14} {'Random':>8} {'MajVote':>8} {'Judger':>8} {'Oracle':>8} {'Δ':>7}"
            )
            lines.append(f"  {'-'*73}")
            for _, r in (
                sub[sub["K"] == k].sort_values(["dataset", "architecture"]).iterrows()
            ):
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
            lines.append(
                f"  {'Architecture':<14} {'θ':>6} {'Accuracy':>10} {'Avg Runs':>10}"
            )
            lines.append(f"  {'-'*44}")
            for _, r in (
                sub[sub["dataset"] == dataset]
                .sort_values(["architecture", "theta"])
                .iterrows()
            ):
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
    parser = argparse.ArgumentParser(
        description="Evaluate Entropy Judger on repeated runs"
    )
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
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Filter by model name(s). Default: all models in scored cache.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Filter by dataset name(s). Default: all datasets in scored cache.",
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
            print(
                "ERROR: No scored runs found. Check --runs-dir and that run.sh has been executed."
            )
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_scored.to_csv(cache_path, index=False)
        print(f"Saved scored runs → {cache_path} ({len(df_scored)} rows)")

    print(
        f"\nScored runs: {len(df_scored)} rows across "
        f"{df_scored['dataset'].nunique()} datasets, "
        f"{df_scored['model_name'].nunique()} models, "
        f"{df_scored['run_k'].max()} max repeats"
    )

    # Apply model/dataset filters
    if args.models:
        unknown = [m for m in args.models if m not in df_scored["model_name"].values]
        if unknown:
            print(f"[WARN] Unknown model(s) ignored: {unknown}")
        df_scored = df_scored[df_scored["model_name"].isin(args.models)]
    if args.datasets:
        unknown = [d for d in args.datasets if d not in df_scored["dataset"].values]
        if unknown:
            print(f"[WARN] Unknown dataset(s) ignored: {unknown}")
        df_scored = df_scored[df_scored["dataset"].isin(args.datasets)]

    if df_scored.empty:
        print("ERROR: No data after filtering. Check --models and --datasets values.")
        return

    if args.models or args.datasets:
        print(
            f"After filtering: {len(df_scored)} rows, "
            f"{df_scored['dataset'].nunique()} dataset(s), "
            f"{df_scored['model_name'].nunique()} model(s)"
        )

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
