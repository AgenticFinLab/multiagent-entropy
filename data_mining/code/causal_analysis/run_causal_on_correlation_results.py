"""
Run the 4-stage causal analysis pipeline against every existing correlation
slice in ``data_mining/exp_*/results_aggregated/``.

The existing causal scripts in this directory are designed to consume the
output of the ablation analyzer (``feature_rankings_combined.csv`` and
``statistical_selection_*.csv``), which the exp_* folders do not contain.
This orchestrator instead derives the "selected features" directly from the
correlation aggregated CSV (which is already ranked by
``mean_importance_normalized``), then invokes the downstream causal stages
(discovery, effect estimation, mediation, unified report) on the same data
slice that the correlation analysis was run on.

Output layout, per slice ``<slice_id>``:

    data_mining/exp_<NAME>/results_causal/<slice_id>/
        _slice_data.csv
        feature_selection/selected_features.csv
        causal_discovery/...
        causal_effects/...
        mediation/...
        causal_analysis_report.txt
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # causal_analysis/
sys.path.insert(0, str(HERE.parent))     # code/

import features as features_mod

# Heavy causal pipeline deps (causal-learn, dowhy, sklearn, ...) are imported
# lazily inside ``run_pipeline_for_slice`` so that ``--dry-run`` can list
# discovered slices on a Python install that lacks those packages.

# Inlined from feature_selection_crossval.NON_ENTROPY_FEATURES so that we don't
# need to import that module (which pulls in sklearn / matplotlib at load time)
# during a dry run. Keep in sync with feature_selection_crossval.py.
NON_ENTROPY_FEATURES = [
    "base_model_accuracy",
    "base_model_format_compliance_rate",
    "base_model_is_finally_correct",
    "base_model_format_compliance",
    "round_1_total_time",
    "round_2_total_time",
    "round_1_num_inferences",
    "round_2_num_inferences",
    "exp_total_time",
    "exp_total_token",
    "round_1_total_token",
    "round_2_total_token",
    "round_1_2_change_tokens",
    "sample_answer_token_count",
    "base_model_answer_token_count",
    "base_sample_token_count",
    "sample_all_agents_token_count",
    "sample_num_agents",
    "sample_round_1_all_agents_total_token",
    "sample_round_2_all_agents_total_token",
    "sample_round_all_agents_total_token_first_last_diff",
    "sample_round_all_agents_total_token_first_last_ratio",
    "sample_round_1_2_change_tokens",
]

logger = logging.getLogger("run_causal_on_correlation_results")

TARGET_COLUMN = "is_finally_correct"
DEFAULT_MAX_FEATURES = 30
DEFAULT_MIN_ROWS = 200

# Default feature exclusion baseline (mirrors data_mining_analyzer behavior).
_DEFAULT_EXCLUDE_GROUPS = ["experiment_identifier"]


# ---------------------------------------------------------------------------
# Slice discovery and parsing
# ---------------------------------------------------------------------------

_ARCH_EXC_RE = re.compile(r"^arch_(?P<arch>[^_]+(?:_[^_]+)*?)_exclude_(?P<exc>.+)$")
_DATASET_EXC_RE = re.compile(r"^dataset_(?P<ds>.+?)_exclude_(?P<exc>.+)$")
_EXC_RE = re.compile(r"^exclude_(?P<exc>.+)$")


def parse_slice_filename(stem: str) -> Tuple[Optional[str], Optional[str], str]:
    """Extract (architecture, dataset, exclude_features_token) from a slice filename stem.

    Supported patterns (observed across exp_* folders):
        ``arch_<ARCH>_exclude_<EXC>``     -> (ARCH, None, EXC)
        ``dataset_<DS>_exclude_<EXC>``    -> (None, DS, EXC)
        ``exclude_<EXC>``                 -> (None, None, EXC)
        ``<EXC>``                         -> (None, None, EXC)
    """
    m = _ARCH_EXC_RE.match(stem)
    if m:
        return m.group("arch"), None, m.group("exc")
    m = _DATASET_EXC_RE.match(stem)
    if m:
        return None, m.group("ds"), m.group("exc")
    m = _EXC_RE.match(stem)
    if m:
        return None, None, m.group("exc")
    return None, None, stem


def discover_exp_folders(root: Path) -> List[Path]:
    """Find ``exp_*`` folders that have a ``results_aggregated/`` directory.

    Note: not all exp_* folders ship their own ``merged_datasets.csv`` — those
    fall back to ``data_mining/data/merged_datasets.csv`` (resolved by
    ``resolve_merged_csv``).
    """
    folders: List[Path] = []
    for child in sorted(root.glob("exp_*")):
        if not child.is_dir():
            continue
        if (child / "results_aggregated").is_dir():
            folders.append(child)
        else:
            logger.debug("Skip %s (missing results_aggregated/)", child)
    return folders


def resolve_merged_csv(exp_folder: Path, root: Path) -> Optional[Path]:
    """Return the merged dataset CSV for an exp folder.

    Prefers ``<exp>/merged_datasets.csv``; falls back to the shared
    ``<root>/data/merged_datasets.csv``.
    """
    local = exp_folder / "merged_datasets.csv"
    if local.exists():
        return local
    shared = root / "data" / "merged_datasets.csv"
    if shared.exists():
        return shared
    return None


def discover_slices(exp_folder: Path) -> List[Tuple[Path, Optional[str], Optional[str], str]]:
    """Return ``(csv_path, architecture, dataset, exclude_token)`` for each slice CSV."""
    out: List[Tuple[Path, Optional[str], Optional[str], str]] = []
    for csv_path in sorted((exp_folder / "results_aggregated").glob("*.csv")):
        arch, ds, exc = parse_slice_filename(csv_path.stem)
        out.append((csv_path, arch, ds, exc))
    return out


# ---------------------------------------------------------------------------
# Exclude-features token resolution
# ---------------------------------------------------------------------------

def resolve_exclude_columns(token: str) -> List[str]:
    """Translate an exclude_features token into a list of column names.

    Tokens may combine groups via ``+`` or ``,`` (matching
    data_mining_analyzer's convention). Group names refer to the FEATURE_GROUPS
    dict in features.py. ``all`` means no exclusions, ``default`` uses
    ``_DEFAULT_EXCLUDE_GROUPS``.
    """
    if not token or token == "all":
        return []
    parts: List[str] = []
    for chunk in token.replace("+", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == "default":
            parts.extend(_DEFAULT_EXCLUDE_GROUPS)
        else:
            parts.append(chunk)

    feature_groups = getattr(features_mod, "FEATURE_GROUPS", None)
    excluded: List[str] = []
    for group_name in parts:
        # Try FEATURE_GROUPS dict first, then module-level constant
        cols = None
        if isinstance(feature_groups, dict) and group_name in feature_groups:
            cols = feature_groups[group_name]
        else:
            const = getattr(features_mod, group_name.upper(), None)
            if isinstance(const, list):
                cols = const
        if cols is None:
            logger.warning("Unknown exclude group '%s' (token=%r); ignoring.", group_name, token)
            continue
        excluded.extend(cols)
    # Deduplicate, preserve order
    seen = set()
    deduped = []
    for c in excluded:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Slice preparation
# ---------------------------------------------------------------------------

def prepare_slice_data(
    merged_csv: Path,
    architecture: Optional[str],
    dataset: Optional[str],
    exclude_columns: List[str],
    out_csv: Path,
) -> Tuple[Path, int]:
    """Filter merged data and drop excluded + non-numeric columns. Return (path, rows).

    The downstream ``CausalDiscovery.load_data`` casts the entire frame to float,
    so we must drop string columns (architecture, dataset, model_name, ...)
    before writing the slice CSV.
    """
    df = pd.read_csv(merged_csv)
    if architecture:
        if "architecture" not in df.columns:
            raise RuntimeError(
                f"{merged_csv} has no 'architecture' column; cannot filter by {architecture!r}"
            )
        df = df[df["architecture"] == architecture]
    if dataset:
        if "dataset" not in df.columns:
            raise RuntimeError(
                f"{merged_csv} has no 'dataset' column; cannot filter by {dataset!r}"
            )
        df = df[df["dataset"] == dataset]
    drop_cols = [c for c in exclude_columns if c in df.columns and c != TARGET_COLUMN]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Keep numeric columns + target only (downstream casts to float).
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    keep = list(numeric_cols)
    if TARGET_COLUMN in df.columns and TARGET_COLUMN not in keep:
        keep.append(TARGET_COLUMN)
    df = df[keep]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv, len(df)


# ---------------------------------------------------------------------------
# Feature selection from correlation results
# ---------------------------------------------------------------------------

def select_features_from_aggregated(
    agg_csv: Path,
    slice_data_csv: Path,
    output_dir: Path,
    max_features: int,
) -> List[str]:
    """Pick top-N features from the correlation aggregated CSV.

    Ranks by ``mean_importance_normalized`` (falling back to ``mean_importance``
    or ``mean_mean_abs_shap`` if missing). Filters to columns that exist in the
    slice data and are not in the NON_ENTROPY_FEATURES blacklist.

    Writes ``selected_features.csv`` (compatible with the downstream causal
    scripts which expect a ``feature`` column).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    agg = pd.read_csv(agg_csv)

    rank_col = next(
        (c for c in ("mean_importance_normalized", "mean_importance", "mean_mean_abs_shap")
         if c in agg.columns),
        None,
    )
    if rank_col is None:
        raise RuntimeError(f"No importance column found in {agg_csv}")
    agg_sorted = agg.sort_values(rank_col, ascending=False)

    # Read enough rows to detect zero-variance / collinear features. fisherz
    # otherwise blows up with a singular correlation matrix.
    slice_df = pd.read_csv(slice_data_csv)
    available = set(slice_df.columns)

    # Zero-variance columns are useless and break the correlation test.
    nunique = slice_df.nunique(dropna=True)
    zero_var = set(nunique[nunique <= 1].index)

    selected: List[str] = []
    selected_data: List[pd.Series] = []
    CORR_DUP_THRESH = 0.999
    for feat in agg_sorted["feature"].tolist():
        if feat == TARGET_COLUMN:
            continue
        if feat in NON_ENTROPY_FEATURES:
            continue
        if feat not in available:
            continue
        if feat in zero_var:
            logger.debug("Drop %s: zero variance", feat)
            continue
        col = pd.to_numeric(slice_df[feat], errors="coerce")
        if col.notna().sum() == 0:
            continue
        # Reject if perfectly (or near-perfectly) correlated with an already
        # selected feature — keeps the fisherz correlation matrix invertible.
        is_dup = False
        for prev in selected_data:
            mask = col.notna() & prev.notna()
            if mask.sum() < 5:
                continue
            v1 = col[mask]
            v2 = prev[mask]
            if v1.std() == 0 or v2.std() == 0:
                continue
            r = float(v1.corr(v2))
            if abs(r) >= CORR_DUP_THRESH:
                is_dup = True
                logger.debug("Drop %s: |corr|=%.4f with already-selected feature", feat, r)
                break
        if is_dup:
            continue
        selected.append(feat)
        selected_data.append(col)
        if len(selected) >= max_features:
            break

    if not selected:
        raise RuntimeError(f"No usable features after filtering {agg_csv}")

    out_df = pd.DataFrame(
        {
            "feature": selected,
            "borda_score": [
                float(agg_sorted.loc[agg_sorted["feature"] == f, rank_col].iloc[0])
                for f in selected
            ],
            "semantic_group": ["correlation_topk"] * len(selected),
        }
    )
    out_path = output_dir / "selected_features.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Selected %d features -> %s", len(selected), out_path)
    return selected


# ---------------------------------------------------------------------------
# Per-slice pipeline
# ---------------------------------------------------------------------------

def run_pipeline_for_slice(
    *,
    agg_csv: Path,
    merged_csv: Path,
    architecture: Optional[str],
    dataset: Optional[str],
    exclude_token: str,
    output_root: Path,
    max_features: int,
    alpha: float,
    discovery_max_sample: int,
    effect_max_sample: int,
    mediation_max_sample: int,
    n_bootstrap: int,
    min_rows: int,
    skip_existing: bool,
) -> bool:
    """Run all 4 stages + report for a single slice. Return True on success."""
    slice_id = agg_csv.stem
    slice_out = output_root / slice_id
    final_report = slice_out / "causal_analysis_report.txt"
    if skip_existing and final_report.exists():
        logger.info("[%s] skipping (exists): %s", slice_id, final_report)
        return True

    # Lazy imports: only loaded when a slice is actually executed, so that
    # ``--dry-run`` and slice discovery don't require causal-learn / dowhy.
    from causal_discovery import CausalDiscovery
    from causal_effect_estimator import CausalEffectEstimator
    from causal_mediation_analyzer import CausalMediationAnalyzer
    from causal_report_generator import CausalReportGenerator

    slice_out.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 80)
    logger.info("[%s] arch=%s dataset=%s exclude=%s", slice_id, architecture, dataset, exclude_token)

    # 1) Materialize the slice CSV
    slice_data_csv = slice_out / "_slice_data.csv"
    exclude_cols = resolve_exclude_columns(exclude_token)
    _, n_rows = prepare_slice_data(merged_csv, architecture, dataset, exclude_cols, slice_data_csv)
    logger.info("[%s] slice rows=%d (after filtering)", slice_id, n_rows)
    if n_rows < min_rows:
        logger.warning("[%s] only %d rows (< min_rows=%d); skipping", slice_id, n_rows, min_rows)
        return False

    # 2) Feature selection (derived from aggregated correlation CSV)
    fs_dir = slice_out / "feature_selection"
    select_features_from_aggregated(agg_csv, slice_data_csv, fs_dir, max_features)
    feature_list_path = fs_dir / "selected_features.csv"

    # 3) Causal discovery
    disc_dir = slice_out / "causal_discovery"
    CausalDiscovery(
        data_path=str(slice_data_csv),
        feature_list_path=str(feature_list_path),
        output_dir=str(disc_dir),
        alpha=alpha,
        max_sample=discovery_max_sample,
    ).run()

    # 4) Causal effect estimation
    eff_dir = slice_out / "causal_effects"
    CausalEffectEstimator(
        data_path=str(slice_data_csv),
        feature_list_path=str(feature_list_path),
        causes_path=str(disc_dir / "direct_causes.json"),
        edges_path=str(disc_dir / "all_edges.csv"),
        output_dir=str(eff_dir),
        max_sample=effect_max_sample,
    ).run()

    # 5) Mediation
    med_dir = slice_out / "mediation"
    CausalMediationAnalyzer(
        data_path=str(slice_data_csv),
        feature_list_path=str(feature_list_path),
        output_dir=str(med_dir),
        n_bootstrap=n_bootstrap,
        max_sample=mediation_max_sample,
    ).run()

    # 6) Unified report
    CausalReportGenerator(base_dir=str(slice_out)).run()
    logger.info("[%s] done -> %s", slice_id, slice_out)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-mining-root",
        default="data_mining",
        help="Path to the data_mining directory (default: data_mining)",
    )
    parser.add_argument(
        "--exp-folder",
        default=None,
        help="Restrict to a single exp_* folder (path or basename).",
    )
    parser.add_argument(
        "--slice",
        dest="slice_regex",
        default=None,
        help="Optional regex to restrict slices by filename stem.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip slices whose causal_analysis_report.txt already exists.",
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--discovery-max-sample", type=int, default=10000)
    parser.add_argument("--effect-max-sample", type=int, default=15000)
    parser.add_argument("--mediation-max-sample", type=int, default=20000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Path to write the full console output to (stdout+stderr+all logs). "
            "Defaults to data_mining/logs/run_causal_on_correlation_results_<TS>.log"
        ),
    )
    args = parser.parse_args()

    # ----- Logging + tee stdout/stderr to a log file for debugging -----
    from datetime import datetime

    log_path: Optional[Path]
    if args.log_file is None:
        log_path = (
            Path(args.data_mining_root) / "logs"
            / f"run_causal_on_correlation_results_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
    elif args.log_file == "":
        log_path = None
    else:
        log_path = Path(args.log_file)

    log_handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
        log_handlers.append(logging.StreamHandler(log_fh))

        # Tee raw stdout/stderr (e.g. ``print`` calls inside causal scripts).
        class _Tee:
            def __init__(self, *streams):
                self._streams = streams

            def write(self, data):
                for s in self._streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass

            def flush(self):
                for s in self._streams:
                    try:
                        s.flush()
                    except Exception:
                        pass

        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=log_handlers,
        force=True,
    )
    if log_path is not None:
        logger.info("Logging full output to %s", log_path)

    root = Path(args.data_mining_root).resolve()
    if not root.is_dir():
        logger.error("data-mining root not found: %s", root)
        return 2

    if args.exp_folder:
        candidate = Path(args.exp_folder)
        if not candidate.is_absolute():
            candidate = (root / candidate.name) if not candidate.exists() else candidate
        exp_folders = [candidate.resolve()]
    else:
        exp_folders = discover_exp_folders(root)

    if not exp_folders:
        logger.error("No exp_* folders discovered under %s", root)
        return 2

    slice_regex = re.compile(args.slice_regex) if args.slice_regex else None
    plan: List[Tuple[Path, Path, Optional[str], Optional[str], str, Path]] = []
    for exp_folder in exp_folders:
        merged_csv = resolve_merged_csv(exp_folder, root)
        if merged_csv is None:
            logger.warning("Skip %s (no merged_datasets.csv local or shared)", exp_folder)
            continue
        logger.debug("Using merged data for %s: %s", exp_folder.name, merged_csv)
        output_root = exp_folder / "results_causal"
        for csv_path, arch, ds, exc in discover_slices(exp_folder):
            if slice_regex and not slice_regex.search(csv_path.stem):
                continue
            plan.append((csv_path, merged_csv, arch, ds, exc, output_root))

    if not plan:
        logger.error("No slices matched.")
        return 2

    logger.info("Discovered %d slice(s) across %d exp folder(s).", len(plan), len(exp_folders))
    if args.dry_run:
        for agg_csv, merged_csv, arch, ds, exc, output_root in plan:
            logger.info(
                "[plan] %s  ->  %s  (arch=%s dataset=%s exclude=%s)",
                agg_csv.relative_to(root),
                (output_root / agg_csv.stem).relative_to(root),
                arch,
                ds,
                exc,
            )
        return 0

    successes = 0
    failures = 0
    for agg_csv, merged_csv, arch, ds, exc, output_root in plan:
        try:
            ok = run_pipeline_for_slice(
                agg_csv=agg_csv,
                merged_csv=merged_csv,
                architecture=arch,
                dataset=ds,
                exclude_token=exc,
                output_root=output_root,
                max_features=args.max_features,
                alpha=args.alpha,
                discovery_max_sample=args.discovery_max_sample,
                effect_max_sample=args.effect_max_sample,
                mediation_max_sample=args.mediation_max_sample,
                n_bootstrap=args.n_bootstrap,
                min_rows=args.min_rows,
                skip_existing=args.skip_existing,
            )
            if ok:
                successes += 1
            else:
                failures += 1
        except Exception:
            logger.exception("[%s] pipeline failed", agg_csv.stem)
            failures += 1

    logger.info("=" * 80)
    logger.info("Done. successes=%d failures=%d total=%d", successes, failures, len(plan))
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
