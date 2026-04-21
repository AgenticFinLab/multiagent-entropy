"""Shared argparse builders for data-mining CLIs.

Consumers: main.py, run_experiments.py, temp_ablation_analysis.py, and the
per-analyzer ``__main__`` blocks. Not all consumers use every group — they
pick the ones they need.

NOTE: These are *additive* helpers. The existing CLI flags on main.py and
run_experiments.py continue to work unchanged; this module just exposes the
same flags in a reusable form so future callers don't reinvent them.
"""

import argparse
from typing import List, Optional


def add_filter_args(p: argparse.ArgumentParser) -> None:
    """Filters: --model-name, --architecture, --dataset, --dataset-type."""
    p.add_argument(
        "--model-name", type=str, nargs="*", default=["all"],
        help="Filter by specific model name(s) (use 'all' for all models)",
    )
    p.add_argument(
        "--architecture", type=str, nargs="*", default=["all"],
        help="Filter by specific architecture(s) (use 'all' for all architectures)",
    )
    p.add_argument(
        "--dataset", type=str, nargs="*", default=["all"],
        help="Filter by specific dataset(s) (use 'all' for all datasets)",
    )
    p.add_argument(
        "--dataset-type", type=str, default="standard",
        choices=["standard", "finagent"],
        help="Dataset type: standard (gsm8k/humaneval etc.) or finagent",
    )


def add_io_args(p: argparse.ArgumentParser) -> None:
    """I/O: --data-path, --output-dir."""
    p.add_argument(
        "--data-path", type=str, default=None,
        help="Path to merged data CSV",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (auto-derived if not provided)",
    )


def add_exclude_features_arg(p: argparse.ArgumentParser) -> None:
    """--exclude-features flag used by all analyzers."""
    p.add_argument(
        "--exclude-features", type=str, default="default",
        help="Feature-exclusion configuration: 'all', 'default', or feature "
             "group name(s) from features.py (comma-separated; '+' to combine).",
    )


def resolve_list_filter(value: Optional[List[str]]) -> Optional[List[str]]:
    """Convert ``['all']`` (or empty) to ``None``; otherwise return the list."""
    if value is None:
        return None
    if value == ["all"] or value == []:
        return None
    return value
