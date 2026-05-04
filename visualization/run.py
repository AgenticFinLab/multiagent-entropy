"""CLI entry point for all visualization plots.

Usage:
    python -m visualization.run accuracy
    python -m visualization.run all
    python -m visualization.run arch datasets
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable, Dict

from visualization.config import get_plot_paths


def _build_accuracy(params):
    from visualization.plots.accuracy import AccuracyPlot, report_accuracy
    from visualization.base.data_loaders import load_csv

    df = load_csv(params["csv_path"])
    report_accuracy(df)
    return AccuracyPlot(
        csv_path=params["csv_path"],
        output_dir=params["output_dir"],
        num_rows=params.get("num_rows", 1),
    )


def _build_single_superiority(params):
    from visualization.plots.single_superiority import SingleSuperiorityPlot
    return SingleSuperiorityPlot(
        csv_path=params["csv_path"],
        output_dir=params["output_dir"],
    )


def _build_arch(params):
    from visualization.plots.arch import ArchPlot
    return ArchPlot(
        summary_json_path=params["summary_json_path"],
        results_dir=params["results_dir"],
        accuracy_data_path=params["accuracy_data_path"],
        output_dir=params["output_dir"],
    )


def _build_datasets(params):
    from visualization.plots.datasets import DatasetsPlot
    return DatasetsPlot(
        shap_data_root=params["shap_data_root"],
        accuracy_data_path=params["accuracy_data_path"],
        output_dir=params["output_dir"],
    )


def _build_mas(params):
    from visualization.plots.mas import MASPlot
    return MASPlot(
        results_dir=params["results_dir"],
        exp_key=params["exp_key"],
        feature_importance_path=params["feature_importance_path"],
        output_dir=params["output_dir"],
        top_features=params["top_features"],
    )


def _build_round(params):
    from visualization.plots.round import RoundPlot
    return RoundPlot(
        r2_summary_path=params["r2_summary_path"],
        r5_summary_path=params["r5_summary_path"],
        r5_math500_data_path=params["r5_math500_data_path"],
        r5_aime_data_path=params["r5_aime_data_path"],
        results_dir=params["results_dir"],
        exp_key=params["exp_key"],
        output_dir=params["output_dir"],
    )


def _build_base_model(params):
    from visualization.plots.base_model import BaseModelPlot
    return BaseModelPlot(
        feature_importance_csv=params["feature_importance_csv"],
        shap_results_dir=params["shap_results_dir"],
        merged_data_path=params["merged_data_path"],
        output_dir=params["output_dir"],
    )


def _build_rl_model(params):
    from visualization.plots.rl_model import RLModelPlot
    return RLModelPlot(
        combined_summary_path=params["combined_summary_path"],
        shap_x_test_path=params["shap_x_test_path"],
        shap_values_path=params["shap_values_path"],
        lightgbm_pred_path=params["lightgbm_pred_path"],
        xgboost_pred_path=params["xgboost_pred_path"],
        merged_data_path=params["merged_data_path"],
        output_dir=params["output_dir"],
        top_features=params["top_features"],
    )


def _build_appendix_arch(params):
    from visualization.plots.appendix_arch import AppendixArchPlot
    return AppendixArchPlot(
        data_dir=params["data_dir"],
        output_dir=params["output_dir"],
        eval_dir=params["eval_dir"],
    )


def _build_causal(params):
    from visualization.plots.causal import CausalPlot
    return CausalPlot(
        causal_data_dir=params["causal_data_dir"],
        output_dir=params["output_dir"],
    )


def _build_causal_appendix(params):
    from visualization.plots.causal_appendix import CausalAppendixPlot
    return CausalAppendixPlot(
        causal_data_dir=params["causal_data_dir"],
        output_dir=params["output_dir"],
    )


def _build_gaia(params):
    from visualization.plots.gaia import GAIAPlot
    return GAIAPlot(
        gaia_aggregated_path=params["gaia_aggregated_path"],
        gaia_aggregated_exclude_agent_path=params["gaia_aggregated_exclude_agent_path"],
        nongaia_merged_path=params["nongaia_merged_path"],
        shap_values_path=params["shap_values_path"],
        shap_x_test_path=params["shap_x_test_path"],
        summary_path=params["summary_path"],
        output_dir=params["output_dir"],
    )


PLOT_BUILDERS: Dict[str, Callable] = {
    "accuracy": _build_accuracy,
    "single_superiority": _build_single_superiority,
    "arch": _build_arch,
    "datasets": _build_datasets,
    "mas": _build_mas,
    "round": _build_round,
    "base_model": _build_base_model,
    "rl_model": _build_rl_model,
    "appendix_arch": _build_appendix_arch,
    "causal": _build_causal,
    "causal_appendix": _build_causal_appendix,
    "gaia": _build_gaia,
}


def run_one(plot_name: str) -> bool:
    print(f"\n=== {plot_name} ===")
    try:
        params = get_plot_paths(plot_name)
        plotter = PLOT_BUILDERS[plot_name](params)
        plotter.compose()
        return True
    except Exception as e:
        print(f"[FAILED] {plot_name}: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run visualization plots.")
    parser.add_argument(
        "plots",
        nargs="+",
        help="Plot names to run, or 'all'. Available: " + ", ".join(sorted(PLOT_BUILDERS)),
    )
    args = parser.parse_args(argv)

    if "all" in args.plots:
        targets = list(PLOT_BUILDERS.keys())
    else:
        unknown = [p for p in args.plots if p not in PLOT_BUILDERS]
        if unknown:
            parser.error(
                f"Unknown plot(s): {unknown}. Available: {sorted(PLOT_BUILDERS)}"
            )
        targets = args.plots

    results = {name: run_one(name) for name in targets}

    print("\n=== Summary ===")
    for name, ok in results.items():
        print(f"  {'OK ' if ok else 'FAIL'}  {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
