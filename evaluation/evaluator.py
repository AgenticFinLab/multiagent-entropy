"""Main evaluation script for multi-agent system experiments.

This module provides a command-line interface for analyzing experiment results,
comparing architectures, and generating evaluation reports. The orchestration
logic that is shared with the temperature-ablation evaluator lives in
``base.evaluator.BaseEvaluator``; this file is a thin wrapper that wires up
the standard data loader and the default ``evaluation/results`` output layout.
"""

import json
import argparse
from pathlib import Path

from .base.constants import DATASETS
from .base.evaluator import BaseEvaluator
from .experiment_analyzer import ExperimentAnalyzer
from .entropy_statistic import EntropyStatistic


class StandardEvaluator(BaseEvaluator):
    """Standard CLI evaluator backed by the default ``DataLoader``."""

    def __init__(self, base_path: str, args: argparse.Namespace):
        super().__init__(base_path, args)
        self.analyzer = ExperimentAnalyzer(base_path)
        self.entropy_statistic = EntropyStatistic(base_path)

    # ----- per-dataset orchestration ------------------------------------

    def run_dataset(self, dataset: str) -> None:
        print(f"\nProcessing dataset: {dataset}")
        if self.args.experiment:
            self._run_specific_experiment(dataset)
        else:
            self._run_all_experiments(dataset)

    def _run_specific_experiment(self, dataset: str) -> None:
        if not self.args.model:
            print("Error: --model is required when analyzing a specific experiment")
            return

        for model_name in self.args.model:
            exp_name = self.args.experiment
            print(f"Analyzing experiment: {model_name}/{exp_name}")
            try:
                metrics = self.analyzer.analyze_experiment(
                    dataset,
                    model_name,
                    exp_name,
                    self.args.task_type,
                    self.args.timeout,
                )
                if self.args.output:
                    output_path = Path(self.args.output)
                else:
                    output_dir = self.get_eval_results_path(dataset) / model_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{exp_name}_metrics.json"
                self.analyzer.save_results(metrics, str(output_path))
                print(f"Results saved to: {output_path}")
            except Exception as e:
                print(f"Warning: Experiment analysis failed for {model_name}: {e}")
                print("Continuing with entropy and trend analysis...")

            print(f"\nAnalyzing entropy for experiment: {model_name}/{exp_name}")
            entropy_results = self.entropy_statistic.analyze_experiment_entropy(
                dataset, model_name, exp_name
            )
            entropy_output_dir = (
                self.get_eval_results_path(dataset) / model_name / "entropy"
            )
            entropy_output_dir.mkdir(parents=True, exist_ok=True)
            json_output_path = entropy_output_dir / f"{exp_name}_entropy.json"
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(entropy_results, f, indent=2, ensure_ascii=False)
            print(f"Entropy JSON saved to: {json_output_path}")

            print(
                f"\nAnalyzing entropy change trends for experiment: {model_name}/{exp_name}"
            )
            self.entropy_statistic.analyze_entropy_change_trends(
                dataset, model_name, exp_name
            )

    def _run_all_experiments(self, dataset: str) -> None:
        print(f"Analyzing all experiments for dataset: {dataset}")
        all_metrics = self.analyzer.analyze_all_experiments(
            dataset, self.args.task_type, self.args.timeout, models=self.args.model
        )

        if self.args.output:
            output_path = Path(self.args.output)
        else:
            output_dir = self.get_eval_results_path(dataset)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "all_metrics.json"
        self.analyzer.save_results(all_metrics, str(output_path))
        print(f"All metrics saved to: {output_path}")

        print(f"\nAnalyzing entropy for all experiments in dataset: {dataset}")
        entropy_results = self.entropy_statistic.analyze_all_experiments_entropy(
            dataset, models=self.args.model
        )
        entropy_output_dir = self.get_eval_results_path(dataset)
        entropy_output_dir.mkdir(parents=True, exist_ok=True)
        json_output_path = entropy_output_dir / "all_entropy_results.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(entropy_results, f, indent=2, ensure_ascii=False)
            print(f"Entropy JSON saved to: {json_output_path}")

        # Per-experiment trend analysis appended in-place
        for model_name, model_data in entropy_results["models"].items():
            for exp_name in list(model_data["experiments"].keys()):
                if "error" in model_data["experiments"][exp_name]:
                    continue
                try:
                    trend_results = (
                        self.entropy_statistic.analyze_entropy_change_trends(
                            dataset, model_name, exp_name
                        )
                    )
                    model_data["experiments"][exp_name][
                        "trend_analysis"
                    ] = trend_results
                except Exception as e:
                    print(f"Error analyzing trends for {model_name}/{exp_name}: {e}")
                    model_data["experiments"][exp_name]["trend_analysis"] = {
                        "error": str(e)
                    }

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(entropy_results, f, indent=2, ensure_ascii=False)

    # ----- top-level ----------------------------------------------------

    def _datasets_to_analyze(self) -> list:
        if self.args.all_datasets:
            return DATASETS
        if self.args.datasets:
            return self.args.datasets
        return ["gsm8k"]

    def run(self) -> None:
        datasets_to_analyze = self._datasets_to_analyze()
        for dataset in datasets_to_analyze:
            self.run_dataset(dataset)

        if self.args.run_aggregator:
            datasets = DATASETS if self.args.aggregate_all else datasets_to_analyze
            for dataset in datasets:
                dataset_path = self.get_eval_results_path(dataset)
                ok = self.run_aggregator(
                    dataset_path / "all_metrics.json",
                    dataset_path / "all_entropy_results.json",
                    dataset_path,
                )
                if ok:
                    print(f"CSV generated for {dataset}: {dataset_path}")

        if self.args.generate_summary:
            datasets = DATASETS if self.args.aggregate_all else datasets_to_analyze
            for dataset in datasets:
                dataset_path = self.get_eval_results_path(dataset)
                if (dataset_path / "all_aggregated_data.csv").exists():
                    print(f"\nGenerating summary for {dataset}...")
                    self.run_summary(dataset_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze multi-agent experiment results"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        choices=DATASETS,
        default=["aime2024_16384", "gsm8k", "humaneval"],
        help="Datasets to analyze (space-separated list)",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Analyze all available datasets",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=["qwen3_14b"],
        help="Model names. If not provided, analyze all models",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["math", "code", "option", "finance", "gaia", "auto"],
        default="auto",
        help="Task type (auto to infer from dataset)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Maximum time in seconds to execute code for code tasks",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to analyze (if not provided, analyze all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (if not provided, save to evaluation/results/)",
    )
    parser.add_argument(
        "--run-aggregator",
        default=True,
        help="Run results aggregator to combine metrics and entropy for data mining",
    )
    parser.add_argument(
        "--aggregate-all",
        default=False,
        help="Aggregate results from all datasets",
    )
    parser.add_argument(
        "--generate-summary",
        default=True,
        help="Generate summary CSV from aggregated data",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_path = str(Path(__file__).parent.parent)
    StandardEvaluator(base_path, args).run()


if __name__ == "__main__":
    main()
