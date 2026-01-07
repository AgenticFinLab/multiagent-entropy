"""Main evaluation script for multi-agent system experiments.

This module provides a command-line interface for analyzing experiment results,
comparing architectures, and generating evaluation reports.
"""

import json
import argparse
from pathlib import Path

from aggregator import Aggregator
from entropy_analyzer import EntropyAnalyzer
from experiment_analyzer import ExperimentAnalyzer


def main():
    """Main entry point for the evaluation script.

    Parses command-line arguments and performs experiment analysis
    based on the provided options.
    """
    parser = argparse.ArgumentParser(
        description="Analyze multi-agent experiment results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "humaneval", "mmlu", "aime2024", "math500"],
        default="gsm8k",
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["math", "code", "option"],
        default="math",
        help="Task type",
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
        "--analyze-entropy",
        default=True,
        help="Perform entropy statistical analysis",
    )
    parser.add_argument(
        "--save-entropy-json",
        default=True,
        help="Save detailed entropy results to JSON file",
    )
    parser.add_argument(
        "--aggregate",
        default=True,
        help="Aggregate results from metrics files",
    )
    parser.add_argument(
        "--analyze-trends",
        default=True,
        help="Analyze entropy change trends between agents across rounds",
    )
    parser.add_argument(
        "--save-trends-json",
        default=True,
        help="Save detailed trend results to JSON file",
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

    args = parser.parse_args()

    base_path = str(Path(__file__).parent.parent)
    analyzer = ExperimentAnalyzer(base_path)
    entropy_analyzer = None

    if args.analyze_entropy or args.analyze_trends:
        entropy_analyzer = EntropyAnalyzer(base_path)

    if args.experiment:
        print(f"Analyzing experiment: {args.experiment}")
        try:
            metrics = analyzer.analyze_experiment(
                args.dataset, args.experiment, args.task_type
            )

            if args.output:
                output_path = args.output
            else:
                output_dir = Path(base_path) / "evaluation" / "results" / args.dataset
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{args.experiment}_metrics.json"

            analyzer.save_results(metrics, output_path)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Experiment analysis failed: {e}")
            print("Continuing with entropy and trend analysis...")

        if entropy_analyzer:
            print(f"\nAnalyzing entropy for experiment: {args.experiment}")
            entropy_results = entropy_analyzer.analyze_experiment_entropy(
                args.dataset, args.experiment
            )

            entropy_output_dir = (
                Path(base_path) / "evaluation" / "results" / args.dataset / "entropy"
            )
            entropy_output_dir.mkdir(parents=True, exist_ok=True)

            if args.save_entropy_json:
                json_output_path = (
                    entropy_output_dir / f"{args.experiment}_entropy.json"
                )
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(entropy_results, f, indent=2, ensure_ascii=False)
                print(f"Entropy JSON saved to: {json_output_path}")

            if args.analyze_trends:
                print(
                    f"\nAnalyzing entropy change trends for experiment: {args.experiment}"
                )
                trend_results = entropy_analyzer.analyze_entropy_change_trends(
                    args.dataset, args.experiment
                )

    else:
        print(f"Analyzing all experiments for dataset: {args.dataset}")
        all_metrics = analyzer.analyze_all_experiments(args.dataset, args.task_type)

        if args.output:
            output_path = args.output
        else:
            output_dir = Path(base_path) / "evaluation" / "results" / args.dataset
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "all_metrics.json"

        analyzer.save_results(all_metrics, output_path)
        print(f"All metrics saved to: {output_path}")

        if entropy_analyzer:
            print(f"\nAnalyzing entropy for all experiments in dataset: {args.dataset}")
            entropy_results = entropy_analyzer.analyze_all_experiments_entropy(
                args.dataset
            )

            entropy_output_dir = (
                Path(base_path) / "evaluation" / "results" / args.dataset
            )
            entropy_output_dir.mkdir(parents=True, exist_ok=True)

            if args.save_entropy_json:
                json_output_path = entropy_output_dir / "all_entropy_results.json"
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(entropy_results, f, indent=2, ensure_ascii=False)
                print(f"Entropy JSON saved to: {json_output_path}")

            if args.analyze_trends:
                for exp_name in entropy_results["experiments"].keys():
                    if "error" not in entropy_results["experiments"][exp_name]:
                        try:
                            trend_results = (
                                entropy_analyzer.analyze_entropy_change_trends(
                                    args.dataset, exp_name
                                )
                            )
                            entropy_results["experiments"][exp_name][
                                "trend_analysis"
                            ] = trend_results
                        except Exception as e:
                            print(f"Error analyzing trends for {exp_name}: {e}")
                            entropy_results["experiments"][exp_name][
                                "trend_analysis"
                            ] = {"error": str(e)}

                if args.save_entropy_json:
                    json_output_path = entropy_output_dir / "all_entropy_results.json"
                    with open(json_output_path, "w", encoding="utf-8") as f:
                        json.dump(entropy_results, f, indent=2, ensure_ascii=False)

    if args.run_aggregator or args.aggregate_all:
        base_results_path = Path(base_path) / "evaluation" / "results"

        if args.aggregate_all:
            datasets = ["gsm8k", "humaneval", "mmlu", "aime2024", "math500"]
            for dataset in datasets:
                dataset_path = base_results_path / dataset
                entropy_file = dataset_path / "all_entropy_results.json"
                metrics_file = dataset_path / "all_metrics.json"
                output_csv = dataset_path / "aggregated"

                if entropy_file.exists() and metrics_file.exists():
                    converter = Aggregator(
                        str(entropy_file), str(metrics_file), str(output_csv)
                    )
                    converter.generate_aggregated_csvs()
                    print(f"CSV generated for {dataset}: {output_csv}")
        else:
            dataset_path = base_results_path / args.dataset
            entropy_file = dataset_path / "all_entropy_results.json"
            metrics_file = dataset_path / "all_metrics.json"
            output_csv = dataset_path / "aggregated"

            if entropy_file.exists() and metrics_file.exists():
                converter = Aggregator(
                    str(entropy_file), str(metrics_file), str(output_csv)
                )
                converter.generate_aggregated_csvs()
                print(f"CSV generated for {args.dataset}: {output_csv}")


if __name__ == "__main__":
    main()
