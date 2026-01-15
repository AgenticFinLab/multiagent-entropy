"""Main evaluation script for multi-agent system experiments.

This module provides a command-line interface for analyzing experiment results,
comparing architectures, and generating evaluation reports.
"""

import json
import argparse
from pathlib import Path

from aggregator import Aggregator
from experiment_analyzer import ExperimentAnalyzer
from entropy_statistic import EntropyStatistic
from metrics_summary import extract_summary_fields

DATASETS = [
    "gsm8k",
    "humaneval",
    "mmlu",
    "math500",
    "aime2024_8192",
    "aime2025_8192",
    "aime2024_16384",
    "aime2025_16384",
]


def main():
    """Main entry point for the evaluation script.

    Parses command-line arguments and performs experiment analysis
    based on the provided options.
    """
    # Create argument parser with description
    parser = argparse.ArgumentParser(
        description="Analyze multi-agent experiment results"
    )
    # Add datasets argument with multiple choices
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        choices=DATASETS,
        default=["aime2024_8192"],
        help="Datasets to analyze (space-separated list)",
    )
    # Add flag to analyze all datasets
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Analyze all available datasets",
    )
    # Add model argument for specifying which model to analyze
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. If not provided, analyze all models",
    )
    # Add task type argument with choices for auto-detection
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["math", "code", "option", "auto"],
        default="auto",
        help="Task type (auto to infer from dataset)",
    )
    # Add experiment argument for analyzing a specific experiment
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to analyze (if not provided, analyze all)",
    )
    # Add output argument for specifying custom output path
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (if not provided, save to evaluation/results/)",
    )
    # Add flag for running aggregator to combine metrics and entropy
    parser.add_argument(
        "--run-aggregator",
        default=True,
        help="Run results aggregator to combine metrics and entropy for data mining",
    )
    # Add flag for aggregating all datasets
    parser.add_argument(
        "--aggregate-all",
        default=False,
        help="Aggregate results from all datasets",
    )
    # Add flag for generating summary CSV
    parser.add_argument(
        "--generate-summary",
        default=True,
        help="Generate summary CSV from aggregated data",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Determine which datasets to analyze
    if args.all_datasets:
        datasets_to_analyze = DATASETS
    elif args.datasets:
        datasets_to_analyze = args.datasets
    else:
        # Default to aime2025_8192 if nothing specified
        datasets_to_analyze = ["aime2025_8192"]

    # Get base path to project directory
    base_path = str(Path(__file__).parent.parent)

    # Process each dataset
    for dataset in datasets_to_analyze:
        print(f"\nProcessing dataset: {dataset}")

        # Initialize experiment analyzer with base path
        analyzer = ExperimentAnalyzer(base_path)
        # Initialize entropy statistic if needed
        entropy_statistic = None

        # Create entropy statistic instance
        entropy_statistic = EntropyStatistic(base_path)

        # Analyze a specific experiment if experiment name is provided
        if args.experiment:
            # Require model name when analyzing specific experiment
            if not args.model:
                print("Error: --model is required when analyzing a specific experiment")
                return

            print(f"Analyzing experiment: {args.model}/{args.experiment}")
            try:
                # Analyze the specified experiment
                metrics = analyzer.analyze_experiment(
                    dataset, args.model, args.experiment, args.task_type
                )

                # Determine output path for results
                if args.output:
                    output_path = args.output
                else:
                    output_dir = (
                        Path(base_path)
                        / "evaluation"
                        / "results"
                        / dataset
                        / args.model
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{args.experiment}_metrics.json"

                # Save analysis results to JSON file
                analyzer.save_results(metrics, output_path)
                print(f"Results saved to: {output_path}")
            except Exception as e:
                print(f"Warning: Experiment analysis failed: {e}")
                print("Continuing with entropy and trend analysis...")

            # Perform entropy analysis if entropy statistic is available
            if entropy_statistic:
                print(
                    f"\nAnalyzing entropy for experiment: {args.model}/{args.experiment}"
                )
                entropy_results = entropy_statistic.analyze_experiment_entropy(
                    dataset, args.model, args.experiment
                )

                # Create output directory for entropy results
                entropy_output_dir = (
                    Path(base_path)
                    / "evaluation"
                    / "results"
                    / dataset
                    / args.model
                    / "entropy"
                )
                entropy_output_dir.mkdir(parents=True, exist_ok=True)

                # Save entropy results to JSON
                json_output_path = (
                    entropy_output_dir / f"{args.experiment}_entropy.json"
                )
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(entropy_results, f, indent=2, ensure_ascii=False)
                print(f"Entropy JSON saved to: {json_output_path}")

                # Analyze entropy change trends
                print(
                    f"\nAnalyzing entropy change trends for experiment: {args.model}/{args.experiment}"
                )
                trend_results = entropy_statistic.analyze_entropy_change_trends(
                    dataset, args.model, args.experiment
                )

        # Analyze all experiments if no specific experiment is provided
        else:
            print(f"Analyzing all experiments for dataset: {dataset}")
            # Analyze all experiments in the dataset
            all_metrics = analyzer.analyze_all_experiments(dataset, args.task_type)

            # Determine output path for all metrics
            if args.output:
                output_path = args.output
            else:
                output_dir = Path(base_path) / "evaluation" / "results" / dataset
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "all_metrics.json"

            # Save all metrics to JSON file
            analyzer.save_results(all_metrics, output_path)
            print(f"All metrics saved to: {output_path}")

            # Perform entropy analysis for all experiments if entropy statistic is available
            if entropy_statistic:
                print(f"\nAnalyzing entropy for all experiments in dataset: {dataset}")
                entropy_results = entropy_statistic.analyze_all_experiments_entropy(
                    dataset
                )

                # Create output directory for entropy results
                entropy_output_dir = (
                    Path(base_path) / "evaluation" / "results" / dataset
                )
                entropy_output_dir.mkdir(parents=True, exist_ok=True)

                # Save entropy results to JSON
                json_output_path = entropy_output_dir / "all_entropy_results.json"
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(entropy_results, f, indent=2, ensure_ascii=False)
                    print(f"Entropy JSON saved to: {json_output_path}")

                # Analyze entropy change trends for each experiment
                for model_name, model_data in entropy_results["models"].items():
                    for exp_name in model_data["experiments"].keys():
                        # Skip experiments with errors
                        if "error" not in model_data["experiments"][exp_name]:
                            try:
                                trend_results = (
                                    entropy_statistic.analyze_entropy_change_trends(
                                        dataset, model_name, exp_name
                                    )
                                )
                                model_data["experiments"][exp_name][
                                    "trend_analysis"
                                ] = trend_results
                            except Exception as e:
                                print(
                                    f"Error analyzing trends for {model_name}/{exp_name}: {e}"
                                )
                                model_data["experiments"][exp_name][
                                    "trend_analysis"
                                ] = {"error": str(e)}

                # Save updated entropy results with trend analysis
                json_output_path = entropy_output_dir / "all_entropy_results.json"
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(entropy_results, f, indent=2, ensure_ascii=False)

    # Run aggregator to combine metrics and entropy data if requested
    if args.run_aggregator:
        base_results_path = Path(base_path) / "evaluation" / "results"

        # Aggregate all datasets if aggregate_all flag is set
        if args.aggregate_all:
            datasets = DATASETS
            for dataset in datasets:
                dataset_path = base_results_path / dataset
                entropy_file = dataset_path / "all_entropy_results.json"
                metrics_file = dataset_path / "all_metrics.json"
                output_csv = dataset_path

                # Run aggregator if both files exist
                if entropy_file.exists() and metrics_file.exists():
                    converter = Aggregator(
                        str(entropy_file), str(metrics_file), str(output_csv)
                    )
                    converter.generate_aggregated_csvs()
                    print(f"CSV generated for {dataset}: {output_csv}")
        # Aggregate only the specified datasets
        else:
            for dataset in datasets_to_analyze:
                dataset_path = base_results_path / dataset
                entropy_file = dataset_path / "all_entropy_results.json"
                metrics_file = dataset_path / "all_metrics.json"
                output_csv = dataset_path

                # Run aggregator if both files exist
                if entropy_file.exists() and metrics_file.exists():
                    aggregator = Aggregator(
                        str(entropy_file), str(metrics_file), str(output_csv)
                    )
                    aggregator.generate_aggregated_csvs()
                    print(f"CSV generated for {dataset}: {output_csv}")

    # Generate summary CSV from aggregated data if requested
    if args.generate_summary:
        base_results_path = Path(base_path) / "evaluation" / "results"

        # Generate summary for all datasets if aggregate_all flag is set
        if args.aggregate_all:
            datasets = DATASETS
            for dataset in datasets:
                dataset_path = base_results_path / dataset
                input_csv = dataset_path / "all_aggregated_data.csv"
                output_csv = dataset_path / "all_summary_data.csv"

                # Generate summary if input CSV exists
                if input_csv.exists():
                    print(f"\nGenerating summary for {dataset}...")
                    extract_summary_fields(input_csv, output_csv)
        # Generate summary only for the specified datasets
        else:
            for dataset in datasets_to_analyze:
                dataset_path = base_results_path / dataset
                input_csv = dataset_path / "all_aggregated_data.csv"
                output_csv = dataset_path / "all_summary_data.csv"

                # Generate summary if input CSV exists
                if input_csv.exists():
                    print(f"\nGenerating summary for {dataset}...")
                    extract_summary_fields(input_csv, output_csv)


if __name__ == "__main__":
    # Execute main function when script is run directly
    main()
