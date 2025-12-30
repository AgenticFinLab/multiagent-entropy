#!/usr/bin/env python3
"""
Result aggregator and visualization tool for MultiAgent-Entropy experiments.

This script provides functionality to:
1. Aggregate results from multiple experiments
2. Extract key metrics from experiment data
3. Generate visualizations (plots, charts)
4. Export results in various formats (CSV, JSON)
"""

import os
import csv
import yaml
import json
import glob
import argparse
import logging
from typing import Dict, List, Any

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiments/logs/result_aggregator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize experiment results"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default="experiments/results/raw",
        help="Directory containing raw experiment results",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="experiments/results/aggregated",
        help="Directory to save aggregated results",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "all"],
        default="all",
        help="Output format for aggregated results",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of the results",
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy", "entropy_mean"],
        help="Metrics to extract and visualize",
    )

    parser.add_argument(
        "--experiment-names",
        nargs="+",
        help="Specific experiment names to process (default: all experiments)",
    )

    return parser.parse_args()


def load_experiment_results(experiment_dir: str) -> Dict[str, Any]:
    """Load results from a single experiment directory.

    Args:
        experiment_dir (str): Path to the experiment directory

    Returns:
        Dict[str, Any]: Experiment results and metadata
    """
    results = {}

    # Load configuration if available
    config_path = os.path.join(experiment_dir, "config.yml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            results["config"] = yaml.safe_load(f)

    # Load merged results if available
    merged_results_path = os.path.join(experiment_dir, "Combined_FinalState.json")
    if os.path.exists(merged_results_path):
        with open(merged_results_path, "r", encoding="utf-8") as f:
            results["merged_results"] = json.load(f)

    # Load batch results
    batch_result_files = glob.glob(os.path.join(experiment_dir, "Batch_*_State.json"))
    batch_results = []
    for file_path in batch_result_files:
        with open(file_path, "r", encoding="utf-8") as f:
            batch_results.append(json.load(f))
    results["batch_results"] = batch_results

    # Extract experiment name from directory
    results["experiment_name"] = os.path.basename(experiment_dir)

    return results


def extract_metrics(results: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
    """Extract specified metrics from experiment results.

    Args:
        results (Dict[str, Any]): Experiment results
        metrics (List[str]): Metrics to extract

    Returns:
        Dict[str, Any]: Extracted metrics
    """
    extracted_metrics = {"experiment_name": results.get("experiment_name", "unknown")}

    # Extract from merged results if available
    if "merged_results" in results and "agent_results" in results["merged_results"]:
        agent_results = results["merged_results"]["agent_results"]

        # Calculate accuracy (if answer is correct)
        if "accuracy" in metrics:
            correct_answers = sum(
                1
                for result in agent_results
                if any(v.get("is_correct", False) for v in result.values())
            )
            total_answers = len(agent_results)
            accuracy = correct_answers / total_answers if total_answers > 0 else 0.0
            extracted_metrics["accuracy"] = accuracy

        # Calculate average entropy
        if "entropy_mean" in metrics:
            entropies = []
            for result in agent_results:
                for agent_output in result.values():
                    if "entropy" in agent_output:
                        entropies.append(agent_output["entropy"])
                    # Handle multi-sample entropy (entropy_1, entropy_2, etc.)
                    elif hasattr(agent_output, "keys") or isinstance(
                        agent_output, dict
                    ):
                        for key, value in agent_output.items():
                            if key.startswith("entropy_") and isinstance(
                                value, (int, float)
                            ):
                                entropies.append(value)
            if entropies:
                extracted_metrics["entropy_mean"] = np.mean(entropies)
                extracted_metrics["entropy_std"] = np.std(entropies)
                extracted_metrics["entropy_min"] = np.min(entropies)
                extracted_metrics["entropy_max"] = np.max(entropies)

        # Calculate average response length
        if "response_length_mean" in metrics:
            response_lengths = []
            for result in agent_results:
                for agent_output in result.values():
                    if "answer" in agent_output:
                        response_lengths.append(len(agent_output["answer"]))
                    # Handle multi-sample answers (answer_1, answer_2, etc.)
                    elif hasattr(agent_output, "keys") or isinstance(
                        agent_output, dict
                    ):
                        for key, value in agent_output.items():
                            if key.startswith("answer_") and isinstance(value, str):
                                response_lengths.append(len(value))
            if response_lengths:
                extracted_metrics["response_length_mean"] = np.mean(response_lengths)
                extracted_metrics["response_length_std"] = np.std(response_lengths)

    # Extract from configuration
    if "config" in results:
        config = results["config"]
        if "data" in config:
            extracted_metrics["data_name"] = config["data"].get("data_name", "unknown")
            extracted_metrics["data_num"] = config["data"].get("data_num", "unknown")
            extracted_metrics["batch_size"] = config["data"].get(
                "batch_size", "unknown"
            )

        if "agents" in config:
            # Get first agent to extract common metrics (all agents share same model/entropy config)
            first_agent_name = next(iter(config["agents"].keys()))
            first_agent = config["agents"][first_agent_name]

            extracted_metrics["lm_name"] = first_agent.get("lm_name", "unknown")
            if "entropy_config" in first_agent:
                extracted_metrics["calculate_entropy"] = first_agent[
                    "entropy_config"
                ].get("calculate_entropy", False)
                extracted_metrics["entropy_type"] = first_agent["entropy_config"].get(
                    "entropy_type", "unknown"
                )

        # Add agent type information
        extracted_metrics["agent_type"] = config.get("agent_type", "unknown")

    return extracted_metrics


def aggregate_results(
    results_list: List[Dict[str, Any]], metrics: List[str]
) -> List[Dict[str, Any]]:
    """Aggregate results from multiple experiments.

    Args:
        results_list (List[Dict[str, Any]]): List of experiment results
        metrics (List[str]): Metrics to extract and aggregate

    Returns:
        List[Dict[str, Any]]: Aggregated metrics for all experiments
    """
    aggregated = []

    for results in results_list:
        try:
            metrics_data = extract_metrics(results, metrics)
            aggregated.append(metrics_data)
        except Exception as e:
            logger.error(
                f"Error extracting metrics from {results.get('experiment_name', 'unknown')}: {str(e)}"
            )

    return aggregated


def save_aggregated_to_csv(
    aggregated_results: List[Dict[str, Any]], save_path: str
) -> None:
    """Save aggregated results to a CSV file.

    Args:
        aggregated_results (List[Dict[str, Any]]): Aggregated results
        save_path (str): Path to save CSV file
    """
    if not aggregated_results:
        logger.warning("No aggregated results to save")
        return

    # Get all unique keys from the results
    fieldnames = set()
    for result in aggregated_results:
        fieldnames.update(result.keys())
    fieldnames = sorted(fieldnames)

    with open(save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(result)

    logger.info(f"Aggregated results saved to CSV: {save_path}")


def save_aggregated_to_json(
    aggregated_results: List[Dict[str, Any]], save_path: str
) -> None:
    """Save aggregated results to a JSON file.

    Args:
        aggregated_results (List[Dict[str, Any]]): Aggregated results
        save_path (str): Path to save JSON file
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Aggregated results saved to JSON: {save_path}")


def generate_accuracy_comparison(
    aggregated_results: List[Dict[str, Any]], save_path: str
) -> None:
    """Generate a bar chart comparing accuracy across experiments.

    Args:
        aggregated_results (List[Dict[str, Any]]): Aggregated results
        save_path (str): Path to save the plot
    """
    # Filter results with accuracy data
    results_with_accuracy = [r for r in aggregated_results if "accuracy" in r]
    if not results_with_accuracy:
        logger.warning("No accuracy data available for visualization")
        return

    # Sort results by accuracy
    results_with_accuracy.sort(key=lambda x: x["accuracy"], reverse=True)

    # Prepare data for plotting
    experiment_names = [r["experiment_name"] for r in results_with_accuracy]
    accuracies = [r["accuracy"] for r in results_with_accuracy]

    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=accuracies, y=experiment_names, palette="viridis")
    plt.xlabel("Accuracy")
    plt.ylabel("Experiment")
    plt.title("Accuracy Comparison Across Experiments")
    plt.grid(axis="x", alpha=0.3)

    # Add accuracy values on bars
    for i, (acc, name) in enumerate(zip(accuracies, experiment_names)):
        plt.text(acc + 0.005, i, f"{acc:.4f}", va="center")

    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Accuracy comparison plot saved to: {save_path}")


def generate_entropy_comparison(
    aggregated_results: List[Dict[str, Any]], save_path: str
) -> None:
    """Generate a box plot comparing entropy across experiments.

    Args:
        aggregated_results (List[Dict[str, Any]]): Aggregated results
        save_path (str): Path to save the plot
    """
    # Filter results with entropy data
    results_with_entropy = [r for r in aggregated_results if "entropy_mean" in r]
    if not results_with_entropy:
        logger.warning("No entropy data available for visualization")
        return

    # Prepare data for plotting
    experiment_names = [r["experiment_name"] for r in results_with_entropy]
    entropy_means = [r["entropy_mean"] for r in results_with_entropy]
    entropy_stds = [r["entropy_std"] for r in results_with_entropy]

    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=entropy_means, y=experiment_names, palette="magma", yerr=entropy_stds)
    plt.xlabel("Mean Entropy")
    plt.ylabel("Experiment")
    plt.title("Mean Entropy Comparison Across Experiments")
    plt.grid(axis="x", alpha=0.3)

    # Add entropy values on bars
    for i, (mean, std, name) in enumerate(
        zip(entropy_means, entropy_stds, experiment_names)
    ):
        plt.text(mean + 0.05, i, f"{mean:.4f} ± {std:.4f}", va="center")

    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Entropy comparison plot saved to: {save_path}")


def generate_visualizations(
    aggregated_results: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generate all visualizations for the aggregated results.

    Args:
        aggregated_results (List[Dict[str, Any]]): Aggregated results
        output_dir (str): Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate accuracy comparison
    accuracy_plot_path = os.path.join(output_dir, "accuracy_comparison.png")
    generate_accuracy_comparison(aggregated_results, accuracy_plot_path)

    # Generate entropy comparison
    entropy_plot_path = os.path.join(output_dir, "entropy_comparison.png")
    generate_entropy_comparison(aggregated_results, entropy_plot_path)

    # Generate combined metrics plot (if both accuracy and entropy are available)
    results_with_both = [
        r for r in aggregated_results if "accuracy" in r and "entropy_mean" in r
    ]
    if results_with_both:
        plt.figure(figsize=(10, 8))

        # Prepare data
        x = [r["entropy_mean"] for r in results_with_both]
        y = [r["accuracy"] for r in results_with_both]
        experiment_names = [r["experiment_name"] for r in results_with_both]

        # Create scatter plot
        scatter = sns.scatterplot(x=x, y=y, s=100, alpha=0.7)

        # Add annotations
        for i, name in enumerate(experiment_names):
            plt.annotate(name, (x[i], y[i]), xytext=(5, 5), textcoords="offset points")

        plt.xlabel("Mean Entropy")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Entropy Across Experiments")
        plt.grid(alpha=0.3)

        # Save plot
        combined_plot_path = os.path.join(output_dir, "accuracy_vs_entropy.png")
        plt.tight_layout()
        plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Accuracy vs Entropy plot saved to: {combined_plot_path}")


def main():
    """Main function to aggregate and visualize experiment results."""
    args = parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get experiment directories
    if args.experiment_names:
        experiment_dirs = [
            os.path.join(args.input_dir, name) for name in args.experiment_names
        ]
        # Filter existing directories
        experiment_dirs = [d for d in experiment_dirs if os.path.isdir(d)]
    else:
        experiment_dirs = [
            d for d in glob.glob(os.path.join(args.input_dir, "*")) if os.path.isdir(d)
        ]

    if not experiment_dirs:
        logger.warning(f"No experiment directories found in {args.input_dir}")
        return

    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    # Load experiment results
    all_results = []
    for exp_dir in experiment_dirs:
        try:
            results = load_experiment_results(exp_dir)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error loading results from {exp_dir}: {str(e)}")

    # Aggregate results
    aggregated_results = aggregate_results(all_results, args.metrics)

    if not aggregated_results:
        logger.warning("No metrics extracted from experiment results")
        return

    # Save aggregated results
    timestamp = os.popen("date '+%Y%m%d_%H%M%S'").read().strip()

    if args.format in ["csv", "all"]:
        csv_path = os.path.join(args.output_dir, f"aggregated_results_{timestamp}.csv")
        save_aggregated_to_csv(aggregated_results, csv_path)

    if args.format in ["json", "all"]:
        json_path = os.path.join(
            args.output_dir, f"aggregated_results_{timestamp}.json"
        )
        save_aggregated_to_json(aggregated_results, json_path)

    # Generate visualizations if requested
    if args.visualize:
        visualizations_dir = os.path.join(
            args.output_dir, f"visualizations_{timestamp}"
        )
        generate_visualizations(aggregated_results, visualizations_dir)

    logger.info("Result aggregation and visualization completed successfully")


if __name__ == "__main__":
    main()
