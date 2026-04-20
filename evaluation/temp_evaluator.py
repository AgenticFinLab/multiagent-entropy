"""Temperature ablation evaluation script for multi-agent system experiments.

This module provides a command-line interface for evaluating experiments
across different temperature settings (0.4, 0.6, 0.8), generating metrics
and entropy analysis results.
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

from aggregator import Aggregator
from experiment_analyzer import ExperimentAnalyzer
from entropy_statistic import EntropyStatistic
from metrics_summary import extract_summary_fields
from temp_data_loader import TempDataLoader


# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_task_type_from_dataset(dataset: str, task_type: str = "auto") -> str:
    """Determine task type from dataset name.

    Args:
        dataset: Dataset name.
        task_type: Explicit task type or "auto" to infer.

    Returns:
        Task type ("math", "code", or "option").
    """
    if task_type != "auto":
        return task_type

    # Map dataset names to their corresponding task types
    dataset_task_map = {
        "humaneval": "code",
        "mmlu": "option",
        "gsm8k": "math",
        "aime2024_16384": "math",
        "aime2025_16384": "math",
        "math500": "math",
        "aime2024_8192": "math",
        "aime2025_8192": "math",
    }
    return dataset_task_map.get(dataset.lower(), "math")


def process_temperature_0_4_or_0_8(
    base_path: str,
    dataset: str,
    model: str,
    temperature: float,
    task_type: str,
    timeout: int,
    output_dir: Path,
) -> bool:
    """Process experiments for temperature 0.4 or 0.8.

    Args:
        base_path: Base path to the project directory.
        dataset: Dataset name.
        model: Model name to analyze.
        temperature: Temperature value (0.4 or 0.8).
        task_type: Task type for evaluation.
        timeout: Timeout for code execution.
        output_dir: Output directory for results.

    Returns:
        True if processing succeeded, False otherwise.
    """
    logger.info(f"Processing temperature {temperature} for {model}/{dataset}")

    # Initialize TempDataLoader to get completed experiments
    temp_data_loader = TempDataLoader(base_path)
    completed_experiments = temp_data_loader.get_completed_experiments(dataset)

    if model not in completed_experiments:
        logger.warning(f"No completed experiments found for model {model}")
        return False

    # Filter experiments for this temperature
    temp_experiments = []
    temp_prefix = f"t_{str(temperature).replace('.', '_')}_"
    for exp_name in completed_experiments.get(model, []):
        if exp_name.startswith(temp_prefix):
            temp_experiments.append(exp_name)

    if not temp_experiments:
        logger.warning(f"No experiments found for temperature {temperature}")
        return False

    logger.info(f"Found {len(temp_experiments)} experiments for temperature {temperature}")

    # Create ExperimentAnalyzer and replace data_loader
    analyzer = ExperimentAnalyzer(base_path)
    analyzer.data_loader = temp_data_loader

    # Initialize all_metrics structure
    inferred_task_type = get_task_type_from_dataset(dataset, task_type)
    all_metrics: Dict[str, Any] = {
        "dataset": dataset,
        "task_type": inferred_task_type,
        "models": {
            model: {
                "experiments": {}
            }
        }
    }

    # Analyze each experiment
    for exp_name in temp_experiments:
        logger.info(f"  Analyzing experiment: {exp_name}")
        try:
            metrics = analyzer.analyze_experiment(
                dataset, model, exp_name, task_type, timeout
            )
            all_metrics["models"][model]["experiments"][exp_name] = metrics
            logger.info(f"    Successfully analyzed {exp_name}")
        except Exception as e:
            logger.error(f"    Error analyzing {exp_name}: {e}")
            all_metrics["models"][model]["experiments"][exp_name] = {"error": str(e)}

    # Create EntropyStatistic and replace data_loader
    entropy_statistic = EntropyStatistic(base_path)
    entropy_statistic.data_loader = temp_data_loader

    # Initialize all_entropy_results structure
    all_entropy_results: Dict[str, Any] = {
        "dataset": dataset,
        "models": {
            model: {
                "experiments": {}
            }
        },
        "architectures": defaultdict(list)
    }

    # Analyze entropy for each experiment
    for exp_name in temp_experiments:
        logger.info(f"  Analyzing entropy for experiment: {exp_name}")
        try:
            entropy_results = entropy_statistic.analyze_experiment_entropy(
                dataset, model, exp_name
            )
            all_entropy_results["models"][model]["experiments"][exp_name] = entropy_results

            # Extract architecture type for grouping
            arch = entropy_results.get("agent_architecture", "unknown")
            all_entropy_results["architectures"][arch].append(f"{model}/{exp_name}")

            # Analyze entropy change trends
            try:
                trend_results = entropy_statistic.analyze_entropy_change_trends(
                    dataset, model, exp_name
                )
                all_entropy_results["models"][model]["experiments"][exp_name][
                    "trend_analysis"
                ] = trend_results
            except Exception as e:
                logger.warning(f"    Error analyzing trends for {exp_name}: {e}")
                all_entropy_results["models"][model]["experiments"][exp_name][
                    "trend_analysis"
                ] = {"error": str(e)}

            logger.info(f"    Successfully analyzed entropy for {exp_name}")
        except Exception as e:
            logger.error(f"    Error analyzing entropy for {exp_name}: {e}")
            all_entropy_results["models"][model]["experiments"][exp_name] = {"error": str(e)}

    # Convert defaultdict to regular dict for JSON serialization
    all_entropy_results["architectures"] = dict(all_entropy_results["architectures"])

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all_metrics.json
    metrics_path = output_dir / "all_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        # Remove response fields to reduce file size
        metrics_copy = _remove_response_fields(all_metrics)
        json.dump(metrics_copy, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save all_entropy_results.json
    entropy_path = output_dir / "all_entropy_results.json"
    with open(entropy_path, "w", encoding="utf-8") as f:
        json.dump(all_entropy_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved entropy results to {entropy_path}")

    # Generate aggregated CSV files using Aggregator
    try:
        aggregator = Aggregator(
            str(entropy_path),
            str(metrics_path),
            str(output_dir)
        )
        aggregator.generate_aggregated_csvs()
        logger.info(f"Generated aggregated CSV files in {output_dir}")
    except Exception as e:
        logger.error(f"Error generating aggregated CSVs: {e}")

    # Generate summary CSV
    try:
        input_csv = output_dir / "all_aggregated_data.csv"
        output_csv = output_dir / "all_summary_data.csv"
        if input_csv.exists():
            extract_summary_fields(input_csv, output_csv)
            logger.info(f"Generated summary CSV: {output_csv}")
    except Exception as e:
        logger.error(f"Error generating summary CSV: {e}")

    return True


def process_temperature_0_6(
    base_path: str,
    dataset: str,
    models: List[str],
    output_dir: Path,
) -> bool:
    """Process temperature 0.6 by filtering existing results.

    Args:
        base_path: Base path to the project directory.
        dataset: Dataset name.
        models: List of model names to filter.
        output_dir: Output directory for results.

    Returns:
        True if processing succeeded, False otherwise.
    """
    logger.info(f"Processing temperature 0.6 for {models}/{dataset}")

    # Path to existing results (results_all directory)
    results_all_path = Path(base_path) / "evaluation" / "results_qwen" / dataset

    # Load existing all_metrics.json
    metrics_file = results_all_path / "all_metrics.json"
    if not metrics_file.exists():
        logger.error(f"Metrics file not found: {metrics_file}")
        return False

    with open(metrics_file, "r", encoding="utf-8") as f:
        all_metrics = json.load(f)

    # Load existing all_entropy_results.json
    entropy_file = results_all_path / "all_entropy_results.json"
    if not entropy_file.exists():
        logger.error(f"Entropy file not found: {entropy_file}")
        return False

    with open(entropy_file, "r", encoding="utf-8") as f:
        all_entropy_results = json.load(f)

    # Filter metrics to only include specified models
    filtered_metrics = {
        "dataset": all_metrics.get("dataset", dataset),
        "task_type": all_metrics.get("task_type", "math"),
        "models": {}
    }
    for model in models:
        if model in all_metrics.get("models", {}):
            filtered_metrics["models"][model] = all_metrics["models"][model]
            logger.info(f"  Included model {model} in filtered metrics")
        else:
            logger.warning(f"  Model {model} not found in existing metrics")

    # Filter entropy results to only include specified models
    filtered_entropy = {
        "dataset": all_entropy_results.get("dataset", dataset),
        "models": {},
        "architectures": {}
    }
    for model in models:
        if model in all_entropy_results.get("models", {}):
            filtered_entropy["models"][model] = all_entropy_results["models"][model]
            logger.info(f"  Included model {model} in filtered entropy results")
        else:
            logger.warning(f"  Model {model} not found in existing entropy results")

    # Filter architectures to only include experiments from specified models
    if "architectures" in all_entropy_results:
        for arch, exp_list in all_entropy_results["architectures"].items():
            filtered_list = []
            for exp_path in exp_list:
                # exp_path format: "model_name/experiment_name"
                model_name = exp_path.split("/")[0] if "/" in exp_path else None
                if model_name in models:
                    filtered_list.append(exp_path)
            if filtered_list:
                filtered_entropy["architectures"][arch] = filtered_list

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered all_metrics.json
    metrics_path = output_dir / "all_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(filtered_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved filtered metrics to {metrics_path}")

    # Save filtered all_entropy_results.json
    entropy_path = output_dir / "all_entropy_results.json"
    with open(entropy_path, "w", encoding="utf-8") as f:
        json.dump(filtered_entropy, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved filtered entropy results to {entropy_path}")

    # Generate aggregated CSV files using Aggregator
    try:
        aggregator = Aggregator(
            str(entropy_path),
            str(metrics_path),
            str(output_dir)
        )
        aggregator.generate_aggregated_csvs()
        logger.info(f"Generated aggregated CSV files in {output_dir}")
    except Exception as e:
        logger.error(f"Error generating aggregated CSVs: {e}")

    # Generate summary CSV
    try:
        input_csv = output_dir / "all_aggregated_data.csv"
        output_csv = output_dir / "all_summary_data.csv"
        if input_csv.exists():
            extract_summary_fields(input_csv, output_csv)
            logger.info(f"Generated summary CSV: {output_csv}")
    except Exception as e:
        logger.error(f"Error generating summary CSV: {e}")

    return True


def _remove_response_fields(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Remove response fields from metrics to reduce file size.

    Args:
        metrics: Dictionary containing analysis metrics.

    Returns:
        Metrics dictionary with response fields removed.
    """
    metrics_copy = metrics.copy()

    if "models" in metrics_copy:
        for model_name, model_data in metrics_copy["models"].items():
            if "experiments" in model_data:
                for exp_name, exp_metrics in model_data["experiments"].items():
                    if "samples" in exp_metrics:
                        for sample_id, sample_data in exp_metrics["samples"].items():
                            if "agents" in sample_data:
                                for agent_key in sample_data["agents"]:
                                    if "response" in sample_data["agents"][agent_key]:
                                        del sample_data["agents"][agent_key]["response"]

    return metrics_copy


def main():
    """Main entry point for the temperature ablation evaluation script.

    Parses command-line arguments and performs experiment analysis
    for different temperature settings.
    """
    # Create argument parser with description
    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent experiments across temperature settings"
    )
    # Add dataset argument
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Dataset to analyze (default: math500)",
    )
    # Add model argument
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=["qwen3_4b"],
        help="Model names to analyze (default: qwen3_4b)",
    )
    # Add temperatures argument
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=[0.4, 0.6, 0.8],
        help="Temperatures to evaluate (default: 0.4 0.6 0.8)",
    )
    # Add task type argument
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["math", "code", "option", "auto"],
        default="auto",
        help="Task type (auto to infer from dataset)",
    )
    # Add timeout argument
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Maximum time in seconds to execute code for code tasks",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Get base path to project directory
    base_path = str(Path(__file__).parent.parent)

    # Define output base directory
    output_base = Path(base_path) / "evaluation" / "results_temp" / args.dataset

    logger.info(f"Starting temperature ablation evaluation")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Models: {args.model}")
    logger.info(f"  Temperatures: {args.temperatures}")

    # Process each temperature
    for temperature in args.temperatures:
        temp_str = f"t_{temperature}".replace(".", "_")
        # Handle the case where temperature might be represented as integer
        if temp_str.endswith("_0"):
            temp_str = temp_str[:-2]
        # Standardize to t_0.X format for output directory
        output_dir = output_base / f"t_{temperature}"

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing temperature: {temperature}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'='*60}")

        if temperature == 0.6:
            # Temperature 0.6 uses existing results from results_all
            success = process_temperature_0_6(
                base_path,
                args.dataset,
                args.model,
                output_dir,
            )
        else:
            # Temperature 0.4 and 0.8 use TempDataLoader
            for model in args.model:
                success = process_temperature_0_4_or_0_8(
                    base_path,
                    args.dataset,
                    model,
                    temperature,
                    args.task_type,
                    args.timeout,
                    output_dir,
                )
                if not success:
                    logger.warning(f"Failed to process temperature {temperature} for model {model}")

    logger.info(f"\n{'='*60}")
    logger.info("Temperature ablation evaluation completed")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    # Execute main function when script is run directly
    main()
