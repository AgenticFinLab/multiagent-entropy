"""
Automated Experiment Runner for Multi-Agent Entropy Data Mining Analysis.

This script runs multiple experiments with different parameter combinations
and collects the results for analysis.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
import concurrent.futures
from threading import Lock
from datetime import datetime
from typing import Dict, List, Optional

from aggregator import ExperimentAggregator
from visualizer import AggregatedResultsVisualizer
from summarizer import VisualizationSummarizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_runner.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Thread lock for safe console output
print_lock = Lock()


def run_single_experiment(
    config: Dict, experiment_id: int, total_experiments: int
) -> Dict:
    """
    Run a single experiment with the given configuration.

    Args:
        config: Dictionary containing experiment configuration
        experiment_id: Current experiment ID for progress tracking
        total_experiments: Total number of experiments to run

    Returns:
        Dictionary containing experiment results
    """
    start_time = time.time()

    # Safely print progress with thread lock
    with print_lock:
        print(
            f"[{experiment_id}/{total_experiments}] Running experiment: {config.get('name', 'unnamed')}"
        )
        print(f"  Parameters: {config['params']}")

    try:
        # Build command to activate conda env, change directory and run the experiment
        # First collect all the parameters
        params_cmd_parts = []
        for param, value in config["params"].items():
            if isinstance(value, list):
                # Handle list arguments like --model-name, --dataset, --architecture
                params_cmd_parts.append(f"--{param.replace('_', '-')}".strip())
                params_cmd_parts.extend([str(v) for v in value])
            elif isinstance(value, bool):
                # Handle boolean flags
                if value:
                    params_cmd_parts.append(f"--{param.replace('_', '-')}".strip())
            else:
                # Handle single-value arguments
                params_cmd_parts.extend([f"--{param.replace('_', '-')}", str(value)])

        # Build the full command as a single bash script
        params_str = " ".join(params_cmd_parts)
        bash_script = f"""#!/bin/bash
set -e  # Exit on any error

cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")" || exit 1

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate maep || exit 1

# Run the experiment
python data_mining/code/main.py {params_str}
"""

        # Write bash script to temporary file
        script_filename = f"/tmp/exp_{config.get('name', 'experiment')}_temp.sh"
        with open(script_filename, "w") as f:
            f.write(bash_script)

        # Make the script executable
        os.chmod(script_filename, 0o755)

        logger.info(f"Running bash script: {script_filename}")
        logger.info(f"Script content: {bash_script.strip()}")

        # Execute the bash script
        result = subprocess.run(
            ["bash", script_filename],
            capture_output=True,
            text=True,
            timeout=config.get("timeout", 3600),  # Default 1 hour timeout
        )

        # Clean up the temporary script file
        try:
            os.remove(script_filename)
        except OSError:
            pass  # Ignore errors when removing temp file

        end_time = time.time()
        duration = end_time - start_time

        # Collect results
        experiment_result = {
            "experiment_id": experiment_id,
            "name": config.get("name", f"experiment_{experiment_id}"),
            "params": config["params"],
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "status": "SUCCESS" if result.returncode == 0 else "FAILED",
        }

        # Safely print completion status
        with print_lock:
            status_msg = "✓ SUCCESS" if result.returncode == 0 else "✗ FAILED"
            print(
                f"[{experiment_id}/{total_experiments}] Completed: {config.get('name', 'unnamed')} - {status_msg} ({duration:.2f}s)"
            )

        return experiment_result

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time

        with print_lock:
            print(
                f"[{experiment_id}/{total_experiments}] TIMEOUT: {config.get('name', 'unnamed')} ({duration:.2f}s)"
            )

        return {
            "experiment_id": experiment_id,
            "name": config.get("name", f"experiment_{experiment_id}"),
            "params": config["params"],
            "return_code": -1,
            "stdout": "",
            "stderr": "Timeout expired",
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "status": "TIMEOUT",
        }

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        with print_lock:
            print(
                f"[{experiment_id}/{total_experiments}] ERROR: {config.get('name', 'unnamed')} - {str(e)} ({duration:.2f}s)"
            )

        return {
            "experiment_id": experiment_id,
            "name": config.get("name", f"experiment_{experiment_id}"),
            "params": config["params"],
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "status": "ERROR",
        }


def run_experiments_serial(configurations: List[Dict]) -> List[Dict]:
    """
    Run experiments serially (one after another).

    Args:
        configurations: List of experiment configurations

    Returns:
        List of experiment results
    """
    results = []
    total_experiments = len(configurations)

    print(f"Starting {total_experiments} experiments in SERIAL mode...")
    print("=" * 80)

    for i, config in enumerate(configurations, 1):
        result = run_single_experiment(config, i, total_experiments)
        results.append(result)

    return results


def run_experiments_parallel(
    configurations: List[Dict], max_workers: Optional[int] = None
) -> List[Dict]:
    """
    Run experiments in parallel using ThreadPoolExecutor.

    Args:
        configurations: List of experiment configurations
        max_workers: Maximum number of parallel workers (default: CPU count)

    Returns:
        List of experiment results
    """
    results = []
    total_experiments = len(configurations)

    print(
        f"Starting {total_experiments} experiments in PARALLEL mode (max {max_workers or 'CPU count'} workers)..."
    )
    print("=" * 80)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_config = {
            executor.submit(run_single_experiment, config, i, total_experiments): (
                i,
                config,
            )
            for i, config in enumerate(configurations, 1)
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_config):
            result = future.result()
            results.append(result)

            # Sort results by experiment_id for consistent ordering
            results.sort(key=lambda x: x["experiment_id"])

    return results


def generate_report(results: List[Dict], output_dir: str = "experiment_reports"):
    """
    Generate a comprehensive report of all experiments.

    Args:
        results: List of experiment results
        output_dir: Directory to save the report files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed JSON results
    json_filename = output_path / f"experiment_results_{timestamp}.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate summary statistics
    total_experiments = len(results)
    successful_experiments = sum(1 for r in results if r["status"] == "SUCCESS")
    failed_experiments = sum(
        1 for r in results if r["status"] in ["FAILED", "ERROR", "TIMEOUT"]
    )
    success_rate = (
        (successful_experiments / total_experiments * 100)
        if total_experiments > 0
        else 0
    )

    # Calculate average duration
    successful_durations = [r["duration"] for r in results if r["status"] == "SUCCESS"]
    avg_duration = (
        sum(successful_durations) / len(successful_durations)
        if successful_durations
        else 0
    )

    # Generate summary report
    summary_lines = [
        "EXPERIMENT SUMMARY REPORT",
        "=" * 50,
        f"Total Experiments: {total_experiments}",
        f"Successful: {successful_experiments}",
        f"Failed/Errored: {failed_experiments}",
        f"Success Rate: {success_rate:.2f}%",
        f"Average Duration (successful): {avg_duration:.2f}s",
        "",
        "DETAILED RESULTS:",
        "-" * 50,
    ]

    for result in results:
        status_icon = "✓" if result["status"] == "SUCCESS" else "✗"
        summary_lines.append(
            f"{status_icon} [{result['experiment_id']}] {result['name']} - {result['status']} ({result['duration']:.2f}s)"
        )

        if result["status"] != "SUCCESS":
            summary_lines.append(
                f"    Error: {result['stderr'][:200]}..."
            )  # Truncate long errors

    summary_filename = output_path / f"experiment_summary_{timestamp}.txt"
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Generate CSV report for easier analysis
    csv_lines = ["experiment_id,name,status,duration,return_code,params,error"]
    for result in results:
        params_str = json.dumps(result["params"]).replace(
            '"', '""'
        )  # Escape quotes for CSV
        error_str = (
            result["stderr"].replace("\n", " ").replace('"', '""')[:100]
        )  # Truncate and escape
        csv_line = f"{result['experiment_id']},{result['name']},{result['status']},{result['duration']:.2f},{result['return_code']},\"{params_str}\",\"{error_str}\""
        csv_lines.append(csv_line)

    csv_filename = output_path / f"experiment_results_{timestamp}.csv"
    with open(csv_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    logger.info(f"Reports saved to {output_path}/")
    logger.info(f"- Detailed JSON: {json_filename.name}")
    logger.info(f"- Summary TXT: {summary_filename.name}")
    logger.info(f"- Analysis CSV: {csv_filename.name}")


def generate_experiment_configs(
    dataset_list: List[str],
    model_list: List[str],
    arch_list: List[str],
    exclude_feature_list: List[str],
) -> List[Dict]:
    """
    Generate experiment configurations based on the provided parameter lists.

    Args:
        dataset_list: List of dataset names
        model_list: List of model names
        arch_list: List of architecture types
        exclude_feature_list: List of exclude feature options

    Returns:
        List of experiment configuration dictionaries
    """
    configurations = []

    # Check if all lists contain only 'all' - if so, run a single experiment
    all_wildcard = (
        len(dataset_list) == 1
        and dataset_list[0] == "all"
        and len(model_list) == 1
        and model_list[0] == "all"
        and len(arch_list) == 1
        and arch_list[0] == "all"
        and len(exclude_feature_list) == 1
        and exclude_feature_list[0] == "all"
    )

    if all_wildcard:
        # Run a single experiment with default parameters
        config = {
            "name": "single_experiment_default",
            "params": {
                "dataset": ["all"],  # Default dataset
                "model_name": ["all"],  # Default model
                "architecture": ["all"],  # Default architecture
                "exclude_features": "default",  # Default exclude features
            },
            "timeout": 3600,
        }
        configurations.append(config)
    else:
        # Generate configurations for all combinations
        from itertools import product

        for dataset, model, arch, exclude_feat in product(
            dataset_list, model_list, arch_list, exclude_feature_list
        ):
            # Create unique experiment name
            exp_name = f"exp_{dataset}_{model}_{arch}_{exclude_feat}_{int(time.time()) % 10000}"

            config = {
                "name": exp_name,
                "params": {
                    "dataset": [dataset],
                    "model_name": [model],
                    "architecture": [arch],
                    "exclude_features": exclude_feat,
                },
                "timeout": 3600,  # Default timeout of 1 hour
            }
            configurations.append(config)

    return configurations


def save_experiment_configs(configurations: List[Dict], output_path: str):
    """
    Save experiment configurations to a JSON file.

    Args:
        configurations: List of experiment configurations
        output_path: Path to save the configuration file
    """
    config_dict = {"experiment_configs": configurations}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Saved {len(configurations)} experiment configurations to {output_path}"
    )


def main():
    """Main entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Automated Experiment Runner for Data Mining Analysis"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to JSON file containing experiment configurations",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run experiments in parallel mode"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_mining/experiment_reports",
        help="Directory to save experiment reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what experiments would be run without executing them",
    )

    # Add new arguments for dataset, model, architecture, and exclude features lists
    parser.add_argument(
        "--dataset-list",
        nargs="+",
        # Changed to None to detect if the argument was provided
        default=["all"],
        help="List of dataset names to use in experiments (use 'all' for all/default)",
    )
    parser.add_argument(
        "--model-list",
        nargs="+",
        # Changed to None to detect if the argument was provided
        default=["all"],
        help="List of model names to use in experiments (use 'all' for all/default)",
    )
    parser.add_argument(
        "--arch-list",
        nargs="+",
        # Changed to None to detect if the argument was provided
        default=["all"],
        help="List of architecture types to use in experiments (use 'all' for all/default)",
    )
    parser.add_argument(
        "--exclude-feature-list",
        nargs="+",
        # Changed to None to detect if the argument was provided
        default=["base_model_all_metrics", "base_model_wo_entropy"],
        help="List of exclude feature options to use in experiments (use 'default' for default and 'all' for all)",
    )
    parser.add_argument(
        "--generate-config-only",
        action="store_true",
        help="Generate configuration file only without running experiments",
    )
    parser.add_argument(
        "--run-aggregation",
        default=True,
        help="Run experiment results aggregation after analysis (default: True)",
    )
    parser.add_argument(
        "--run-visualization",
        default=True,
        help="Run visualization of aggregated results after aggregation (default: True)",
    )
    parser.add_argument(
        "--run-summarization",
        default=True,
        help="Run summarization of generated images after visualization (default: True)",
    )
    parser.add_argument(
        "--n-top-analysis",
        type=int,
        default=5,
        help="Number of top features to summarize (default: 5)",
    )

    args = parser.parse_args()

    if args.config_file == None or args.config_file == "":
        logger.info(f"Generating experiment configurations from parameter lists...")
        logger.info(f"Dataset list: {args.dataset_list}")
        logger.info(f"Model list: {args.model_list}")
        logger.info(f"Architecture list: {args.arch_list}")
        logger.info(f"Exclude feature list: {args.exclude_feature_list}")

        # Generate configurations based on the parameter lists
        configurations = generate_experiment_configs(
            args.dataset_list,
            args.model_list,
            args.arch_list,
            args.exclude_feature_list,
        )

        # Define config file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_output_path = (
            f"data_mining/configs/generated_experiment_config_{timestamp}.json"
        )

        # Save the generated configurations
        save_experiment_configs(configurations, config_output_path)

        # If only generating config, exit here
        if args.generate_config_only:
            print(f"Configuration file generated: {config_output_path}")
            print(f"Number of experiments configured: {len(configurations)}")
            return

        # Otherwise, use the generated config file for running experiments
        args.config_file = config_output_path
    else:
        logger.info(f"Using specified configuration file: {args.config_file}")

    # Get experiment configurations from file (either original or generated)
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            logger.error(f"Configuration file {args.config_file} not found")
            sys.exit(1)

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # Handle both direct array and object with 'experiment_configs' key
        if isinstance(config_data, list):
            configurations = config_data
        elif isinstance(config_data, dict) and "experiment_configs" in config_data:
            configurations = config_data["experiment_configs"]
        else:
            logger.error(
                f"Configuration file must contain either an array of configs or an object with 'experiment_configs' key"
            )
            sys.exit(1)

        logger.info(
            f"Loaded {len(configurations)} experiment configurations from {args.config_file}"
        )
    else:
        raise ValueError("No configuration file provided")

    if args.dry_run:
        print("DRY RUN MODE - Would run the following experiments:")
        print("=" * 80)
        for i, config in enumerate(configurations, 1):
            print(f"[{i}] {config['name']}")
            print(f"    Parameters: {config['params']}")
            print(f"    Timeout: {config.get('timeout', 3600)}s")
            print()
        print(f"Total: {len(configurations)} experiments")
        return

    logger.info(f"Starting {len(configurations)} experiments")

    # Run experiments
    start_time = time.time()

    if args.parallel:
        results = run_experiments_parallel(configurations, args.max_workers)
    else:
        results = run_experiments_serial(configurations)

    end_time = time.time()
    total_duration = end_time - start_time

    # Generate reports
    generate_report(results, args.output_dir)

    # Run aggregation if requested
    if args.run_aggregation:
        logger.info("\nStarting experiment results aggregation...")
        try:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            results_dir = project_root / "results"
            output_dir = project_root / "results_aggregated"
            output_dir.mkdir(parents=True, exist_ok=True)

            aggregator = ExperimentAggregator(str(results_dir), str(output_dir))
            aggregator.aggregate_all_experiments()
            logger.info("Experiment results aggregation completed!")
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}", exc_info=True)
            raise

    # Run visualization if requested
    if args.run_visualization:
        logger.info("\nStarting visualization of aggregated results...")
        try:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            input_dir = project_root / "results_aggregated"
            output_dir = project_root / "results_visualizations"
            shap_data_dir = project_root / "results"

            visualizer = AggregatedResultsVisualizer(
                str(input_dir),
                str(output_dir),
                n_features=20,
                feature_importance_from="mean_importance_normalized",
                shap_data_dir=str(shap_data_dir),
            )
            visualizer.visualize_all_experiments()
            logger.info("Visualization of aggregated results completed!")
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}", exc_info=True)
            raise

    if args.run_summarization:
        logger.info("\nStarting summarization of generated images...")
        try:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            input_dir = project_root / "results_aggregated"
            output_dir = project_root / "results_summaries"

            summarizer = VisualizationSummarizer(
                str(input_dir),
                str(output_dir),
                sort_column="mean_importance_normalized",
            )
            summarizer.analyze_visualizations(n=args.n_top_analysis)
            # Perform hierarchical statistical analysis on the summary data
            summarizer.perform_hierarchical_statistical_analysis()
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}", exc_info=True)
            raise

    # Print final summary
    successful_count = sum(1 for r in results if r["status"] == "SUCCESS")
    total_count = len(results)
    success_rate = (successful_count / total_count * 100) if total_count > 0 else 0

    print("")
    print("=" * 80)
    print("EXPERIMENT BATCH COMPLETED")
    print("=" * 80)
    print(f"Total Experiments: {total_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {total_count - successful_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total Duration: {total_duration:.2f}s")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
