#!/usr/bin/env python3
"""
Main evaluation script for MultiAgent-Entropy experiments.

This script provides a command-line interface to evaluate experiment results,
including accuracy evaluation for different task types (math, code, option)
and comprehensive entropy analysis.
"""

import os
import sys
import json
import yaml
import argparse
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

from evaluation.math_evaluator import MathEvaluator
from evaluation.code_evaluator import CodeEvaluator
from evaluation.option_evaluator import OptionEvaluator
from evaluation.entropy_analyzer import EntropyAnalyzer
from evaluation.report_generator import ReportGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation/logs/evaluation.log"),
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
        description="Evaluate MultiAgent-Entropy experiment results"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Path to experiment directory or parent directory containing multiple experiments",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="evaluation/results",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "-t",
        "--task-type",
        type=str,
        choices=["math", "code", "option", "auto"],
        default="auto",
        help="Task type for evaluation (math/code/option/auto)",
    )

    parser.add_argument(
        "--experiment-names",
        nargs="+",
        help="Specific experiment names to evaluate (default: all experiments)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "html", "both"],
        default="both",
        help="Report format (txt/html/both)",
    )

    parser.add_argument(
        "--no-entropy",
        action="store_true",
        help="Skip entropy analysis",
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization generation",
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison across multiple experiments",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def get_task_type_from_experiment(experiment_dir: str) -> Optional[str]:
    """
    Infer task type from experiment directory or configuration.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        Task type (math/code/option) or None if cannot determine
    """
    dir_name = os.path.basename(experiment_dir).lower()

    if "gsm8k" in dir_name or "math" in dir_name:
        return "math"
    elif "humaneval" in dir_name or "code" in dir_name:
        return "code"
    elif "mmlu" in dir_name or "option" in dir_name:
        return "option"

    config_path = os.path.join(experiment_dir, "config.yml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if "data" in config:
                    data_name = config["data"].get("data_name", "").lower()
                    if "gsm8k" in data_name or "math" in data_name:
                        return "math"
                    elif "humaneval" in data_name or "code" in data_name:
                        return "code"
                    elif "mmlu" in data_name or "option" in data_name:
                        return "option"
        except Exception as e:
            logger.warning(f"Error reading config: {e}")

    return None


def get_evaluator(task_type: str):
    """
    Get the appropriate evaluator for the task type.

    Args:
        task_type: Type of task (math, code, option)

    Returns:
        Evaluator instance
    """
    if task_type == "math":
        return MathEvaluator()
    elif task_type == "code":
        return CodeEvaluator()
    elif task_type == "option":
        return OptionEvaluator()
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def evaluate_single_experiment(
    experiment_dir: str,
    task_type: str,
    output_dir: str,
    analyze_entropy: bool = True,
    generate_visualizations: bool = True,
    report_format: str = "both",
) -> Dict[str, Any]:
    """
    Evaluate a single experiment.

    Args:
        experiment_dir: Path to experiment directory
        task_type: Type of task (math/code/option)
        output_dir: Directory to save results
        analyze_entropy: Whether to perform entropy analysis
        generate_visualizations: Whether to generate visualizations
        report_format: Report format (txt/html/both)

    Returns:
        Dictionary containing evaluation results
    """
    experiment_name = os.path.basename(experiment_dir)
    logger.info(f"Evaluating experiment: {experiment_name}")

    results = {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "task_type": task_type,
    }

    evaluator = get_evaluator(task_type)

    try:
        accuracy_results = evaluator.evaluate_experiment(experiment_dir)
        results.update(accuracy_results)

        if "error" in accuracy_results:
            logger.warning(f"Accuracy evaluation failed: {accuracy_results['error']}")
            return results

        logger.info(
            f"Accuracy: {accuracy_results.get('accuracy', 0):.4f} "
            f"({accuracy_results.get('correct_samples', 0)}/{accuracy_results.get('total_samples', 0)})"
        )

    except Exception as e:
        logger.error(f"Error during accuracy evaluation: {e}")
        results["accuracy_error"] = str(e)
        return results

    entropy_results = {}
    if analyze_entropy:
        try:
            entropy_analyzer = EntropyAnalyzer(experiment_dir)
            entropy_data = entropy_analyzer.load_entropy_data()

            if entropy_data:
                statistics = entropy_analyzer.calculate_statistics()
                step_statistics = entropy_analyzer.calculate_step_statistics()
                sample_statistics = entropy_analyzer.calculate_sample_statistics()
                comparison = entropy_analyzer.compare_agents()

                entropy_results["statistics"] = statistics
                entropy_results["step_statistics"] = step_statistics
                entropy_results["sample_statistics"] = sample_statistics
                entropy_results["comparison"] = comparison

                correlation = entropy_analyzer.analyze_entropy_accuracy_correlation(
                    accuracy_results
                )
                entropy_results["correlation"] = correlation

                results["mean_entropy"] = np.mean(
                    [s["mean"] for s in statistics.values()]
                )
                results["std_entropy"] = np.mean(
                    [s["std"] for s in statistics.values()]
                )

                logger.info(f"Entropy analysis completed for {len(statistics)} agents")
                logger.info(f"Step statistics calculated for {len(step_statistics)} agents")
                logger.info(f"Sample statistics calculated for {len(sample_statistics)} samples")

        except Exception as e:
            logger.error(f"Error during entropy analysis: {e}")
            results["entropy_error"] = str(e)

    report_generator = ReportGenerator(output_dir)

    try:
        if generate_visualizations and analyze_entropy and "statistics" in entropy_results:
            entropy_analyzer = EntropyAnalyzer(experiment_dir)
            entropy_analyzer.load_entropy_data()

            generated_files = report_generator.generate_full_report(
                accuracy_results, entropy_results, experiment_name, entropy_analyzer
            )
        else:
            generated_files = report_generator.generate_full_report(
                accuracy_results, entropy_results, experiment_name, None
            )

        results["report_files"] = generated_files

        logger.info(f"Report generated: {generated_files.get('html_report', 'N/A')}")

    except Exception as e:
        logger.error(f"Error during report generation: {e}")
        results["report_error"] = str(e)

    return results


def evaluate_multiple_experiments(
    input_dir: str,
    task_type: str,
    output_dir: str,
    experiment_names: Optional[List[str]] = None,
    analyze_entropy: bool = True,
    generate_visualizations: bool = True,
    report_format: str = "both",
    generate_comparison: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple experiments.

    Args:
        input_dir: Path to parent directory containing experiments
        task_type: Type of task (math/code/option/auto)
        output_dir: Directory to save results
        experiment_names: Specific experiment names to evaluate
        analyze_entropy: Whether to perform entropy analysis
        generate_visualizations: Whether to generate visualizations
        report_format: Report format (txt/html/both)
        generate_comparison: Whether to generate comparison across experiments

    Returns:
        List of evaluation results for each experiment
    """
    all_results = []

    if experiment_names:
        experiment_dirs = [
            os.path.join(input_dir, name) for name in experiment_names
        ]
        experiment_dirs = [d for d in experiment_dirs if os.path.isdir(d)]
    else:
        experiment_dirs = [
            d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
        ]
        experiment_dirs = [os.path.join(input_dir, d) for d in experiment_dirs]

    if not experiment_dirs:
        logger.warning(f"No experiment directories found in {input_dir}")
        return all_results

    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    for experiment_dir in experiment_dirs:
        current_task_type = task_type

        if task_type == "auto":
            inferred_type = get_task_type_from_experiment(experiment_dir)
            if inferred_type:
                current_task_type = inferred_type
                logger.info(f"Inferred task type: {current_task_type}")
            else:
                logger.warning(
                    f"Could not infer task type for {experiment_dir}, skipping"
                )
                continue

        try:
            result = evaluate_single_experiment(
                experiment_dir=experiment_dir,
                task_type=current_task_type,
                output_dir=output_dir,
                analyze_entropy=analyze_entropy,
                generate_visualizations=generate_visualizations,
                report_format=report_format,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {experiment_dir}: {e}")

    if generate_comparison and len(all_results) > 1:
        try:
            report_generator = ReportGenerator(output_dir)

            comparison_plot_path = os.path.join(output_dir, "experiments_comparison.png")
            report_generator.generate_comparison_plot(all_results, comparison_plot_path)

            summary_table_path = os.path.join(output_dir, "experiments_summary.csv")
            report_generator.generate_summary_table(all_results, summary_table_path)

            logger.info(f"Comparison plot saved to: {comparison_plot_path}")
            logger.info(f"Summary table saved to: {summary_table_path}")

        except Exception as e:
            logger.error(f"Error generating comparison: {e}")

    return all_results


def main():
    """Main function to run evaluation."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs("evaluation/logs", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Starting evaluation with input directory: {args.input_dir}")

    results = evaluate_multiple_experiments(
        input_dir=args.input_dir,
        task_type=args.task_type,
        output_dir=args.output_dir,
        experiment_names=args.experiment_names,
        analyze_entropy=not args.no_entropy,
        generate_visualizations=not args.no_visualization,
        report_format=args.format,
        generate_comparison=args.compare,
    )

    if not results:
        logger.warning("No experiments were evaluated")
        return

    logger.info(f"Evaluation completed for {len(results)} experiments")

    results_summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(results_summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation summary saved to: {results_summary_path}")


if __name__ == "__main__":
    main()
