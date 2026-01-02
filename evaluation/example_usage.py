#!/usr/bin/env python3
"""
Example script demonstrating how to use the evaluation package.

This script shows how to:
1. Evaluate a single experiment
2. Perform entropy analysis
3. Generate reports
4. Compare multiple experiments
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.math_evaluator import MathEvaluator
from evaluation.code_evaluator import CodeEvaluator
from evaluation.option_evaluator import OptionEvaluator
from evaluation.entropy_analyzer import EntropyAnalyzer
from evaluation.report_generator import ReportGenerator


def example_single_experiment():
    """Example: Evaluate a single math experiment."""
    print("=" * 80)
    print("Example 1: Evaluating a Single Math Experiment")
    print("=" * 80)

    experiment_dir = "experiments/results/raw/gsm8k/qwen3-0.6b_gsm8k_single_agent_20260102_183452"
    output_dir = "evaluation/example_results"

    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        print("Please run an experiment first or update the path.")
        return

    evaluator = MathEvaluator()
    results = evaluator.evaluate_experiment(experiment_dir)

    print(f"\nExperiment: {results.get('experiment_dir', 'Unknown')}")
    print(f"Task Type: {results.get('task_type', 'Unknown')}")
    print(f"Accuracy: {results.get('accuracy', 0):.4f}")
    print(f"Correct: {results.get('correct_samples', 0)}/{results.get('total_samples', 0)}")

    entropy_analyzer = EntropyAnalyzer(experiment_dir)
    entropy_data = entropy_analyzer.load_entropy_data()

    if entropy_data:
        statistics = entropy_analyzer.calculate_statistics()
        print(f"\nEntropy Statistics:")
        for agent_name, stats in statistics.items():
            print(f"  {agent_name}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std: {stats['std']:.4f}")

    report_generator = ReportGenerator(output_dir)
    entropy_results = {"statistics": statistics} if entropy_data else {}

    generated_files = report_generator.generate_full_report(
        results, entropy_results, "example_math_experiment", entropy_analyzer
    )

    print(f"\nGenerated Reports:")
    for file_type, file_path in generated_files.items():
        if file_type != "visualizations":
            print(f"  {file_type}: {file_path}")


def example_multiple_experiments():
    """Example: Compare multiple experiments."""
    print("\n" + "=" * 80)
    print("Example 2: Comparing Multiple Experiments")
    print("=" * 80)

    input_dir = "experiments/results/raw"
    output_dir = "evaluation/example_results"

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    experiments = []
    for dataset in ["gsm8k", "humaneval", "mmlu"]:
        dataset_dir = os.path.join(input_dir, dataset)
        if os.path.exists(dataset_dir):
            for exp_name in os.listdir(dataset_dir):
                exp_dir = os.path.join(dataset_dir, exp_name)
                if os.path.isdir(exp_dir):
                    task_type = "math" if "gsm8k" in dataset else ("code" if "humaneval" in dataset else "option")

                    try:
                        if task_type == "math":
                            evaluator = MathEvaluator()
                        elif task_type == "code":
                            evaluator = CodeEvaluator()
                        else:
                            evaluator = OptionEvaluator()

                        results = evaluator.evaluate_experiment(exp_dir)
                        results["experiment_name"] = exp_name

                        entropy_analyzer = EntropyAnalyzer(exp_dir)
                        entropy_data = entropy_analyzer.load_entropy_data()

                        if entropy_data:
                            statistics = entropy_analyzer.calculate_statistics()
                            if statistics:
                                results["mean_entropy"] = np.mean(
                                    [s["mean"] for s in statistics.values()]
                                )
                                results["std_entropy"] = np.mean(
                                    [s["std"] for s in statistics.values()]
                                )

                        experiments.append(results)

                    except Exception as e:
                        print(f"Error evaluating {exp_name}: {e}")

    if experiments:
        report_generator = ReportGenerator(output_dir)

        comparison_plot_path = os.path.join(output_dir, "example_comparison.png")
        report_generator.generate_comparison_plot(experiments, comparison_plot_path)

        summary_table_path = os.path.join(output_dir, "example_summary.csv")
        report_generator.generate_summary_table(experiments, summary_table_path)

        print(f"\nComparison plot saved to: {comparison_plot_path}")
        print(f"Summary table saved to: {summary_table_path}")


def example_custom_analysis():
    """Example: Custom entropy analysis."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Entropy Analysis")
    print("=" * 80)

    experiment_dir = "experiments/results/raw/gsm8k/qwen3-0.6b_gsm8k_single_agent_20260102_183452"

    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        return

    analyzer = EntropyAnalyzer(experiment_dir)
    entropy_data = analyzer.load_entropy_data()

    if not entropy_data:
        print("No entropy data found")
        return

    statistics = analyzer.calculate_statistics()
    print("\nEntropy Statistics:")
    for agent_name, stats in statistics.items():
        print(f"\n{agent_name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Q25: {stats['q25']:.4f}")
        print(f"  Q75: {stats['q75']:.4f}")

    comparison = analyzer.compare_agents()
    if comparison.get("pairwise_comparisons"):
        print("\nAgent Comparisons:")
        for comp_name, comp_stats in comparison["pairwise_comparisons"].items():
            print(f"\n{comp_name}:")
            print(f"  T-test: t={comp_stats['t_statistic']:.4f}, p={comp_stats['t_p_value']:.4f}")
            print(f"  KS-test: statistic={comp_stats['ks_statistic']:.4f}, p={comp_stats['ks_p_value']:.4f}")


if __name__ == "__main__":
    import numpy as np

    print("\nMultiAgent-Entropy Evaluation Package Examples")
    print("=" * 80)

    example_single_experiment()
    example_multiple_experiments()
    example_custom_analysis()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
