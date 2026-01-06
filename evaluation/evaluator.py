"""Main evaluation script for multi-agent system experiments.

This module provides a command-line interface for analyzing experiment results,
comparing architectures, and generating evaluation reports.
"""

import json
import argparse
from pathlib import Path

from entropy_analyzer import EntropyAnalyzer
from experiment_analyzer import ExperimentAnalyzer
from results_aggregator import ResultsAggregator


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
        "--compare",
        action="store_true",
        help="Compare experiments by architecture",
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

    args = parser.parse_args()

    base_path = str(Path.cwd())
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

            print(f"Architecture: {entropy_results['agent_architecture']}")
            print(
                f"Total entropy: {entropy_results['macro_statistics']['experiment_level']['total_entropy']:.4f}"
            )
            print(
                f"Average entropy: {entropy_results['macro_statistics']['experiment_level']['average_entropy']:.4f}"
            )

            if args.analyze_trends:
                print(f"\nAnalyzing entropy change trends for experiment: {args.experiment}")
                trend_results = entropy_analyzer.analyze_entropy_change_trends(
                    args.dataset, args.experiment
                )

                print(f"\nTrend Analysis Summary:")
                print(f"  Architecture: {trend_results['agent_architecture']}")
                print(f"  Number of rounds: {trend_results['num_rounds']}")
                
                if "intra_round_stats" in trend_results["trend_statistics"]:
                    intra_stats = trend_results["trend_statistics"]["intra_round_stats"]
                    print(f"\n  Intra-round Statistics:")
                    print(f"    Mean agent difference: {intra_stats.get('mean_agent_difference', 0):.4f}")
                    print(f"    Max agent difference: {intra_stats.get('max_agent_difference', 0):.4f}")
                
                if "inter_round_stats" in trend_results["trend_statistics"]:
                    inter_stats = trend_results["trend_statistics"]["inter_round_stats"]
                    print(f"\n  Inter-round Statistics:")
                    print(f"    Mean round-to-round change: {inter_stats.get('mean_round_to_round_change', 0):.4f}")
                    print(f"    Max round-to-round change: {inter_stats.get('max_round_to_round_change', 0):.4f}")
                
                if "overall_summary" in trend_results["trend_statistics"]:
                    summary = trend_results["trend_statistics"]["overall_summary"]
                    print(f"\n  Overall Summary:")
                    print(f"    Total agents analyzed: {summary.get('total_agents_analyzed', 0)}")
                    print(f"    Agents with increasing trend: {summary.get('agents_with_increasing_trend', 0)}")
                    print(f"    Agents with decreasing trend: {summary.get('agents_with_decreasing_trend', 0)}")
                    print(f"    Dominant trend: {summary.get('dominant_trend', 'unknown')}")
    else:
        if args.compare:
            print(f"Comparing all experiments for dataset: {args.dataset}")
            comparison = analyzer.compare_experiments(args.dataset, args.task_type)

            if args.output:
                output_path = args.output
            else:
                output_dir = Path(base_path) / "evaluation" / "results" / args.dataset
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "comparison.json"

            analyzer.save_results(comparison, output_path)
            print(f"Comparison results saved to: {output_path}")

            for arch, exps in comparison["architectures"].items():
                print(f"\n{arch.upper()} Architecture:")
                for exp in exps:
                    print(f"  {exp['experiment_name']}")
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
                print(f"\nAnalyzing entropy change trends for all experiments in dataset: {args.dataset}")
                
                for exp_name in entropy_results["experiments"].keys():
                    if "error" not in entropy_results["experiments"][exp_name]:
                        try:
                            trend_results = entropy_analyzer.analyze_entropy_change_trends(
                                args.dataset, exp_name
                            )
                            entropy_results["experiments"][exp_name]["trend_analysis"] = trend_results
                        except Exception as e:
                            print(f"Error analyzing trends for {exp_name}: {e}")
                            entropy_results["experiments"][exp_name]["trend_analysis"] = {
                                "error": str(e)
                            }
                
                if args.save_entropy_json:
                    json_output_path = entropy_output_dir / "all_entropy_results.json"
                    with open(json_output_path, "w", encoding="utf-8") as f:
                        json.dump(entropy_results, f, indent=2, ensure_ascii=False)
                    print(f"Updated entropy JSON with trend analysis saved to: {json_output_path}")
                    
                    print(f"\nTrend Analysis Summary Across Experiments:")
                    for exp_name, exp_data in entropy_results["experiments"].items():
                        if "trend_analysis" in exp_data and "error" not in exp_data["trend_analysis"]:
                            trend_data = exp_data["trend_analysis"]
                            arch = trend_data["agent_architecture"]
                            num_rounds = trend_data["num_rounds"]
                            print(f"\n  {exp_name} ({arch}, {num_rounds} rounds):")
                            
                            if "overall_summary" in trend_data["trend_statistics"]:
                                summary = trend_data["trend_statistics"]["overall_summary"]
                                print(f"    Dominant trend: {summary.get('dominant_trend', 'unknown')}")
                                print(f"    Agents analyzed: {summary.get('total_agents_analyzed', 0)}")

    if args.aggregate:
        print(f"\nAggregating results for dataset: {args.dataset}")
        aggregator = ResultsAggregator(base_path, args.dataset)
        aggregator.run_aggregation()


if __name__ == "__main__":
    main()
