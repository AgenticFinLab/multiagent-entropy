import argparse
import json
from pathlib import Path

from analyzer import ExperimentAnalyzer
from entropy_statistics import EntropyStatisticsAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-agent experiment results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "humaneval", "mmlu"],
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
        "--save-csv",
        default=True,
        help="Save last_agent_stats summary to CSV file",
    )
    parser.add_argument(
        "--analyze-entropy",
        action="store_true",
        help="Perform entropy statistical analysis",
    )
    parser.add_argument(
        "--entropy-compare",
        action="store_true",
        help="Compare entropy statistics across architectures",
    )
    parser.add_argument(
        "--save-entropy-json",
        action="store_true",
        help="Save detailed entropy results to JSON file",
    )

    args = parser.parse_args()

    base_path = str(Path.cwd())
    analyzer = ExperimentAnalyzer(base_path)
    entropy_analyzer = None

    if args.analyze_entropy or args.entropy_compare:
        entropy_analyzer = EntropyStatisticsAnalyzer(base_path)

    if args.experiment:
        print(f"Analyzing experiment: {args.experiment}")
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
                json_output_path = entropy_output_dir / f"{args.experiment}_entropy.json"
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(entropy_results, f, indent=2, ensure_ascii=False)
                print(f"Entropy JSON saved to: {json_output_path}")

            print(f"Architecture: {entropy_results['agent_architecture']}")
            print(f"Total entropy: {entropy_results['macro_statistics']['experiment_level']['total_entropy']:.4f}")
            print(f"Average entropy: {entropy_results['macro_statistics']['experiment_level']['average_entropy']:.4f}")
    else:
        if args.compare:
            print(f"Comparing all experiments for dataset: {args.dataset}")
            comparison = analyzer.compare_experiments(args.dataset, args.task_type)

            if args.output:
                output_path = args.output
            else:
                output_dir = (
                    Path(base_path) / "evaluation" / "results" / args.dataset
                )
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
                output_dir = (
                    Path(base_path) / "evaluation" / "results" / args.dataset
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "all_metrics.json"

            analyzer.save_results(all_metrics, output_path)
            print(f"All metrics saved to: {output_path}")

        if args.save_csv:
            csv_output_path = (
                Path(base_path)
                / "evaluation"
                / "results"
                / args.dataset
                / "all_metrics_summary.csv"
            )
            analyzer.save_last_agent_stats_to_csv(
                args.dataset, str(csv_output_path), args.task_type
            )
            print(f"CSV summary saved to: {csv_output_path}")

        if entropy_analyzer:
            print(f"\nAnalyzing entropy for all experiments in dataset: {args.dataset}")
            entropy_results = entropy_analyzer.analyze_all_experiments_entropy(
                args.dataset
            )

            entropy_output_dir = (
                Path(base_path) / "evaluation" / "results" / args.dataset / "entropy"
            )
            entropy_output_dir.mkdir(parents=True, exist_ok=True)

            entropy_analyzer.save_all_entropy_statistics_to_csv(
                args.dataset, str(entropy_output_dir)
            )
            print(f"Entropy statistics saved to: {entropy_output_dir}")

            if args.entropy_compare:
                print("\nComparing entropy statistics across architectures...")
                comparison = entropy_analyzer.compare_architectures_entropy(args.dataset)

                print(f"\nArchitecture Comparison:")
                for arch, trends in comparison["trends"].items():
                    print(f"\n{arch.upper()} Architecture:")
                    print(f"  Mean entropy: {trends['mean']:.4f}")
                    print(f"  Std: {trends['std']:.4f}")
                    print(f"  Min: {trends['min']:.4f}")
                    print(f"  Max: {trends['max']:.4f}")
                    print(f"  Experiments: {trends['count']}")

            if args.save_entropy_json:
                json_output_path = entropy_output_dir / "all_entropy_results.json"
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(entropy_results, f, indent=2, ensure_ascii=False)
                print(f"Entropy JSON saved to: {json_output_path}")


if __name__ == "__main__":
    main()
