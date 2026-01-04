import argparse
from pathlib import Path

from analyzer import ExperimentAnalyzer


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

    args = parser.parse_args()

    base_path = str(Path.cwd())
    analyzer = ExperimentAnalyzer(base_path)

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


if __name__ == "__main__":
    main()
