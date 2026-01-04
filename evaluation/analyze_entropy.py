import argparse
from pathlib import Path

from entropy_statistics import EntropyStatisticsAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze entropy statistics for multi-agent experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "humaneval", "mmlu"],
        default="gsm8k",
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to analyze (if not provided, analyze all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (if not provided, save to evaluation/results/)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare entropy statistics across architectures",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save detailed results to JSON file",
    )

    args = parser.parse_args()

    base_path = str(Path.cwd())
    analyzer = EntropyStatisticsAnalyzer(base_path)

    if args.experiment:
        print(f"Analyzing entropy for experiment: {args.experiment}")
        results = analyzer.analyze_experiment_entropy(
            args.dataset, args.experiment
        )

        if args.save_json:
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = (
                    Path(base_path)
                    / "evaluation"
                    / "results"
                    / args.dataset
                    / "entropy"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{args.experiment}_entropy_stats.json"
            analyzer.save_results_json(results, str(output_path))

        print(f"Experiment entropy analysis completed")
        print(f"Architecture: {results['agent_architecture']}")
        print(f"Total entropy: {results['macro_statistics']['experiment_level']['total_entropy']:.4f}")
        print(f"Average entropy: {results['macro_statistics']['experiment_level']['average_entropy']:.4f}")
    else:
        if args.compare:
            print(f"Comparing entropy statistics for dataset: {args.dataset}")
            comparison = analyzer.compare_architectures_entropy(args.dataset)

            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = (
                    Path(base_path)
                    / "evaluation"
                    / "results"
                    / args.dataset
                    / "entropy"
                )
            output_dir.mkdir(parents=True, exist_ok=True)

            if args.save_json:
                output_path = output_dir / "architecture_comparison.json"
                analyzer.save_results_json(comparison, str(output_path))

            print(f"\nArchitecture Comparison:")
            for arch, trends in comparison["trends"].items():
                print(f"\n{arch.upper()} Architecture:")
                print(f"  Mean entropy: {trends['mean']:.4f}")
                print(f"  Std: {trends['std']:.4f}")
                print(f"  Min: {trends['min']:.4f}")
                print(f"  Max: {trends['max']:.4f}")
                print(f"  Experiments: {trends['count']}")
        else:
            print(f"Analyzing entropy statistics for all experiments: {args.dataset}")
            all_results = analyzer.analyze_all_experiments_entropy(args.dataset)

            if args.save_json:
                if args.output_dir:
                    output_dir = Path(args.output_dir)
                else:
                    output_dir = (
                        Path(base_path)
                        / "evaluation"
                        / "results"
                        / args.dataset
                        / "entropy"
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "all_entropy_statistics.json"
                analyzer.save_results_json(all_results, str(output_path))

            print(f"\nSummary:")
            print(f"  Total experiments analyzed: {len(all_results['experiments'])}")
            for arch, exp_names in all_results["architectures"].items():
                print(f"  {arch.upper()}: {len(exp_names)} experiments")

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = (
                Path(base_path) / "evaluation" / "results" / args.dataset / "entropy"
            )

        print(f"\nSaving CSV files to: {output_dir}")
        analyzer.save_all_entropy_statistics_to_csv(args.dataset, str(output_dir))


if __name__ == "__main__":
    main()
