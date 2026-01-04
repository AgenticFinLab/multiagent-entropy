"""Aggregator for multi-agent experiment results.

This module provides functionality to aggregate statistical results from
the evaluation/results directory, separating correct and incorrect samples
into separate CSV files for machine learning analysis.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any


class ResultsAggregator:
    """Aggregator for multi-agent experiment results.

    Traverses the evaluation/results directory, parses statistical data,
    and aggregates sample-level information into CSV files separated by
    correctness for machine learning analysis.
    """

    def __init__(self, base_path: str, dataset: str):
        """Initialize the aggregator with base path.

        Args:
            base_path: Base path to the project directory.
            dataset: Dataset name to filter results.
        """
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "evaluation" / "results" / f"{dataset}"

    def find_all_metrics_files(self) -> List[Path]:
        """Find all metrics JSON files in the results directory.

        Returns:
            List of paths to metrics JSON files.
        """
        metrics_files = []

        all_metrics_file = self.results_path / "all_metrics.json"
        if all_metrics_file.exists():
            metrics_files.append(all_metrics_file)

        for exp_file in self.results_path.glob("*_metrics.json"):
            if exp_file.name != "all_metrics.json":
                metrics_files.append(exp_file)

        return sorted(metrics_files)

    def parse_metrics_file(self, metrics_file: Path) -> Dict[str, Any]:
        """Parse a metrics JSON file.

        Args:
            metrics_file: Path to the metrics JSON file.

        Returns:
            Dictionary containing parsed metrics data.
        """
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def extract_sample_data(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sample-level data from metrics.

        Args:
            metrics_data: Dictionary containing metrics data.

        Returns:
            List of dictionaries containing sample-level information.
        """
        samples_data = []

        dataset = metrics_data.get("dataset", "")
        task_type = metrics_data.get("task_type", "")

        experiments = metrics_data.get("experiments", {})

        for exp_name, exp_data in experiments.items():
            experiment_name = exp_data.get("experiment_name", "")
            agent_architecture = exp_data.get("agent_architecture", "")
            num_rounds = exp_data.get("num_rounds", 0)

            samples = exp_data.get("samples", {})

            for sample_id, sample_data in samples.items():
                main_id = sample_data.get("main_id", "")
                ground_truth = sample_data.get("ground_truth", "")

                agents = sample_data.get("agents", {})

                for agent_key, agent_data in agents.items():
                    agent_type = agent_data.get("agent_type", "")
                    execution_order = agent_data.get("execution_order", 0)
                    time_cost = agent_data.get("time_cost", 0.0)
                    average_entropy = agent_data.get("average_entropy", 0.0)
                    predicted_answer = agent_data.get("predicted_answer", "")
                    is_correct = agent_data.get("is_correct", False)

                    round_num = 0
                    if "_round_" in agent_key:
                        round_num = int(agent_key.split("_round_")[1])

                    sample_record = {
                        "dataset": dataset,
                        "task_type": task_type,
                        "experiment_name": experiment_name,
                        "agent_architecture": agent_architecture,
                        "num_rounds": num_rounds,
                        "main_id": main_id,
                        "ground_truth": ground_truth,
                        "agent_type": agent_type,
                        "execution_order": execution_order,
                        "round": round_num,
                        "time_cost": time_cost,
                        "average_entropy": average_entropy,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                    }

                    samples_data.append(sample_record)

        return samples_data

    def aggregate_all_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate all results from metrics files.

        Returns:
            Dictionary containing correct and incorrect samples.
        """
        all_samples = {"correct": [], "incorrect": []}

        metrics_files = self.find_all_metrics_files()

        for metrics_file in metrics_files:
            print(f"Processing: {metrics_file}")

            try:
                metrics_data = self.parse_metrics_file(metrics_file)
                samples_data = self.extract_sample_data(metrics_data)

                for sample in samples_data:
                    if sample["is_correct"]:
                        all_samples["correct"].append(sample)
                    else:
                        all_samples["incorrect"].append(sample)

            except Exception as e:
                print(f"Error processing {metrics_file}: {e}")

        return all_samples

    def save_to_csv(self, samples: List[Dict[str, Any]], output_path: Path) -> None:
        """Save samples to a CSV file.

        Args:
            samples: List of sample dictionaries.
            output_path: Path to the output CSV file.
        """
        if not samples:
            print(f"No samples to save to {output_path}")
            return

        fieldnames = [
            "dataset",
            "task_type",
            "experiment_name",
            "agent_architecture",
            "num_rounds",
            "main_id",
            "ground_truth",
            "agent_type",
            "execution_order",
            "round",
            "time_cost",
            "average_entropy",
            "predicted_answer",
            "is_correct",
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        print(f"Saved {len(samples)} samples to {output_path}")

    def run_aggregation(self) -> None:
        """Run the complete aggregation process.

        Traverses all metrics files, extracts sample data,
        and saves correct and incorrect samples to separate CSV files.
        """
        print("Starting aggregation of experiment results...")

        all_samples = self.aggregate_all_results()

        output_dir = self.results_path / "aggregated"
        correct_csv = output_dir / "correct_samples.csv"
        incorrect_csv = output_dir / "incorrect_samples.csv"

        self.save_to_csv(all_samples["correct"], correct_csv)
        self.save_to_csv(all_samples["incorrect"], incorrect_csv)

        print(f"\nAggregation complete!")
        print(f"Correct samples: {len(all_samples['correct'])}")
        print(f"Incorrect samples: {len(all_samples['incorrect'])}")
        print(
            f"Total samples: {len(all_samples['correct']) + len(all_samples['incorrect'])}"
        )


def main():
    """Main entry point for the aggregator script.

    Parses command-line arguments and runs the aggregation process.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate multi-agent experiment results"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=str(Path.cwd()),
        help="Base path to the project directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset name to filter results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for aggregated CSV files",
    )

    args = parser.parse_args()

    aggregator = ResultsAggregator(args.base_path, args.dataset)
    aggregator.run_aggregation()


if __name__ == "__main__":
    main()
