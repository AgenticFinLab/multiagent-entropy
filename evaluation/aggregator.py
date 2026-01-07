"""Aggregator for multi-agent experiment results.

This module provides functionality to aggregate experiment results
from `all_entropy_results.json` and `all_metrics.json` into a unified format csv file which suitable for data mining and analysis.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List


class Aggregator:
    """Convert JSON results to CSV format for data mining."""

    def __init__(self, entropy_file: str, metrics_file: str, output_csv: str):
        """Initialize the converter.

        Args:
            entropy_file: Path to all_entropy_results.json
            metrics_file: Path to all_metrics.json
            output_csv: Path to output CSV file
        """
        self.entropy_file = Path(entropy_file)
        self.metrics_file = Path(metrics_file)
        self.output_csv = Path(output_csv)

    def load_json_files(self) -> tuple:
        """Load both JSON files.

        Returns:
            Tuple of (entropy_data, metrics_data)
        """
        with open(self.entropy_file, "r", encoding="utf-8") as f:
            entropy_data = json.load(f)

        with open(self.metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        return entropy_data, metrics_data

    def extract_sample_level_data(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract sample-level data from both JSON files.

        Args:
            entropy_data: Entropy analysis results
            metrics_data: Performance metrics

        Returns:
            List of dictionaries containing sample-level data
        """
        records = []

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            exp_metrics = metrics_data.get("experiments", {}).get(exp_name, {})
            if not exp_metrics:
                continue

            architecture = exp_entropy.get("agent_architecture", "unknown")
            num_rounds = exp_entropy.get("num_rounds", 0)

            sample_level_entropy = exp_entropy.get("micro_statistics", {}).get(
                "sample_level", {}
            )
            samples_metrics = exp_metrics.get("samples", {})

            for sample_id, sample_metrics in samples_metrics.items():
                if sample_id not in sample_level_entropy:
                    continue

                sample_entropy = sample_level_entropy[sample_id]

                ground_truth = sample_metrics.get("ground_truth", "")
                final_predicted_answer = sample_metrics.get(
                    "final_predicted_answer", ""
                )
                is_finally_correct = sample_metrics.get("is_finally_correct", False)

                for agent_key, agent_metrics in sample_metrics.get(
                    "agents", {}
                ).items():
                    agent_type = agent_metrics.get(
                        "agent_type", agent_key.split("_")[0]
                    )
                    execution_order = agent_metrics.get("execution_order", 0)
                    time_cost = agent_metrics.get("time_cost", 0)
                    avg_entropy = agent_metrics.get("average_entropy", 0)
                    predicted_answer = agent_metrics.get("predicted_answer", "")
                    is_correct = agent_metrics.get("is_correct", False)

                    record = {
                        "sample_id": sample_id,
                        "experiment_name": exp_name,
                        "architecture": architecture,
                        "ground_truth": ground_truth,
                        "agent_name": agent_type,
                        "agent_key": agent_key,
                        "execution_order": execution_order,
                        "time_cost": time_cost,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "final_predicted_answer": final_predicted_answer,
                        "is_finally_correct": is_finally_correct,
                        "sample_total_entropy": sample_entropy.get("total_entropy", 0),
                        "sample_max_entropy": sample_entropy.get("max_entropy", 0),
                        "sample_min_entropy": sample_entropy.get("min_entropy", 0),
                        "sample_mean_entropy": sample_entropy.get("mean_entropy", 0),
                        "sample_median_entropy": sample_entropy.get(
                            "median_entropy", 0
                        ),
                        "sample_std_entropy": sample_entropy.get("std_entropy", 0),
                        "sample_variance_entropy": sample_entropy.get(
                            "variance_entropy", 0
                        ),
                        "sample_q1_entropy": sample_entropy.get("q1_entropy", 0),
                        "sample_q3_entropy": sample_entropy.get("q3_entropy", 0),
                        "sample_token_count": sample_entropy.get("token_count", 0),
                        "sample_count": sample_entropy.get("sample_count", 0),
                        "sample_avg_entropy_per_token": sample_entropy.get(
                            "average_entropy_per_token", 0
                        ),
                        "agent_avg_entropy": avg_entropy,
                    }

                    records.append(record)

        return records

    def extract_round_level_data(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract round-level statistics.

        Args:
            entropy_data: Entropy analysis results
            metrics_data: Performance metrics

        Returns:
            Dictionary mapping (exp_name, round_num) to round statistics
        """
        round_stats = {}

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            round_level = exp_entropy.get("macro_statistics", {}).get("round_level", {})

            for round_num, round_data in round_level.items():
                key = (exp_name, int(round_num))
                round_stats[key] = {
                    "round_total_entropy": round_data.get("total_entropy", 0),
                    "round_count": round_data.get("count", 0),
                    "round_avg_entropy": round_data.get("average_entropy", 0),
                }

        return round_stats

    def extract_agent_level_data(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract agent-level statistics.

        Args:
            entropy_data: Entropy analysis results
            metrics_data: Performance metrics

        Returns:
            Dictionary mapping (exp_name, agent_name) to agent statistics
        """
        agent_stats = {}

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            agent_level = exp_entropy.get("macro_statistics", {}).get("agent_level", {})

            for agent_name, agent_data in agent_level.items():
                key = (exp_name, agent_name)
                agent_stats[key] = {
                    "agent_total_entropy": agent_data.get("total_entropy", 0),
                    "agent_sample_count": agent_data.get("sample_count", 0),
                    "agent_total_tokens": agent_data.get("total_tokens", 0),
                    "agent_avg_entropy": agent_data.get("average_entropy", 0),
                    "agent_mean_entropy": agent_data.get("mean_entropy", 0),
                    "agent_max_entropy": agent_data.get("max_entropy", 0),
                    "agent_min_entropy": agent_data.get("min_entropy", 0),
                    "agent_median_entropy": agent_data.get("median_entropy", 0),
                    "agent_std_entropy": agent_data.get("std_entropy", 0),
                    "agent_variance_entropy": agent_data.get("variance_entropy", 0),
                    "agent_q1_entropy": agent_data.get("q1_entropy", 0),
                    "agent_q3_entropy": agent_data.get("q3_entropy", 0),
                }

        return agent_stats

    def extract_experiment_level_data(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract experiment-level statistics.

        Args:
            entropy_data: Entropy analysis results
            metrics_data: Performance metrics

        Returns:
            Dictionary mapping exp_name to experiment statistics
        """
        exp_stats = {}

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            exp_metrics = metrics_data.get("experiments", {}).get(exp_name, {})
            if not exp_metrics:
                continue

            exp_level = exp_entropy.get("macro_statistics", {}).get(
                "experiment_level", {}
            )

            samples = exp_metrics.get("samples", {})
            total_correct = 0
            total_predictions = 0
            total_time = 0

            for sample_id, sample_data in samples.items():
                for agent_key, agent_data in sample_data.get("agents", {}).items():
                    total_time += agent_data.get("time_cost", 0)
                    if agent_data.get("predicted_answer") is not None:
                        total_predictions += 1
                        if agent_data.get("is_correct", False):
                            total_correct += 1

            accuracy = total_correct / total_predictions if total_predictions > 0 else 0
            avg_time = total_time / total_predictions if total_predictions > 0 else 0

            exp_stats[exp_name] = {
                "exp_total_entropy": exp_level.get("total_entropy", 0),
                "exp_avg_entropy": exp_level.get("average_entropy", 0),
                "exp_total_samples": exp_level.get("total_samples", 0),
                "exp_accuracy": accuracy,
                "exp_total_time": total_time,
                "exp_avg_time": avg_time,
            }

        return exp_stats

    def merge_all_data(
        self,
        sample_records: List[Dict[str, Any]],
        round_stats: Dict[str, Dict[str, Any]],
        agent_stats: Dict[str, Dict[str, Any]],
        exp_stats: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge all levels of data into sample records.

        Args:
            sample_records: Sample-level records
            round_stats: Round-level statistics
            agent_stats: Agent-level statistics
            exp_stats: Experiment-level statistics

        Returns:
            List of merged records
        """
        merged_records = []

        for record in sample_records:
            exp_name = record["experiment_name"]
            agent_name = record["agent_name"]

            key = (exp_name, agent_name)
            if key in agent_stats:
                record.update(agent_stats[key])

            if exp_name in exp_stats:
                record.update(exp_stats[exp_name])

            merged_records.append(record)

        return merged_records

    def convert_to_csv(self):
        """Convert JSON files to CSV format."""
        entropy_data, metrics_data = self.load_json_files()

        sample_records = self.extract_sample_level_data(entropy_data, metrics_data)

        round_stats = self.extract_round_level_data(entropy_data, metrics_data)

        agent_stats = self.extract_agent_level_data(entropy_data, metrics_data)

        exp_stats = self.extract_experiment_level_data(entropy_data, metrics_data)

        merged_records = self.merge_all_data(
            sample_records, round_stats, agent_stats, exp_stats
        )

        if not merged_records:
            return

        fieldnames = list(merged_records[0].keys())

        with open(self.output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_records)

        return merged_records


def main():
    """Main function to run the converter."""
    base_path = Path("/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k")

    entropy_file = base_path / "all_entropy_results.json"
    metrics_file = base_path / "all_metrics.json"
    output_csv = base_path / "aggregated_data.csv"

    converter = JSONToCSVConverter(
        str(entropy_file), str(metrics_file), str(output_csv)
    )

    converter.convert_to_csv()


if __name__ == "__main__":
    main()
