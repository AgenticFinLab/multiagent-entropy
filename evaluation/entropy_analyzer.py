"""Entropy analyzer for multi-agent system experiments.

This module provides functionality to analyze entropy statistics
from experiment results, including macro and micro level analysis,
and comparison across different agent architectures.
"""

import csv
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import torch
import numpy as np

from data_loader import DataLoader
from utils import save_csv, save_json


class EntropyAnalyzer:
    """Analyzer for entropy statistics in multi-agent experiments.

    Provides methods to analyze entropy at multiple levels:
    - Experiment level: Overall entropy statistics
    - Round level: Entropy per round
    - Agent level: Entropy per agent type
    - Sample level: Entropy per sample
    - Sequence level: Entropy per execution sequence
    - Token position level: Entropy distribution across token positions
    """

    def __init__(self, base_path: str):
        """Initialize the entropy analyzer with base path.

        Args:
            base_path: Base path to the project directory.
        """
        self.data_loader = DataLoader(base_path)
        self.base_path = Path(base_path)

    def analyze_all_experiments_entropy(self, dataset: str) -> Dict[str, Any]:
        """Analyze entropy for all experiments in a dataset.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").

        Returns:
            Dictionary containing entropy analysis results for all experiments,
            organized by architecture and experiment name.
        """
        experiments = self.data_loader.get_experiments_by_dataset(dataset)

        all_results = {
            "dataset": dataset,
            "architectures": defaultdict(list),
            "experiments": {},
        }

        for experiment_name in experiments:
            try:
                experiment_results = self.analyze_experiment_entropy(
                    dataset, experiment_name
                )
                all_results["experiments"][experiment_name] = experiment_results

                arch = experiment_results["agent_architecture"]
                all_results["architectures"][arch].append(experiment_name)
            except Exception as e:
                print(f"Error analyzing experiment {experiment_name}: {e}")
                all_results["experiments"][experiment_name] = {"error": str(e)}

        all_results["architectures"] = dict(all_results["architectures"])

        return all_results

    def analyze_experiment_entropy(
        self, dataset: str, experiment_name: str
    ) -> Dict[str, Any]:
        """Analyze entropy for a single experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing macro and micro level entropy statistics.
        """
        config = self.data_loader.load_experiment_config(experiment_name)
        agent_architecture = config.get("agent_type", "unknown")
        num_rounds = config.get("round", 1)

        info = self.data_loader.load_result_store_info(dataset, experiment_name)

        entropy_data = self._collect_entropy_data(dataset, experiment_name, info)

        macro_stats = self._calculate_macro_statistics(
            entropy_data, agent_architecture, num_rounds
        )
        micro_stats = self._calculate_micro_statistics(entropy_data)

        results = {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "agent_architecture": agent_architecture,
            "num_rounds": num_rounds,
            "num_samples": len(entropy_data),
            "macro_statistics": macro_stats,
            "micro_statistics": micro_stats,
        }

        return results

    def _collect_entropy_data(
        self, dataset: str, experiment_name: str, info: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect entropy tensors for all results in an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment.
            info: Result store information.

        Returns:
            Dictionary mapping sequence IDs to entropy data.
        """
        entropy_data = defaultdict(list)

        for block_name, block_info in info.items():
            for result_id in block_info["ids"]:
                parsed = self.data_loader.parse_result_id(result_id)
                main_id = parsed["main_id"]
                agent_type = parsed["agent_type"]
                execution_order = parsed["execution_order"]
                sample_number = parsed["sample_number"]

                sequence_id = f"{main_id}-{agent_type}-{execution_order}"

                entropy_tensor = self.data_loader.load_entropy_tensor(
                    dataset, experiment_name, result_id
                )

                if entropy_tensor is not None:
                    entropy_data[sequence_id].append(
                        {
                            "result_id": result_id,
                            "agent_type": agent_type,
                            "execution_order": execution_order,
                            "sample_number": sample_number,
                            "entropy_tensor": entropy_tensor,
                        }
                    )

        return dict(entropy_data)

    def _calculate_macro_statistics(
        self,
        entropy_data: Dict[str, List[Dict[str, Any]]],
        agent_architecture: str,
        num_rounds: int,
    ) -> Dict[str, Any]:
        """Calculate macro-level entropy statistics.

        Args:
            entropy_data: Dictionary of entropy data by sequence.
            agent_architecture: Type of agent architecture.
            num_rounds: Number of rounds in the experiment.

        Returns:
            Dictionary containing experiment, round, and agent level statistics.
        """
        macro_stats = {
            "experiment_level": {},
            "round_level": defaultdict(lambda: {"total_entropy": 0.0, "count": 0}),
            "agent_level": defaultdict(
                lambda: {
                    "total_entropy": 0.0,
                    "sample_count": 0,
                    "total_tokens": 0,
                    "all_entropies": [],
                }
            ),
            "architecture_comparison": {},
        }

        total_experiment_entropy = 0.0
        total_count = 0

        for main_id, sample_entropies in entropy_data.items():
            for entropy_info in sample_entropies:
                entropy_tensor = entropy_info["entropy_tensor"]
                entropy_sum = float(entropy_tensor.sum().item())

                total_experiment_entropy += entropy_sum
                total_count += 1

                round_num = self._get_round_number(
                    entropy_info, agent_architecture, num_rounds
                )
                macro_stats["round_level"][round_num]["total_entropy"] += entropy_sum
                macro_stats["round_level"][round_num]["count"] += 1

                agent_type = entropy_info["agent_type"]
                macro_stats["agent_level"][agent_type]["total_entropy"] += entropy_sum
                macro_stats["agent_level"][agent_type]["sample_count"] += 1
                macro_stats["agent_level"][agent_type]["total_tokens"] += len(
                    entropy_tensor
                )

                if isinstance(entropy_tensor, torch.Tensor):
                    entropy_array = entropy_tensor.cpu().numpy()
                else:
                    entropy_array = np.array(entropy_tensor)
                macro_stats["agent_level"][agent_type]["all_entropies"].append(
                    entropy_array
                )

        macro_stats["experiment_level"]["total_entropy"] = total_experiment_entropy
        macro_stats["experiment_level"]["average_entropy"] = (
            total_experiment_entropy / total_count if total_count > 0 else 0.0
        )
        macro_stats["experiment_level"]["total_samples"] = len(entropy_data)
        macro_stats["experiment_level"]["total_results"] = total_count

        for round_num, round_data in macro_stats["round_level"].items():
            if round_data["count"] > 0:
                round_data["average_entropy"] = (
                    round_data["total_entropy"] / round_data["count"]
                )

        macro_stats["round_level"] = dict(macro_stats["round_level"])

        for agent_type, agent_data in macro_stats["agent_level"].items():
            if agent_data["sample_count"] > 0:
                all_flat_entropies = np.concatenate(agent_data["all_entropies"])
                agent_data["total_entropy"] = float(agent_data["total_entropy"])
                agent_data["average_entropy"] = (
                    agent_data["total_entropy"] / agent_data["sample_count"]
                )
                agent_data["mean_entropy"] = float(np.mean(all_flat_entropies))
                agent_data["max_entropy"] = float(np.max(all_flat_entropies))
                agent_data["min_entropy"] = float(np.min(all_flat_entropies))
                agent_data["median_entropy"] = float(np.median(all_flat_entropies))
                agent_data["std_entropy"] = float(np.std(all_flat_entropies))
                agent_data["variance_entropy"] = float(np.var(all_flat_entropies))
                agent_data["q1_entropy"] = float(np.percentile(all_flat_entropies, 25))
                agent_data["q3_entropy"] = float(np.percentile(all_flat_entropies, 75))
                del agent_data["all_entropies"]

        macro_stats["agent_level"] = dict(macro_stats["agent_level"])

        return macro_stats

    def _calculate_micro_statistics(
        self, entropy_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate micro-level entropy statistics.

        Args:
            entropy_data: Dictionary of entropy data by sequence.

        Returns:
            Dictionary containing sample, sequence, and token position level statistics.
        """
        micro_stats = {
            "sample_level": defaultdict(
                lambda: {
                    "total_entropy": 0.0,
                    "max_entropy": 0.0,
                    "mean_entropy": 0.0,
                    "variance_entropy": 0.0,
                    "median_entropy": 0.0,
                    "q1_entropy": 0.0,
                    "q3_entropy": 0.0,
                    "std_entropy": 0.0,
                    "min_entropy": 0.0,
                    "token_count": 0,
                    "sample_count": 0,
                }
            ),
            "sequence_level": defaultdict(
                lambda: {
                    "total_entropy": 0.0,
                    "max_entropy": 0.0,
                    "mean_entropy": 0.0,
                    "variance_entropy": 0.0,
                    "median_entropy": 0.0,
                    "q1_entropy": 0.0,
                    "q3_entropy": 0.0,
                    "std_entropy": 0.0,
                    "min_entropy": 0.0,
                    "token_count": 0,
                    "sample_count": 0,
                }
            ),
            "token_position_level": defaultdict(list),
        }

        for sequence_id, sample_entropies in entropy_data.items():
            sequence_total_entropy = 0.0
            sequence_max_entropy = 0.0
            sequence_mean_entropy = 0.0
            sequence_variance_entropy = 0.0
            sequence_median_entropy = 0.0
            sequence_q1_entropy = 0.0
            sequence_q3_entropy = 0.0
            sequence_std_entropy = 0.0
            sequence_min_entropy = 0.0
            sequence_token_count = 0
            sequence_sample_count = len(sample_entropies)

            for entropy_info in sample_entropies:
                entropy_tensor = entropy_info["entropy_tensor"]

                if isinstance(entropy_tensor, torch.Tensor):
                    entropy_array = entropy_tensor.cpu().numpy()
                else:
                    entropy_array = np.array(entropy_tensor)

                max_entropy = float(np.max(entropy_array))
                mean_entropy = float(np.mean(entropy_array))
                variance_entropy = float(np.var(entropy_array))
                median_entropy = float(np.median(entropy_array))
                q1_entropy = float(np.percentile(entropy_array, 25))
                q3_entropy = float(np.percentile(entropy_array, 75))
                std_entropy = float(np.std(entropy_array))
                min_entropy = float(np.min(entropy_array))
                entropy_sum = float(np.sum(entropy_array))
                token_count = len(entropy_array)

                sequence_total_entropy += entropy_sum
                sequence_max_entropy += max_entropy
                sequence_mean_entropy += mean_entropy
                sequence_variance_entropy += variance_entropy
                sequence_median_entropy += median_entropy
                sequence_q1_entropy += q1_entropy
                sequence_q3_entropy += q3_entropy
                sequence_std_entropy += std_entropy
                sequence_min_entropy += min_entropy
                sequence_token_count += token_count

                for pos, entropy_val in enumerate(entropy_array):
                    micro_stats["token_position_level"][pos].append(float(entropy_val))

            if sequence_sample_count > 0:
                seq_stats = micro_stats["sequence_level"][sequence_id]
                seq_stats["total_entropy"] = sequence_total_entropy
                seq_stats["max_entropy"] = sequence_max_entropy / sequence_sample_count
                seq_stats["mean_entropy"] = (
                    sequence_mean_entropy / sequence_sample_count
                )
                seq_stats["variance_entropy"] = (
                    sequence_variance_entropy / sequence_sample_count
                )
                seq_stats["median_entropy"] = (
                    sequence_median_entropy / sequence_sample_count
                )
                seq_stats["q1_entropy"] = sequence_q1_entropy / sequence_sample_count
                seq_stats["q3_entropy"] = sequence_q3_entropy / sequence_sample_count
                seq_stats["std_entropy"] = sequence_std_entropy / sequence_sample_count
                seq_stats["min_entropy"] = sequence_min_entropy / sequence_sample_count
                seq_stats["token_count"] = sequence_token_count
                seq_stats["sample_count"] = sequence_sample_count
                seq_stats["average_entropy_per_token"] = (
                    sequence_total_entropy / sequence_token_count
                    if sequence_token_count > 0
                    else 0.0
                )

            main_id = sequence_id.split("-")[0]
            if main_id not in micro_stats["sample_level"]:
                micro_stats["sample_level"][main_id] = {
                    "total_entropy": 0.0,
                    "max_entropy": 0.0,
                    "mean_entropy": 0.0,
                    "variance_entropy": 0.0,
                    "median_entropy": 0.0,
                    "q1_entropy": 0.0,
                    "q3_entropy": 0.0,
                    "std_entropy": 0.0,
                    "min_entropy": 0.0,
                    "token_count": 0,
                    "sample_count": 0,
                }

            stats = micro_stats["sample_level"][main_id]
            stats["total_entropy"] += sequence_total_entropy
            stats["max_entropy"] += (
                sequence_max_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["mean_entropy"] += (
                sequence_mean_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["variance_entropy"] += (
                sequence_variance_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["median_entropy"] += (
                sequence_median_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["q1_entropy"] += (
                sequence_q1_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["q3_entropy"] += (
                sequence_q3_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["std_entropy"] += (
                sequence_std_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["min_entropy"] += (
                sequence_min_entropy / sequence_sample_count
                if sequence_sample_count > 0
                else 0.0
            )
            stats["token_count"] += sequence_token_count
            stats["sample_count"] += sequence_sample_count

        micro_stats["sample_level"] = dict(micro_stats["sample_level"])

        for main_id, stats in micro_stats["sample_level"].items():
            if stats["sample_count"] > 0:
                stats["max_entropy"] = stats["max_entropy"] / stats["sample_count"]
                stats["mean_entropy"] = stats["mean_entropy"] / stats["sample_count"]
                stats["variance_entropy"] = (
                    stats["variance_entropy"] / stats["sample_count"]
                )
                stats["median_entropy"] = (
                    stats["median_entropy"] / stats["sample_count"]
                )
                stats["q1_entropy"] = stats["q1_entropy"] / stats["sample_count"]
                stats["q3_entropy"] = stats["q3_entropy"] / stats["sample_count"]
                stats["std_entropy"] = stats["std_entropy"] / stats["sample_count"]
                stats["min_entropy"] = stats["min_entropy"] / stats["sample_count"]
                stats["average_entropy_per_token"] = (
                    stats["total_entropy"] / stats["token_count"]
                    if stats["token_count"] > 0
                    else 0.0
                )

        for sequence_key, seq_stats in micro_stats["sequence_level"].items():
            if seq_stats["sample_count"] > 0:
                seq_stats["max_entropy"] = (
                    seq_stats["max_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["mean_entropy"] = (
                    seq_stats["mean_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["variance_entropy"] = (
                    seq_stats["variance_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["median_entropy"] = (
                    seq_stats["median_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["q1_entropy"] = (
                    seq_stats["q1_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["q3_entropy"] = (
                    seq_stats["q3_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["std_entropy"] = (
                    seq_stats["std_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["min_entropy"] = (
                    seq_stats["min_entropy"] / seq_stats["sample_count"]
                )
                seq_stats["average_entropy_per_token"] = (
                    seq_stats["total_entropy"] / seq_stats["token_count"]
                    if seq_stats["token_count"] > 0
                    else 0.0
                )

        micro_stats["sequence_level"] = dict(micro_stats["sequence_level"])

        for pos, entropies in micro_stats["token_position_level"].items():
            entropies_array = np.array(entropies)
            micro_stats["token_position_level"][pos] = {
                "mean": float(np.mean(entropies_array)),
                "std": float(np.std(entropies_array)),
                "median": float(np.median(entropies_array)),
                "count": len(entropies),
                "min": float(np.min(entropies_array)),
                "max": float(np.max(entropies_array)),
                "q1": float(np.percentile(entropies_array, 25)),
                "q3": float(np.percentile(entropies_array, 75)),
            }

        micro_stats["token_position_level"] = dict(micro_stats["token_position_level"])

        return micro_stats

    def _get_round_number(
        self,
        entropy_info: Dict[str, Any],
        agent_architecture: str,
        num_rounds: int,
    ) -> int:
        """Get the round number for an entropy result.

        Args:
            entropy_info: Dictionary containing entropy information.
            agent_architecture: Type of agent architecture.
            num_rounds: Total number of rounds.

        Returns:
            Round number for the entropy result.
        """
        execution_order = entropy_info["execution_order"]
        agent_type = entropy_info["agent_type"]

        if agent_architecture == "single":
            return execution_order
        elif agent_architecture == "debate":
            if agent_type == "orchestrator":
                return num_rounds
            else:
                return (execution_order - 1) // 3 + 1
        else:
            return (execution_order - 1) // 4 + 1

    def compare_architectures_entropy(self, dataset: str) -> Dict[str, Any]:
        """Compare entropy statistics across different architectures.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").

        Returns:
            Dictionary containing architecture comparison results.
        """
        all_results = self.analyze_all_experiments_entropy(dataset)

        comparison = {
            "dataset": dataset,
            "architectures": {},
            "trends": {},
            "distribution_analysis": {},
        }

        for exp_name, results in all_results["experiments"].items():
            if "error" in results:
                continue

            arch = results["agent_architecture"]
            if arch not in comparison["architectures"]:
                comparison["architectures"][arch] = []

            comparison["architectures"][arch].append(
                {
                    "experiment_name": exp_name,
                    "total_entropy": results["macro_statistics"]["experiment_level"][
                        "total_entropy"
                    ],
                    "average_entropy": results["macro_statistics"]["experiment_level"][
                        "average_entropy"
                    ],
                    "num_samples": results["macro_statistics"]["experiment_level"][
                        "total_samples"
                    ],
                }
            )

        for arch, exps in comparison["architectures"].items():
            if len(exps) > 0:
                avg_entropies = [exp["average_entropy"] for exp in exps]
                comparison["trends"][arch] = {
                    "mean": float(np.mean(avg_entropies)),
                    "std": float(np.std(avg_entropies)),
                    "min": float(np.min(avg_entropies)),
                    "max": float(np.max(avg_entropies)),
                    "count": len(exps),
                }

        comparison["distribution_analysis"] = self._analyze_entropy_distribution(
            all_results
        )

        return comparison

    def _analyze_entropy_distribution(
        self, all_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze entropy distribution across architectures.

        Args:
            all_results: Dictionary containing all experiment results.

        Returns:
            Dictionary containing distribution analysis results.
        """
        distribution = {"architecture_comparison": {}}

        for exp_name, results in all_results["experiments"].items():
            if "error" in results:
                continue

            arch = results["agent_architecture"]
            if arch not in distribution["architecture_comparison"]:
                distribution["architecture_comparison"][arch] = {
                    "all_entropies": [],
                    "agent_entropies": defaultdict(list),
                }

            macro_stats = results["macro_statistics"]
            for agent_type, agent_stats in macro_stats["agent_level"].items():
                distribution["architecture_comparison"][arch]["agent_entropies"][
                    agent_type
                ].append(agent_stats["mean_entropy"])

        for arch, data in distribution["architecture_comparison"].items():
            arch_comparison = {}
            for agent_type, entropies in data["agent_entropies"].items():
                entropies_array = np.array(entropies)
                arch_comparison[agent_type] = {
                    "mean": float(np.mean(entropies_array)),
                    "std": float(np.std(entropies_array)),
                    "min": float(np.min(entropies_array)),
                    "max": float(np.max(entropies_array)),
                }

            distribution["architecture_comparison"][arch] = arch_comparison

        return distribution

    def save_macro_statistics_to_csv(self, dataset: str, output_path: str):
        """Save macro-level entropy statistics to CSV file.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            output_path: Path to output CSV file.
        """
        all_results = self.analyze_all_experiments_entropy(dataset)

        rows = []
        for exp_name, results in all_results["experiments"].items():
            if "error" in results:
                continue

            macro = results["macro_statistics"]
            exp_level = macro["experiment_level"]

            rows.append(
                {
                    "experiment_name": exp_name,
                    "agent_architecture": results["agent_architecture"],
                    "num_rounds": results["num_rounds"],
                    "num_samples": exp_level["total_samples"],
                    "total_results": exp_level["total_results"],
                    "total_entropy": exp_level["total_entropy"],
                    "average_entropy": exp_level["average_entropy"],
                }
            )

            for round_num, round_data in macro["round_level"].items():
                rows.append(
                    {
                        "experiment_name": exp_name,
                        "agent_architecture": results["agent_architecture"],
                        "level": "round",
                        "round_number": round_num,
                        "total_entropy": round_data["total_entropy"],
                        "average_entropy": round_data["average_entropy"],
                        "count": round_data["count"],
                    }
                )

        fieldnames = [
            "experiment_name",
            "agent_architecture",
            "num_rounds",
            "num_samples",
            "total_results",
            "total_entropy",
            "average_entropy",
            "level",
            "round_number",
            "count",
        ]

        save_csv(rows, output_path, fieldnames)
        print(f"Macro statistics saved to: {output_path}")

    def save_micro_statistics_to_csv(self, dataset: str, output_path: str):
        """Save micro-level entropy statistics to CSV file.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            output_path: Path to output CSV file.
        """
        all_results = self.analyze_all_experiments_entropy(dataset)

        rows = []
        for exp_name, results in all_results["experiments"].items():
            if "error" in results:
                continue

            micro = results["micro_statistics"]

            for sequence_id, seq_stats in micro["sequence_level"].items():
                rows.append(
                    {
                        "experiment_name": exp_name,
                        "agent_architecture": results["agent_architecture"],
                        "sequence_id": sequence_id,
                        "total_entropy": seq_stats["total_entropy"],
                        "max_entropy": seq_stats["max_entropy"],
                        "mean_entropy": seq_stats["mean_entropy"],
                        "variance_entropy": seq_stats["variance_entropy"],
                        "median_entropy": seq_stats["median_entropy"],
                        "q1_entropy": seq_stats["q1_entropy"],
                        "q3_entropy": seq_stats["q3_entropy"],
                        "std_entropy": seq_stats["std_entropy"],
                        "min_entropy": seq_stats["min_entropy"],
                        "token_count": seq_stats["token_count"],
                        "sample_count": seq_stats["sample_count"],
                        "average_entropy_per_token": seq_stats[
                            "average_entropy_per_token"
                        ],
                    }
                )

        fieldnames = [
            "experiment_name",
            "agent_architecture",
            "sequence_id",
            "total_entropy",
            "max_entropy",
            "mean_entropy",
            "variance_entropy",
            "median_entropy",
            "q1_entropy",
            "q3_entropy",
            "std_entropy",
            "min_entropy",
            "token_count",
            "sample_count",
            "average_entropy_per_token",
        ]

        save_csv(rows, output_path, fieldnames)
        print(f"Micro statistics saved to: {output_path}")

    def save_token_position_statistics_to_csv(self, dataset: str, output_path: str):
        """Save token position level entropy statistics to CSV file.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            output_path: Path to output CSV file.
        """
        all_results = self.analyze_all_experiments_entropy(dataset)

        rows = []
        for exp_name, results in all_results["experiments"].items():
            if "error" in results:
                continue

            micro = results["micro_statistics"]

            for pos, pos_stats in micro["token_position_level"].items():
                rows.append(
                    {
                        "experiment_name": exp_name,
                        "agent_architecture": results["agent_architecture"],
                        "token_position": pos,
                        "mean_entropy": pos_stats["mean"],
                        "std_entropy": pos_stats["std"],
                        "median_entropy": pos_stats["median"],
                        "min_entropy": pos_stats["min"],
                        "max_entropy": pos_stats["max"],
                        "q1_entropy": pos_stats["q1"],
                        "q3_entropy": pos_stats["q3"],
                        "count": pos_stats["count"],
                    }
                )

        fieldnames = [
            "experiment_name",
            "agent_architecture",
            "token_position",
            "mean_entropy",
            "std_entropy",
            "median_entropy",
            "min_entropy",
            "max_entropy",
            "q1_entropy",
            "q3_entropy",
            "count",
        ]

        save_csv(rows, output_path, fieldnames)
        print(f"Token position statistics saved to: {output_path}")

    def save_all_entropy_statistics_to_csv(self, dataset: str, output_dir: str):
        """Save all entropy statistics to CSV files.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            output_dir: Directory to save CSV files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        macro_csv = output_path / "macro_statistics.csv"
        micro_csv = output_path / "micro_statistics.csv"
        token_pos_csv = output_path / "token_position_statistics.csv"

        self.save_macro_statistics_to_csv(dataset, str(macro_csv))
        self.save_micro_statistics_to_csv(dataset, str(micro_csv))
        self.save_token_position_statistics_to_csv(dataset, str(token_pos_csv))

        comparison = self.compare_architectures_entropy(dataset)
        comparison_csv = output_path / "architecture_comparison.csv"

        comparison_rows = []
        for arch, exps in comparison["architectures"].items():
            for exp in exps:
                comparison_rows.append(
                    {
                        "agent_architecture": arch,
                        "experiment_name": exp["experiment_name"],
                        "total_entropy": exp["total_entropy"],
                        "average_entropy": exp["average_entropy"],
                        "num_samples": exp["num_samples"],
                    }
                )

        if arch in comparison["trends"]:
            trend = comparison["trends"][arch]
            comparison_rows.append(
                {
                    "agent_architecture": arch,
                    "experiment_name": "TREND_SUMMARY",
                    "total_entropy": None,
                    "average_entropy": trend["mean"],
                    "num_samples": trend["count"],
                    "trend_std": trend["std"],
                    "trend_min": trend["min"],
                    "trend_max": trend["max"],
                }
            )

        fieldnames = [
            "agent_architecture",
            "experiment_name",
            "total_entropy",
            "average_entropy",
            "num_samples",
            "trend_std",
            "trend_min",
            "trend_max",
        ]

        with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        print(f"Architecture comparison saved to: {comparison_csv}")

        print(f"\nAll entropy statistics saved to: {output_dir}")

    def save_results_json(self, results: Dict[str, Any], output_path: str):
        """Save entropy analysis results to JSON file.

        Args:
            results: Dictionary containing entropy analysis results.
            output_path: Path to output JSON file.
        """
        save_json(results, output_path)
        print(f"Results saved to: {output_path}")
