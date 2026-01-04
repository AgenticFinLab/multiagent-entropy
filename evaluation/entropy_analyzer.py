import csv
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import torch
import numpy as np

from data_loader import DataLoader
from utils import save_csv, save_json


class EntropyAnalyzer:
    def __init__(self, base_path: str):
        self.data_loader = DataLoader(base_path)
        self.base_path = Path(base_path)

    def analyze_all_experiments_entropy(self, dataset: str) -> Dict[str, Any]:
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
        entropy_data = defaultdict(list)

        for block_name, block_info in info.items():
            for result_id in block_info["ids"]:
                parsed = self.data_loader.parse_result_id(result_id)
                main_id = parsed["main_id"]
                agent_type = parsed["agent_type"]
                execution_order = parsed["execution_order"]
                sample_number = parsed["sample_number"]

                entropy_tensor = self.data_loader.load_entropy_tensor(
                    dataset, experiment_name, result_id
                )

                if entropy_tensor is not None:
                    entropy_data[main_id].append(
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
        macro_stats = {
            "experiment_level": {},
            "round_level": defaultdict(lambda: {"total_entropy": 0.0, "count": 0}),
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

        return macro_stats

    def _calculate_micro_statistics(
        self, entropy_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        micro_stats = {
            "agent_level": defaultdict(
                lambda: {
                    "max_entropy": 0.0,
                    "mean_entropy": 0.0,
                    "variance_entropy": 0.0,
                    "median_entropy": 0.0,
                    "q1_entropy": 0.0,
                    "q3_entropy": 0.0,
                    "mean_change_rate": 0.0,
                    "std_change_rate": 0.0,
                    "token_count": 0,
                    "sample_count": 0,
                    "all_entropies": [],
                }
            ),
            "token_position_level": defaultdict(list),
        }

        for main_id, sample_entropies in entropy_data.items():
            for entropy_info in sample_entropies:
                agent_type = entropy_info["agent_type"]
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

                if len(entropy_array) > 1:
                    change_rates = np.abs(np.diff(entropy_array))
                    mean_change_rate = float(np.mean(change_rates))
                    std_change_rate = float(np.std(change_rates))
                else:
                    mean_change_rate = 0.0
                    std_change_rate = 0.0

                token_count = len(entropy_array)

                stats = micro_stats["agent_level"][agent_type]
                stats["max_entropy"] += max_entropy
                stats["mean_entropy"] += mean_entropy
                stats["variance_entropy"] += variance_entropy
                stats["median_entropy"] += median_entropy
                stats["q1_entropy"] += q1_entropy
                stats["q3_entropy"] += q3_entropy
                stats["mean_change_rate"] += mean_change_rate
                stats["std_change_rate"] += std_change_rate
                stats["token_count"] += token_count
                stats["sample_count"] += 1
                stats["all_entropies"].append(entropy_array)

                for pos, entropy_val in enumerate(entropy_array):
                    micro_stats["token_position_level"][pos].append(float(entropy_val))

        for agent_type, stats in micro_stats["agent_level"].items():
            if stats["sample_count"] > 0:
                count = stats["sample_count"]
                stats["max_entropy"] /= count
                stats["mean_entropy"] /= count
                stats["variance_entropy"] /= count
                stats["median_entropy"] /= count
                stats["q1_entropy"] /= count
                stats["q3_entropy"] /= count
                stats["mean_change_rate"] /= count
                stats["std_change_rate"] /= count
                stats["average_token_count"] = stats["token_count"] / count

                all_flat_entropies = np.concatenate(stats["all_entropies"])
                stats["overall_std"] = float(np.std(all_flat_entropies))
                stats["overall_min"] = float(np.min(all_flat_entropies))
                stats["overall_max"] = float(np.max(all_flat_entropies))
                del stats["all_entropies"]

        micro_stats["agent_level"] = dict(micro_stats["agent_level"])

        for pos, entropies in micro_stats["token_position_level"].items():
            entropies_array = np.array(entropies)
            micro_stats["token_position_level"][pos] = {
                "mean": float(np.mean(entropies_array)),
                "std": float(np.std(entropies_array)),
                "median": float(np.median(entropies_array)),
                "count": len(entropies),
            }

        micro_stats["token_position_level"] = dict(micro_stats["token_position_level"])

        return micro_stats

    def _get_round_number(
        self,
        entropy_info: Dict[str, Any],
        agent_architecture: str,
        num_rounds: int,
    ) -> int:
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

            micro_stats = results["micro_statistics"]
            for agent_type, agent_stats in micro_stats["agent_level"].items():
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
        all_results = self.analyze_all_experiments_entropy(dataset)

        rows = []
        for exp_name, results in all_results["experiments"].items():
            if "error" in results:
                continue

            micro = results["micro_statistics"]

            for agent_type, agent_stats in micro["agent_level"].items():
                rows.append(
                    {
                        "experiment_name": exp_name,
                        "agent_architecture": results["agent_architecture"],
                        "agent_type": agent_type,
                        "max_entropy": agent_stats["max_entropy"],
                        "mean_entropy": agent_stats["mean_entropy"],
                        "variance_entropy": agent_stats["variance_entropy"],
                        "median_entropy": agent_stats["median_entropy"],
                        "q1_entropy": agent_stats["q1_entropy"],
                        "q3_entropy": agent_stats["q3_entropy"],
                        "mean_change_rate": agent_stats["mean_change_rate"],
                        "std_change_rate": agent_stats["std_change_rate"],
                        "average_token_count": agent_stats["average_token_count"],
                        "sample_count": agent_stats["sample_count"],
                        "overall_std": agent_stats["overall_std"],
                        "overall_min": agent_stats["overall_min"],
                        "overall_max": agent_stats["overall_max"],
                    }
                )

        fieldnames = [
            "experiment_name",
            "agent_architecture",
            "agent_type",
            "max_entropy",
            "mean_entropy",
            "variance_entropy",
            "median_entropy",
            "q1_entropy",
            "q3_entropy",
            "mean_change_rate",
            "std_change_rate",
            "average_token_count",
            "sample_count",
            "overall_std",
            "overall_min",
            "overall_max",
        ]

        save_csv(rows, output_path, fieldnames)
        print(f"Micro statistics saved to: {output_path}")

    def save_token_position_statistics_to_csv(self, dataset: str, output_path: str):
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
            "count",
        ]

        save_csv(rows, output_path, fieldnames)
        print(f"Token position statistics saved to: {output_path}")

    def save_all_entropy_statistics_to_csv(self, dataset: str, output_dir: str):
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
        save_json(results, output_path)
        print(f"Results saved to: {output_path}")
