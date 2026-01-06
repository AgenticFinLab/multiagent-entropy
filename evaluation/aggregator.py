"""Aggregator for multi-agent experiment results.

This module provides functionality to aggregate experiment results
from multiple sources into a unified format suitable for data mining
and analysis.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json


class ResultsAggregator:
    """Aggregator for multi-agent experiment results.

    Combines entropy statistics, performance metrics, and architecture
    information into a unified format for data mining and analysis.
    """

    def __init__(self, base_path: str):
        """Initialize the results aggregator.

        Args:
            base_path: Base path to the project directory.
        """
        self.base_path = Path(base_path)
        if (self.base_path / "results").exists():
            self.results_dir = self.base_path / "results"
        elif (self.base_path / "evaluation" / "results").exists():
            self.results_dir = self.base_path / "evaluation" / "results"
        else:
            self.results_dir = self.base_path / "results"

    def aggregate_dataset_results(
        self, dataset: str
    ) -> Dict[str, Any]:
        """Aggregate all results for a specific dataset.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").

        Returns:
            Dictionary containing aggregated results in a data-mining friendly format.
        """
        dataset_dir = self.results_dir / dataset

        entropy_file = dataset_dir / "all_entropy_results.json"
        metrics_file = dataset_dir / "all_metrics.json"

        if not entropy_file.exists() or not metrics_file.exists():
            raise FileNotFoundError(
                f"Results files not found for dataset {dataset}. "
                f"Please run evaluation first."
            )

        with open(entropy_file, "r", encoding="utf-8") as f:
            entropy_data = json.load(f)

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        aggregated = {
            "dataset": dataset,
            "summary": self._create_dataset_summary(entropy_data, metrics_data),
            "experiments": self._aggregate_experiments(entropy_data, metrics_data),
            "architecture_comparison": self._compare_architectures(entropy_data, metrics_data),
            "entropy_performance_correlation": self._analyze_entropy_performance_correlation(
                entropy_data, metrics_data
            ),
            "round_analysis": self._analyze_rounds(entropy_data, metrics_data),
            "agent_analysis": self._analyze_agents(entropy_data, metrics_data),
        }

        return aggregated

    def _calculate_experiment_metrics(
        self, exp_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate metrics from experiment data.

        Args:
            exp_metrics: Experiment metrics data.

        Returns:
            Dictionary containing calculated metrics.
        """
        samples = exp_metrics.get("samples", {})
        
        total_samples = len(samples)
        correct_predictions = 0
        total_time = 0.0
        total_predictions = 0
        
        for sample_id, sample_data in samples.items():
            agents = sample_data.get("agents", {})
            for agent_name, agent_data in agents.items():
                total_time += agent_data.get("time_cost", 0)
                if agent_data.get("predicted_answer") is not None:
                    total_predictions += 1
                    if agent_data.get("is_correct", False):
                        correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        average_time = total_time / total_predictions if total_predictions > 0 else 0
        
        return {
            "accuracy": accuracy,
            "total_time": total_time,
            "average_time": average_time,
            "total_samples": total_samples,
            "total_predictions": total_predictions,
        }

    def _create_dataset_summary(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a high-level summary of the dataset.

        Args:
            entropy_data: Entropy analysis results.
            metrics_data: Performance metrics.

        Returns:
            Dictionary containing dataset summary.
        """
        summary = {
            "total_experiments": len(entropy_data.get("experiments", {})),
            "architectures": list(entropy_data.get("architectures", {}).keys()),
            "total_samples": 0,
            "total_rounds": 0,
            "average_accuracy": 0.0,
            "average_entropy": 0.0,
        }

        total_accuracy = 0.0
        total_entropy = 0.0
        exp_count = 0

        for exp_name, exp_data in entropy_data.get("experiments", {}).items():
            if "error" in exp_data:
                continue

            exp_count += 1
            macro_stats = exp_data.get("macro_statistics", {})
            exp_level = macro_stats.get("experiment_level", {})

            summary["total_samples"] += exp_level.get("total_samples", 0)
            summary["total_rounds"] += exp_data.get("num_rounds", 0)
            total_entropy += exp_level.get("average_entropy", 0)

            if exp_name in metrics_data.get("experiments", {}):
                metrics = metrics_data["experiments"][exp_name]
                calculated_metrics = self._calculate_experiment_metrics(metrics)
                total_accuracy += calculated_metrics["accuracy"]

        if exp_count > 0:
            summary["average_accuracy"] = total_accuracy / exp_count
            summary["average_entropy"] = total_entropy / exp_count

        return summary

    def _aggregate_experiments(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Aggregate individual experiment data.

        Args:
            entropy_data: Entropy analysis results.
            metrics_data: Performance metrics.

        Returns:
            List of aggregated experiment records.
        """
        experiments = []

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            if exp_name not in metrics_data.get("experiments", {}):
                continue

            exp_metrics = metrics_data["experiments"][exp_name]
            calculated_metrics = self._calculate_experiment_metrics(exp_metrics)

            aggregated_exp = {
                "experiment_name": exp_name,
                "architecture": exp_entropy.get("agent_architecture"),
                "num_rounds": exp_entropy.get("num_rounds"),
                "num_samples": exp_entropy.get("num_samples"),
                "entropy": {
                    "total": exp_entropy["macro_statistics"]["experiment_level"].get(
                        "total_entropy", 0
                    ),
                    "average": exp_entropy["macro_statistics"]["experiment_level"].get(
                        "average_entropy", 0
                    ),
                    "per_round": exp_entropy["macro_statistics"].get("round_level", {}),
                },
                "performance": {
                    "accuracy": calculated_metrics["accuracy"],
                    "total_time": calculated_metrics["total_time"],
                    "average_time": calculated_metrics["average_time"],
                },
                "agents": self._aggregate_agents_for_experiment(exp_entropy, exp_metrics),
                "trends": self._extract_trend_data(exp_entropy),
            }

            experiments.append(aggregated_exp)

        return experiments

    def _aggregate_agents_for_experiment(
        self, exp_entropy: Dict[str, Any], exp_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Aggregate agent-level data for an experiment.

        Args:
            exp_entropy: Entropy data for the experiment.
            exp_metrics: Performance metrics for the experiment.

        Returns:
            List of aggregated agent records.
        """
        agents = []

        agent_level_entropy = exp_entropy.get("macro_statistics", {}).get(
            "agent_level", {}
        )

        samples = exp_metrics.get("samples", {})

        agent_performance = defaultdict(lambda: {"correct": 0, "total": 0, "time": 0.0})

        for sample_id, sample_data in samples.items():
            agents_data = sample_data.get("agents", {})
            for agent_name, agent_data in agents_data.items():
                agent_type = agent_data.get("agent_type", agent_name.split("_")[0])
                agent_performance[agent_type]["time"] += agent_data.get("time_cost", 0)
                if agent_data.get("predicted_answer") is not None:
                    agent_performance[agent_type]["total"] += 1
                    if agent_data.get("is_correct", False):
                        agent_performance[agent_type]["correct"] += 1

        for agent_name, agent_entropy in agent_level_entropy.items():
            perf = agent_performance.get(agent_name, {"correct": 0, "total": 0, "time": 0.0})
            accuracy = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
            avg_time = perf["time"] / perf["total"] if perf["total"] > 0 else 0

            agent_data = {
                "agent_name": agent_name,
                "entropy": {
                    "total": agent_entropy.get("total_entropy", 0),
                    "average": agent_entropy.get("average_entropy", 0),
                    "mean": agent_entropy.get("mean_entropy", 0),
                    "std": agent_entropy.get("std_entropy", 0),
                    "max": agent_entropy.get("max_entropy", 0),
                    "min": agent_entropy.get("min_entropy", 0),
                    "median": agent_entropy.get("median_entropy", 0),
                    "variance": agent_entropy.get("variance_entropy", 0),
                },
                "performance": {
                    "accuracy": accuracy,
                    "average_time": avg_time,
                },
            }

            agents.append(agent_data)

        return agents

    def _compare_architectures(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare different architectures.

        Args:
            entropy_data: Entropy analysis results.
            metrics_data: Performance metrics.

        Returns:
            Dictionary containing architecture comparison data.
        """
        comparison = {}

        for arch, exp_names in entropy_data.get("architectures", {}).items():
            arch_data = {
                "experiments": [],
                "average_entropy": 0.0,
                "average_accuracy": 0.0,
                "average_time": 0.0,
                "entropy_std": 0.0,
                "accuracy_std": 0.0,
            }

            entropies = []
            accuracies = []
            times = []

            for exp_name in exp_names:
                if exp_name not in entropy_data.get("experiments", {}):
                    continue

                exp_entropy = entropy_data["experiments"][exp_name]
                if "error" in exp_entropy:
                    continue

                exp_metrics = metrics_data.get("experiments", {}).get(exp_name, {})
                if not exp_metrics:
                    continue

                calculated_metrics = self._calculate_experiment_metrics(exp_metrics)

                exp_level = exp_entropy["macro_statistics"]["experiment_level"]
                entropy = exp_level.get("average_entropy", 0)
                accuracy = calculated_metrics["accuracy"]
                time_cost = calculated_metrics["average_time"]

                entropies.append(entropy)
                accuracies.append(accuracy)
                times.append(time_cost)

                arch_data["experiments"].append(
                    {
                        "name": exp_name,
                        "entropy": entropy,
                        "accuracy": accuracy,
                        "time": time_cost,
                    }
                )

            if entropies:
                import numpy as np

                arch_data["average_entropy"] = np.mean(entropies)
                arch_data["average_accuracy"] = np.mean(accuracies)
                arch_data["average_time"] = np.mean(times)
                arch_data["entropy_std"] = np.std(entropies)
                arch_data["accuracy_std"] = np.std(accuracies)

            comparison[arch] = arch_data

        return comparison

    def _analyze_entropy_performance_correlation(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlation between entropy and performance.

        Args:
            entropy_data: Entropy analysis results.
            metrics_data: Performance metrics.

        Returns:
            Dictionary containing correlation analysis.
        """
        import numpy as np

        data_points = []

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            if exp_name not in metrics_data.get("experiments", {}):
                continue

            exp_metrics = metrics_data["experiments"][exp_name]
            calculated_metrics = self._calculate_experiment_metrics(exp_metrics)

            exp_level = exp_entropy["macro_statistics"]["experiment_level"]
            entropy = exp_level.get("average_entropy", 0)
            accuracy = calculated_metrics["accuracy"]
            time_cost = calculated_metrics["average_time"]
            architecture = exp_entropy.get("agent_architecture")

            data_points.append(
                {
                    "experiment": exp_name,
                    "architecture": architecture,
                    "entropy": entropy,
                    "accuracy": accuracy,
                    "time": time_cost,
                }
            )

        if len(data_points) < 2:
            return {
                "correlation_entropy_accuracy": 0.0,
                "correlation_entropy_time": 0.0,
                "correlation_accuracy_time": 0.0,
                "data_points": data_points,
            }

        entropies = [dp["entropy"] for dp in data_points]
        accuracies = [dp["accuracy"] for dp in data_points]
        times = [dp["time"] for dp in data_points]

        corr_entropy_accuracy = np.corrcoef(entropies, accuracies)[0, 1]
        corr_entropy_time = np.corrcoef(entropies, times)[0, 1]
        corr_accuracy_time = np.corrcoef(accuracies, times)[0, 1]

        return {
            "correlation_entropy_accuracy": float(corr_entropy_accuracy)
            if not np.isnan(corr_entropy_accuracy)
            else 0.0,
            "correlation_entropy_time": float(corr_entropy_time)
            if not np.isnan(corr_entropy_time)
            else 0.0,
            "correlation_accuracy_time": float(corr_accuracy_time)
            if not np.isnan(corr_accuracy_time)
            else 0.0,
            "data_points": data_points,
        }

    def _analyze_rounds(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze entropy and performance across rounds.

        Args:
            entropy_data: Entropy analysis results.
            metrics_data: Performance metrics.

        Returns:
            Dictionary containing round-level analysis.
        """
        import numpy as np

        round_data = defaultdict(lambda: {"entropies": [], "accuracies": [], "times": []})

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            if exp_name not in metrics_data.get("experiments", {}):
                continue

            exp_metrics = metrics_data["experiments"][exp_name]
            round_level_entropy = exp_entropy.get("macro_statistics", {}).get(
                "round_level", {}
            )

            for round_num, round_stats in round_level_entropy.items():
                round_data[round_num]["entropies"].append(
                    round_stats.get("average_entropy", 0)
                )

        round_analysis = {}

        for round_num, data in round_data.items():
            if data["entropies"]:
                round_analysis[round_num] = {
                    "average_entropy": np.mean(data["entropies"]),
                    "entropy_std": np.std(data["entropies"]),
                    "min_entropy": np.min(data["entropies"]),
                    "max_entropy": np.max(data["entropies"]),
                    "num_experiments": len(data["entropies"]),
                }

        return round_analysis

    def _analyze_agents(
        self, entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze entropy and performance by agent type.

        Args:
            entropy_data: Entropy analysis results.
            metrics_data: Performance metrics.

        Returns:
            Dictionary containing agent-level analysis.
        """
        import numpy as np

        agent_data = defaultdict(
            lambda: {
                "entropies": [],
                "mean_entropies": [],
                "std_entropies": [],
                "max_entropies": [],
                "experiments": [],
            }
        )

        for exp_name, exp_entropy in entropy_data.get("experiments", {}).items():
            if "error" in exp_entropy:
                continue

            agent_level = exp_entropy.get("macro_statistics", {}).get("agent_level", {})

            for agent_name, agent_stats in agent_level.items():
                agent_data[agent_name]["entropies"].append(
                    agent_stats.get("average_entropy", 0)
                )
                agent_data[agent_name]["mean_entropies"].append(
                    agent_stats.get("mean_entropy", 0)
                )
                agent_data[agent_name]["std_entropies"].append(
                    agent_stats.get("std_entropy", 0)
                )
                agent_data[agent_name]["max_entropies"].append(
                    agent_stats.get("max_entropy", 0)
                )
                agent_data[agent_name]["experiments"].append(exp_name)

        agent_analysis = {}

        for agent_name, data in agent_data.items():
            if data["entropies"]:
                agent_analysis[agent_name] = {
                    "average_entropy": np.mean(data["entropies"]),
                    "entropy_std": np.std(data["entropies"]),
                    "min_entropy": np.min(data["entropies"]),
                    "max_entropy": np.max(data["entropies"]),
                    "average_mean_entropy": np.mean(data["mean_entropies"]),
                    "average_std_entropy": np.mean(data["std_entropies"]),
                    "average_max_entropy": np.mean(data["max_entropies"]),
                    "num_experiments": len(data["entropies"]),
                    "experiments": data["experiments"],
                }

        return agent_analysis

    def _extract_trend_data(self, exp_entropy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract trend analysis data from entropy results.

        Args:
            exp_entropy: Entropy data for the experiment.

        Returns:
            Trend analysis data if available, None otherwise.
        """
        if "trend_analysis" not in exp_entropy:
            return None

        trend_data = exp_entropy["trend_analysis"]
        trend_stats = trend_data.get("trend_statistics", {})

        return {
            "num_rounds": trend_data.get("num_rounds", 0),
            "intra_round_stats": trend_stats.get("intra_round_stats", {}),
            "inter_round_stats": trend_stats.get("inter_round_stats", {}),
            "overall_summary": trend_stats.get("overall_summary", {}),
        }

    def save_aggregated_results(
        self, dataset: str, output_path: Optional[str] = None
    ) -> str:
        """Aggregate and save results for a dataset.

        Args:
            dataset: Dataset name.
            output_path: Optional output file path. If not provided,
                        saves to evaluation/results/{dataset}/aggregated_results.json.

        Returns:
            Path to the saved file.
        """
        aggregated = self.aggregate_dataset_results(dataset)

        if output_path is None:
            output_path = self.results_dir / dataset / "aggregated_results.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def aggregate_all_datasets(self) -> Dict[str, Any]:
        """Aggregate results for all available datasets.

        Returns:
            Dictionary containing aggregated results for all datasets.
        """
        all_results = {}

        for dataset_dir in self.results_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset = dataset_dir.name
            entropy_file = dataset_dir / "all_entropy_results.json"
            metrics_file = dataset_dir / "all_metrics.json"

            if entropy_file.exists() and metrics_file.exists():
                try:
                    all_results[dataset] = self.aggregate_dataset_results(dataset)
                except Exception as e:
                    print(f"Warning: Failed to aggregate {dataset}: {e}")

        return all_results

    def save_all_aggregated_results(self, output_path: Optional[str] = None) -> str:
        """Aggregate and save results for all datasets.

        Args:
            output_path: Optional output file path. If not provided,
                        saves to evaluation/results/all_datasets_aggregated.json.

        Returns:
            Path to the saved file.
        """
        all_results = self.aggregate_all_datasets()

        if output_path is None:
            output_path = self.results_dir / "all_datasets_aggregated.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        return str(output_path)
