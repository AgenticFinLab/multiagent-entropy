import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

from data_loader import DataLoader
from metrics_calculator import MetricsCalculator


class ExperimentAnalyzer:
    def __init__(self, base_path: str):
        self.data_loader = DataLoader(base_path)
        self.metrics_calculator = MetricsCalculator()

    def analyze_experiment(
        self, dataset: str, experiment_name: str, task_type: str = "math"
    ) -> Dict[str, Any]:
        config = self.data_loader.load_experiment_config(experiment_name)
        agent_architecture = config.get("agent_type", "unknown")
        num_rounds = config.get("round", 1)

        ground_truths = self.data_loader.load_ground_truth(dataset)
        all_results = self.data_loader.load_all_results(dataset, experiment_name)

        results_by_sample = defaultdict(list)
        for result_id, result_data in all_results.items():
            parsed = self.data_loader.parse_result_id(result_id)
            main_id = parsed["main_id"]
            results_by_sample[main_id].append(
                {
                    "result_id": result_id,
                    "agent_type": parsed["agent_type"],
                    "execution_order": parsed["execution_order"],
                    "data": result_data,
                }
            )

        metrics = {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "task_type": task_type,
            "agent_architecture": agent_architecture,
            "num_rounds": num_rounds,
            "num_samples": len(results_by_sample),
            "samples": {},
        }

        for main_id, sample_results in results_by_sample.items():
            sample_metrics = self._analyze_sample(
                main_id,
                sample_results,
                ground_truths.get(main_id),
                agent_architecture,
                dataset,
                experiment_name,
            )
            metrics["samples"][main_id] = sample_metrics

        metrics["summary"] = self._calculate_summary(
            metrics["samples"], agent_architecture
        )

        return metrics

    def _analyze_sample(
        self,
        main_id: str,
        sample_results: List[Dict[str, Any]],
        ground_truth: Optional[Dict[str, Any]],
        agent_architecture: str,
        dataset: str,
        experiment_name: str,
    ) -> Dict[str, Any]:
        sample_metrics = {
            "main_id": main_id,
            "ground_truth": ground_truth["groundtruth"] if ground_truth else None,
            "agents": {},
        }

        for result_info in sample_results:
            result_id = result_info["result_id"]
            agent_type = result_info["agent_type"]
            execution_order = result_info["execution_order"]
            result_data = result_info["data"]

            time_cost = self.metrics_calculator.calculate_time_cost(result_data)

            entropy_tensor = self.data_loader.load_entropy_tensor(
                dataset, experiment_name, result_id
            )
            avg_entropy = self.metrics_calculator.calculate_average_entropy(
                entropy_tensor
            )

            if "final_answer" in result_data:
                response = result_data["final_answer"]
                predicted_answer = response
            else:
                response = result_data.get("response", "")
                predicted_answer = self.metrics_calculator.extract_boxed_answer(
                    response
                )

            is_correct = False
            if ground_truth and predicted_answer:
                is_correct = self.metrics_calculator.is_answer_correct(
                    predicted_answer, ground_truth["groundtruth"]
                )

            if agent_architecture == "single":
                round_num = execution_order
                agent_key = f"{agent_type}_round_{round_num}"
            elif agent_architecture == "debate":
                if agent_type == "orchestrator":
                    agent_key = "orchestrator"
                else:
                    round_num = (execution_order - 1) // 3 + 1
                    agent_key = f"{agent_type}_round_{round_num}"
            else:
                round_num = (execution_order - 1) // 4 + 1
                agent_key = f"{agent_type}_round_{round_num}"
            sample_metrics["agents"][agent_key] = {
                "agent_type": agent_type,
                "execution_order": execution_order,
                "time_cost": time_cost,
                "average_entropy": avg_entropy,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
            }

        return sample_metrics

    def _calculate_summary(
        self, samples: Dict[str, Any], agent_architecture: str
    ) -> Dict[str, Any]:
        summary = {
            "total_samples": len(samples),
            "last_agent_stats": {},
            "agent_stats": defaultdict(
                lambda: {
                    "count": 0,
                    "correct": 0,
                    "has_format": 0,
                    "total_time": 0.0,
                    "total_entropy": 0.0,
                }
            ),
        }

        for main_id, sample in samples.items():
            ground_truth = sample["ground_truth"]

            for agent_key, agent_data in sample["agents"].items():
                stats = summary["agent_stats"][agent_key]
                stats["count"] += 1
                stats["total_time"] += agent_data["time_cost"]
                stats["total_entropy"] += agent_data["average_entropy"]

                if agent_data["is_correct"]:
                    stats["correct"] += 1

                if agent_data["predicted_answer"] is not None:
                    stats["has_format"] += 1

        for agent_key, stats in summary["agent_stats"].items():
            if stats["count"] > 0:
                stats["accuracy"] = stats["correct"] / stats["count"]
                stats["format_compliance_rate"] = stats["has_format"] / stats["count"]
                stats["average_time"] = stats["total_time"] / stats["count"]
                stats["average_entropy"] = stats["total_entropy"] / stats["count"]

        summary["agent_stats"] = dict(summary["agent_stats"])
        summary["last_agent_stats"] = dict(
            summary["agent_stats"][self._get_final_agent_keys(agent_architecture)[-1]]
        )

        return summary

    def _get_final_agent_keys(self, agent_architecture: str) -> List[str]:
        if agent_architecture == "single":
            return ["SingleSolver_round_1", "SingleSolver_round_2"]
        elif agent_architecture == "sequential":
            return ["judger_round_1", "judger_round_2"]
        elif agent_architecture == "centralized":
            return ["OrchestratorAgent_round_1", "OrchestratorAgent_round_2"]
        elif agent_architecture == "debate":
            return ["orchestrator"]
        elif agent_architecture == "hybrid":
            return ["OrchestratorAgent_round_1", "OrchestratorAgent_round_2"]
        else:
            return []

    def analyze_all_experiments(
        self, dataset: str, task_type: str = "math"
    ) -> Dict[str, Any]:
        experiments = self.data_loader.get_experiments_by_dataset(dataset)

        all_metrics = {"dataset": dataset, "task_type": task_type, "experiments": {}}

        for experiment_name in experiments:
            try:
                metrics = self.analyze_experiment(dataset, experiment_name, task_type)
                all_metrics["experiments"][experiment_name] = metrics
            except Exception as e:
                print(f"Error analyzing experiment {experiment_name}: {e}")
                all_metrics["experiments"][experiment_name] = {"error": str(e)}

        return all_metrics

    def save_results(self, metrics: Dict[str, Any], output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def compare_experiments(
        self, dataset: str, task_type: str = "math"
    ) -> Dict[str, Any]:
        all_metrics = self.analyze_all_experiments(dataset, task_type)

        comparison = {"dataset": dataset, "task_type": task_type, "architectures": {}}

        for exp_name, metrics in all_metrics["experiments"].items():
            if "error" in metrics:
                continue

            arch = metrics["agent_architecture"]
            if arch not in comparison["architectures"]:
                comparison["architectures"][arch] = []

            comparison["architectures"][arch].append(
                {
                    "experiment_name": exp_name,
                    "accuracy": metrics["summary"]["accuracy"],
                    "format_compliance_rate": metrics["summary"][
                        "format_compliance_rate"
                    ],
                    "average_time_cost": metrics["summary"]["average_time_cost"],
                    "average_entropy": metrics["summary"]["average_entropy"],
                }
            )

        return comparison

    def save_last_agent_stats_to_csv(
        self, dataset: str, output_path: str, task_type: str = "math"
    ):
        all_metrics = self.analyze_all_experiments(dataset, task_type)

        rows = []
        for exp_name, metrics in all_metrics["experiments"].items():
            if "error" in metrics:
                continue

            last_agent_stats = metrics["summary"]["last_agent_stats"]
            rows.append(
                {
                    "experiment_name": exp_name,
                    "agent_architecture": metrics["agent_architecture"],
                    "num_rounds": metrics["num_rounds"],
                    "num_samples": metrics["num_samples"],
                    "count": last_agent_stats["count"],
                    "correct": last_agent_stats["correct"],
                    "accuracy": last_agent_stats["accuracy"],
                    "format_compliance_rate": last_agent_stats["format_compliance_rate"],
                    "average_time": last_agent_stats["average_time"],
                    "average_entropy": last_agent_stats["average_entropy"],
                }
            )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "experiment_name",
            "agent_architecture",
            "num_rounds",
            "num_samples",
            "count",
            "correct",
            "accuracy",
            "format_compliance_rate",
            "average_time",
            "average_entropy",
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
