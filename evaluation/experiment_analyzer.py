"""Experiment analyzer for multi-agent system evaluation.

This module provides functionality to analyze experiment results,
calculate metrics, and compare different agent architectures.
"""

from collections import defaultdict
from typing import Dict, List, Any, Optional

from data_loader import DataLoader
from utils import save_json
from metrics_calculator import MetricsCalculator


class ExperimentAnalyzer:
    """Analyzer for multi-agent experiment results.

    Provides methods to analyze individual experiments and compare
    architectures.
    """

    def __init__(self, base_path: str):
        """Initialize the experiment analyzer.

        Args:
            base_path: Base path to the project directory.
        """
        self.data_loader = DataLoader(base_path)
        self.metrics_calculator = MetricsCalculator()

    def analyze_experiment(
        self, dataset: str, experiment_name: str, task_type: str = "math"
    ) -> Dict[str, Any]:
        """Analyze a single experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment to analyze.
            task_type: Type of task (e.g., "math", "code", "option").

        Returns:
            Dictionary containing experiment metrics and analysis results.
        """
        config = self.data_loader.load_experiment_config(dataset, experiment_name)
        agent_architecture = config["agent_type"]
        num_rounds = config["round"]

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
        """Analyze metrics for a single sample.

        Args:
            main_id: Main sample identifier.
            sample_results: List of results for this sample.
            ground_truth: Ground truth data for this sample.
            agent_architecture: Type of agent architecture.
            dataset: Dataset name.
            experiment_name: Experiment name.

        Returns:
            Dictionary containing sample-level metrics.
        """
        sample_metrics = {
            "main_id": main_id,
            "ground_truth": ground_truth["groundtruth"] if ground_truth else None,
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
            sample_metrics["agents"] = sample_metrics.get("agents", {})
            sample_metrics["agents"][agent_key] = {
                "agent_type": agent_type,
                "execution_order": execution_order,
                "time_cost": time_cost,
                "average_entropy": avg_entropy,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
            }

        final_agent_key = self._get_final_agent_key(
            sample_metrics["agents"], agent_architecture
        )
        if final_agent_key and final_agent_key in sample_metrics["agents"]:
            final_agent_data = sample_metrics["agents"][final_agent_key]
            sample_metrics["final_predicted_answer"] = final_agent_data[
                "predicted_answer"
            ]
            sample_metrics["is_finally_correct"] = final_agent_data["is_correct"]
        else:
            sample_metrics["final_predicted_answer"] = None
            sample_metrics["is_finally_correct"] = False

        sample_metrics["agents"] = sample_metrics.pop("agents")

        return sample_metrics

    def _get_final_agent_keys(self, agent_architecture: str) -> List[str]:
        """Get the keys for the final agent in each architecture.

        Args:
            agent_architecture: Type of agent architecture.

        Returns:
            List of agent keys for the final agent.
        """
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

    def _get_final_agent_key(
        self, agents: Dict[str, Any], agent_architecture: str
    ) -> Optional[str]:
        """Get the key of the final agent for a given sample.

        Args:
            agents: Dictionary of agent data for this sample.
            agent_architecture: Type of agent architecture.

        Returns:
            The key of the final agent, or None if not found.
        """
        if not agents:
            return None

        if agent_architecture == "single":
            final_agent_type = "SingleSolver"
        elif agent_architecture == "sequential":
            final_agent_type = "judger"
        elif agent_architecture == "centralized":
            final_agent_type = "OrchestratorAgent"
        elif agent_architecture == "debate":
            final_agent_type = "orchestrator"
        elif agent_architecture == "hybrid":
            final_agent_type = "OrchestratorAgent"
        else:
            return None

        if agent_architecture == "debate":
            return "orchestrator"

        max_execution_order = -1
        final_agent_key = None

        for agent_key, agent_data in agents.items():
            if agent_data["agent_type"] == final_agent_type:
                execution_order = agent_data["execution_order"]
                if execution_order > max_execution_order:
                    max_execution_order = execution_order
                    final_agent_key = agent_key

        return final_agent_key

    def analyze_all_experiments(
        self, dataset: str, task_type: str = "math"
    ) -> Dict[str, Any]:
        """Analyze all experiments for a given dataset.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            task_type: Type of task (e.g., "math", "code", "option").

        Returns:
            Dictionary containing metrics for all experiments.
        """
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
        """Save analysis results to JSON file.

        Args:
            metrics: Dictionary containing analysis metrics.
            output_path: Path to save the results.
        """
        metrics_copy = metrics.copy()
        
        if "experiments" in metrics_copy:
            for exp_name, exp_metrics in metrics_copy["experiments"].items():
                if "samples" in exp_metrics:
                    for sample_id, sample_data in exp_metrics["samples"].items():
                        if "agents" in sample_data:
                            for agent_key in sample_data["agents"]:
                                if "predicted_answer" in sample_data["agents"][agent_key]:
                                    del sample_data["agents"][agent_key]["predicted_answer"]
                                if "is_correct" in sample_data["agents"][agent_key]:
                                    del sample_data["agents"][agent_key]["is_correct"]
        
        save_json(metrics_copy, output_path)
