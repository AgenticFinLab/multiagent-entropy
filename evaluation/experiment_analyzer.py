"""Experiment analyzer for multi-agent system evaluation.

This module provides functionality to analyze experiment results,
calculate metrics, and compare different agent architectures.
"""

from collections import defaultdict
from typing import Dict, List, Any, Optional

from utils import save_json
from data_loader import DataLoader
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

    def _get_task_type_from_dataset(self, dataset: str) -> str:
        """Determine task type from dataset name.

        Args:
            dataset: Dataset name.

        Returns:
            Task type ("math", "code", or "option").
        """
        dataset_task_map = {
            "humaneval": "code",
            "mmlu": "option",
            "gsm8k": "math",
            "aime2024": "math",
            "aime2025": "math",
            "math500": "math",
        }
        return dataset_task_map.get(dataset.lower(), "math")

    def analyze_experiment(
        self,
        dataset: str,
        model_name: str,
        experiment_name: str,
        task_type: str = "math",
    ) -> Dict[str, Any]:
        """Analyze a single experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment to analyze.
            task_type: Type of task (e.g., "math", "code", "option").
                       If not provided or "auto", will be inferred from dataset.

        Returns:
            Dictionary containing experiment metrics and analysis results.
        """
        if task_type == "auto" or task_type == "math":
            task_type = self._get_task_type_from_dataset(dataset)

        try:
            config = self.data_loader.load_experiment_config(
                dataset, experiment_name, model_name
            )
        except Exception as e:
            raise Exception(
                f"Failed to load config for {dataset}/{model_name}/{experiment_name}: {e}"
            )

        if not isinstance(config, dict):
            raise Exception(f"Config is not a dictionary, got {type(config)}: {config}")

        agent_architecture = config.get("agent_type")
        if agent_architecture is None:
            raise Exception(
                f"Config missing 'agent_type' key. Available keys: {list(config.keys())}"
            )

        num_rounds = config.get("round", 1)

        ground_truths = self.data_loader.load_ground_truth(dataset)
        all_results = self.data_loader.load_all_results(
            dataset, model_name, experiment_name
        )

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
            "model_name": model_name,
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
                model_name,
                experiment_name,
                task_type,
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
        model_name: str,
        experiment_name: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Analyze metrics for a single sample.

        Args:
            main_id: Main sample identifier.
            sample_results: List of results for this sample.
            ground_truth: Ground truth data for this sample.
            agent_architecture: Type of agent architecture.
            dataset: Dataset name.
            model_name: Model name.
            experiment_name: Experiment name.
            task_type: Type of task (e.g., "math", "code", "option").

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

            agent_time_cost = self.metrics_calculator.calculate_agent_time_cost(
                result_data
            )

            entropy_tensor = self.data_loader.load_entropy_tensor(
                dataset, model_name, experiment_name, result_id
            )
            avg_entropy = self.metrics_calculator.calculate_average_entropy(
                entropy_tensor
            )

            if task_type == "code":
                if "final_answer" in result_data:
                    response = result_data["final_answer"]
                    response_formatted = "```python\n" + response + "\n```"
                    predicted_answer, format_compliance = (
                        self.metrics_calculator.extract_code_answer(response_formatted)
                    )
                else:
                    response = result_data.get("response", "")
                    predicted_answer, format_compliance = (
                        self.metrics_calculator.extract_code_answer(response)
                    )
            else:
                if "final_answer" in result_data:
                    response = result_data["final_answer"]
                    predicted_answer = response
                    response_formatted = "\\boxed{" + response + "}"
                    _, format_compliance = self.metrics_calculator.extract_boxed_answer(
                        response_formatted
                    )
                else:
                    response = result_data.get("response", "")
                    predicted_answer, format_compliance = (
                        self.metrics_calculator.extract_boxed_answer(response)
                    )

            is_correct = False
            if ground_truth and predicted_answer and format_compliance:
                test_cases = ground_truth.get("test_cases") if ground_truth else None
                is_correct = self.metrics_calculator.is_answer_correct_by_task_type(
                    predicted_answer, ground_truth["groundtruth"], task_type, test_cases
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
                "agent_time_cost": agent_time_cost,
                "average_entropy": avg_entropy,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response,
                "format_compliance": format_compliance,
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
            sample_metrics["final_format_compliance"] = final_agent_data[
                "format_compliance"
            ]
        else:
            sample_metrics["final_predicted_answer"] = None
            sample_metrics["is_finally_correct"] = False
            sample_metrics["final_format_compliance"] = False

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
                       If not provided or "auto", will be inferred from dataset.

        Returns:
            Dictionary containing metrics for all experiments, grouped by model.
        """
        if task_type == "auto" or task_type == "math":
            task_type = self._get_task_type_from_dataset(dataset)

        experiments_by_model = self.data_loader.get_experiments_by_dataset(dataset)

        all_metrics = {"dataset": dataset, "task_type": task_type, "models": {}}

        for model_name, experiments in experiments_by_model.items():
            all_metrics["models"][model_name] = {"experiments": {}}
            for experiment_name in experiments:
                try:
                    metrics = self.analyze_experiment(
                        dataset, model_name, experiment_name, task_type
                    )
                    all_metrics["models"][model_name]["experiments"][
                        experiment_name
                    ] = metrics
                except Exception as e:
                    import traceback

                    print(
                        f"Error analyzing experiment {model_name}/{experiment_name}: {e}"
                    )
                    traceback.print_exc()
                    all_metrics["models"][model_name]["experiments"][
                        experiment_name
                    ] = {"error": str(e)}

        return all_metrics

    def save_results(self, metrics: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file.

        Args:
            metrics: Dictionary containing analysis metrics.
            output_path: Path to save the results.
        """
        metrics_copy = metrics.copy()

        if "models" in metrics_copy:
            for model_name, model_data in metrics_copy["models"].items():
                if "experiments" in model_data:
                    for exp_name, exp_metrics in model_data["experiments"].items():
                        if "samples" in exp_metrics:
                            for sample_id, sample_data in exp_metrics[
                                "samples"
                            ].items():
                                if "agents" in sample_data:
                                    for agent_key in sample_data["agents"]:
                                        if (
                                            "response"
                                            in sample_data["agents"][agent_key]
                                        ):
                                            del sample_data["agents"][agent_key][
                                                "response"
                                            ]

        save_json(metrics_copy, output_path)
