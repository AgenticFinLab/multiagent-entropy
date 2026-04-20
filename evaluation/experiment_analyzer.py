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
        # Initialize data loader for reading experiment data
        self.data_loader = DataLoader(base_path)
        # Initialize metrics calculator for computing performance metrics
        self.metrics_calculator = MetricsCalculator()

    def _get_task_type_from_dataset(self, dataset: str) -> str:
        """Determine task type from dataset name.

        Args:
            dataset: Dataset name.

        Returns:
            Task type ("math", "code", or "option").
        """
        # Map dataset names to their corresponding task types
        dataset_task_map = {
            "humaneval": "code",
            "mmlu": "option",
            "gsm8k": "math",
            "aime2024_16384": "math",
            "aime2025_16384": "math",
            "math500": "math",
            "aime2024_8192": "math",
            "aime2025_8192": "math",
            "finagent": "finance",
        }
        # Return task type or default to "math" if dataset not found
        return dataset_task_map.get(dataset.lower(), "math")

    def analyze_experiment(
        self,
        dataset: str,
        model_name: str,
        experiment_name: str,
        task_type: str = "math",
        timeout: int = 10,
    ) -> Dict[str, Any]:
        """Analyze a single experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment to analyze.
            task_type: Type of task (e.g., "math", "code", "option").
                       If not provided or "auto", will be inferred from dataset.
            timeout: Maximum time in seconds to execute code for code tasks.

        Returns:
            Dictionary containing experiment metrics and analysis results.
        """
        # Infer task type from dataset if not provided or set to auto/math
        if task_type == "auto" or task_type == "math":
            task_type = self._get_task_type_from_dataset(dataset)

        # Load experiment configuration
        try:
            config = self.data_loader.load_experiment_config(
                dataset, experiment_name, model_name
            )
        except Exception as e:
            raise Exception(
                f"Failed to load config for {dataset}/{model_name}/{experiment_name}: {e}"
            )

        # Validate configuration format
        if not isinstance(config, dict):
            raise Exception(f"Config is not a dictionary, got {type(config)}: {config}")

        # Extract agent architecture from configuration
        agent_architecture = config.get("agent_type")
        if agent_architecture is None:
            raise Exception(
                f"Config missing 'agent_type' key. Available keys: {list(config.keys())}"
            )

        # Extract number of rounds from configuration
        num_rounds = config.get("round", 1)

        # Load finagent pre-computed evaluation results if applicable
        finagent_eval_results = None
        if dataset.lower() == "finagent":
            try:
                finagent_eval_results = (
                    self.data_loader.load_finagent_evaluation_results(
                        model_name, experiment_name
                    )
                )
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        # Load ground truth data for the dataset (skip for finagent)
        if dataset.lower() == "finagent":
            ground_truths = {}
        else:
            ground_truths = self.data_loader.load_ground_truth(dataset)
        # Load all results for the experiment
        all_results = self.data_loader.load_all_results(
            dataset, model_name, experiment_name
        )

        # Group results by sample ID for analysis
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

        # Initialize metrics dictionary with experiment metadata
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

        # Analyze each sample individually
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
                finagent_eval_results=finagent_eval_results,
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
        timeout: int = 10,
        finagent_eval_results: Optional[Dict] = None,
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
            timeout: Maximum time in seconds to execute code for code tasks.

        Returns:
            Dictionary containing sample-level metrics.
        """
        # Initialize sample metrics with ground truth
        sample_metrics = {
            "main_id": main_id,
            "ground_truth": ground_truth["groundtruth"] if ground_truth else None,
        }

        # For finagent, set ground truth and question type from eval results
        if (
            task_type == "finance"
            and finagent_eval_results
            and main_id in finagent_eval_results
        ):
            sample_metrics["ground_truth"] = finagent_eval_results[main_id].get(
                "expected_answer", ""
            )
            sample_metrics["question_type"] = finagent_eval_results[main_id].get(
                "question_type", ""
            )

        # Process each agent result for this sample
        for result_info in sample_results:
            result_id = result_info["result_id"]
            agent_type = result_info["agent_type"]
            execution_order = result_info["execution_order"]
            result_data = result_info["data"]

            # Calculate agent time cost
            agent_time_cost = self.metrics_calculator.calculate_agent_time_cost(
                result_data
            )

            # Load entropy tensor and calculate average entropy
            entropy_tensor = self.data_loader.load_entropy_tensor(
                dataset, model_name, experiment_name, result_id
            )
            avg_entropy = self.metrics_calculator.calculate_average_entropy(
                entropy_tensor
            )

            # Extract predicted answer based on task type
            if task_type == "finance":
                # For finagent, use pre-computed evaluation results
                response = result_data.get("response", "")
                predicted_answer = response[:200] if response else ""
                format_compliance = True
                is_correct = False
                evaluation_score = 0.0
                if finagent_eval_results and main_id in finagent_eval_results:
                    eval_data = finagent_eval_results[main_id]
                    is_correct = eval_data.get("evaluation_result", False)
                    evaluation_score = eval_data.get("evaluation_score", 0.0)
            elif task_type == "code":
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
                # Determine if answer is correct
                is_correct = False
                if ground_truth and predicted_answer and format_compliance:
                    test_cases = (
                        ground_truth.get("test_cases") if ground_truth else None
                    )
                    is_correct = self.metrics_calculator.is_answer_correct_by_task_type(
                        predicted_answer,
                        ground_truth["groundtruth"],
                        task_type,
                        test_cases,
                        timeout,
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
                # Determine if answer is correct
                is_correct = False
                if ground_truth and predicted_answer and format_compliance:
                    test_cases = (
                        ground_truth.get("test_cases") if ground_truth else None
                    )
                    is_correct = self.metrics_calculator.is_answer_correct_by_task_type(
                        predicted_answer,
                        ground_truth["groundtruth"],
                        task_type,
                        test_cases,
                        timeout,
                    )

            # Generate agent key based on architecture and execution order
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

            # Store agent metrics
            sample_metrics["agents"] = sample_metrics.get("agents", {})
            agent_metrics = {
                "agent_type": agent_type,
                "execution_order": execution_order,
                "agent_time_cost": agent_time_cost,
                "average_entropy": avg_entropy,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response,
                "format_compliance": format_compliance,
            }
            if task_type == "finance":
                agent_metrics["evaluation_score"] = evaluation_score
            sample_metrics["agents"][agent_key] = agent_metrics

        # Determine final agent key based on architecture
        final_agent_key = self._get_final_agent_key(
            sample_metrics["agents"], agent_architecture
        )

        # Extract final answer from the last agent
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

        # Move agents dict to top level
        sample_metrics["agents"] = sample_metrics.pop("agents")

        return sample_metrics

    def _get_final_agent_keys(self, agent_architecture: str) -> List[str]:
        """Get the keys for the final agent in each architecture.

        Args:
            agent_architecture: Type of agent architecture.

        Returns:
            List of agent keys for the final agent.
        """
        # Return final agent keys based on architecture type
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
        # Return None if no agents are present
        if not agents:
            return None

        # Determine final agent type based on architecture
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

        # For debate architecture, return orchestrator directly
        if agent_architecture == "debate":
            return "orchestrator"

        # Find the agent with the highest execution order
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
        self,
        dataset: str,
        task_type: str = "math",
        timeout: int = 10,
        models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Analyze all experiments for a given dataset.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            task_type: Type of task (e.g., "math", "code", "option").
                       If not provided or "auto", will be inferred from dataset.
            timeout: Maximum time in seconds to execute code for code tasks.
            models: Optional list of model names to analyze. If not provided, analyze all.

        Returns:
            Dictionary containing metrics for all experiments, grouped by model.
        """
        # Infer task type from dataset if not provided or set to auto/math
        if task_type == "auto" or task_type == "math":
            task_type = self._get_task_type_from_dataset(dataset)

        # Get all experiments grouped by model
        experiments_by_model = self.data_loader.get_experiments_by_dataset(dataset)

        # Filter models if a list is provided
        if models:
            experiments_by_model = {
                model_name: experiments
                for model_name, experiments in experiments_by_model.items()
                if model_name in models
            }

        # Initialize all metrics dictionary
        all_metrics = {"dataset": dataset, "task_type": task_type, "models": {}}

        # Analyze each experiment for each model
        for model_name, experiments in experiments_by_model.items():
            all_metrics["models"][model_name] = {"experiments": {}}
            for experiment_name in experiments:
                try:
                    metrics = self.analyze_experiment(
                        dataset, model_name, experiment_name, task_type, timeout
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
        # Create a copy of metrics to avoid modifying the original
        metrics_copy = metrics.copy()

        # Remove response fields to reduce file size
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

        # Save metrics to JSON file
        save_json(metrics_copy, output_path)
