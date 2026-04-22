"""Entropy analyzer for multi-agent system experiments.

This module provides functionality to analyze entropy statistics
from experiment results, including macro and micro level analysis.
"""

import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer

from .base.analyzer import BaseAnalyzer
from .base.architecture import (
    get_round_number as _arch_get_round_number,
    get_final_result_id_from_entropy,
)
from .utils import save_json
from .metrics_calculator import MetricsCalculator


class EntropyStatistic(BaseAnalyzer):
    """Analyzer for entropy statistics in multi-agent experiments.

    Provides methods to analyze entropy at multiple levels:
    - Experiment level: Overall entropy statistics
    - Round level: Entropy per round
    - Agent level: Entropy per agent type
    - Sample level: Entropy per sample
    - Sequence level: Entropy per execution sequence
    - Token position level: Entropy distribution across token positions
    """

    def __init__(self, base_path: str, data_loader: Optional[Any] = None):
        super().__init__(base_path, data_loader=data_loader)
        self.tokenizer_cache: Dict[str, Any] = {}

    def analyze_all_experiments_entropy(
        self, dataset: str, models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze entropy for all experiments in a dataset.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            models: Optional list of model names to analyze. If not provided, analyze all.

        Returns:
            Dictionary containing entropy analysis results for all experiments,
            organized by model and architecture.
        """
        # Get all experiments grouped by model for the specified dataset
        experiments_by_model = self.data_loader.get_experiments_by_dataset(dataset)

        # Filter models if a list is provided
        if models:
            experiments_by_model = {
                model_name: experiments
                for model_name, experiments in experiments_by_model.items()
                if model_name in models
            }

        # Initialize results dictionary with dataset and model/architecture structures
        all_results = {
            "dataset": dataset,
            "models": {},
            "architectures": defaultdict(list),
        }

        # Iterate through each model and its experiments
        for model_name, experiments in experiments_by_model.items():
            # Initialize model-specific results structure
            all_results["models"][model_name] = {"experiments": {}}
            # Process each experiment for the current model
            for experiment_name in experiments:
                try:
                    # Analyze entropy for the current experiment
                    experiment_results = self.analyze_experiment_entropy(
                        dataset, model_name, experiment_name
                    )
                    # Store experiment results in the results dictionary
                    all_results["models"][model_name]["experiments"][
                        experiment_name
                    ] = experiment_results

                    # Extract architecture type from experiment results
                    arch = experiment_results["agent_architecture"]
                    # Add experiment to architecture grouping
                    all_results["architectures"][arch].append(
                        f"{model_name}/{experiment_name}"
                    )
                except Exception as e:
                    # Handle errors during experiment analysis
                    print(
                        f"Error analyzing experiment {model_name}/{experiment_name}: {e}"
                    )
                    # Store error information in results
                    all_results["models"][model_name]["experiments"][
                        experiment_name
                    ] = {"error": str(e)}

        # Convert defaultdict to regular dict for JSON serialization
        all_results["architectures"] = dict(all_results["architectures"])

        return all_results

    def analyze_experiment_entropy(
        self, dataset: str, model_name: str, experiment_name: str
    ) -> Dict[str, Any]:
        """Analyze entropy for a single experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing macro and micro level entropy statistics.
        """
        # Load experiment configuration to get architecture and round settings
        config = self.data_loader.load_experiment_config(
            dataset, experiment_name, model_name
        )
        # Extract agent architecture type from configuration
        agent_architecture = config.get("agent_type", "unknown")
        # Extract number of rounds from configuration
        num_rounds = config.get("round", 1)
        # Extract language model name
        lm_name = config.get("lm_name")
        # Extract task type from configuration, default to "math"
        task_type = config.get("task_type", "math")

        # Load result store information to get block structure
        info = self.data_loader.load_result_store_info(
            dataset, model_name, experiment_name
        )

        # Load all results to get responses
        all_results = self.data_loader.load_all_results(
            dataset, model_name, experiment_name
        )

        # Collect entropy tensors from all results in the experiment
        entropy_data = self._collect_entropy_data(
            dataset, model_name, experiment_name, info
        )

        # Calculate macro-level statistics (experiment, round, agent level)
        macro_stats = self._calculate_macro_statistics(
            entropy_data, agent_architecture, num_rounds
        )
        # Calculate micro-level statistics (sample, sequence, token position level)
        micro_stats = self._calculate_micro_statistics(
            entropy_data,
            agent_architecture,
            num_rounds,
            lm_name,
            all_results,
            task_type,
        )

        # Compile results with metadata and statistics
        results = {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "model_name": model_name,
            "agent_architecture": agent_architecture,
            "num_rounds": num_rounds,
            "num_inferences": len(
                entropy_data
            ),  # Number of unique sequences (main_id-agent_type-execution_order combinations)
            "macro_statistics": macro_stats,
            "micro_statistics": micro_stats,
        }

        return results

    def _collect_entropy_data(
        self, dataset: str, model_name: str, experiment_name: str, info: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect entropy tensors for all results in an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.
            info: Result store information.

        Returns:
            Dictionary mapping sequence IDs to entropy data.
        """
        # Initialize dictionary to store entropy data by sequence ID
        entropy_data = defaultdict(list)

        # Iterate through all result blocks
        for block_name, block_info in info.items():
            # Process each result ID in the block
            for result_id in block_info["ids"]:
                # Parse result ID to extract components
                parsed = self.data_loader.parse_result_id(result_id)
                main_id = parsed["main_id"]
                agent_type = parsed["agent_type"]
                execution_order = parsed["execution_order"]
                sample_number = parsed["sample_number"]

                # Create unique sequence ID for grouping
                sequence_id = f"{main_id}-{agent_type}-{execution_order}"

                # Load entropy tensor for the current result
                entropy_tensor = self.data_loader.load_entropy_tensor(
                    dataset, model_name, experiment_name, result_id
                )

                # Load step-level entropy tensors (for finagent ReAct steps)
                step_entropy_tensors = self.data_loader.load_step_entropy_tensors(
                    dataset, model_name, experiment_name, result_id
                )

                # Store entropy data if tensor exists
                if entropy_tensor is not None:
                    entropy_data[sequence_id].append(
                        {
                            "result_id": result_id,
                            "agent_type": agent_type,
                            "execution_order": execution_order,
                            "sample_number": sample_number,
                            "entropy_tensor": entropy_tensor,
                            "step_entropy_tensors": step_entropy_tensors,
                        }
                    )

        # Convert defaultdict to regular dict for JSON serialization
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
        # Initialize macro statistics structure
        macro_stats = {
            "experiment_level": {},
            "round_level": defaultdict(
                lambda: {"total_entropy": 0.0, "num_inferences": 0}
            ),
        }

        # Initialize accumulators for experiment-level statistics
        total_experiment_entropy = 0.0
        total_count = 0

        # Iterate through all sequences and their entropy data
        for main_id, sample_entropies in entropy_data.items():
            for entropy_info in sample_entropies:
                # Get entropy tensor for current inference
                entropy_tensor = entropy_info["entropy_tensor"]
                # Calculate total entropy sum for this inference
                entropy_sum = float(entropy_tensor.sum().item())

                # Accumulate experiment-level entropy
                total_experiment_entropy += entropy_sum
                total_count += 1

                # Determine round number for this inference
                round_num = self._get_round_number(
                    entropy_info, agent_architecture, num_rounds
                )
                # Accumulate round-level entropy statistics
                macro_stats["round_level"][round_num]["total_entropy"] += entropy_sum
                macro_stats["round_level"][round_num]["num_inferences"] += 1

                # Extract agent type for potential agent-level analysis
                agent_type = entropy_info["agent_type"]

                # Convert tensor to numpy array for processing
                if isinstance(entropy_tensor, torch.Tensor):
                    entropy_array = entropy_tensor.cpu().numpy()
                else:
                    entropy_array = np.array(entropy_tensor)

        # Calculate experiment-level average entropy
        macro_stats["experiment_level"]["total_entropy"] = total_experiment_entropy
        macro_stats["experiment_level"]["infer_average_entropy"] = (
            total_experiment_entropy / total_count if total_count > 0 else 0.0
        )

        # Calculate round-level average entropy
        for round_num, round_data in macro_stats["round_level"].items():
            if round_data["num_inferences"] > 0:
                round_data["infer_average_entropy"] = (
                    round_data["total_entropy"] / round_data["num_inferences"]
                )

        # Convert defaultdict to regular dict for JSON serialization
        macro_stats["round_level"] = dict(macro_stats["round_level"])

        return macro_stats

    def _calculate_micro_statistics(
        self,
        entropy_data: Dict[str, List[Dict[str, Any]]],
        agent_architecture: str,
        num_rounds: int,
        lm_name: Optional[str] = None,
        all_results: Optional[Dict[str, Any]] = None,
        task_type: str = "math",
    ) -> Dict[str, Any]:
        """Calculate micro-level entropy statistics.

        Args:
            entropy_data: Dictionary of entropy data by sequence.
            agent_architecture: Type of agent architecture.
            num_rounds: Total number of rounds.
            lm_name: Language model name.
            all_results: Dictionary containing all experiment results.
            task_type: Type of task (e.g., "math", "code", "option").

        Returns:
            Dictionary containing sample, sequence, and token position level statistics.
        """
        # Initialize micro statistics structure for samples and token positions
        micro_stats = {
            "samples": defaultdict(
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
                    "all_agents_token_count": 0,
                    "num_agents": 0,
                    "agents": {},
                }
            ),
            "token_position_level": defaultdict(list),
        }

        # Group all agents by main_id to find the final agent later
        agents_by_main_id = defaultdict(list)

        # Iterate through all sequences and their entropy data
        for sequence_id, sample_entropies in entropy_data.items():
            # sequence_id = "{main_id}-{agent_type}-{order}"; main_id may be a UUID
            # containing hyphens, so strip only the trailing "-<agent_type>-<order>" part.
            m_seq = re.match(r"^(.+)-([^-]+)-(\d+)$", sequence_id)
            main_id = m_seq.group(1) if m_seq else sequence_id.split("-")[0]
            for entropy_info in sample_entropies:
                agents_by_main_id[main_id].append(entropy_info)

            # Initialize sequence-level accumulators
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

            # Initialize sequence metadata
            sequence_agent_type = None
            sequence_execution_order = None

            # Get tokenizer once for the experiment
            tokenizer = self._get_tokenizer(lm_name) if lm_name else None

            # Process each entropy sample in the sequence
            for entropy_info in sample_entropies:
                # Capture agent type and execution order from first sample
                if sequence_agent_type is None:
                    sequence_agent_type = entropy_info["agent_type"]
                    sequence_execution_order = entropy_info["execution_order"]

                # Get entropy tensor for current sample
                entropy_tensor = entropy_info["entropy_tensor"]
                result_id = entropy_info["result_id"]

                # Calculate predicted_answer_entropy for this specific agent inference
                predicted_answer_entropy = None
                if (
                    tokenizer
                    and all_results
                    and result_id in all_results
                    and task_type != "finance"
                ):
                    response = all_results[result_id].get("response", "")
                    # Extract answer based on task type
                    if task_type in ["math", "option"]:
                        answer, ok = MetricsCalculator.extract_boxed_answer(response)
                    elif task_type == "code":
                        answer, ok = MetricsCalculator.extract_code_answer(response)
                    else:
                        answer, ok = MetricsCalculator.extract_boxed_answer(response)

                    if ok and answer:
                        predicted_answer_entropy = self._get_answer_token_entropy(
                            response, answer, tokenizer, result_id, [entropy_info]
                        )

                # Store it back in entropy_info to use it when building agents dict later
                entropy_info["predicted_answer_entropy"] = predicted_answer_entropy

                # Convert tensor to numpy array for statistical calculations
                if isinstance(entropy_tensor, torch.Tensor):
                    entropy_array = entropy_tensor.cpu().numpy()
                else:
                    entropy_array = np.array(entropy_tensor)

                # Calculate statistical measures for the entropy array
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

                # Accumulate sequence-level statistics
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

                # Collect entropy values by token position for position-level analysis
                for pos, entropy_val in enumerate(entropy_array):
                    micro_stats["token_position_level"][pos].append(float(entropy_val))

            # Extract main_id from sequence_id (main_id may be a UUID with hyphens)
            m_seq2 = re.match(r"^(.+)-([^-]+)-(\d+)$", sequence_id)
            main_id = m_seq2.group(1) if m_seq2 else sequence_id.split("-")[0]
            # Initialize sample statistics if not already present
            if main_id not in micro_stats["samples"]:
                micro_stats["samples"][main_id] = {
                    "total_entropy": 0.0,
                    "max_entropy": 0.0,
                    "mean_entropy": 0.0,
                    "variance_entropy": 0.0,
                    "median_entropy": 0.0,
                    "q1_entropy": 0.0,
                    "q3_entropy": 0.0,
                    "std_entropy": 0.0,
                    "min_entropy": 0.0,
                    "all_agents_token_count": 0,
                    "num_agents": 0,
                    "agents": {},
                }

            # Get reference to sample statistics
            stats = micro_stats["samples"][main_id]
            # Accumulate sequence statistics into sample statistics
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
            stats["all_agents_token_count"] += sequence_token_count
            stats["num_agents"] += 1

            # Create agent-specific statistics if samples exist
            if sequence_sample_count > 0:
                # Determine round number for this sequence
                round_number = self._get_round_number(
                    {
                        "agent_type": sequence_agent_type,
                        "execution_order": sequence_execution_order,
                    },
                    agent_architecture,
                    num_rounds,
                )
                # Create agent key for this round
                agent_key = f"{sequence_agent_type}_round_{round_number}"
                # Store agent-specific statistics
                stats["agents"][agent_key] = {
                    "agent_type": sequence_agent_type,
                    "execution_order": sequence_execution_order,
                    "round_number": round_number,
                    "predicted_answer_entropy": (
                        sample_entropies[0].get("predicted_answer_entropy")
                        if sequence_sample_count > 0
                        else None
                    ),
                    "total_entropy": sequence_total_entropy,
                    "max_entropy": sequence_max_entropy / sequence_sample_count,
                    "mean_entropy": sequence_mean_entropy / sequence_sample_count,
                    "variance_entropy": sequence_variance_entropy
                    / sequence_sample_count,
                    "median_entropy": sequence_median_entropy / sequence_sample_count,
                    "q1_entropy": sequence_q1_entropy / sequence_sample_count,
                    "q3_entropy": sequence_q3_entropy / sequence_sample_count,
                    "std_entropy": sequence_std_entropy / sequence_sample_count,
                    "min_entropy": sequence_min_entropy / sequence_sample_count,
                    "token_count": sequence_token_count,
                    "average_entropy_per_token": (
                        sequence_total_entropy / sequence_token_count
                        if sequence_token_count > 0
                        else 0.0
                    ),
                }

                # Calculate step-level entropy statistics (for finagent ReAct steps)
                step_entropy_list = []
                # Collect step tensors from all samples in this sequence
                all_step_tensors = []
                for ent_info in sample_entropies:
                    st = ent_info.get("step_entropy_tensors", [])
                    if st:
                        all_step_tensors = (
                            st  # Use the first non-empty one (samples share same steps)
                        )
                        break

                for step_idx, step_tensor in all_step_tensors:
                    if isinstance(step_tensor, torch.Tensor):
                        step_array = step_tensor.cpu().numpy()
                    else:
                        step_array = np.array(step_tensor)

                    step_entropy_list.append(
                        {
                            "step_index": step_idx,
                            "mean_entropy": float(np.mean(step_array)),
                            "std_entropy": float(np.std(step_array)),
                            "max_entropy": float(np.max(step_array)),
                            "min_entropy": float(np.min(step_array)),
                            "median_entropy": float(np.median(step_array)),
                            "total_entropy": float(np.sum(step_array)),
                            "token_count": len(step_array),
                        }
                    )

                if step_entropy_list:
                    stats["agents"][agent_key]["step_entropy"] = step_entropy_list

                    # Calculate step-level dynamics
                    num_steps = len(step_entropy_list)
                    first_mean = step_entropy_list[0]["mean_entropy"]
                    last_mean = step_entropy_list[-1]["mean_entropy"]
                    entropy_decay_rate = (
                        (first_mean - last_mean) / num_steps if num_steps > 1 else 0.0
                    )

                    dynamics = {
                        "num_steps": num_steps,
                        "entropy_decay_rate": entropy_decay_rate,
                        "first_step_mean_entropy": first_mean,
                        "last_step_mean_entropy": last_mean,
                        "step_mean_entropies": [
                            s["mean_entropy"] for s in step_entropy_list
                        ],
                    }

                    # Tool-call entropy and statistics (for ReAct steps with tool calls)
                    react_steps_raw = None
                    if all_results and result_id in all_results:
                        extras = all_results[result_id].get("extras", {}) or {}
                        react_steps_raw = extras.get("react_steps")

                    if react_steps_raw:
                        # --- agent-level tool-call statistics ---
                        tool_stats = self._parse_tool_call_result_stats(react_steps_raw)
                        dynamics.update(tool_stats)

                        # --- per-step and aggregate tool-call entropy ---
                        all_tool_entropy_values = []
                        step_tensor_map = {idx: t for idx, t in all_step_tensors}
                        for step_data in react_steps_raw:
                            step_idx = step_data.get("step_index", 0)
                            if step_data.get("tool_calls") is None:
                                continue
                            step_tensor = step_tensor_map.get(step_idx)
                            if step_tensor is None:
                                continue
                            response_text = step_data.get("response", "")
                            tc_entropy = self._extract_tool_call_token_entropy(
                                response_text, step_tensor, tokenizer
                            )
                            if tc_entropy is not None:
                                dynamics[f"step_{step_idx}_tool_call_mean_entropy"] = (
                                    tc_entropy["mean"]
                                )
                                dynamics[f"step_{step_idx}_tool_call_token_count"] = (
                                    tc_entropy["token_count"]
                                )
                                all_tool_entropy_values.extend(tc_entropy["values"])

                        # aggregate across all tool-call steps
                        dynamics["tool_call_steps_count"] = tool_stats.get(
                            "tool_total_calls", 0
                        )
                        if all_tool_entropy_values:
                            arr = np.array(all_tool_entropy_values)
                            dynamics["tool_call_mean_entropy"] = float(np.mean(arr))
                            dynamics["tool_call_max_entropy"] = float(np.max(arr))
                            dynamics["tool_call_min_entropy"] = float(np.min(arr))
                            dynamics["tool_call_std_entropy"] = float(np.std(arr))
                            dynamics["tool_call_median_entropy"] = float(np.median(arr))
                        else:
                            dynamics["tool_call_mean_entropy"] = None
                            dynamics["tool_call_max_entropy"] = None
                            dynamics["tool_call_min_entropy"] = None
                            dynamics["tool_call_std_entropy"] = None
                            dynamics["tool_call_median_entropy"] = None

                    stats["agents"][agent_key]["step_entropy_dynamics"] = dynamics

        # Convert defaultdict to regular dict for JSON serialization
        micro_stats["samples"] = dict(micro_stats["samples"])

        # Normalize sample statistics by number of agents
        for main_id, stats in micro_stats["samples"].items():
            if stats["num_agents"] > 0:
                stats["max_entropy"] = stats["max_entropy"] / stats["num_agents"]
                stats["mean_entropy"] = stats["mean_entropy"] / stats["num_agents"]
                stats["variance_entropy"] = (
                    stats["variance_entropy"] / stats["num_agents"]
                )
                stats["median_entropy"] = stats["median_entropy"] / stats["num_agents"]
                stats["q1_entropy"] = stats["q1_entropy"] / stats["num_agents"]
                stats["q3_entropy"] = stats["q3_entropy"] / stats["num_agents"]
                stats["std_entropy"] = stats["std_entropy"] / stats["num_agents"]
                stats["min_entropy"] = stats["min_entropy"] / stats["num_agents"]
                stats["average_entropy_per_token"] = (
                    stats["total_entropy"] / stats["all_agents_token_count"]
                    if stats["all_agents_token_count"] > 0
                    else 0.0
                )

        # Calculate statistics for each token position
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

        # Convert defaultdict to regular dict for JSON serialization
        micro_stats["token_position_level"] = dict(micro_stats["token_position_level"])

        # Calculate final_predicted_answer_entropy for each sample
        if lm_name and all_results and task_type != "finance":
            tokenizer = self._get_tokenizer(lm_name)
            if tokenizer:
                for main_id, agents_data in agents_by_main_id.items():
                    final_result_id = self._get_final_agent_key(
                        agents_data, agent_architecture
                    )
                    if final_result_id and final_result_id in all_results:
                        response = all_results[final_result_id].get("response", "")
                        # Extract answer based on task type
                        if task_type in ["math", "option"]:
                            # Extract boxed answer for math and option tasks
                            answer, ok = MetricsCalculator.extract_boxed_answer(
                                response
                            )
                        elif task_type == "code":
                            # Extract code answer for code tasks
                            answer, ok = MetricsCalculator.extract_code_answer(response)
                        else:
                            # Default to boxed answer extraction
                            answer, ok = MetricsCalculator.extract_boxed_answer(
                                response
                            )

                        if ok and answer:
                            # Find answer token entropy
                            answer_entropy = self._get_answer_token_entropy(
                                response,
                                answer,
                                tokenizer,
                                final_result_id,
                                agents_data,
                            )
                            if answer_entropy is not None:
                                micro_stats["samples"][main_id][
                                    "final_predicted_answer_entropy"
                                ] = answer_entropy

        return micro_stats

    def _get_tokenizer(self, lm_name: str):
        """Get or load tokenizer for the given model name.

        Args:
            lm_name: Name of the language model.

        Returns:
            Tokenizer object or None if loading fails.
        """
        if not lm_name:
            return None
        if lm_name in self.tokenizer_cache:
            return self.tokenizer_cache[lm_name]

        try:
            # Resolve relative paths (configs often store ../models/ModelName)
            resolved = str(Path(self.base_path) / lm_name) if lm_name.startswith(".") else lm_name
            tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True)
            # Set padding side to 'left' for decoder-only architecture
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "<pad>"})
            # Cache the tokenizer
            self.tokenizer_cache[lm_name] = tokenizer
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer for {lm_name}: {e}")
            return None

    def _get_final_agent_key(
        self, agents_data: List[Dict[str, Any]], agent_architecture: str
    ) -> Optional[str]:
        """Return the result_id of the final agent (delegates to base.architecture)."""
        return get_final_result_id_from_entropy(agents_data, agent_architecture)

    def _extract_tool_call_token_entropy(
        self,
        response: str,
        step_tensor: Any,
        tokenizer: Any,
    ) -> Optional[Dict[str, Any]]:
        """Extract entropy statistics for tokens in the tool-call decision span.

        Identifies the 'Action: ... Action Input: {...}' span in the response and
        returns entropy stats for those tokens from the step entropy tensor.

        Returns None if no tool-call span is found or tokenizer is unavailable.
        """
        if tokenizer is None or not response:
            return None

        # Find 'Action:' start position
        action_match = re.search(r"Action\s*:", response)
        if not action_match:
            return None
        start_char = action_match.start()

        # Find 'Action Input:' and then the end of the JSON block following it
        action_input_match = re.search(r"Action Input\s*:", response[start_char:])
        if not action_input_match:
            return None
        json_start = start_char + action_input_match.end()

        # Use bracket counting to find the closing '}' of the JSON
        depth = 0
        end_char = None
        in_string = False
        escape_next = False
        for i in range(json_start, len(response)):
            ch = response[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_char = i + 1
                    break

        if end_char is None:
            # Fall back to end of line after Action Input
            newline = response.find("\n", json_start)
            end_char = newline if newline != -1 else len(response)

        try:
            encoding = tokenizer(
                response,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = encoding["offset_mapping"]
        except Exception:
            return None

        if isinstance(step_tensor, torch.Tensor):
            entropy_array = step_tensor.cpu().numpy()
        else:
            entropy_array = np.array(step_tensor)

        # Collect token indices that overlap [start_char, end_char)
        token_indices = [
            i
            for i, (tok_s, tok_e) in enumerate(offsets)
            if i < len(entropy_array) and tok_e > start_char and tok_s < end_char
        ]
        if not token_indices:
            return None

        values = [float(entropy_array[i]) for i in token_indices]
        return {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "token_count": len(values),
            "values": values,
        }

    @staticmethod
    def _parse_tool_call_result_stats(react_steps_list: List[Dict]) -> Dict[str, Any]:
        """Parse tool-call statistics from a list of react step dicts.

        A call is 'successful' if tool_result JSON has success=true.
        A call is 'effective' if successful AND the result value contains no error
        indicators (traceback, error, exception, stderr, nameerror, typeerror).
        """
        ERROR_INDICATORS = (
            "traceback",
            "error",
            "exception",
            "stderr",
            "nameerror",
            "typeerror",
        )
        total_calls = 0
        success_count = 0
        effective_count = 0
        tool_names = set()

        for step in react_steps_list:
            tc = step.get("tool_calls")
            if tc is None:
                continue
            total_calls += 1
            tool_name = tc.get("tool_name", "")
            if tool_name:
                tool_names.add(tool_name)

            tool_result_str = tc.get("tool_result", "") or ""
            # Extract JSON payload after "Observation: [tool_name] "
            json_start = tool_result_str.find("{")
            if json_start == -1:
                continue
            try:
                import json as _json

                payload = _json.loads(tool_result_str[json_start:].rstrip())
            except Exception:
                # Try to find the first complete JSON object
                depth = 0
                end = None
                in_str = False
                esc = False
                for i in range(json_start, len(tool_result_str)):
                    c = tool_result_str[i]
                    if esc:
                        esc = False
                        continue
                    if c == "\\":
                        esc = True
                        continue
                    if c == '"' and not esc:
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end is None:
                    continue
                try:
                    import json as _json

                    payload = _json.loads(tool_result_str[json_start:end])
                except Exception:
                    continue

            if not payload.get("success", False):
                continue
            success_count += 1

            result_val = str(payload.get("result", "")).lower()
            if not any(ind in result_val for ind in ERROR_INDICATORS):
                effective_count += 1

        return {
            "tool_total_calls": total_calls,
            "tool_success_count": success_count,
            "tool_effective_count": effective_count,
            "tool_success_rate": (
                success_count / total_calls if total_calls > 0 else 0.0
            ),
            "tool_effective_rate": (
                effective_count / total_calls if total_calls > 0 else 0.0
            ),
            "tool_unique_tools_count": len(tool_names),
        }

    def _get_answer_token_entropy(
        self,
        response: str,
        answer: str,
        tokenizer: Any,
        result_id: str,
        agents_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find the entropy statistics of tokens containing the final answer.

        Args:
            response: Response text from the agent.
            answer: Extracted final answer.
            tokenizer: Tokenizer to use.
            result_id: Result ID of the final agent.
            agents_data: List of entropy data for all agents in a sample.

        Returns:
            Dictionary containing answer token entropy statistics:
            - answer_token_count: Number of tokens in the answer
            - max_answer_token_entropy: Maximum entropy value among answer tokens
            - mean_answer_token_entropy: Average entropy value of answer tokens
            - min_answer_token_entropy: Minimum entropy value among answer tokens
            - std_answer_token_entropy: Standard deviation of answer token entropies
            - median_answer_token_entropy: Median entropy value of answer tokens
            - answer_token_entropies: List of entropy values for all answer tokens
            Returns None if answer tokens not found.
        """
        # Find the last \boxed{...} that contains the answer
        # MetricsCalculator.extract_boxed_answer returns the content of the last \boxed match
        matches = list(re.finditer(r"\\boxed\{([^}]*)\}", response))

        if not matches:
            # Fallback to simple find if \boxed pattern not found
            start_char = response.find(answer)
        else:
            # Get the content position of the last \boxed match
            last_match = matches[-1]
            start_char = last_match.start(1)

            # Adjust start_char if MetricsCalculator modified the answer (e.g. nested braces)
            content = last_match.group(1)
            if content.startswith("{") and not answer.startswith("{"):
                start_char += 1
            elif content.startswith("(") and not answer.startswith("("):
                start_char += 1

        if start_char == -1:
            return None

        try:
            # Use tokenizer to get offsets
            # Explicitly set add_special_tokens=False to match sequence tokens
            encoding = tokenizer(
                response, return_offsets_mapping=True, add_special_tokens=False
            )
            offsets = encoding["offset_mapping"]

            # Find the answer end position in the text
            end_char = start_char + len(answer)

            # Find all token indices that overlap with the answer span
            answer_token_indices = []
            for i, (token_start, token_end) in enumerate(offsets):
                # Token overlaps with answer if it starts before answer ends and ends after answer starts
                if token_start < end_char and token_end > start_char:
                    answer_token_indices.append(i)

            if len(answer_token_indices) > 0:
                # Find the entropy tensor for this result_id
                entropy_tensor = None
                for data in agents_data:
                    if data["result_id"] == result_id:
                        entropy_tensor = data["entropy_tensor"]
                        break

                if entropy_tensor is not None:
                    # Extract entropies for all answer tokens within bounds
                    answer_entropies = []
                    for idx in answer_token_indices:
                        if idx < len(entropy_tensor):
                            answer_entropies.append(float(entropy_tensor[idx].item()))

                    if len(answer_entropies) > 0:
                        # Convert to numpy array for statistical calculations
                        answer_entropies_array = np.array(answer_entropies)

                        # Calculate statistics
                        return {
                            "answer_token_count": len(answer_entropies),
                            "max_answer_token_entropy": float(
                                np.max(answer_entropies_array)
                            ),
                            "mean_answer_token_entropy": float(
                                np.mean(answer_entropies_array)
                            ),
                            "min_answer_token_entropy": float(
                                np.min(answer_entropies_array)
                            ),
                            "std_answer_token_entropy": float(
                                np.std(answer_entropies_array)
                            ),
                            "median_answer_token_entropy": float(
                                np.median(answer_entropies_array)
                            ),
                            "answer_token_entropies": answer_entropies,
                        }
        except Exception as e:
            print(f"Error calculating answer token entropy: {e}")

        return None

    def _get_round_number(
        self,
        entropy_info: Dict[str, Any],
        agent_architecture: str,
        num_rounds: int,
    ) -> int:
        """Return the round number for an entropy result (delegates to base.architecture)."""
        return _arch_get_round_number(
            entropy_info["execution_order"],
            entropy_info["agent_type"],
            agent_architecture,
            num_rounds,
        )

    def _analyze_entropy_distribution(
        self, all_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze entropy distribution across architectures.

        Args:
            all_results: Dictionary containing all experiment results.

        Returns:
            Dictionary containing distribution analysis results.
        """
        # Initialize distribution analysis structure
        distribution = {"architecture_comparison": {}}

        # Iterate through all experiments
        for exp_name, results in all_results["experiments"].items():
            # Skip experiments with errors
            if "error" in results:
                continue

            # Extract architecture type from experiment results
            arch = results["agent_architecture"]
            # Initialize architecture comparison structure if not exists
            if arch not in distribution["architecture_comparison"]:
                distribution["architecture_comparison"][arch] = {
                    "all_entropies": [],
                    "agent_entropies": defaultdict(list),
                }

            # Get macro statistics for the experiment
            macro_stats = results["macro_statistics"]

        # Calculate statistical measures for each architecture
        for arch, data in distribution["architecture_comparison"].items():
            arch_comparison = {}
            # Calculate statistics for each agent type within the architecture
            for agent_type, entropies in data["agent_entropies"].items():
                entropies_array = np.array(entropies)
                arch_comparison[agent_type] = {
                    "mean": float(np.mean(entropies_array)),
                    "std": float(np.std(entropies_array)),
                    "min": float(np.min(entropies_array)),
                    "max": float(np.max(entropies_array)),
                }

            # Store architecture comparison results
            distribution["architecture_comparison"][arch] = arch_comparison

        return distribution

    def save_results_json(self, results: Dict[str, Any], output_path: str):
        """Save entropy analysis results to JSON file.

        Args:
            results: Dictionary containing entropy analysis results.
            output_path: Path to output JSON file.
        """
        # Use utility function to save results as JSON
        save_json(results, output_path)

    def analyze_entropy_change_trends(
        self, dataset: str, model_name: str, experiment_name: str
    ) -> Dict[str, Any]:
        """Analyze entropy change trends between agents across rounds.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing entropy change trend analysis results.
        """
        # Load experiment configuration to get architecture and round settings
        config = self.data_loader.load_experiment_config(
            dataset, experiment_name, model_name
        )
        # Extract agent architecture type from configuration
        agent_architecture = config.get("agent_type", "unknown")
        # Extract number of rounds from configuration
        num_rounds = config.get("round", 1)

        # Load result store information to get block structure
        info = self.data_loader.load_result_store_info(
            dataset, model_name, experiment_name
        )
        # Collect entropy tensors from all results in the experiment
        entropy_data = self._collect_entropy_data(
            dataset, model_name, experiment_name, info
        )

        # Initialize trend results structure
        trend_results = {
            "experiment_name": experiment_name,
            "dataset": dataset,
            "model_name": model_name,
            "agent_architecture": agent_architecture,
            "num_rounds": num_rounds,
            "entropy_by_round_agent": {},
            "intra_round_trends": {},
            "inter_round_trends": {},
            "trend_statistics": {},
        }

        # Extract entropy values organized by round and agent
        trend_results["entropy_by_round_agent"] = self._extract_entropy_by_round_agent(
            entropy_data, agent_architecture, num_rounds
        )

        # Calculate entropy trends within each round
        trend_results["intra_round_trends"] = self._calculate_intra_round_trends(
            trend_results["entropy_by_round_agent"]
        )

        # Calculate entropy trends between rounds
        trend_results["inter_round_trends"] = self._calculate_inter_round_trends(
            trend_results["entropy_by_round_agent"]
        )

        # Calculate statistical measures for trends
        trend_results["trend_statistics"] = self._calculate_trend_statistics(
            trend_results["intra_round_trends"],
            trend_results["inter_round_trends"],
        )

        return trend_results

    def _extract_entropy_by_round_agent(
        self,
        entropy_data: Dict[str, List[Dict[str, Any]]],
        agent_architecture: str,
        num_rounds: int,
    ) -> Dict[int, Dict[str, List[float]]]:
        """Extract entropy values organized by round and agent.

        Args:
            entropy_data: Dictionary of entropy data by sequence.
            agent_architecture: Type of agent architecture.
            num_rounds: Number of rounds in the experiment.

        Returns:
            Dictionary mapping round numbers to agent types to entropy values.
        """
        # Initialize nested defaultdict for round-agent entropy organization
        round_agent_entropy = defaultdict(
            lambda: defaultdict(lambda: {"entropies": [], "sums": []})
        )

        # Iterate through all sequences and their entropy data
        for sequence_id, sample_entropies in entropy_data.items():
            # Process each entropy sample in the sequence
            for entropy_info in sample_entropies:
                # Get entropy tensor for current sample
                entropy_tensor = entropy_info["entropy_tensor"]
                # Extract agent type
                agent_type = entropy_info["agent_type"]

                # Convert tensor to numpy array for processing
                if isinstance(entropy_tensor, torch.Tensor):
                    entropy_array = entropy_tensor.cpu().numpy()
                else:
                    entropy_array = np.array(entropy_tensor)

                # Determine round number for this entropy result
                round_num = self._get_round_number(
                    entropy_info, agent_architecture, num_rounds
                )

                # Calculate entropy statistics
                entropy_sum = float(np.sum(entropy_array))
                entropy_mean = float(np.mean(entropy_array))

                # Store entropy values by round and agent
                round_agent_entropy[round_num][agent_type]["entropies"].append(
                    entropy_mean
                )
                round_agent_entropy[round_num][agent_type]["sums"].append(entropy_sum)

        # Compile results with statistical measures
        result = {}
        # Sort rounds for consistent ordering
        for round_num in sorted(round_agent_entropy.keys()):
            result[round_num] = {}
            # Sort agent types for consistent ordering
            for agent_type in sorted(round_agent_entropy[round_num].keys()):
                result[round_num][agent_type] = {
                    "mean_entropy": float(
                        np.mean(round_agent_entropy[round_num][agent_type]["entropies"])
                    ),
                    "std_entropy": float(
                        np.std(round_agent_entropy[round_num][agent_type]["entropies"])
                    ),
                    "median_entropy": float(
                        np.median(
                            round_agent_entropy[round_num][agent_type]["entropies"]
                        )
                    ),
                    "min_entropy": float(
                        np.min(round_agent_entropy[round_num][agent_type]["entropies"])
                    ),
                    "max_entropy": float(
                        np.max(round_agent_entropy[round_num][agent_type]["entropies"])
                    ),
                    "total_entropy": float(
                        np.sum(round_agent_entropy[round_num][agent_type]["sums"])
                    ),
                }

        return result

    def _calculate_intra_round_trends(
        self, entropy_by_round_agent: Dict[int, Dict[str, Dict[str, float]]]
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Calculate entropy change trends between agents within the same round.

        Args:
            entropy_by_round_agent: Dictionary mapping rounds to agents to entropy stats.

        Returns:
            Dictionary containing intra-round trend analysis.
        """
        # Initialize intra-round trends dictionary
        intra_round_trends = {}

        # Iterate through each round and its agent data
        for round_num, agents_data in entropy_by_round_agent.items():
            # Get list of agent types for this round
            agent_types = list(agents_data.keys())

            # Handle case with only one agent
            if len(agent_types) < 2:
                intra_round_trends[round_num] = {
                    "trends": {},
                    "differences": {},
                    "summary": "Only one agent in this round",
                }
                continue

            # Initialize structure for multi-agent round
            intra_round_trends[round_num] = {
                "trends": {},
                "differences": {},
                "summary": "",
            }

            # Initialize dictionaries for trends and differences
            trends = {}
            differences = {}

            # Calculate differences between all pairs of agents
            for i in range(len(agent_types)):
                for j in range(i + 1, len(agent_types)):
                    agent1 = agent_types[i]
                    agent2 = agent_types[j]

                    # Calculate absolute difference in mean entropy
                    diff = (
                        agents_data[agent1]["mean_entropy"]
                        - agents_data[agent2]["mean_entropy"]
                    )
                    # Calculate percentage change relative to agent2
                    pct_change = (
                        (diff / agents_data[agent2]["mean_entropy"]) * 100
                        if agents_data[agent2]["mean_entropy"] != 0
                        else 0.0
                    )

                    # Create pair key for identification
                    pair_key = f"{agent1}_vs_{agent2}"
                    differences[pair_key] = {
                        "absolute_difference": diff,
                        "percentage_difference": pct_change,
                        "agent1_entropy": agents_data[agent1]["mean_entropy"],
                        "agent2_entropy": agents_data[agent2]["mean_entropy"],
                        "agent1_std": agents_data[agent1]["std_entropy"],
                        "agent2_std": agents_data[agent2]["std_entropy"],
                    }

            # Store differences in results
            intra_round_trends[round_num]["differences"] = differences

            # Calculate ranking for rounds with 4 agents
            if len(agent_types) == 4:
                # Sort agents by mean entropy
                sorted_agents = sorted(
                    agent_types, key=lambda x: agents_data[x]["mean_entropy"]
                )
                # Create trend description with ranking
                trend_desc = " -> ".join(
                    [
                        f"{a}({agents_data[a]['mean_entropy']:.4f})"
                        for a in sorted_agents
                    ]
                )
                trends["ranking"] = trend_desc
                # Identify highest and lowest entropy agents
                trends["highest_entropy_agent"] = sorted_agents[-1]
                trends["lowest_entropy_agent"] = sorted_agents[0]
                # Calculate entropy range
                trends["entropy_range"] = (
                    agents_data[sorted_agents[-1]]["mean_entropy"]
                    - agents_data[sorted_agents[0]]["mean_entropy"]
                )

            # Store trends in results
            intra_round_trends[round_num]["trends"] = trends

        return intra_round_trends

    def _calculate_inter_round_trends(
        self, entropy_by_round_agent: Dict[int, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, List[float]]]:
        """Calculate entropy change trends of individual agents across consecutive rounds.

        Args:
            entropy_by_round_agent: Dictionary mapping rounds to agents to entropy stats.

        Returns:
            Dictionary containing inter-round trend analysis.
        """
        # Initialize inter-round trends structure
        inter_round_trends = {
            "agent_trends": {},
            "round_to_round_changes": {},
            "summary": {},
        }

        # Get sorted list of round numbers
        round_numbers = sorted(entropy_by_round_agent.keys())

        # Collect entropy data for each agent across rounds
        for round_num, agents_data in entropy_by_round_agent.items():
            for agent_type, stats in agents_data.items():
                # Initialize agent trend structure if not exists
                if agent_type not in inter_round_trends["agent_trends"]:
                    inter_round_trends["agent_trends"][agent_type] = {
                        "mean_entropies": [],
                        "rounds": [],
                        "changes": [],
                        "percentage_changes": [],
                    }

                # Append mean entropy and round number for this agent
                inter_round_trends["agent_trends"][agent_type]["mean_entropies"].append(
                    stats["mean_entropy"]
                )
                inter_round_trends["agent_trends"][agent_type]["rounds"].append(
                    round_num
                )

        # Calculate changes between consecutive rounds for each agent
        for agent_type, agent_data in inter_round_trends["agent_trends"].items():
            mean_entropies = agent_data["mean_entropies"]
            rounds = agent_data["rounds"]

            # Iterate through consecutive pairs of rounds
            for i in range(1, len(mean_entropies)):
                # Calculate absolute change
                change = mean_entropies[i] - mean_entropies[i - 1]
                # Calculate percentage change
                pct_change = (
                    (change / mean_entropies[i - 1]) * 100
                    if mean_entropies[i - 1] != 0
                    else 0.0
                )

                # Store changes in agent data
                agent_data["changes"].append(change)
                agent_data["percentage_changes"].append(pct_change)

                # Create round pair key
                round_pair = f"{rounds[i - 1]}_to_{rounds[i]}"
                # Initialize round pair structure if not exists
                if round_pair not in inter_round_trends["round_to_round_changes"]:
                    inter_round_trends["round_to_round_changes"][round_pair] = {}

                # Store round-to-round change data
                inter_round_trends["round_to_round_changes"][round_pair][agent_type] = {
                    "change": change,
                    "percentage_change": pct_change,
                    "from_entropy": mean_entropies[i - 1],
                    "to_entropy": mean_entropies[i],
                }

        # Calculate summary statistics for each agent
        for agent_type, agent_data in inter_round_trends["agent_trends"].items():
            if len(agent_data["changes"]) > 0:
                inter_round_trends["summary"][agent_type] = {
                    "total_change": (
                        agent_data["changes"][-1]
                        if len(agent_data["changes"]) > 0
                        else 0.0
                    ),
                    "average_change": float(np.mean(agent_data["changes"])),
                    "total_percentage_change": (
                        agent_data["percentage_changes"][-1]
                        if len(agent_data["percentage_changes"]) > 0
                        else 0.0
                    ),
                    "average_percentage_change": float(
                        np.mean(agent_data["percentage_changes"])
                    ),
                    # Determine trend direction based on average change
                    "trend_direction": (
                        "increasing"
                        if np.mean(agent_data["changes"]) > 0
                        else "decreasing"
                    ),
                    # Calculate volatility as standard deviation of changes
                    "volatility": float(np.std(agent_data["changes"])),
                }

        return inter_round_trends

    def _calculate_trend_statistics(
        self,
        intra_round_trends: Dict[int, Dict[str, Dict[str, float]]],
        inter_round_trends: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, Any]:
        """Calculate overall trend statistics.

        Args:
            intra_round_trends: Intra-round trend analysis results.
            inter_round_trends: Inter-round trend analysis results.

        Returns:
            Dictionary containing overall trend statistics.
        """
        # Initialize statistics structure
        statistics = {
            "intra_round_stats": {},
            "inter_round_stats": {},
            "overall_summary": {},
        }

        # Collect all absolute differences from intra-round trends
        all_differences = []
        for round_num, round_data in intra_round_trends.items():
            if "differences" in round_data:
                for diff_key, diff_data in round_data["differences"].items():
                    all_differences.append(abs(diff_data["absolute_difference"]))

        # Calculate intra-round statistics if differences exist
        if all_differences:
            statistics["intra_round_stats"] = {
                "mean_agent_difference": float(np.mean(all_differences)),
                "max_agent_difference": float(np.max(all_differences)),
                "min_agent_difference": float(np.min(all_differences)),
                "std_agent_difference": float(np.std(all_differences)),
            }

        # Collect all agent changes from inter-round trends
        all_agent_changes = []
        for agent_type, agent_data in inter_round_trends["agent_trends"].items():
            all_agent_changes.extend(agent_data["changes"])

        # Calculate inter-round statistics if changes exist
        if all_agent_changes:
            statistics["inter_round_stats"] = {
                "mean_round_to_round_change": float(np.mean(all_agent_changes)),
                "max_round_to_round_change": float(np.max(all_agent_changes)),
                "min_round_to_round_change": float(np.min(all_agent_changes)),
                "std_round_to_round_change": float(np.std(all_agent_changes)),
            }

        # Calculate overall summary if summary data exists
        if "summary" in inter_round_trends:
            # Count agents with increasing trends
            num_increasing = sum(
                1
                for summary in inter_round_trends["summary"].values()
                if summary["trend_direction"] == "increasing"
            )
            # Count agents with decreasing trends
            num_decreasing = len(inter_round_trends["summary"]) - num_increasing

            # Compile overall summary statistics
            statistics["overall_summary"] = {
                "total_agents_analyzed": len(inter_round_trends["summary"]),
                "agents_with_increasing_trend": num_increasing,
                "agents_with_decreasing_trend": num_decreasing,
                # Determine dominant trend direction
                "dominant_trend": (
                    "increasing" if num_increasing > num_decreasing else "decreasing"
                ),
            }

        return statistics
