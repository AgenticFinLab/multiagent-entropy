"""Aggregator for multi-agent experiment results.

This module provides functionality to aggregate experiment results
from `all_entropy_results.json` and `all_metrics.json` into a unified format csv file which suitable for data mining and analysis.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


class Aggregator:
    """Convert JSON results to CSV format for data mining."""

    def __init__(self, entropy_file: str, metrics_file: str, output_dir: str):
        """Initialize the converter.

        Args:
            entropy_file: Path to all_entropy_results.json
            metrics_file: Path to all_metrics.json
            output_dir: Path to output directory for CSV files
        """
        # Convert file paths to Path objects for consistent path handling
        self.entropy_file = Path(entropy_file)
        self.metrics_file = Path(metrics_file)
        self.output_dir = Path(output_dir)
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_json_files(self) -> tuple:
        """Load both JSON files.

        Returns:
            Tuple of (entropy_data, metrics_data)
        """
        # Load entropy analysis results from JSON file
        with open(self.entropy_file, "r", encoding="utf-8") as f:
            entropy_data = json.load(f)

        # Load performance metrics from JSON file
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        return entropy_data, metrics_data

    def _extract_base_model_data(
        self, metrics_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract base model data from metrics for each model.

        Base model is defined as the first round agent in single agent architecture.
        This data is shared across all architectures for the same dataset and model.

        Args:
            metrics_data: Performance metrics

        Returns:
            Dictionary mapping model_name to sample_id to base model data
        """
        base_model_data = {}

        # Iterate through each model's metrics
        for model_name, model_metrics in metrics_data.get("models", {}).items():
            model_base_data = {}
            # Iterate through each experiment within the model
            for exp_name, exp_metrics in model_metrics.get("experiments", {}).items():
                # Only extract from single agent architecture experiments
                architecture = exp_metrics.get("agent_architecture", "unknown")
                if architecture == "single":
                    samples = exp_metrics.get("samples", {})
                    # Extract first round agent data as base model
                    for sample_id, sample_data in samples.items():
                        for agent_key, agent_data in sample_data.get(
                            "agents", {}
                        ).items():
                            if agent_key.endswith("_round_1"):
                                model_base_data[sample_id] = {
                                    "predicted_answer": agent_data["predicted_answer"],
                                    "is_correct": agent_data["is_correct"],
                                    "format_compliance": agent_data[
                                        "format_compliance"
                                    ],
                                }
                                break
            base_model_data[model_name] = model_base_data

        return base_model_data

    def _calculate_base_model_stats(
        self, metrics_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate base model statistics across all samples for each model.

        Base model is defined as the first round agent in single agent architecture.
        These statistics are shared across all architectures for the same dataset and model.

        Args:
            metrics_data: Performance metrics

        Returns:
            Dictionary mapping model_name to base model statistics
        """
        base_model_stats = {}

        # Iterate through each model to calculate statistics
        for model_name, model_metrics in metrics_data.get("models", {}).items():
            base_model_correct = 0
            base_model_format_compliant = 0
            base_model_predictions = 0

            # Count correct predictions and format compliance
            for exp_name, exp_metrics in model_metrics.get("experiments", {}).items():
                architecture = exp_metrics.get("agent_architecture", "unknown")
                if architecture == "single":
                    samples = exp_metrics.get("samples", {})
                    for sample_id, sample_data in samples.items():
                        for agent_key, agent_data in sample_data.get(
                            "agents", {}
                        ).items():
                            if agent_key.endswith("_round_1"):
                                base_model_predictions += 1
                                if agent_data["is_correct"]:
                                    base_model_correct += 1
                                if agent_data["format_compliance"]:
                                    base_model_format_compliant += 1
                                break

            # Calculate accuracy and format compliance rate
            accuracy = (
                base_model_correct / base_model_predictions
                if base_model_predictions > 0
                else 0
            )
            format_compliance_rate = (
                base_model_format_compliant / base_model_predictions
                if base_model_predictions > 0
                else 0
            )

            base_model_stats[model_name] = {
                "accuracy": accuracy,
                "format_compliance_rate": format_compliance_rate,
            }

        return base_model_stats

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

        # Extract base model data for comparison
        base_model_data = self._extract_base_model_data(metrics_data)

        # Iterate through models and experiments to extract sample-level data
        for model_name, model_entropy in entropy_data.get("models", {}).items():
            for exp_name, exp_entropy in model_entropy.get("experiments", {}).items():
                # Skip experiments with errors
                if "error" in exp_entropy:
                    continue

                # Get corresponding metrics data
                model_metrics = metrics_data.get("models", {}).get(model_name, {})
                exp_metrics = model_metrics.get("experiments", {}).get(exp_name, {})
                if not exp_metrics:
                    continue

                # Extract experiment metadata
                architecture = exp_entropy.get("agent_architecture", "unknown")
                num_rounds = exp_entropy.get("num_rounds", 0)

                # Get sample-level entropy and metrics
                micro_stats = exp_entropy.get("micro_statistics", {})
                samples_entropy = micro_stats.get("samples", {})
                samples_metrics = exp_metrics.get("samples", {})

                # Process each sample
                for sample_id, sample_metrics in samples_metrics.items():
                    if sample_id not in samples_entropy:
                        continue

                    sample_entropy = samples_entropy[sample_id]

                    # Extract sample-level performance metrics
                    ground_truth = sample_metrics.get("ground_truth", "")
                    final_predicted_answer = sample_metrics.get(
                        "final_predicted_answer", ""
                    )
                    is_finally_correct = sample_metrics.get("is_finally_correct", False)
                    final_format_compliance = sample_metrics.get(
                        "final_format_compliance", False
                    )

                    # Extract base model data for this sample
                    base_model_sample_data = base_model_data.get(model_name, {}).get(
                        sample_id, {}
                    )
                    base_model_predicted_answer = base_model_sample_data.get(
                        "predicted_answer", ""
                    )
                    base_model_is_finally_correct = base_model_sample_data.get(
                        "is_correct", False
                    )
                    base_model_format_compliance = base_model_sample_data.get(
                        "format_compliance", False
                    )

                    # Process each agent within the sample
                    for agent_key, agent_metrics in sample_metrics.get(
                        "agents", {}
                    ).items():
                        agent_entropy_data = sample_entropy.get("agents", {}).get(
                            agent_key, {}
                        )

                        if not agent_entropy_data:
                            continue

                        # Extract agent type and execution details
                        agent_type = agent_entropy_data.get(
                            "agent_type", agent_key.split("_")[0]
                        )

                        execution_order = agent_entropy_data["execution_order"]
                        agent_time_cost = agent_metrics["agent_time_cost"]
                        avg_entropy = agent_metrics["average_entropy"]

                        # Skip orchestrator agents in debate architecture
                        if architecture == "debate" and agent_type == "orchestrator":
                            continue

                        # Build comprehensive record with all metrics
                        record = {
                            "model_name": model_name,
                            "sample_id": sample_id,
                            "experiment_name": exp_name,
                            "architecture": architecture,
                            "num_rounds": num_rounds,
                            "ground_truth": ground_truth,
                            "agent_name": agent_type,
                            "agent_key": agent_key,
                            "execution_order": execution_order,
                            "agent_time_cost": agent_time_cost,
                            "final_predicted_answer": final_predicted_answer,
                            "is_finally_correct": is_finally_correct,
                            "final_format_compliance": final_format_compliance,
                            "base_model_predicted_answer": base_model_predicted_answer,
                            "base_model_is_finally_correct": base_model_is_finally_correct,
                            "base_model_format_compliance": base_model_format_compliance,
                            "sample_total_entropy": sample_entropy.get(
                                "total_entropy", 0
                            ),
                            "sample_max_entropy": sample_entropy.get("max_entropy", 0),
                            "sample_min_entropy": sample_entropy.get("min_entropy", 0),
                            "sample_mean_entropy": sample_entropy.get(
                                "mean_entropy", 0
                            ),
                            "sample_median_entropy": sample_entropy.get(
                                "median_entropy", 0
                            ),
                            "sample_std_entropy": sample_entropy.get("std_entropy", 0),
                            "sample_variance_entropy": sample_entropy.get(
                                "variance_entropy", 0
                            ),
                            "sample_q1_entropy": sample_entropy.get("q1_entropy", 0),
                            "sample_q3_entropy": sample_entropy.get("q3_entropy", 0),
                            "sample_num_agents": sample_entropy.get("num_agents", 0),
                            "sample_all_agents_token_count": sample_entropy.get(
                                "all_agents_token_count", 0
                            ),
                            "sample_avg_entropy_per_token": sample_entropy.get(
                                "average_entropy_per_token", 0
                            ),
                            "agent_total_entropy": agent_entropy_data.get(
                                "total_entropy", 0
                            ),
                            "agent_max_entropy": agent_entropy_data.get(
                                "max_entropy", 0
                            ),
                            "agent_min_entropy": agent_entropy_data.get(
                                "min_entropy", 0
                            ),
                            "agent_mean_entropy": agent_entropy_data.get(
                                "mean_entropy", 0
                            ),
                            "agent_median_entropy": agent_entropy_data.get(
                                "median_entropy", 0
                            ),
                            "agent_std_entropy": agent_entropy_data.get(
                                "std_entropy", 0
                            ),
                            "agent_variance_entropy": agent_entropy_data.get(
                                "variance_entropy", 0
                            ),
                            "agent_q1_entropy": agent_entropy_data.get("q1_entropy", 0),
                            "agent_q3_entropy": agent_entropy_data.get("q3_entropy", 0),
                            "agent_token_count": agent_entropy_data.get(
                                "token_count", 0
                            ),
                            "agent_avg_entropy_per_token": agent_entropy_data.get(
                                "average_entropy_per_token", 0
                            ),
                            "agent_round_number": agent_entropy_data.get(
                                "round_number", 0
                            ),
                            "agent_avg_entropy": avg_entropy,
                        }

                        # Calculate derived features with numerical stability
                        sample_mean_entropy = sample_entropy.get("mean_entropy", 0)
                        sample_std_entropy = sample_entropy.get("std_entropy", 0)
                        sample_total_entropy = sample_entropy.get("total_entropy", 0)
                        sample_num_agents = sample_entropy.get("num_agents", 0)
                        agent_total_entropy = agent_entropy_data.get("total_entropy", 0)

                        # sample_entropy_stability_index: 1 - (std / mean)
                        if sample_mean_entropy > 0:
                            record["sample_entropy_stability_index"] = 1 - (
                                sample_std_entropy / sample_mean_entropy
                            )
                        else:
                            record["sample_entropy_stability_index"] = 0.0

                        # agent_entropy_contribution: agent_total / sample_total
                        if sample_total_entropy > 0:
                            record["agent_entropy_contribution"] = (
                                agent_total_entropy / sample_total_entropy
                            )
                        else:
                            record["agent_entropy_contribution"] = 0.0

                        # sample_avg_entropy_per_agent: sample_total / sample_num_agents
                        if sample_num_agents > 0:
                            record["sample_avg_entropy_per_agent"] = (
                                sample_total_entropy / sample_num_agents
                            )
                        else:
                            record["sample_avg_entropy_per_agent"] = 0.0

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

        # Iterate through models and experiments to extract round-level data
        for model_name, model_entropy in entropy_data.get("models", {}).items():
            for exp_name, exp_entropy in model_entropy.get("experiments", {}).items():
                # Skip experiments with errors
                if "error" in exp_entropy:
                    continue

                # Extract round-level entropy statistics
                macro_stats = exp_entropy.get("macro_statistics", {})
                round_level = macro_stats.get("round_level", {})

                # Initialize round statistics from entropy data
                for round_num, round_data in round_level.items():
                    key = (exp_name, int(round_num))
                    round_stats[key] = {
                        "round_total_entropy": round_data.get("total_entropy", 0),
                        "round_num_inferences": round_data.get("num_inferences", 0),
                        "round_infer_avg_entropy": round_data.get(
                            "infer_average_entropy", 0
                        ),
                    }

                # Get corresponding metrics data
                model_metrics = metrics_data.get("models", {}).get(model_name, {})
                exp_metrics = model_metrics.get("experiments", {}).get(exp_name, {})
                if not exp_metrics:
                    continue

                # Get architecture type for time calculation
                architecture = exp_entropy.get("agent_architecture", "unknown")
                micro_stats = exp_entropy.get("micro_statistics", {})
                samples_entropy = micro_stats.get("samples", {})
                samples_metrics = exp_metrics.get("samples", {})

                # Initialize time tracking for centralized architecture
                if architecture == "centralized":
                    sample_round_times = defaultdict(lambda: defaultdict(list))

                # Process each sample to calculate round-level metrics
                for sample_id, sample_metrics in samples_metrics.items():
                    if sample_id not in samples_entropy:
                        continue

                    sample_entropy = samples_entropy[sample_id]

                    # Process each agent within the sample
                    for agent_key, agent_metrics in sample_metrics.get(
                        "agents", {}
                    ).items():
                        agent_entropy_data = sample_entropy.get("agents", {}).get(
                            agent_key, {}
                        )

                        agent_type = agent_entropy_data.get(
                            "agent_type", agent_key.split("_")[0]
                        )

                        # Skip orchestrator agents in debate architecture
                        if architecture == "debate" and agent_type == "orchestrator":
                            continue

                        round_number = agent_entropy_data.get("round_number", 0)
                        if round_number == 0:
                            continue

                        # Initialize round statistics if not exists
                        key = (exp_name, round_number)
                        if key not in round_stats:
                            round_stats[key] = {
                                "round_total_entropy": 0,
                                "round_num_inferences": 0,
                                "round_infer_avg_entropy": 0,
                                "round_total_time": 0,
                                "round_total_token": 0,
                            }

                        round_stats[key].setdefault("round_total_time", 0)
                        round_stats[key].setdefault("round_total_token", 0)

                        agent_time_cost = agent_metrics["agent_time_cost"]

                        # Handle time calculation differently for centralized architecture
                        if architecture == "centralized":
                            if agent_type == "OrchestratorAgent":
                                sample_round_times[sample_id][round_number].append(
                                    ("orchestrator", agent_time_cost)
                                )
                            else:
                                sample_round_times[sample_id][round_number].append(
                                    ("parallel", agent_time_cost)
                                )
                        else:
                            round_stats[key]["round_total_time"] += agent_time_cost

                        # Accumulate token count
                        round_stats[key]["round_total_token"] += agent_entropy_data[
                            "token_count"
                        ]

                # Calculate total time for centralized architecture (parallel execution)
                if architecture == "centralized":
                    for sample_id, rounds in sample_round_times.items():
                        for round_number, agents in rounds.items():
                            parallel_times = []
                            orchestrator_time = 0
                            for agent_type, time_cost in agents:
                                if agent_type == "orchestrator":
                                    orchestrator_time = time_cost
                                else:
                                    parallel_times.append(time_cost)

                            # Total time = max parallel time + orchestrator time
                            round_time = (
                                max(parallel_times) + orchestrator_time
                                if parallel_times
                                else orchestrator_time
                            )
                            key = (exp_name, round_number)
                            round_stats[key]["round_total_time"] += round_time

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

        # Iterate through models and experiments to extract agent-level data
        for model_name, model_entropy in entropy_data.get("models", {}).items():
            for exp_name, exp_entropy in model_entropy.get("experiments", {}).items():
                # Skip experiments with errors
                if "error" in exp_entropy:
                    continue

                # Extract agent-level entropy statistics
                macro_stats = exp_entropy.get("macro_statistics", {})
                agent_level = macro_stats.get("agent_level", {})

                # Process each agent's statistics
                for agent_name, agent_data in agent_level.items():
                    key = (exp_name, agent_name)
                    # Build comprehensive agent statistics record
                    agent_stats[key] = {
                        "agent_total_entropy": agent_data.get("total_entropy", 0),
                        "agent_num_inferences": agent_data.get("num_inferences", 0),
                        "agent_total_tokens": agent_data.get("total_tokens", 0),
                        "agent_avg_entropy": agent_data.get("infer_average_entropy", 0),
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

        # Calculate base model statistics for comparison
        base_model_stats = self._calculate_base_model_stats(metrics_data)

        # Iterate through models and experiments to extract experiment-level data
        for model_name, model_entropy in entropy_data.get("models", {}).items():
            for exp_name, exp_entropy in model_entropy.get("experiments", {}).items():
                # Skip experiments with errors
                if "error" in exp_entropy:
                    continue

                # Get corresponding metrics data
                model_metrics = metrics_data.get("models", {}).get(model_name, {})
                exp_metrics = model_metrics.get("experiments", {}).get(exp_name, {})
                if not exp_metrics:
                    continue

                # Extract experiment metadata
                architecture = exp_entropy.get("agent_architecture", "unknown")
                macro_stats = exp_entropy.get("macro_statistics", {})
                exp_level = macro_stats.get("experiment_level", {})

                # Initialize counters for experiment-level metrics
                samples = exp_metrics.get("samples", {})
                total_correct = 0
                total_format_compliance = 0
                total_predictions = 0
                total_time = 0
                total_token = 0

                # Get sample-level entropy data
                micro_stats = exp_entropy.get("micro_statistics", {})
                samples_entropy = micro_stats.get("samples", {})

                # Initialize time tracking for centralized architecture
                if architecture == "centralized":
                    sample_round_times = defaultdict(lambda: defaultdict(list))

                # Process each sample to calculate experiment-level metrics
                for sample_id, sample_data in samples.items():
                    if sample_id in samples_entropy:
                        sample_entropy = samples_entropy[sample_id]
                        # Accumulate token count from all agents
                        for agent_key, agent_entropy_data in sample_entropy.get(
                            "agents", {}
                        ).items():
                            agent_type = agent_entropy_data.get(
                                "agent_type", agent_key.split("_")[0]
                            )
                            # Skip orchestrator in debate architecture
                            if (
                                architecture == "debate"
                                and agent_type == "orchestrator"
                            ):
                                continue
                            total_token += agent_entropy_data.get("token_count", 0)

                    # Process each agent within the sample
                    for agent_key, agent_data in sample_data.get("agents", {}).items():
                        agent_type = agent_data.get(
                            "agent_type", agent_key.split("_")[0]
                        )
                        # Skip orchestrator in debate architecture
                        if architecture == "debate" and agent_type == "orchestrator":
                            continue

                        # Handle time calculation for centralized architecture
                        if architecture == "centralized":
                            round_number = (
                                int(agent_key.split("_round_")[-1])
                                if "_round_" in agent_key
                                else 0
                            )
                            if agent_type == "OrchestratorAgent":
                                sample_round_times[sample_id][round_number].append(
                                    ("orchestrator", agent_data["agent_time_cost"])
                                )
                            else:
                                sample_round_times[sample_id][round_number].append(
                                    ("parallel", agent_data["agent_time_cost"])
                                )
                        else:
                            # Accumulate time for non-centralized architectures
                            total_time += agent_data["agent_time_cost"]

                    # Count predictions and correctness
                    if sample_data.get("final_predicted_answer") is not None:
                        total_predictions += 1
                        if sample_data.get("is_finally_correct", False):
                            total_correct += 1
                        if sample_data.get("final_format_compliance", False):
                            total_format_compliance += 1

                # Calculate total time for centralized architecture (parallel execution)
                if architecture == "centralized":
                    for sample_id, rounds in sample_round_times.items():
                        for round_number, agents in rounds.items():
                            parallel_times = []
                            orchestrator_time = 0
                            for agent_type, time_cost in agents:
                                if agent_type == "orchestrator":
                                    orchestrator_time = time_cost
                                else:
                                    parallel_times.append(time_cost)

                            # Total time = max parallel time + orchestrator time
                            round_time = (
                                max(parallel_times) + orchestrator_time
                                if parallel_times
                                else orchestrator_time
                            )
                            total_time += round_time

                # Calculate accuracy and format compliance rate
                accuracy = (
                    total_correct / total_predictions if total_predictions > 0 else 0
                )
                format_compliance_rate = (
                    total_format_compliance / total_predictions
                    if total_predictions > 0
                    else 0
                )

                # Get base model statistics for comparison
                model_base_stats = base_model_stats.get(model_name, {})
                # Build comprehensive experiment statistics record
                exp_stats[exp_name] = {
                    "exp_total_entropy": exp_level.get("total_entropy", 0),
                    "exp_infer_average_entropy": exp_level.get(
                        "infer_average_entropy", 0
                    ),
                    "exp_num_inferences": exp_entropy.get("num_inferences", 0),
                    "exp_accuracy": accuracy,
                    "exp_format_compliance_rate": format_compliance_rate,
                    "exp_total_time": total_time,
                    "exp_total_token": total_token,
                    "base_model_accuracy": model_base_stats.get("accuracy", 0),
                    "base_model_format_compliance_rate": model_base_stats.get(
                        "format_compliance_rate", 0
                    ),
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

        # Merge each sample record with corresponding round, agent, and experiment stats
        for record in sample_records:
            exp_name = record["experiment_name"]
            agent_name = record["agent_name"]
            agent_round_number = record.get("agent_round_number", 0)

            # Merge agent-level statistics
            key = (exp_name, agent_name)
            if key in agent_stats:
                record.update(agent_stats[key])

            # Merge round-level statistics
            round_key = (exp_name, agent_round_number)
            if round_key in round_stats:
                record.update(round_stats[round_key])

            # Merge experiment-level statistics
            if exp_name in exp_stats:
                record.update(exp_stats[exp_name])

            merged_records.append(record)

        return merged_records

    def add_dynamic_round_features(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add dynamic round comparison features to records.

        This method adds features that compare different rounds:
        - round_{x}_{y}_change_tokens: Token change from round x to round y
        - round_{x}_{y}_change_entropy: Entropy change from round x to round y

        Args:
            records: List of sample records with round-level data

        Returns:
            List of records with added dynamic features
        """
        # Group records by experiment to enable round comparisons
        exp_records = defaultdict(list)
        for record in records:
            exp_name = record["experiment_name"]
            exp_records[exp_name].append(record)

        # For each experiment, calculate round comparison features
        for exp_name, exp_record_list in exp_records.items():
            # Collect round data for this experiment
            round_data = {}
            for record in exp_record_list:
                round_num = record.get("agent_round_number", 0)
                if round_num > 0:
                    round_data[round_num] = {
                        "round_total_token": record.get("round_total_token", 0),
                        "round_total_entropy": record.get("round_total_entropy", 0),
                    }

            # Get all valid round numbers sorted
            round_numbers = sorted(round_data.keys())

            # Calculate comparison features for all valid round pairs
            for i, x in enumerate(round_numbers):
                for y in round_numbers[i + 1 :]:
                    if x in round_data and y in round_data:
                        # Calculate token change
                        token_change = (
                            round_data[y]["round_total_token"]
                            - round_data[x]["round_total_token"]
                        )
                        token_feature_name = f"round_{x}_{y}_change_tokens"

                        # Calculate entropy change
                        entropy_change = (
                            round_data[y]["round_total_entropy"]
                            - round_data[x]["round_total_entropy"]
                        )
                        entropy_feature_name = f"round_{x}_{y}_change_entropy"

                        # Add features to all records in this experiment
                        for record in exp_record_list:
                            record[token_feature_name] = token_change
                            record[entropy_feature_name] = entropy_change

        return records

    def write_csv(self, records: List[Dict[str, Any]], filename: str):
        """Write records to CSV file.

        Args:
            records: List of dictionaries to write
            filename: Output filename
        """
        # Check if there are records to write
        if not records:
            print(f"No records to write for {filename}")
            return

        # Construct full output path
        output_path = self.output_dir / filename
        # Extract field names from first record
        fieldnames = list(records[0].keys())

        # Write records to CSV file
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"Successfully wrote {len(records)} records to {output_path}")

    def generate_exclude_agent_csv(self, input_csv_path: str, output_csv_path: str):
        """Generate CSV file excluding agent-specific columns and merging records by sample.

        This method processes an existing CSV file to:
        1. Remove agent-specific columns (agent_name, agent_key, execution_order,
           agent_time_cost, final_predicted_answer, base_model_predicted_answer,
           and all columns starting with "agent_" except agent_round_number which is used for round detection)
        2. Merge multiple records per sample into a single record
        3. Rename round-related columns to include round number (e.g., round_1_total_entropy)
        4. Save to a new CSV file

        Args:
            input_csv_path: Path to the input CSV file
            output_csv_path: Path to the output CSV file
        """
        # Define columns to remove
        columns_to_remove = {
            "experiment_name",
            "ground_truth",
            "agent_name",
            "agent_key",
            "execution_order",
            "agent_time_cost",
            "final_predicted_answer",
            "base_model_predicted_answer",
        }

        # Read the input CSV file
        input_path = Path(input_csv_path)
        if not input_path.exists():
            print(f"Input CSV file not found: {input_path}")
            return

        records = []
        with open(input_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames

            # Process each row and extract round number before removing agent columns
            for row in reader:
                # Extract round number from agent_round_number before removing agent columns
                round_num = None
                if "agent_round_number" in row:
                    try:
                        round_num = int(row["agent_round_number"])
                    except (ValueError, TypeError):
                        pass
                
                # Determine which columns to keep (excluding agent-specific columns)
                columns_to_keep = []
                for field in fieldnames:
                    if field in columns_to_remove:
                        continue
                    if field.startswith("agent_") and field != "agent_round_number":
                        continue
                    columns_to_keep.append(field)

                # Create new row with only the columns we want to keep
                new_row = {col: row[col] for col in columns_to_keep}
                
                # Store round number for later processing
                if round_num is not None:
                    new_row["_round_number"] = round_num
                
                records.append(new_row)

        # Group records by sample (using model_name, sample_id, architecture, num_rounds)
        sample_groups = defaultdict(list)
        for record in records:
            sample_key = (
                record.get("model_name", ""),
                record.get("sample_id", ""),
                record.get("architecture", ""),
                record.get("num_rounds", ""),
            )
            sample_groups[sample_key].append(record)

        # Merge records for each sample
        merged_records = []
        for sample_key, sample_records in sample_groups.items():
            if not sample_records:
                continue

            # Use the first record as base
            base_record = sample_records[0].copy()
            
            # Get num_rounds from the base record
            num_rounds = None
            if "num_rounds" in base_record and base_record["num_rounds"]:
                try:
                    num_rounds = int(base_record["num_rounds"])
                except (ValueError, TypeError):
                    pass

            # Collect round data from all records
            round_data = {}
            for record in sample_records:
                # Get round number from the stored _round_number field
                round_num = None
                if "_round_number" in record:
                    round_num = record["_round_number"]
                
                # Validate round number against num_rounds if available
                if round_num is not None and num_rounds is not None:
                    if round_num < 1 or round_num > num_rounds:
                        print(f"Warning: Round number {round_num} exceeds num_rounds {num_rounds}, skipping")
                        continue

                # If we have a valid round number, collect round-specific fields
                if round_num is not None:
                    # If we already have data for this round, verify it's consistent
                    if round_num in round_data:
                        existing_data = round_data[round_num]
                        new_data = {
                            "round_total_entropy": record.get("round_total_entropy", ""),
                            "round_num_inferences": record.get("round_num_inferences", ""),
                            "round_infer_avg_entropy": record.get(
                                "round_infer_avg_entropy", ""
                            ),
                            "round_total_time": record.get("round_total_time", ""),
                            "round_total_token": record.get("round_total_token", ""),
                        }
                        # Check if the new data is consistent with existing data
                        # (they should be the same for all agents in the same round)
                        for key, value in new_data.items():
                            if value and existing_data[key] and existing_data[key] != value:
                                print(f"Warning: Inconsistent {key} for round {round_num}: {existing_data[key]} vs {value}")
                    else:
                        round_data[round_num] = {
                            "round_total_entropy": record.get("round_total_entropy", ""),
                            "round_num_inferences": record.get("round_num_inferences", ""),
                            "round_infer_avg_entropy": record.get(
                                "round_infer_avg_entropy", ""
                            ),
                            "round_total_time": record.get("round_total_time", ""),
                            "round_total_token": record.get("round_total_token", ""),
                        }

            # Remove original round fields from base record
            for field in [
                "round_total_entropy",
                "round_num_inferences",
                "round_infer_avg_entropy",
                "round_total_time",
                "round_total_token",
                "_round_number",
            ]:
                if field in base_record:
                    del base_record[field]

            # Add round-specific fields with round number in the name
            for round_num, data in sorted(round_data.items()):
                base_record[f"round_{round_num}_total_entropy"] = data[
                    "round_total_entropy"
                ]
                base_record[f"round_{round_num}_num_inferences"] = data[
                    "round_num_inferences"
                ]
                base_record[f"round_{round_num}_infer_avg_entropy"] = data[
                    "round_infer_avg_entropy"
                ]
                base_record[f"round_{round_num}_total_time"] = data["round_total_time"]
                base_record[f"round_{round_num}_total_token"] = data[
                    "round_total_token"
                ]

            merged_records.append(base_record)

        # Determine final fieldnames (may include dynamic round fields)
        all_fieldnames = set()
        for record in merged_records:
            all_fieldnames.update(record.keys())

        # Sort fieldnames for consistent ordering
        # Put common fields first, then round fields
        common_fields = [
            "model_name",
            "sample_id",
            "architecture",
            "num_rounds",
            "is_finally_correct",
            "final_format_compliance",
            "base_model_is_finally_correct",
            "base_model_format_compliance",
            "sample_total_entropy",
            "sample_max_entropy",
            "sample_min_entropy",
            "sample_mean_entropy",
            "sample_median_entropy",
            "sample_std_entropy",
            "sample_variance_entropy",
            "sample_q1_entropy",
            "sample_q3_entropy",
            "sample_num_agents",
            "sample_all_agents_token_count",
            "sample_avg_entropy_per_token",
            "sample_entropy_stability_index",
            "sample_avg_entropy_per_agent",
            "exp_total_entropy",
            "exp_infer_average_entropy",
            "exp_num_inferences",
            "exp_accuracy",
            "exp_format_compliance_rate",
            "exp_total_time",
            "exp_total_token",
            "base_model_accuracy",
            "base_model_format_compliance_rate",
        ]

        # Add round fields
        round_fields = sorted([f for f in all_fieldnames if f.startswith("round_")])

        # Add any other fields not in common_fields or round_fields
        other_fields = sorted(
            [
                f
                for f in all_fieldnames
                if f not in common_fields and not f.startswith("round_")
            ]
        )

        final_fieldnames = common_fields + other_fields + round_fields

        # Write the output CSV file
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not merged_records:
            print(f"No records to write to {output_path}")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=final_fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(merged_records)

        print(
            f"Successfully wrote {len(merged_records)} merged records to {output_path}"
        )
        print(
            f"Merged {len(records)} original records into {len(merged_records)} sample records"
        )
        print(
            f"Removed {len(fieldnames) - len(columns_to_keep)} agent-specific columns"
        )

    def generate_aggregated_csvs(self):
        """Generate multiple CSV files based on different experimental conditions."""
        # Load JSON data files
        entropy_data, metrics_data = self.load_json_files()

        # Extract data at different levels of granularity
        sample_records = self.extract_sample_level_data(entropy_data, metrics_data)
        round_stats = self.extract_round_level_data(entropy_data, metrics_data)
        agent_stats = self.extract_agent_level_data(entropy_data, metrics_data)
        exp_stats = self.extract_experiment_level_data(entropy_data, metrics_data)

        # Merge all levels of data into comprehensive records
        merged_records = self.merge_all_data(
            sample_records, round_stats, agent_stats, exp_stats
        )

        # Exit early if no records were found
        if not merged_records:
            print("No records found to aggregate")
            return

        # Add dynamic round comparison features
        merged_records = self.add_dynamic_round_features(merged_records)

        # Write all aggregated data to a single CSV file
        self.write_csv(merged_records, "all_aggregated_data.csv")

        # Generate CSV file excluding agent-specific columns
        input_csv_path = self.output_dir / "all_aggregated_data.csv"
        output_csv_path = self.output_dir / "all_aggregated_data_exclude_agent.csv"
        self.generate_exclude_agent_csv(str(input_csv_path), str(output_csv_path))

        # Group records by model name for model-specific CSV files
        records_by_model = defaultdict(list)
        for record in merged_records:
            model_name = record["model_name"]
            records_by_model[model_name].append(record)

        # Generate CSV files for each model
        for model_name, model_records in records_by_model.items():
            # Save original output directory
            original_output_dir = self.output_dir
            # Create model-specific subdirectory
            self.output_dir = self.output_dir / model_name
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Group records by experiment name for experiment-specific CSV files
            records_by_experiment = defaultdict(list)
            for record in model_records:
                exp = record["experiment_name"]
                records_by_experiment[exp].append(record)

            # Write CSV file for each experiment
            for exp, records in records_by_experiment.items():
                filename = f"aggregated_data_{exp}.csv"
                self.write_csv(records, filename)

            # Write CSV file containing all data for this model
            self.write_csv(model_records, "aggregated_data.csv")
            # Restore original output directory
            self.output_dir = original_output_dir

        return merged_records
