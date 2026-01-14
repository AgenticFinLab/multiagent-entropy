"""Feature enhancement utilities for multi-agent evaluation.

This module encapsulates advanced feature engineering logic that augments
basic aggregated results with richer entropy-related sample-level features.

It is designed to be used by the Aggregator to keep aggregation code simple
while centralizing complex feature calculations here.
"""

from typing import Dict, Any, List
from collections import defaultdict

import numpy as np


class FeatureEnhancer:
    """Feature enhancer for sample-level and round-level entropy features."""

    @staticmethod
    def _extract_base_model_data(
        metrics_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract base model data from metrics for each model.

        Base model is defined as the first round agent in single agent architecture.
        This data is shared across all architectures for the same dataset and model.

        Args:
            metrics_data: Performance metrics

        Returns:
            Dictionary mapping model_name to sample_id to base model data
        """
        base_model_data: Dict[str, Dict[str, Any]] = {}

        # Iterate through each model's metrics
        for model_name, model_metrics in metrics_data.get("models", {}).items():
            model_base_data: Dict[str, Any] = {}
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

    @staticmethod
    def _extract_base_model_entropy(
        entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract base model entropy statistics for each model and sample.

        Base model is defined as the first round agent in single agent architecture,
        consistent with `_extract_base_model_data`.

        The extracted statistics are shared across all architectures for the same
        dataset and model, and are used as a reference baseline for entropy-based
        features.

        Args:
            entropy_data: Entropy analysis results
            metrics_data: Performance metrics (used to locate the base agent)

        Returns:
            Dictionary mapping model_name -> sample_id -> base entropy stats
        """
        base_model_entropy: Dict[str, Dict[str, Any]] = {}

        for model_name, model_metrics in metrics_data.get("models", {}).items():
            model_entropy = entropy_data.get("models", {}).get(model_name, {})
            model_base_entropy: Dict[str, Any] = {}

            for exp_name, exp_metrics in model_metrics.get("experiments", {}).items():
                architecture = exp_metrics.get("agent_architecture", "unknown")
                if architecture != "single":
                    continue

                # Find corresponding entropy experiment
                exp_entropy = model_entropy.get("experiments", {}).get(exp_name, {})
                if not exp_entropy:
                    continue

                micro_stats = exp_entropy.get("micro_statistics", {})
                samples_entropy = micro_stats.get("samples", {})
                samples_metrics = exp_metrics.get("samples", {})

                for sample_id, sample_metrics in samples_metrics.items():
                    if sample_id not in samples_entropy:
                        continue

                    sample_entropy = samples_entropy[sample_id]
                    agents_entropy = sample_entropy.get("agents", {})

                    # Reuse the same base agent selection rule: first agent in round 1
                    for agent_key, agent_data in sample_metrics.get("agents", {}).items():
                        if agent_key.endswith("_round_1"):
                            agent_entropy_data = agents_entropy.get(agent_key, {})
                            if agent_entropy_data:
                                total_entropy = agent_entropy_data.get("total_entropy", 0.0)
                                token_count = agent_entropy_data.get("token_count", 0)
                                avg_entropy_per_token = agent_entropy_data.get(
                                    "average_entropy_per_token", 0.0
                                )
                                model_base_entropy[sample_id] = {
                                    "total_entropy": float(total_entropy),
                                    "token_count": int(token_count),
                                    "avg_entropy_per_token": float(avg_entropy_per_token),
                                }
                            break

            base_model_entropy[model_name] = model_base_entropy

        return base_model_entropy

    @staticmethod
    def build_sample_records(
        entropy_data: Dict[str, Any], metrics_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build enriched sample-level records from entropy and metrics JSON data.

        This method contains the feature-enhanced implementation that was
        previously embedded in Aggregator.extract_sample_level_data.
        """
        records: List[Dict[str, Any]] = []

        # Extract base model data for comparison
        base_model_data = FeatureEnhancer._extract_base_model_data(metrics_data)
        base_model_entropy = FeatureEnhancer._extract_base_model_entropy(
            entropy_data, metrics_data
        )

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

                    # Pre-compute sample-level entropy statistics
                    sample_total_entropy = sample_entropy.get("total_entropy", 0)
                    sample_max_entropy = sample_entropy.get("max_entropy", 0)
                    sample_min_entropy = sample_entropy.get("min_entropy", 0)
                    sample_mean_entropy = sample_entropy.get("mean_entropy", 0)
                    sample_median_entropy = sample_entropy.get("median_entropy", 0)
                    sample_std_entropy = sample_entropy.get("std_entropy", 0)
                    sample_variance_entropy = sample_entropy.get("variance_entropy", 0)
                    sample_q1_entropy = sample_entropy.get("q1_entropy", 0)
                    sample_q3_entropy = sample_entropy.get("q3_entropy", 0)
                    sample_num_agents = sample_entropy.get("num_agents", 0)
                    sample_all_agents_token_count = sample_entropy.get(
                        "all_agents_token_count", 0
                    )
                    sample_avg_entropy_per_token = sample_entropy.get(
                        "average_entropy_per_token", 0
                    )

                    # Sample-level distribution shape features
                    sample_shape_features: Dict[str, Any] = {}
                    # Range and IQR
                    sample_shape_features["sample_entropy_range"] = (
                        float(sample_max_entropy) - float(sample_min_entropy)
                    )
                    sample_iqr = float(sample_q3_entropy) - float(sample_q1_entropy)
                    sample_shape_features["sample_entropy_iqr"] = sample_iqr

                    # Relative IQR w.r.t. mean and range
                    if sample_mean_entropy > 0:
                        sample_shape_features["sample_entropy_relative_iqr_mean"] = (
                            sample_iqr / float(sample_mean_entropy)
                        )
                    else:
                        sample_shape_features["sample_entropy_relative_iqr_mean"] = 0.0

                    range_denominator = float(sample_max_entropy) - float(
                        sample_min_entropy
                    )
                    if range_denominator != 0:
                        sample_shape_features[
                            "sample_entropy_relative_iqr_range"
                        ] = sample_iqr / range_denominator
                    else:
                        sample_shape_features["sample_entropy_relative_iqr_range"] = 0.0

                    # Bowley skewness based on quartiles
                    iqr_denominator = float(sample_q3_entropy) - float(sample_q1_entropy)
                    if iqr_denominator != 0:
                        sample_shape_features["sample_entropy_bowley_skewness"] = (
                            float(sample_q3_entropy)
                            + float(sample_q1_entropy)
                            - 2.0 * float(sample_median_entropy)
                        ) / iqr_denominator
                    else:
                        sample_shape_features["sample_entropy_bowley_skewness"] = 0.0

                    # Median over mean
                    if sample_mean_entropy > 0:
                        sample_shape_features["sample_entropy_median_over_mean"] = (
                            float(sample_median_entropy) / float(sample_mean_entropy)
                        )
                    else:
                        sample_shape_features["sample_entropy_median_over_mean"] = 0.0

                    # Tail weight: (max - q3) / IQR
                    if iqr_denominator != 0:
                        sample_shape_features["sample_entropy_tail_weight"] = (
                            float(sample_max_entropy) - float(sample_q3_entropy)
                        ) / iqr_denominator
                    else:
                        sample_shape_features["sample_entropy_tail_weight"] = 0.0

                    # Coefficient of variation: std / mean
                    if sample_mean_entropy > 0:
                        sample_shape_features["sample_entropy_cv"] = (
                            float(sample_std_entropy) / float(sample_mean_entropy)
                        )
                    else:
                        sample_shape_features["sample_entropy_cv"] = 0.0

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

                    # Extract base model entropy data for this sample (single-agent baseline)
                    base_sample_entropy_data = base_model_entropy.get(model_name, {}).get(
                        sample_id, {}
                    )
                    base_sample_total_entropy = float(
                        base_sample_entropy_data.get("total_entropy", 0.0)
                    )
                    base_sample_token_count = int(
                        base_sample_entropy_data.get("token_count", 0)
                    )
                    base_sample_avg_entropy_per_token = float(
                        base_sample_entropy_data.get("avg_entropy_per_token", 0.0)
                    )

                    sample_base_entropy_features: Dict[str, Any] = {
                        "base_sample_total_entropy": base_sample_total_entropy,
                        "base_sample_token_count": base_sample_token_count,
                        "base_sample_avg_entropy_per_token": base_sample_avg_entropy_per_token,
                    }

                    if base_sample_total_entropy > 0:
                        sample_base_entropy_features["sample_entropy_ratio_vs_base_total"] = (
                            float(sample_total_entropy) / base_sample_total_entropy
                        )
                        sample_base_entropy_features[
                            "sample_entropy_reduction_vs_base_total"
                        ] = base_sample_total_entropy - float(sample_total_entropy)
                    else:
                        sample_base_entropy_features["sample_entropy_ratio_vs_base_total"] = 0.0
                        sample_base_entropy_features[
                            "sample_entropy_reduction_vs_base_total"
                        ] = 0.0

                    if base_sample_avg_entropy_per_token > 0:
                        sample_base_entropy_features[
                            "sample_avg_entropy_per_token_ratio_vs_base"
                        ] = (
                            float(sample_avg_entropy_per_token)
                            / base_sample_avg_entropy_per_token
                        )
                        sample_base_entropy_features[
                            "sample_avg_entropy_per_token_diff_vs_base"
                        ] = float(sample_avg_entropy_per_token) - base_sample_avg_entropy_per_token
                    else:
                        sample_base_entropy_features[
                            "sample_avg_entropy_per_token_ratio_vs_base"
                        ] = 0.0
                        sample_base_entropy_features[
                            "sample_avg_entropy_per_token_diff_vs_base"
                        ] = 0.0

                    # Collect round-based agent entropy statistics
                    # Group agents by round number and collect their entropy metrics
                    round_agent_entropy_data: Dict[int, Dict[str, List[float]]] = defaultdict(
                        lambda: defaultdict(list)
                    )
                    sample_round_total_tokens: Dict[int, int] = defaultdict(int)
                    sample_round_total_entropy: Dict[int, float] = defaultdict(float)

                    for agent_key, agent_entropy_data in sample_entropy.get("agents", {}).items():
                        agent_type = agent_entropy_data.get(
                            "agent_type", agent_key.split("_")[0]
                        )

                        # Skip orchestrator agents in debate architecture
                        if architecture == "debate" and agent_type == "orchestrator":
                            continue

                        round_number = agent_entropy_data.get("round_number", 0)
                        if round_number > 0:
                            # Accumulate per-round total entropy and token statistics for this sample
                            token_count = agent_entropy_data.get("token_count", 0)
                            total_entropy_value = agent_entropy_data.get(
                                "total_entropy", 0.0
                            )
                            sample_round_total_tokens[round_number] += int(token_count)
                            sample_round_total_entropy[round_number] += float(
                                total_entropy_value
                            )

                            # Collect all entropy types for this agent
                            entropy_types = [
                                "max_entropy",
                                "min_entropy",
                                "mean_entropy",
                                "median_entropy",
                                "std_entropy",
                                "variance_entropy",
                                "q1_entropy",
                                "q3_entropy",
                                "total_entropy",
                            ]
                            for entropy_type in entropy_types:
                                value = agent_entropy_data.get(entropy_type, 0)
                                round_agent_entropy_data[round_number][entropy_type].append(value)

                    # Calculate round-based statistics
                    round_statistics: Dict[str, Any] = {}
                    for round_num, entropy_dict in round_agent_entropy_data.items():
                        for entropy_type, values in entropy_dict.items():
                            if len(values) > 1:
                                # Multiple agents in this round, calculate statistics
                                values_array = np.array(values)
                                stats = {
                                    "max": float(np.max(values_array)),
                                    "min": float(np.min(values_array)),
                                    "mean": float(np.mean(values_array)),
                                    "median": float(np.median(values_array)),
                                    "std": float(np.std(values_array)),
                                    "variance": float(np.var(values_array)),
                                    "q1": float(np.percentile(values_array, 25)),
                                    "q3": float(np.percentile(values_array, 75)),
                                }
                            elif len(values) == 1:
                                # Single agent in this round, all statistics are the same value
                                single_value = float(values[0])
                                stats = {
                                    "max": single_value,
                                    "min": single_value,
                                    "mean": single_value,
                                    "median": single_value,
                                    "std": 0.0,
                                    "variance": 0.0,
                                    "q1": single_value,
                                    "q3": single_value,
                                }
                            else:
                                continue

                            # Store statistics with proper naming convention
                            for stat_name, stat_value in stats.items():
                                key = f"sample_round_{round_num}_{stat_name}_agent_{entropy_type}"
                                round_statistics[key] = stat_value

                    # Add per-round sample-level total entropy, token, and density statistics
                    sample_round_entropy_per_token: Dict[int, float] = {}
                    for round_num in sorted(sample_round_total_tokens.keys()):
                        total_tokens = float(sample_round_total_tokens[round_num])
                        total_entropy_round = float(
                            sample_round_total_entropy.get(round_num, 0.0)
                        )
                        key_prefix = f"sample_round_{round_num}_all_agents"
                        round_statistics[f"{key_prefix}_total_entropy"] = total_entropy_round
                        round_statistics[f"{key_prefix}_total_token"] = total_tokens
                        if total_tokens > 0:
                            density = total_entropy_round / total_tokens
                        else:
                            density = 0.0
                        round_statistics[f"{key_prefix}_entropy_per_token"] = density
                        sample_round_entropy_per_token[round_num] = density

                    # Sample-level cross-round dynamics based on per-round totals and densities
                    sample_round_dynamic_features: Dict[str, Any] = {}
                    if sample_round_total_tokens:
                        sorted_rounds = sorted(sample_round_total_tokens.keys())
                        first_round = sorted_rounds[0]
                        last_round = sorted_rounds[-1]

                        first_total_entropy = float(
                            sample_round_total_entropy.get(first_round, 0.0)
                        )
                        last_total_entropy = float(
                            sample_round_total_entropy.get(last_round, 0.0)
                        )
                        sample_round_dynamic_features[
                            "sample_round_all_agents_total_entropy_first_last_diff"
                        ] = last_total_entropy - first_total_entropy
                        if first_total_entropy > 0:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_total_entropy_first_last_ratio"
                            ] = last_total_entropy / first_total_entropy
                        else:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_total_entropy_first_last_ratio"
                            ] = 0.0

                        first_total_tokens = float(
                            sample_round_total_tokens.get(first_round, 0)
                        )
                        last_total_tokens = float(
                            sample_round_total_tokens.get(last_round, 0)
                        )
                        sample_round_dynamic_features[
                            "sample_round_all_agents_total_token_first_last_diff"
                        ] = last_total_tokens - first_total_tokens
                        if first_total_tokens > 0:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_total_token_first_last_ratio"
                            ] = last_total_tokens / first_total_tokens
                        else:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_total_token_first_last_ratio"
                            ] = 0.0

                        first_density = sample_round_entropy_per_token.get(
                            first_round, 0.0
                        )
                        last_density = sample_round_entropy_per_token.get(
                            last_round, 0.0
                        )
                        sample_round_dynamic_features[
                            "sample_round_all_agents_entropy_per_token_first_last_diff"
                        ] = last_density - first_density
                        if first_density > 0:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_entropy_per_token_first_last_ratio"
                            ] = last_density / first_density
                        else:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_entropy_per_token_first_last_ratio"
                            ] = 0.0

                        num_round_steps = max(len(sorted_rounds) - 1, 1)
                        sample_round_dynamic_features[
                            "sample_round_all_agents_entropy_per_token_slope_per_round"
                        ] = (last_density - first_density) / float(num_round_steps)

                        density_values = [
                            sample_round_entropy_per_token[r] for r in sorted_rounds
                        ]
                        if len(density_values) > 1:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_entropy_per_token_volatility"
                            ] = float(np.std(np.array(density_values)))
                        else:
                            sample_round_dynamic_features[
                                "sample_round_all_agents_entropy_per_token_volatility"
                            ] = 0.0

                        # Pairwise change between round 1 and 2 if both exist
                        if 1 in sample_round_total_tokens and 2 in sample_round_total_tokens:
                            sample_round_dynamic_features[
                                "sample_round_1_2_change_tokens"
                            ] = float(
                                sample_round_total_tokens[2]
                                - sample_round_total_tokens[1]
                            )
                            sample_round_dynamic_features[
                                "sample_round_1_2_change_entropy"
                            ] = float(
                                sample_round_total_entropy[2]
                                - sample_round_total_entropy[1]
                            )

                    # Cross-round features based on aggregated agent mean and total entropy
                    sample_round_trend_features: Dict[str, Any] = {}
                    mean_agent_mean_entropy_by_round: Dict[int, float] = {}
                    mean_agent_total_entropy_by_round: Dict[int, float] = {}
                    for round_num in round_agent_entropy_data.keys():
                        key_mean = f"sample_round_{round_num}_mean_agent_mean_entropy"
                        key_total = f"sample_round_{round_num}_mean_agent_total_entropy"
                        if key_mean in round_statistics:
                            mean_agent_mean_entropy_by_round[round_num] = float(
                                round_statistics[key_mean]
                            )
                        if key_total in round_statistics:
                            mean_agent_total_entropy_by_round[round_num] = float(
                                round_statistics[key_total]
                            )

                    if mean_agent_mean_entropy_by_round:
                        sorted_rounds_mean = sorted(mean_agent_mean_entropy_by_round.keys())
                        first_round_mean = sorted_rounds_mean[0]
                        last_round_mean = sorted_rounds_mean[-1]
                        first_mean = mean_agent_mean_entropy_by_round[first_round_mean]
                        last_mean = mean_agent_mean_entropy_by_round[last_round_mean]
                        sample_round_trend_features[
                            "sample_round_mean_agent_mean_entropy_first_last_diff"
                        ] = last_mean - first_mean
                        if first_mean > 0:
                            sample_round_trend_features[
                                "sample_round_mean_agent_mean_entropy_first_last_ratio"
                            ] = last_mean / first_mean
                        else:
                            sample_round_trend_features[
                                "sample_round_mean_agent_mean_entropy_first_last_ratio"
                            ] = 0.0

                        num_round_steps_mean = max(len(sorted_rounds_mean) - 1, 1)
                        sample_round_trend_features[
                            "sample_round_mean_agent_mean_entropy_slope_per_round"
                        ] = (last_mean - first_mean) / float(num_round_steps_mean)

                        mean_values = [
                            mean_agent_mean_entropy_by_round[r] for r in sorted_rounds_mean
                        ]
                        if len(mean_values) > 1:
                            sample_round_trend_features[
                                "sample_round_mean_agent_mean_entropy_volatility"
                            ] = float(np.std(np.array(mean_values)))
                        else:
                            sample_round_trend_features[
                                "sample_round_mean_agent_mean_entropy_volatility"
                            ] = 0.0

                        # Trend sign: -1 (decrease), 0 (flat), 1 (increase)
                        if last_mean > first_mean:
                            trend_sign = 1
                        elif last_mean < first_mean:
                            trend_sign = -1
                        else:
                            trend_sign = 0
                        sample_round_trend_features[
                            "sample_round_mean_agent_mean_entropy_trend_sign"
                        ] = trend_sign

                    if mean_agent_total_entropy_by_round:
                        sorted_rounds_total = sorted(
                            mean_agent_total_entropy_by_round.keys()
                        )
                        first_round_total = sorted_rounds_total[0]
                        last_round_total = sorted_rounds_total[-1]
                        first_total_mean = mean_agent_total_entropy_by_round[
                            first_round_total
                        ]
                        last_total_mean = mean_agent_total_entropy_by_round[
                            last_round_total
                        ]
                        sample_round_trend_features[
                            "sample_round_mean_agent_total_entropy_first_last_diff"
                        ] = last_total_mean - first_total_mean
                        if first_total_mean > 0:
                            sample_round_trend_features[
                                "sample_round_mean_agent_total_entropy_first_last_ratio"
                            ] = last_total_mean / first_total_mean
                        else:
                            sample_round_trend_features[
                                "sample_round_mean_agent_total_entropy_first_last_ratio"
                            ] = 0.0

                        num_round_steps_total = max(len(sorted_rounds_total) - 1, 1)
                        sample_round_trend_features[
                            "sample_round_mean_agent_total_entropy_slope_per_round"
                        ] = (last_total_mean - first_total_mean) / float(
                            num_round_steps_total
                        )

                        total_mean_values = [
                            mean_agent_total_entropy_by_round[r]
                            for r in sorted_rounds_total
                        ]
                        if len(total_mean_values) > 1:
                            sample_round_trend_features[
                                "sample_round_mean_agent_total_entropy_volatility"
                            ] = float(np.std(np.array(total_mean_values)))
                        else:
                            sample_round_trend_features[
                                "sample_round_mean_agent_total_entropy_volatility"
                            ] = 0.0

                    # Intra-round agent distribution features (mean and total entropy)
                    for round_num in round_agent_entropy_data.keys():
                        # Mean entropy across agents
                        max_mean = round_statistics.get(
                            f"sample_round_{round_num}_max_agent_mean_entropy", 0.0
                        )
                        min_mean = round_statistics.get(
                            f"sample_round_{round_num}_min_agent_mean_entropy", 0.0
                        )
                        mean_mean = round_statistics.get(
                            f"sample_round_{round_num}_mean_agent_mean_entropy", 0.0
                        )
                        std_mean = round_statistics.get(
                            f"sample_round_{round_num}_std_agent_mean_entropy", 0.0
                        )
                        q1_mean = round_statistics.get(
                            f"sample_round_{round_num}_q1_agent_mean_entropy", 0.0
                        )
                        q3_mean = round_statistics.get(
                            f"sample_round_{round_num}_q3_agent_mean_entropy", 0.0
                        )
                        median_mean = round_statistics.get(
                            f"sample_round_{round_num}_median_agent_mean_entropy", 0.0
                        )

                        spread_mean = float(max_mean) - float(min_mean)
                        round_statistics[
                            f"sample_round_{round_num}_agent_mean_entropy_spread"
                        ] = spread_mean

                        if mean_mean > 0:
                            round_statistics[
                                f"sample_round_{round_num}_agent_mean_entropy_cv"
                            ] = float(std_mean) / float(mean_mean)
                        else:
                            round_statistics[
                                f"sample_round_{round_num}_agent_mean_entropy_cv"
                            ] = 0.0

                        iqr_mean = float(q3_mean) - float(q1_mean)
                        if iqr_mean != 0:
                            bowley_mean = (
                                float(q3_mean)
                                + float(q1_mean)
                                - 2.0 * float(median_mean)
                            ) / iqr_mean
                        else:
                            bowley_mean = 0.0
                        round_statistics[
                            f"sample_round_{round_num}_agent_mean_entropy_bowley_skewness"
                        ] = bowley_mean

                        # Total entropy across agents
                        max_total = round_statistics.get(
                            f"sample_round_{round_num}_max_agent_total_entropy", 0.0
                        )
                        min_total = round_statistics.get(
                            f"sample_round_{round_num}_min_agent_total_entropy", 0.0
                        )
                        spread_total = float(max_total) - float(min_total)
                        round_statistics[
                            f"sample_round_{round_num}_agent_total_entropy_spread"
                        ] = spread_total

                    # Cross-round change in agent entropy spread (first vs last round)
                    if round_agent_entropy_data:
                        sorted_rounds_spread = sorted(round_agent_entropy_data.keys())
                        first_round_spread = sorted_rounds_spread[0]
                        last_round_spread = sorted_rounds_spread[-1]

                        first_spread_mean = round_statistics.get(
                            f"sample_round_{first_round_spread}_agent_mean_entropy_spread",
                            0.0,
                        )
                        last_spread_mean = round_statistics.get(
                            f"sample_round_{last_round_spread}_agent_mean_entropy_spread",
                            0.0,
                        )
                        sample_round_trend_features[
                            "sample_round_agent_mean_entropy_spread_first_last_diff"
                        ] = last_spread_mean - first_spread_mean

                        first_spread_total = round_statistics.get(
                            f"sample_round_{first_round_spread}_agent_total_entropy_spread",
                            0.0,
                        )
                        last_spread_total = round_statistics.get(
                            f"sample_round_{last_round_spread}_agent_total_entropy_spread",
                            0.0,
                        )
                        sample_round_trend_features[
                            "sample_round_agent_total_entropy_spread_first_last_diff"
                        ] = last_spread_total - first_spread_total

                    # Shared sample-level features used by all agent records of this sample
                    shared_sample_features: Dict[str, Any] = {}
                    shared_sample_features.update(round_statistics)
                    shared_sample_features.update(sample_round_dynamic_features)
                    shared_sample_features.update(sample_round_trend_features)
                    shared_sample_features.update(sample_shape_features)
                    shared_sample_features.update(sample_base_entropy_features)

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

                        record: Dict[str, Any] = {
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
                            "sample_total_entropy": sample_total_entropy,
                            "sample_max_entropy": sample_max_entropy,
                            "sample_min_entropy": sample_min_entropy,
                            "sample_mean_entropy": sample_mean_entropy,
                            "sample_median_entropy": sample_median_entropy,
                            "sample_std_entropy": sample_std_entropy,
                            "sample_variance_entropy": sample_variance_entropy,
                            "sample_q1_entropy": sample_q1_entropy,
                            "sample_q3_entropy": sample_q3_entropy,
                            "sample_num_agents": sample_num_agents,
                            "sample_all_agents_token_count": sample_all_agents_token_count,
                            "sample_avg_entropy_per_token": sample_avg_entropy_per_token,
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

                        # Add sample-level shared statistics to the record
                        record.update(shared_sample_features)

                        # Calculate derived features with numerical stability
                        sample_mean_entropy_local = sample_mean_entropy
                        sample_std_entropy_local = sample_std_entropy
                        sample_total_entropy_local = sample_total_entropy
                        sample_num_agents_local = sample_num_agents
                        agent_total_entropy_local = agent_entropy_data.get(
                            "total_entropy", 0
                        )

                        # sample_entropy_stability_index: 1 - (std / mean)
                        if sample_mean_entropy_local > 0:
                            record["sample_entropy_stability_index"] = 1 - (
                                sample_std_entropy_local / sample_mean_entropy_local
                            )
                        else:
                            record["sample_entropy_stability_index"] = 0.0

                        # agent_entropy_contribution: agent_total / sample_total
                        if sample_total_entropy_local > 0:
                            record["agent_entropy_contribution"] = (
                                agent_total_entropy_local / sample_total_entropy_local
                            )
                        else:
                            record["agent_entropy_contribution"] = 0.0

                        # sample_avg_entropy_per_agent: sample_total / sample_num_agents
                        if sample_num_agents_local > 0:
                            record["sample_avg_entropy_per_agent"] = (
                                sample_total_entropy_local / sample_num_agents_local
                            )
                        else:
                            record["sample_avg_entropy_per_agent"] = 0.0

                        records.append(record)

        return records

    @staticmethod
    def add_dynamic_round_features(
        records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add experiment-level dynamic round comparison features to records.

        This mirrors the original Aggregator.add_dynamic_round_features implementation.
        """
        # Group records by experiment to enable round comparisons
        exp_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            exp_name = record["experiment_name"]
            exp_records[exp_name].append(record)

        # For each experiment, calculate round comparison features
        for exp_name, exp_record_list in exp_records.items():
            # Collect round data for this experiment
            round_data: Dict[int, Dict[str, float]] = {}
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
