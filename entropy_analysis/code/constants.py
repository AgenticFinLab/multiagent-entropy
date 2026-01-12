"""Constants module for multi-agent entropy analysis.

This module defines shared constants used across the entropy analysis system,
including architecture types, feature groups, and configuration parameters.
"""

ARCHITECTURES = ["centralized", "debate", "hybrid", "sequential", "single"]

MULTI_AGENT_ARCHITECTURES = ["centralized", "debate", "hybrid", "sequential"]

SINGLE_AGENT_ARCHITECTURES = ["single"]

FEATURE_GROUPS = {
    "sample_features": [
        "sample_id",
        "sample_mean_entropy",
        "sample_std_entropy",
        "sample_min_entropy",
        "sample_max_entropy",
        "sample_all_agents_token_count",
    ],
    "agent_features": [
        "agent_name",
        "agent_key",
        "agent_round_number",
        "agent_mean_entropy",
        "agent_std_entropy",
        "agent_token_count",
    ],
    "round_features": [
        "agent_round_number",
        "round_mean_entropy",
        "round_std_entropy",
        "round_token_count",
    ],
    "exp_features": [
        "experiment_name",
        "exp_accuracy",
        "exp_total_entropy",
        "exp_infer_average_entropy",
        "exp_num_inferences",
        "exp_total_time",
    ],
    "metadata": [
        "sample_id",
        "experiment_name",
        "architecture",
        "num_rounds",
        "ground_truth",
        "agent_name",
        "agent_key",
        "execution_order",
        "time_cost",
        "final_predicted_answer",
        "is_finally_correct",
    ],
}

ENTROPY_FEATURES = [
    "sample_mean_entropy",
    "sample_std_entropy",
    "sample_min_entropy",
    "sample_max_entropy",
    "agent_mean_entropy",
    "agent_std_entropy",
    "round_mean_entropy",
    "round_std_entropy",
    "exp_total_entropy",
    "exp_infer_average_entropy",
]

METADATA_COLUMNS = [
    "sample_id",
    "experiment_name",
    "architecture",
    "num_rounds",
    "ground_truth",
    "agent_name",
    "agent_key",
    "execution_order",
    "time_cost",
    "final_predicted_answer",
    "is_finally_correct",
    "model_name",
    "base_model_predicted_answer",
    "base_model_is_finally_correct",
    "base_model_format_compliance",
    "base_model_accuracy",
    "base_model_format_compliance_rate",
]

BASE_MODEL_COLUMNS = [
    "base_model_predicted_answer",
    "base_model_is_finally_correct",
    "base_model_format_compliance",
    "base_model_accuracy",
    "base_model_format_compliance_rate",
]
