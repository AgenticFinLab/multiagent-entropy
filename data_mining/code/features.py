# Experiment identifier features
EXPERIMENT_IDENTIFIER = [
    "model_name",
    "sample_id",
    "num_rounds",
]

# Sample identifier
SAMPLE_IDENTIFIER = [
    "is_finally_correct",
    "final_format_compliance",
]

# Experiment statistics
EXPERIMENT_METRICS = [
    "exp_num_inferences",
    "exp_accuracy",
    "exp_format_compliance_rate",
]

# Exclude identifiers and statistics that would leak target information
# ALL experiment will exclude this feature group
DEFAULT_EXCLUDE_COLUMNS = (
    EXPERIMENT_IDENTIFIER 
    + SAMPLE_IDENTIFIER 
    + EXPERIMENT_METRICS
)

# Base model metrics
BASE_MODEL_METRICS_EXPERIMENT_LEVEL = [
    "base_model_accuracy",
    "base_model_format_compliance_rate",
]

BASE_MODEL_METRICS_SAMPLE_LEVEL = [
    "base_model_is_finally_correct",
    "base_model_format_compliance",
]

# Sample-level baseline entropy features relative to single-agent base model
SAMPLE_BASELINE_ENTROPY = [
    "base_sample_total_entropy",
    "base_sample_token_count",
    "base_sample_avg_entropy_per_token",
    "sample_entropy_ratio_vs_base_total",
    "sample_entropy_reduction_vs_base_total",
    "sample_avg_entropy_per_token_ratio_vs_base",
    "sample_avg_entropy_per_token_diff_vs_base",
    # new: token entropy statistics of base model finally predicted answer
    "base_model_answer_token_count",
    "base_model_max_answer_token_entropy",
    "base_model_mean_answer_token_entropy",
    "base_model_min_answer_token_entropy",
    "base_model_std_answer_token_entropy",
    "base_model_median_answer_token_entropy",
    "base_model_vs_sample_final_answer_entropy_diff",
    "base_model_vs_sample_final_answer_entropy_ratio",
    "answer_token_entropy_change",
    "answer_token_entropy_change_direction",
]

# Base model metrics without entropy, means exclude the accuracy metrics of base model in the experiment, remain the entropy metrics of base model in the experiment
BASE_MODEL_WO_ENTROPY = (
    DEFAULT_EXCLUDE_COLUMNS
    + BASE_MODEL_METRICS_EXPERIMENT_LEVEL
    + BASE_MODEL_METRICS_SAMPLE_LEVEL
)

# Base model metrics with entropy, means exclude all the metrics of base model in the experiment
BASE_MODEL_ALL_METRICS = BASE_MODEL_WO_ENTROPY + SAMPLE_BASELINE_ENTROPY

EXPERIMENT_STATISTICS = [
    "exp_total_entropy",
    "exp_infer_average_entropy",
    "exp_total_time",
    "exp_total_token",
    "architecture",
]

# Round statistics
ROUND_STATISTICS = [
    "round_1_2_change_entropy",
    "round_1_2_change_tokens",
    "round_1_infer_avg_entropy",
    "round_1_num_inferences",
    "round_1_total_entropy",
    "round_1_total_time",
    "round_1_total_token",
    "round_2_infer_avg_entropy",
    "round_2_num_inferences",
    "round_2_total_entropy",
    "round_2_total_time",
    "round_2_total_token",
]

# Sample statistics
SAMPLE_STATISTICS = [
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
    # new: token entropy statistics of MAS finally predicted answer
    "sample_answer_token_count",
    "sample_max_answer_token_entropy",
    "sample_mean_answer_token_entropy",
    "sample_min_answer_token_entropy",
    "sample_std_answer_token_entropy",
    "sample_median_answer_token_entropy",
]


# Sample distribution shape features
SAMPLE_DISTRIBUTION_SHAPE = [
    "sample_entropy_range",
    "sample_entropy_iqr",
    "sample_entropy_relative_iqr_mean",
    "sample_entropy_relative_iqr_range",
    "sample_entropy_bowley_skewness",
    "sample_entropy_median_over_mean",
    "sample_entropy_tail_weight",
    "sample_entropy_cv",
]


# Aggregation over agents
AGGREGATION_OVER_AGENTS = [
    "sample_avg_entropy_per_agent",
]


# Sample-level round-wise aggregated features over all agents
SAMPLE_ROUND_WISE_AGGREGATED = [
    "sample_round_1_all_agents_total_entropy",
    "sample_round_1_all_agents_total_token",
    "sample_round_1_all_agents_entropy_per_token",
    "sample_round_2_all_agents_total_entropy",
    "sample_round_2_all_agents_total_token",
    "sample_round_2_all_agents_entropy_per_token",
    "sample_round_all_agents_total_entropy_first_last_diff",
    "sample_round_all_agents_total_entropy_first_last_ratio",
    "sample_round_all_agents_total_token_first_last_diff",
    "sample_round_all_agents_total_token_first_last_ratio",
    "sample_round_all_agents_entropy_per_token_first_last_diff",
    "sample_round_all_agents_entropy_per_token_first_last_ratio",
    "sample_round_all_agents_entropy_per_token_slope_per_round",
    "sample_round_all_agents_entropy_per_token_volatility",
    "sample_round_1_2_change_tokens",
    "sample_round_1_2_change_entropy",
]


# Cross-round features based on aggregated agent mean and total entropy
CROSS_ROUND_AGGREGATED = [
    "sample_round_mean_agent_mean_entropy_first_last_diff",
    "sample_round_mean_agent_mean_entropy_first_last_ratio",
    "sample_round_mean_agent_mean_entropy_slope_per_round",
    "sample_round_mean_agent_mean_entropy_volatility",
    "sample_round_mean_agent_mean_entropy_trend_sign",
    "sample_round_mean_agent_total_entropy_first_last_diff",
    "sample_round_mean_agent_total_entropy_first_last_ratio",
    "sample_round_mean_agent_total_entropy_slope_per_round",
    "sample_round_mean_agent_total_entropy_volatility",
]


# Intra-round agent distribution features (per round)
INTRA_ROUND_AGENT_DISTRIBUTION = [
    "sample_round_1_agent_mean_entropy_spread",
    "sample_round_1_agent_mean_entropy_cv",
    "sample_round_1_agent_mean_entropy_bowley_skewness",
    "sample_round_1_agent_total_entropy_spread",
    "sample_round_2_agent_mean_entropy_spread",
    "sample_round_2_agent_mean_entropy_cv",
    "sample_round_2_agent_mean_entropy_bowley_skewness",
    "sample_round_2_agent_total_entropy_spread",
]


# Cross-round change of agent entropy spread
CROSS_ROUND_AGENT_SPREAD_CHANGE = [
    "sample_round_agent_mean_entropy_spread_first_last_diff",
    "sample_round_agent_total_entropy_spread_first_last_diff",
]


# Sample round1 statistics based on every agent statistics
SAMPLE_ROUND1_AGENT_STATISTICS = [
    "sample_round_1_max_agent_max_entropy",
    "sample_round_1_max_agent_mean_entropy",
    "sample_round_1_max_agent_median_entropy",
    "sample_round_1_max_agent_min_entropy",
    "sample_round_1_max_agent_q1_entropy",
    "sample_round_1_max_agent_q3_entropy",
    "sample_round_1_max_agent_std_entropy",
    "sample_round_1_max_agent_total_entropy",
    "sample_round_1_max_agent_variance_entropy",
    "sample_round_1_mean_agent_max_entropy",
    "sample_round_1_mean_agent_mean_entropy",
    "sample_round_1_mean_agent_median_entropy",
    "sample_round_1_mean_agent_min_entropy",
    "sample_round_1_mean_agent_q1_entropy",
    "sample_round_1_mean_agent_q3_entropy",
    "sample_round_1_mean_agent_std_entropy",
    "sample_round_1_mean_agent_total_entropy",
    "sample_round_1_mean_agent_variance_entropy",
    "sample_round_1_median_agent_max_entropy",
    "sample_round_1_median_agent_mean_entropy",
    "sample_round_1_median_agent_median_entropy",
    "sample_round_1_median_agent_min_entropy",
    "sample_round_1_median_agent_q1_entropy",
    "sample_round_1_median_agent_q3_entropy",
    "sample_round_1_median_agent_std_entropy",
    "sample_round_1_median_agent_total_entropy",
    "sample_round_1_median_agent_variance_entropy",
    "sample_round_1_min_agent_max_entropy",
    "sample_round_1_min_agent_mean_entropy",
    "sample_round_1_min_agent_median_entropy",
    "sample_round_1_min_agent_min_entropy",
    "sample_round_1_min_agent_q1_entropy",
    "sample_round_1_min_agent_q3_entropy",
    "sample_round_1_min_agent_std_entropy",
    "sample_round_1_min_agent_total_entropy",
    "sample_round_1_min_agent_variance_entropy",
    "sample_round_1_q1_agent_max_entropy",
    "sample_round_1_q1_agent_mean_entropy",
    "sample_round_1_q1_agent_median_entropy",
    "sample_round_1_q1_agent_min_entropy",
    "sample_round_1_q1_agent_q1_entropy",
    "sample_round_1_q1_agent_q3_entropy",
    "sample_round_1_q1_agent_std_entropy",
    "sample_round_1_q1_agent_total_entropy",
    "sample_round_1_q1_agent_variance_entropy",
    "sample_round_1_q3_agent_max_entropy",
    "sample_round_1_q3_agent_mean_entropy",
    "sample_round_1_q3_agent_median_entropy",
    "sample_round_1_q3_agent_min_entropy",
    "sample_round_1_q3_agent_q1_entropy",
    "sample_round_1_q3_agent_q3_entropy",
    "sample_round_1_q3_agent_std_entropy",
    "sample_round_1_q3_agent_total_entropy",
    "sample_round_1_q3_agent_variance_entropy",
    "sample_round_1_std_agent_max_entropy",
    "sample_round_1_std_agent_mean_entropy",
    "sample_round_1_std_agent_median_entropy",
    "sample_round_1_std_agent_min_entropy",
    "sample_round_1_std_agent_q1_entropy",
    "sample_round_1_std_agent_q3_entropy",
    "sample_round_1_std_agent_std_entropy",
    "sample_round_1_std_agent_total_entropy",
    "sample_round_1_std_agent_variance_entropy",
    "sample_round_1_variance_agent_max_entropy",
    "sample_round_1_variance_agent_mean_entropy",
    "sample_round_1_variance_agent_median_entropy",
    "sample_round_1_variance_agent_min_entropy",
    "sample_round_1_variance_agent_q1_entropy",
    "sample_round_1_variance_agent_q3_entropy",
    "sample_round_1_variance_agent_std_entropy",
    "sample_round_1_variance_agent_total_entropy",
    "sample_round_1_variance_agent_variance_entropy",
]


# Sample round2 statistics based on every agent statistics
SAMPLE_ROUND2_AGENT_STATISTICS = [
    "sample_round_2_max_agent_max_entropy",
    "sample_round_2_max_agent_mean_entropy",
    "sample_round_2_max_agent_median_entropy",
    "sample_round_2_max_agent_min_entropy",
    "sample_round_2_max_agent_q1_entropy",
    "sample_round_2_max_agent_q3_entropy",
    "sample_round_2_max_agent_std_entropy",
    "sample_round_2_max_agent_total_entropy",
    "sample_round_2_max_agent_variance_entropy",
    "sample_round_2_mean_agent_max_entropy",
    "sample_round_2_mean_agent_mean_entropy",
    "sample_round_2_mean_agent_median_entropy",
    "sample_round_2_mean_agent_min_entropy",
    "sample_round_2_mean_agent_q1_entropy",
    "sample_round_2_mean_agent_q3_entropy",
    "sample_round_2_mean_agent_std_entropy",
    "sample_round_2_mean_agent_total_entropy",
    "sample_round_2_mean_agent_variance_entropy",
    "sample_round_2_median_agent_max_entropy",
    "sample_round_2_median_agent_mean_entropy",
    "sample_round_2_median_agent_median_entropy",
    "sample_round_2_median_agent_min_entropy",
    "sample_round_2_median_agent_q1_entropy",
    "sample_round_2_median_agent_q3_entropy",
    "sample_round_2_median_agent_std_entropy",
    "sample_round_2_median_agent_total_entropy",
    "sample_round_2_median_agent_variance_entropy",
    "sample_round_2_min_agent_max_entropy",
    "sample_round_2_min_agent_mean_entropy",
    "sample_round_2_min_agent_median_entropy",
    "sample_round_2_min_agent_min_entropy",
    "sample_round_2_min_agent_q1_entropy",
    "sample_round_2_min_agent_q3_entropy",
    "sample_round_2_min_agent_std_entropy",
    "sample_round_2_min_agent_total_entropy",
    "sample_round_2_min_agent_variance_entropy",
    "sample_round_2_q1_agent_max_entropy",
    "sample_round_2_q1_agent_mean_entropy",
    "sample_round_2_q1_agent_median_entropy",
    "sample_round_2_q1_agent_min_entropy",
    "sample_round_2_q1_agent_q1_entropy",
    "sample_round_2_q1_agent_q3_entropy",
    "sample_round_2_q1_agent_std_entropy",
    "sample_round_2_q1_agent_total_entropy",
    "sample_round_2_q1_agent_variance_entropy",
    "sample_round_2_q3_agent_max_entropy",
    "sample_round_2_q3_agent_mean_entropy",
    "sample_round_2_q3_agent_median_entropy",
    "sample_round_2_q3_agent_min_entropy",
    "sample_round_2_q3_agent_q1_entropy",
    "sample_round_2_q3_agent_q3_entropy",
    "sample_round_2_q3_agent_std_entropy",
    "sample_round_2_q3_agent_total_entropy",
    "sample_round_2_q3_agent_variance_entropy",
    "sample_round_2_std_agent_max_entropy",
    "sample_round_2_std_agent_mean_entropy",
    "sample_round_2_std_agent_median_entropy",
    "sample_round_2_std_agent_min_entropy",
    "sample_round_2_std_agent_q1_entropy",
    "sample_round_2_std_agent_q3_entropy",
    "sample_round_2_std_agent_std_entropy",
    "sample_round_2_std_agent_total_entropy",
    "sample_round_2_std_agent_variance_entropy",
    "sample_round_2_variance_agent_max_entropy",
    "sample_round_2_variance_agent_mean_entropy",
    "sample_round_2_variance_agent_median_entropy",
    "sample_round_2_variance_agent_min_entropy",
    "sample_round_2_variance_agent_q1_entropy",
    "sample_round_2_variance_agent_q3_entropy",
    "sample_round_2_variance_agent_std_entropy",
    "sample_round_2_variance_agent_total_entropy",
    "sample_round_2_variance_agent_variance_entropy",
]

# All features combined
ALL_FEATURES = (
    DEFAULT_EXCLUDE_COLUMNS
    + BASE_MODEL_METRICS_EXPERIMENT_LEVEL
    + BASE_MODEL_METRICS_SAMPLE_LEVEL
    + EXPERIMENT_STATISTICS
    + ROUND_STATISTICS
    + SAMPLE_STATISTICS
    + SAMPLE_DISTRIBUTION_SHAPE
    + SAMPLE_BASELINE_ENTROPY
    + AGGREGATION_OVER_AGENTS
    + SAMPLE_ROUND_WISE_AGGREGATED
    + CROSS_ROUND_AGGREGATED
    + INTRA_ROUND_AGENT_DISTRIBUTION
    + CROSS_ROUND_AGENT_SPREAD_CHANGE
    + SAMPLE_ROUND1_AGENT_STATISTICS
    + SAMPLE_ROUND2_AGENT_STATISTICS
)

# Define feature groups for easy access
FEATURE_GROUPS = {
    "default": DEFAULT_EXCLUDE_COLUMNS,
    "base_model_wo_entropy": BASE_MODEL_WO_ENTROPY,
    "base_model_all_metrics": BASE_MODEL_ALL_METRICS,
    "experiment_identifier": EXPERIMENT_IDENTIFIER,
    "sample_identifier": SAMPLE_IDENTIFIER,
    "experiment_statistics": EXPERIMENT_STATISTICS,
    "round_statistics": ROUND_STATISTICS,
    "sample_statistics": SAMPLE_STATISTICS,
    "sample_distribution_shape": SAMPLE_DISTRIBUTION_SHAPE,
    "sample_baseline_entropy": SAMPLE_BASELINE_ENTROPY,
    "aggregation_over_agents": AGGREGATION_OVER_AGENTS,
    "sample_round_wise_aggregated": SAMPLE_ROUND_WISE_AGGREGATED,
    "cross_round_aggregated": CROSS_ROUND_AGGREGATED,
    "intra_round_agent_distribution": INTRA_ROUND_AGENT_DISTRIBUTION,
    "cross_round_agent_spread_change": CROSS_ROUND_AGENT_SPREAD_CHANGE,
    "sample_round1_agent_statistics": SAMPLE_ROUND1_AGENT_STATISTICS,
    "sample_round2_agent_statistics": SAMPLE_ROUND2_AGENT_STATISTICS,
    "all_features": ALL_FEATURES,
}
