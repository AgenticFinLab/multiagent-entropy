"""Utility functions for multi-agent entropy analysis.

This module provides shared utility functions used across the entropy analysis
system, including data preprocessing, feature grouping, and statistical analysis.
"""

from typing import Dict, List

import pandas as pd

from constants import ARCHITECTURES, METADATA_COLUMNS, BASE_MODEL_COLUMNS


def get_feature_groups(data_columns: List[str]) -> Dict[str, List[str]]:
    """Categorize features into different groups based on their prefixes.

    Args:
        data_columns: List of column names from the data.

    Returns:
        Dictionary mapping feature group names to lists of feature names.
    """
    feature_groups = {
        "sample_features": [col for col in data_columns if col.startswith("sample_")],
        "agent_features": [col for col in data_columns if col.startswith("agent_")],
        "round_features": [col for col in data_columns if col.startswith("round_")],
        "exp_features": [col for col in data_columns if col.startswith("exp_")],
        "metadata": [col for col in data_columns if col in METADATA_COLUMNS],
    }

    return feature_groups


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for analysis.

    Converts data types, handles missing values, and ensures proper formatting
    for entropy features.

    Args:
        data: DataFrame containing raw experimental data.

    Returns:
        DataFrame containing preprocessed data.
    """
    df = data.copy()

    if "architecture" in df.columns:
        df["architecture"] = df["architecture"].astype("category")

    if "is_finally_correct" in df.columns:
        df["is_finally_correct"] = df["is_finally_correct"].astype(bool)

    entropy_cols = [col for col in df.columns if "entropy" in col.lower()]
    for col in entropy_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_summary_statistics(data: pd.DataFrame) -> Dict[str, object]:
    """Generate summary statistics for the data.

    Args:
        data: DataFrame containing experimental data.

    Returns:
        Dictionary containing summary statistics.
    """
    summary = {
        "total_rows": len(data),
        "unique_samples": data["sample_id"].nunique(),
        "unique_experiments": data["experiment_name"].nunique(),
    }

    if "architecture" in data.columns:
        summary["architectures"] = data["architecture"].unique().tolist()

    if "model_name" in data.columns:
        summary["models"] = data["model_name"].unique().tolist()

    if "num_rounds" in data.columns:
        summary["rounds"] = data["num_rounds"].unique().tolist()

    if "exp_accuracy" in data.columns:
        summary["accuracy_range"] = (
            data["exp_accuracy"].min(),
            data["exp_accuracy"].max(),
        )

    if "base_model_accuracy" in data.columns:
        summary["base_model_accuracy_range"] = (
            data["base_model_accuracy"].min(),
            data["base_model_accuracy"].max(),
        )

    summary["entropy_features"] = [
        col for col in data.columns if "entropy" in col.lower()
    ]

    summary["base_model_features"] = [
        col for col in data.columns if col in BASE_MODEL_COLUMNS
    ]

    return summary


def get_architecture_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Generate comparison statistics across architectures.

    Args:
        data: DataFrame containing experimental data.

    Returns:
        DataFrame containing comparison metrics for each architecture.
    """
    comparison = {}

    for arch in ARCHITECTURES:
        arch_data = data[data["architecture"] == arch]
        if len(arch_data) > 0:
            comparison[arch] = {
                "sample_count": arch_data["sample_id"].nunique(),
                "experiment_count": arch_data["experiment_name"].nunique(),
            }

            if "exp_accuracy" in arch_data.columns:
                comparison[arch]["avg_accuracy"] = arch_data["exp_accuracy"].mean()

            if "sample_mean_entropy" in arch_data.columns:
                comparison[arch]["avg_entropy"] = arch_data[
                    "sample_mean_entropy"
                ].mean()

            if "sample_all_agents_token_count" in arch_data.columns:
                comparison[arch]["avg_token_count"] = arch_data[
                    "sample_all_agents_token_count"
                ].mean()

            if "round_total_time" in arch_data.columns:
                comparison[arch]["avg_round_time"] = arch_data[
                    "round_total_time"
                ].mean()

            if "round_total_token" in arch_data.columns:
                comparison[arch]["avg_round_token"] = arch_data[
                    "round_total_token"
                ].mean()

            if "exp_total_token" in arch_data.columns:
                comparison[arch]["avg_exp_token"] = arch_data[
                    "exp_total_token"
                ].mean()

            if "exp_total_time" in arch_data.columns:
                comparison[arch]["avg_exp_time"] = arch_data[
                    "exp_total_time"
                ].mean()

    return pd.DataFrame(comparison).T


def calculate_metrics_from_data(data: pd.DataFrame) -> Dict:
    """Calculate metrics from loaded data.

    Args:
        data: DataFrame containing experimental data.

    Returns:
        Dictionary containing calculated metrics.
    """
    metrics = {
        "total_samples": data["sample_id"].nunique(),
    }

    if "exp_accuracy" in data.columns:
        metrics["accuracy"] = data["exp_accuracy"].mean()
        metrics["std_accuracy"] = data["exp_accuracy"].std()

    if "sample_mean_entropy" in data.columns:
        metrics["mean_entropy"] = data["sample_mean_entropy"].mean()

    if "sample_std_entropy" in data.columns:
        metrics["std_entropy"] = data["sample_std_entropy"].mean()

    if "architecture" in data.columns:
        metrics["architectures"] = data["architecture"].unique().tolist()

    if "base_model_accuracy" in data.columns:
        metrics["base_model_accuracy"] = data["base_model_accuracy"].mean()
        metrics["base_model_std_accuracy"] = data["base_model_accuracy"].std()

    if "base_model_format_compliance_rate" in data.columns:
        metrics["base_model_format_compliance_rate"] = data[
            "base_model_format_compliance_rate"
        ].mean()

    return metrics
