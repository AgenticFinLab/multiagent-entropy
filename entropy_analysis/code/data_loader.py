"""Data loading and preprocessing module for multi-agent system entropy analysis.

This module provides functionality for loading, preprocessing, and managing
experimental data from multi-agent system evaluations. It supports data
aggregation at different levels: experiment, sample, agent, and round.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class DataLoader:
    """Handles data loading, preprocessing, and management for entropy analysis.

    This class provides methods to load raw experimental data, preprocess it,
    and extract data at different granularity levels (experiment, sample, agent,
    round). It also supports filtering by architecture and generating summary
    statistics.

    Attributes:
        data_path: Path to the raw data CSV file.
        raw_data: DataFrame containing the raw loaded data.
        processed_data: DataFrame containing the preprocessed data.
        architectures: List of available architecture types.
    """

    def __init__(self, data_path: str):
        """Initialize the DataLoader with a data file path.

        Args:
            data_path: Path to the CSV file containing experimental data.
        """
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        self.architectures = ["centralized", "debate", "hybrid", "sequential", "single"]

    def load_data(self) -> pd.DataFrame:
        """Load raw data from the CSV file.

        Returns:
            DataFrame containing the raw experimental data.
        """
        self.raw_data = pd.read_csv(self.data_path)
        print(f"Data loaded successfully, {len(self.raw_data)} rows")
        return self.raw_data

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Categorize features into different groups based on their prefixes.

        Returns:
            Dictionary mapping feature group names to lists of feature names.
        """
        if self.raw_data is None:
            self.load_data()

        feature_groups = {
            "sample_features": [
                col for col in self.raw_data.columns if col.startswith("sample_")
            ],
            "agent_features": [
                col for col in self.raw_data.columns if col.startswith("agent_")
            ],
            "round_features": [
                col for col in self.raw_data.columns if col.startswith("round_")
            ],
            "exp_features": [
                col for col in self.raw_data.columns if col.startswith("exp_")
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

        return feature_groups

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the raw data for analysis.

        Converts data types, handles missing values, and ensures proper formatting
        for entropy features.

        Returns:
            DataFrame containing the preprocessed data.
        """
        if self.raw_data is None:
            self.load_data()

        df = self.raw_data.copy()

        df["architecture"] = df["architecture"].astype("category")
        df["is_finally_correct"] = df["is_finally_correct"].astype(bool)

        entropy_cols = [col for col in df.columns if "entropy" in col.lower()]
        for col in entropy_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print("Data preprocessing completed")
        print(f"Architecture types: {df['architecture'].unique()}")
        print(f"Number of samples: {df['sample_id'].nunique()}")
        print(f"Number of experiments: {df['experiment_name'].nunique()}")

        self.processed_data = df
        return df

    def get_experiment_level_data(self) -> pd.DataFrame:
        """Extract data at the experiment level.

        Returns:
            DataFrame containing experiment-level aggregated data.
        """
        if self.processed_data is None:
            self.preprocess_data()

        exp_cols = [
            "experiment_name",
            "architecture",
            "exp_accuracy",
            "exp_total_entropy",
            "exp_infer_average_entropy",
            "exp_num_inferences",
            "exp_total_time",
        ]

        exp_data = self.processed_data[exp_cols].drop_duplicates(
            subset=["experiment_name"]
        )
        return exp_data

    def get_sample_level_data(self) -> pd.DataFrame:
        """Extract data at the sample level.

        Returns:
            DataFrame containing sample-level aggregated data.
        """
        if self.processed_data is None:
            self.preprocess_data()

        sample_cols = [
            "sample_id",
            "experiment_name",
            "architecture",
            "num_rounds",
            "ground_truth",
            "is_finally_correct",
        ] + [col for col in self.processed_data.columns if col.startswith("sample_")]

        sample_data = self.processed_data[sample_cols].drop_duplicates(
            subset=["sample_id"]
        )
        return sample_data

    def get_agent_level_data(self) -> pd.DataFrame:
        """Extract data at the agent level.

        Returns:
            DataFrame containing agent-level data.
        """
        if self.processed_data is None:
            self.preprocess_data()

        agent_cols = [
            "sample_id",
            "experiment_name",
            "architecture",
            "agent_name",
            "agent_key",
            "agent_round_number",
            "is_finally_correct",
        ] + [col for col in self.processed_data.columns if col.startswith("agent_")]

        agent_data = self.processed_data[agent_cols]
        return agent_data

    def get_round_level_data(self) -> pd.DataFrame:
        """Extract data at the round level.

        Returns:
            DataFrame containing round-level aggregated data.
        """
        if self.processed_data is None:
            self.preprocess_data()

        round_cols = [
            "sample_id",
            "experiment_name",
            "architecture",
            "agent_round_number",
            "is_finally_correct",
        ] + [col for col in self.processed_data.columns if col.startswith("round_")]

        round_data = self.processed_data[round_cols].drop_duplicates(
            subset=["sample_id", "agent_round_number"]
        )
        return round_data

    def filter_by_architecture(self, architecture: str) -> pd.DataFrame:
        """Filter data by a specific architecture type.

        Args:
            architecture: The architecture type to filter by.

        Returns:
            DataFrame containing data for the specified architecture.

        Raises:
            ValueError: If the architecture type is not recognized.
        """
        if self.processed_data is None:
            self.preprocess_data()

        if architecture not in self.architectures:
            raise ValueError(
                f"Unknown architecture type: {architecture}. "
                f"Available options: {self.architectures}"
            )

        return self.processed_data[self.processed_data["architecture"] == architecture]

    def get_architecture_comparison(self) -> pd.DataFrame:
        """Generate comparison statistics across different architectures.

        Returns:
            DataFrame containing comparison metrics for each architecture.
        """
        if self.processed_data is None:
            self.preprocess_data()

        comparison = {}
        for arch in self.architectures:
            arch_data = self.filter_by_architecture(arch)
            comparison[arch] = {
                "sample_count": arch_data["sample_id"].nunique(),
                "experiment_count": arch_data["experiment_name"].nunique(),
                "avg_accuracy": arch_data["exp_accuracy"].mean(),
                "avg_entropy": arch_data["sample_mean_entropy"].mean(),
                "avg_token_count": arch_data["sample_all_agents_token_count"].mean(),
            }

        return pd.DataFrame(comparison).T

    def save_processed_data(self, output_path: str) -> None:
        """Save the processed data to a CSV file.

        Args:
            output_path: Path where the processed data should be saved.
        """
        if self.processed_data is None:
            self.preprocess_data()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

    def get_summary_statistics(self) -> Dict[str, object]:
        """Generate summary statistics for the dataset.

        Returns:
            Dictionary containing various summary statistics.
        """
        if self.processed_data is None:
            self.preprocess_data()

        summary = {
            "total_rows": len(self.processed_data),
            "unique_samples": self.processed_data["sample_id"].nunique(),
            "unique_experiments": self.processed_data["experiment_name"].nunique(),
            "architectures": self.processed_data["architecture"].unique().tolist(),
            "rounds": self.processed_data["num_rounds"].unique().tolist(),
            "accuracy_range": (
                self.processed_data["exp_accuracy"].min(),
                self.processed_data["exp_accuracy"].max(),
            ),
            "entropy_features": [
                col for col in self.processed_data.columns if "entropy" in col.lower()
            ],
        }

        return summary


if __name__ == "__main__":
    data_path = (
        "/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated/"
        "aggregated_data.csv"
    )

    loader = DataLoader(data_path)

    raw_data = loader.load_data()
    print("\nFirst 5 rows of raw data:")
    print(raw_data.head())

    processed_data = loader.preprocess_data()
    print("\nProcessed data info:")
    print(processed_data.info())

    feature_groups = loader.get_feature_groups()
    print("\nFeature groups:")
    for group, features in feature_groups.items():
        print(f"{group}: {len(features)} features")

    summary = loader.get_summary_statistics()
    print("\nData summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    comparison = loader.get_architecture_comparison()
    print("\nArchitecture comparison:")
    print(comparison)

    loader.save_processed_data(
        "/home/yuxuanzhao/multiagent-entropy/entropy_analysis/results/processed_data.csv"
    )
