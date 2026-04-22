#!/usr/bin/env python3
"""
Data Collection Module for Multi-Agent Entropy Analysis

This module collects and merges data from multiple datasets for comprehensive analysis.
It reads all_aggregated_data_exclude_agent.csv files from different dataset folders and combines them into a single dataset with dataset identification.

Supports two dataset types:
- 'standard': Standard benchmark datasets (GPQA, AIME, etc.)
- 'finagent': Financial agent evaluation dataset with dynamic step entropy columns
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal

import pandas as pd

from features import discover_step_entropy_features

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and merges data from multiple dataset folders."""

    def __init__(
        self,
        base_dir: str,
        target_datasets: Optional[List[str]] = None,
        dataset_type: Literal["standard", "finagent", "gaia"] = "standard",
    ):
        """
        Initialize the DataCollector.

        Args:
            base_dir: Base directory containing dataset folders
            target_datasets: Optional list of specific dataset names to collect.
                           If None, discovers all available datasets automatically.
            dataset_type: Type of dataset to collect:
                         - 'standard': Standard benchmark datasets (default)
                         - 'finagent': Financial agent dataset with dynamic step entropy columns
                         - 'gaia': GAIA benchmark dataset
        """
        self.base_dir = Path(base_dir)
        self.target_datasets = target_datasets
        self.dataset_type = dataset_type
        self.datasets = []
        self.merged_data = None

    def _get_csv_path(self, dataset_name: str) -> Path:
        """
        Get the CSV file path for a dataset based on dataset_type.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Path to the CSV file
        """
        if self.dataset_type == "finagent":
            # finagent has a fixed path structure
            return self.base_dir / "finagent" / "all_aggregated_data_exclude_agent.csv"
        elif self.dataset_type == "gaia":
            # gaia has a fixed path structure
            return self.base_dir / "gaia" / "all_aggregated_data_exclude_agent.csv"
        else:
            # standard datasets use subdirectory structure
            return (
                self.base_dir / dataset_name / "all_aggregated_data_exclude_agent.csv"
            )

    def discover_datasets(self) -> List[str]:
        """
        Discover dataset folders in the base directory.
        If target_datasets is specified, only validates and uses those datasets.
        Otherwise, discovers all available datasets.

        For finagent mode, only returns ['finagent'] if the data file exists.

        Returns:
            List of dataset folder names
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

        # Special handling for finagent dataset type
        if self.dataset_type == "finagent":
            csv_file = self._get_csv_path("finagent")
            if csv_file.exists():
                self.datasets = ["finagent"]
                logger.info(f"Found finagent dataset: {csv_file}")
            else:
                logger.warning(f"finagent CSV not found: {csv_file}")
                self.datasets = []
            return self.datasets

        # Special handling for gaia dataset type
        if self.dataset_type == "gaia":
            csv_file = self._get_csv_path("gaia")
            if csv_file.exists():
                self.datasets = ["gaia"]
                logger.info(f"Found gaia dataset: {csv_file}")
            else:
                logger.warning(f"gaia CSV not found: {csv_file}")
                self.datasets = []
            return self.datasets

        # Standard dataset discovery
        if self.target_datasets:
            datasets = []
            for dataset_name in self.target_datasets:
                csv_file = self._get_csv_path(dataset_name)

                if not csv_file.parent.exists():
                    logger.warning(f"Target dataset folder not found: {dataset_name}")
                elif not csv_file.exists():
                    logger.warning(f"Target dataset CSV not found: {dataset_name}")
                else:
                    datasets.append(dataset_name)
                    logger.info(f"Found target dataset: {dataset_name}")

            self.datasets = datasets
        else:
            # Find all subdirectories that contain the target CSV file
            datasets = []
            for item in self.base_dir.iterdir():
                if item.is_dir():
                    csv_file = item / "all_aggregated_data_exclude_agent.csv"
                    if csv_file.exists():
                        datasets.append(item.name)
                        logger.info(f"Found dataset: {item.name}")

            self.datasets = sorted(datasets)

        return self.datasets

    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load a single dataset CSV file.

        Args:
            dataset_name: Name of the dataset folder

        Returns:
            DataFrame containing the dataset data, or None if file not found
        """
        csv_path = self._get_csv_path(dataset_name)

        if not csv_path.exists():
            logger.warning(f"CSV file not found for dataset {dataset_name}: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            logger.info(
                f"Loaded dataset {dataset_name}: {len(df)} records, {len(df.columns)} columns"
            )

            # Log step entropy columns if present (finagent and gaia)
            if self.dataset_type in ("finagent", "gaia"):
                step_cols = discover_step_entropy_features(df.columns)
                if step_cols:
                    logger.info(
                        f"Found {len(step_cols)} step entropy columns: {step_cols[0]}...{step_cols[-1]}"
                    )

            return df
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            return None

    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all discovered datasets into a single DataFrame.

        Handles dynamic columns (e.g., step_N_mean_entropy) by filling missing
        values with NaN. Preserves finagent-specific fields when present.

        Returns:
            Merged DataFrame with dataset identification column
        """
        if not self.datasets:
            self.discover_datasets()

        if not self.datasets:
            raise ValueError("No datasets found to merge")

        all_dataframes = []

        for dataset_name in self.datasets:
            df = self.load_dataset(dataset_name)
            if df is not None:
                # Add dataset identification column
                df = df.copy()
                df.insert(0, "dataset", dataset_name)
                all_dataframes.append(df)

        if not all_dataframes:
            raise ValueError("No valid datasets could be loaded")

        # Collect all unique columns across dataframes for proper alignment
        # This handles dynamic step_N_mean_entropy columns from finagent
        all_columns = set()
        for df in all_dataframes:
            all_columns.update(df.columns)

        # Discover step entropy columns for logging
        step_entropy_cols = discover_step_entropy_features(list(all_columns))
        if step_entropy_cols:
            logger.info(
                f"Found {len(step_entropy_cols)} unique step entropy columns across all datasets"
            )

        # Merge all dataframes (pd.concat automatically fills missing columns with NaN)
        self.merged_data = pd.concat(all_dataframes, ignore_index=True)
        logger.info(
            f"Merged {len(all_dataframes)} datasets: {len(self.merged_data)} total records, {len(self.merged_data.columns)} columns"
        )

        return self.merged_data

    def save_merged_data(self, output_path: str = None) -> str:
        """
        Save the merged dataset to a CSV file.

        Args:
            output_path: Path to save the merged data. If None, uses default path.
                        If target_datasets contains single dataset, saves to
                        data_mining/results/{dataset_name}/merged_datasets.csv

        Returns:
            Path where the file was saved
        """
        if self.merged_data is None:
            self.merge_datasets()

        if output_path is None:
            # If single target dataset specified, save to its results folder
            if self.target_datasets and len(self.target_datasets) == 1:
                output_path = (
                    f"data_mining/results/{self.target_datasets[0]}/merged_datasets.csv"
                )
            else:
                output_path = "data_mining/data/merged_datasets.csv"

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        self.merged_data.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to: {output_path}")

        return output_path

    def get_data_summary(self) -> Dict:
        """
        Get summary statistics about the merged data.

        Returns:
            Dictionary containing summary information
        """
        if self.merged_data is None:
            self.merge_datasets()

        summary = {
            "total_records": len(self.merged_data),
            "total_columns": len(self.merged_data.columns),
            "datasets": list(self.merged_data["dataset"].unique()),
            "dataset_counts": self.merged_data["dataset"].value_counts().to_dict(),
            "columns": list(self.merged_data.columns),
        }

        return summary


def collect_standard_data(
    base_dir: str = "evaluation/results_all",
    target_datasets: Optional[List[str]] = None,
) -> DataCollector:
    """
    Collect standard benchmark datasets.

    Args:
        base_dir: Base directory containing dataset folders
        target_datasets: Optional list of specific datasets to collect

    Returns:
        Configured DataCollector instance with loaded data
    """
    collector = DataCollector(
        base_dir=base_dir,
        target_datasets=target_datasets,
        dataset_type="standard",
    )
    collector.discover_datasets()
    collector.merge_datasets()
    return collector


def collect_finagent_data(
    base_dir: str = "evaluation/results_finagent",
) -> DataCollector:
    """
    Collect finagent evaluation data.

    The finagent data includes:
    - evaluation_score (float 0-1)
    - num_steps, entropy_decay_rate, first_step_mean_entropy, last_step_mean_entropy
    - step_0_mean_entropy, step_1_mean_entropy, ... (dynamic columns)

    Args:
        base_dir: Base directory containing finagent results

    Returns:
        Configured DataCollector instance with loaded data
    """
    collector = DataCollector(
        base_dir=base_dir,
        dataset_type="finagent",
    )
    collector.discover_datasets()
    collector.merge_datasets()
    return collector


def collect_gaia_data(
    base_dir: str = "evaluation/results_gaia",
) -> DataCollector:
    """
    Collect GAIA benchmark evaluation data.

    Args:
        base_dir: Base directory containing gaia results

    Returns:
        Configured DataCollector instance with loaded data
    """
    collector = DataCollector(
        base_dir=base_dir,
        dataset_type="gaia",
    )
    collector.discover_datasets()
    collector.merge_datasets()
    return collector


def main(dataset_type: str = "standard"):
    """Main function to execute data collection and merging.

    Args:
        dataset_type: Type of data to collect ('standard' or 'finagent')
    """
    logger.info(f"Starting data collection and merging process (type={dataset_type})")

    if dataset_type == "finagent":
        collector = collect_finagent_data()
    elif dataset_type == "gaia":
        collector = collect_gaia_data()
    else:
        collector = collect_standard_data()

    # Save merged data
    output_path = collector.save_merged_data()

    # Print summary
    summary = collector.get_data_summary()
    logger.info("Data collection completed successfully")
    logger.info(f"Summary: {summary}")

    return output_path, summary


if __name__ == "__main__":
    import sys

    dtype = sys.argv[1] if len(sys.argv) > 1 else "standard"
    output_path, summary = main(dataset_type=dtype)
