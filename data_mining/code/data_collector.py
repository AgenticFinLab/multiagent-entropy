#!/usr/bin/env python3
"""
Data Collection Module for Multi-Agent Entropy Analysis

This module collects and merges data from multiple datasets for comprehensive analysis.
It reads all_aggregated_data_exclude_agent.csv files from different dataset folders and combines them into a single dataset with dataset identification.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and merges data from multiple dataset folders."""

    def __init__(
        self, base_dir: str = "evaluation/results",
        target_datasets: Optional[List[str]] = None
    ):
        """
        Initialize the DataCollector.

        Args:
            base_dir: Base directory containing dataset folders
            target_datasets: Optional list of specific dataset names to collect.
                           If None, discovers all available datasets automatically.
        """
        self.base_dir = Path(base_dir)
        self.target_datasets = target_datasets
        self.datasets = []
        self.merged_data = None

    def discover_datasets(self) -> List[str]:
        """
        Discover dataset folders in the base directory.
        If target_datasets is specified, only validates and uses those datasets.
        Otherwise, discovers all available datasets.

        Returns:
            List of dataset folder names
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

        # If target_datasets is specified, validate and use them
        if self.target_datasets:
            datasets = []
            for dataset_name in self.target_datasets:
                dataset_path = self.base_dir / dataset_name
                csv_file = dataset_path / "all_aggregated_data_exclude_agent.csv"
                
                if not dataset_path.exists():
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
        csv_path = (
            self.base_dir / dataset_name / "all_aggregated_data_exclude_agent.csv"
        )

        if not csv_path.exists():
            logger.warning(f"CSV file not found for dataset {dataset_name}: {csv_path}")
            return None

        try:
            df = pd.read_csv(csv_path)
            logger.info(
                f"Loaded dataset {dataset_name}: {len(df)} records, {len(df.columns)} columns"
            )
            return df
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            return None

    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge all discovered datasets into a single DataFrame.

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

        # Merge all dataframes
        self.merged_data = pd.concat(all_dataframes, ignore_index=True)
        logger.info(
            f"Merged {len(all_dataframes)} datasets: {len(self.merged_data)} total records"
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
                output_path = f"data_mining/results/{self.target_datasets[0]}/merged_datasets.csv"
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


def main():
    """Main function to execute data collection and merging."""
    logger.info("Starting data collection and merging process")

    # Initialize data collector
    collector = DataCollector()

    # Discover datasets
    datasets = collector.discover_datasets()
    logger.info(f"Found {len(datasets)} datasets: {datasets}")

    # Merge datasets
    merged_data = collector.merge_datasets()

    # Save merged data
    output_path = collector.save_merged_data()

    # Print summary
    summary = collector.get_data_summary()
    logger.info("Data collection completed successfully")
    logger.info(f"Summary: {summary}")

    return output_path, summary


if __name__ == "__main__":
    output_path, summary = main()
