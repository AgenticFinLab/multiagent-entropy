"""Unified data loading and analysis module for multi-agent system entropy analysis.

This module provides comprehensive functionality for loading, preprocessing, and analyzing experimental data from multi-agent system evaluations. It supports data loading at multiple hierarchical levels (dataset, model, and experiment), enabling flexible multi-level analysis and cross-level comparisons.

Key Features:
- Hierarchical data loading from dataset/model/experiment structure
- Data preprocessing and feature engineering
- Multi-level analysis (dataset, model, experiment)
- Cross-level comparisons and aggregations
- Comprehensive statistics and reporting
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from constants import ARCHITECTURES
from error_handling import (
    DatasetNotFoundError,
    ModelNotFoundError,
    FileNotFoundError,
    ErrorHandler,
)
from utils import (
    get_feature_groups,
    preprocess_data,
    get_summary_statistics,
    get_architecture_comparison,
    calculate_metrics_from_data,
)


class DataLoader:
    """Unified data loader and analyzer for multi-agent system entropy analysis.

    This class provides comprehensive functionality for:
    - Loading data at multiple hierarchical levels (dataset, model, experiment)
    - Preprocessing and feature engineering
    - Multi-level analysis and comparisons
    - Generating comprehensive reports

    Attributes:
        data_path: Path to the data file or base directory.
        base_path: Base path to the evaluation results directory (for hierarchical loading).
        raw_data: DataFrame containing the raw loaded data.
        processed_data: DataFrame containing the preprocessed data.
        architectures: List of available architecture types.
        results: Dictionary storing analysis results.
        error_handler: ErrorHandler instance for error management.
    """

    def __init__(
        self,
        data_path: str,
        base_path: Optional[str] = None,
        raise_exceptions: bool = True,
    ):
        """Initialize the DataLoader.

        Args:
            data_path: Path to a CSV file or base directory for hierarchical loading.
            base_path: Optional base path to evaluation results directory.
            raise_exceptions: Whether to raise exceptions on errors (default: True).
        """
        self.data_path = Path(data_path)
        self.base_path = Path(base_path) if base_path else self.data_path.parent
        self.raw_data = None
        self.processed_data = None
        self.architectures = ARCHITECTURES
        self.results = {}
        self.error_handler = ErrorHandler(raise_exceptions=raise_exceptions)

    def load_data(self) -> pd.DataFrame:
        """Load raw data from a CSV file.

        Returns:
            DataFrame containing the raw experimental data.
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully, {len(self.raw_data)} rows")
            return self.raw_data
        except Exception as e:
            self.error_handler.handle_error(
                e, {"operation": "load_data", "file_path": str(self.data_path)}
            )

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Categorize features into different groups based on their prefixes.

        Returns:
            Dictionary mapping feature group names to lists of feature names.
        """
        if self.raw_data is None:
            self.load_data()

        return get_feature_groups(self.raw_data.columns.tolist())

    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the raw data for analysis.

        Converts data types, handles missing values, and ensures proper formatting
        for entropy features.

        Returns:
            DataFrame containing the preprocessed data.
        """
        if self.raw_data is None:
            self.load_data()

        df = preprocess_data(self.raw_data)

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
            "exp_total_token",
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

        return get_architecture_comparison(self.processed_data)

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

        return get_summary_statistics(self.processed_data)

    def get_metrics(self) -> Dict:
        """Calculate metrics from the loaded data.

        Returns:
            Dictionary containing calculated metrics.
        """
        if self.processed_data is None:
            self.preprocess_data()

        return calculate_metrics_from_data(self.processed_data)

    def _discover_datasets(self) -> List[str]:
        """Discover all available datasets in the results directory.

        Returns:
            List of dataset names.
        """
        if not self.base_path.exists():
            return []

        datasets = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                datasets.append(item.name)

        return sorted(datasets)

    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets.

        Returns:
            List of dataset names.
        """
        return self._discover_datasets()

    def get_available_models(self, dataset: str) -> List[str]:
        """Get list of available models for a specific dataset.

        Args:
            dataset: Name of the dataset.

        Returns:
            List of model names.

        Raises:
            ValueError: If dataset does not exist.
        """
        dataset_path = self.base_path / dataset
        if not dataset_path.exists():
            raise DatasetNotFoundError(dataset, self.base_path)

        models = []
        for item in dataset_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                models.append(item.name)

        return sorted(models)

    def get_available_experiments(self, dataset: str, model: str) -> List[str]:
        """Get list of available experiments for a specific dataset and model.

        Args:
            dataset: Name of the dataset.
            model: Name of the model.

        Returns:
            List of experiment names (CSV file names without extension).

        Raises:
            ValueError: If dataset or model does not exist.
        """
        model_path = self.base_path / dataset / model
        if not model_path.exists():
            raise ModelNotFoundError(model, dataset, self.base_path)

        experiments = []
        for item in model_path.iterdir():
            if (
                item.is_file()
                and item.suffix == ".csv"
                and item.name != "aggregated_data.csv"
            ):
                experiments.append(item.stem)

        return sorted(experiments)

    def load_dataset_level_data(
        self, dataset: str, include_models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load dataset-level aggregated data.

        Args:
            dataset: Name of the dataset.
            include_models: Optional list of models to include. If None, includes all models.

        Returns:
            DataFrame containing dataset-level aggregated data.

        Raises:
            ValueError: If dataset does not exist or no data is found.
            FileNotFoundError: If dataset-level CSV file does not exist.
        """
        dataset_path = self.base_path / dataset
        if not dataset_path.exists():
            raise DatasetNotFoundError(dataset, self.base_path)

        csv_path = dataset_path / "all_aggregated_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset-level aggregated data not found at {csv_path}"
            )

        df = pd.read_csv(csv_path)

        if include_models:
            df = df[df["model_name"].isin(include_models)]

        return df

    def load_model_level_data(self, dataset: str, model: str) -> pd.DataFrame:
        """Load model-level aggregated data.

        Args:
            dataset: Name of the dataset.
            model: Name of the model.

        Returns:
            DataFrame containing model-level aggregated data.

        Raises:
            ValueError: If dataset or model does not exist.
            FileNotFoundError: If model-level CSV file does not exist.
        """
        model_path = self.base_path / dataset / model
        if not model_path.exists():
            raise ModelNotFoundError(model, dataset, self.base_path)

        csv_path = model_path / "aggregated_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Model-level aggregated data not found at {csv_path}"
            )

        return pd.read_csv(csv_path)

    def load_experiment_level_data(
        self, dataset: str, model: str, experiment: str
    ) -> pd.DataFrame:
        """Load experiment-level data.

        Args:
            dataset: Name of the dataset.
            model: Name of the model.
            experiment: Name of the experiment (CSV file name without extension).

        Returns:
            DataFrame containing experiment-level data.

        Raises:
            ValueError: If dataset, model, or experiment does not exist.
            FileNotFoundError: If experiment CSV file does not exist.
        """
        model_path = self.base_path / dataset / model
        if not model_path.exists():
            raise ValueError(
                f"Model '{model}' for dataset '{dataset}' does not exist at {model_path}"
            )

        csv_path = model_path / f"{experiment}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Experiment data not found at {csv_path}")

        return pd.read_csv(csv_path)

    def load_dataset_metrics(self, dataset: str) -> Dict:
        """Load dataset-level metrics JSON file.

        Args:
            dataset: Name of the dataset.

        Returns:
            Dictionary containing dataset-level metrics.

        Raises:
            ValueError: If dataset does not exist.
            FileNotFoundError: If metrics JSON file does not exist.
        """
        dataset_path = self.base_path / dataset
        if not dataset_path.exists():
            raise ValueError(f"Dataset '{dataset}' does not exist at {dataset_path}")

        json_path = dataset_path / "all_metrics.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset-level metrics not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dataset_entropy_results(self, dataset: str) -> Dict:
        """Load dataset-level entropy results JSON file.

        Args:
            dataset: Name of the dataset.

        Returns:
            Dictionary containing dataset-level entropy results.

        Raises:
            ValueError: If dataset does not exist.
            FileNotFoundError: If entropy results JSON file does not exist.
        """
        dataset_path = self.base_path / dataset
        if not dataset_path.exists():
            raise ValueError(f"Dataset '{dataset}' does not exist at {dataset_path}")

        json_path = dataset_path / "all_entropy_results.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Dataset-level entropy results not found at {json_path}"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_levels(
        self,
        dataset: str,
        model: Optional[str] = None,
        experiment: Optional[str] = None,
    ) -> Dict[str, Union[pd.DataFrame, Dict]]:
        """Load data at multiple levels based on provided parameters.

        This method provides a flexible way to load data at different levels:
        - If only dataset is provided: loads dataset-level data
        - If dataset and model are provided: loads model-level data
        - If dataset, model, and experiment are provided: loads experiment-level data

        Args:
            dataset: Name of the dataset.
            model: Optional name of the model.
            experiment: Optional name of the experiment.

        Returns:
            Dictionary containing loaded data and metadata.

        Raises:
            ValueError: If any of the specified levels do not exist.
        """
        result = {
            "dataset": dataset,
            "model": model,
            "experiment": experiment,
            "data": None,
            "metrics": None,
            "entropy_results": None,
        }

        if experiment and model:
            result["data"] = self.load_experiment_level_data(dataset, model, experiment)
        elif model:
            result["data"] = self.load_model_level_data(dataset, model)
        else:
            result["data"] = self.load_dataset_level_data(dataset)
            try:
                result["metrics"] = self.load_dataset_metrics(dataset)
                result["entropy_results"] = self.load_dataset_entropy_results(dataset)
            except FileNotFoundError:
                pass

        return result

    def get_hierarchy_info(self, dataset: Optional[str] = None) -> Dict:
        """Get information about the hierarchical structure.

        Args:
            dataset: Optional dataset name. If provided, returns info for that dataset only.

        Returns:
            Dictionary containing hierarchy information.
        """
        info = {}

        if dataset:
            if dataset not in self._discover_datasets():
                raise ValueError(f"Dataset '{dataset}' does not exist")

            info[dataset] = {"models": {}}
            for model in self.get_available_models(dataset):
                info[dataset]["models"][model] = {
                    "experiments": self.get_available_experiments(dataset, model)
                }
        else:
            for ds in self._discover_datasets():
                info[ds] = {"models": {}}
                for model in self.get_available_models(ds):
                    info[ds]["models"][model] = {
                        "experiments": self.get_available_experiments(ds, model)
                    }

        return info

    def aggregate_across_models(
        self, dataset: str, models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Aggregate data across multiple models for a dataset.

        Args:
            dataset: Name of the dataset.
            models: Optional list of models to include. If None, includes all models.

        Returns:
            DataFrame containing aggregated data across specified models.

        Raises:
            ValueError: If dataset does not exist or no valid models are found.
        """
        if models is None:
            models = self.get_available_models(dataset)

        if not models:
            raise ValueError(f"No models found for dataset '{dataset}'")

        dfs = []
        for model in models:
            try:
                df = self.load_model_level_data(dataset, model)
                dfs.append(df)
            except FileNotFoundError:
                print(f"Warning: Model-level data not found for model '{model}'")

        if not dfs:
            raise ValueError(
                f"No valid data found for any model in dataset '{dataset}'"
            )

        return pd.concat(dfs, ignore_index=True)

    def aggregate_across_experiments(
        self, dataset: str, model: str, experiments: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Aggregate data across multiple experiments for a model.

        Args:
            dataset: Name of the dataset.
            model: Name of the model.
            experiments: Optional list of experiments to include. If None, includes all experiments.

        Returns:
            DataFrame containing aggregated data across specified experiments.

        Raises:
            ValueError: If dataset or model does not exist, or no valid experiments are found.
        """
        if experiments is None:
            experiments = self.get_available_experiments(dataset, model)

        if not experiments:
            raise ValueError(
                f"No experiments found for model '{model}' in dataset '{dataset}'"
            )

        dfs = []
        for experiment in experiments:
            try:
                df = self.load_experiment_level_data(dataset, model, experiment)
                dfs.append(df)
            except FileNotFoundError:
                print(
                    f"Warning: Experiment data not found for experiment '{experiment}'"
                )

        if not dfs:
            raise ValueError(
                f"No valid data found for any experiment in model '{model}'"
            )

        return pd.concat(dfs, ignore_index=True)

    def analyze_dataset_level(
        self, dataset: str, include_models: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Analyze data at the dataset level.

        Performs comprehensive analysis including model comparison, architecture
        comparison, and overall statistics.

        Args:
            dataset: Name of the dataset.
            include_models: Optional list of models to include.

        Returns:
            Dictionary containing dataset-level analysis results.

        Raises:
            ValueError: If dataset does not exist.
        """
        print(f"Analyzing dataset-level data for '{dataset}'...")

        data = self.load_dataset_level_data(dataset, include_models)

        model_comparison = self._compare_models(data)
        architecture_comparison = self._compare_architectures(data)
        overall_statistics = self._calculate_overall_statistics(data)

        self.results[f"{dataset}_dataset_level"] = {
            "model_comparison": model_comparison,
            "architecture_comparison": architecture_comparison,
            "overall_statistics": overall_statistics,
        }

        return self.results[f"{dataset}_dataset_level"]

    def analyze_model_level(self, dataset: str, model: str) -> Dict[str, pd.DataFrame]:
        """Analyze data at the model level.

        Performs analysis including architecture comparison, experiment comparison,
        and model-specific statistics.

        Args:
            dataset: Name of the dataset.
            model: Name of the model.

        Returns:
            Dictionary containing model-level analysis results.

        Raises:
            ValueError: If dataset or model does not exist.
        """
        print(f"Analyzing model-level data for '{dataset}/{model}'...")

        data = self.load_model_level_data(dataset, model)

        architecture_comparison = self._compare_architectures(data)
        experiment_comparison = self._compare_experiments(data)
        model_statistics = self._calculate_overall_statistics(data)

        self.results[f"{dataset}_{model}_model_level"] = {
            "architecture_comparison": architecture_comparison,
            "experiment_comparison": experiment_comparison,
            "model_statistics": model_statistics,
        }

        return self.results[f"{dataset}_{model}_model_level"]

    def analyze_experiment_level(
        self, dataset: str, model: str, experiment: str
    ) -> Dict[str, pd.DataFrame]:
        """Analyze data at the experiment level.

        Performs fine-grained analysis including sample-level statistics,
        agent-level analysis, and round-level evolution.

        Args:
            dataset: Name of the dataset.
            model: Name of the model.
            experiment: Name of the experiment.

        Returns:
            Dictionary containing experiment-level analysis results.

        Raises:
            ValueError: If dataset, model, or experiment does not exist.
        """
        print(
            f"Analyzing experiment-level data for '{dataset}/{model}/{experiment}'..."
        )

        data = self.load_experiment_level_data(dataset, model, experiment)

        sample_statistics = self._calculate_sample_statistics(data)
        agent_statistics = self._calculate_agent_statistics(data)
        round_evolution = self._analyze_round_evolution(data)

        self.results[f"{dataset}_{model}_{experiment}_experiment_level"] = {
            "sample_statistics": sample_statistics,
            "agent_statistics": agent_statistics,
            "round_evolution": round_evolution,
        }

        return self.results[f"{dataset}_{model}_{experiment}_experiment_level"]

    def compare_across_datasets(
        self, datasets: List[str], metric: str = "exp_accuracy"
    ) -> pd.DataFrame:
        """Compare performance across multiple datasets.

        Args:
            datasets: List of dataset names to compare.
            metric: Metric to use for comparison (default: exp_accuracy).

        Returns:
            DataFrame containing cross-dataset comparison.

        Raises:
            ValueError: If any dataset does not exist.
        """
        print(f"Comparing across {len(datasets)} datasets...")

        comparison_data = []
        for dataset in datasets:
            try:
                data = self.load_dataset_level_data(dataset)
                dataset_stats = {
                    "dataset": dataset,
                    "mean_accuracy": data["exp_accuracy"].mean(),
                    "std_accuracy": data["exp_accuracy"].std(),
                    "median_accuracy": data["exp_accuracy"].median(),
                    "num_models": data["model_name"].nunique(),
                    "num_samples": data["sample_id"].nunique(),
                }
                comparison_data.append(dataset_stats)
            except Exception as e:
                print(f"Warning: Could not load dataset '{dataset}': {e}")

        comparison_df = pd.DataFrame(comparison_data)

        self.results["cross_dataset_comparison"] = comparison_df

        return comparison_df

    def compare_across_models(
        self, dataset: str, models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare performance across multiple models within a dataset.

        Args:
            dataset: Name of the dataset.
            models: Optional list of models to compare. If None, compares all models.

        Returns:
            DataFrame containing cross-model comparison.

        Raises:
            ValueError: If dataset does not exist.
        """
        print(f"Comparing across models in dataset '{dataset}'...")

        data = self.load_dataset_level_data(dataset, models)

        model_comparison = self._compare_models(data)

        self.results[f"{dataset}_cross_model_comparison"] = model_comparison

        return model_comparison

    def compare_architectures_across_models(
        self, dataset: str, models: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare architecture performance across multiple models.

        Args:
            dataset: Name of the dataset.
            models: Optional list of models to include. If None, includes all models.

        Returns:
            DataFrame containing architecture comparison across models.

        Raises:
            ValueError: If dataset does not exist.
        """
        print(f"Comparing architectures across models in dataset '{dataset}'...")

        data = self.load_dataset_level_data(dataset, models)

        comparison = []
        for model in data["model_name"].unique():
            model_data = data[data["model_name"] == model]
            for arch in model_data["architecture"].unique():
                arch_data = model_data[model_data["architecture"] == arch]
                arch_stats = {
                    "model": model,
                    "architecture": arch,
                    "accuracy": arch_data["exp_accuracy"].mean(),
                    "mean_entropy": arch_data["sample_mean_entropy"].mean(),
                    "std_entropy": arch_data["sample_std_entropy"].mean(),
                    "num_samples": arch_data["sample_id"].nunique(),
                }

                if "base_model_accuracy" in arch_data.columns:
                    arch_stats["base_model_accuracy"] = arch_data["base_model_accuracy"].mean()

                if "base_model_format_compliance_rate" in arch_data.columns:
                    arch_stats["base_model_format_compliance_rate"] = arch_data["base_model_format_compliance_rate"].mean()

                comparison.append(arch_stats)

        comparison_df = pd.DataFrame(comparison)

        self.results[f"{dataset}_architecture_across_models"] = comparison_df

        return comparison_df

    def _compare_models(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compare performance across models.

        Args:
            data: DataFrame containing dataset-level data.

        Returns:
            DataFrame containing model comparison statistics.
        """
        comparison = []
        for model in data["model_name"].unique():
            model_data = data[data["model_name"] == model]
            model_stats = {
                "model": model,
                "mean_accuracy": model_data["exp_accuracy"].mean(),
                "std_accuracy": model_data["exp_accuracy"].std(),
                "median_accuracy": model_data["exp_accuracy"].median(),
                "mean_entropy": model_data["sample_mean_entropy"].mean(),
                "std_entropy": model_data["sample_std_entropy"].mean(),
                "num_samples": model_data["sample_id"].nunique(),
                "num_experiments": model_data["experiment_name"].nunique(),
            }

            if "base_model_accuracy" in model_data.columns:
                model_stats["base_model_accuracy"] = model_data["base_model_accuracy"].mean()
                model_stats["base_model_std_accuracy"] = model_data["base_model_accuracy"].std()

            if "base_model_format_compliance_rate" in model_data.columns:
                model_stats["base_model_format_compliance_rate"] = model_data["base_model_format_compliance_rate"].mean()

            comparison.append(model_stats)

        return pd.DataFrame(comparison)

    def _compare_architectures(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compare performance across architectures.

        Args:
            data: DataFrame containing data.

        Returns:
            DataFrame containing architecture comparison statistics.
        """
        comparison = []
        for arch in data["architecture"].unique():
            arch_data = data[data["architecture"] == arch]
            comparison.append(
                {
                    "architecture": arch,
                    "mean_accuracy": arch_data["exp_accuracy"].mean(),
                    "std_accuracy": arch_data["exp_accuracy"].std(),
                    "median_accuracy": arch_data["exp_accuracy"].median(),
                    "mean_entropy": arch_data["sample_mean_entropy"].mean(),
                    "std_entropy": arch_data["sample_std_entropy"].mean(),
                    "num_samples": arch_data["sample_id"].nunique(),
                }
            )

        return pd.DataFrame(comparison)

    def _compare_experiments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compare performance across experiments.

        Args:
            data: DataFrame containing model-level data.

        Returns:
            DataFrame containing experiment comparison statistics.
        """
        comparison = []
        for exp in data["experiment_name"].unique():
            exp_data = data[data["experiment_name"] == exp]
            comparison.append(
                {
                    "experiment": exp,
                    "architecture": exp_data["architecture"].unique()[0],
                    "mean_accuracy": exp_data["exp_accuracy"].mean(),
                    "std_accuracy": exp_data["exp_accuracy"].std(),
                    "mean_entropy": exp_data["sample_mean_entropy"].mean(),
                    "std_entropy": exp_data["sample_std_entropy"].mean(),
                    "num_samples": exp_data["sample_id"].nunique(),
                }
            )

        return pd.DataFrame(comparison)

    def _calculate_overall_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate overall statistics for the data.

        Args:
            data: DataFrame containing data.

        Returns:
            Dictionary containing overall statistics.
        """
        return {
            "total_rows": len(data),
            "unique_samples": data["sample_id"].nunique(),
            "unique_experiments": data["experiment_name"].nunique(),
            "unique_architectures": data["architecture"].nunique(),
            "mean_accuracy": data["exp_accuracy"].mean(),
            "std_accuracy": data["exp_accuracy"].std(),
            "mean_entropy": data["sample_mean_entropy"].mean(),
            "std_entropy": data["sample_std_entropy"].mean(),
        }

    def _calculate_sample_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate sample-level statistics.

        Args:
            data: DataFrame containing experiment-level data.

        Returns:
            DataFrame containing sample-level statistics.
        """
        sample_cols = [
            "sample_id",
            "ground_truth",
            "is_finally_correct",
        ] + [col for col in data.columns if col.startswith("sample_")]

        sample_data = data[sample_cols].drop_duplicates(subset=["sample_id"])

        return sample_data

    def _calculate_agent_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate agent-level statistics.

        Args:
            data: DataFrame containing experiment-level data.

        Returns:
            DataFrame containing agent-level statistics.
        """
        agent_cols = [
            "agent_name",
            "agent_key",
            "agent_round_number",
        ] + [col for col in data.columns if col.startswith("agent_")]

        agent_data = (
            data[agent_cols]
            .groupby(["agent_name", "agent_round_number"])
            .agg(
                {
                    "agent_total_entropy": ["mean", "std", "min", "max"],
                    "agent_mean_entropy": ["mean", "std"],
                    "agent_token_count": "mean",
                }
            )
            .reset_index()
        )

        agent_data.columns = ["_".join(col).strip("_") for col in agent_data.columns]

        return agent_data

    def _analyze_round_evolution(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze entropy evolution across rounds.

        Args:
            data: DataFrame containing experiment-level data.

        Returns:
            DataFrame containing round evolution statistics.
        """
        round_data = (
            data.groupby(["sample_id", "agent_round_number"])
            .agg(
                {
                    "round_total_entropy": "first",
                    "round_infer_avg_entropy": "first",
                    "is_finally_correct": "first",
                }
            )
            .reset_index()
        )

        round_stats = round_data.groupby("agent_round_number").agg(
            {
                "round_total_entropy": ["mean", "std", "median"],
                "round_infer_avg_entropy": ["mean", "std", "median"],
                "is_finally_correct": "mean",
            }
        )

        return round_stats

    def generate_comprehensive_report(
        self, dataset: str, models: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """Generate a comprehensive multi-level analysis report.

        Args:
            dataset: Name of the dataset.
            models: Optional list of models to analyze. If None, analyzes all models.

        Returns:
            Dictionary containing comprehensive analysis results.
        """
        print(f"Generating comprehensive report for dataset '{dataset}'...")

        report = {
            "dataset_level": self.analyze_dataset_level(dataset, models),
        }

        if models is None:
            models = self.get_available_models(dataset)

        for model in models:
            try:
                report[f"{model}_model_level"] = self.analyze_model_level(
                    dataset, model
                )
            except Exception as e:
                print(f"Warning: Could not analyze model '{model}': {e}")

        return report

    def save_results(self, output_dir: str) -> None:
        """Save analysis results to CSV files.

        Args:
            output_dir: Directory path where results should be saved.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for key, value in self.results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.DataFrame):
                        sub_value.to_csv(
                            output_path / f"{key}_{sub_key}.csv", index=True
                        )
                    elif isinstance(sub_value, dict):
                        pd.DataFrame([sub_value]).to_csv(
                            output_path / f"{key}_{sub_key}.csv", index=False
                        )
            elif isinstance(value, pd.DataFrame):
                value.to_csv(output_path / f"{key}.csv", index=True)

        print(f"Analysis results saved to: {output_path}")
