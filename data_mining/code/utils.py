"""
Utility functions and constants for the data mining analysis project.
This module contains reusable components shared across different analyzers.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from features import EXPERIMENT_IDENTIFIER, DEFAULT_EXCLUDE_COLUMNS

# Define EXCLUDE_COLUMNS for backward compatibility
EXCLUDE_COLUMNS = EXPERIMENT_IDENTIFIER

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Visualization settings
VISUALIZATION_STYLE = "seaborn-v0_8"
VISUALIZATION_PALETTE = "husl"


def setup_visualization_style():
    """Set up visualization style for better plots."""
    plt.style.use(VISUALIZATION_STYLE)
    sns.set_palette(VISUALIZATION_PALETTE)


def load_data_from_path(data_path: Path) -> pd.DataFrame:
    """
    Load data from a given path.

    Args:
        data_path: Path to the CSV file

    Returns:
        Loaded DataFrame
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {len(df)} records, {len(df.columns)} columns")

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode non-numeric features to numeric values (0, 1, 2, 3, 4, ...).

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()

    # Identify categorical columns (non-numeric)
    categorical_cols = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()

    # Encode each categorical column
    for col in categorical_cols:
        if col == "dataset":
            # Skip dataset column as it's used for identification
            continue

        # Get unique values and create mapping
        unique_values = df_encoded[col].unique()
        value_mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}

        # Apply encoding
        df_encoded[col] = df_encoded[col].map(value_mapping)

    return df_encoded


def prepare_features(
    df: pd.DataFrame,
    target_column: str,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for modeling.

    Args:
        df: Input DataFrame
        target_column: Name of the target column
        exclude_columns: List of columns to exclude from features (uses EXCLUDE_COLUMNS if None)

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if exclude_columns is None:
        exclude_columns = DEFAULT_EXCLUDE_COLUMNS.copy()

    # Always exclude the target column and dataset identifier from features
    exclude_columns = exclude_columns + [target_column]

    # Get feature columns (all columns except excluded ones)
    feature_columns = [col for col in df.columns if col not in exclude_columns]

    # Select only numeric features
    numeric_features = (
        df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    )

    X = df[numeric_features].copy()
    y = df[target_column].copy()

    # Handle missing values
    X = X.fillna(X.median())

    logger.info(f"Prepared features: {len(numeric_features)} numeric features")
    logger.info(f"Target variable: {target_column}, shape: {y.shape}")

    return X, y


def create_output_directory(output_dir: Path) -> Path:
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir: Path to the output directory

    Returns:
        The created/output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[pd.Series] = None,
):
    """
    Split data into train and test sets.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion of test set
        random_state: Random state for reproducibility
        stratify: Column to stratify on (for classification)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def determine_output_directory(
    base_output_dir: str,
    target_dataset: Optional[str] = None,
    analyzer_type: str = "",
    dataset_type: str = "standard",
) -> str:
    """
    Determine output directory based on target dataset and dataset type.

    Args:
        base_output_dir: Base output directory
        target_dataset: Target dataset name
        analyzer_type: Type of analyzer (e.g., 'regression', 'classification', 'shap')
        dataset_type: Type of dataset ('standard' or 'finagent')

    Returns:
        Determined output directory path
    """
    import os

    # Adjust base output directory based on dataset_type
    if dataset_type == "finagent":
        # Replace 'data_mining/results' with 'data_mining/results_finagent'
        if base_output_dir.startswith(
            "data_mining/results"
        ) and not base_output_dir.startswith("data_mining/results_finagent"):
            base_output_dir = base_output_dir.replace(
                "data_mining/results", "data_mining/results_finagent", 1
            )
        elif not base_output_dir.startswith("data_mining/results"):
            # For custom paths, append _finagent suffix
            base_output_dir = f"{base_output_dir}_finagent"

    # Build the output directory path
    if target_dataset:
        if analyzer_type:
            output_dir = f"{base_output_dir}/{target_dataset}/{analyzer_type}"
        else:
            output_dir = f"{base_output_dir}/{target_dataset}"
    else:
        if analyzer_type:
            output_dir = f"{base_output_dir}/{analyzer_type}"
        else:
            output_dir = base_output_dir

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def get_default_data_path() -> str:
    """
    Get the default data path.

    Returns:
        Default data path string
    """
    return "data_mining/data/merged_datasets.csv"


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate that dataframe has required structure.

    Args:
        df: DataFrame to validate
        required_columns: List of required columns

    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        logger.warning("DataFrame is empty")
        return False

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False

    return True


def filter_dataframe(
    df: pd.DataFrame,
    model_names: Optional[List[str]] = None,
    architectures: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Filter dataframe based on model names, architectures, and datasets.

    Args:
        df: Input DataFrame
        model_names: List of model names to filter (None or ['all'] for all)
        architectures: List of architectures to filter (None or ['all'] for all)
        datasets: List of datasets to filter (None or ['all'] for all)

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    # Filter by model names
    if model_names and model_names != ["all"] and "model_name" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["model_name"].isin(model_names)]
        logger.info(
            f"Filtered by model_names: {model_names}, remaining records: {len(filtered_df)}"
        )

    # Filter by architectures
    if (
        architectures
        and architectures != ["all"]
        and "architecture" in filtered_df.columns
    ):
        filtered_df = filtered_df[filtered_df["architecture"].isin(architectures)]
        logger.info(
            f"Filtered by architectures: {architectures}, remaining records: {len(filtered_df)}"
        )

    # Filter by datasets
    if datasets and datasets != ["all"] and "dataset" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["dataset"].isin(datasets)]
        logger.info(
            f"Filtered by datasets: {datasets}, remaining records: {len(filtered_df)}"
        )

    return filtered_df


def generate_filter_suffix(
    model_names: Optional[List[str]] = None,
    architectures: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> str:
    """
    Generate a suffix for output directory based on filter parameters.

    Args:
        model_names: List of model names to filter
        architectures: List of architectures to filter
        datasets: List of datasets to filter

    Returns:
        Directory suffix string (e.g., "model_gpt4_arch_react_dataset_aime2025")
    """
    suffix_parts = []

    # Add model names to suffix
    if model_names and model_names != ["all"]:
        model_str = "_".join(model_names)
        suffix_parts.append(f"model_{model_str}")

    # Add architectures to suffix
    if architectures and architectures != ["all"]:
        arch_str = "_".join(architectures)
        suffix_parts.append(f"arch_{arch_str}")

    # Add datasets to suffix
    if datasets and datasets != ["all"]:
        dataset_str = "_".join(datasets)
        suffix_parts.append(f"dataset_{dataset_str}")

    return "_".join(suffix_parts) if suffix_parts else ""


def get_exclude_columns_from_config(exclude_features: str) -> List[str]:
    """
    Get exclude columns list based on configuration.

    Args:
        exclude_features: Configuration string that can be:
            - 'all': Use no exclusions (empty list)
            - 'default': Use DEFAULT_EXCLUDE_COLUMNS from features.py
            - Feature group name(s) from features.py (comma-separated)
            - Can also use '+' to combine multiple groups

    Returns:
        List of column names to exclude

    Examples:
        - 'all' -> [] (no exclusions, use all features)
        - 'default' -> DEFAULT_EXCLUDE_COLUMNS
        - 'base_model_metrics' -> BASE_MODEL_METRICS columns
        - 'base_model_metrics,experiment_identifier' -> combined columns
        - 'default+base_model_metrics' -> DEFAULT_EXCLUDE_COLUMNS + BASE_MODEL_METRICS
    """
    from features import FEATURE_GROUPS, DEFAULT_EXCLUDE_COLUMNS

    # Handle wildcard - no exclusions
    if exclude_features == "all":
        logger.info("Using all features (no exclusions)")
        return []

    # Handle default configuration
    if exclude_features == "default" or exclude_features is None:
        logger.info(f"Using default exclusions: {len(DEFAULT_EXCLUDE_COLUMNS)} columns")
        return DEFAULT_EXCLUDE_COLUMNS.copy()

    # Parse configuration string
    exclude_list = []

    # Split by '+' for combining with default
    if "+" in exclude_features:
        parts = [p.strip() for p in exclude_features.split("+")]
        if "default" in parts:
            exclude_list.extend(DEFAULT_EXCLUDE_COLUMNS)
            parts.remove("default")
        exclude_features = ",".join(parts)

    # Split by comma for multiple groups
    feature_groups = [fg.strip() for fg in exclude_features.split(",") if fg.strip()]

    # Collect columns from specified feature groups
    for group_name in feature_groups:
        if group_name in FEATURE_GROUPS:
            group_columns = FEATURE_GROUPS[group_name]
            exclude_list.extend(group_columns)
            logger.info(
                f"Adding feature group '{group_name}': {len(group_columns)} columns"
            )
        else:
            logger.warning(f"Unknown feature group: '{group_name}'")
            logger.info(f"Available groups: {', '.join(FEATURE_GROUPS.keys())}")

    # Remove duplicates while preserving order
    exclude_list = list(dict.fromkeys(exclude_list))

    logger.info(f"Total exclusions configured: {len(exclude_list)} columns")
    return exclude_list
