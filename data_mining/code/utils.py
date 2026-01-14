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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Common constants
EXCLUDE_COLUMNS = [
    # ignored identifier
    "dataset",
    "model_name",
    # "architecture",
    "sample_id",
    # useless data
    # "num_rounds",
    # "exp_num_inferences",
    # "round_1_num_inferences",
    # "round_2_num_inferences",
    # base model metrics
    # "base_model_accuracy",
    # "base_model_is_finally_correct",
    # "base_model_format_compliance",
    # "base_model_format_compliance_rate",
]

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
        exclude_columns = EXCLUDE_COLUMNS.copy()

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
    base_output_dir: str, target_dataset: Optional[str] = None, analyzer_type: str = ""
) -> str:
    """
    Determine output directory based on target dataset.

    Args:
        base_output_dir: Base output directory
        target_dataset: Target dataset name
        analyzer_type: Type of analyzer (e.g., 'regression', 'classification', 'shap')

    Returns:
        Determined output directory path
    """
    if target_dataset:
        if analyzer_type:
            return f"{base_output_dir}/{target_dataset}/{analyzer_type}"
        else:
            return f"{base_output_dir}/{target_dataset}"
    else:
        if analyzer_type:
            return f"{base_output_dir}/{analyzer_type}"
        else:
            return base_output_dir


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
