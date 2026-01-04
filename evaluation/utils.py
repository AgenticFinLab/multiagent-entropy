"""Utility functions for the evaluation module.

This module provides helper functions for file operations, data formatting,
and common utilities used across the evaluation package.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any


def ensure_directory_exists(file_path: Path) -> None:
    """Ensure that the parent directory of a file path exists.

    Args:
        file_path: Path to a file whose parent directory should exist.

    Returns:
        None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save data to a JSON file.

    Args:
        data: Dictionary data to save.
        output_path: Path to the output JSON file.

    Returns:
        None
    """
    output_file = Path(output_path)
    ensure_directory_exists(output_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(
    rows: List[Dict[str, Any]], output_path: str, fieldnames: List[str]
) -> None:
    """Save data to a CSV file.

    Args:
        rows: List of dictionaries containing data to save.
        output_path: Path to the output CSV file.
        fieldnames: List of column names for the CSV file.

    Returns:
        None
    """
    output_file = Path(output_path)
    ensure_directory_exists(output_file)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_output_directory(base_path: Path, dataset: str, subdirectory: str = "") -> Path:
    """Get the output directory path for evaluation results.

    Args:
        base_path: Base path to the project directory.
        dataset: Dataset name (e.g., "gsm8k", "humaneval").
        subdirectory: Optional subdirectory within the dataset folder.

    Returns:
        Path object pointing to the output directory.
    """
    output_dir = base_path / "evaluation" / "results" / dataset
    if subdirectory:
        output_dir = output_dir / subdirectory
    return output_dir


def format_float(value: float, precision: int = 4) -> float:
    """Format a float value to specified precision.

    Args:
        value: Float value to format.
        precision: Number of decimal places to keep.

    Returns:
        Rounded float value.
    """
    return round(value, precision)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, handling division by zero.

    Args:
        numerator: Numerator value.
        denominator: Denominator value.
        default: Default value to return if denominator is zero.

    Returns:
        Result of division or default value if denominator is zero.
    """
    return numerator / denominator if denominator != 0 else default
