"""Error handling utilities for hierarchical data loading and analysis.

This module provides custom exceptions and error handling utilities for the
hierarchical data loading and analysis system.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any


class HierarchicalDataError(Exception):
    """Base exception for hierarchical data operations."""

    pass


class DatasetNotFoundError(HierarchicalDataError):
    """Exception raised when a dataset is not found."""

    def __init__(self, dataset: str, base_path: Optional[Path] = None):
        message = f"Dataset '{dataset}' not found"
        if base_path:
            message += f" in {base_path}"
        super().__init__(message)
        self.dataset = dataset
        self.base_path = base_path


class ModelNotFoundError(HierarchicalDataError):
    """Exception raised when a model is not found."""

    def __init__(self, model: str, dataset: str, base_path: Optional[Path] = None):
        message = f"Model '{model}' not found for dataset '{dataset}'"
        if base_path:
            message += f" in {base_path}"
        super().__init__(message)
        self.model = model
        self.dataset = dataset
        self.base_path = base_path


class ExperimentNotFoundError(HierarchicalDataError):
    """Exception raised when an experiment is not found."""

    def __init__(self, experiment: str, model: str, dataset: str):
        message = (
            f"Experiment '{experiment}' not found for model '{model}' "
            f"in dataset '{dataset}'"
        )
        super().__init__(message)
        self.experiment = experiment
        self.model = model
        self.dataset = dataset


class FileNotFoundError(HierarchicalDataError):
    """Exception raised when a required file is not found."""

    def __init__(self, file_path: Path, file_type: str = "file"):
        message = f"{file_type.capitalize()} not found at {file_path}"
        super().__init__(message)
        self.file_path = file_path
        self.file_type = file_type


class DataFormatError(HierarchicalDataError):
    """Exception raised when data format is invalid or unexpected."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class MissingColumnError(DataFormatError):
    """Exception raised when required columns are missing from data."""

    def __init__(self, missing_columns: List[str], available_columns: List[str]):
        message = (
            f"Missing required columns: {', '.join(missing_columns)}. "
            f"Available columns: {', '.join(available_columns)}"
        )
        super().__init__(message)
        self.missing_columns = missing_columns
        self.available_columns = available_columns


class InvalidDataError(HierarchicalDataError):
    """Exception raised when data is invalid or corrupted."""

    def __init__(self, message: str, data_info: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.data_info = data_info or {}


class AnalysisError(HierarchicalDataError):
    """Exception raised when an analysis operation fails."""

    def __init__(self, message: str, analysis_type: Optional[str] = None):
        super().__init__(message)
        self.analysis_type = analysis_type


def handle_file_not_found(
    file_path: Path, file_type: str = "file", raise_exception: bool = True
) -> Optional[FileNotFoundError]:
    """Handle file not found errors.

    Args:
        file_path: Path to the file that was not found.
        file_type: Type of file (e.g., 'CSV', 'JSON').
        raise_exception: Whether to raise an exception or return None.

    Returns:
        FileNotFoundError if raise_exception is False, otherwise raises the exception.

    Raises:
        FileNotFoundError: If raise_exception is True.
    """
    error = FileNotFoundError(file_path, file_type)
    if raise_exception:
        raise error
    return error


def validate_file_exists(
    file_path: Path, file_type: str = "file", raise_exception: bool = True
) -> bool:
    """Validate that a file exists.

    Args:
        file_path: Path to the file to validate.
        file_type: Type of file (for error message).
        raise_exception: Whether to raise an exception if file doesn't exist.

    Returns:
        True if file exists, False otherwise.

    Raises:
        FileNotFoundError: If file doesn't exist and raise_exception is True.
    """
    if not file_path.exists():
        handle_file_not_found(file_path, file_type, raise_exception)
        return False
    return True


def validate_directory_exists(dir_path: Path, raise_exception: bool = True) -> bool:
    """Validate that a directory exists.

    Args:
        dir_path: Path to the directory to validate.
        raise_exception: Whether to raise an exception if directory doesn't exist.

    Returns:
        True if directory exists, False otherwise.

    Raises:
        FileNotFoundError: If directory doesn't exist and raise_exception is True.
    """
    if not dir_path.exists():
        error = FileNotFoundError(dir_path, "directory")
        if raise_exception:
            raise error
        return False
    return True


def validate_columns(
    data_columns: List[str], required_columns: List[str], raise_exception: bool = True
) -> bool:
    """Validate that required columns are present in data.

    Args:
        data_columns: List of columns present in the data.
        required_columns: List of required columns.
        raise_exception: Whether to raise an exception if columns are missing.

    Returns:
        True if all required columns are present, False otherwise.

    Raises:
        MissingColumnError: If required columns are missing and raise_exception is True.
    """
    missing_columns = set(required_columns) - set(data_columns)
    if missing_columns:
        error = MissingColumnError(list(missing_columns), data_columns)
        if raise_exception:
            raise error
        return False
    return True


def safe_load_json(
    file_path: Path, raise_exception: bool = True
) -> Optional[Dict[str, Any]]:
    """Safely load a JSON file with error handling.

    Args:
        file_path: Path to the JSON file.
        raise_exception: Whether to raise an exception on error.

    Returns:
        Dictionary containing JSON data, or None if loading fails and
        raise_exception is False.

    Raises:
        FileNotFoundError: If file doesn't exist and raise_exception is True.
        DataFormatError: If JSON is invalid and raise_exception is True.
    """
    import json

    try:
        validate_file_exists(file_path, "JSON file", raise_exception)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        error = DataFormatError(f"Invalid JSON format in {file_path}: {str(e)}")
        if raise_exception:
            raise error
        return None
    except Exception as e:
        if raise_exception:
            raise
        return None


def safe_load_csv(file_path: Path, raise_exception: bool = True) -> Optional[Any]:
    """Safely load a CSV file with error handling.

    Args:
        file_path: Path to the CSV file.
        raise_exception: Whether to raise an exception on error.

    Returns:
        DataFrame containing CSV data, or None if loading fails and
        raise_exception is False.

    Raises:
        FileNotFoundError: If file doesn't exist and raise_exception is True.
        DataFormatError: If CSV is invalid and raise_exception is True.
    """
    import pandas as pd

    try:
        validate_file_exists(file_path, "CSV file", raise_exception)
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError as e:
        error = DataFormatError(f"Empty CSV file at {file_path}")
        if raise_exception:
            raise error
        return None
    except pd.errors.ParserError as e:
        error = DataFormatError(f"Invalid CSV format in {file_path}: {str(e)}")
        if raise_exception:
            raise error
        return None
    except Exception as e:
        if raise_exception:
            raise
        return None


def log_error(
    error: Exception, context: Optional[Dict[str, Any]] = None, logger=None
) -> None:
    """Log an error with optional context information.

    Args:
        error: The exception to log.
        context: Optional dictionary containing context information.
        logger: Optional logger object. If None, prints to stdout.
    """
    message = f"Error: {type(error).__name__}: {str(error)}"
    if context:
        message += f"\nContext: {context}"

    if logger:
        logger.error(message)
    else:
        print(message)


def get_error_context(
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    experiment: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create error context dictionary.

    Args:
        dataset: Optional dataset name.
        model: Optional model name.
        experiment: Optional experiment name.
        additional_info: Optional additional context information.

    Returns:
        Dictionary containing error context.
    """
    context = {}
    if dataset:
        context["dataset"] = dataset
    if model:
        context["model"] = model
    if experiment:
        context["experiment"] = experiment
    if additional_info:
        context.update(additional_info)
    return context


class ErrorHandler:
    """Centralized error handler for hierarchical data operations."""

    def __init__(self, raise_exceptions: bool = True, logger=None):
        """Initialize the ErrorHandler.

        Args:
            raise_exceptions: Whether to raise exceptions or handle them gracefully.
            logger: Optional logger object for logging errors.
        """
        self.raise_exceptions = raise_exceptions
        self.logger = logger
        self.error_count = 0
        self.errors = []

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        raise_exception: Optional[bool] = None,
    ) -> None:
        """Handle an error.

        Args:
            error: The exception to handle.
            context: Optional context information.
            raise_exception: Whether to raise the exception. If None, uses
                            the instance's default setting.
        """
        self.error_count += 1
        self.errors.append(
            {"error": error, "context": context, "type": type(error).__name__}
        )

        log_error(error, context, self.logger)

        if raise_exception is None:
            raise_exception = self.raise_exceptions

        if raise_exception:
            raise error

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors encountered.

        Returns:
            Dictionary containing error summary.
        """
        return {
            "total_errors": self.error_count,
            "error_types": {},
            "errors": self.errors,
        }

    def reset(self) -> None:
        """Reset error tracking."""
        self.error_count = 0
        self.errors = []
