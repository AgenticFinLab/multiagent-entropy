"""Evaluation package for multi-agent system experiments.

This package provides tools for analyzing and evaluating multi-agent system
performance, including metrics calculation, experiment analysis, and entropy
statistics.
"""

# Import data loading functionality for accessing experiment results
from .data_loader import DataLoader

# Import entropy analysis functionality for analyzing entropy statistics
from .entropy_statistic import EntropyStatistic

# Import metrics calculation functionality for evaluating system performance
from .metrics_calculator import MetricsCalculator

# Import experiment analysis functionality for analyzing experiment results
from .experiment_analyzer import ExperimentAnalyzer


# Define public API for the evaluation package
__all__ = [
    "DataLoader",
    "MetricsCalculator",
    "ExperimentAnalyzer",
    "EntropyStatistic",
]
