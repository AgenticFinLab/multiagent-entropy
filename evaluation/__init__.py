"""Evaluation package for multi-agent system experiments.

This package provides tools for analyzing and evaluating multi-agent system
performance, including metrics calculation, experiment analysis, and entropy
statistics.
"""

from .data_loader import DataLoader
from .entropy_analyzer import EntropyAnalyzer
from .metrics_calculator import MetricsCalculator
from .experiment_analyzer import ExperimentAnalyzer


__all__ = [
    "DataLoader",
    "MetricsCalculator",
    "ExperimentAnalyzer",
    "EntropyAnalyzer",
]
