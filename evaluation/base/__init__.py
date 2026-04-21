"""Base abstractions and shared constants for the evaluation package."""

from .constants import DATASETS, DATASET_TASK_MAP, infer_task_type
from .architecture import (
    ARCHITECTURE_FINAL_AGENT,
    get_final_agent_type,
    get_round_number,
    get_final_agent_key_from_metrics,
    get_final_result_id_from_entropy,
)
from .data_loader import BaseDataLoader
from .analyzer import BaseAnalyzer
from .evaluator import BaseEvaluator

__all__ = [
    "DATASETS",
    "DATASET_TASK_MAP",
    "infer_task_type",
    "ARCHITECTURE_FINAL_AGENT",
    "get_final_agent_type",
    "get_round_number",
    "get_final_agent_key_from_metrics",
    "get_final_result_id_from_entropy",
    "BaseDataLoader",
    "BaseAnalyzer",
    "BaseEvaluator",
]
