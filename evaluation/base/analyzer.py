"""Base class for analyzers (ExperimentAnalyzer, EntropyStatistic).

Provides shared state (data_loader + base_path) and the response-stripping
logic that was previously duplicated between `ExperimentAnalyzer.save_results`
and `temperature_ablation_evaluator._remove_response_fields`.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from utils import save_json


class BaseAnalyzer:
    """Abstract base for analyzers that consume experiment results."""

    def __init__(self, base_path: str, data_loader: Optional[Any] = None):
        self.base_path = Path(base_path)
        if data_loader is None:
            # Lazy import to avoid a hard cycle: BaseAnalyzer is imported by
            # `evaluation.base.__init__`, while the concrete `DataLoader` lives
            # one level up and pulls `BaseDataLoader` from this package.
            from data_loader import DataLoader
            data_loader = DataLoader(str(base_path))
        self.data_loader = data_loader

    @staticmethod
    def _remove_response_fields(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Strip `response` from every agent record (in place safe shallow copy).

        Used to keep persisted JSON files small. The original input dict is
        mutated only at the leaf level (response keys deleted), but the wrapping
        copy keeps the caller's reference safe to reuse.
        """
        metrics_copy = metrics.copy()
        if "models" not in metrics_copy:
            return metrics_copy
        for model_data in metrics_copy["models"].values():
            for exp_metrics in model_data.get("experiments", {}).values():
                for sample_data in exp_metrics.get("samples", {}).values():
                    for agent_record in sample_data.get("agents", {}).values():
                        agent_record.pop("response", None)
        return metrics_copy

    def save_metrics_json(self, metrics: Dict[str, Any], output_path: str) -> None:
        """Persist metrics JSON with response fields stripped."""
        save_json(self._remove_response_fields(metrics), output_path)

    @staticmethod
    def save_entropy_json(results: Dict[str, Any], output_path: str) -> None:
        """Persist entropy results JSON unchanged."""
        save_json(results, output_path)
