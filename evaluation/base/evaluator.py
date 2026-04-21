"""Base class for evaluator CLI scripts.

Both the standard evaluator and the temperature-ablation evaluator share the
same downstream pipeline: take a directory holding `all_metrics.json` and
`all_entropy_results.json`, run `Aggregator` to produce
`all_aggregated_data.csv`, then run `extract_summary_fields` to produce
`all_summary_data.csv`. This base centralizes that shared tail and the
`results/<dataset>` path convention so subclasses can focus on the
mode-specific orchestration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseEvaluator:
    """Abstract base for evaluator CLIs."""

    def __init__(self, base_path: str, args: Any):
        self.base_path = Path(base_path)
        self.args = args

    # --- output path conventions ----------------------------------------

    def get_eval_results_path(self, dataset: str) -> Path:
        """Default standard layout: `evaluation/results[/finagent]/<dataset>`."""
        if dataset.lower() == "finagent":
            return self.base_path / "evaluation" / "results_finagent" / dataset
        return self.base_path / "evaluation" / "results" / dataset

    # --- shared post-pipeline --------------------------------------------

    @staticmethod
    def run_aggregator(metrics_path: Path, entropy_path: Path, output_dir: Path) -> bool:
        """Generate the aggregated CSVs. No-op if either input is missing."""
        from aggregator import Aggregator

        if not metrics_path.exists() or not entropy_path.exists():
            logger.warning(
                f"Skipping aggregator for {output_dir}: missing {metrics_path.name} "
                f"or {entropy_path.name}"
            )
            return False
        aggregator = Aggregator(str(entropy_path), str(metrics_path), str(output_dir))
        aggregator.generate_aggregated_csvs()
        logger.info(f"Generated aggregated CSV files in {output_dir}")
        return True

    @staticmethod
    def run_summary(output_dir: Path) -> bool:
        """Generate the summary CSV from `all_aggregated_data.csv`. No-op if missing."""
        from metrics_summary import extract_summary_fields

        input_csv = output_dir / "all_aggregated_data.csv"
        if not input_csv.exists():
            return False
        output_csv = output_dir / "all_summary_data.csv"
        extract_summary_fields(input_csv, output_csv)
        logger.info(f"Generated summary CSV: {output_csv}")
        return True

    # --- subclass entry --------------------------------------------------

    def run(self) -> None:
        raise NotImplementedError("BaseEvaluator subclasses must override run()")
