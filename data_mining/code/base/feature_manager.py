"""FeatureManager — encapsulates FinAgent vs Standard feature handling.

Replaces the duplicated `_log_finagent_features` blocks that lived in
regression_analyzer.py, classification_analyzer.py, pca_analyzer.py, and
feature_ablation_analyzer.py.
"""

import logging
from typing import List, Optional

import pandas as pd

from features import (
    FINAGENT_EVALUATION_FEATURES,
    FINAGENT_STEP_ENTROPY_FEATURES,
    discover_step_entropy_features,
)

logger = logging.getLogger(__name__)


class FeatureManager:
    """Holds dataset-aware feature metadata and logging."""

    def __init__(self, dataset_type: str = "standard"):
        self.dataset_type = dataset_type

    def log_finagent_features(self, X: pd.DataFrame) -> None:
        """Log FinAgent-specific features present in the feature matrix.

        Mirrors the per-analyzer ``_log_finagent_features`` helper exactly.
        Safe to call regardless of dataset_type — it is a no-op when no
        FinAgent columns are present.
        """
        eval_features = [f for f in FINAGENT_EVALUATION_FEATURES if f in X.columns]
        if eval_features:
            logger.info(f"FinAgent evaluation features detected: {eval_features}")

        step_features = [f for f in FINAGENT_STEP_ENTROPY_FEATURES if f in X.columns]
        if step_features:
            logger.info(f"FinAgent step entropy features detected: {step_features}")

        dynamic_step_features = discover_step_entropy_features(X.columns)
        if dynamic_step_features:
            logger.info(
                f"Dynamic step entropy features discovered: {dynamic_step_features}"
            )

    @staticmethod
    def filter_existing(columns: List[str], df: pd.DataFrame) -> List[str]:
        """Drop column names that are not present in ``df`` — preserves order."""
        return [c for c in columns if c in df.columns]
