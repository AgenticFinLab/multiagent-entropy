"""Base subpackage: shared abstractions for the data-mining analyzers.

Public exports:
    - constants: MODEL_NAMES, default hyperparameters, plot defaults, ...
    - FeatureManager: encapsulates the (FinAgent vs Standard) feature-discovery
      logic that was duplicated across analyzers.
    - ModelFactory: single source of truth for sklearn/XGBoost/LightGBM model
      construction and unified feature-importance extraction.
    - OutputManager + save_plot + load_dataset_csv: I/O helpers.
    - cli: argparse builders shared by main.py / run_experiments.py / analyzer
      __main__ blocks.
    - BaseAnalyzer: template-method abstract class for the analyzers
      (load -> encode -> prepare -> split -> train -> evaluate -> save).
    - BasePostProcessor: shared iteration helper for aggregator / visualizer /
      summarizer.
"""

from .constants import (
    MODEL_NAMES,
    DATASET_TYPES,
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_REG_PARAMS,
    DEFAULT_XGB_CLF_PARAMS,
    DEFAULT_LGBM_REG_PARAMS,
    DEFAULT_LGBM_CLF_PARAMS,
    PLOT_DEFAULTS,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
)
from .feature_manager import FeatureManager
from .model_factory import ModelFactory, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
from .io_utils import OutputManager, save_plot, load_dataset_csv
from . import cli
from .analyzer import BaseAnalyzer
from .post_processor import BasePostProcessor

__all__ = [
    "MODEL_NAMES",
    "DATASET_TYPES",
    "DEFAULT_RF_PARAMS",
    "DEFAULT_XGB_REG_PARAMS",
    "DEFAULT_XGB_CLF_PARAMS",
    "DEFAULT_LGBM_REG_PARAMS",
    "DEFAULT_LGBM_CLF_PARAMS",
    "PLOT_DEFAULTS",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_RANDOM_STATE",
    "FeatureManager",
    "ModelFactory",
    "XGBOOST_AVAILABLE",
    "LIGHTGBM_AVAILABLE",
    "OutputManager",
    "save_plot",
    "load_dataset_csv",
    "cli",
    "BaseAnalyzer",
    "BasePostProcessor",
]
