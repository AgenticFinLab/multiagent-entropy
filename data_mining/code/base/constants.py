"""Shared constants for data-mining analyzers.

Hyperparameters mirror the values previously hardcoded inside each analyzer
(regression_analyzer, classification_analyzer, pca_analyzer, etc.) — kept
identical so behavior does not drift after the refactor.
"""

MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM"]
DATASET_TYPES = ["standard", "finagent"]

DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_XGB_REG_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_XGB_CLF_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
    "use_label_encoder": False,
    "eval_metric": "logloss",
}

DEFAULT_LGBM_REG_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

DEFAULT_LGBM_CLF_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
    "importance_type": "gain",
}

PLOT_DEFAULTS = {
    "dpi": 300,
    "bbox_inches": "tight",
    "figsize_importance": (12, 8),
    "figsize_heatmap": (16, 14),
}

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
