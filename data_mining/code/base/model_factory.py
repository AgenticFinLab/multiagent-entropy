"""ModelFactory — single source of truth for sklearn / XGBoost / LightGBM
model construction.

Replaces the per-analyzer ``RandomForestRegressor(...)``, ``XGBRegressor(...)``,
``LGBMRegressor(...)`` (and corresponding classifier) instantiations that
used to be inlined in train_models() across analyzers.

The factory falls back to ``None`` when the optional dependency is missing,
matching the pre-existing ``XGBOOST_AVAILABLE`` / ``LIGHTGBM_AVAILABLE``
guards in the original code.
"""

import logging

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .constants import (
    DEFAULT_RF_PARAMS,
    DEFAULT_XGB_REG_PARAMS,
    DEFAULT_XGB_CLF_PARAMS,
    DEFAULT_LGBM_REG_PARAMS,
    DEFAULT_LGBM_CLF_PARAMS,
)

logger = logging.getLogger(__name__)


try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")


class ModelFactory:
    """Build sklearn-compatible estimators by canonical name.

    Canonical names: ``"RandomForest"``, ``"XGBoost"``, ``"LightGBM"`` —
    matching the keys used in the original analyzers' result dictionaries.
    """

    @staticmethod
    def regressor(name: str, **overrides):
        """Return a regressor instance, or ``None`` if the dependency is missing."""
        if name == "RandomForest":
            params = {**DEFAULT_RF_PARAMS, **overrides}
            return RandomForestRegressor(**params)
        if name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                return None
            params = {**DEFAULT_XGB_REG_PARAMS, **overrides}
            return xgb.XGBRegressor(**params)
        if name == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                return None
            params = {**DEFAULT_LGBM_REG_PARAMS, **overrides}
            return lgb.LGBMRegressor(**params)
        raise ValueError(f"Unknown regressor name: {name}")

    @staticmethod
    def classifier(name: str, **overrides):
        """Return a classifier instance, or ``None`` if the dependency is missing."""
        if name == "RandomForest":
            params = {**DEFAULT_RF_PARAMS, **overrides}
            return RandomForestClassifier(**params)
        if name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                return None
            params = {**DEFAULT_XGB_CLF_PARAMS, **overrides}
            return xgb.XGBClassifier(**params)
        if name == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                return None
            params = {**DEFAULT_LGBM_CLF_PARAMS, **overrides}
            return lgb.LGBMClassifier(**params)
        raise ValueError(f"Unknown classifier name: {name}")

    @staticmethod
    def feature_importance(model):
        """Unified feature-importance extraction across the supported model types."""
        return model.feature_importances_
