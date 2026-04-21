"""BaseAnalyzer — shared skeleton for the sklearn/XGBoost/LightGBM analyzers.

Subclasses (RegressionAnalyzer, ClassificationAnalyzer, PCAAnalysis,
FeatureAblationAnalyzer, CalibrationAnalyzer, ShapAnalyzer) inherit the
load/filter/encode/prepare/split pipeline plus a standardized per-model
training loop. Subclass-specific metric computation lives in ``_metrics``;
optional per-model post-processing (e.g. predict_proba for classifiers)
lives in ``_postprocess_model``.

This class is deliberately thin: the goal is to delete duplicated code
without changing any output paths or numbers. Subclasses that need
different behavior can simply override the relevant hook.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import (
    setup_visualization_style,
    encode_categorical_features,
    prepare_features,
    split_data,
    filter_dataframe,
    get_exclude_columns_from_config,
    get_default_data_path,
)

from .feature_manager import FeatureManager
from .io_utils import OutputManager, load_dataset_csv, save_plot
from .model_factory import ModelFactory
from .constants import (
    MODEL_NAMES,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    PLOT_DEFAULTS,
)

logger = logging.getLogger(__name__)


class BaseAnalyzer:
    """Template-method base class for tabular data-mining analyzers."""

    # ---- subclass configuration knobs ----------------------------------
    target_column: str = ""                 # e.g. "exp_accuracy"
    analyzer_type: str = ""                 # e.g. "regression"
    is_classification: bool = False         # True => stratified split + classifier factory
    default_base_output_dir: str = "data_mining/results"

    # ---- construction --------------------------------------------------
    def __init__(
        self,
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        target_dataset: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        architectures: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        exclude_features: str = "default",
        dataset_type: str = "standard",
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        if data_path is None:
            data_path = get_default_data_path()

        if output_dir is None:
            output_dir = OutputManager(
                self.default_base_output_dir,
                analyzer_type=self.analyzer_type,
                target_dataset=target_dataset,
                dataset_type=dataset_type,
            ).resolve()

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.target_dataset = target_dataset
        self.model_names = model_names
        self.architectures = architectures
        self.datasets = datasets
        self.exclude_features = exclude_features
        self.dataset_type = dataset_type
        self.test_size = test_size
        self.random_state = random_state

        self.df: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        self.feature_manager = FeatureManager(dataset_type=dataset_type)

        # Ensure output dir exists (resolve() already does this for the auto
        # path, but a user-supplied path must be created too).
        self.output_dir.mkdir(parents=True, exist_ok=True)

        setup_visualization_style()

    # ---- data pipeline (template methods) ------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load CSV and apply (model_name, architecture, dataset) filters."""
        self.df = load_dataset_csv(self.data_path)
        self.df = filter_dataframe(
            self.df,
            model_names=self.model_names,
            architectures=self.architectures,
            datasets=self.datasets,
        )
        return self.df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return encode_categorical_features(df)

    def prepare_features(
        self,
        target_column: Optional[str] = None,
        exclude_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Encode categoricals, resolve exclusions, return (X, y)."""
        if self.df is None:
            self.load_data()

        target = target_column or self.target_column
        self.df = self.encode_categorical_features(self.df)

        if exclude_columns is None:
            exclude_columns = get_exclude_columns_from_config(self.exclude_features)

        exclude_columns = exclude_columns + [target]
        exclude_columns = [c for c in exclude_columns if c in self.df.columns]

        X, y = prepare_features(self.df, target, exclude_columns)
        self.feature_manager.log_finagent_features(X)
        return X, y

    def split(self, X: pd.DataFrame, y: pd.Series):
        """Train/test split — stratifies on ``y`` for classifiers."""
        return split_data(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=(y if self.is_classification else None),
        )

    # ---- model training loop -------------------------------------------

    def _make_model(self, name: str):
        """Return a fresh estimator, or ``None`` if the dependency is missing."""
        if self.is_classification:
            return ModelFactory.classifier(name)
        return ModelFactory.regressor(name)

    def _metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Subclasses that use ``train_models`` return their metric dict
        (e.g. MSE/MAE/R² or Acc/P/R/F1). Analyzers with fully custom training
        loops (PCA component search, feature ablation, calibration) can leave
        this as the default empty dict.
        """
        return {}

    def _postprocess_model(self, name: str, model, X_test) -> Dict[str, Any]:
        """Hook for subclass-specific per-model artifacts (default: none)."""
        return {}

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the configured model set and collect metrics/predictions."""
        X_train, X_test, y_train, y_test = self.split(X, y)

        models: Dict[str, Any] = {}
        predictions: Dict[str, Any] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        extras: Dict[str, Dict[str, Any]] = {}

        for name in MODEL_NAMES:
            model = self._make_model(name)
            if model is None:
                continue
            logger.info(f"Training {name} ({self.analyzer_type})...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            models[name] = model
            predictions[name] = pred
            metrics[name] = self._metrics(y_test, pred)
            extra = self._postprocess_model(name, model, X_test)
            if extra:
                extras[name] = extra

        self._log_metrics(metrics)

        return {
            "models": models,
            "predictions": predictions,
            "metrics": metrics,
            "extras": extras,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def _log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        for name, m in metrics.items():
            parts = ", ".join(f"{k}: {v:.4f}" for k, v in m.items())
            logger.info(f"{name} - {parts}")

    # ---- standard plots ------------------------------------------------

    def plot_feature_importance(
        self, model, feature_names: List[str], title: str,
        save_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Bar plot + CSV dump of feature importances."""
        if save_path is None:
            save_path = self.output_dir / f"{title.replace(' ', '_')}.png"

        importances = ModelFactory.feature_importance(model)
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        csv_path = Path(str(save_path).replace(".png", ".csv"))
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"Feature importance data saved to CSV: {csv_path}")

        plt.figure(figsize=PLOT_DEFAULTS["figsize_importance"])
        sns.barplot(data=importance_df.head(20), x="Importance", y="Feature")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        save_plot(save_path)
        logger.info(f"Feature importance plot saved: {save_path}")
        return importance_df

    def plot_correlation_heatmap(
        self, X: pd.DataFrame, target_column: Optional[str] = None,
        title: str = "Feature Correlation Heatmap",
        save_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Lower-triangle correlation heatmap + CSV dump."""
        if save_path is None:
            save_path = self.output_dir / f"{title.replace(' ', '_')}.png"

        if target_column is not None and self.df is not None and target_column in self.df.columns:
            X_with_target = X.copy()
            X_with_target[target_column] = self.df[target_column]
            corr_matrix = X_with_target.corr()
            logger.info(f"Included '{target_column}' in correlation analysis")
        else:
            corr_matrix = X.corr()

        csv_path = Path(str(save_path).replace(".png", ".csv"))
        corr_matrix.to_csv(csv_path)
        logger.info(f"Correlation matrix saved to CSV: {csv_path}")

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        plt.figure(figsize=PLOT_DEFAULTS["figsize_heatmap"])
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 2},
        )
        plt.title(title, fontsize=16, fontweight="bold")
        save_plot(save_path)
        logger.info(f"Correlation heatmap saved: {save_path}")
        return corr_matrix

    # ---- top-level pipeline -------------------------------------------

    def run_analysis(self) -> Dict[str, Any]:
        """Subclasses orchestrate prepare -> train -> plot -> save.

        Default is a no-op; subclasses that expose ``run_full_pipeline``
        through the base class implementation must override this.
        """
        return {}

    def generate_report(self) -> Optional[Path]:
        """Subclasses write a text report and return its path.

        Default returns None so analyzers with custom pipelines (e.g. PCA,
        feature ablation) can implement their own reporting flows without
        being forced into this shape.
        """
        return None

    def run_full_pipeline(self):
        logger.info(f"Starting {self.analyzer_type} analysis pipeline...")
        self.load_data()
        self.run_analysis()
        report_path = self.generate_report()
        logger.info(f"{self.analyzer_type.capitalize()} analysis pipeline completed.")
        return self.results, report_path
