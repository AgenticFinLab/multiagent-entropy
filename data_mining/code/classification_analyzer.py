"""
Classification Analyzer for Multi-Agent Entropy Analysis

This module performs sample-level classification analysis to predict is_finally_correct.
Uses Random Forest, XGBoost, and LightGBM algorithms.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Import utilities
from utils import (
    setup_visualization_style,
    load_data_from_path,
    encode_categorical_features,
    prepare_features,
    create_output_directory,
    split_data,
    determine_output_directory,
    get_default_data_path,
    filter_dataframe,
    get_exclude_columns_from_config,
)

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClassificationAnalyzer:
    """Performs sample-level classification analysis on multi-agent entropy data."""

    def __init__(
        self,
        data_path: str = None,
        output_dir: str = None,
        target_dataset: str = None,
        model_names: List[str] = None,
        architectures: List[str] = None,
        datasets: List[str] = None,
        exclude_features: str = "default",
    ):
        """
        Initialize the ClassificationAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file
            output_dir: Directory to save analysis results
            target_dataset: Target dataset name for determining output directory
            model_names: List of model names to filter (None or ['*'] for all)
            architectures: List of architectures to filter (None or ['*'] for all)
            datasets: List of datasets to filter (None or ['*'] for all)
            exclude_features: Feature exclusion configuration ('*', 'default', or feature group names)
        """
        if data_path is None:
            data_path = get_default_data_path()

        # Determine output directory based on target_dataset
        if output_dir is None:
            output_dir = determine_output_directory(
                "data_mining/results", target_dataset, "classification"
            )

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.target_dataset = target_dataset
        self.model_names = model_names
        self.architectures = architectures
        self.datasets = datasets
        self.exclude_features = exclude_features
        self.df = None
        self.results = {}

        # Create output directory if it doesn't exist
        create_output_directory(self.output_dir)

        # Set style for better visualizations
        setup_visualization_style()

    def load_data(self) -> pd.DataFrame:
        """
        Load the merged dataset and apply filters.

        Returns:
            Loaded and filtered DataFrame
        """
        self.df = load_data_from_path(self.data_path)

        # Apply filters if specified
        self.df = filter_dataframe(
            self.df,
            model_names=self.model_names,
            architectures=self.architectures,
            datasets=self.datasets,
        )

        return self.df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode non-numeric features to numeric values (0, 1, 2, 3, 4, ...).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical features
        """
        return encode_categorical_features(df)

    def prepare_features(
        self,
        target_column: str = "is_finally_correct",
        exclude_columns: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.

        Args:
            target_column: Name of the target column (default: is_finally_correct)
            exclude_columns: List of columns to exclude from features

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.df is None:
            self.load_data()

        # Encode categorical features
        self.df = self.encode_categorical_features(self.df)

        if exclude_columns is None:
            # Use the configured exclude features
            exclude_columns = get_exclude_columns_from_config(self.exclude_features)

        # Always exclude the target column and dataset identifier from features
        exclude_columns = exclude_columns + [target_column]

        # Use utility function to prepare features
        X, y = prepare_features(self.df, target_column, exclude_columns)

        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train classification models for sample-level analysis.

        Args:
            X: Feature matrix
            y: Target variable (is_finally_correct)

        Returns:
            Dictionary containing trained models and their performance
        """
        logger.info("Training classification models...")

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {}
        predictions = {}
        metrics = {}

        # Random Forest Classifier
        logger.info("Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        models["RandomForest"] = rf_model
        predictions["RandomForest"] = rf_pred
        metrics["RandomForest"] = {
            "Accuracy": accuracy_score(y_test, rf_pred),
            "Precision": precision_score(y_test, rf_pred),
            "Recall": recall_score(y_test, rf_pred),
            "F1": f1_score(y_test, rf_pred),
        }

        # XGBoost Classifier
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost Classifier...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)

            models["XGBoost"] = xgb_model
            predictions["XGBoost"] = xgb_pred
            metrics["XGBoost"] = {
                "Accuracy": accuracy_score(y_test, xgb_pred),
                "Precision": precision_score(y_test, xgb_pred),
                "Recall": recall_score(y_test, xgb_pred),
                "F1": f1_score(y_test, xgb_pred),
            }

        # LightGBM Classifier
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM Classifier...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                importance_type="gain",
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)

            models["LightGBM"] = lgb_model
            predictions["LightGBM"] = lgb_pred
            metrics["LightGBM"] = {
                "Accuracy": accuracy_score(y_test, lgb_pred),
                "Precision": precision_score(y_test, lgb_pred),
                "Recall": recall_score(y_test, lgb_pred),
                "F1": f1_score(y_test, lgb_pred),
            }

        # Log metrics
        for model_name, model_metrics in metrics.items():
            logger.info(
                f"{model_name} - Accuracy: {model_metrics['Accuracy']:.4f}, "
                f"Precision: {model_metrics['Precision']:.4f}, "
                f"Recall: {model_metrics['Recall']:.4f}, "
                f"F1: {model_metrics['F1']:.4f}"
            )

        return {
            "models": models,
            "predictions": predictions,
            "metrics": metrics,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def plot_feature_importance(
        self, model, feature_names: List[str], title: str, save_path: str = None
    ):
        """
        Plot and save feature importance.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = self.output_dir / f"{title.replace(' ', '_')}.png"

        # Get feature importances
        importances = model.feature_importances_

        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x="Importance", y="Feature")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance plot saved: {save_path}")

        return importance_df

    def plot_correlation_heatmap(
        self,
        X: pd.DataFrame,
        target_column: str = None,
        title: str = "Feature Correlation Heatmap",
        save_path: str = None,
    ):
        """
        Plot and save correlation heatmap (lower triangle only).

        Args:
            X: Feature DataFrame
            target_column: Optional target column to include in correlation analysis
            title: Plot title
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = self.output_dir / f"{title.replace(' ', '_')}.png"

        # Include target column if specified
        if target_column is not None and target_column in self.df.columns:
            X_with_target = X.copy()
            X_with_target[target_column] = self.df[target_column]
            corr_matrix = X_with_target.corr()
            logger.info(f"Included '{target_column}' in correlation analysis")
        else:
            corr_matrix = X.corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Plot
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 2},
        )
        plt.title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Correlation heatmap saved: {save_path}")

        return corr_matrix

    def run_analysis(self):
        """
        Perform sample-level classification analysis.
        """
        logger.info("=" * 80)
        logger.info("Starting Sample-Level Analysis (Classification)")
        logger.info("=" * 80)

        # Prepare features for classification
        X, y = self.prepare_features(
            target_column="is_finally_correct", exclude_columns=self.exclude_columns
        )

        # Train models
        classification_results = self.train_models(X, y)

        # Plot correlation heatmap with is_finally_correct included
        corr_matrix = self.plot_correlation_heatmap(
            X,
            target_column="is_finally_correct",
            title="Feature Correlation Heatmap - Sample Level Classification",
        )

        # Plot feature importance for each model
        feature_names = X.columns.tolist()
        importance_results = {}

        for model_name, model in classification_results["models"].items():
            importance_df = self.plot_feature_importance(
                model,
                feature_names,
                title=f"Feature Importance - {model_name} (Classification)",
            )
            importance_results[model_name] = importance_df

        # Save results
        self.results = {
            "classification_results": classification_results,
            "importance_results": importance_results,
            "correlation_matrix": corr_matrix,
        }

        logger.info("Sample-level classification analysis completed")

        return self.results

    def generate_report(self):
        """
        Generate a comprehensive analysis report.
        """
        logger.info("Generating classification analysis report...")

        report_path = self.output_dir / "classification_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SAMPLE-LEVEL CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("CLASSIFICATION ANALYSIS (Predicting is_finally_correct)\n")
            f.write("-" * 80 + "\n\n")

            for model_name, metrics in self.results["classification_results"][
                "metrics"
            ].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy: {metrics['Accuracy']:.6f}\n")
                f.write(f"  Precision: {metrics['Precision']:.6f}\n")
                f.write(f"  Recall: {metrics['Recall']:.6f}\n")
                f.write(f"  F1-Score: {metrics['F1']:.6f}\n\n")

            # Top 10 important features for each model
            f.write("TOP 10 IMPORTANT FEATURES:\n")
            f.write("-" * 80 + "\n\n")

            for model_name, importance_df in self.results["importance_results"].items():
                f.write(f"{model_name}:\n")
                for idx, row in importance_df.head(10).iterrows():
                    f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                f.write("\n")

        logger.info(f"Classification analysis report saved: {report_path}")

        return report_path

    def run_full_pipeline(self):
        """
        Run complete classification analysis pipeline.
        """
        logger.info("Starting classification analysis pipeline...")

        # Load data
        self.load_data()

        # Run classification analysis
        self.run_analysis()

        # Generate report
        report_path = self.generate_report()

        logger.info("Classification analysis pipeline completed successfully!")

        return self.results, report_path


def main():
    """Main function to execute the classification analysis."""
    logger.info("Initializing Classification Analyzer...")

    # Initialize analyzer
    analyzer = ClassificationAnalyzer()

    # Run full analysis
    results, report_path = analyzer.run_full_pipeline()

    logger.info(f"Classification analysis complete. Report saved to: {report_path}")

    return results, report_path


if __name__ == "__main__":
    results, report_path = main()
