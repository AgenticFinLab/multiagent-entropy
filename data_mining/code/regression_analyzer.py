#!/usr/bin/env python3
"""
Regression Analyzer for Multi-Agent Entropy Analysis

This module performs experiment-level regression analysis to predict exp_accuracy.
Uses Random Forest, XGBoost, and LightGBM algorithms.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Import utilities
from utils import (
    EXCLUDE_COLUMNS,
    setup_visualization_style,
    load_data_from_path,
    encode_categorical_features,
    prepare_features,
    create_output_directory,
    split_data,
    determine_output_directory,
    get_default_data_path
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


class RegressionAnalyzer:
    """Performs experiment-level regression analysis on multi-agent entropy data."""

    def __init__(
        self,
        data_path: str = None,
        output_dir: str = None,
        target_dataset: str = None,
    ):
        """
        Initialize the RegressionAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file
            output_dir: Directory to save analysis results
            target_dataset: Target dataset name for determining output directory
        """
        if data_path is None:
            data_path = get_default_data_path()

        # Determine output directory based on target_dataset
        if output_dir is None:
            output_dir = determine_output_directory(
                "data_mining/results", target_dataset, "regression"
            )

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.target_dataset = target_dataset
        self.df = None
        self.results = {}

        # Create output directory if it doesn't exist
        create_output_directory(self.output_dir)

        # Set style for better visualizations
        setup_visualization_style()

    def load_data(self) -> pd.DataFrame:
        """
        Load the merged dataset.

        Returns:
            Loaded DataFrame
        """
        self.df = load_data_from_path(self.data_path)
        
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
        target_column: str = "exp_accuracy",
        exclude_columns: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.

        Args:
            target_column: Name of the target column (default: exp_accuracy)
            exclude_columns: List of columns to exclude from features

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.df is None:
            self.load_data()

        # Encode categorical features
        self.df = self.encode_categorical_features(self.df)

        if exclude_columns is None:
            exclude_columns = EXCLUDE_COLUMNS.copy()

        # Use utility function to prepare features
        X, y = prepare_features(self.df, target_column, exclude_columns)

        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train regression models for experiment-level analysis.

        Args:
            X: Feature matrix
            y: Target variable (exp_accuracy)

        Returns:
            Dictionary containing trained models and their performance
        """
        logger.info("Training regression models...")

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )

        models = {}
        predictions = {}
        metrics = {}

        # Random Forest Regressor
        logger.info("Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        models["RandomForest"] = rf_model
        predictions["RandomForest"] = rf_pred
        metrics["RandomForest"] = {
            "MSE": mean_squared_error(y_test, rf_pred),
            "MAE": mean_absolute_error(y_test, rf_pred),
            "R2": r2_score(y_test, rf_pred),
        }

        # XGBoost Regressor
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost Regressor...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)

            models["XGBoost"] = xgb_model
            predictions["XGBoost"] = xgb_pred
            metrics["XGBoost"] = {
                "MSE": mean_squared_error(y_test, xgb_pred),
                "MAE": mean_absolute_error(y_test, xgb_pred),
                "R2": r2_score(y_test, xgb_pred),
            }

        # LightGBM Regressor
        if LIGHTGBM_AVAILABLE:
            logger.info("Training LightGBM Regressor...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)

            models["LightGBM"] = lgb_model
            predictions["LightGBM"] = lgb_pred
            metrics["LightGBM"] = {
                "MSE": mean_squared_error(y_test, lgb_pred),
                "MAE": mean_absolute_error(y_test, lgb_pred),
                "R2": r2_score(y_test, lgb_pred),
            }

        # Log metrics
        for model_name, model_metrics in metrics.items():
            logger.info(
                f"{model_name} - MSE: {model_metrics['MSE']:.4f}, "
                f"MAE: {model_metrics['MAE']:.4f}, R2: {model_metrics['R2']:.4f}"
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
        Perform experiment-level regression analysis.
        """
        logger.info("=" * 80)
        logger.info("Starting Experiment-Level Analysis (Regression)")
        logger.info("=" * 80)

        # Prepare features for regression
        # Exclude is_finally_correct as it's used to calculate exp_accuracy
        X, y = self.prepare_features(
            target_column="exp_accuracy", exclude_columns=EXCLUDE_COLUMNS
        )

        # Train models
        regression_results = self.train_models(X, y)

        # Plot correlation heatmap with exp_accuracy included
        corr_matrix = self.plot_correlation_heatmap(
            X,
            target_column="exp_accuracy",
            title="Feature Correlation Heatmap - Experiment Level Regression",
        )

        # Plot feature importance for each model
        feature_names = X.columns.tolist()
        importance_results = {}

        for model_name, model in regression_results["models"].items():
            importance_df = self.plot_feature_importance(
                model,
                feature_names,
                title=f"Feature Importance - {model_name} (Regression)",
            )
            importance_results[model_name] = importance_df

        # Save results
        self.results = {
            "regression_results": regression_results,
            "importance_results": importance_results,
            "correlation_matrix": corr_matrix,
        }

        logger.info("Experiment-level regression analysis completed")

        return self.results

    def generate_report(self):
        """
        Generate a comprehensive analysis report.
        """
        logger.info("Generating regression analysis report...")

        report_path = self.output_dir / "regression_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT-LEVEL REGRESSION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("REGRESSION ANALYSIS (Predicting exp_accuracy)\n")
            f.write("-" * 80 + "\n\n")

            for model_name, metrics in self.results["regression_results"]["metrics"].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Mean Squared Error (MSE): {metrics['MSE']:.6f}\n")
                f.write(f"  Mean Absolute Error (MAE): {metrics['MAE']:.6f}\n")
                f.write(f"  R-squared (R2): {metrics['R2']:.6f}\n\n")

            # Top 10 important features for each model
            f.write("TOP 10 IMPORTANT FEATURES:\n")
            f.write("-" * 80 + "\n\n")

            for model_name, importance_df in self.results["importance_results"].items():
                f.write(f"{model_name}:\n")
                for idx, row in importance_df.head(10).iterrows():
                    f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                f.write("\n")

        logger.info(f"Regression analysis report saved: {report_path}")

        return report_path

    def run_full_pipeline(self):
        """
        Run complete regression analysis pipeline.
        """
        logger.info("Starting regression analysis pipeline...")

        # Load data
        self.load_data()

        # Run regression analysis
        self.run_analysis()

        # Generate report
        report_path = self.generate_report()

        logger.info("Regression analysis pipeline completed successfully!")

        return self.results, report_path


def main():
    """Main function to execute the regression analysis."""
    logger.info("Initializing Regression Analyzer...")

    # Initialize analyzer
    analyzer = RegressionAnalyzer()

    # Run full analysis
    results, report_path = analyzer.run_full_pipeline()

    logger.info(f"Regression analysis complete. Report saved to: {report_path}")

    return results, report_path


if __name__ == "__main__":
    results, report_path = main()
