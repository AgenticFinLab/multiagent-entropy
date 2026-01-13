#!/usr/bin/env python3
"""
Data Mining Analyzer Module for Multi-Agent Entropy Analysis

This module performs comprehensive data mining analysis including:
1. Experiment-level analysis: Regression models to predict exp_accuracy
2. Sample-level analysis: Classification models to predict is_finally_correct

Uses Random Forest, XGBoost, and LightGBM algorithms.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")

# Machine Learning libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataMiningAnalyzer:
    """Performs comprehensive data mining analysis on multi-agent entropy data."""

    def __init__(self, data_path: str = None, output_dir: str = None):
        """
        Initialize the DataMiningAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file
            output_dir: Directory to save analysis results
        """
        if data_path is None:
            data_path = "data_mining/data/merged_datasets.csv"
        if output_dir is None:
            output_dir = "data_mining/results"

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.df = None
        self.results = {}

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style for better visualizations
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def load_data(self) -> pd.DataFrame:
        """
        Load the merged dataset.

        Returns:
            Loaded DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        logger.info(
            f"Loaded data: {len(self.df)} records, {len(self.df.columns)} columns"
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
        df_encoded = df.copy()

        # Identify categorical columns (non-numeric)
        categorical_cols = df_encoded.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        # Encode each categorical column
        for col in categorical_cols:
            if col == "dataset":
                # Skip dataset column as it's used for identification
                continue

            # Get unique values and create mapping
            unique_values = df_encoded[col].unique()
            value_mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}

            # Apply encoding
            df_encoded[col] = df_encoded[col].map(value_mapping)

            logger.info(
                f"Encoded '{col}': {len(value_mapping)} unique values mapped to 0-{len(value_mapping)-1}"
            )
            logger.info(f"  Mapping: {value_mapping}")

        return df_encoded

    def prepare_features(
        self,
        target_column: str,
        exclude_columns: List[str] = None,
        include_target_in_corr: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.

        Args:
            target_column: Name of the target column
            exclude_columns: List of columns to exclude from features
            include_target_in_corr: Whether to include target column in correlation analysis

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.df is None:
            self.load_data()

        # Encode categorical features
        self.df = self.encode_categorical_features(self.df)

        if exclude_columns is None:
            exclude_columns = []

        # Always exclude the target column and dataset identifier from features
        exclude_columns = exclude_columns + [target_column, "dataset"]

        # Get feature columns (all columns except excluded ones)
        feature_columns = [col for col in self.df.columns if col not in exclude_columns]

        # Select only numeric features
        numeric_features = (
            self.df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        )

        X = self.df[numeric_features].copy()
        y = self.df[target_column].copy()

        # Handle missing values
        X = X.fillna(X.median())

        logger.info(f"Prepared features: {len(numeric_features)} numeric features")
        logger.info(f"Target variable: {target_column}, shape: {y.shape}")

        return X, y

    def train_regression_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
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
        X_train, X_test, y_train, y_test = train_test_split(
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
            "X_test": X_test,
            "y_test": y_test,
        }

    def train_classification_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
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
        X_train, X_test, y_train, y_test = train_test_split(
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
            "X_test": X_test,
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
            annot_kws={"size": 6},
        )
        plt.title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Correlation heatmap saved: {save_path}")

        return corr_matrix

    def run_experiment_level_analysis(self):
        """
        Perform experiment-level analysis (regression on exp_accuracy).
        """
        logger.info("=" * 80)
        logger.info("Starting Experiment-Level Analysis (Regression)")
        logger.info("=" * 80)

        # Prepare features for regression
        # Exclude is_finally_correct as it's used to calculate exp_accuracy
        X, y = self.prepare_features(
            target_column="exp_accuracy", exclude_columns=["is_finally_correct"]
        )

        # Train models
        regression_results = self.train_regression_models(X, y)

        # Plot correlation heatmap with exp_accuracy included
        corr_matrix = self.plot_correlation_heatmap(
            X,
            target_column="exp_accuracy",
            title="Feature Correlation Heatmap - Experiment Level",
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
        self.results["experiment_level"] = {
            "regression_results": regression_results,
            "importance_results": importance_results,
            "correlation_matrix": corr_matrix,
        }

        logger.info("Experiment-level analysis completed")

        return self.results["experiment_level"]

    def run_sample_level_analysis(self):
        """
        Perform sample-level analysis (classification on is_finally_correct).
        """
        logger.info("=" * 80)
        logger.info("Starting Sample-Level Analysis (Classification)")
        logger.info("=" * 80)

        # Prepare features for classification
        X, y = self.prepare_features(
            target_column="is_finally_correct", exclude_columns=[]
        )

        # Train models
        classification_results = self.train_classification_models(X, y)

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
        self.results["sample_level"] = {
            "classification_results": classification_results,
            "importance_results": importance_results,
        }

        logger.info("Sample-level analysis completed")

        return self.results["sample_level"]

    def generate_report(self):
        """
        Generate a comprehensive analysis report.
        """
        logger.info("Generating comprehensive analysis report...")

        report_path = self.output_dir / "analysis_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("DATA MINING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Experiment-level results
            if "experiment_level" in self.results:
                f.write("EXPERIMENT-LEVEL ANALYSIS (Regression on exp_accuracy)\n")
                f.write("-" * 80 + "\n\n")

                for model_name, metrics in self.results["experiment_level"][
                    "regression_results"
                ]["metrics"].items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Mean Squared Error (MSE): {metrics['MSE']:.6f}\n")
                    f.write(f"  Mean Absolute Error (MAE): {metrics['MAE']:.6f}\n")
                    f.write(f"  R-squared (R2): {metrics['R2']:.6f}\n\n")

                # Top 10 important features for each model
                f.write("TOP 10 IMPORTANT FEATURES (Experiment Level):\n")
                f.write("-" * 80 + "\n\n")

                for model_name, importance_df in self.results["experiment_level"][
                    "importance_results"
                ].items():
                    f.write(f"{model_name}:\n")
                    for idx, row in importance_df.head(10).iterrows():
                        f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                    f.write("\n")

            # Sample-level results
            if "sample_level" in self.results:
                f.write("\n" + "=" * 80 + "\n")
                f.write(
                    "SAMPLE-LEVEL ANALYSIS (Classification on is_finally_correct)\n"
                )
                f.write("-" * 80 + "\n\n")

                for model_name, metrics in self.results["sample_level"][
                    "classification_results"
                ]["metrics"].items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Accuracy: {metrics['Accuracy']:.6f}\n")
                    f.write(f"  Precision: {metrics['Precision']:.6f}\n")
                    f.write(f"  Recall: {metrics['Recall']:.6f}\n")
                    f.write(f"  F1-Score: {metrics['F1']:.6f}\n\n")

                # Top 10 important features for each model
                f.write("TOP 10 IMPORTANT FEATURES (Sample Level):\n")
                f.write("-" * 80 + "\n\n")

                for model_name, importance_df in self.results["sample_level"][
                    "importance_results"
                ].items():
                    f.write(f"{model_name}:\n")
                    for idx, row in importance_df.head(10).iterrows():
                        f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                    f.write("\n")

        logger.info(f"Analysis report saved: {report_path}")

        return report_path

    def run_full_analysis(self):
        """
        Run complete analysis pipeline.
        """
        logger.info("Starting full data mining analysis pipeline...")

        # Load data
        self.load_data()

        # Run experiment-level analysis
        self.run_experiment_level_analysis()

        # Run sample-level analysis
        self.run_sample_level_analysis()

        # Generate report
        report_path = self.generate_report()

        logger.info("Full analysis pipeline completed successfully!")

        return self.results, report_path


def main():
    """Main function to execute the data mining analysis."""
    logger.info("Initializing Data Mining Analyzer...")

    # Initialize analyzer
    analyzer = DataMiningAnalyzer()

    # Run full analysis
    results, report_path = analyzer.run_full_analysis()

    logger.info(f"Analysis complete. Report saved to: {report_path}")

    return results, report_path


if __name__ == "__main__":
    results, report_path = main()
