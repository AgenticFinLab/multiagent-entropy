#!/usr/bin/env python3
"""Regression Analyzer for multi-agent entropy data.

Experiment-level regression to predict ``exp_accuracy`` using RandomForest,
XGBoost, and LightGBM. The load/encode/prepare/split/train skeleton lives in
``base.BaseAnalyzer``; this file only defines the regression-specific metric
set and the orchestration of plots and reports.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from base import BaseAnalyzer

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RegressionAnalyzer(BaseAnalyzer):
    """Experiment-level regression on ``exp_accuracy``."""

    target_column = "exp_accuracy"
    analyzer_type = "regression"
    is_classification = False

    def _metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "MSE": mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    def run_analysis(self):
        logger.info("=" * 80)
        logger.info("Starting Experiment-Level Analysis (Regression)")
        logger.info("=" * 80)

        X, y = self.prepare_features(target_column=self.target_column)
        regression_results = self.train_models(X, y)

        corr_matrix = self.plot_correlation_heatmap(
            X,
            target_column=self.target_column,
            title="Feature Correlation Heatmap - Experiment Level Regression",
        )

        feature_names = X.columns.tolist()
        importance_results = {}
        for model_name, model in regression_results["models"].items():
            importance_df = self.plot_feature_importance(
                model, feature_names,
                title=f"Feature Importance - {model_name} (Regression)",
            )
            importance_results[model_name] = importance_df

        self.results = {
            "regression_results": regression_results,
            "importance_results": importance_results,
            "correlation_matrix": corr_matrix,
        }
        logger.info("Experiment-level regression analysis completed")
        return self.results

    def generate_report(self) -> Path:
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

            f.write("TOP 10 IMPORTANT FEATURES:\n")
            f.write("-" * 80 + "\n\n")
            for model_name, importance_df in self.results["importance_results"].items():
                f.write(f"{model_name}:\n")
                for _, row in importance_df.head(10).iterrows():
                    f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                f.write("\n")

        logger.info(f"Regression analysis report saved: {report_path}")
        return report_path


def main():
    logger.info("Initializing Regression Analyzer...")
    analyzer = RegressionAnalyzer()
    results, report_path = analyzer.run_full_pipeline()
    logger.info(f"Regression analysis complete. Report saved to: {report_path}")
    return results, report_path


if __name__ == "__main__":
    results, report_path = main()
