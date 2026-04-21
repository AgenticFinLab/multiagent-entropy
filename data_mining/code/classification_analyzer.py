"""Classification Analyzer for multi-agent entropy data.

Sample-level classification to predict ``is_finally_correct`` using
RandomForest, XGBoost, and LightGBM. The load/encode/prepare/split/train
skeleton lives in ``base.BaseAnalyzer``; this file only defines the
classification-specific metric set, per-model probability capture, and the
plot/report orchestration.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from base import BaseAnalyzer

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClassificationAnalyzer(BaseAnalyzer):
    """Sample-level classification on ``is_finally_correct``."""

    target_column = "is_finally_correct"
    analyzer_type = "classification"
    is_classification = True

    def _metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
        }

    def _postprocess_model(self, name, model, X_test) -> Dict[str, Any]:
        return {"proba": model.predict_proba(X_test)}

    def save_prediction_probabilities(self, classification_results: Dict) -> None:
        """Dump per-model ``predict_proba`` output to CSV."""
        extras = classification_results.get("extras", {})
        for model_name, extra in extras.items():
            probabilities = extra.get("proba")
            if probabilities is None:
                continue
            proba_df = pd.DataFrame(
                probabilities,
                columns=[f"prob_class_{i}" for i in range(probabilities.shape[1])],
            )
            proba_df.insert(0, "sample_index", range(len(proba_df)))
            csv_path = self.output_dir / f"prediction_probabilities_{model_name}.csv"
            proba_df.to_csv(csv_path, index=False)
            logger.info(
                f"Prediction probabilities for {model_name} saved to: {csv_path}"
            )

    def run_analysis(self):
        logger.info("=" * 80)
        logger.info("Starting Sample-Level Analysis (Classification)")
        logger.info("=" * 80)

        X, y = self.prepare_features(target_column=self.target_column)
        classification_results = self.train_models(X, y)

        # Preserve backward-compatible flat key for downstream consumers.
        classification_results["prediction_probabilities"] = {
            name: extra["proba"]
            for name, extra in classification_results.get("extras", {}).items()
            if "proba" in extra
        }

        self.save_prediction_probabilities(classification_results)

        corr_matrix = self.plot_correlation_heatmap(
            X,
            target_column=self.target_column,
            title="Feature Correlation Heatmap - Sample Level Classification",
        )

        feature_names = X.columns.tolist()
        importance_results = {}
        for model_name, model in classification_results["models"].items():
            importance_df = self.plot_feature_importance(
                model, feature_names,
                title=f"Feature Importance - {model_name} (Classification)",
            )
            importance_results[model_name] = importance_df

        self.results = {
            "classification_results": classification_results,
            "importance_results": importance_results,
            "correlation_matrix": corr_matrix,
        }
        logger.info("Sample-level classification analysis completed")
        return self.results

    def generate_report(self) -> Path:
        logger.info("Generating classification analysis report...")
        report_path = self.output_dir / "classification_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SAMPLE-LEVEL CLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("CLASSIFICATION ANALYSIS (Predicting is_finally_correct)\n")
            f.write("-" * 80 + "\n\n")

            for model_name, metrics in self.results["classification_results"]["metrics"].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy: {metrics['Accuracy']:.6f}\n")
                f.write(f"  Precision: {metrics['Precision']:.6f}\n")
                f.write(f"  Recall: {metrics['Recall']:.6f}\n")
                f.write(f"  F1-Score: {metrics['F1']:.6f}\n\n")

            f.write("TOP 10 IMPORTANT FEATURES:\n")
            f.write("-" * 80 + "\n\n")
            for model_name, importance_df in self.results["importance_results"].items():
                f.write(f"{model_name}:\n")
                for _, row in importance_df.head(10).iterrows():
                    f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                f.write("\n")

        logger.info(f"Classification analysis report saved: {report_path}")
        return report_path


def main():
    logger.info("Initializing Classification Analyzer...")
    analyzer = ClassificationAnalyzer()
    results, report_path = analyzer.run_full_pipeline()
    logger.info(f"Classification analysis complete. Report saved to: {report_path}")
    return results, report_path


if __name__ == "__main__":
    results, report_path = main()
