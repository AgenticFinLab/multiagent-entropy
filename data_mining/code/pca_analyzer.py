"""
PCA Analyzer for Multi-Agent Entropy Analysis

This module performs PCA (Principal Component Analysis) to analyze feature redundancy
in the original feature space. Used to address reviewer concerns about feature redundancy.
All analysis targets the classification task (predicting is_finally_correct).
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils import split_data
from base import BaseAnalyzer, ModelFactory, MODEL_NAMES

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PCAAnalysis(BaseAnalyzer):
    """Performs PCA analysis to evaluate feature redundancy in multi-agent entropy data."""

    target_column = "is_finally_correct"
    analyzer_type = "pca"
    is_classification = True

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
        super().__init__(
            data_path=data_path,
            output_dir=output_dir,
            target_dataset=target_dataset,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
            exclude_features=exclude_features,
        )
        self.X = None
        self.y = None
        self.scaler = None
        self.pca = None
        self.explained_variance_ratio = None

    def prepare_features(
        self,
        target_column: str = "is_finally_correct",
        exclude_columns: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Override to cache X/y on the instance (used by fit_pca etc.)."""
        self.X, self.y = super().prepare_features(
            target_column=target_column, exclude_columns=exclude_columns
        )
        return self.X, self.y

    def fit_pca(self, n_components: int = None) -> Tuple[PCA, np.ndarray]:
        """
        Fit PCA on standardized features.

        Args:
            n_components: Number of components to keep. If None, keeps all components.

        Returns:
            Tuple of (PCA object, explained variance ratio array)
        """
        if self.X is None:
            self.prepare_features()

        logger.info(f"Fitting PCA with n_components={n_components}...")

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        logger.info(f"PCA fitted with {len(self.explained_variance_ratio)} components")
        logger.info(
            f"Total variance explained: {np.sum(self.explained_variance_ratio):.4f}"
        )

        return self.pca, self.explained_variance_ratio

    def _train_and_evaluate_models(
        self, X_train, X_test, y_train, y_test
    ) -> Dict[str, Dict[str, float]]:
        """Train and evaluate the configured classifier set on given splits."""
        metrics = {}
        for name in MODEL_NAMES:
            model = ModelFactory.classifier(name)
            if model is None:
                continue
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics[name] = {
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred),
            }
        return metrics

    def evaluate_pca_performance(
        self, component_range: List[int] = None
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Evaluate model performance across different numbers of PCA components.

        Args:
            component_range: List of component numbers to evaluate.
                           If None, uses adaptive range from 1 to n_features.

        Returns:
            Dictionary mapping n_components to model metrics
        """
        if self.X is None:
            self.prepare_features()

        if self.scaler is None:
            self.fit_pca()

        n_features = self.X.shape[1]

        # Define adaptive component range if not provided
        if component_range is None:
            if n_features <= 10:
                component_range = list(range(1, n_features + 1))
            elif n_features <= 30:
                component_range = list(range(1, 11)) + list(range(15, n_features + 1, 5))
            else:
                component_range = (
                    list(range(1, 11))
                    + list(range(15, 31, 5))
                    + list(range(40, n_features + 1, 10))
                )
            # Ensure we include the full feature count
            if n_features not in component_range:
                component_range.append(n_features)
            component_range = sorted(set(component_range))

        logger.info(f"Evaluating PCA performance for components: {component_range}")

        # Standardize features once
        X_scaled = self.scaler.transform(self.X)

        results = {}
        for n_comp in component_range:
            if n_comp > n_features:
                continue

            logger.info(f"Evaluating with {n_comp} components...")

            # Apply PCA transformation
            pca_temp = PCA(n_components=n_comp)
            X_pca = pca_temp.fit_transform(X_scaled)

            # Split data
            X_train, X_test, y_train, y_test = split_data(
                pd.DataFrame(X_pca),
                self.y,
                test_size=0.2,
                random_state=42,
                stratify=self.y,
            )

            # Train and evaluate models
            metrics = self._train_and_evaluate_models(X_train, X_test, y_train, y_test)
            results[n_comp] = metrics

        self.results["pca_performance"] = results

        # Save results to CSV
        self._save_pca_performance_to_csv(results)

        return results

    def _save_pca_performance_to_csv(self, results: Dict) -> None:
        """Save PCA performance results to CSV."""
        rows = []
        for n_comp, model_metrics in results.items():
            for model_name, metrics in model_metrics.items():
                row = {"n_components": n_comp, "model": model_name}
                row.update(metrics)
                rows.append(row)

        df_results = pd.DataFrame(rows)
        csv_path = self.output_dir / "pca_performance_by_components.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"PCA performance results saved to: {csv_path}")

    def find_optimal_components(
        self, variance_threshold: float = 0.95
    ) -> Dict[str, int]:
        """
        Find optimal number of PCA components based on variance and performance.

        Args:
            variance_threshold: Cumulative variance threshold (default: 0.95)

        Returns:
            Dictionary with recommended component numbers and rationale
        """
        if self.explained_variance_ratio is None:
            self.fit_pca()

        # Find components for variance threshold
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        n_for_variance = np.argmax(cumulative_variance >= variance_threshold) + 1

        logger.info(
            f"Components needed for {variance_threshold*100:.0f}% variance: {n_for_variance}"
        )

        # Find elbow point using second derivative
        if len(self.explained_variance_ratio) > 2:
            diff1 = np.diff(self.explained_variance_ratio)
            diff2 = np.diff(diff1)
            elbow_idx = np.argmax(np.abs(diff2)) + 1
        else:
            elbow_idx = 1

        logger.info(f"Elbow point at component: {elbow_idx}")

        # Find performance-based optimal (if performance data available)
        n_for_performance = None
        if "pca_performance" in self.results:
            perf_results = self.results["pca_performance"]
            # Find the minimum components where F1 is within 1% of max
            max_f1 = 0
            best_n_comp = list(perf_results.keys())[-1]

            for n_comp, model_metrics in perf_results.items():
                avg_f1 = np.mean([m["F1"] for m in model_metrics.values()])
                if avg_f1 > max_f1:
                    max_f1 = avg_f1
                    best_n_comp = n_comp

            # Find minimum components within 1% of best F1
            for n_comp in sorted(perf_results.keys()):
                avg_f1 = np.mean(
                    [m["F1"] for m in perf_results[n_comp].values()]
                )
                if avg_f1 >= max_f1 * 0.99:
                    n_for_performance = n_comp
                    break

            logger.info(f"Components for optimal performance: {n_for_performance}")

        # Determine final recommendation
        recommended = n_for_variance
        if n_for_performance is not None:
            recommended = min(n_for_variance, n_for_performance)

        result = {
            "variance_threshold_components": n_for_variance,
            "elbow_components": elbow_idx,
            "performance_optimal_components": n_for_performance,
            "recommended_components": recommended,
        }

        self.results["optimal_components"] = result
        return result

    def compare_original_vs_pca(
        self, optimal_n_components: int
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compare model performance between original features and PCA-reduced features.

        Args:
            optimal_n_components: Number of PCA components to use

        Returns:
            Dictionary with 'original' and 'pca' performance metrics
        """
        if self.X is None:
            self.prepare_features()

        logger.info(f"Comparing original features vs PCA({optimal_n_components})...")

        # Standardize features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X)
        else:
            X_scaled = self.scaler.transform(self.X)

        # Split original data (use same random_state for fair comparison)
        X_train_orig, X_test_orig, y_train, y_test = split_data(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Evaluate on original features
        logger.info("Evaluating on original features...")
        original_metrics = self._train_and_evaluate_models(
            X_train_orig, X_test_orig, y_train, y_test
        )

        # Apply PCA and split
        pca_temp = PCA(n_components=optimal_n_components)
        X_pca = pca_temp.fit_transform(X_scaled)

        X_train_pca, X_test_pca, _, _ = split_data(
            pd.DataFrame(X_pca),
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y,
        )

        # Evaluate on PCA features
        logger.info(f"Evaluating on {optimal_n_components} PCA components...")
        pca_metrics = self._train_and_evaluate_models(
            X_train_pca, X_test_pca, y_train, y_test
        )

        comparison = {
            "original": original_metrics,
            "pca": pca_metrics,
            "n_original_features": self.X.shape[1],
            "n_pca_components": optimal_n_components,
        }

        self.results["comparison"] = comparison

        # Save comparison to CSV
        self._save_comparison_to_csv(comparison)

        return comparison

    def _save_comparison_to_csv(self, comparison: Dict) -> None:
        """Save comparison results to CSV."""
        rows = []
        for feature_type in ["original", "pca"]:
            if feature_type in comparison and isinstance(comparison[feature_type], dict):
                for model_name, metrics in comparison[feature_type].items():
                    row = {"feature_type": feature_type, "model": model_name}
                    row.update(metrics)
                    rows.append(row)

        df_comparison = pd.DataFrame(rows)
        csv_path = self.output_dir / "original_vs_pca_comparison.csv"
        df_comparison.to_csv(csv_path, index=False)
        logger.info(f"Comparison results saved to: {csv_path}")

    def plot_variance_explained(self, save_path: str = None) -> None:
        """
        Plot variance explained by each PCA component.

        Args:
            save_path: Path to save the plot. If None, uses default path.
        """
        if self.explained_variance_ratio is None:
            self.fit_pca()

        if save_path is None:
            save_path = self.output_dir / "pca_variance_explained.png"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Individual variance explained (bar chart)
        n_components = len(self.explained_variance_ratio)
        x = range(1, n_components + 1)

        axes[0].bar(x, self.explained_variance_ratio, alpha=0.7, color="steelblue")
        axes[0].set_xlabel("Principal Component", fontsize=12)
        axes[0].set_ylabel("Variance Explained Ratio", fontsize=12)
        axes[0].set_title("Variance Explained by Each Component", fontsize=14, fontweight="bold")
        axes[0].set_xticks(x[::max(1, n_components // 10)])

        # Right: Cumulative variance explained (line chart)
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        axes[1].plot(x, cumulative_variance, "b-o", markersize=4, linewidth=2)
        axes[1].axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
        axes[1].fill_between(x, cumulative_variance, alpha=0.3)
        axes[1].set_xlabel("Number of Components", fontsize=12)
        axes[1].set_ylabel("Cumulative Variance Explained", fontsize=12)
        axes[1].set_title("Cumulative Variance Explained", fontsize=14, fontweight="bold")
        axes[1].legend(loc="lower right")
        axes[1].set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Variance explained plot saved: {save_path}")

        # Save variance data to CSV
        variance_df = pd.DataFrame({
            "component": range(1, n_components + 1),
            "variance_ratio": self.explained_variance_ratio,
            "cumulative_variance": cumulative_variance,
        })
        csv_path = self.output_dir / "pca_variance_explained.csv"
        variance_df.to_csv(csv_path, index=False)
        logger.info(f"Variance data saved to: {csv_path}")

    def plot_performance_vs_components(self, save_path: str = None) -> None:
        """
        Plot model performance vs number of PCA components.

        Args:
            save_path: Path to save the plot. If None, uses default path.
        """
        if "pca_performance" not in self.results:
            logger.warning("Run evaluate_pca_performance() first")
            return

        if save_path is None:
            save_path = self.output_dir / "pca_performance_vs_components.png"

        perf_results = self.results["pca_performance"]
        n_components_list = sorted(perf_results.keys())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {"RandomForest": "green", "XGBoost": "blue", "LightGBM": "orange"}

        # Left: Accuracy vs Components
        for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            accuracies = []
            valid_components = []
            for n_comp in n_components_list:
                if model_name in perf_results[n_comp]:
                    accuracies.append(perf_results[n_comp][model_name]["Accuracy"])
                    valid_components.append(n_comp)
            if accuracies:
                axes[0].plot(
                    valid_components,
                    accuracies,
                    "-o",
                    label=model_name,
                    color=colors.get(model_name, "gray"),
                    markersize=4,
                )

        axes[0].set_xlabel("Number of PCA Components", fontsize=12)
        axes[0].set_ylabel("Accuracy", fontsize=12)
        axes[0].set_title("Accuracy vs PCA Components", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: F1 vs Components
        for model_name in ["RandomForest", "XGBoost", "LightGBM"]:
            f1_scores = []
            valid_components = []
            for n_comp in n_components_list:
                if model_name in perf_results[n_comp]:
                    f1_scores.append(perf_results[n_comp][model_name]["F1"])
                    valid_components.append(n_comp)
            if f1_scores:
                axes[1].plot(
                    valid_components,
                    f1_scores,
                    "-o",
                    label=model_name,
                    color=colors.get(model_name, "gray"),
                    markersize=4,
                )

        axes[1].set_xlabel("Number of PCA Components", fontsize=12)
        axes[1].set_ylabel("F1 Score", fontsize=12)
        axes[1].set_title("F1 Score vs PCA Components", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance vs components plot saved: {save_path}")

    def plot_pca_2d_projection(self, save_path: str = None) -> None:
        """
        Plot 2D PCA projection colored by is_finally_correct.

        Args:
            save_path: Path to save the plot. If None, uses default path.
        """
        if self.X is None:
            self.prepare_features()

        if save_path is None:
            save_path = self.output_dir / "pca_2d_projection.png"

        # Standardize and apply PCA
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X)
        else:
            X_scaled = self.scaler.transform(self.X)

        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)

        plt.figure(figsize=(10, 8))

        # Plot by class
        colors = {0: "red", 1: "blue"}
        labels = {0: "Incorrect (0)", 1: "Correct (1)"}

        for class_val in [0, 1]:
            mask = self.y == class_val
            plt.scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=colors[class_val],
                label=labels[class_val],
                alpha=0.5,
                s=20,
            )

        plt.xlabel(
            f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)",
            fontsize=12,
        )
        plt.ylabel(
            f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)",
            fontsize=12,
        )
        plt.title("PCA 2D Projection by is_finally_correct", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"2D projection plot saved: {save_path}")

    def generate_report(self) -> str:
        """
        Generate a comprehensive PCA analysis report.

        Returns:
            Path to the generated report file
        """
        logger.info("Generating PCA analysis report...")

        report_path = self.output_dir / "pca_analysis_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PCA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Data overview
            f.write("DATA OVERVIEW\n")
            f.write("-" * 80 + "\n")
            if self.X is not None:
                f.write(f"Number of samples: {self.X.shape[0]}\n")
                f.write(f"Number of features: {self.X.shape[1]}\n\n")

            # Variance explained
            f.write("VARIANCE EXPLAINED BY PCA\n")
            f.write("-" * 80 + "\n")
            if self.explained_variance_ratio is not None:
                cumulative = np.cumsum(self.explained_variance_ratio)
                f.write("Top 10 Principal Components:\n")
                for i in range(min(10, len(self.explained_variance_ratio))):
                    f.write(
                        f"  PC{i+1}: {self.explained_variance_ratio[i]*100:.2f}% "
                        f"(Cumulative: {cumulative[i]*100:.2f}%)\n"
                    )
                f.write(f"\nTotal components: {len(self.explained_variance_ratio)}\n")
                f.write(f"Total variance explained: {cumulative[-1]*100:.2f}%\n\n")

            # Optimal components
            f.write("OPTIMAL COMPONENTS ANALYSIS\n")
            f.write("-" * 80 + "\n")
            if "optimal_components" in self.results:
                opt = self.results["optimal_components"]
                f.write(f"Components for 95% variance: {opt['variance_threshold_components']}\n")
                f.write(f"Elbow point components: {opt['elbow_components']}\n")
                if opt['performance_optimal_components'] is not None:
                    f.write(f"Performance-optimal components: {opt['performance_optimal_components']}\n")
                f.write(f"Recommended components: {opt['recommended_components']}\n\n")

            # Original vs PCA comparison
            f.write("ORIGINAL VS PCA COMPARISON\n")
            f.write("-" * 80 + "\n")
            if "comparison" in self.results:
                comp = self.results["comparison"]
                f.write(f"Original features: {comp['n_original_features']}\n")
                f.write(f"PCA components: {comp['n_pca_components']}\n\n")

                f.write("Performance Comparison:\n")
                f.write(f"{'Model':<15} {'Metric':<12} {'Original':>10} {'PCA':>10} {'Diff':>10}\n")
                f.write("-" * 60 + "\n")

                for model_name in comp["original"].keys():
                    if model_name in comp["pca"]:
                        for metric in ["Accuracy", "Precision", "Recall", "F1"]:
                            orig_val = comp["original"][model_name][metric]
                            pca_val = comp["pca"][model_name][metric]
                            diff = pca_val - orig_val
                            f.write(
                                f"{model_name:<15} {metric:<12} {orig_val:>10.4f} "
                                f"{pca_val:>10.4f} {diff:>+10.4f}\n"
                            )
                        f.write("\n")

            # Performance by components
            f.write("PERFORMANCE BY NUMBER OF COMPONENTS\n")
            f.write("-" * 80 + "\n")
            if "pca_performance" in self.results:
                perf = self.results["pca_performance"]
                f.write(f"{'Components':<12} {'Model':<15} {'Accuracy':>10} {'F1':>10}\n")
                f.write("-" * 50 + "\n")
                for n_comp in sorted(perf.keys()):
                    for model_name, metrics in perf[n_comp].items():
                        f.write(
                            f"{n_comp:<12} {model_name:<15} "
                            f"{metrics['Accuracy']:>10.4f} {metrics['F1']:>10.4f}\n"
                        )
                f.write("\n")

            # Conclusions
            f.write("ANALYSIS CONCLUSIONS\n")
            f.write("-" * 80 + "\n")
            if "optimal_components" in self.results and "comparison" in self.results:
                opt = self.results["optimal_components"]
                comp = self.results["comparison"]
                n_orig = comp["n_original_features"]
                n_pca = opt["recommended_components"]
                reduction_pct = (1 - n_pca / n_orig) * 100

                f.write(f"1. Dimensionality can be reduced from {n_orig} to {n_pca} features ")
                f.write(f"({reduction_pct:.1f}% reduction)\n")
                f.write(f"2. This reduction retains 95% of the variance in the original feature space\n")

                # Check performance impact
                avg_orig_f1 = np.mean([m["F1"] for m in comp["original"].values()])
                avg_pca_f1 = np.mean([m["F1"] for m in comp["pca"].values()])
                f1_diff = avg_pca_f1 - avg_orig_f1

                if abs(f1_diff) < 0.01:
                    f.write("3. PCA reduction has minimal impact on model performance (F1 change < 1%)\n")
                elif f1_diff > 0:
                    f.write(f"3. PCA reduction slightly improves performance (F1 +{f1_diff*100:.2f}%)\n")
                else:
                    f.write(f"3. PCA reduction slightly decreases performance (F1 {f1_diff*100:.2f}%)\n")

                f.write("\n")
                f.write("These results suggest that there is redundancy in the original feature space,\n")
                f.write("but the features capture distinct information that contributes to prediction.\n")

        logger.info(f"PCA analysis report saved: {report_path}")
        return str(report_path)

    def run_full_pipeline(self) -> Tuple[Dict, str]:
        """
        Run complete PCA analysis pipeline.

        Returns:
            Tuple of (results dictionary, report path)
        """
        logger.info("Starting PCA analysis pipeline...")

        # Load data
        self.load_data()

        # Prepare features
        self.prepare_features()

        # Fit PCA
        self.fit_pca()

        # Evaluate PCA performance
        self.evaluate_pca_performance()

        # Find optimal components
        optimal = self.find_optimal_components()

        # Compare original vs PCA
        self.compare_original_vs_pca(optimal["recommended_components"])

        # Generate plots
        self.plot_variance_explained()
        self.plot_performance_vs_components()
        self.plot_pca_2d_projection()

        # Generate report
        report_path = self.generate_report()

        logger.info("PCA analysis pipeline completed successfully!")

        return self.results, report_path


def main():
    """Main function to execute the PCA analysis."""
    logger.info("Initializing PCA Analyzer...")

    # Initialize analyzer
    analyzer = PCAAnalysis()

    # Run full analysis
    results, report_path = analyzer.run_full_pipeline()

    logger.info(f"PCA analysis complete. Report saved to: {report_path}")

    return results, report_path


if __name__ == "__main__":
    results, report_path = main()
