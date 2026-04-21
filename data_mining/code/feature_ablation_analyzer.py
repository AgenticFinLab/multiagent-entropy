"""
Feature Ablation Analyzer for Multi-Agent Entropy Analysis

This module performs feature ablation experiments to evaluate feature importance
and validate the effectiveness of the feature set for classification tasks.
Uses multiple methods: SHAP, tree importance, RFE, and statistical tests.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE, chi2, mutual_info_classif, f_classif
from sklearn.metrics import accuracy_score, f1_score

from utils import split_data
from base import BaseAnalyzer, ModelFactory, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE

# Aliases preserved for any downstream / internal references
if XGBOOST_AVAILABLE:
    import xgboost as xgb
if LIGHTGBM_AVAILABLE:
    import lightgbm as lgb

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureAblationAnalyzer(BaseAnalyzer):
    """Performs feature ablation experiments for multi-agent entropy classification."""

    target_column = "is_finally_correct"
    analyzer_type = "feature_ablation"
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

    def compute_feature_importance_shap(
        self, model, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Compute feature importance using SHAP values.

        Args:
            model: Trained tree-based model
            X_train: Training feature matrix
            X_test: Test feature matrix

        Returns:
            DataFrame with Feature and Importance columns, or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP importance calculation")
            return None

        try:
            logger.info("Computing SHAP feature importance...")
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Handle binary classification (shap_values may be a list)
            if isinstance(shap_values, list):
                # For binary classification, use class 1 importance
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Calculate mean absolute SHAP value per feature
            feature_importance = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                "Feature": X_test.columns.tolist(),
                "Importance": feature_importance
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

            logger.info("SHAP importance calculation completed")
            return importance_df

        except Exception as e:
            logger.warning(f"SHAP calculation failed: {str(e)}")
            return None

    def compute_feature_importance_coefficients(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """
        Compute feature importance using Logistic Regression coefficients.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            DataFrame with Feature and Importance columns
        """
        logger.info("Computing Logistic Regression coefficient importance...")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Logistic Regression
        lr_model = LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, solver="lbfgs"
        )
        lr_model.fit(X_scaled, y)

        # Get absolute coefficients as importance
        importance = np.abs(lr_model.coef_[0])

        importance_df = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Importance": importance
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        logger.info("Logistic Regression coefficient importance completed")
        return importance_df

    def compute_feature_importance_tree(
        self, model, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.

        Args:
            model: Trained tree model with feature_importances_ attribute
            feature_names: List of feature names

        Returns:
            DataFrame with Feature and Importance columns
        """
        logger.info("Extracting tree-based feature importance...")

        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        return importance_df

    def rank_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Rank features using multiple methods and compute average ranking.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Tuple of (combined ranking DataFrame, dict of individual method rankings)
        """
        logger.info("=" * 60)
        logger.info("Computing feature rankings using multiple methods...")
        logger.info("=" * 60)

        # Split data for SHAP calculation
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train RandomForest for tree importance and SHAP
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        rankings = {}

        # Method 1: Tree-based importance
        tree_importance = self.compute_feature_importance_tree(
            rf_model, X.columns.tolist()
        )
        tree_importance["Rank_Tree"] = range(1, len(tree_importance) + 1)
        rankings["tree"] = tree_importance.copy()

        # Method 2: SHAP importance
        shap_importance = self.compute_feature_importance_shap(
            rf_model, X_train, X_test
        )
        if shap_importance is not None:
            shap_importance["Rank_SHAP"] = range(1, len(shap_importance) + 1)
            rankings["shap"] = shap_importance.copy()

        # Method 3: Logistic Regression coefficients
        lr_importance = self.compute_feature_importance_coefficients(X, y)
        lr_importance["Rank_LR"] = range(1, len(lr_importance) + 1)
        rankings["logistic_regression"] = lr_importance.copy()

        # Combine rankings
        combined = tree_importance[["Feature", "Rank_Tree"]].copy()
        combined = combined.merge(
            lr_importance[["Feature", "Rank_LR"]], on="Feature", how="left"
        )

        if "shap" in rankings:
            combined = combined.merge(
                shap_importance[["Feature", "Rank_SHAP"]], on="Feature", how="left"
            )
            combined["Avg_Rank"] = combined[["Rank_Tree", "Rank_SHAP", "Rank_LR"]].mean(axis=1)
        else:
            combined["Avg_Rank"] = combined[["Rank_Tree", "Rank_LR"]].mean(axis=1)

        # Sort by average rank
        combined = combined.sort_values("Avg_Rank").reset_index(drop=True)
        combined["Combined_Rank"] = range(1, len(combined) + 1)

        # Store results
        self.results["feature_rankings"] = combined
        self.results["individual_rankings"] = rankings

        # Save to CSV
        combined.to_csv(self.output_dir / "feature_rankings_combined.csv", index=False)
        for method, df in rankings.items():
            df.to_csv(self.output_dir / f"feature_rankings_{method}.csv", index=False)

        logger.info(f"Feature ranking completed. Top 5 features: {combined['Feature'].head(5).tolist()}")

        return combined, rankings

    def run_rfe(
        self, X: pd.DataFrame, y: pd.Series, n_features_range: List[int] = None
    ) -> Dict:
        """
        Run Recursive Feature Elimination with different feature counts.

        Args:
            X: Feature matrix
            y: Target variable
            n_features_range: List of feature counts to evaluate (default: auto-generated)

        Returns:
            Dictionary with RFE results
        """
        logger.info("=" * 60)
        logger.info("Running Recursive Feature Elimination (RFE)...")
        logger.info("=" * 60)

        n_total_features = X.shape[1]

        # Generate feature range if not provided
        if n_features_range is None:
            # Use 10 steps from 1 to total features
            step = max(1, n_total_features // 10)
            n_features_range = list(range(1, n_total_features + 1, step))
            if n_total_features not in n_features_range:
                n_features_range.append(n_total_features)
            n_features_range = sorted(n_features_range)

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = {
            "n_features": [],
            "accuracy": [],
            "f1": [],
            "selected_features": []
        }

        base_estimator = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )

        for n_features in n_features_range:
            logger.info(f"RFE with {n_features} features...")

            # Create and fit RFE
            rfe = RFE(
                estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                n_features_to_select=n_features,
                step=1
            )
            rfe.fit(X_train, y_train)

            # Get selected features
            selected_mask = rfe.support_
            selected_features = X.columns[selected_mask].tolist()

            # Train model on selected features
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train[selected_features], y_train)
            y_pred = rf_model.predict(X_test[selected_features])

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results["n_features"].append(n_features)
            results["accuracy"].append(accuracy)
            results["f1"].append(f1)
            results["selected_features"].append(selected_features)

            logger.info(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # Store results
        self.results["rfe"] = results

        # Save to CSV
        rfe_df = pd.DataFrame({
            "n_features": results["n_features"],
            "accuracy": results["accuracy"],
            "f1": results["f1"]
        })
        rfe_df.to_csv(self.output_dir / "rfe_results.csv", index=False)

        return results

    def run_statistical_selection(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, pd.DataFrame]:
        """
        Run statistical feature selection methods.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary with chi2, mutual_info, and f_classif results
        """
        logger.info("=" * 60)
        logger.info("Running statistical feature selection...")
        logger.info("=" * 60)

        results = {}

        # Chi2 (requires non-negative values)
        logger.info("Computing chi2 scores...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        chi2_scores, chi2_pvalues = chi2(X_scaled, y)
        results["chi2"] = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Score": chi2_scores,
            "P_Value": chi2_pvalues
        }).sort_values("Score", ascending=False).reset_index(drop=True)

        # Mutual Information
        logger.info("Computing mutual information scores...")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        results["mutual_info"] = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Score": mi_scores
        }).sort_values("Score", ascending=False).reset_index(drop=True)

        # ANOVA F-value
        logger.info("Computing ANOVA F-scores...")
        f_scores, f_pvalues = f_classif(X, y)
        results["f_classif"] = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Score": f_scores,
            "P_Value": f_pvalues
        }).sort_values("Score", ascending=False).reset_index(drop=True)

        # Store results
        self.results["statistical_selection"] = results

        # Save to CSV
        for method, df in results.items():
            df.to_csv(self.output_dir / f"statistical_selection_{method}.csv", index=False)

        logger.info("Statistical feature selection completed")

        return results

    def run_ablation_study(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        importance_ranking: pd.DataFrame
    ) -> Dict:
        """
        Run ablation study by progressively removing least important features.

        Args:
            X: Feature matrix
            y: Target variable
            importance_ranking: DataFrame with feature importance ranking

        Returns:
            Dictionary with ablation results
        """
        logger.info("=" * 60)
        logger.info("Running feature ablation study...")
        logger.info("=" * 60)

        # Get features sorted by importance (least to most important)
        features_by_importance = importance_ranking["Feature"].tolist()[::-1]

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = {
            "n_remaining_features": [],
            "removed_feature": [],
            "accuracy": [],
            "f1": []
        }

        current_features = X.columns.tolist()

        # Initial performance with all features
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train[current_features], y_train)
        y_pred = rf_model.predict(X_test[current_features])

        results["n_remaining_features"].append(len(current_features))
        results["removed_feature"].append(None)
        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["f1"].append(f1_score(y_test, y_pred))

        # Progressively remove features
        for feature_to_remove in features_by_importance:
            if feature_to_remove not in current_features:
                continue

            current_features = [f for f in current_features if f != feature_to_remove]

            if len(current_features) == 0:
                break

            # Train and evaluate
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train[current_features], y_train)
            y_pred = rf_model.predict(X_test[current_features])

            results["n_remaining_features"].append(len(current_features))
            results["removed_feature"].append(feature_to_remove)
            results["accuracy"].append(accuracy_score(y_test, y_pred))
            results["f1"].append(f1_score(y_test, y_pred))

            logger.info(
                f"Removed '{feature_to_remove}': {len(current_features)} features remain, "
                f"Acc={results['accuracy'][-1]:.4f}, F1={results['f1'][-1]:.4f}"
            )

        # Store results
        self.results["ablation"] = results

        # Save to CSV
        ablation_df = pd.DataFrame(results)
        ablation_df.to_csv(self.output_dir / "ablation_results.csv", index=False)

        return results

    def plot_feature_importance_ranking(self, save_path: str = None):
        """
        Plot feature importance ranking as horizontal bar chart.

        Args:
            save_path: Path to save the plot
        """
        if "feature_rankings" not in self.results:
            logger.warning("No feature rankings available. Run rank_features first.")
            return

        if save_path is None:
            save_path = self.output_dir / "feature_importance_ranking.png"

        rankings = self.results["feature_rankings"]
        individual = self.results.get("individual_rankings", {})

        # Create figure with subplots
        n_methods = len(individual) + 1
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 10))

        if n_methods == 1:
            axes = [axes]

        # Plot combined ranking
        top20 = rankings.head(20)
        ax = axes[0]
        ax.barh(range(len(top20)), top20["Avg_Rank"].values[::-1])
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["Feature"].values[::-1])
        ax.set_xlabel("Average Rank (lower is more important)")
        ax.set_title("Combined Feature Ranking (Top 20)", fontsize=12, fontweight="bold")
        ax.invert_xaxis()

        # Plot individual method rankings
        for idx, (method, df) in enumerate(individual.items(), 1):
            if idx >= len(axes):
                break
            ax = axes[idx]
            top20 = df.head(20)
            ax.barh(range(len(top20)), top20["Importance"].values[::-1])
            ax.set_yticks(range(len(top20)))
            ax.set_yticklabels(top20["Feature"].values[::-1])
            ax.set_xlabel("Importance Score")
            ax.set_title(f"{method.upper()} Importance (Top 20)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Feature importance ranking plot saved: {save_path}")

    def plot_ablation_curve(self, save_path: str = None):
        """
        Plot accuracy and F1 curves as features are removed.

        Args:
            save_path: Path to save the plot
        """
        if "ablation" not in self.results:
            logger.warning("No ablation results available. Run run_ablation_study first.")
            return

        if save_path is None:
            save_path = self.output_dir / "ablation_curve.png"

        ablation = self.results["ablation"]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = ablation["n_remaining_features"]
        ax.plot(x, ablation["accuracy"], "b-o", label="Accuracy", markersize=4)
        ax.plot(x, ablation["f1"], "r-s", label="F1 Score", markersize=4)

        ax.set_xlabel("Number of Remaining Features", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Feature Ablation Study: Performance vs. Feature Count", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Invert x-axis so it goes from all features to few features
        ax.invert_xaxis()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Ablation curve plot saved: {save_path}")

    def plot_rfe_performance(self, save_path: str = None):
        """
        Plot RFE performance curves.

        Args:
            save_path: Path to save the plot
        """
        if "rfe" not in self.results:
            logger.warning("No RFE results available. Run run_rfe first.")
            return

        if save_path is None:
            save_path = self.output_dir / "rfe_performance.png"

        rfe = self.results["rfe"]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(rfe["n_features"], rfe["accuracy"], "b-o", label="Accuracy", markersize=6)
        ax.plot(rfe["n_features"], rfe["f1"], "r-s", label="F1 Score", markersize=6)

        ax.set_xlabel("Number of Features", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("RFE: Performance vs. Number of Features", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Find optimal number of features
        best_idx = np.argmax(rfe["f1"])
        best_n = rfe["n_features"][best_idx]
        ax.axvline(x=best_n, color="green", linestyle="--", alpha=0.7, label=f"Best: {best_n} features")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"RFE performance plot saved: {save_path}")

    def plot_statistical_selection_scores(self, save_path: str = None):
        """
        Plot statistical selection scores for each method.

        Args:
            save_path: Path to save the plot
        """
        if "statistical_selection" not in self.results:
            logger.warning("No statistical selection results. Run run_statistical_selection first.")
            return

        if save_path is None:
            save_path = self.output_dir / "statistical_selection_scores.png"

        stats = self.results["statistical_selection"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        methods = ["chi2", "mutual_info", "f_classif"]
        titles = ["Chi-Square Test", "Mutual Information", "ANOVA F-Value"]

        for ax, method, title in zip(axes, methods, titles):
            if method not in stats:
                continue

            df = stats[method].head(20)
            ax.barh(range(len(df)), df["Score"].values[::-1])
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["Feature"].values[::-1])
            ax.set_xlabel("Score")
            ax.set_title(f"{title} (Top 20)", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Statistical selection scores plot saved: {save_path}")

    def generate_report(self) -> str:
        """
        Generate comprehensive feature ablation report.

        Returns:
            Path to the generated report
        """
        logger.info("Generating feature ablation report...")

        report_path = self.output_dir / "feature_ablation_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE ABLATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Data overview
            f.write("DATA OVERVIEW\n")
            f.write("-" * 80 + "\n")
            if self.df is not None:
                f.write(f"Total samples: {len(self.df)}\n")
                if "is_finally_correct" in self.df.columns:
                    positive_rate = self.df["is_finally_correct"].mean()
                    f.write(f"Positive class rate: {positive_rate:.4f}\n")
            f.write("\n")

            # Feature importance rankings
            if "feature_rankings" in self.results:
                f.write("FEATURE IMPORTANCE RANKINGS\n")
                f.write("-" * 80 + "\n\n")

                f.write("Combined Ranking (Top 20):\n")
                rankings = self.results["feature_rankings"]
                for idx, row in rankings.head(20).iterrows():
                    f.write(f"  {row['Combined_Rank']:2d}. {row['Feature']}: Avg Rank = {row['Avg_Rank']:.2f}\n")
                f.write("\n")

                # Individual method rankings
                if "individual_rankings" in self.results:
                    for method, df in self.results["individual_rankings"].items():
                        f.write(f"{method.upper()} Method (Top 20):\n")
                        for idx, row in df.head(20).iterrows():
                            f.write(f"  {idx+1:2d}. {row['Feature']}: {row['Importance']:.6f}\n")
                        f.write("\n")

            # RFE results
            if "rfe" in self.results:
                f.write("RFE RESULTS SUMMARY\n")
                f.write("-" * 80 + "\n\n")

                rfe = self.results["rfe"]
                best_idx = np.argmax(rfe["f1"])
                best_n = rfe["n_features"][best_idx]
                best_acc = rfe["accuracy"][best_idx]
                best_f1 = rfe["f1"][best_idx]

                f.write(f"Recommended number of features: {best_n}\n")
                f.write(f"  Accuracy at optimal: {best_acc:.4f}\n")
                f.write(f"  F1 Score at optimal: {best_f1:.4f}\n\n")

                f.write("Performance at different feature counts:\n")
                for i in range(len(rfe["n_features"])):
                    f.write(f"  {rfe['n_features'][i]:3d} features: Acc={rfe['accuracy'][i]:.4f}, F1={rfe['f1'][i]:.4f}\n")
                f.write("\n")

            # Statistical selection
            if "statistical_selection" in self.results:
                f.write("STATISTICAL SELECTION RESULTS\n")
                f.write("-" * 80 + "\n\n")

                for method, df in self.results["statistical_selection"].items():
                    f.write(f"{method.upper()} Top 10:\n")
                    for idx, row in df.head(10).iterrows():
                        score_str = f"{row['Score']:.6f}"
                        if "P_Value" in row:
                            score_str += f" (p={row['P_Value']:.2e})"
                        f.write(f"  {idx+1:2d}. {row['Feature']}: {score_str}\n")
                    f.write("\n")

            # Ablation study
            if "ablation" in self.results:
                f.write("ABLATION STUDY RESULTS\n")
                f.write("-" * 80 + "\n\n")

                ablation = self.results["ablation"]
                initial_acc = ablation["accuracy"][0]
                initial_f1 = ablation["f1"][0]

                f.write(f"Initial performance (all {ablation['n_remaining_features'][0]} features):\n")
                f.write(f"  Accuracy: {initial_acc:.4f}\n")
                f.write(f"  F1 Score: {initial_f1:.4f}\n\n")

                f.write("Features causing significant performance drop when removed:\n")
                significant_drops = []
                for i in range(1, len(ablation["accuracy"])):
                    acc_drop = ablation["accuracy"][i-1] - ablation["accuracy"][i]
                    f1_drop = ablation["f1"][i-1] - ablation["f1"][i]
                    if acc_drop > 0.01 or f1_drop > 0.01:
                        significant_drops.append({
                            "feature": ablation["removed_feature"][i],
                            "acc_drop": acc_drop,
                            "f1_drop": f1_drop
                        })

                if significant_drops:
                    for drop in significant_drops:
                        f.write(f"  - {drop['feature']}: Acc drop={drop['acc_drop']:.4f}, F1 drop={drop['f1_drop']:.4f}\n")
                else:
                    f.write("  No individual feature removal caused >1% performance drop.\n")
                f.write("\n")

            # Conclusions
            f.write("CONCLUSIONS\n")
            f.write("-" * 80 + "\n\n")

            if "rfe" in self.results and "ablation" in self.results:
                rfe = self.results["rfe"]
                ablation = self.results["ablation"]

                # Calculate redundancy
                full_features = ablation["n_remaining_features"][0]
                best_rfe_idx = np.argmax(rfe["f1"])
                optimal_features = rfe["n_features"][best_rfe_idx]
                redundancy_ratio = 1 - (optimal_features / full_features)

                f.write(f"1. Feature redundancy analysis:\n")
                f.write(f"   - Total features: {full_features}\n")
                f.write(f"   - Optimal feature subset (by RFE): {optimal_features}\n")
                f.write(f"   - Estimated redundancy: {redundancy_ratio*100:.1f}%\n\n")

                if redundancy_ratio > 0.5:
                    f.write("   FINDING: Significant feature redundancy detected. Feature selection recommended.\n\n")
                elif redundancy_ratio > 0.2:
                    f.write("   FINDING: Moderate feature redundancy detected. Some features may be pruned.\n\n")
                else:
                    f.write("   FINDING: Low feature redundancy. Most features contribute to model performance.\n\n")

                f.write("2. Feature importance findings:\n")
                if "feature_rankings" in self.results:
                    top5 = self.results["feature_rankings"]["Feature"].head(5).tolist()
                    f.write(f"   - Top 5 most important features: {', '.join(top5)}\n\n")

                f.write("3. Recommendation:\n")
                if redundancy_ratio > 0.3:
                    f.write(f"   Consider reducing feature set to top {optimal_features} features for improved model interpretability.\n")
                else:
                    f.write("   The current feature set is effective with minimal redundancy.\n")

        logger.info(f"Report saved: {report_path}")
        return str(report_path)

    def run_full_pipeline(self) -> Tuple[Dict, str]:
        """
        Run the complete feature ablation analysis pipeline.

        Returns:
            Tuple of (results dictionary, report path)
        """
        logger.info("=" * 80)
        logger.info("Starting Feature Ablation Analysis Pipeline")
        logger.info("=" * 80)

        # Load data
        self.load_data()

        # Prepare features
        X, y = self.prepare_features(target_column="is_finally_correct")
        logger.info(f"Prepared {X.shape[1]} features, {X.shape[0]} samples")

        # Step 1: Rank features
        combined_ranking, individual_rankings = self.rank_features(X, y)

        # Step 2: Run RFE
        rfe_results = self.run_rfe(X, y)

        # Step 3: Statistical selection
        stat_results = self.run_statistical_selection(X, y)

        # Step 4: Ablation study
        ablation_results = self.run_ablation_study(X, y, combined_ranking)

        # Step 5: Generate plots
        self.plot_feature_importance_ranking()
        self.plot_ablation_curve()
        self.plot_rfe_performance()
        self.plot_statistical_selection_scores()

        # Step 6: Generate report
        report_path = self.generate_report()

        logger.info("=" * 80)
        logger.info("Feature Ablation Analysis Pipeline Completed Successfully!")
        logger.info("=" * 80)

        return self.results, report_path


def main():
    """Main function to execute the feature ablation analysis."""
    logger.info("Initializing Feature Ablation Analyzer...")

    # Initialize analyzer
    analyzer = FeatureAblationAnalyzer()

    # Run full analysis
    results, report_path = analyzer.run_full_pipeline()

    logger.info(f"Feature ablation analysis complete. Report saved to: {report_path}")

    return results, report_path


if __name__ == "__main__":
    results, report_path = main()
