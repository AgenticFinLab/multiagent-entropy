#!/usr/bin/env python3
"""
Temperature Ablation Data Mining Analysis

This module performs comprehensive data mining analysis on temperature ablation
experiments, including accuracy comparison, factor identification, and entropy
distribution analysis across different temperature settings.
"""
import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from features import DEFAULT_EXCLUDE_COLUMNS
from utils import (
    setup_visualization_style,
    encode_categorical_features,
    create_output_directory,
    get_exclude_columns_from_config,
)

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AccuracyComparisonAnalyzer:
    """Analyze accuracy across temperatures for single-agent vs multi-agent systems."""

    def __init__(self, data_dir: str, output_dir: str, temperatures: List[float]):
        """
        Initialize the AccuracyComparisonAnalyzer.

        Args:
            data_dir: Path to evaluation/results_temp directory
            output_dir: Directory to save analysis results
            temperatures: List of temperature values to analyze
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.temperatures = sorted(temperatures)
        self.temperature_data: Dict[float, pd.DataFrame] = {}

        create_output_directory(self.output_dir)
        setup_visualization_style()

    def load_temperature_data(self, dataset: str = "math500", model: str = "qwen3_4b") -> Dict[float, pd.DataFrame]:
        """
        Load CSV data for each temperature from data_dir/dataset/t_{temp}/all_aggregated_data_exclude_agent.csv

        Args:
            dataset: Dataset name (default: math500)
            model: Model name (default: qwen3_4b)

        Returns:
            Dictionary mapping temperature to DataFrame
        """
        logger.info("Loading temperature data for accuracy comparison...")

        for temp in self.temperatures:
            csv_path = self.data_dir / dataset / f"t_{temp}" / "all_aggregated_data_exclude_agent.csv"

            if not csv_path.exists():
                logger.warning(f"Data file not found for temperature {temp}: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
                df["temperature"] = temp
                self.temperature_data[temp] = df
                logger.info(f"Loaded {len(df)} records for temperature {temp}")
            except Exception as e:
                logger.error(f"Error loading data for temperature {temp}: {e}")

        logger.info(f"Successfully loaded data for {len(self.temperature_data)} temperatures")
        return self.temperature_data

    def generate_accuracy_table(self) -> pd.DataFrame:
        """
        Generate temperature x architecture accuracy comparison table.

        Uses exp_accuracy from Aggregator to ensure consistency with the standard
        accuracy calculation (correct / samples_with_valid_predictions).

        Returns:
            DataFrame with columns: temperature, architecture, accuracy, num_samples, num_correct
        """
        logger.info("Generating accuracy comparison table...")

        if not self.temperature_data:
            logger.warning("No temperature data loaded. Call load_temperature_data first.")
            return pd.DataFrame()

        results = []
        architectures = ["single", "sequential", "centralized", "debate", "hybrid"]

        for temp, df in self.temperature_data.items():
            if "architecture" not in df.columns or "is_finally_correct" not in df.columns:
                logger.warning(f"Missing required columns for temperature {temp}")
                continue

            # Check if exp_accuracy column exists for Aggregator-consistent calculation
            has_exp_accuracy = "exp_accuracy" in df.columns

            # Per-architecture accuracy
            for arch in architectures:
                arch_df = df[df["architecture"] == arch]
                if len(arch_df) == 0:
                    continue

                num_correct = int(arch_df["is_finally_correct"].sum())

                if has_exp_accuracy:
                    # Use exp_accuracy from Aggregator (correct / samples_with_predictions)
                    accuracy = arch_df["exp_accuracy"].iloc[0]
                    # Back-calculate num_samples (samples with valid predictions)
                    if accuracy > 0:
                        num_samples = int(round(num_correct / accuracy))
                    else:
                        num_samples = 0
                else:
                    # Fallback to original calculation if exp_accuracy not available
                    num_samples = len(arch_df)
                    accuracy = arch_df["is_finally_correct"].mean()

                results.append({
                    "temperature": temp,
                    "architecture": arch,
                    "accuracy": accuracy,
                    "num_samples": num_samples,
                    "num_correct": num_correct,
                })

            # Multi-agent average (all architectures except single)
            multi_agent_df = df[df["architecture"] != "single"]
            if len(multi_agent_df) > 0:
                # Calculate weighted average accuracy for multi-agent architectures
                total_correct = 0
                total_samples = 0
                for arch in ["sequential", "centralized", "debate", "hybrid"]:
                    arch_df = multi_agent_df[multi_agent_df["architecture"] == arch]
                    if len(arch_df) == 0:
                        continue
                    arch_correct = int(arch_df["is_finally_correct"].sum())
                    total_correct += arch_correct
                    if has_exp_accuracy:
                        arch_accuracy = arch_df["exp_accuracy"].iloc[0]
                        if arch_accuracy > 0:
                            total_samples += int(round(arch_correct / arch_accuracy))
                    else:
                        total_samples += len(arch_df)

                if total_samples > 0:
                    multi_agent_accuracy = total_correct / total_samples
                else:
                    multi_agent_accuracy = 0.0

                results.append({
                    "temperature": temp,
                    "architecture": "multi_agent_avg",
                    "accuracy": multi_agent_accuracy,
                    "num_samples": total_samples,
                    "num_correct": total_correct,
                })

        accuracy_df = pd.DataFrame(results)

        # Save to CSV
        csv_path = self.output_dir / "accuracy_comparison_table.csv"
        accuracy_df.to_csv(csv_path, index=False)
        logger.info(f"Accuracy comparison table saved to: {csv_path}")

        return accuracy_df

    def run_significance_tests(self) -> pd.DataFrame:
        """
        McNemar test between temperature pairs for same samples.

        Returns:
            DataFrame with: temp_pair, architecture, statistic, p_value, significant
        """
        logger.info("Running significance tests between temperature pairs...")

        if not self.temperature_data:
            logger.warning("No temperature data loaded.")
            return pd.DataFrame()

        results = []
        temp_pairs = list(combinations(sorted(self.temperature_data.keys()), 2))
        architectures = ["single", "sequential", "centralized", "debate", "hybrid"]

        for temp1, temp2 in temp_pairs:
            df1 = self.temperature_data[temp1]
            df2 = self.temperature_data[temp2]

            for arch in architectures:
                try:
                    arch_df1 = df1[df1["architecture"] == arch]
                    arch_df2 = df2[df2["architecture"] == arch]

                    if len(arch_df1) == 0 or len(arch_df2) == 0:
                        continue

                    # Match samples by sample_id
                    merged = pd.merge(
                        arch_df1[["sample_id", "is_finally_correct"]],
                        arch_df2[["sample_id", "is_finally_correct"]],
                        on="sample_id",
                        suffixes=("_t1", "_t2"),
                    )

                    if len(merged) < 10:
                        logger.warning(f"Insufficient paired samples for {arch} at temps {temp1} vs {temp2}")
                        continue

                    # Build 2x2 contingency table for McNemar test
                    # [correct_both, correct_t1_only]
                    # [correct_t2_only, incorrect_both]
                    a = ((merged["is_finally_correct_t1"] == 1) & (merged["is_finally_correct_t2"] == 1)).sum()
                    b = ((merged["is_finally_correct_t1"] == 1) & (merged["is_finally_correct_t2"] == 0)).sum()
                    c = ((merged["is_finally_correct_t1"] == 0) & (merged["is_finally_correct_t2"] == 1)).sum()
                    d = ((merged["is_finally_correct_t1"] == 0) & (merged["is_finally_correct_t2"] == 0)).sum()

                    contingency_table = [[a, b], [c, d]]

                    if STATSMODELS_AVAILABLE:
                        result = mcnemar(contingency_table, exact=True)
                        statistic = result.statistic
                        p_value = result.pvalue
                    else:
                        # Fallback to chi-squared approximation
                        if b + c > 0:
                            statistic = (abs(b - c) - 1) ** 2 / (b + c)
                            p_value = stats.chi2.sf(statistic, df=1)
                        else:
                            statistic = 0.0
                            p_value = 1.0

                    results.append({
                        "temp_pair": f"{temp1}_vs_{temp2}",
                        "architecture": arch,
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "num_paired_samples": len(merged),
                    })

                except Exception as e:
                    logger.error(f"Error in significance test for {arch} at {temp1} vs {temp2}: {e}")

        significance_df = pd.DataFrame(results)

        # Save to CSV
        csv_path = self.output_dir / "significance_tests.csv"
        significance_df.to_csv(csv_path, index=False)
        logger.info(f"Significance tests saved to: {csv_path}")

        return significance_df

    def plot_accuracy_trends(self) -> str:
        """
        Plot temperature vs accuracy trend lines, one line per architecture.

        Returns:
            Path to saved plot
        """
        logger.info("Plotting accuracy trends across temperatures...")

        accuracy_df = self.generate_accuracy_table()

        if accuracy_df.empty:
            logger.warning("No accuracy data to plot")
            return ""

        plt.figure(figsize=(12, 8))
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Define markers and colors for each architecture
        markers = {"single": "o", "sequential": "s", "centralized": "^", "debate": "D", "hybrid": "p", "multi_agent_avg": "*"}
        colors = sns.color_palette("husl", n_colors=len(markers))
        color_map = {arch: colors[i] for i, arch in enumerate(markers.keys())}

        for arch in markers.keys():
            arch_data = accuracy_df[accuracy_df["architecture"] == arch].sort_values("temperature")
            if len(arch_data) > 0:
                label = "Multi-Agent (avg)" if arch == "multi_agent_avg" else arch.capitalize()
                linewidth = 2.5 if arch == "multi_agent_avg" else 1.5
                markersize = 12 if arch == "multi_agent_avg" else 8

                plt.plot(
                    arch_data["temperature"],
                    arch_data["accuracy"],
                    marker=markers[arch],
                    label=label,
                    color=color_map[arch],
                    linewidth=linewidth,
                    markersize=markersize,
                )

        plt.xlabel("Temperature", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title("Accuracy Trends Across Temperatures by Architecture", fontsize=16, fontweight="bold")
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(self.temperatures)

        plot_path = self.output_dir / "accuracy_trend_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Accuracy trend plot saved to: {plot_path}")
        return str(plot_path)

    def run_full_analysis(self, dataset: str = "math500") -> Dict:
        """
        Run all accuracy comparison analyses.

        Args:
            dataset: Dataset name to analyze

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Running full accuracy comparison analysis...")

        results = {}

        try:
            self.load_temperature_data(dataset)
            results["accuracy_table"] = self.generate_accuracy_table()
            results["significance_tests"] = self.run_significance_tests()
            results["trend_plot"] = self.plot_accuracy_trends()
        except Exception as e:
            logger.error(f"Error in accuracy comparison analysis: {e}")

        return results


class FactorIdentificationAnalyzer:
    """Identify key factors affecting MAS effectiveness at each temperature."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        temperatures: List[float],
        exclude_features: str = "default",
    ):
        """
        Initialize the FactorIdentificationAnalyzer.

        Args:
            data_dir: Path to evaluation/results_temp directory
            output_dir: Directory to save analysis results
            temperatures: List of temperature values to analyze
            exclude_features: Feature exclusion group name (see features.py FEATURE_GROUPS)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.temperatures = sorted(temperatures)
        self.exclude_features = exclude_features
        self.temperature_data: Dict[float, pd.DataFrame] = {}
        self.importance_results: Dict[float, Dict] = {}

        create_output_directory(self.output_dir)
        setup_visualization_style()

    def load_temperature_data(self, dataset: str = "math500") -> Dict[float, pd.DataFrame]:
        """Load CSV data for each temperature."""
        logger.info("Loading temperature data for factor identification...")

        for temp in self.temperatures:
            csv_path = self.data_dir / dataset / f"t_{temp}" / "all_aggregated_data_exclude_agent.csv"

            if not csv_path.exists():
                logger.warning(f"Data file not found for temperature {temp}: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
                df["temperature"] = temp
                self.temperature_data[temp] = df
                logger.info(f"Loaded {len(df)} records for temperature {temp}")
            except Exception as e:
                logger.error(f"Error loading data for temperature {temp}: {e}")

        return self.temperature_data

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling."""
        exclude_cols = get_exclude_columns_from_config(self.exclude_features)
        exclude_cols = list(exclude_cols) + ["dataset", "temperature"]

        # Get target
        target_col = "is_finally_correct"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")

        y = df[target_col].copy()

        # Handle categorical columns
        df_encoded = encode_categorical_features(df)

        # Get feature columns
        feature_cols = [
            col for col in df_encoded.columns
            if col not in exclude_cols and col != target_col
        ]

        # Select numeric features only
        X = df_encoded[feature_cols].select_dtypes(include=[np.number]).copy()

        # Fill NaN with median
        X = X.fillna(X.median())

        return X, y

    def train_classification_models(self, df: pd.DataFrame, temperature: float) -> Dict:
        """
        Train XGBoost and LightGBM classifiers for is_finally_correct.

        Args:
            df: Input DataFrame
            temperature: Temperature value for labeling

        Returns:
            Dict with models, feature importance DataFrames, and X_train/X_test
        """
        logger.info(f"Training classification models for temperature {temperature}...")

        X, y = self._prepare_features(df)

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "models": {},
            "importance": {},
        }

        # Train XGBoost
        if XGBOOST_AVAILABLE:
            try:
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
                results["models"]["XGBoost"] = xgb_model

                # Extract feature importance
                xgb_importance = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": xgb_model.feature_importances_,
                }).sort_values("importance", ascending=False)
                results["importance"]["XGBoost"] = xgb_importance

                logger.info(f"XGBoost trained for temperature {temperature}")
            except Exception as e:
                logger.error(f"Error training XGBoost for temperature {temperature}: {e}")

        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
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
                results["models"]["LightGBM"] = lgb_model

                # Extract feature importance
                lgb_importance = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": lgb_model.feature_importances_,
                }).sort_values("importance", ascending=False)
                results["importance"]["LightGBM"] = lgb_importance

                logger.info(f"LightGBM trained for temperature {temperature}")
            except Exception as e:
                logger.error(f"Error training LightGBM for temperature {temperature}: {e}")

        return results

    def extract_feature_importance(self) -> pd.DataFrame:
        """
        Extract and compare feature importance across temperatures.

        Returns:
            DataFrame with: feature, temperature, xgb_importance, lgb_importance, avg_importance, rank
        """
        logger.info("Extracting feature importance across temperatures...")

        all_results = []

        for temp, df in self.temperature_data.items():
            try:
                model_results = self.train_classification_models(df, temp)
                self.importance_results[temp] = model_results

                # Combine XGBoost and LightGBM importance
                xgb_imp = model_results["importance"].get("XGBoost", pd.DataFrame())
                lgb_imp = model_results["importance"].get("LightGBM", pd.DataFrame())

                if not xgb_imp.empty and not lgb_imp.empty:
                    # Merge importance scores
                    merged = pd.merge(
                        xgb_imp.rename(columns={"importance": "xgb_importance"}),
                        lgb_imp.rename(columns={"importance": "lgb_importance"}),
                        on="feature",
                        how="outer",
                    ).fillna(0)

                    merged["temperature"] = temp
                    merged["avg_importance"] = (merged["xgb_importance"] + merged["lgb_importance"]) / 2
                    merged["rank"] = merged["avg_importance"].rank(ascending=False).astype(int)

                    all_results.append(merged)

            except Exception as e:
                logger.error(f"Error extracting importance for temperature {temp}: {e}")

        if not all_results:
            return pd.DataFrame()

        importance_df = pd.concat(all_results, ignore_index=True)
        importance_df = importance_df.sort_values(["temperature", "rank"])

        # Save to CSV
        csv_path = self.output_dir / "feature_importance_all_temperatures.csv"
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"Feature importance saved to: {csv_path}")

        return importance_df

    def run_shap_analysis(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                          temperature: float, model_name: str):
        """
        Run SHAP TreeExplainer, generate summary plot per temperature.

        Args:
            model: Trained model
            X_train: Training features
            X_test: Test features
            temperature: Temperature value
            model_name: Name of the model
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP analysis.")
            return

        logger.info(f"Running SHAP analysis for {model_name} at temperature {temperature}...")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # For binary classification, take positive class
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]

            # Generate summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values,
                X_test,
                plot_type="dot",
                show=False,
                max_display=20,
            )

            plot_path = self.output_dir / f"shap_summary_t_{temperature}_{model_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"SHAP summary plot saved to: {plot_path}")

        except Exception as e:
            logger.error(f"Error in SHAP analysis for {model_name} at temperature {temperature}: {e}")
            plt.close()

    def compute_correlation_heatmap(self, df: pd.DataFrame, temperature: float):
        """
        Compute Pearson correlation matrix for top features, plot heatmap.

        Args:
            df: Input DataFrame
            temperature: Temperature value
        """
        logger.info(f"Computing correlation heatmap for temperature {temperature}...")

        try:
            X, y = self._prepare_features(df)

            # Get top 20 features by importance
            if temperature in self.importance_results:
                importance = self.importance_results[temperature].get("importance", {})
                if "XGBoost" in importance:
                    top_features = importance["XGBoost"].head(20)["feature"].tolist()
                elif "LightGBM" in importance:
                    top_features = importance["LightGBM"].head(20)["feature"].tolist()
                else:
                    top_features = X.columns[:20].tolist()
            else:
                top_features = X.columns[:20].tolist()

            # Filter to available features
            top_features = [f for f in top_features if f in X.columns]

            if len(top_features) < 2:
                logger.warning(f"Not enough features for correlation analysis at temperature {temperature}")
                return

            # Compute correlation matrix
            corr_matrix = X[top_features].corr()

            # Plot heatmap
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
            )
            plt.title(f"Feature Correlation Heatmap (Temperature {temperature})", fontsize=14, fontweight="bold")
            plt.tight_layout()

            plot_path = self.output_dir / f"feature_correlation_heatmap_t_{temperature}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Correlation heatmap saved to: {plot_path}")

        except Exception as e:
            logger.error(f"Error computing correlation heatmap for temperature {temperature}: {e}")
            plt.close()

    def analyze_cross_temperature_ranking_changes(self) -> pd.DataFrame:
        """
        Track how feature importance rankings change across temperatures.

        Returns:
            DataFrame with: feature, rank_t0.4, rank_t0.6, rank_t0.8, max_rank_change
        """
        logger.info("Analyzing cross-temperature ranking changes...")

        importance_df = self.extract_feature_importance()

        if importance_df.empty:
            return pd.DataFrame()

        # Pivot to get rank per temperature
        pivot_df = importance_df.pivot_table(
            index="feature",
            columns="temperature",
            values="rank",
            aggfunc="first",
        ).reset_index()

        # Rename columns
        pivot_df.columns = ["feature"] + [f"rank_t{t}" for t in sorted(self.temperatures)]

        # Compute max rank change
        rank_cols = [col for col in pivot_df.columns if col.startswith("rank_t")]
        pivot_df["max_rank_change"] = pivot_df[rank_cols].max(axis=1) - pivot_df[rank_cols].min(axis=1)

        pivot_df = pivot_df.sort_values("max_rank_change", ascending=False)

        # Save to CSV
        csv_path = self.output_dir / "cross_temperature_ranking_changes.csv"
        pivot_df.to_csv(csv_path, index=False)
        logger.info(f"Cross-temperature ranking changes saved to: {csv_path}")

        return pivot_df

    def plot_cross_temperature_comparison(self):
        """Plot grouped bar chart comparing top feature importance across temperatures."""
        logger.info("Plotting cross-temperature feature importance comparison...")

        importance_df = self.extract_feature_importance()

        if importance_df.empty:
            logger.warning("No importance data to plot")
            return

        # Get top 15 features by average importance across all temperatures
        avg_importance = importance_df.groupby("feature")["avg_importance"].mean().nlargest(15)
        top_features = avg_importance.index.tolist()

        # Filter data for top features
        plot_data = importance_df[importance_df["feature"].isin(top_features)]

        # Create grouped bar chart
        plt.figure(figsize=(14, 8))
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        ax = sns.barplot(
            data=plot_data,
            x="feature",
            y="avg_importance",
            hue="temperature",
            palette="husl",
        )

        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Average Importance", fontsize=12)
        plt.title("Top Feature Importance Comparison Across Temperatures", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.legend(title="Temperature", loc="upper right")
        plt.tight_layout()

        plot_path = self.output_dir / "cross_temperature_importance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Cross-temperature comparison plot saved to: {plot_path}")

    def run_analysis(self, dataset: str = "math500", model: str = "qwen3_4b") -> Dict:
        """
        Run all factor identification analyses.

        Args:
            dataset: Dataset name to analyze
            model: Model name to analyze (unused, for API compatibility)

        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Running factor identification analysis (exclude_features='{self.exclude_features}')...")

        results = {}

        try:
            self.load_temperature_data(dataset)
            results["importance"] = self.extract_feature_importance()
            results["ranking_changes"] = self.analyze_cross_temperature_ranking_changes()

            # Run SHAP analysis and correlation heatmaps for each temperature
            for temp, model_results in self.importance_results.items():
                X_train = model_results.get("X_train")
                X_test = model_results.get("X_test")
                models = model_results.get("models", {})

                for model_name, model in models.items():
                    if X_train is not None and X_test is not None:
                        self.run_shap_analysis(model, X_train, X_test, temp, model_name)

                if temp in self.temperature_data:
                    self.compute_correlation_heatmap(self.temperature_data[temp], temp)

            self.plot_cross_temperature_comparison()

        except Exception as e:
            logger.error(f"Error in factor identification analysis: {e}")

        return results


class EntropyDistributionAnalyzer:
    """Analyze entropy distribution patterns across temperatures."""

    def __init__(self, data_dir: str, output_dir: str, temperatures: List[float]):
        """
        Initialize the EntropyDistributionAnalyzer.

        Args:
            data_dir: Path to evaluation/results_temp directory
            output_dir: Directory to save analysis results
            temperatures: List of temperature values to analyze
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.temperatures = sorted(temperatures)
        self.temperature_data: Dict[float, pd.DataFrame] = {}

        create_output_directory(self.output_dir)
        setup_visualization_style()

    def load_temperature_data(self, dataset: str = "math500") -> Dict[float, pd.DataFrame]:
        """Load CSV data for each temperature."""
        logger.info("Loading temperature data for entropy distribution analysis...")

        for temp in self.temperatures:
            csv_path = self.data_dir / dataset / f"t_{temp}" / "all_aggregated_data_exclude_agent.csv"

            if not csv_path.exists():
                logger.warning(f"Data file not found for temperature {temp}: {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
                df["temperature"] = temp
                self.temperature_data[temp] = df
                logger.info(f"Loaded {len(df)} records for temperature {temp}")
            except Exception as e:
                logger.error(f"Error loading data for temperature {temp}: {e}")

        return self.temperature_data

    def compute_distribution_statistics(self) -> pd.DataFrame:
        """
        Compute entropy distribution statistics per temperature x architecture.

        Returns:
            DataFrame with: temperature, architecture, stat_name, stat_value
        """
        logger.info("Computing entropy distribution statistics...")

        if not self.temperature_data:
            logger.warning("No temperature data loaded.")
            return pd.DataFrame()

        results = []
        entropy_cols = ["sample_mean_entropy", "sample_total_entropy", "sample_avg_entropy_per_token"]
        architectures = ["single", "sequential", "centralized", "debate", "hybrid"]

        for temp, df in self.temperature_data.items():
            for arch in architectures:
                arch_df = df[df["architecture"] == arch] if "architecture" in df.columns else df

                if len(arch_df) == 0:
                    continue

                for entropy_col in entropy_cols:
                    if entropy_col not in arch_df.columns:
                        continue

                    entropy_values = arch_df[entropy_col].dropna()

                    if len(entropy_values) < 2:
                        continue

                    # Compute statistics
                    statistics = {
                        "mean": entropy_values.mean(),
                        "std": entropy_values.std(),
                        "median": entropy_values.median(),
                        "min": entropy_values.min(),
                        "max": entropy_values.max(),
                        "q25": entropy_values.quantile(0.25),
                        "q75": entropy_values.quantile(0.75),
                        "skewness": stats.skew(entropy_values),
                        "kurtosis": stats.kurtosis(entropy_values),
                    }

                    for stat_name, stat_value in statistics.items():
                        results.append({
                            "temperature": temp,
                            "architecture": arch,
                            "entropy_metric": entropy_col,
                            "stat_name": stat_name,
                            "stat_value": stat_value,
                        })

        stats_df = pd.DataFrame(results)

        # Save to CSV
        csv_path = self.output_dir / "entropy_distribution_statistics.csv"
        stats_df.to_csv(csv_path, index=False)
        logger.info(f"Entropy distribution statistics saved to: {csv_path}")

        return stats_df

    def plot_entropy_distributions(self):
        """Plot entropy distribution histograms/density for each temperature."""
        logger.info("Plotting entropy distributions...")

        if not self.temperature_data:
            logger.warning("No temperature data loaded.")
            return

        entropy_col = "sample_mean_entropy"
        n_temps = len(self.temperature_data)

        if n_temps == 0:
            return

        fig, axes = plt.subplots(1, n_temps, figsize=(6 * n_temps, 6), sharey=True)
        plt.style.use("seaborn-v0_8")

        if n_temps == 1:
            axes = [axes]

        colors = sns.color_palette("husl", n_colors=5)
        architectures = ["single", "sequential", "centralized", "debate", "hybrid"]
        arch_colors = {arch: colors[i] for i, arch in enumerate(architectures)}

        for idx, (temp, df) in enumerate(sorted(self.temperature_data.items())):
            ax = axes[idx]

            if entropy_col not in df.columns:
                ax.set_title(f"Temperature {temp}\n(No entropy data)")
                continue

            for arch in architectures:
                if "architecture" in df.columns:
                    arch_df = df[df["architecture"] == arch]
                else:
                    arch_df = df

                if len(arch_df) == 0 or entropy_col not in arch_df.columns:
                    continue

                entropy_values = arch_df[entropy_col].dropna()

                if len(entropy_values) > 1:
                    # Plot histogram
                    ax.hist(
                        entropy_values,
                        bins=30,
                        density=True,
                        alpha=0.5,
                        label=arch.capitalize(),
                        color=arch_colors[arch],
                    )
                    # Add KDE overlay
                    try:
                        sns.kdeplot(entropy_values, ax=ax, color=arch_colors[arch], linewidth=2)
                    except Exception:
                        pass

            ax.set_title(f"Temperature {temp}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Sample Mean Entropy", fontsize=12)
            if idx == 0:
                ax.set_ylabel("Density", fontsize=12)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Entropy Distribution Across Temperatures", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        plot_path = self.output_dir / "entropy_distribution_histogram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Entropy distribution plot saved to: {plot_path}")

    def generate_stats_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table of key entropy statistics across temperatures.

        Returns:
            Pivot table with temperature as rows, statistics as columns
        """
        logger.info("Generating entropy statistics comparison table...")

        stats_df = self.compute_distribution_statistics()

        if stats_df.empty:
            return pd.DataFrame()

        # Filter to sample_mean_entropy and create pivot
        filtered = stats_df[stats_df["entropy_metric"] == "sample_mean_entropy"]

        if filtered.empty:
            return pd.DataFrame()

        pivot_df = filtered.pivot_table(
            index=["temperature", "architecture"],
            columns="stat_name",
            values="stat_value",
            aggfunc="first",
        ).reset_index()

        # Save to CSV
        csv_path = self.output_dir / "entropy_stats_comparison_table.csv"
        pivot_df.to_csv(csv_path, index=False)
        logger.info(f"Entropy stats comparison table saved to: {csv_path}")

        return pivot_df

    def run_full_analysis(self, dataset: str = "math500") -> Dict:
        """
        Run all entropy distribution analyses.

        Args:
            dataset: Dataset name to analyze

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Running full entropy distribution analysis...")

        results = {}

        try:
            self.load_temperature_data(dataset)
            results["statistics"] = self.compute_distribution_statistics()
            results["comparison_table"] = self.generate_stats_comparison_table()
            self.plot_entropy_distributions()
        except Exception as e:
            logger.error(f"Error in entropy distribution analysis: {e}")

        return results


class TemperatureAblationAnalyzer:
    """Unified entry point for temperature ablation analysis."""

    def __init__(
        self,
        data_dir: str = "evaluation/results_temp",
        output_dir: str = "data_mining/results_temp_ablation",
        temperatures: List[float] = None,
        dataset: str = "math500",
        model: str = "qwen3_4b",
        exclude_features: str = "default,base_model_wo_entropy,base_model_all_metrics",
    ):
        """
        Initialize the TemperatureAblationAnalyzer.

        Args:
            data_dir: Path to evaluation/results_temp directory
            output_dir: Directory to save analysis results
            temperatures: List of temperature values to analyze
            dataset: Dataset name to analyze
            model: Model name to analyze
            exclude_features: Comma-separated list of feature exclusion groups for factor identification
        """
        if temperatures is None:
            temperatures = [0.4, 0.6, 0.8]

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.temperatures = sorted(temperatures)
        self.dataset = dataset
        self.model = model
        self.exclude_features = exclude_features
        self.results: Dict[str, Any] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-analyzers (factor analyzer created dynamically in run_full_analysis)
        self.accuracy_analyzer = AccuracyComparisonAnalyzer(
            str(self.data_dir), str(self.output_dir / "accuracy"), self.temperatures
        )
        self.entropy_analyzer = EntropyDistributionAnalyzer(
            str(self.data_dir), str(self.output_dir / "entropy"), self.temperatures
        )

        logger.info(f"TemperatureAblationAnalyzer initialized")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Temperatures: {self.temperatures}")
        logger.info(f"  Dataset: {self.dataset}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Exclude features: {self.exclude_features}")

    def run_accuracy_analysis(self) -> Dict:
        """
        Run accuracy comparison analysis.

        Returns:
            Dictionary containing accuracy analysis results
        """
        logger.info("=" * 80)
        logger.info("Running Accuracy Comparison Analysis")
        logger.info("=" * 80)

        try:
            results = self.accuracy_analyzer.run_full_analysis(self.dataset)
            self.results["accuracy"] = results
            return results
        except Exception as e:
            logger.error(f"Error in accuracy analysis: {e}")
            return {}

    def run_factor_analysis(self) -> Dict:
        """
        Run factor identification analysis for all exclude feature groups.

        Returns:
            Dictionary containing factor analysis results for each exclude group
        """
        logger.info("=" * 80)
        logger.info("Running Factor Identification Analysis")
        logger.info("=" * 80)

        results = {}
        exclude_groups = [g.strip() for g in self.exclude_features.split(",")]

        for exclude_group in exclude_groups:
            try:
                logger.info(f"Running factor identification with exclude_features='{exclude_group}'...")

                # Create output subdirectory for this exclusion config
                factor_output = str(self.output_dir / "factors" / f"exclude_{exclude_group}")

                factor_analyzer = FactorIdentificationAnalyzer(
                    str(self.data_dir),
                    factor_output,
                    self.temperatures,
                    exclude_features=exclude_group,
                )
                factor_results = factor_analyzer.run_analysis(
                    dataset=self.dataset, model=self.model
                )
                results[f"factor_identification_exclude_{exclude_group}"] = factor_results

            except Exception as e:
                logger.error(f"Error in factor analysis for exclude_features='{exclude_group}': {e}")
                results[f"factor_identification_exclude_{exclude_group}"] = {}

        self.results["factors"] = results
        return results

    def run_entropy_analysis(self) -> Dict:
        """
        Run entropy distribution analysis.

        Returns:
            Dictionary containing entropy analysis results
        """
        logger.info("=" * 80)
        logger.info("Running Entropy Distribution Analysis")
        logger.info("=" * 80)

        try:
            results = self.entropy_analyzer.run_full_analysis(self.dataset)
            self.results["entropy"] = results
            return results
        except Exception as e:
            logger.error(f"Error in entropy analysis: {e}")
            return {}

    def generate_unified_report(self, results: Dict):
        """
        Generate temperature_ablation_report.txt with all results.

        Args:
            results: Dictionary containing all analysis results
        """
        logger.info("Generating unified temperature ablation report...")

        report_path = self.output_dir / "temperature_ablation_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("TEMPERATURE ABLATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Temperatures analyzed: {self.temperatures}\n")
            f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n\n")

            # Accuracy Analysis Section
            if "accuracy" in results:
                f.write("-" * 80 + "\n")
                f.write("ACCURACY COMPARISON ANALYSIS\n")
                f.write("-" * 80 + "\n\n")

                accuracy_table = results["accuracy"].get("accuracy_table")
                if accuracy_table is not None and not accuracy_table.empty:
                    f.write("Accuracy by Temperature and Architecture:\n\n")
                    for temp in self.temperatures:
                        temp_data = accuracy_table[accuracy_table["temperature"] == temp]
                        if not temp_data.empty:
                            f.write(f"  Temperature {temp}:\n")
                            for _, row in temp_data.iterrows():
                                f.write(f"    {row['architecture']}: {row['accuracy']:.4f} "
                                       f"({row['num_correct']}/{row['num_samples']})\n")
                            f.write("\n")

                significance_df = results["accuracy"].get("significance_tests")
                if significance_df is not None and not significance_df.empty:
                    f.write("Significance Tests (McNemar):\n\n")
                    for _, row in significance_df.iterrows():
                        sig_marker = "*" if row["significant"] else ""
                        f.write(f"  {row['temp_pair']} | {row['architecture']}: "
                               f"p={row['p_value']:.4f} {sig_marker}\n")
                    f.write("\n")

            # Factor Analysis Section
            if "factors" in results:
                f.write("-" * 80 + "\n")
                f.write("FACTOR IDENTIFICATION ANALYSIS\n")
                f.write("-" * 80 + "\n\n")

                importance_df = results["factors"].get("importance")
                if importance_df is not None and not importance_df.empty:
                    f.write("Top 10 Features by Average Importance:\n\n")
                    top_features = importance_df.groupby("feature")["avg_importance"].mean().nlargest(10)
                    for i, (feature, importance) in enumerate(top_features.items(), 1):
                        f.write(f"  {i}. {feature}: {importance:.6f}\n")
                    f.write("\n")

                ranking_df = results["factors"].get("ranking_changes")
                if ranking_df is not None and not ranking_df.empty:
                    f.write("Features with Largest Ranking Changes Across Temperatures:\n\n")
                    for _, row in ranking_df.head(10).iterrows():
                        f.write(f"  {row['feature']}: max_rank_change={row['max_rank_change']:.0f}\n")
                    f.write("\n")

            # Entropy Analysis Section
            if "entropy" in results:
                f.write("-" * 80 + "\n")
                f.write("ENTROPY DISTRIBUTION ANALYSIS\n")
                f.write("-" * 80 + "\n\n")

                stats_df = results["entropy"].get("statistics")
                if stats_df is not None and not stats_df.empty:
                    f.write("Mean Entropy Statistics by Temperature:\n\n")
                    mean_entropy = stats_df[
                        (stats_df["entropy_metric"] == "sample_mean_entropy") &
                        (stats_df["stat_name"] == "mean")
                    ]
                    for temp in self.temperatures:
                        temp_data = mean_entropy[mean_entropy["temperature"] == temp]
                        if not temp_data.empty:
                            f.write(f"  Temperature {temp}:\n")
                            for _, row in temp_data.iterrows():
                                f.write(f"    {row['architecture']}: {row['stat_value']:.4f}\n")
                            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Unified report saved to: {report_path}")

    def run_full_analysis(self, analysis_type: str = "all") -> Dict:
        """
        Run specified analysis type(s).

        Args:
            analysis_type: 'all', 'accuracy', 'factor', or 'entropy'

        Returns:
            Dictionary containing all requested analysis results
        """
        logger.info("=" * 80)
        logger.info(f"Starting Temperature Ablation Analysis (type: {analysis_type})")
        logger.info("=" * 80)

        results = {}

        if analysis_type in ["all", "accuracy"]:
            results["accuracy"] = self.run_accuracy_analysis()

        if analysis_type in ["all", "factor"]:
            # Parse exclude_features into a list and run factor analysis for each
            exclude_groups = [g.strip() for g in self.exclude_features.split(",")]

            for exclude_group in exclude_groups:
                try:
                    logger.info(f"Running factor identification with exclude_features='{exclude_group}'...")

                    # Create output subdirectory for this exclusion config
                    factor_output = str(self.output_dir / "factors" / f"exclude_{exclude_group}")

                    factor_analyzer = FactorIdentificationAnalyzer(
                        str(self.data_dir),
                        factor_output,
                        self.temperatures,
                        exclude_features=exclude_group,
                    )
                    factor_results = factor_analyzer.run_analysis(
                        dataset=self.dataset, model=self.model
                    )
                    results[f"factor_identification_exclude_{exclude_group}"] = factor_results

                except Exception as e:
                    logger.error(f"Error in factor analysis for exclude_features='{exclude_group}': {e}")
                    results[f"factor_identification_exclude_{exclude_group}"] = {}

        if analysis_type in ["all", "entropy"]:
            results["entropy"] = self.run_entropy_analysis()

        # Generate unified report   
        self.generate_unified_report(results)

        logger.info("=" * 80)
        logger.info("Temperature Ablation Analysis Complete")
        logger.info("=" * 80)

        return results


def main():
    """Main function to execute the temperature ablation analysis."""
    parser = argparse.ArgumentParser(
        description="Temperature Ablation Data Mining Analysis"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["all", "accuracy", "factor", "entropy"],
        default="all",
        help="Type of analysis to run (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="evaluation/results_temp",
        help="Path to evaluation results directory (default: evaluation/results_temp)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_mining/results_temp_ablation",
        help="Output directory for results (default: data_mining/results_temp_ablation)",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=[0.4, 0.6, 0.8],
        help="Temperature values to analyze (default: 0.4 0.6 0.8)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        help="Dataset name to analyze (default: math500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3_4b",
        help="Model name to analyze (default: qwen3_4b)",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default="default,base_model_wo_entropy,base_model_all_metrics",
        help="""Feature exclusion groups for factor identification (comma-separated).
        Default runs three analyses excluding: default, base_model_wo_entropy, base_model_all_metrics.
        Use 'all' for no exclusions. See features.py FEATURE_GROUPS for available groups.""",
    )

    args = parser.parse_args()

    logger.info("Initializing Temperature Ablation Analyzer...")

    analyzer = TemperatureAblationAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        temperatures=args.temperatures,
        dataset=args.dataset,
        model=args.model,
        exclude_features=args.exclude_features,
    )

    results = analyzer.run_full_analysis(analysis_type=args.analysis_type)

    logger.info("Temperature ablation analysis completed successfully!")

    return results


if __name__ == "__main__":
    results = main()
