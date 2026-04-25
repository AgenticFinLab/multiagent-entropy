"""Comprehensive visualization for base-model analysis.

Refactored from results_plot/base_model/analyze_base_model.py into the
`visualization` package layout.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from visualization.base import ARCH_COLORS, BaseVisualizer
from visualization.base.data_loaders import load_csv, load_shap

warnings.filterwarnings("ignore")


class BaseModelPlot(BaseVisualizer):
    """Comprehensive visualizer for base model analysis."""

    def __init__(
        self,
        feature_importance_csv: Path | str,
        shap_results_dir: Path | str,
        merged_data_path: Path | str,
        output_dir: Path | str,
        shap_exp_key: str = "exclude_base_model_wo_entropy",
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=14)
        self.feature_importance_csv = Path(feature_importance_csv)
        self.shap_results_dir = Path(shap_results_dir)
        self.merged_data_path = Path(merged_data_path)
        self.shap_exp_key = shap_exp_key

    # ---------- data loading helpers ----------

    def _load_feature_importance(self) -> pd.DataFrame | None:
        if not self.feature_importance_csv.exists():
            print(f"Feature importance file not found: {self.feature_importance_csv}")
            return None
        return load_csv(self.feature_importance_csv)

    def _load_shap_data(self):
        shap_df, x_df, _ = load_shap(self.shap_results_dir, self.shap_exp_key)
        return shap_df, x_df

    def _load_merged_data(self) -> pd.DataFrame | None:
        if not self.merged_data_path.exists():
            print(f"Merged data file not found: {self.merged_data_path}")
            return None
        return load_csv(self.merged_data_path)

    # ---------- subplot: standalone feature importance bar ----------

    def plot_feature_importance(self, ax, top_n: int = 10) -> None:
        """Plot feature importance bar chart (standalone variant)."""
        fi_df = self._load_feature_importance()
        if fi_df is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return

        for col in ("lightgbm_importance", "xgboost_importance"):
            lo, hi = fi_df[col].min(), fi_df[col].max()
            fi_df[f"{col}_normalized"] = (
                (fi_df[col] - lo) / (hi - lo) if hi > lo else fi_df[col]
            )

        fi_df = fi_df.sort_values("mean_importance_normalized", ascending=False).head(top_n)

        y_pos = np.arange(len(fi_df))
        bar_height = 0.25
        colors = {"lightgbm": "#91BFD8", "xgboost": "#4575B4", "mean": "#D73027"}

        ax.barh(y_pos - bar_height, fi_df["lightgbm_importance_normalized"].values,
                height=bar_height, color=colors["lightgbm"], edgecolor="white",
                linewidth=0.8, label="LightGBM")
        ax.barh(y_pos, fi_df["xgboost_importance_normalized"].values,
                height=bar_height, color=colors["xgboost"], edgecolor="white",
                linewidth=0.8, label="XGBoost")
        ax.barh(y_pos + bar_height, fi_df["mean_importance_normalized"].values,
                height=bar_height, color=colors["mean"], edgecolor="white",
                linewidth=0.8, label="Mean")

        ax.set_yticks(y_pos)

        feature_mapping = {
            "base_model_answer_token_count": "base model answer token count",
            "base_sample_total_entropy": "base model entropy",
            "base_sample_token_count": "base model total token count",
        }
        labels = []
        for text in fi_df["feature"].values:
            mapped = feature_mapping.get(text, text)
            labels.append(textwrap.fill(mapped.replace("_", " "), width=24))
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()

        ax.set_xlabel("Feature Importance", fontsize=14)
        ax.text(0.5, -0.15, "(c)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False, fontsize=12)
        sns.despine(ax=ax, top=True, right=True)

    # ---------- subplot: SHAP scatter with feature importance inset ----------

    def plot_shap_with_importance_inset(self, ax) -> None:
        """SHAP scatter for base-model features with feature-importance inset."""
        shap_df, x_test_df = self._load_shap_data()
        if shap_df is None or x_test_df is None:
            ax.text(0.5, 0.5, "No SHAP data available",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.15, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        focus_features = [
            "base_model_answer_token_count",
            "base_sample_total_entropy",
        ]
        available_features = [
            f for f in focus_features if f in shap_df.columns and f in x_test_df.columns
        ]
        if not available_features:
            ax.text(0.5, 0.5, "No base model features available in SHAP data",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.15, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        colors = ["#4575B4", "#D73027"]
        markers = ["o", "s", "^"]
        feature_map = {
            "base_model_answer_token_count": "Base Model Answer Token",
            "base_sample_total_entropy": "Base Model Entropy",
            "base_sample_token_count": "Base Model Total Token",
        }

        for idx, feature in enumerate(available_features):
            feature_values = x_test_df[feature].values.copy()
            shap_values = shap_df[feature].values

            fv_min, fv_max = np.nanmin(feature_values), np.nanmax(feature_values)
            if fv_max > fv_min:
                feature_values_norm = (feature_values - fv_min) / (fv_max - fv_min)
            else:
                feature_values_norm = np.full_like(feature_values, 0.5)

            corr, _ = pearsonr(feature_values, shap_values)

            ax.scatter(
                feature_values_norm,
                shap_values,
                alpha=0.6,
                s=30,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                label=f"{feature_map.get(feature, feature)}\n(Pearson Correlation={corr:.3f})",
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Normalized Feature Value", fontsize=14)
        ax.set_ylabel("SHAP Value", fontsize=14)
        ax.text(0.5, -0.15, "(c)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, fontweight="bold")
        ax.legend(loc="lower right", frameon=False, fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

        inset_ax = ax.inset_axes([0.5, 0.72, 0.40, 0.25])
        self._plot_importance_inset(inset_ax)

    def _plot_importance_inset(self, ax, top_n: int = 5) -> None:
        """Simplified feature importance bar chart for use as inset."""
        fi_df = self._load_feature_importance()
        if fi_df is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            return

        for col in ("lightgbm_importance", "xgboost_importance"):
            lo, hi = fi_df[col].min(), fi_df[col].max()
            fi_df[f"{col}_normalized"] = (
                (fi_df[col] - lo) / (hi - lo) if hi > lo else fi_df[col]
            )

        fi_df = fi_df.sort_values("mean_importance_normalized", ascending=False).head(top_n)

        y_pos = np.arange(len(fi_df))
        bar_height = 0.25
        colors = {"lightgbm": "#91BFD8", "xgboost": "#4575B4", "mean": "#D73027"}

        ax.barh(y_pos - bar_height, fi_df["lightgbm_importance_normalized"].values,
                height=bar_height, color=colors["lightgbm"], edgecolor="white",
                linewidth=0.5, label="LightGBM")
        ax.barh(y_pos, fi_df["xgboost_importance_normalized"].values,
                height=bar_height, color=colors["xgboost"], edgecolor="white",
                linewidth=0.5, label="XGBoost")
        ax.barh(y_pos + bar_height, fi_df["mean_importance_normalized"].values,
                height=bar_height, color=colors["mean"], edgecolor="white",
                linewidth=0.5, label="Mean")

        feature_mapping = {
            "base_model_answer_token_count": "base model answer token",
            "base_sample_total_entropy": "base model entropy",
            "base_sample_token_count": "base model total token",
            "base_sample_avg_entropy_per_token": "base model avg entropy/token",
        }
        ax.set_yticks(y_pos)
        labels = []
        for f in fi_df["feature"].values:
            label = feature_mapping.get(f, f.replace("_", " "))
            if len(label) > 24:
                label = label[:24] + "..."
            labels.append(label)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()

        ax.set_xlabel("Importance", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(loc="lower right", frameon=False, fontsize=7, ncol=1)
        ax.set_facecolor("white")
        ax.patch.set_alpha(0.9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(0.8)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    # ---------- subplot: accuracy trends ----------

    def plot_accuracy_trends(self, ax) -> None:
        """Accuracy trends across architectures binned by base-model entropy."""
        df = self._load_merged_data()
        if df is None:
            ax.text(0.5, 0.5, "No merged data available",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.15, "(d)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        qwen_models = ["qwen3_0_6b", "qwen3_4b", "qwen3_8b"]
        df_qwen = df[df["model_name"].isin(qwen_models)]

        if len(df_qwen) == 0:
            ax.text(0.5, 0.5, "No qwen model data available",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.15, "(d)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        focus_feature = "base_sample_total_entropy"
        if focus_feature not in df_qwen.columns or "is_finally_correct" not in df_qwen.columns:
            ax.text(0.5, 0.5, "Required columns not found in data",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.15, "(d)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        architectures = ["centralized", "debate", "hybrid", "sequential", "single"]
        available_archs = [a for a in architectures if a in df_qwen["architecture"].unique()]

        base_stats = []
        for model in qwen_models:
            model_df = df[df["model_name"] == model]
            if not model_df.empty:
                base_stats.append({
                    "model": model,
                    "avg_entropy": model_df["base_sample_total_entropy"].mean(),
                    "avg_acc": model_df["base_model_accuracy"].mean() * 100,
                })

        df_qwen["feature_bin"] = pd.qcut(df_qwen[focus_feature], q=10, duplicates="drop")

        for arch in available_archs:
            arch_df = df_qwen[df_qwen["architecture"] == arch]
            grouped = arch_df.groupby("feature_bin")["is_finally_correct"].mean()
            bin_centers = [(iv.left + iv.right) / 2 for iv in grouped.index]
            ax.plot(
                bin_centers,
                grouped.values * 100,
                marker="o",
                linewidth=2,
                markersize=6,
                color=ARCH_COLORS.get(arch, "#999999"),
                label=arch,
                alpha=0.8,
            )

        models_map = {"qwen3_0_6b": "Qwen3-0.6B", "qwen3_4b": "Qwen3-4B", "qwen3_8b": "Qwen3-8B"}
        markers = ["X", "^", "s"]
        scatter_colors = ["#D73027", "#56B4E9", "#FEE090"]
        for i, stat in enumerate(base_stats):
            ax.scatter(
                stat["avg_entropy"],
                stat["avg_acc"],
                color=scatter_colors[i % len(scatter_colors)],
                marker=markers[i % len(markers)],
                s=120,
                label=f'{models_map[stat["model"]]} (Base)',
                edgecolors="white",
                linewidths=1.5,
                zorder=5,
            )

        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("Base Model Entropy", fontsize=14)
        ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.text(0.5, -0.15, "(d)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, fontweight="bold")
        ax.legend(loc="best", frameon=False, fontsize=11, ncol=2)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    # ---------- main entry ----------

    def compose(self, filename: str = "base_model_analysis.pdf", save_individual: bool = True) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        print("Generating subplot 1: SHAP Scatter with Feature Importance Inset...")
        self.plot_shap_with_importance_inset(axes[0])

        print("Generating subplot 2: Accuracy Trends...")
        self.plot_accuracy_trends(axes[1])

        plt.tight_layout()
        self.save_figure(fig, filename)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots()

    def _save_individual_subplots(self) -> None:
        configs = [
            ("shap_scatter_with_inset", self.plot_shap_with_importance_inset, (7, 6)),
            ("accuracy_trends", self.plot_accuracy_trends, (7, 6)),
        ]
        print("\nSaving individual subplots...")
        for name, func, figsize in configs:
            fig, ax = plt.subplots(figsize=figsize)
            func(ax)
            plt.tight_layout()
            self.save_subplot(fig, name)
            plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    feature_importance_csv = root / "data_mining" / "results_qwen" / "results_aggregated" / "exclude_base_model_wo_entropy.csv"
    shap_results_dir = root / "data_mining" / "results_qwen" / "results"
    merged_data_path = root / "data_mining" / "data" / "merged_datasets.csv"
    output_dir = root / "visualization" / "outputs" / "base_model"

    plotter = BaseModelPlot(
        feature_importance_csv=feature_importance_csv,
        shap_results_dir=shap_results_dir,
        merged_data_path=merged_data_path,
        output_dir=output_dir,
    )
    plotter.compose()


if __name__ == "__main__":
    main()
