"""MAS comprehensive analysis figure (SHAP scatter + entropy scatter).

Refactored from results_plot/mas/analyze_mas.py into the
`visualization` package layout.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from visualization.base import BaseVisualizer
from visualization.base.data_loaders import load_csv, load_shap

warnings.filterwarnings("ignore")


class MASPlot(BaseVisualizer):
    """Two-subplot MAS visualization: SHAP scatter + top-2 entropy scatter."""

    def __init__(
        self,
        results_dir: Path | str,
        exp_key: str,
        feature_importance_path: Path | str,
        output_dir: Path | str,
        top_features: Optional[List[str]] = None,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=14)

        self.feature_importance_path = Path(feature_importance_path)

        # SHAP trio for LightGBM (shap_values, X_test, prediction_probabilities)
        self.shap_df, self.x_test_df, self.lightgbm_pred_df = load_shap(
            results_dir, exp_key, model="LightGBM", task="classification"
        )
        # XGBoost prediction probabilities (loaded separately; not part of trio)
        self.xgboost_pred_df, _ = self._load_xgb_preds(results_dir, exp_key)

        if top_features is None:
            self.feature1 = "sample_variance_entropy"
            self.feature2 = "sample_round_1_q3_agent_variance_entropy"
            self.focus_features = [self.feature1, self.feature2]
        else:
            self.focus_features = top_features
            self.feature1 = top_features[0]
            self.feature2 = top_features[1] if len(top_features) > 1 else top_features[0]

    @staticmethod
    def _load_xgb_preds(results_dir: Path | str, exp_key: str):
        path = Path(results_dir) / exp_key / "shap" / "shap_prediction_probabilities_XGBoost_classification.csv"
        return (pd.read_csv(path) if path.exists() else None), path

    def plot_shap_with_importance_inset(self, ax: plt.Axes) -> None:
        """SHAP value scatter for the focus features (combines original subplots 1+2)."""
        shap_df, x_test_df = self.shap_df, self.x_test_df
        if shap_df is None or x_test_df is None:
            ax.text(0.5, 0.5, "SHAP data not available",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.15, "(a)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        available_features = [
            f for f in self.focus_features
            if f in shap_df.columns and f in x_test_df.columns
        ]

        if not available_features:
            ax.text(0.5, 0.5, "No specified features available in SHAP data",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.15, "(a)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        colors = ["#4575B4", "#D73027", "#91BFD8"]
        markers = ["o", "s", "^"]

        feature_map = {
            "sample_variance_entropy": "Variance Entropy",
            "sample_round_1_q3_agent_variance_entropy": "Round 1 Q3 Var Entropy",
            "sample_answer_token_count": "Answer Token Count",
        }

        for idx, feature in enumerate(available_features):
            feature_values = x_test_df[feature].values.copy()
            shap_values = shap_df[feature].values

            # Normalize feature values to [0, 1] for consistent scale
            fv_min, fv_max = np.nanmin(feature_values), np.nanmax(feature_values)
            if fv_max > fv_min:
                feature_values_norm = (feature_values - fv_min) / (fv_max - fv_min)
            else:
                feature_values_norm = np.full_like(feature_values, 0.5)

            corr, _ = pearsonr(feature_values, shap_values)
            display_name = feature_map.get(feature, feature.replace("_", " ").title())

            ax.scatter(
                feature_values_norm,
                shap_values,
                alpha=0.6,
                s=30,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                label=f"{display_name}\n(Pearson Correlation {corr:.3f})",
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Normalized Feature Value", fontsize=14)
        ax.set_ylabel("SHAP Value", fontsize=14)
        ax.text(0.5, -0.15, "(a)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, fontweight="bold")
        ax.legend(loc="best", frameon=True, fancybox=False,
                  edgecolor="black", framealpha=0.95, fontsize=12.5)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    def _plot_importance_inset(self, ax: plt.Axes, top_n: int = 5) -> None:
        """Simplified feature importance bar chart, intended for use as an inset."""
        if not self.feature_importance_path.exists():
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
            return

        fi_df = pd.read_csv(self.feature_importance_path)

        lightgbm_min = fi_df["lightgbm_importance"].min()
        lightgbm_max = fi_df["lightgbm_importance"].max()
        if lightgbm_max > lightgbm_min:
            fi_df["lightgbm_importance_normalized"] = (fi_df["lightgbm_importance"] - lightgbm_min) / (lightgbm_max - lightgbm_min)
        else:
            fi_df["lightgbm_importance_normalized"] = fi_df["lightgbm_importance"]

        xgb_min = fi_df["xgboost_importance"].min()
        xgb_max = fi_df["xgboost_importance"].max()
        if xgb_max > xgb_min:
            fi_df["xgboost_importance_normalized"] = (fi_df["xgboost_importance"] - xgb_min) / (xgb_max - xgb_min)
        else:
            fi_df["xgboost_importance_normalized"] = fi_df["xgboost_importance"]

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
            "sample_variance_entropy": "var entropy",
            "sample_round_1_q3_agent_variance_entropy": "r1 q3 var entropy",
            "sample_answer_token_count": "answer token",
            "sample_total_entropy": "total entropy",
            "sample_mean_entropy": "mean entropy",
        }

        ax.set_yticks(y_pos)
        labels = []
        for f in fi_df["feature"].values:
            label = feature_mapping.get(f, f.replace("sample_", "").replace("_", " "))
            if len(label) > 18:
                label = label[:18] + "..."
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

    def plot_top2_entropy_scatter(self, ax: plt.Axes) -> None:
        """Two-feature entropy scatter; markers separate SAS vs MAS, color by predicted class."""
        x_test = self.x_test_df
        if x_test is None:
            print("Warning: X_test data unavailable")
            return

        f1, f2 = self.feature1, self.feature2

        if f1 not in x_test.columns or f2 not in x_test.columns:
            print(f"Warning: Required features not found ({f1}, {f2}). Available features: {list(x_test.columns[:10])}...")
            return

        if "architecture" not in x_test.columns:
            print("Warning: 'architecture' column not found in X_test data")
            return

        if self.lightgbm_pred_df is None or self.xgboost_pred_df is None:
            print("Warning: Prediction probability files not available")
            return

        lgbm_prob0 = pd.to_numeric(self.lightgbm_pred_df["prob_class_0"].values, errors="coerce")
        lgbm_prob1 = pd.to_numeric(self.lightgbm_pred_df["prob_class_1"].values, errors="coerce")
        xgb_prob0 = pd.to_numeric(self.xgboost_pred_df["prob_class_0"].values, errors="coerce")
        xgb_prob1 = pd.to_numeric(self.xgboost_pred_df["prob_class_1"].values, errors="coerce")

        x1 = x_test[f1].values
        x2 = x_test[f2].values
        architecture = x_test["architecture"].values

        min_len = min(len(lgbm_prob0), len(xgb_prob0), len(x1), len(x2), len(architecture))
        lgbm_prob0, lgbm_prob1 = lgbm_prob0[:min_len], lgbm_prob1[:min_len]
        xgb_prob0, xgb_prob1 = xgb_prob0[:min_len], xgb_prob1[:min_len]
        x1, x2, architecture = x1[:min_len], x2[:min_len], architecture[:min_len]

        mean_prob0 = (lgbm_prob0 + xgb_prob0) / 2.0
        mean_prob1 = (lgbm_prob1 + xgb_prob1) / 2.0

        mask_valid = ~(np.isnan(mean_prob0) | np.isnan(mean_prob1) | np.isnan(x1) | np.isnan(x2))
        x1, x2 = x1[mask_valid], x2[mask_valid]
        mean_prob0, mean_prob1 = mean_prob0[mask_valid], mean_prob1[mask_valid]
        architecture = architecture[mask_valid]

        positive_mask = mean_prob1 > mean_prob0
        negative_mask = mean_prob0 >= mean_prob1

        # SAS: architecture == 4 (single); MAS: architecture in [0, 1, 2, 3]
        sas_mask = architecture == 4
        mas_mask = architecture < 4

        size_scale = 100
        base_size = 10
        sizes_pos = mean_prob1 * size_scale + base_size
        sizes_neg = mean_prob0 * size_scale + base_size

        mask_sas_neg = negative_mask & sas_mask
        if np.any(mask_sas_neg):
            ax.scatter(x1[mask_sas_neg], x2[mask_sas_neg],
                       s=sizes_neg[mask_sas_neg], c="#4575B4", alpha=0.6,
                       edgecolors="white", linewidths=0.5,
                       label="SAS Negative", marker="s")

        mask_mas_neg = negative_mask & mas_mask
        if np.any(mask_mas_neg):
            ax.scatter(x1[mask_mas_neg], x2[mask_mas_neg],
                       s=sizes_neg[mask_mas_neg], c="#91BFD8", alpha=0.6,
                       edgecolors="white", linewidths=0.5,
                       label="MAS Negative", marker="^")

        mask_sas_pos = positive_mask & sas_mask
        if np.any(mask_sas_pos):
            ax.scatter(x1[mask_sas_pos], x2[mask_sas_pos],
                       s=sizes_pos[mask_sas_pos], c="#D73027", alpha=0.6,
                       edgecolors="white", linewidths=0.5,
                       label="SAS Positive", marker="o")

        mask_mas_pos = positive_mask & mas_mask
        if np.any(mask_mas_pos):
            ax.scatter(x1[mask_mas_pos], x2[mask_mas_pos],
                       s=sizes_pos[mask_mas_pos], c="#FC8D59", alpha=0.6,
                       edgecolors="white", linewidths=0.5,
                       label="MAS Positive", marker="*")

        def format_label(name: str) -> str:
            return name.replace("_", " ").title()

        ax.set_xlabel(format_label(f1), fontsize=15)
        ax.set_ylabel(format_label(f2), fontsize=15)
        ax.text(0.5, -0.15, "(b)", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, fontweight="bold")

        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="best", frameon=True, fancybox=False,
                  edgecolor="black", framealpha=0.95)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def compose(self, filename: str = "mas_analysis.pdf", save_individual: bool = True) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        print("Generating subplot 1: SHAP Scatter with Feature Importance Inset...")
        self.plot_shap_with_importance_inset(axes[0])

        print("Generating subplot 2: Top 2 Entropy Features Scatter...")
        self.plot_top2_entropy_scatter(axes[1])

        plt.tight_layout()
        self.save_figure(fig, filename)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots()

    def _save_individual_subplots(self) -> None:
        subplot_configs = [
            {"name": "shap_scatter_with_inset",
             "plot_func": self.plot_shap_with_importance_inset,
             "figsize": (7, 6)},
            {"name": "entropy_scatter",
             "plot_func": self.plot_top2_entropy_scatter,
             "figsize": (7, 6)},
        ]

        print("\nSaving individual subplots...")
        for config in subplot_configs:
            fig_individual, ax_individual = plt.subplots(figsize=config["figsize"])
            config["plot_func"](ax_individual)
            plt.tight_layout()
            self.save_subplot(fig_individual, config["name"])
            plt.close(fig_individual)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    results_dir = base_dir / "data_mining" / "results_exclue_all_metrics" / "results"
    exp_key = "exclude_base_model_all_metrics"
    feature_importance_path = (
        base_dir / "data_mining" / "results_exclue_all_metrics"
        / "results_aggregated" / "exclude_base_model_all_metrics.csv"
    )
    output_dir = base_dir / "visualization" / "outputs" / "mas"

    top_features = [
        "sample_round_1_max_agent_total_entropy",
        "sample_total_entropy",
    ]

    plotter = MASPlot(
        results_dir=results_dir,
        exp_key=exp_key,
        feature_importance_path=feature_importance_path,
        output_dir=output_dir,
        top_features=top_features,
    )
    plotter.compose()


if __name__ == "__main__":
    main()
