"""Round comparison figure: R=2 vs R=5 + entropy trends + entropy scatter.

Refactored from results_plot/round/analyze_round.py into the
`visualization` package layout.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from visualization.base import ARCH_COLORS, BaseVisualizer
from visualization.base.data_loaders import load_csv, load_shap

warnings.filterwarnings("ignore")


class RoundPlot(BaseVisualizer):
    """Three-subplot round-based visualization."""

    def __init__(
        self,
        r2_summary_path: Path | str,
        r5_summary_path: Path | str,
        r5_math500_data_path: Path | str,
        r5_aime_data_path: Path | str,
        results_dir: Path | str,
        exp_key: str,
        output_dir: Path | str,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=16)

        self.r2_summary_path = Path(r2_summary_path)
        self.r5_summary_path = Path(r5_summary_path)
        self.r5_math500_data_path = Path(r5_math500_data_path)
        self.r5_aime_data_path = Path(r5_aime_data_path)

        # SHAP X_test + LightGBM prediction probabilities
        _, self.x_test_df, self.lightgbm_pred_df = load_shap(
            results_dir, exp_key, model="LightGBM", task="classification"
        )
        # XGBoost prediction probabilities (loaded separately)
        xgb_path = Path(results_dir) / exp_key / "shap" / "shap_prediction_probabilities_XGBoost_classification.csv"
        self.xgboost_pred_df: Optional[pd.DataFrame] = (
            pd.read_csv(xgb_path) if xgb_path.exists() else None
        )

        # Architecture color palette comes from shared ARCH_COLORS.
        self.color_map = ARCH_COLORS

        # Model color mapping
        self.model_colors = {
            "qwen3_0_6b": "#91BFD8",
            "qwen3_4b": "#4575B4",
        }

        # Architecture marker mapping
        self.arch_markers = {
            "single": "v",
            "centralized": "o",
            "debate": "s",
            "hybrid": "^",
            "sequential": "D",
        }

    def plot_r2_r5_comparison(self, ax: plt.Axes) -> None:
        """R=2 vs R=5 model performance, dual Y-axis (Accuracy / Total Tokens)."""
        r2_df = load_csv(self.r2_summary_path)
        r5_df = load_csv(self.r5_summary_path)

        r2_df["round"] = "R=2"
        r5_df["round"] = "R=5"

        combined_df = pd.concat([r2_df, r5_df], ignore_index=True)

        datasets = ["math500", "aime2025_16384"]
        combined_df = combined_df[combined_df["dataset"].isin(datasets)]

        dataset_name_map = {"math500": "Math500", "aime2025_16384": "AIME25"}
        combined_df["dataset_display"] = combined_df["dataset"].map(dataset_name_map)
        combined_df["accuracy_pct"] = combined_df["accuracy"] * 100
        combined_df["token_100k"] = combined_df["token"]

        ax2 = ax.twinx()

        architectures = ["single", "centralized", "debate", "hybrid", "sequential"]
        rounds = ["R=2", "R=5"]
        models = ["qwen3_0_6b", "qwen3_4b"]

        dataset_list = list(dataset_name_map.values())
        n_datasets = len(dataset_list)
        n_archs = len(architectures)
        n_rounds = len(rounds)
        n_models = len(models)

        bar_width = 0.08
        group_width = bar_width * n_archs * n_rounds * n_models + 0.2

        dataset_positions = np.arange(n_datasets) * (group_width + 0.3)

        for d_idx, dataset_display in enumerate(dataset_list):
            base_x = dataset_positions[d_idx]
            bar_idx = 0

            for r_idx, round_val in enumerate(rounds):
                for m_idx, model in enumerate(models):
                    for a_idx, arch in enumerate(architectures):
                        mask = (
                            (combined_df["dataset_display"] == dataset_display) &
                            (combined_df["round"] == round_val) &
                            (combined_df["model"] == model) &
                            (combined_df["architecture"] == arch)
                        )
                        data = combined_df[mask]

                        if len(data) > 0:
                            x_pos = base_x + bar_idx * bar_width
                            acc = data["accuracy_pct"].values[0]
                            tok = data["token_100k"].values[0]

                            color = self.model_colors[model]
                            hatch = "" if round_val == "R=2" else "///"

                            ax.bar(x_pos, acc, bar_width * 0.9,
                                   color=color, edgecolor="white",
                                   linewidth=0.5, hatch=hatch, alpha=0.85)

                            ax2.scatter(x_pos, tok, marker=self.arch_markers[arch],
                                        color=color, s=25, edgecolors="black",
                                        linewidths=0.5, alpha=0.9, zorder=5)

                        bar_idx += 1

        ax.set_ylabel("Accuracy (%)", fontsize=17)
        ax.set_xlabel("Dataset", fontsize=17)
        ax.set_xticks(dataset_positions + group_width / 2 - 0.1)
        ax.set_xticklabels(dataset_list, fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        ax2.set_ylabel("Total Tokens (100K)", fontsize=17)
        ax2.set_ylim(0, max(combined_df["token_100k"]) * 1.2)

        model_legend = [
            Patch(facecolor=self.model_colors["qwen3_0_6b"], label="Qwen3-0.6B"),
            Patch(facecolor=self.model_colors["qwen3_4b"], label="Qwen3-4B"),
        ]

        round_legend = [
            Patch(facecolor="white", edgecolor="black", label="R=2"),
            Patch(facecolor="white", edgecolor="black", hatch="///", label="R=5"),
        ]

        arch_legend = [
            Line2D([0], [0], marker=self.arch_markers["single"], color="gray",
                   linestyle="None", markersize=8, label="Single"),
            Line2D([0], [0], marker=self.arch_markers["debate"], color="gray",
                   linestyle="None", markersize=8, label="Debate"),
            Line2D([0], [0], marker=self.arch_markers["centralized"], color="gray",
                   linestyle="None", markersize=8, label="Centralized"),
            Line2D([0], [0], marker=self.arch_markers["sequential"], color="gray",
                   linestyle="None", markersize=8, label="Sequential"),
            Line2D([0], [0], marker=self.arch_markers["hybrid"], color="gray",
                   linestyle="None", markersize=8, label="Hybrid"),
        ]

        legend_elements = model_legend + round_legend + arch_legend
        ax.legend(handles=legend_elements, loc="upper left", frameon=False,
                  fontsize=10, ncol=3)

        ax.text(0.5, -0.25, "(a)", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, fontweight="bold")

        sns.despine(ax=ax, top=True, right=False)

    def plot_round_entropy_trends(self, ax: plt.Axes) -> None:
        """Entropy trends across rounds for Math500 + AIME25."""
        dfs = []
        for path, dataset in [(self.r5_math500_data_path, "Math500"),
                              (self.r5_aime_data_path, "AIME25")]:
            if Path(path).exists():
                df = pd.read_csv(path)
                df["dataset"] = dataset
                dfs.append(df)

        if not dfs:
            ax.text(0.5, 0.5, "Data files not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(b)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=18, fontweight="bold")
            return

        combined_df = pd.concat(dfs, ignore_index=True)

        entropy_features = [
            "all_agents_total_entropy",
            "mean_agent_total_entropy",
            "max_agent_total_entropy",
        ]

        feature_styles = {
            "all_agents_total_entropy": {"linestyle": "-", "color": "#D73027"},
            "mean_agent_total_entropy": {"linestyle": "--", "color": "#4575B4"},
            "max_agent_total_entropy": {"linestyle": "-.", "color": "#FC8D59"},
        }

        feature_labels = {
            "all_agents_total_entropy": "Total Entropy",
            "mean_agent_total_entropy": "Mean Agent Entropy",
            "max_agent_total_entropy": "Max Agent Entropy",
        }

        rounds = range(1, 6)

        for feature in entropy_features:
            round_values = []
            round_stds = []

            for r in rounds:
                col_name = f"sample_round_{r}_{feature}"
                if col_name in combined_df.columns:
                    values = combined_df[col_name].dropna()
                    if len(values) > 0:
                        round_values.append(values.mean())
                        round_stds.append(values.std())
                    else:
                        round_values.append(np.nan)
                        round_stds.append(np.nan)
                else:
                    round_values.append(np.nan)
                    round_stds.append(np.nan)

            style = feature_styles[feature]
            ax.plot(list(rounds), round_values,
                    linestyle=style["linestyle"],
                    color=style["color"],
                    marker="o", markersize=8,
                    linewidth=2.5,
                    label=feature_labels[feature],
                    alpha=0.9)

            round_values = np.array(round_values)
            round_stds = np.array(round_stds)
            ax.fill_between(list(rounds),
                            round_values - round_stds,
                            round_values + round_stds,
                            color=style["color"], alpha=0.15)

        ax.set_xlabel("Round Number", fontsize=17)
        ax.set_ylabel("Entropy Value", fontsize=17)
        ax.set_xticks(list(rounds))
        ax.set_xticklabels([str(r) for r in rounds], fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="best", frameon=False, fontsize=12)

        ax.text(0.5, -0.25, "(b)", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, fontweight="bold")

        sns.despine(ax=ax, top=True, right=True)

    def plot_entropy_scatter(self, ax: plt.Axes) -> None:
        """Two-feature entropy scatter; markers separate SAS vs MAS, color by predicted class."""
        x_test = self.x_test_df

        if x_test is None:
            ax.text(0.5, 0.5, "X_test file not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=18, fontweight="bold")
            return

        x_feature = "sample_round_1_max_agent_total_entropy"
        y_feature = "sample_total_entropy"

        if x_feature not in x_test.columns or y_feature not in x_test.columns:
            ax.text(0.5, 0.5, "Required features not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=18, fontweight="bold")
            return

        if "architecture" not in x_test.columns:
            ax.text(0.5, 0.5, "Architecture column not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=18, fontweight="bold")
            return

        if self.lightgbm_pred_df is None or self.xgboost_pred_df is None:
            ax.text(0.5, 0.5, "Prediction probability files not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=18, fontweight="bold")
            return

        lgbm_prob0 = pd.to_numeric(self.lightgbm_pred_df["prob_class_0"].values, errors="coerce")
        lgbm_prob1 = pd.to_numeric(self.lightgbm_pred_df["prob_class_1"].values, errors="coerce")
        xgb_prob0 = pd.to_numeric(self.xgboost_pred_df["prob_class_0"].values, errors="coerce")
        xgb_prob1 = pd.to_numeric(self.xgboost_pred_df["prob_class_1"].values, errors="coerce")

        x1 = x_test[x_feature].values
        x2 = x_test[y_feature].values
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

        ax.set_xlabel("Round 1 Max Agent Total Entropy", fontsize=17)
        ax.set_ylabel("Sample Total Entropy", fontsize=17)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", frameon=False, fontsize=11, ncol=2)

        ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def compose(self, filename: str = "round_analysis.pdf", save_individual: bool = True) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        print("Generating subplot 1: R=2 vs R=5 Performance Comparison...")
        self.plot_r2_r5_comparison(axes[0])

        print("Generating subplot 2: Round Entropy Trends...")
        self.plot_round_entropy_trends(axes[1])

        print("Generating subplot 3: Entropy Feature Scatter...")
        self.plot_entropy_scatter(axes[2])

        plt.tight_layout()
        self.save_figure(fig, filename)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots()

    def _save_individual_subplots(self) -> None:
        subplot_configs = [
            {"name": "r2_r5_comparison",
             "plot_func": self.plot_r2_r5_comparison,
             "figsize": (8, 5)},
            {"name": "round_entropy_trends",
             "plot_func": self.plot_round_entropy_trends,
             "figsize": (7, 5)},
            {"name": "entropy_scatter",
             "plot_func": self.plot_entropy_scatter,
             "figsize": (7, 5)},
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

    r2_summary_path = base_dir / "evaluation" / "results_R_2" / "combined_summary_data.csv"
    r5_summary_path = base_dir / "evaluation" / "results_R_5" / "combined_summary_data.csv"

    r5_math500_data_path = base_dir / "evaluation" / "results_R_5" / "math500" / "all_aggregated_data_exclude_agent.csv"
    r5_aime_data_path = base_dir / "evaluation" / "results_R_5" / "aime2025_16384" / "all_aggregated_data_exclude_agent.csv"

    results_dir = base_dir / "data_mining" / "results_round_5" / "results"
    exp_key = "exclude_base_model_all_metrics"

    output_dir = base_dir / "visualization" / "outputs" / "round"

    plotter = RoundPlot(
        r2_summary_path=r2_summary_path,
        r5_summary_path=r5_summary_path,
        r5_math500_data_path=r5_math500_data_path,
        r5_aime_data_path=r5_aime_data_path,
        results_dir=results_dir,
        exp_key=exp_key,
        output_dir=output_dir,
    )
    plotter.compose()


if __name__ == "__main__":
    main()
