"""Comprehensive visualization for RL-model analysis.

Refactored from results_plot/rl_model/analyze_rl_model.py into the
`visualization` package layout. Labels translated to English.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.base import ARCH_COLORS, ARCH_ORDER_WITH_BASE, BaseVisualizer
from visualization.base.data_loaders import load_csv

warnings.filterwarnings("ignore")


class RLModelPlot(BaseVisualizer):
    """Visualizer for RL model comprehensive analysis."""

    def __init__(
        self,
        combined_summary_path: Path | str,
        shap_x_test_path: Path | str,
        shap_values_path: Path | str,
        lightgbm_pred_path: Path | str,
        xgboost_pred_path: Path | str,
        merged_data_path: Path | str,
        output_dir: Path | str,
        top_features: list | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=14)
        self.combined_summary_path = Path(combined_summary_path)
        self.shap_x_test_path = Path(shap_x_test_path)
        self.shap_values_path = Path(shap_values_path)
        self.lightgbm_pred_path = Path(lightgbm_pred_path)
        self.xgboost_pred_path = Path(xgboost_pred_path)
        self.merged_data_path = Path(merged_data_path)

        if top_features is None:
            self.feature1 = "sample_round_1_median_agent_total_entropy"
            self.feature2 = "sample_round_1_q3_agent_total_entropy"
        else:
            self.feature1 = top_features[0]
            self.feature2 = top_features[1] if len(top_features) > 1 else top_features[0]

        # Override RC params to match original RL plot font sizing.
        plt.rcParams["font.size"] = 16
        plt.rcParams["xtick.labelsize"] = 15
        plt.rcParams["ytick.labelsize"] = 15
        plt.rcParams["axes.labelsize"] = 17
        plt.rcParams["legend.title_fontsize"] = 16
        plt.rcParams["legend.fontsize"] = 15
        plt.rcParams["axes.unicode_minus"] = False

    # ---------- subplot 1: accuracy bar ----------

    def plot_accuracy_bar(self, ax) -> None:
        if not self.combined_summary_path.exists():
            ax.text(0.5, 0.5, "Combined summary not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(a)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=18, fontweight="bold")
            return

        df = load_csv(self.combined_summary_path)

        # Custom dataset order: GSM8K at the end.
        all_datasets = df["dataset"].unique()
        dataset_order = []
        gsm8k_dataset = None
        for dataset in sorted(all_datasets):
            if "gsm8k" in dataset.lower():
                gsm8k_dataset = dataset
            else:
                dataset_order.append(dataset)
        if gsm8k_dataset:
            dataset_order.append(gsm8k_dataset)
        datasets = dataset_order

        base_df_list = []
        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            for model in dataset_df["model"].unique():
                model_data = dataset_df[dataset_df["model"] == model].iloc[0]
                base_df_list.append({
                    "dataset": dataset,
                    "model": model,
                    "architecture": "base",
                    "accuracy": model_data["base model accuracy"] / 100.0,
                })
        base_df = pd.DataFrame(base_df_list)
        combined_df = pd.concat([base_df, df], ignore_index=True)
        combined_df["dataset_model"] = combined_df["dataset"]

        dataset_name_map = {
            "math500": "Math500",
            "aime2024_16384": "AIME24",
            "aime2025_16384": "AIME25",
            "gsm8k": "GSM8K",
            "humaneval": "HE",
            "mmlu": "MMLU",
        }
        combined_df["dataset_model"] = combined_df["dataset_model"].map(dataset_name_map)
        dataset_display_order = ["AIME24", "AIME25", "HE", "Math500", "MMLU", "GSM8K"]

        combined_df["accuracy"] = combined_df["accuracy"] * 100

        with sns.plotting_context("paper", font_scale=1.4):
            sns.barplot(
                data=combined_df,
                x="dataset_model",
                y="accuracy",
                hue="architecture",
                hue_order=ARCH_ORDER_WITH_BASE,
                order=dataset_display_order,
                ax=ax,
                palette=ARCH_COLORS,
                edgecolor="white",
                linewidth=0.8,
                saturation=0.9,
            )

        ax.set_xlabel("Dataset", fontsize=17)
        ax.set_ylabel("Accuracy (%)", fontsize=17)
        ax.text(0.5, -0.25, "(a)", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
        ax.set_ylim(bottom=0)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)

        arch_label = {
            "base": "Base",
            "centralized": "Centralized",
            "debate": "Debate",
            "hybrid": "Hybrid",
            "sequential": "Sequential",
            "single": "Single",
        }
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [arch_label.get(l, l) for l in labels],
                  loc="best", frameon=False, fontsize=14, ncol=2)

    # ---------- subplot 2: accuracy trends ----------

    def plot_accuracy_trends(self, ax) -> None:
        if not self.merged_data_path.exists():
            ax.text(0.5, 0.5, "Data file not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(b)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        df = load_csv(self.merged_data_path)

        if "is_finally_correct" not in df.columns or "architecture" not in df.columns:
            ax.text(0.5, 0.5, "Required columns not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(b)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        focus_features = ["base_sample_total_entropy"]
        focus_feature = next((f for f in focus_features if f in df.columns), None)
        if focus_feature is None:
            ax.text(0.5, 0.5, "No applicable feature found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(b)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        architectures = ["centralized", "debate", "hybrid", "sequential", "single"]
        available_archs = [a for a in architectures if a in df["architecture"].unique()]

        base_stats = []
        if "model_name" in df.columns and "base_model_accuracy" in df.columns:
            for model in df["model_name"].unique():
                model_df = df[df["model_name"] == model]
                if not model_df.empty and focus_feature in model_df.columns:
                    base_stats.append({
                        "model": model,
                        "avg_entropy": model_df[focus_feature].mean(),
                        "avg_acc": model_df["base_model_accuracy"].mean() * 100,
                    })

        try:
            df["feature_bin"] = pd.qcut(df[focus_feature], q=10, duplicates="drop")
        except Exception:
            df["feature_bin"] = pd.cut(df[focus_feature], bins=10)

        arch_label = {
            "centralized": "Centralized",
            "debate": "Debate",
            "hybrid": "Hybrid",
            "sequential": "Sequential",
            "single": "Single",
        }

        for arch in available_archs:
            arch_df = df[df["architecture"] == arch]
            grouped = arch_df.groupby("feature_bin")["is_finally_correct"].mean()
            bin_centers = [(iv.left + iv.right) / 2 for iv in grouped.index]
            ax.plot(
                bin_centers,
                grouped.values * 100,
                marker="o",
                linewidth=2,
                markersize=6,
                color=ARCH_COLORS.get(arch, "#999999"),
                label=arch_label.get(arch, arch),
                alpha=0.8,
            )

        model_name_map = {"qwen_2_5_7b_simplerl_zoo": "Qwen-2.5-7B-RL"}
        if base_stats:
            markers = ["X", "^", "s"]
            colors = ["#D73027", "#56B4E9", "#FEE090"]
            for i, stat in enumerate(base_stats):
                if stat.get("model") in model_name_map:
                    stat["model"] = model_name_map[stat["model"]]
                ax.scatter(
                    stat["avg_entropy"],
                    stat["avg_acc"],
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    s=120,
                    label=f'{stat["model"]} (Base)',
                    edgecolors="white",
                    linewidths=1.5,
                    zorder=5,
                )

        feature_map = {"base_sample_total_entropy": "Base Model Entropy"}
        feature_label = feature_map.get(focus_feature, focus_feature.replace("_", " ").title())
        ax.set_xlabel(feature_label, fontsize=16)
        ax.set_ylabel("Accuracy (%)", fontsize=16)
        ax.text(0.5, -0.25, "(b)", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, fontweight="bold")
        ax.legend(loc="best", frameon=False, fontsize=12)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    # ---------- subplot 3: top-2 entropy scatter ----------

    def plot_top2_entropy_scatter(self, ax) -> None:
        if not self.shap_x_test_path.exists():
            ax.text(0.5, 0.5, "X_test file not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        x_test = load_csv(self.shap_x_test_path)
        f1, f2 = self.feature1, self.feature2

        if f1 not in x_test.columns or f2 not in x_test.columns:
            print(f"Warning: Required features not found ({f1}, {f2}).")
            ax.text(0.5, 0.5, f"Required features not found\n({f1}, {f2})",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        if "architecture" not in x_test.columns:
            ax.text(0.5, 0.5, "Architecture column not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        if not self.lightgbm_pred_path.exists() or not self.xgboost_pred_path.exists():
            ax.text(0.5, 0.5, "Prediction files not found",
                    ha="center", va="center", fontsize=12)
            ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, fontweight="bold")
            return

        lightgbm_df = load_csv(self.lightgbm_pred_path)
        xgboost_df = load_csv(self.xgboost_pred_path)

        lgbm_prob0 = pd.to_numeric(lightgbm_df["prob_class_0"].values, errors="coerce")
        lgbm_prob1 = pd.to_numeric(lightgbm_df["prob_class_1"].values, errors="coerce")
        xgb_prob0 = pd.to_numeric(xgboost_df["prob_class_0"].values, errors="coerce")
        xgb_prob1 = pd.to_numeric(xgboost_df["prob_class_1"].values, errors="coerce")

        x1 = x_test[f1].values
        x2 = x_test[f2].values
        architecture = x_test["architecture"].values

        # round_2_total_entropy is on a much larger scale; bring it down for plotting.
        if f2 == "round_2_total_entropy":
            x2 = x_test[f2].values / 1000.0
        elif f1 == "round_2_total_entropy":
            x1 = x_test[f1].values / 1000.0

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

        # SAS: architecture == 4 (single); MAS: architecture in [0, 1, 2, 3].
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

        feature_map = {
            "sample_round_1_median_agent_total_entropy": "Round 1 Median Agent Entropy",
            "sample_round_1_q3_agent_total_entropy": "Round 1 Q3 Total Entropy",
            "round_2_total_entropy": "Round 2 Total Entropy",
            "round_1_2_change_entropy": "Round 1-2 Change Entropy",
            "base_sample_total_entropy": "Base Sample Total Entropy",
        }
        ax.set_xlabel(f"{feature_map.get(self.feature1, self.feature1)}", fontsize=17)
        ax.set_ylabel(f"{feature_map.get(self.feature2, self.feature2)}", fontsize=17)
        ax.text(0.5, -0.25, "(c)", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", frameon=True, fancybox=False,
                  edgecolor="black", framealpha=0.95, fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---------- main entry ----------

    def compose(self, filename: str = "rl_model_analysis.pdf", save_individual: bool = True) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        print("Generating subplot 1: Accuracy Bar Chart...")
        self.plot_accuracy_bar(axes[0])

        print("Generating subplot 2: Accuracy Trends...")
        self.plot_accuracy_trends(axes[1])

        print("Generating subplot 3: Top 2 Entropy Features Scatter...")
        self.plot_top2_entropy_scatter(axes[2])

        plt.tight_layout()
        self.save_figure(fig, filename)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots()

    def _save_individual_subplots(self) -> None:
        configs = [
            ("accuracy_bar", self.plot_accuracy_bar, (8, 5)),
            ("accuracy_trends", self.plot_accuracy_trends, (7, 5)),
            ("entropy_scatter", self.plot_top2_entropy_scatter, (7, 5)),
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

    combined_summary_path = root / "evaluation" / "results_rl" / "combined_summary_data.csv"
    shap_dir = root / "data_mining" / "results_rl" / "results" / "exclude_base_model_wo_entropy" / "shap"
    shap_x_test_path = shap_dir / "X_test_LightGBM_classification.csv"
    shap_values_path = shap_dir / "shap_values_LightGBM_classification.csv"
    lightgbm_pred_path = shap_dir / "shap_prediction_probabilities_LightGBM_classification.csv"
    xgboost_pred_path = shap_dir / "shap_prediction_probabilities_XGBoost_classification.csv"
    merged_data_path = root / "results_plot" / "rl_model" / "merged_datasets.csv"
    output_dir = root / "visualization" / "outputs" / "rl_model"

    top_features = [
        "round_2_total_entropy",
        "sample_round_1_median_agent_total_entropy",
    ]

    plotter = RLModelPlot(
        combined_summary_path=combined_summary_path,
        shap_x_test_path=shap_x_test_path,
        shap_values_path=shap_values_path,
        lightgbm_pred_path=lightgbm_pred_path,
        xgboost_pred_path=xgboost_pred_path,
        merged_data_path=merged_data_path,
        output_dir=output_dir,
        top_features=top_features,
    )
    plotter.compose()


if __name__ == "__main__":
    main()
