"""Dataset comprehensive analysis: SHAP scatter + combined box plot.

Refactored from results_plot/datasets/analyze_datasets.py into the
`visualization` package layout.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.base import BaseVisualizer
from visualization.base.data_loaders import load_csv, load_shap

warnings.filterwarnings("ignore")


class DatasetsPlot(BaseVisualizer):
    """Visualizer for cross-dataset analysis with ICML-style formatting."""

    DATASETS = [
        "aime2025_16384", "aime2024_16384",
        "math500", "gsm8k", "humaneval", "mmlu",
    ]

    DATASET_DISPLAY_NAMES = {
        "aime2025_16384": "AIME25",
        "aime2024_16384": "AIME24",
        "humaneval": "HE",
        "math500": "MATH500",
        "mmlu": "MMLU",
        "gsm8k": "GSM8K",
    }

    FEATURES = [
        "sample_round_1_max_agent_total_entropy",
        "exp_infer_average_entropy",
    ]

    FEATURE_DISPLAY_NAMES = {
        "sample_variance_entropy": "Variance Entropy",
        "exp_infer_average_entropy": "Average Agent Entropy",
        "sample_round_1_max_agent_total_entropy": "Round 1 Max Agent Entropy",
    }

    def __init__(
        self,
        shap_data_root: Path | str,
        accuracy_data_path: Path | str,
        output_dir: Path | str,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=14)
        self.shap_data_root = Path(shap_data_root)
        self.accuracy_data_path = Path(accuracy_data_path)

        # ICML color palette for datasets
        self.dataset_colors = {
            "aime2024_16384": "#D73027",
            "aime2025_16384": "#FC8D59",
            "gsm8k": "#FEE090",
            "humaneval": "#4575B4",
            "math500": "#91BFD8",
            "mmlu": "#313695",
        }

        self.feature_markers = {
            "sample_round_1_max_agent_total_entropy": "o",
            "exp_infer_average_entropy": "s",
        }

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    def _exp_key(self, dataset_name: str) -> str:
        return f"dataset_{dataset_name}_exclude_base_model_all_metrics"

    def load_shap_data_for_dataset(self, dataset_name: str):
        """Return (shap_values_df, X_test_df) for a dataset."""
        shap_df, x_test_df, _ = load_shap(self.shap_data_root, self._exp_key(dataset_name))
        if shap_df is None or x_test_df is None:
            print(f"Warning: SHAP data not found for {dataset_name}")
            return None, None
        return shap_df, x_test_df

    def load_accuracy_data(self) -> pd.DataFrame:
        return load_csv(self.accuracy_data_path)

    # ------------------------------------------------------------------
    # Subplot helpers
    # ------------------------------------------------------------------
    def plot_shap_scatter(self, ax: plt.Axes) -> None:
        """SHAP value scatter plot for all datasets (subplot a)."""
        all_data = []

        for dataset_name in self.DATASETS:
            shap_df, x_test_df = self.load_shap_data_for_dataset(dataset_name)
            if shap_df is None or x_test_df is None:
                continue

            for feature in self.FEATURES:
                if feature not in shap_df.columns or feature not in x_test_df.columns:
                    print(f"Warning: Feature {feature} not found in {dataset_name}")
                    continue

                feature_values = x_test_df[feature].values
                shap_values = shap_df[feature].values
                for fv, sv in zip(feature_values, shap_values):
                    all_data.append(
                        {
                            "dataset": dataset_name,
                            "feature": feature,
                            "feature_value": fv,
                            "shap_value": sv,
                        }
                    )

        plot_df = pd.DataFrame(all_data)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No SHAP data available", ha="center", va="center", fontsize=12)
            return

        for feature in self.FEATURES:
            mask = plot_df["feature"] == feature
            if mask.any():
                fv_min = plot_df.loc[mask, "feature_value"].min()
                fv_max = plot_df.loc[mask, "feature_value"].max()
                if fv_max > fv_min:
                    plot_df.loc[mask, "feature_value_norm"] = (
                        plot_df.loc[mask, "feature_value"] - fv_min
                    ) / (fv_max - fv_min)
                else:
                    plot_df.loc[mask, "feature_value_norm"] = 0.5

        for dataset_name in self.DATASETS:
            for feature in self.FEATURES:
                mask = (plot_df["dataset"] == dataset_name) & (plot_df["feature"] == feature)
                subset = plot_df[mask]
                if subset.empty:
                    continue

                ax.scatter(
                    subset["feature_value_norm"],
                    subset["shap_value"],
                    alpha=0.5,
                    s=25,
                    color=self.dataset_colors[dataset_name],
                    marker=self.feature_markers[feature],
                    edgecolors="white",
                    linewidth=0.3,
                )

        dataset_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=self.dataset_colors[ds], markersize=8,
                       label=self.DATASET_DISPLAY_NAMES[ds])
            for ds in self.DATASETS
        ]
        feature_handles = [
            plt.Line2D([0], [0], marker=self.feature_markers[f], color="w",
                       markerfacecolor="gray", markersize=8,
                       label=self.FEATURE_DISPLAY_NAMES[f])
            for f in self.FEATURES
        ]

        ax.legend(handles=dataset_handles + feature_handles, loc="lower right",
                  frameon=False, fontsize=10, ncol=2)

        ax.set_xlabel("Normalized Feature Value", fontsize=12)
        ax.set_ylabel("SHAP Value", fontsize=12)
        ax.text(0.5, -0.15, "(a)", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    def plot_combined_violin(self, ax: plt.Axes) -> None:
        """Combined box plot for both features with MAS accuracy annotation (subplot b)."""
        acc_df = self.load_accuracy_data()

        feature_colors = {
            "sample_round_1_max_agent_total_entropy": "#91BFD8",
            "exp_infer_average_entropy": "#FEE090",
        }

        all_data = []
        dataset_accuracies: dict = {}

        for dataset_name in self.DATASETS:
            _, x_test_df = self.load_shap_data_for_dataset(dataset_name)
            if x_test_df is None:
                continue

            # MAS average accuracy excludes 'single' architecture
            dataset_acc = acc_df[
                (acc_df["dataset"] == dataset_name)
                & (acc_df["architecture"] != "single")
            ]["accuracy"].mean()
            dataset_accuracies[dataset_name] = dataset_acc

            for feature in self.FEATURES:
                if feature not in x_test_df.columns:
                    continue

                feature_values = x_test_df[feature].values
                fv_min, fv_max = np.nanmin(feature_values), np.nanmax(feature_values)
                if fv_max > fv_min:
                    normalized_values = (feature_values - fv_min) / (fv_max - fv_min)
                else:
                    normalized_values = np.full_like(feature_values, 0.5)

                for val in normalized_values:
                    if not np.isnan(val):
                        all_data.append({"dataset": dataset_name, "feature": feature, "value": val})

        plot_df = pd.DataFrame(all_data)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
            return

        order = [ds for ds in self.DATASETS if ds in plot_df["dataset"].values]

        width = 0.35
        positions_f1 = np.arange(len(order)) - width / 2
        positions_f2 = np.arange(len(order)) + width / 2

        for fidx, feature in enumerate(self.FEATURES):
            positions = positions_f1 if fidx == 0 else positions_f2
            feature_data = plot_df[plot_df["feature"] == feature]

            data_list = []
            valid_positions = []
            for idx, ds in enumerate(order):
                ds_data = feature_data[feature_data["dataset"] == ds]["value"].values
                if len(ds_data) > 0:
                    data_list.append(ds_data)
                    valid_positions.append(positions[idx])

            if data_list:
                bp = ax.boxplot(
                    data_list,
                    positions=valid_positions,
                    widths=width * 0.8,
                    patch_artist=True,
                    showfliers=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="#333333", markersize=4),
                    flierprops=dict(marker="o", markerfacecolor="none",
                                    markeredgecolor=feature_colors[feature],
                                    markersize=3, alpha=0.5),
                )

                for box in bp["boxes"]:
                    box.set_facecolor(feature_colors[feature])
                    box.set_alpha(0.7)
                    box.set_edgecolor("#333333")

                for element in ["whiskers", "caps"]:
                    for item in bp[element]:
                        item.set_color("#333333")
                        item.set_linewidth(1)

                for median in bp["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.0)

        for idx, ds in enumerate(order):
            acc = dataset_accuracies.get(ds, 0)
            ax.text(idx, 1.05, f"{acc:.0%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#1a1a1a")

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([self.DATASET_DISPLAY_NAMES[ds] for ds in order],
                           rotation=0, ha="center", fontsize=11)

        feature_handles = [
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=feature_colors[f], markersize=10,
                       label=self.FEATURE_DISPLAY_NAMES[f])
            for f in self.FEATURES
        ]
        mean_handle = plt.Line2D([0], [0], marker="D", color="w",
                                 markerfacecolor="white", markeredgecolor="#333333",
                                 markersize=5, label="Mean")
        feature_handles.append(mean_handle)

        ax.legend(handles=feature_handles, loc="upper left",
                  bbox_to_anchor=(0.29, 0.88), frameon=True, fontsize=8.5)

        ax.set_ylabel("Normalized Feature Value", fontsize=12)
        ax.set_xlabel("Dataset", fontsize=12)
        ax.text(0.5, -0.15, "(b)", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    def plot_dual_feature_scatter(self, ax: plt.Axes) -> None:
        """Scatter of two features with dataset and sample type encoding (subplot c)."""
        all_data = []

        for dataset_name in self.DATASETS:
            shap_dir = self.shap_data_root / self._exp_key(dataset_name)
            x_test_path = shap_dir / "shap" / "X_test_LightGBM_classification.csv"

            if not x_test_path.exists():
                continue

            try:
                x_test_df = pd.read_csv(x_test_path, index_col=0)
            except Exception as e:
                print(f"Error loading X_test for {dataset_name}: {e}")
                continue

            f1, f2 = self.FEATURES[0], self.FEATURES[1]
            if f1 not in x_test_df.columns or f2 not in x_test_df.columns:
                continue

            lgbm_pred_path = shap_dir / "shap" / "lightgbm_predictions.csv"
            if lgbm_pred_path.exists():
                try:
                    lgbm_df = pd.read_csv(lgbm_pred_path)
                    prob0 = lgbm_df["prob_class_0"].values
                    prob1 = lgbm_df["prob_class_1"].values
                    positive_mask = prob1 > prob0
                except Exception:
                    positive_mask = np.ones(len(x_test_df), dtype=bool)
            else:
                positive_mask = np.ones(len(x_test_df), dtype=bool)

            for idx in range(min(len(x_test_df), len(positive_mask))):
                all_data.append(
                    {
                        "dataset": dataset_name,
                        "f1": x_test_df[f1].iloc[idx],
                        "f2": x_test_df[f2].iloc[idx],
                        "is_positive": positive_mask[idx],
                    }
                )

        plot_df = pd.DataFrame(all_data)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
            return

        for dataset_name in self.DATASETS:
            subset = plot_df[plot_df["dataset"] == dataset_name]
            if subset.empty:
                continue

            pos_subset = subset[subset["is_positive"]]
            if not pos_subset.empty:
                ax.scatter(
                    pos_subset["f1"], pos_subset["f2"],
                    color=self.dataset_colors[dataset_name],
                    marker="o", s=30, alpha=0.6,
                    edgecolors="white", linewidth=0.5,
                    label=f"{self.DATASET_DISPLAY_NAMES[dataset_name]} (Pos)",
                )

            neg_subset = subset[~subset["is_positive"]]
            if not neg_subset.empty:
                ax.scatter(
                    neg_subset["f1"], neg_subset["f2"],
                    color=self.dataset_colors[dataset_name],
                    marker="s", s=30, alpha=0.6,
                    edgecolors="white", linewidth=0.5,
                    label=f"{self.DATASET_DISPLAY_NAMES[dataset_name]} (Neg)",
                )

        dataset_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=self.dataset_colors[ds], markersize=8,
                       label=self.DATASET_DISPLAY_NAMES[ds])
            for ds in self.DATASETS if ds in plot_df["dataset"].values
        ]
        type_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="gray", markersize=8, label="Positive"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor="gray", markersize=8, label="Negative"),
        ]

        ax.legend(handles=dataset_handles + type_handles, loc="center right",
                  frameon=False, fontsize=9, ncol=1, bbox_to_anchor=(1.0, 0.5))

        ax.set_xlabel(self.FEATURE_DISPLAY_NAMES[self.FEATURES[0]], fontsize=12)
        ax.set_ylabel(self.FEATURE_DISPLAY_NAMES[self.FEATURES[1]], fontsize=12)
        ax.text(0.5, -0.15, "(c)", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------
    def compose(self, filename: str = "dataset_analysis.pdf", save_individual: bool = True) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        print("Plotting SHAP scatter (a)...")
        self.plot_shap_scatter(axes[0])

        print("Plotting combined violin (b)...")
        self.plot_combined_violin(axes[1])

        plt.tight_layout()
        out_path = self.save_figure(fig, filename)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots()

        return out_path

    def _save_individual_subplots(self) -> None:
        subplot_configs = [
            ("shap_scatter", self.plot_shap_scatter, (7, 6)),
            ("combined_violin", self.plot_combined_violin, (7, 6)),
        ]

        print("\nSaving individual subplots...")
        for name, plot_func, figsize in subplot_configs:
            fig, ax = plt.subplots(figsize=figsize)
            plot_func(ax)
            plt.tight_layout()
            self.save_subplot(fig, name)
            plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]

    shap_data_root = base_dir / "data_mining" / "results_all" / "results"
    accuracy_data_path = base_dir / "evaluation" / "results_all" / "combined_summary_data.csv"
    output_dir = base_dir / "visualization" / "outputs" / "datasets"

    plotter = DatasetsPlot(
        shap_data_root=shap_data_root,
        accuracy_data_path=accuracy_data_path,
        output_dir=output_dir,
    )
    plotter.compose("dataset_analysis.pdf")


if __name__ == "__main__":
    main()
