"""Architecture analysis: SHAP scatter + entropy box plot.

Refactored from results_plot/arch/analyze_arch.py into the
`visualization` package layout.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visualization.base import (
    ARCH_COLORS,
    ARCH_ORDER,
    FEATURE_COLORS,
    BaseVisualizer,
)
from visualization.base.data_loaders import load_csv, load_json, load_shap

warnings.filterwarnings("ignore")


class ArchPlot(BaseVisualizer):
    """Visualizer for architecture-based entropy analysis with ICML-style formatting."""

    ARCHITECTURES = ARCH_ORDER
    ARCH_DISPLAY_NAMES = {
        "centralized": "Centralized",
        "debate": "Debate",
        "hybrid": "Hybrid",
        "sequential": "Sequential",
        "single": "Single",
    }

    # Entropy type categories with representative features
    ENTROPY_TYPES = {
        "Peak Entropy": [
            "sample_round_1_max_agent_max_entropy",
            "sample_round_1_q3_agent_max_entropy",
            "sample_max_entropy",
        ],
        "Answer Entropy": [
            "sample_max_answer_token_entropy",
            "sample_mean_answer_token_entropy",
            "sample_answer_token_count",
        ],
        "Cumulative Entropy": [
            "exp_total_entropy",
            "sample_total_entropy",
            "exp_infer_average_entropy",
        ],
        "Temporal Stability": [
            "sample_variance_entropy",
            "sample_std_entropy",
            "sample_round_1_2_change_entropy",
        ],
    }

    BOX_FEATURES = [
        "sample_round_1_q3_agent_max_entropy",
        "sample_round_1_max_agent_std_entropy",
    ]

    SCATTER_FEATURES = [
        "sample_round_1_q3_agent_max_entropy",
        "sample_round_1_max_agent_std_entropy",
    ]

    FEATURE_DISPLAY_NAMES = {
        "sample_round_1_q3_agent_max_entropy": "R1 Q3 Agent Max Entropy",
        "sample_max_answer_token_entropy": "Max Answer Token Entropy",
        "sample_round_1_max_agent_max_entropy": "R1 Max Agent Max Entropy",
        "sample_round_1_max_agent_std_entropy": "R1 Max Agent Std Entropy",
    }

    FEATURE_MARKERS = {
        "sample_round_1_q3_agent_max_entropy": "o",
        "sample_round_1_max_agent_std_entropy": "s",
    }

    def __init__(
        self,
        summary_json_path: Path | str,
        results_dir: Path | str,
        accuracy_data_path: Path | str,
        output_dir: Path | str,
    ) -> None:
        super().__init__(output_dir=output_dir, base_font_size=14)
        self.summary_json_path = Path(summary_json_path)
        self.results_dir = Path(results_dir)
        self.accuracy_data_path = Path(accuracy_data_path)

        self.arch_colors = ARCH_COLORS
        self.feature_colors = FEATURE_COLORS

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    def _exp_key(self, arch: str) -> str:
        return f"arch_{arch}_exclude_base_model_all_metrics"

    def load_summary_data(self) -> dict:
        return load_json(self.summary_json_path)

    def load_accuracy_data(self) -> pd.DataFrame:
        return load_csv(self.accuracy_data_path)

    def load_arch_data(self, arch: str):
        """Return (X_test_df, predictions_df) for an architecture."""
        shap_df, x_test_df, pred_df = load_shap(self.results_dir, self._exp_key(arch))
        if x_test_df is None:
            print(f"Warning: X_test not found for {arch}")
            return None, None
        return x_test_df, pred_df

    def load_shap_data_for_arch(self, arch: str):
        """Return (shap_values_df, X_test_df) for an architecture."""
        shap_df, x_test_df, _ = load_shap(self.results_dir, self._exp_key(arch))
        if shap_df is None or x_test_df is None:
            print(f"Warning: SHAP data not found for {arch}")
            return None, None
        return shap_df, x_test_df

    def compute_heatmap_data(self) -> pd.DataFrame:
        """Compute Architecture x Entropy Type importance matrix."""
        summary_data = self.load_summary_data()

        heatmap_data: dict = {}

        for arch in self.ARCHITECTURES:
            arch_key = self._exp_key(arch)
            if arch_key not in summary_data:
                continue

            arch_features = {
                item["feature_name"]: item["mean_importance_normalized"]
                for item in summary_data[arch_key]
            }

            arch_row = {}
            for entropy_type, features in self.ENTROPY_TYPES.items():
                importances = [arch_features[f] for f in features if f in arch_features]
                arch_row[entropy_type] = float(np.mean(importances)) if importances else 0.0

            heatmap_data[self.ARCH_DISPLAY_NAMES[arch]] = arch_row

        return pd.DataFrame(heatmap_data).T

    # ------------------------------------------------------------------
    # Subplot helpers
    # ------------------------------------------------------------------
    def plot_shap_scatter(self, ax: plt.Axes) -> None:
        """SHAP scatter for all architectures (subplot c). Colors=arch, markers=feature."""
        all_data = []

        for arch in self.ARCHITECTURES:
            shap_df, x_test_df = self.load_shap_data_for_arch(arch)
            if shap_df is None or x_test_df is None:
                continue

            for feature in self.SCATTER_FEATURES:
                if feature not in shap_df.columns or feature not in x_test_df.columns:
                    print(f"Warning: Feature {feature} not found in {arch}")
                    continue

                feature_values = x_test_df[feature].values
                shap_values = shap_df[feature].values
                for fv, sv in zip(feature_values, shap_values):
                    all_data.append(
                        {
                            "architecture": arch,
                            "feature": feature,
                            "feature_value": fv,
                            "shap_value": sv,
                        }
                    )

        plot_df = pd.DataFrame(all_data)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No SHAP data available", ha="center", va="center", fontsize=12)
            return

        # Normalize feature values per feature to [0, 1]
        for feature in self.SCATTER_FEATURES:
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

        for arch in self.ARCHITECTURES:
            for feature in self.SCATTER_FEATURES:
                mask = (plot_df["architecture"] == arch) & (plot_df["feature"] == feature)
                subset = plot_df[mask]
                if subset.empty:
                    continue

                ax.scatter(
                    subset["feature_value_norm"],
                    subset["shap_value"],
                    alpha=0.5,
                    s=25,
                    color=self.arch_colors[arch],
                    marker=self.FEATURE_MARKERS[feature],
                    edgecolors="white",
                    linewidth=0.3,
                )

        arch_handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=self.arch_colors[arch],
                markersize=8, label=self.ARCH_DISPLAY_NAMES[arch],
            )
            for arch in self.ARCHITECTURES
        ]
        feature_handles = [
            plt.Line2D(
                [0], [0], marker=self.FEATURE_MARKERS[feature], color="w",
                markerfacecolor="gray", markersize=8,
                label=self.FEATURE_DISPLAY_NAMES[feature],
            )
            for feature in self.SCATTER_FEATURES
        ]

        ax.legend(handles=arch_handles + feature_handles, loc="lower left",
                  frameon=False, fontsize=11, ncol=1)

        ax.set_xlabel("Normalized Feature Value", fontsize=12)
        ax.set_ylabel("SHAP Value", fontsize=12)
        ax.text(0.5, -0.15, "(c)", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    def plot_arch_boxplot(self, ax: plt.Axes) -> None:
        """Box plot of entropy features by architecture with accuracy annotation (subplot d)."""
        acc_df = self.load_accuracy_data()

        all_data = []
        arch_accuracies: dict = {}

        for arch in self.ARCHITECTURES:
            x_test_df, _ = self.load_arch_data(arch)
            if x_test_df is None:
                continue

            arch_acc = acc_df[acc_df["architecture"] == arch]["accuracy"].mean()
            arch_accuracies[arch] = arch_acc

            for feature in self.BOX_FEATURES:
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
                        all_data.append({"architecture": arch, "feature": feature, "value": val})

        plot_df = pd.DataFrame(all_data)

        if plot_df.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
            return

        order = [a for a in self.ARCHITECTURES if a in plot_df["architecture"].values]

        n_features = len(self.BOX_FEATURES)
        width = 0.25

        for fidx, feature in enumerate(self.BOX_FEATURES):
            offset = (fidx - (n_features - 1) / 2) * width
            positions = np.arange(len(order)) + offset
            feature_data = plot_df[plot_df["feature"] == feature]

            data_list = []
            valid_positions = []
            for idx, arch in enumerate(order):
                arch_data = feature_data[feature_data["architecture"] == arch]["value"].values
                if len(arch_data) > 0:
                    data_list.append(arch_data)
                    valid_positions.append(positions[idx])

            if data_list:
                bp = ax.boxplot(
                    data_list,
                    positions=valid_positions,
                    widths=width * 0.85,
                    patch_artist=True,
                    showfliers=True,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="#333333", markersize=4),
                    flierprops=dict(marker="o", markerfacecolor="none",
                                    markeredgecolor=self.feature_colors[feature],
                                    markersize=3, alpha=0.5),
                )

                for box in bp["boxes"]:
                    box.set_facecolor(self.feature_colors[feature])
                    box.set_alpha(0.7)
                    box.set_edgecolor("#333333")

                for element in ["whiskers", "caps"]:
                    for item in bp[element]:
                        item.set_color("#333333")
                        item.set_linewidth(1)

                for median in bp["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.0)

        for idx, arch in enumerate(order):
            acc = arch_accuracies.get(arch, 0)
            ax.text(idx, 1.05, f"{acc:.0%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#1a1a1a")

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([self.ARCH_DISPLAY_NAMES[a] for a in order],
                           rotation=0, ha="center", fontsize=11)

        feature_handles = [
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=self.feature_colors[f], markersize=10,
                       label=self.FEATURE_DISPLAY_NAMES[f])
            for f in self.BOX_FEATURES
        ]
        mean_handle = plt.Line2D([0], [0], marker="D", color="w",
                                 markerfacecolor="white", markeredgecolor="#333333",
                                 markersize=5, label="Mean")
        feature_handles.append(mean_handle)

        ax.legend(handles=feature_handles, loc="upper center",
                  bbox_to_anchor=(0.62, 0.88), frameon=True, fontsize=8.5)

        ax.set_ylabel("Normalized Feature Value", fontsize=12)
        ax.set_xlabel("Architecture", fontsize=12)
        ax.text(0.5, -0.15, "(d)", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, fontweight="bold")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------
    def compose(self, filename: str = "arch_analysis.pdf", save_individual: bool = True) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        print("Generating subplot 1: SHAP Scatter Plot...")
        self.plot_shap_scatter(axes[0])

        print("Generating subplot 2: Architecture Box Plot...")
        self.plot_arch_boxplot(axes[1])

        plt.tight_layout()
        out_path = self.save_figure(fig, filename)
        plt.close(fig)

        if save_individual:
            self._save_individual_subplots()

        return out_path

    def _save_individual_subplots(self) -> None:
        subplot_configs = [
            ("shap_scatter", self.plot_shap_scatter, (7, 6)),
            ("arch_boxplot", self.plot_arch_boxplot, (7, 6)),
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

    summary_json_path = base_dir / "data_mining" / "results_arch" / "results_summaries" / "summary.json"
    results_dir = base_dir / "data_mining" / "results_arch" / "results"
    accuracy_data_path = base_dir / "evaluation" / "results_all" / "combined_summary_data.csv"
    output_dir = base_dir / "visualization" / "outputs" / "arch"

    plotter = ArchPlot(
        summary_json_path=summary_json_path,
        results_dir=results_dir,
        accuracy_data_path=accuracy_data_path,
        output_dir=output_dir,
    )
    plotter.compose()


if __name__ == "__main__":
    main()
