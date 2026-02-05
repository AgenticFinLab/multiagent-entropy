"""Visualization module for multi-agent system entropy analysis.

This module provides classes and functions for generating various types of visualizations to analyze entropy patterns, correlations, and trends in multi-agent systems.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from constants import (
    ARCHITECTURES,
    MULTI_AGENT_ARCHITECTURES,
    SINGLE_AGENT_ARCHITECTURES,
)

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class EntropyVisualizer:
    """Visualizes entropy analysis results for multi-agent system performance.

    This class provides methods to create various types of visualizations
    including architecture comparisons, correlation analysis, evolution plots,
    collaboration pattern comparisons, and comprehensive dashboards.

    Attributes:
        data: DataFrame containing the preprocessed experimental data.
        output_dir: Path to the directory where visualizations will be saved.
        architectures: List of available architecture types.
        analysis_level: Level of analysis ('dataset' or 'model').
    """

    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str,
        analysis_level: str = "model",
        num_models: int = 1,
    ) -> None:
        """Initialize the EntropyVisualizer with data and output directory.

        Args:
            data: DataFrame containing preprocessed experimental data.
            output_dir: Directory path where visualizations will be saved.
            analysis_level: Level of analysis ('dataset' or 'model').
            num_models: Number of models in the dataset for visualization purposes.
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.architectures = ARCHITECTURES
        self.analysis_level = analysis_level
        self.num_models = (
            num_models
            if num_models != 1
            else (data["model_name"].nunique() if "model_name" in data.columns else 1)
        )

    def plot_architecture_entropy_comparison(self) -> None:
        """Generate box plots comparing entropy features across architectures.

        Creates a grid of box plots showing the distribution of entropy
        features for different architecture types.
        """
        print("Generating architecture entropy comparison plots...")

        entropy_features = [
            col
            for col in self.data.columns
            if "entropy" in col.lower() and "mean" in col
        ]

        num_features = len(entropy_features)
        if num_features == 0:
            print("No entropy features found for comparison.")
            return

        # If at dataset level with multiple models, create a 2xn grid layout where n is number of models
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            unique_models = self.data["model_name"].unique()
            num_models = len(unique_models)

            # Create 2 x num_models grid layout
            rows = 2
            cols = num_models

            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 10))
            if num_models == 1:
                axes = np.array([[axes[0]], [axes[1]]])

            fig.suptitle(
                f"Architecture Entropy Comparison by Model ({self.num_models} Models)",
                fontsize=16,
                fontweight="bold",
            )

            for col_idx, model in enumerate(unique_models):
                model_data = self.data[self.data["model_name"] == model]

                for row_idx, feature in enumerate(
                    entropy_features[:2]
                ):  # Use first 2 features, or fewer if available
                    ax = axes[row_idx, col_idx]

                    data_to_plot = []
                    labels = []
                    for arch in self.architectures:
                        arch_data = model_data[model_data["architecture"] == arch][
                            feature
                        ].dropna()
                        data_to_plot.append(arch_data.values)
                        labels.append(arch)

                    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

                    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
                    for patch, color in zip(bp["boxes"], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax.set_xlabel("Architecture Type", fontsize=10)
                    ax.set_ylabel("Entropy Value", fontsize=10)
                    ax.set_title(
                        f"{feature.replace('_', ' ').title()}\nModel: {model}",
                        fontsize=11,
                    )
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis="x", rotation=45)

                # If we have more features, we can add more rows or handle differently
                for remaining_row in range(len(entropy_features), 2):
                    # Hide unused subplots
                    if remaining_row < rows:
                        fig.delaxes(axes[remaining_row, col_idx])

            # Add model names as column titles at the top
            for col_idx, model in enumerate(unique_models):
                axes[0, col_idx].set_title(
                    f"{entropy_features[0].replace('_', ' ').title()}\nModel: {model}",
                    fontsize=11,
                )
                if len(entropy_features) > 1:
                    axes[1, col_idx].set_title(
                        f"{entropy_features[1].replace('_', ' ').title()}\nModel: {model}",
                        fontsize=11,
                    )
        else:
            # Original behavior for non-dataset level or single model
            cols = min(2, num_features)
            rows = (num_features + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(16, 12 * rows / 2))
            if num_features == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)

            fig.suptitle(
                "Architecture Entropy Comparison",
                fontsize=16,
                fontweight="bold",
            )

            for idx, feature in enumerate(entropy_features):
                ax = axes[idx // cols, idx % cols]

                data_to_plot = []
                labels = []
                for arch in self.architectures:
                    arch_data = self.data[self.data["architecture"] == arch][
                        feature
                    ].dropna()
                    data_to_plot.append(arch_data.values)
                    labels.append(arch)

                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

                colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xlabel("Architecture Type", fontsize=12)
                ax.set_ylabel("Entropy Value", fontsize=12)
                ax.set_title(feature.replace("_", " ").title(), fontsize=13)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="x", rotation=45)

            for idx in range(num_features, rows * cols):
                fig.delaxes(axes[idx // cols, idx % cols])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "architecture_entropy_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'architecture_entropy_comparison.png'}")

    def plot_entropy_accuracy_correlation(self) -> None:
        """Generate scatter plots showing correlation between entropy and accuracy.

        Creates a 3x3 grid of scatter plots with regression lines showing
        the relationship between each entropy feature and accuracy.
        """
        print("Generating entropy-accuracy correlation plots...")

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # Set title based on analysis level and number of models
        if self.analysis_level == "dataset" and self.num_models > 1:
            fig.suptitle(
                f"Correlation Analysis Between Entropy Features and Accuracy ({self.num_models} Models)",
                fontsize=16,
                fontweight="bold",
            )
        else:
            fig.suptitle(
                "Correlation Analysis Between Entropy Features and Accuracy",
                fontsize=16,
                fontweight="bold",
            )

        # Define markers for architectures and colors for models to enable dual classification
        markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h"]
        max_models = max(
            (
                len(self.data["model_name"].unique())
                if "model_name" in self.data.columns
                else 1
            ),
            10,
        )
        colors = plt.cm.tab10(np.linspace(0, 1, max_models))

        for idx, feature in enumerate(entropy_features[:9]):
            ax = axes[idx // 3, idx % 3]

            x = self.data[feature].dropna()
            y = self.data.loc[x.index, "exp_accuracy"]

            # If we have model information and are at dataset level, use dual classification (model=color, architecture-marker)
            if (
                self.analysis_level == "dataset"
                and self.num_models > 1
                and "model_name" in self.data.columns
            ):
                model_data = self.data.loc[x.index].copy()
                model_data = model_data[model_data["exp_accuracy"].notna() & x.notna()]
                model_data = model_data[
                    ~np.isinf(model_data[feature])
                    & ~np.isinf(model_data["exp_accuracy"])
                ]

                if len(model_data) < 2:
                    ax.text(
                        0.5,
                        0.5,
                        "Insufficient Data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                # Create a color map for models and marker map for architectures
                unique_models = model_data["model_name"].unique()
                unique_archs = model_data["architecture"].unique()

                # Assign colors to models and markers to architectures
                model_colors = {}
                for i, model in enumerate(unique_models):
                    model_colors[model] = colors[i % len(colors)]

                arch_markers = {}
                for j, arch in enumerate(unique_archs):
                    arch_markers[arch] = markers[j % len(markers)]

                # Plot each combination of model and architecture
                for model in unique_models:
                    for arch in unique_archs:
                        subset = model_data[
                            (model_data["model_name"] == model)
                            & (model_data["architecture"] == arch)
                        ]
                        if len(subset) > 0:
                            ax.scatter(
                                subset[feature],
                                subset["exp_accuracy"],
                                alpha=0.7,
                                s=50,
                                label=f"{model}-{arch}",
                                color=model_colors[model],
                                marker=arch_markers[arch],
                            )

                ax.legend(
                    title="Model-Architecture",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    fontsize=8,
                )

                # Calculate overall correlation
                corr = model_data[feature].corr(model_data["exp_accuracy"])

                # Fit regression line to all data
                try:
                    # Validate data before fitting
                    x_vals = model_data[feature].values
                    y_vals = model_data["exp_accuracy"].values

                    # Check for finite values and remove non-finite entries
                    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if mask.sum() < 2:
                        continue

                    x_clean = x_vals[mask]
                    y_clean = y_vals[mask]

                    # Check if x values have sufficient variance
                    if np.var(x_clean) == 0:
                        continue

                    # Perform polynomial fitting
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    ax.plot(
                        x_clean,
                        p(x_clean),
                        "r--",
                        alpha=0.8,
                        linewidth=2,
                        label="Overall Trend",
                    )
                except Exception:
                    pass
            else:
                # Original behavior for single model or non-dataset level
                valid_idx = ~(x.isna() | y.isna() | np.isinf(x) | np.isinf(y))
                x = x[valid_idx]
                y = y[valid_idx]

                if len(x) < 2:
                    ax.text(
                        0.5,
                        0.5,
                        "Insufficient Data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                if x.std() == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "Zero Variance",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                corr = x.corr(y)

                ax.scatter(x, y, alpha=0.5, s=20)

                try:
                    # Validate data before fitting
                    x_vals = x.values
                    y_vals = y.values

                    # Check for finite values and remove non-finite entries
                    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                    if mask.sum() < 2:
                        continue

                    x_clean = x_vals[mask]
                    y_clean = y_vals[mask]

                    # Check if x values have sufficient variance
                    if np.var(x_clean) == 0:
                        continue

                    # Perform polynomial fitting
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    ax.plot(x_clean, p(x_clean), "r--", alpha=0.8, linewidth=2)
                except Exception:
                    pass

            ax.set_xlabel(feature.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.set_title(f"Correlation Coefficient: {corr:.3f}", fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "entropy_accuracy_correlation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'entropy_accuracy_correlation.png'}")

    def plot_round_entropy_evolution(self) -> None:
        """Generate plots showing entropy evolution across processing rounds.

        Creates a 2x2 grid of plots showing how entropy values change
        throughout multi-round processing, including comparisons between
        correct and incorrect samples.
        """
        print("Generating round entropy evolution plots...")

        if self.analysis_level == "dataset" and self.num_models > 1:
            round_data = (
                self.data.groupby(
                    ["model_name", "sample_id", "agent_round_number", "architecture"]
                )
                .agg(
                    {
                        "round_total_entropy": "first",
                        "round_infer_avg_entropy": "first",
                        "round_total_time": "first",
                        "round_total_token": "first",
                        "is_finally_correct": "first",
                    }
                )
                .reset_index()
            )
        else:
            round_data = (
                self.data.groupby(["sample_id", "agent_round_number", "architecture"])
                .agg(
                    {
                        "round_total_entropy": "first",
                        "round_infer_avg_entropy": "first",
                        "round_total_time": "first",
                        "round_total_token": "first",
                        "is_finally_correct": "first",
                    }
                )
                .reset_index()
            )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        if self.analysis_level == "dataset" and self.num_models > 1:
            fig.suptitle(
                f"Dataset-Level Round Entropy Evolution ({self.num_models} Models)",
                fontsize=16,
                fontweight="bold",
            )
        else:
            fig.suptitle(
                "Round Entropy Evolution Patterns",
                fontsize=16,
                fontweight="bold",
            )

        if self.analysis_level == "dataset" and self.num_models > 1:
            round_stats = round_data.groupby("agent_round_number").agg(
                {
                    "round_total_entropy": ["mean", "std"],
                    "round_infer_avg_entropy": ["mean", "std"],
                }
            )

            ax = axes[0, 0]
            ax.errorbar(
                round_stats.index,
                round_stats[("round_total_entropy", "mean")],
                yerr=round_stats[("round_total_entropy", "std")],
                marker="o",
                linewidth=2,
                markersize=8,
                capsize=5,
            )
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Total Entropy", fontsize=12)
            ax.set_title("Total Entropy Change Across Rounds", fontsize=13)
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.errorbar(
                round_stats.index,
                round_stats[("round_infer_avg_entropy", "mean")],
                yerr=round_stats[("round_infer_avg_entropy", "std")],
                marker="s",
                linewidth=2,
                markersize=8,
                capsize=5,
                color="orange",
            )
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Average Entropy", fontsize=12)
            ax.set_title("Average Entropy Change Across Rounds", fontsize=13)
            ax.grid(True, alpha=0.3)

            correct_data = round_data[round_data["is_finally_correct"] == True]
            incorrect_data = round_data[round_data["is_finally_correct"] == False]

            correct_stats = correct_data.groupby("agent_round_number")[
                "round_infer_avg_entropy"
            ].mean()
            incorrect_stats = incorrect_data.groupby("agent_round_number")[
                "round_infer_avg_entropy"
            ].mean()

            ax = axes[1, 0]
            ax.plot(
                correct_stats.index,
                correct_stats.values,
                marker="o",
                linewidth=2,
                markersize=8,
                label="Correct Samples",
                color="green",
            )
            ax.plot(
                incorrect_stats.index,
                incorrect_stats.values,
                marker="s",
                linewidth=2,
                markersize=8,
                label="Incorrect Samples",
                color="red",
            )
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Average Entropy", fontsize=12)
            ax.set_title("Entropy Evolution: Correct vs Incorrect", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            if "model_name" in round_data.columns:
                for model in round_data["model_name"].unique():
                    model_round_data = round_data[round_data["model_name"] == model]
                    model_stats = model_round_data.groupby("agent_round_number")[
                        "round_infer_avg_entropy"
                    ].mean()
                    ax.plot(
                        model_stats.index,
                        model_stats.values,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=model,
                    )
                ax.set_xlabel("Round", fontsize=12)
                ax.set_ylabel("Average Entropy", fontsize=12)
                ax.set_title("Round Entropy Evolution by Model", fontsize=13)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                for arch in self.architectures:
                    arch_data = round_data[round_data["architecture"] == arch]
                    arch_stats = arch_data.groupby("agent_round_number")[
                        "round_infer_avg_entropy"
                    ].mean()
                    ax.plot(
                        arch_stats.index,
                        arch_stats.values,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=arch,
                    )
                ax.set_xlabel("Round", fontsize=12)
                ax.set_ylabel("Average Entropy", fontsize=12)
                ax.set_title("Entropy Evolution Across Architectures", fontsize=13)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
        else:
            round_stats = round_data.groupby("agent_round_number").agg(
                {
                    "round_total_entropy": ["mean", "std"],
                    "round_infer_avg_entropy": ["mean", "std"],
                }
            )

            ax = axes[0, 0]
            ax.errorbar(
                round_stats.index,
                round_stats[("round_total_entropy", "mean")],
                yerr=round_stats[("round_total_entropy", "std")],
                marker="o",
                linewidth=2,
                markersize=8,
                capsize=5,
            )
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Total Entropy", fontsize=12)
            ax.set_title("Total Entropy Change Across Rounds", fontsize=13)
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.errorbar(
                round_stats.index,
                round_stats[("round_infer_avg_entropy", "mean")],
                yerr=round_stats[("round_infer_avg_entropy", "std")],
                marker="s",
                linewidth=2,
                markersize=8,
                capsize=5,
                color="orange",
            )
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Average Entropy", fontsize=12)
            ax.set_title("Average Entropy Change Across Rounds", fontsize=13)
            ax.grid(True, alpha=0.3)

            correct_data = round_data[round_data["is_finally_correct"] == True]
            incorrect_data = round_data[round_data["is_finally_correct"] == False]

            correct_stats = correct_data.groupby("agent_round_number")[
                "round_infer_avg_entropy"
            ].mean()
            incorrect_stats = incorrect_data.groupby("agent_round_number")[
                "round_infer_avg_entropy"
            ].mean()

            ax = axes[1, 0]
            ax.plot(
                correct_stats.index,
                correct_stats.values,
                marker="o",
                linewidth=2,
                markersize=8,
                label="Correct Samples",
                color="green",
            )
            ax.plot(
                incorrect_stats.index,
                incorrect_stats.values,
                marker="s",
                linewidth=2,
                markersize=8,
                label="Incorrect Samples",
                color="red",
            )
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Average Entropy", fontsize=12)
            ax.set_title(
                "Entropy Evolution Comparison: Correct vs Incorrect", fontsize=13
            )
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            for arch in self.architectures:
                arch_data = round_data[round_data["architecture"] == arch]
                arch_stats = arch_data.groupby("agent_round_number")[
                    "round_infer_avg_entropy"
                ].mean()
                ax.plot(
                    arch_stats.index,
                    arch_stats.values,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=arch,
                )
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Average Entropy", fontsize=12)
            ax.set_title(
                "Entropy Evolution Comparison Across Architectures", fontsize=13
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "round_entropy_evolution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'round_entropy_evolution.png'}")

    def plot_collaboration_comparison(self) -> None:
        """Generate comparison plots between multi-agent and single-agent systems.

        Creates a 2x2 grid of plots comparing performance metrics between
        multi-agent architectures and single-agent systems.
        """
        print("Generating collaboration pattern comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        if self.analysis_level == "dataset" and self.num_models > 1:
            fig.suptitle(
                f"Dataset-Level Multi-Agent vs Single-Agent Comparison ({self.num_models} Models)",
                fontsize=16,
                fontweight="bold",
            )
        else:
            fig.suptitle(
                "Multi-Agent vs Single-Agent System Comparison",
                fontsize=16,
                fontweight="bold",
            )

        multi_agent_data = self.data[
            self.data["architecture"].isin(MULTI_AGENT_ARCHITECTURES)
        ]
        single_agent_data = self.data[
            self.data["architecture"].isin(SINGLE_AGENT_ARCHITECTURES)
        ]

        ax = axes[0, 0]
        # If we have multiple models at dataset level, compare multi-agent vs single-agent for each model
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Create comparison by model
            comparison_data = []
            for model in self.data["model_name"].unique():
                model_data = self.data[self.data["model_name"] == model]
                model_multi = model_data[
                    model_data["architecture"].isin(MULTI_AGENT_ARCHITECTURES)
                ]
                model_single = model_data[
                    model_data["architecture"].isin(SINGLE_AGENT_ARCHITECTURES)
                ]

                if len(model_multi) > 0 and len(model_single) > 0:
                    comparison_data.append(
                        {
                            "Model": model,
                            "Multi-Agent": model_multi["exp_accuracy"].mean(),
                            "Single-Agent": model_single["exp_accuracy"].mean(),
                        }
                    )

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.set_index("Model", inplace=True)
                comparison_df.plot(kind="bar", ax=ax, alpha=0.7, width=0.8)
            else:
                # Fallback to overall comparison if no model-specific data
                accuracy_comparison = pd.DataFrame(
                    {
                        "Multi-Agent": [multi_agent_data["exp_accuracy"].mean()],
                        "Single-Agent": [single_agent_data["exp_accuracy"].mean()],
                    }
                )
                accuracy_comparison.plot(
                    kind="bar", ax=ax, color=["#4ECDC4", "#FF6B6B"], alpha=0.7
                )
        else:
            accuracy_comparison = pd.DataFrame(
                {
                    "Multi-Agent": [multi_agent_data["exp_accuracy"].mean()],
                    "Single-Agent": [single_agent_data["exp_accuracy"].mean()],
                }
            )
            accuracy_comparison.plot(
                kind="bar", ax=ax, color=["#4ECDC4", "#FF6B6B"], alpha=0.7
            )
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            "Accuracy Comparison: Multi-Agent vs Single-Agent by Model", fontsize=13
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[0, 1]
        # Similar approach for entropy comparison
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Create entropy comparison by model
            entropy_comparison_data = []
            for model in self.data["model_name"].unique():
                model_data = self.data[self.data["model_name"] == model]
                model_multi = model_data[
                    model_data["architecture"].isin(MULTI_AGENT_ARCHITECTURES)
                ]
                model_single = model_data[
                    model_data["architecture"].isin(SINGLE_AGENT_ARCHITECTURES)
                ]

                if len(model_multi) > 0 and len(model_single) > 0:
                    entropy_comparison_data.append(
                        {
                            "Model": model,
                            "Multi-Agent": model_multi["sample_mean_entropy"].mean(),
                            "Single-Agent": model_single["sample_mean_entropy"].mean(),
                        }
                    )

            if entropy_comparison_data:
                entropy_comparison_df = pd.DataFrame(entropy_comparison_data)
                entropy_comparison_df.set_index("Model", inplace=True)
                entropy_comparison_df.plot(kind="bar", ax=ax, alpha=0.7, width=0.8)
            else:
                # Fallback to overall comparison if no model-specific data
                entropy_comparison = pd.DataFrame(
                    {
                        "Multi-Agent": [multi_agent_data["sample_mean_entropy"].mean()],
                        "Single-Agent": [
                            single_agent_data["sample_mean_entropy"].mean()
                        ],
                    }
                )
                entropy_comparison.plot(
                    kind="bar", ax=ax, color=["#4ECDC4", "#FF6B6B"], alpha=0.7
                )
        else:
            entropy_comparison = pd.DataFrame(
                {
                    "Multi-Agent": [multi_agent_data["sample_mean_entropy"].mean()],
                    "Single-Agent": [single_agent_data["sample_mean_entropy"].mean()],
                }
            )
            entropy_comparison.plot(
                kind="bar", ax=ax, color=["#4ECDC4", "#FF6B6B"], alpha=0.7
            )
        ax.set_ylabel("Average Entropy", fontsize=12)
        ax.set_title(
            "Entropy Comparison: Multi-Agent vs Single-Agent by Model", fontsize=13
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[1, 0]
        # Architecture accuracy ranking with model differentiation if at dataset level
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Group by both architecture and model to see how each model performs on each architecture
            arch_model_accuracy = (
                self.data.groupby(["model_name", "architecture"])["exp_accuracy"]
                .mean()
                .unstack(level=1, fill_value=0)
            )
            arch_model_accuracy.plot(
                kind="bar",
                ax=ax,
                color=plt.cm.Set3(range(len(arch_model_accuracy.columns))),
            )
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Architecture Accuracy by Model", fontsize=13)
            ax.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            arch_accuracy = (
                self.data.groupby("architecture")["exp_accuracy"]
                .mean()
                .sort_values(ascending=False)
            )
            arch_accuracy.plot(
                kind="bar", ax=ax, color=plt.cm.Set3(range(len(arch_accuracy)))
            )
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Architecture Accuracy Ranking", fontsize=13)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 1]
        # Architecture entropy ranking with model differentiation if at dataset level
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Group by both architecture and model to see how each model performs on each architecture
            arch_model_entropy = (
                self.data.groupby(["model_name", "architecture"])["sample_mean_entropy"]
                .mean()
                .unstack(level=1, fill_value=0)
            )
            arch_model_entropy.plot(
                kind="bar",
                ax=ax,
                color=plt.cm.Set3(range(len(arch_model_entropy.columns))),
            )
            ax.set_ylabel("Average Entropy", fontsize=12)
            ax.set_title("Architecture Entropy by Model", fontsize=13)
            ax.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")
        else:
            arch_entropy = (
                self.data.groupby("architecture")["sample_mean_entropy"]
                .mean()
                .sort_values(ascending=False)
            )
            arch_entropy.plot(
                kind="bar", ax=ax, color=plt.cm.Set3(range(len(arch_entropy)))
            )
            ax.set_ylabel("Average Entropy", fontsize=12)
            ax.set_title("Architecture Entropy Ranking", fontsize=13)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "collaboration_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'collaboration_comparison.png'}")

    def plot_entropy_heatmap(self) -> None:
        """Generate a heatmap showing correlations between entropy features.

        Creates a correlation matrix heatmap displaying the relationships
        between all entropy features and accuracy in the dataset.
        """
        print("Generating entropy feature heatmap...")

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]

        features_to_correlate = entropy_features + ["exp_accuracy"]

        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Create separate heatmaps for each model
            unique_models = self.data["model_name"].unique()
            num_models = len(unique_models)

            # Calculate the layout dimensions for subplots
            cols = min(3, num_models)  # Maximum 3 columns
            rows = (num_models + cols - 1) // cols  # Calculate required rows

            fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 8 * rows))

            if num_models == 1:
                axes = [axes]
            elif num_models == 2:
                axes = axes.flatten()
            else:
                axes = (
                    axes.flatten()
                    if hasattr(axes, "flatten")
                    else [axes] if not isinstance(axes, (list, np.ndarray)) else axes
                )

            fig.suptitle(
                f"Dataset-Level Entropy Feature Correlation Heatmaps by Model ({self.num_models} Models)",
                fontsize=16,
                fontweight="bold",
            )

            # Determine the global correlation range for consistent color scaling
            all_corr_matrices = []
            for model in unique_models:
                model_data = self.data[self.data["model_name"] == model]
                model_corr_matrix = model_data[features_to_correlate].corr()
                all_corr_matrices.append(model_corr_matrix)

            # Get global min and max for consistent color scale
            vmin = min([m.min().min() for m in all_corr_matrices])
            vmax = max([m.max().max() for m in all_corr_matrices])

            for idx, model in enumerate(unique_models):
                model_data = self.data[self.data["model_name"] == model]
                correlation_matrix = model_data[features_to_correlate].corr()

                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    ax=axes[idx],
                    vmin=vmin,
                    vmax=vmax,
                    annot_kws={"fontsize": 6},
                    mask=mask,
                )

                axes[idx].set_title(
                    f"{model} - Correlation Heatmap", fontsize=10, fontweight="bold"
                )
                axes[idx].set_xticklabels(
                    axes[idx].get_xticklabels(), rotation=45, ha="right", fontsize=7
                )
                axes[idx].set_yticklabels(
                    axes[idx].get_yticklabels(), rotation=0, fontsize=7
                )

            # Hide any unused subplots
            for idx in range(len(unique_models), len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "entropy_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            print(f"Saved: {self.output_dir / 'entropy_heatmap.png'}")
        else:
            # Original behavior for non-dataset level or single model
            correlation_matrix = self.data[features_to_correlate].corr()

            fig, ax = plt.subplots(figsize=(18, 15))

            ax.set_title(
                "Entropy Feature Correlation Heatmap",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax,
                annot_kws={"fontsize": 8},
                mask=mask,
            )

            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", fontsize=8
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "entropy_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            print(f"Saved: {self.output_dir / 'entropy_heatmap.png'}")

    def plot_accuracy_entropy_scatter(self) -> None:
        """Generate scatter plots showing accuracy vs entropy relationships.

        Creates a 2x2 grid of scatter plots displaying the relationship
        between accuracy and various entropy metrics.
        """
        print("Generating accuracy-entropy scatter plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        if self.analysis_level == "dataset" and self.num_models > 1:
            fig.suptitle(
                f"Dataset-Level Accuracy vs Entropy Relationships ({self.num_models} Models)",
                fontsize=16,
                fontweight="bold",
            )
        else:
            fig.suptitle(
                "Relationship Between Accuracy and Entropy Metrics",
                fontsize=16,
                fontweight="bold",
            )

        # Define markers for architectures and colors for models to enable dual classification
        markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h"]
        colors = plt.cm.tab10(
            np.linspace(
                0,
                1,
                max(
                    (
                        len(self.data["model_name"].unique())
                        if "model_name" in self.data.columns
                        else 1
                    ),
                    10,
                ),
            )
        )

        ax = axes[0, 0]
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Dual classification: model (color) and architecture (marker)
            legend_elements = []
            for i, model in enumerate(self.data["model_name"].unique()):
                model_data = self.data[self.data["model_name"] == model]
                for j, arch in enumerate(self.data["architecture"].unique()):
                    arch_data = model_data[model_data["architecture"] == arch]
                    if len(arch_data) > 0:
                        marker = markers[j % len(markers)]
                        color = colors[i % len(colors)]
                        scatter = ax.scatter(
                            arch_data["sample_mean_entropy"],
                            arch_data["exp_accuracy"],
                            alpha=0.7,
                            s=50,
                            label=f"{model}-{arch}",
                            c=[color],
                            marker=marker,
                        )

            ax.set_xlabel("Sample Mean Entropy", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(
                "Sample Mean Entropy vs Accuracy (Model & Architecture)", fontsize=13
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            for arch in self.architectures:
                arch_data = self.data[self.data["architecture"] == arch]
                ax.scatter(
                    arch_data["sample_mean_entropy"],
                    arch_data["exp_accuracy"],
                    alpha=0.5,
                    s=30,
                    label=arch,
                )
            ax.set_xlabel("Sample Mean Entropy", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Sample Mean Entropy vs Accuracy", fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Dual classification: model (color) and architecture (marker)
            for i, model in enumerate(self.data["model_name"].unique()):
                model_data = self.data[self.data["model_name"] == model]
                for j, arch in enumerate(self.data["architecture"].unique()):
                    arch_data = model_data[model_data["architecture"] == arch]
                    if len(arch_data) > 0:
                        marker = markers[j % len(markers)]
                        color = colors[i % len(colors)]
                        ax.scatter(
                            arch_data["sample_std_entropy"],
                            arch_data["exp_accuracy"],
                            alpha=0.7,
                            s=50,
                            label=f"{model}-{arch}",
                            c=[color],
                            marker=marker,
                        )
            ax.set_xlabel("Sample Entropy Std Dev", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(
                "Sample Entropy Std Dev vs Accuracy (Model & Architecture)", fontsize=13
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            for arch in self.architectures:
                arch_data = self.data[self.data["architecture"] == arch]
                ax.scatter(
                    arch_data["sample_std_entropy"],
                    arch_data["exp_accuracy"],
                    alpha=0.5,
                    s=30,
                    label=arch,
                )
            ax.set_xlabel("Sample Entropy Std Dev", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Sample Entropy Std Dev vs Accuracy", fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Dual classification: model (color) and architecture (marker)
            for i, model in enumerate(self.data["model_name"].unique()):
                model_data = self.data[self.data["model_name"] == model]
                for j, arch in enumerate(self.data["architecture"].unique()):
                    arch_data = model_data[model_data["architecture"] == arch]
                    if len(arch_data) > 0:
                        marker = markers[j % len(markers)]
                        color = colors[i % len(colors)]
                        ax.scatter(
                            arch_data["sample_max_entropy"],
                            arch_data["exp_accuracy"],
                            alpha=0.7,
                            s=50,
                            label=f"{model}-{arch}",
                            c=[color],
                            marker=marker,
                        )
            ax.set_xlabel("Sample Max Entropy", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(
                "Sample Max Entropy vs Accuracy (Model & Architecture)", fontsize=13
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            for arch in self.architectures:
                arch_data = self.data[self.data["architecture"] == arch]
                ax.scatter(
                    arch_data["sample_max_entropy"],
                    arch_data["exp_accuracy"],
                    alpha=0.5,
                    s=30,
                    label=arch,
                )
            ax.set_xlabel("Sample Max Entropy", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Sample Max Entropy vs Accuracy", fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if (
            self.analysis_level == "dataset"
            and self.num_models > 1
            and "model_name" in self.data.columns
        ):
            # Dual classification: model (color) and architecture (marker)
            for i, model in enumerate(self.data["model_name"].unique()):
                model_data = self.data[self.data["model_name"] == model]
                for j, arch in enumerate(self.data["architecture"].unique()):
                    arch_data = model_data[model_data["architecture"] == arch]
                    if len(arch_data) > 0:
                        marker = markers[j % len(markers)]
                        color = colors[i % len(colors)]
                        ax.scatter(
                            arch_data["sample_all_agents_token_count"],
                            arch_data["exp_accuracy"],
                            alpha=0.7,
                            s=50,
                            label=f"{model}-{arch}",
                            c=[color],
                            marker=marker,
                        )
            ax.set_xlabel("Total Token Count", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(
                "Total Token Count vs Accuracy (Model & Architecture)", fontsize=13
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            for arch in self.architectures:
                arch_data = self.data[self.data["architecture"] == arch]
                ax.scatter(
                    arch_data["sample_all_agents_token_count"],
                    arch_data["exp_accuracy"],
                    alpha=0.5,
                    s=30,
                    label=arch,
                )
            ax.set_xlabel("Total Token Count", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("Total Token Count vs Accuracy", fontsize=13)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "accuracy_entropy_scatter.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'accuracy_entropy_scatter.png'}")

    def plot_distribution_comparison(self) -> None:
        """Generate distribution plots comparing entropy features across architectures.

        Creates a 2x3 grid of histogram plots showing the distribution
        of various entropy features for different architecture types.
        """
        print("Generating distribution comparison plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        if self.analysis_level == "dataset" and self.num_models > 1:
            fig.suptitle(
                f"Dataset-Level Entropy Distribution Comparison ({self.num_models} Models)",
                fontsize=16,
                fontweight="bold",
            )
        else:
            fig.suptitle(
                "Entropy Distribution Comparison Across Architectures",
                fontsize=16,
                fontweight="bold",
            )

        features = [
            "sample_mean_entropy",
            "sample_std_entropy",
            "sample_max_entropy",
            "sample_min_entropy",
            "sample_median_entropy",
            "sample_q3_entropy",
        ]

        for idx, feature in enumerate(features):
            ax = axes[idx // 3, idx % 3]

            if (
                self.analysis_level == "dataset"
                and self.num_models > 1
                and "model_name" in self.data.columns
            ):
                # Implement dual classification visual encoding: different colors for models and architectures
                unique_models = self.data["model_name"].unique()
                unique_archs = self.data["architecture"].unique()

                # Create a color map for each unique combination of model and architecture
                from itertools import product

                combinations = list(product(unique_models, unique_archs))

                # Use a colormap to generate distinct colors
                total_combinations = len(combinations)
                colors = plt.cm.tab20(np.linspace(0, 1, total_combinations))

                # Create legend labels and color mapping
                color_map = {}
                for i, (model, arch) in enumerate(combinations):
                    color_map[(model, arch)] = colors[i]

                # Plot each combination with its unique color
                legend_elements = []
                for model in unique_models:
                    for arch in unique_archs:
                        if (model, arch) in color_map:
                            model_arch_data = self.data[
                                (self.data["model_name"] == model)
                                & (self.data["architecture"] == arch)
                            ][feature].dropna()

                            if len(model_arch_data) > 0:
                                ax.hist(
                                    model_arch_data,
                                    bins=30,
                                    alpha=0.6,
                                    label=f"{model}-{arch}",
                                    color=color_map[(model, arch)],
                                    density=True,
                                )

                ax.set_xlabel(feature.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.set_title(feature.replace("_", " ").title(), fontsize=11)
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                for arch in self.architectures:
                    arch_data = self.data[self.data["architecture"] == arch][
                        feature
                    ].dropna()
                    ax.hist(arch_data, bins=30, alpha=0.5, label=arch, density=True)

                ax.set_xlabel(feature.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.set_title(feature.replace("_", " ").title(), fontsize=11)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "distribution_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'distribution_comparison.png'}")

    def plot_cross_model_architecture_comparison(self) -> None:
        """Generate bar charts comparing architecture performance across different models.

        Creates side-by-side bar charts showing how each architecture performs
        across different models for accuracy and entropy metrics.
        """
        print("Generating cross-model architecture comparison...")

        if "model_name" not in self.data.columns:
            print(
                "Cross-model architecture comparison skipped: No model data available"
            )
            return

        # Group by model and architecture to get average metrics
        model_arch_data = (
            self.data.groupby(["model_name", "architecture"])
            .agg(
                {
                    "exp_accuracy": "mean",
                    "sample_mean_entropy": "mean",
                    "sample_std_entropy": "mean",
                }
            )
            .reset_index()
        )

        unique_models = model_arch_data["model_name"].unique()
        unique_architectures = model_arch_data["architecture"].unique()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Cross-Model Architecture Performance Comparison ({self.num_models} Models)",
            fontsize=16,
            fontweight="bold",
        )

        # Accuracy comparison
        ax = axes[0, 0]
        pivot_acc = model_arch_data.pivot(
            index="architecture", columns="model_name", values="exp_accuracy"
        )
        pivot_acc.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Accuracy by Architecture and Model")
        ax.set_ylabel("Accuracy")
        ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # Mean entropy comparison
        ax = axes[0, 1]
        pivot_ent = model_arch_data.pivot(
            index="architecture", columns="model_name", values="sample_mean_entropy"
        )
        pivot_ent.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Mean Entropy by Architecture and Model")
        ax.set_ylabel("Mean Entropy")
        ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # Std entropy comparison
        ax = axes[1, 0]
        pivot_std = model_arch_data.pivot(
            index="architecture", columns="model_name", values="sample_std_entropy"
        )
        pivot_std.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("Std Entropy by Architecture and Model")
        ax.set_ylabel("Std Entropy")
        ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # Architecture comparison stacked by model
        ax = axes[1, 1]
        acc_by_arch = (
            self.data.groupby(["architecture", "model_name"])["exp_accuracy"]
            .mean()
            .unstack(level=1, fill_value=0)
        )
        acc_by_arch.plot(kind="bar", ax=ax, stacked=False)
        ax.set_title("Architecture Performance Stacked by Model")
        ax.set_ylabel("Average Accuracy")
        ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "cross_model_architecture_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'cross_model_architecture_comparison.png'}")

    def plot_base_model_comparison(self) -> None:
        """Generate plots comparing multi-agent system performance with base model performance.

        Creates visualizations showing how multi-agent systems compare to base models
        in terms of accuracy and other metrics.
        """
        print("Generating base model comparison plots...")

        # Check if base model columns exist in the data
        base_model_cols = [
            "base_model_accuracy",
            "base_model_format_compliance_rate",
            "base_model_is_finally_correct",
        ]

        available_base_cols = [
            col for col in base_model_cols if col in self.data.columns
        ]

        if not available_base_cols:
            print(
                "No base model columns found in data. Skipping base model comparison."
            )
            return

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"Multi-Agent vs Base Model Performance Comparison ({self.num_models} Models)",
            fontsize=16,
            fontweight="bold",
        )

        # Accuracy comparison
        ax = axes[0, 0]
        if (
            "base_model_accuracy" in self.data.columns
            and "exp_accuracy" in self.data.columns
        ):
            # For dataset level, compare model averages
            if self.analysis_level == "dataset" and "model_name" in self.data.columns:
                model_comparison = (
                    self.data.groupby("model_name")
                    .agg({"exp_accuracy": "mean", "base_model_accuracy": "mean"})
                    .reset_index()
                )

                x = np.arange(len(model_comparison))
                width = 0.35

                ax.bar(
                    x - width / 2,
                    model_comparison["exp_accuracy"],
                    width,
                    label="Multi-Agent Accuracy",
                    alpha=0.8,
                )
                ax.bar(
                    x + width / 2,
                    model_comparison["base_model_accuracy"],
                    width,
                    label="Base Model Accuracy",
                    alpha=0.8,
                )

                ax.set_xlabel("Model")
                ax.set_ylabel("Accuracy")
                ax.set_title("Accuracy: Multi-Agent vs Base Model by Model")
                ax.set_xticks(x)
                ax.set_xticklabels(model_comparison["model_name"], rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")
            else:
                # For model level, create a scatter plot
                ax.scatter(
                    self.data["base_model_accuracy"],
                    self.data["exp_accuracy"],
                    alpha=0.6,
                )
                # Add diagonal line for reference
                min_val = min(
                    self.data["base_model_accuracy"].min(),
                    self.data["exp_accuracy"].min(),
                )
                max_val = max(
                    self.data["base_model_accuracy"].max(),
                    self.data["exp_accuracy"].max(),
                )
                ax.plot(
                    [min_val, max_val],
                    [min_val, max_val],
                    "r--",
                    alpha=0.8,
                    label="Equal Performance",
                )
                ax.set_xlabel("Base Model Accuracy")
                ax.set_ylabel("Multi-Agent Accuracy")
                ax.set_title("Multi-Agent vs Base Model Accuracy")
                ax.legend()
                ax.grid(True, alpha=0.3)

        # Format compliance comparison
        ax = axes[0, 1]
        if "base_model_format_compliance_rate" in self.data.columns:
            if self.analysis_level == "dataset" and "model_name" in self.data.columns:
                model_comparison = (
                    self.data.groupby("model_name")
                    .agg(
                        {
                            "exp_format_compliance_rate": "mean",
                            "base_model_format_compliance_rate": "mean",
                        }
                    )
                    .reset_index()
                )

                x = np.arange(len(model_comparison))
                width = 0.35

                ax.bar(
                    x - width / 2,
                    model_comparison["exp_format_compliance_rate"],
                    width,
                    label="Multi-Agent Format Compliance",
                    alpha=0.8,
                )
                ax.bar(
                    x + width / 2,
                    model_comparison["base_model_format_compliance_rate"],
                    width,
                    label="Base Model Format Compliance",
                    alpha=0.8,
                )

                ax.set_xlabel("Model")
                ax.set_ylabel("Format Compliance Rate")
                ax.set_title("Format Compliance: Multi-Agent vs Base Model")
                ax.set_xticks(x)
                ax.set_xticklabels(model_comparison["model_name"], rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")

        # Performance improvement heatmap
        ax = axes[1, 0]
        if (
            "base_model_accuracy" in self.data.columns
            and "exp_accuracy" in self.data.columns
        ):
            # Calculate improvement over base model
            improvement_data = self.data.copy()
            improvement_data["accuracy_improvement"] = (
                improvement_data["exp_accuracy"]
                - improvement_data["base_model_accuracy"]
            )

            if "model_name" in self.data.columns:
                improvement_by_model = (
                    improvement_data.groupby("model_name")["accuracy_improvement"]
                    .mean()
                    .reset_index()
                )

                ax.bar(
                    improvement_by_model["model_name"],
                    improvement_by_model["accuracy_improvement"],
                )
                ax.set_xlabel("Model")
                ax.set_ylabel("Accuracy Improvement")
                ax.set_title("Accuracy Improvement Over Base Model by Model")
                ax.tick_params(axis="x", rotation=45)
                ax.grid(True, alpha=0.3, axis="y")

                # Add horizontal line at y=0 to show baseline
                ax.axhline(y=0, color="r", linestyle="--", alpha=0.7)

        # Architecture-level comparison
        ax = axes[1, 1]
        if (
            "base_model_accuracy" in self.data.columns
            and "exp_accuracy" in self.data.columns
            and "architecture" in self.data.columns
        ):
            arch_comparison = (
                self.data.groupby("architecture")
                .agg({"exp_accuracy": "mean", "base_model_accuracy": "mean"})
                .reset_index()
            )

            x = np.arange(len(arch_comparison))
            width = 0.35

            ax.bar(
                x - width / 2,
                arch_comparison["exp_accuracy"],
                width,
                label="Multi-Agent Accuracy",
                alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                arch_comparison["base_model_accuracy"],
                width,
                label="Base Model Accuracy",
                alpha=0.8,
            )

            ax.set_xlabel("Architecture")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy by Architecture: Multi-Agent vs Base Model")
            ax.set_xticks(x)
            ax.set_xticklabels(arch_comparison["architecture"], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "base_model_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'base_model_comparison.png'}")

    def generate_all_visualizations(self) -> None:
        """Generate all visualization plots for the entropy analysis.

        Executes all plotting methods to create a comprehensive set of
        visualizations for the analysis results.
        """
        print("Generating all visualization plots...")

        self.plot_architecture_entropy_comparison()
        self.plot_entropy_accuracy_correlation()
        self.plot_round_entropy_evolution()
        self.plot_collaboration_comparison()
        self.plot_entropy_heatmap()
        self.plot_accuracy_entropy_scatter()
        self.plot_distribution_comparison()
        self.plot_base_model_comparison()  # Add base model comparison

        # Additional hierarchical visualizations
        if self.analysis_level == "dataset" and "model_name" in self.data.columns:
            self.plot_cross_model_architecture_comparison()

        print(f"Visualization charts saved to: {self.output_dir}")
