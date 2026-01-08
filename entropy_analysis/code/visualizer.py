"""Visualization module for entropy analysis results.

This module provides comprehensive visualization capabilities for displaying
entropy analysis results from multi-agent system performance evaluation. It
includes methods for creating various types of plots including box plots,
scatter plots, heatmaps, and comprehensive dashboards.
"""

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    """

    def __init__(self, data: pd.DataFrame, output_dir: str) -> None:
        """Initialize the EntropyVisualizer with data and output directory.

        Args:
            data: DataFrame containing preprocessed experimental data.
            output_dir: Directory path where visualizations will be saved.
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.architectures = ["centralized", "debate", "hybrid", "sequential", "single"]

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

        cols = min(2, num_features)
        rows = (num_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 12 * rows / 2))
        if num_features == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            "Entropy Feature Comparison Across Architectures",
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
        fig.suptitle(
            "Correlation Analysis Between Entropy Features and Accuracy",
            fontsize=16,
            fontweight="bold",
        )

        for idx, feature in enumerate(entropy_features[:9]):
            ax = axes[idx // 3, idx % 3]

            x = self.data[feature].dropna()
            y = self.data.loc[x.index, "exp_accuracy"]

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
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
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

        round_data = (
            self.data.groupby(["sample_id", "agent_round_number", "architecture"])
            .agg(
                {
                    "round_total_entropy": "first",
                    "round_avg_entropy": "first",
                    "is_finally_correct": "first",
                }
            )
            .reset_index()
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Entropy Evolution Patterns Across Processing Rounds",
            fontsize=16,
            fontweight="bold",
        )

        round_stats = round_data.groupby("agent_round_number").agg(
            {
                "round_total_entropy": ["mean", "std"],
                "round_avg_entropy": ["mean", "std"],
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
            round_stats[("round_avg_entropy", "mean")],
            yerr=round_stats[("round_avg_entropy", "std")],
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
            "round_avg_entropy"
        ].mean()
        incorrect_stats = incorrect_data.groupby("agent_round_number")[
            "round_avg_entropy"
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
        ax.set_title("Entropy Evolution Comparison: Correct vs Incorrect", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for arch in self.architectures:
            arch_data = round_data[round_data["architecture"] == arch]
            arch_stats = arch_data.groupby("agent_round_number")[
                "round_avg_entropy"
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
        ax.set_title("Entropy Evolution Comparison Across Architectures", fontsize=13)
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

        multi_agent_archs = ["centralized", "debate", "hybrid", "sequential"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Multi-Agent vs Single-Agent System Comparison",
            fontsize=16,
            fontweight="bold",
        )

        multi_agent_data = self.data[self.data["architecture"].isin(multi_agent_archs)]
        single_agent_data = self.data[self.data["architecture"] == "single"]

        ax = axes[0, 0]
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
        ax.set_title("Average Accuracy Comparison", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[0, 1]
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
        ax.set_title("Average Entropy Comparison", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 0]
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
        between all entropy features in the dataset.
        """
        print("Generating entropy feature heatmap...")

        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]

        correlation_matrix = self.data[entropy_features].corr()

        fig, ax = plt.subplots(figsize=(16, 14))

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
        )

        ax.set_title(
            "Entropy Feature Correlation Heatmap",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

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
        fig.suptitle(
            "Relationship Between Accuracy and Entropy Metrics",
            fontsize=16,
            fontweight="bold",
        )

        ax = axes[0, 0]
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

    def plot_summary_dashboard(self) -> None:
        """Generate a comprehensive dashboard summarizing key findings.

        Creates a multi-panel dashboard displaying architecture performance,
        entropy-accuracy correlations, scatter plots, evolution patterns,
        and summary statistics.
        """
        print("Generating summary dashboard...")

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        arch_accuracy = (
            self.data.groupby("architecture")["exp_accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        arch_accuracy.plot(
            kind="bar", ax=ax1, color=plt.cm.Set2(range(len(arch_accuracy)))
        )
        ax1.set_title("Architecture Accuracy", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Accuracy")
        ax1.tick_params(axis="x", rotation=45)

        ax2 = fig.add_subplot(gs[0, 1])
        arch_entropy = (
            self.data.groupby("architecture")["sample_mean_entropy"]
            .mean()
            .sort_values(ascending=False)
        )
        arch_entropy.plot(
            kind="bar", ax=ax2, color=plt.cm.Set2(range(len(arch_entropy)))
        )
        ax2.set_title("Architecture Mean Entropy", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Mean Entropy")
        ax2.tick_params(axis="x", rotation=45)

        ax3 = fig.add_subplot(gs[0, 2:])
        entropy_features = [
            col for col in self.data.columns if "entropy" in col.lower()
        ]
        correlations = [
            self.data[feature].corr(self.data["exp_accuracy"])
            for feature in entropy_features[:10]
        ]
        ax3.barh(
            range(len(correlations)),
            correlations,
            color=plt.cm.RdYlGn_r(np.array(correlations)),
        )
        ax3.set_yticks(range(len(correlations)))
        ax3.set_yticklabels(
            [f.replace("_", " ")[:20] for f in entropy_features[:10]], fontsize=8
        )
        ax3.set_xlabel("Correlation Coefficient")
        ax3.set_title("Entropy-Accuracy Correlations", fontsize=12, fontweight="bold")
        ax3.axvline(x=0, color="black", linestyle="--", linewidth=0.5)

        ax4 = fig.add_subplot(gs[1, :2])
        for arch in self.architectures:
            arch_data = self.data[self.data["architecture"] == arch]
            ax4.scatter(
                arch_data["sample_mean_entropy"],
                arch_data["exp_accuracy"],
                alpha=0.5,
                s=20,
                label=arch,
            )
        ax4.set_xlabel("Mean Entropy")
        ax4.set_ylabel("Accuracy")
        ax4.set_title(
            "Entropy vs Accuracy Scatter Plot", fontsize=12, fontweight="bold"
        )
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 2:])
        round_data = (
            self.data.groupby(["sample_id", "agent_round_number"])
            .agg({"round_avg_entropy": "first", "is_finally_correct": "first"})
            .reset_index()
        )
        round_stats = round_data.groupby("agent_round_number")[
            "round_avg_entropy"
        ].mean()
        ax5.plot(
            round_stats.index, round_stats.values, marker="o", linewidth=2, markersize=8
        )
        ax5.set_xlabel("Round")
        ax5.set_ylabel("Average Entropy")
        ax5.set_title("Round Entropy Evolution", fontsize=12, fontweight="bold")
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[2, :])
        summary_text = f"""
        Dataset Summary Statistics:
        - Total Samples: {self.data['sample_id'].nunique()}
        - Number of Experiments: {self.data['experiment_name'].nunique()}
        - Architecture Types: {', '.join(self.architectures)}
        - Average Accuracy: {self.data['exp_accuracy'].mean():.3f}
        - Average Entropy: {self.data['sample_mean_entropy'].mean():.3f}
        - Average Token Count: {self.data['sample_all_agents_token_count'].mean():.0f}
        - Highest Accuracy Architecture: {arch_accuracy.index[0]} ({arch_accuracy.iloc[0]:.3f})
        - Lowest Entropy Architecture: {arch_entropy.index[-1]} ({arch_entropy.iloc[-1]:.3f})
        """
        ax6.text(
            0.1,
            0.5,
            summary_text,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax6.axis("off")
        ax6.set_title("Analysis Summary", fontsize=14, fontweight="bold")

        fig.suptitle(
            "Multi-Agent System Entropy Analysis Summary Dashboard",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )
        plt.savefig(
            self.output_dir / "summary_dashboard.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Saved: {self.output_dir / 'summary_dashboard.png'}")

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
        self.plot_summary_dashboard()

        print("\nAll visualization plots generated successfully!")
        print(f"Plots saved in: {self.output_dir}")


if __name__ == "__main__":
    from data_loader import DataLoader

    data_path = (
        "/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated/"
        "aggregated_data.csv"
    )

    loader = DataLoader(data_path)
    processed_data = loader.preprocess_data()

    visualizer = EntropyVisualizer(
        processed_data,
        "/home/yuxuanzhao/multiagent-entropy/entropy_analysis/visualizations/",
    )

    visualizer.generate_all_visualizations()
