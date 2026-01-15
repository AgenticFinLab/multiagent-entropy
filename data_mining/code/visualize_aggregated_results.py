"""Visualization script for aggregated experiment results.

This script generates comprehensive visualizations for each CSV file in the
results_aggregated directory, including:
1. Feature importance comparison
2. SHAP value analysis
3. Impact direction statistics
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["font.size"] = 10


class AggregatedResultsVisualizer:
    """Visualizer for aggregated experiment results."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        n_features: int = 20,
        feature_importance_from: str = "mean_importance_normalized",
    ):
        """Initializes the visualizer.

        Args:
            input_dir: Directory containing aggregated CSV files
            output_dir: Directory to save visualization images
            n_features: Number of top features to display
            feature_importance_from: Column name to use for feature importance ranking.
                Options: 'lightgbm_importance', 'xgboost_importance',
                        'mean_importance', 'mean_importance_normalized'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_features = n_features
        self.feature_importance_from = feature_importance_from

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_all_experiments(self):
        """Visualizes all CSV files in the input directory."""
        if not self.input_dir.exists():
            print(f"Error: Input directory does not exist: {self.input_dir}")
            return

        csv_files = list(self.input_dir.glob("*.csv"))

        if not csv_files:
            print(f"Warning: No CSV files found in {self.input_dir}")
            return

        print(f"Found {len(csv_files)} CSV files to visualize")

        successful = 0
        failed = 0

        for csv_file in sorted(csv_files):
            exp_name = csv_file.stem
            print(f"\n{'='*80}")
            print(f"Processing: {exp_name}")
            print(f"{'='*80}")

            try:
                self.visualize_experiment(csv_file, exp_name)
                successful += 1
                print(f"Successfully visualized: {exp_name}")
            except Exception as e:
                failed += 1
                print(f"Failed: {exp_name}")
                print(f"   Error message: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"Visualization completion statistics:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Total: {len(csv_files)}")
        print(f"{'='*80}\n")

    def visualize_experiment(self, csv_file: Path, exp_name: str):
        """Creates visualizations for a single experiment.

        Args:
            csv_file: Path to the CSV file
            exp_name: Name of the experiment
        """
        # Read data
        df = pd.read_csv(csv_file)
        print(f"   Loaded {len(df)} features")

        # Validate feature_importance_from column exists
        if self.feature_importance_from not in df.columns:
            print(
                f"   Warning: Column '{self.feature_importance_from}' not found. "
                f"Using 'mean_importance_normalized' instead."
            )
            self.feature_importance_from = "mean_importance_normalized"

        # Sort by feature importance and select top N features
        df_sorted = df.sort_values(
            by=self.feature_importance_from, ascending=False
        ).head(self.n_features)

        # Create figure with subplots
        fig = plt.figure(figsize=(8, 16))
        gs = fig.add_gridspec(3, 1, hspace=0.35, wspace=0.1)

        # 1. Feature Importance Comparison (Top subplot, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_feature_importance(ax1, df_sorted)

        # 2. SHAP Value Visualization
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_shap_values(ax2, df_sorted)

        # # 3. Impact Direction Distribution
        # ax4 = fig.add_subplot(gs[0, 1])
        # self._plot_impact_direction(ax4, df_sorted)

        # 4. SHAP Statistics - Positive vs Negative Ratios
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_shap_ratios(ax5, df_sorted)

        # # 5. Feature Importance Distribution
        # ax6 = fig.add_subplot(gs[2, 0])
        # self._plot_importance_distribution(ax6, df)

        # Add main title
        fig.suptitle(
            f"Experiment Analysis: {exp_name}\n(Top {self.n_features} features by {self.feature_importance_from})",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        # Save figure
        output_file = self.output_dir / f"{exp_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved visualization to: {output_file}")

    def _plot_feature_importance(self, ax, df: pd.DataFrame):
        """Plots feature importance comparison."""
        features = df["feature"].values
        lgb_imp = df["lightgbm_importance"].values
        xgb_imp = df["xgboost_importance"].values
        mean_imp_norm = df[self.feature_importance_from].values

        x = np.arange(len(features))
        width = 0.25

        # Normalize for better comparison
        lgb_norm = (lgb_imp - lgb_imp.min()) / (lgb_imp.max() - lgb_imp.min()) if lgb_imp.max() > lgb_imp.min() else lgb_imp * 0
        xgb_norm = (xgb_imp - xgb_imp.min()) / (xgb_imp.max() - xgb_imp.min()) if xgb_imp.max() > xgb_imp.min() else xgb_imp * 0

        ax.barh(x - width, lgb_norm, width, label="LightGBM (normalized)", alpha=0.8, color="#2E86AB")
        ax.barh(x, xgb_norm, width, label="XGBoost (normalized)", alpha=0.8, color="#A23B72")
        ax.barh(x + width, mean_imp_norm, width, label="Mean Normalized", alpha=0.8, color="#F18F01")

        ax.set_yticks(x)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Normalized Importance", fontweight="bold")
        ax.set_title("Feature Importance Comparison", fontweight="bold", fontsize=12)
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    def _plot_shap_values(self, ax, df: pd.DataFrame):
        """Plots SHAP values for top features."""
        features = df["feature"].values[:20]  # Top 20 for clarity
        mean_shap = df["mean_shap"].values[:20]

        colors = ["#2E8B57" if val > 0 else "#DC143C" for val in mean_shap]

        ax.barh(features, mean_shap, color=colors, alpha=0.7)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Mean SHAP Value", fontweight="bold")
        ax.set_title("SHAP Value Impact\n(Top 20 Features)", fontweight="bold", fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (feature, value) in enumerate(zip(features, mean_shap)):
            ax.text(
                value,
                i,
                f" {value:.3f}",
                va="center",
                ha="left" if value > 0 else "right",
                fontsize=8,
            )

    def _plot_impact_direction(self, ax, df: pd.DataFrame):
        """Plots impact direction distribution."""
        direction_counts = df["overall_impact_direction"].value_counts()

        colors = {"positive": "#2E8B57", "negative": "#DC143C", "mixed": "#FFA500"}
        plot_colors = [colors.get(dir, "#888888") for dir in direction_counts.index]

        wedges, texts, autotexts = ax.pie(
            direction_counts.values,
            labels=direction_counts.index,
            autopct="%1.1f%%",
            colors=plot_colors,
            startangle=90,
            textprops={"fontsize": 10, "weight": "bold"},
        )

        for autotext in autotexts:
            autotext.set_color("white")

        ax.set_title("Impact Direction Distribution", fontweight="bold", fontsize=12)

    def _plot_shap_ratios(self, ax, df: pd.DataFrame):
        """Plots positive vs negative SHAP ratios."""
        features = df["feature"].values[:20]
        pos_ratio = df["mean_positive_ratio"].values[:20]
        neg_ratio = df["mean_negative_ratio"].values[:20]

        x = np.arange(len(features))
        width = 0.35

        ax.barh(x - width / 2, pos_ratio, width, label="Positive Ratio", color="#2E8B57", alpha=0.8)
        ax.barh(x + width / 2, neg_ratio, width, label="Negative Ratio", color="#DC143C", alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Ratio", fontweight="bold")
        ax.set_title("SHAP Positive/Negative Ratios\n(Top 20 Features)", fontweight="bold", fontsize=12)
        ax.legend()
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    def _plot_importance_distribution(self, ax, df: pd.DataFrame):
        """Plots distribution of feature importance."""
        mean_imp_norm = df[self.feature_importance_from].values

        ax.hist(mean_imp_norm, bins=30, color="#6A4C93", alpha=0.7, edgecolor="black")
        ax.axvline(
            mean_imp_norm.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_imp_norm.mean():.3f}",
        )
        ax.axvline(
            np.median(mean_imp_norm),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(mean_imp_norm):.3f}",
        )

        ax.set_xlabel(f"{self.feature_importance_from}", fontweight="bold")
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_title("Feature Importance Distribution\n(All Features)", fontweight="bold", fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)


def main():
    """Main function."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Set paths
    input_dir = project_root / "results_aggregated"
    output_dir = project_root / "results_visualizations"

    # Configuration
    n_features = 20  # Number of top features to display
    feature_importance_from = "mean_importance_normalized"  # Column to use for ranking

    print("=" * 80)
    print("Aggregated Results Visualization Tool")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Top features to display: {n_features}")
    print(f"Feature importance ranking from: {feature_importance_from}")
    print("=" * 80)

    # Create visualizer and run
    visualizer = AggregatedResultsVisualizer(
        str(input_dir),
        str(output_dir),
        n_features=n_features,
        feature_importance_from=feature_importance_from,
    )
    visualizer.visualize_all_experiments()

    print("\nAll visualizations completed!")


if __name__ == "__main__":
    main()
