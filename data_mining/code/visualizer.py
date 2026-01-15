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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP importance plots will be skipped.")

warnings.filterwarnings("ignore")

# Set enhanced plotting style
sns.set_style("whitegrid", {"axes.spines.top": False, "axes.spines.right": False})


class AggregatedResultsVisualizer:
    """Visualizer for aggregated experiment results."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        n_features: int = 20,
        feature_importance_from: str = "mean_importance_normalized",
        shap_data_dir: str = None,
    ):
        """Initializes the visualizer.

        Args:
            input_dir: Directory containing aggregated CSV files
            output_dir: Directory to save visualization images
            n_features: Number of top features to display
            feature_importance_from: Column name to use for feature importance ranking.
                Options: 'lightgbm_importance', 'xgboost_importance',
                        'mean_importance', 'mean_importance_normalized'
            shap_data_dir: Base directory containing SHAP results for experiments
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.n_features = n_features
        self.feature_importance_from = feature_importance_from
        self.shap_data_dir = Path(shap_data_dir) if shap_data_dir else None

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store experiment summaries for final report
        self.experiment_summaries = []

    def _load_shap_data(self, exp_name: str, model_name: str, task_type: str):
        """Loads SHAP values and X_test data for a specific model.
        
        Args:
            exp_name: Name of the experiment
            model_name: Name of the model (e.g., 'LightGBM', 'XGBoost')
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Tuple of (shap_values_df, X_test_df) or (None, None) if files not found
        """
        if self.shap_data_dir is None:
            return None, None
        
        # Construct path to SHAP directory for this experiment
        shap_dir = self.shap_data_dir / exp_name / "shap"
        
        if not shap_dir.exists():
            print(f"   Warning: SHAP directory not found: {shap_dir}")
            return None, None
        
        # Load SHAP values CSV
        shap_csv_path = shap_dir / f"shap_values_{model_name}_{task_type}.csv"
        if not shap_csv_path.exists():
            print(f"   Warning: SHAP values file not found: {shap_csv_path}")
            return None, None
        
        # Load X_test CSV
        x_test_csv_path = shap_dir / f"X_test_{model_name}_{task_type}.csv"
        if not x_test_csv_path.exists():
            print(f"   Warning: X_test file not found: {x_test_csv_path}")
            return None, None
        
        try:
            shap_values_df = pd.read_csv(shap_csv_path, index_col='sample_index')
            X_test_df = pd.read_csv(x_test_csv_path, index_col=0)
            print(f"   Loaded SHAP data: {len(shap_values_df)} samples, {len(shap_values_df.columns)} features")
            return shap_values_df, X_test_df
        except Exception as e:
            print(f"   Error loading SHAP data: {str(e)}")
            return None, None

    def _plot_shap_importance(self, ax, shap_values_df, X_test_df, model_name: str):
        """Plots SHAP importance subplot similar to shap_analyzer.py."""
        if not SHAP_AVAILABLE:
            ax.text(0.5, 0.5, 'SHAP not available',
                   ha='center', va='center', fontsize=8)
            ax.set_title(f"SHAP Importance - {model_name}", fontweight="bold", fontsize=10)
            return
        
        if shap_values_df is None or X_test_df is None:
            ax.text(0.5, 0.5, 'SHAP data not available',
                   ha='center', va='center', fontsize=8)
            ax.set_title(f"SHAP Importance - {model_name}", fontweight="bold", fontsize=10)
            return
        
        try:
            shap_values = shap_values_df.values
            X_test_values = X_test_df.values
            feature_names = list(shap_values_df.columns)

            plt.sca(ax)
            shap.summary_plot(
                shap_values,
                X_test_values,
                feature_names=None,  # Hide feature names
                plot_type="dot",
                show=False,
                max_display=20,
                color=plt.get_cmap("coolwarm")
            )

            ax.set_title(f"SHAP Importance - {model_name}", fontweight="bold", fontsize=11, pad=15)
            ax.set_facecolor('#f8f9fa')
            
            # Adjust y-axis tick labels
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylabel("Features", fontsize=9, fontweight="bold")
            ax.set_xlabel("SHAP value (impact on model output)", fontsize=9, fontweight="bold")

        except Exception as e:
            print(f"   Error plotting SHAP importance for {model_name}: {str(e)}")
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {str(e)}',
                   ha='center', va='center', fontsize=8, wrap=True)
            ax.set_title(f"SHAP Importance - {model_name}", fontweight="bold", fontsize=12)

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
        
        # Generate summary file after processing all experiments
        self._generate_summary_file()

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
        
        # Try to load SHAP data for LightGBM and XGBoost
        shap_lgb, X_test_lgb = self._load_shap_data(exp_name, "LightGBM", "classification")
        shap_xgb, X_test_xgb = self._load_shap_data(exp_name, "XGBoost", "classification")

        # Create figure with subplots
        fig = plt.figure(figsize=(8, 12))
        gs = fig.add_gridspec(8, 2, hspace=4, wspace=0.6)

        # 1. Feature Importance Comparison
        ax1 = fig.add_subplot(gs[:4, 0])
        self._plot_feature_importance(ax1, df_sorted)

        # 2. SHAP Importance - LightGBM
        ax2 = fig.add_subplot(gs[4:8, 1])
        self._plot_shap_importance(ax2, shap_lgb, X_test_lgb, "LightGBM")

        # 3. SHAP Importance - XGBoost
        ax3 = fig.add_subplot(gs[:4, 1])
        self._plot_shap_importance(ax3, shap_xgb, X_test_xgb, "XGBoost")

        # 4. SHAP Value Impact (Mean SHAP)
        ax4 = fig.add_subplot(gs[4:8, 0])
        self._plot_shap_values(ax4, df_sorted)      

        # Add main title
        fig.suptitle(
            f"Experiment Analysis: {exp_name}\n(Top {self.n_features} features by {self.feature_importance_from})",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        # Save figure
        output_file = self.output_dir / f"{exp_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"   Saved visualization to: {output_file}")
        
        # Collect summary information for this experiment
        self._collect_experiment_summary(exp_name, df_sorted)

    def _plot_feature_importance(self, ax, df: pd.DataFrame):
        """Plots feature importance comparison."""
        features = df["feature"].values
        lgb_imp = df["lightgbm_importance"].values
        xgb_imp = df["xgboost_importance"].values
        mean_imp_norm = df[self.feature_importance_from].values

        # Normalize for better comparison
        lgb_norm = (lgb_imp - lgb_imp.min()) / (lgb_imp.max() - lgb_imp.min()) if lgb_imp.max() > lgb_imp.min() else lgb_imp * 0
        xgb_norm = (xgb_imp - xgb_imp.min()) / (xgb_imp.max() - xgb_imp.min()) if xgb_imp.max() > xgb_imp.min() else xgb_imp * 0

        x = np.arange(len(features))
        width = 0.25

        ax.barh(x - width, lgb_norm, width, label="LightGBM (normalized)", alpha=0.8, 
                color="#1f77b4", edgecolor='white', linewidth=0.8)
        ax.barh(x, xgb_norm, width, label="XGBoost (normalized)", alpha=0.8, 
                color="#ff7f0e", edgecolor='white', linewidth=0.8)
        ax.barh(x + width, mean_imp_norm, width, label="Mean Normalized", alpha=0.8, 
                color="#2ca02c", edgecolor='white', linewidth=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(features, fontsize=8, ha='right')
        ax.set_xlabel("Normalized Importance", fontweight="bold", fontsize=9)
        ax.set_title("Feature Importance Comparison", fontweight="bold", fontsize=11, pad=15)
        ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        # Automatically adjust y-axis labels, avoid overlap
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
        # Removed inner tight_layout call to avoid interfering with global layout


    def _plot_shap_values(self, ax, df: pd.DataFrame):
        """Plots SHAP values for top features."""
        features = df["feature"].values[:self.n_features]
        mean_shap = df["mean_shap"].values[:self.n_features]

        y_positions = np.arange(len(features))

        colors = ["#2E8B57" if val > 0 else "#DC143C" for val in mean_shap]

        bars = ax.barh(y_positions, mean_shap, color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Mean SHAP Value", fontweight="bold", fontsize=9)
        ax.set_title("SHAP Value Impact", fontweight="bold", fontsize=11, pad=15)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(features, fontsize=8, ha='right')
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        # Add value labels, avoid overlap
        for i, (feature, value) in enumerate(zip(features, mean_shap)):
            if abs(value) > 0.001:
                ax.text(
                    value + (0.01 if value >= 0 else -0.01),
                    y_positions[i],
                    f"{value:.3f}",
                    va="center",
                    ha="left" if value > 0 else "right",
                    fontsize=7,
                    weight='normal',
                )

    def _collect_experiment_summary(self, exp_name: str, df_sorted: pd.DataFrame):
        """Collects summary information for a single experiment.
        
        Args:
            exp_name: Name of the experiment
            df_sorted: DataFrame with top features sorted by importance
        """
        # Extract top features and their impact directions
        summary_data = []
        for _, row in df_sorted.iterrows():
            feature = row['feature']
            importance = row[self.feature_importance_from]
            impact_direction = row['overall_impact_direction']
            
            summary_data.append({
                'feature': feature,
                'importance': importance,
                'impact_direction': impact_direction
            })
        
        # Store experiment summary
        self.experiment_summaries.append({
            'experiment_name': exp_name,
            'features': summary_data
        })
    
    def _generate_summary_file(self):
        """Generates a summary.txt file with feature importance and impact directions."""
        if not self.experiment_summaries:
            print("No experiment summaries to generate report.")
            return
        
        summary_file_path = self.output_dir / "summary.txt"
        
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write("Experiment Feature Importance Summary\n")
            f.write("===================================\n\n")
            
            for exp_summary in self.experiment_summaries:
                exp_name = exp_summary['experiment_name']
                features = exp_summary['features']
                
                f.write(f"Experiment: {exp_name}\n")
                f.write(f"Top {self.n_features} Most Important Features:\n")
                f.write("-" * 40 + "\n")
                
                # Sort features by importance in descending order
                sorted_features = sorted(features, key=lambda x: x['importance'], reverse=True)
                
                for i, feature_data in enumerate(sorted_features, 1):
                    feature = feature_data['feature']
                    importance = feature_data['importance']
                    impact_direction = feature_data['impact_direction']
                    
                    # Determine impact direction description
                    direction_desc = {
                        'positive': 'Positive influence',
                        'negative': 'Negative influence',
                        'mixed': 'Mixed influence'
                    }.get(impact_direction, f'{impact_direction} influence')
                    
                    f.write(f"  {i:2d}. {feature:<30} (Importance: {importance:>8.4f}, {direction_desc})\n")
                
                f.write("\n")
        
        print(f"Summary file saved to: {summary_file_path}")


def main():
    """Main function."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Set paths
    input_dir = project_root / "results_aggregated"
    output_dir = project_root / "results_visualizations"
    shap_data_dir = project_root / "results"  # Base directory for SHAP data

    # Configuration
    n_features = 20  # Number of top features to display
    feature_importance_from = "mean_importance_normalized"  # Column to use for ranking

    print("=" * 80)
    print("Aggregated Results Visualization Tool")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"SHAP data directory: {shap_data_dir}")
    print(f"Top features to display: {n_features}")
    print(f"Feature importance ranking from: {feature_importance_from}")
    print("=" * 80)

    # Create visualizer and run
    visualizer = AggregatedResultsVisualizer(
        str(input_dir),
        str(output_dir),
        n_features=n_features,
        feature_importance_from=feature_importance_from,
        shap_data_dir=str(shap_data_dir),
    )
    visualizer.visualize_all_experiments()

    print("\nAll visualizations completed!")


if __name__ == "__main__":
    main()
