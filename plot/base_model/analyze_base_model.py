"""
Comprehensive Visualization for Base Model Analysis

This script generates a four-subplot figure with ICML-compliant styling:
1. Feature importance bar chart
2. SHAP value scatter plots with correlation annotations
3. Accuracy trends across different architectures and base model features
4. Impact of base model accuracy on different architectures
"""

import warnings
from pathlib import Path

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


class BaseModelVisualizer:
    """Comprehensive visualizer for base model analysis."""
    
    def __init__(
        self,
        feature_importance_dir: str,
        shap_dir: str,
        merged_data_path: str,
        combined_summary_path: str,
        output_dir: str
    ):
        """
        Initialize the visualizer.
        
        Args:
            feature_importance_dir: Directory containing feature importance CSVs
            shap_dir: Directory containing SHAP analysis results
            merged_data_path: Path to merged_datasets.csv
            combined_summary_path: Path to combined_summary_data.csv
            output_dir: Directory to save output figures
        """
        self.feature_importance_dir = Path(feature_importance_dir)
        self.shap_dir = Path(shap_dir)
        self.merged_data_path = Path(merged_data_path)
        self.combined_summary_path = Path(combined_summary_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ICML color palette (consistent with analyze_accuracy.py)
        self.color_map = {
            'centralized': '#D73027',
            'debate': '#FC8D59',
            'hybrid': '#FEE090',
            'sequential': '#4575B4',
            'single': '#91BFD8'
        }
        
        # Configure ICML-style plotting
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure global plotting parameters for ICML style."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['legend.title_fontsize'] = 14
        plt.rcParams['legend.fontsize'] = 13
    
    def load_feature_importance(self, model_name='LightGBM'):
        """
        Load feature importance data.
        
        Args:
            model_name: Model name (LightGBM, XGBoost, or RandomForest)
            
        Returns:
            DataFrame containing feature importance data
        """
        csv_path = self.feature_importance_dir / f"Feature_Importance_-_{model_name}_(Classification).csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature importance file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded feature importance data: {len(df)} features")
        return df
    
    def load_shap_data(self, model_name='LightGBM'):
        """
        Load SHAP values and X_test data.
        
        Args:
            model_name: Model name
            
        Returns:
            Tuple of (shap_values_df, X_test_df)
        """
        shap_csv_path = self.shap_dir / f"shap_values_{model_name}_classification.csv"
        x_test_csv_path = self.shap_dir / f"X_test_{model_name}_classification.csv"
        
        if not shap_csv_path.exists():
            raise FileNotFoundError(f"SHAP values file not found: {shap_csv_path}")
        if not x_test_csv_path.exists():
            raise FileNotFoundError(f"X_test file not found: {x_test_csv_path}")
        
        shap_df = pd.read_csv(shap_csv_path, index_col='sample_index')
        x_test_df = pd.read_csv(x_test_csv_path, index_col=0)
        
        print(f"Loaded SHAP data: {len(shap_df)} samples, {len(shap_df.columns)} features")
        return shap_df, x_test_df
    
    def load_merged_data(self):
        """
        Load merged datasets CSV.
        
        Returns:
            DataFrame containing merged experiment data
        """
        if not self.merged_data_path.exists():
            raise FileNotFoundError(f"Merged data file not found: {self.merged_data_path}")
        
        df = pd.read_csv(self.merged_data_path)
        print(f"Loaded merged data: {len(df)} samples")
        return df
    
    def plot_feature_importance(self, ax, top_n=10):
        """
        Plot feature importance bar chart (Subplot 1).
        
        Args:
            ax: Matplotlib axis object
            top_n: Number of top features to display
        """
        # Load feature importance data from the new CSV file
        csv_path = Path('data_mining/results_qwen/results_aggregated/exclude_base_model_wo_entropy.csv')
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature importance file not found: {csv_path}")
        
        fi_df = pd.read_csv(csv_path)
        
        # Apply min-max normalization to lightgbm_importance
        lightgbm_min = fi_df['lightgbm_importance'].min()
        lightgbm_max = fi_df['lightgbm_importance'].max()
        if lightgbm_max > lightgbm_min:
            fi_df['lightgbm_importance_normalized'] = (fi_df['lightgbm_importance'] - lightgbm_min) / (lightgbm_max - lightgbm_min)
        else:
            fi_df['lightgbm_importance_normalized'] = fi_df['lightgbm_importance']
        
        # Sort by mean_importance_normalized and get top N features
        fi_df = fi_df.sort_values('mean_importance_normalized', ascending=False).head(top_n)
        
        # Calculate bar positions (y-axis coordinates)
        y_pos = np.arange(len(fi_df))
        bar_height = 0.25  # Height of each bar
        
        # Define colors for the three bar types
        colors = {
            'lightgbm': '#91BFD8',      # Sky Blue
            'xgboost': '#4575B4',        # Blue
            'mean': '#D73027'            # Red
        }
        
        # Create three horizontal bar plots for each feature
        ax.barh(
            y_pos - bar_height,
            fi_df['lightgbm_importance_normalized'].values,
            height=bar_height,
            color=colors['lightgbm'],
            edgecolor='white',
            linewidth=0.8,
            label='LightGBM (Normalized)'
        )
        
        ax.barh(
            y_pos,
            fi_df['xgboost_importance'].values,
            height=bar_height,
            color=colors['xgboost'],
            edgecolor='white',
            linewidth=0.8,
            label='XGBoost'
        )
        
        ax.barh(
            y_pos + bar_height,
            fi_df['mean_importance_normalized'].values,
            height=bar_height,
            color=colors['mean'],
            edgecolor='white',
            linewidth=0.8,
            label='Mean (Normalized)'
        )
        
        # Set labels and ticks
        ax.set_yticks(y_pos)
        
        # Auto-wrap text to prevent overlap
        labels = [textwrap.fill(text.replace('_', ' '), width=24) for text in fi_df['feature'].values]
        ax.set_yticklabels(labels, fontsize=11)
        
        ax.invert_yaxis()
        
        # Styling
        ax.set_xlabel('Feature Importance', fontsize=14)
        ax.text(0.5, -0.2, '(d)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='lower right', frameon=False, fontsize=12)
        
        # Remove top and right spines for a cleaner look
        sns.despine(ax=ax, top=True, right=True)
    
    def plot_shap_scatter(self, ax):
        """
        Plot SHAP value scatter plots with correlation (Subplot 2).
        
        Args:
            ax: Matplotlib axis object
        """
        # Load SHAP data
        shap_df, x_test_df = self.load_shap_data(model_name='LightGBM')
        
        # Focus features
        focus_features = [
            'base_model_answer_token_count',
            'base_sample_total_entropy',
            # 'base_sample_token_count'
        ]
        
        # Check if features exist
        available_features = [f for f in focus_features if f in shap_df.columns and f in x_test_df.columns]
        
        if not available_features:
            ax.text(0.5, 0.5, 'No base model features available in SHAP data',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.2, '(e)', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Plot scatter for each feature
        colors = ['#D73027', '#4575B4', '#91BFD8']
        markers = ['o', 's', '^']
        
        for idx, feature in enumerate(available_features):
            feature_values = x_test_df[feature].values
            shap_values = shap_df[feature].values
            
            # Calculate correlation
            corr, _ = pearsonr(feature_values, shap_values)
            
            feature_map = {
                'base_model_answer_token_count': 'Base Model Answer Token Count',
                'base_sample_total_entropy': 'Base Model Entropy'
            }

            # Plot scatter
            ax.scatter(
                feature_values,
                shap_values,
                alpha=0.6,
                s=30,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                label=f'{feature_map.get(feature, feature)}\n(Pearson Correlation ={corr:.3f})',
                edgecolors='white',
                linewidth=0.5
            )
        
        # Styling
        ax.set_xlabel('Feature Value', fontsize=14)
        ax.set_ylabel('SHAP Value', fontsize=14)
        ax.text(0.5, -0.2, '(e)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.legend(loc='best', frameon=False, fontsize=13)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
    
    def plot_accuracy_trends(self, ax):
        """
        Plot accuracy trends across architectures (Subplot 3).
        
        Args:
            ax: Matplotlib axis object
        """
        # Load merged data
        df = self.load_merged_data()
        
        # Filter for qwen models only
        qwen_models = ['qwen3_0_6b', 'qwen3_4b', 'qwen3_8b']
        df_qwen = df[df['model_name'].isin(qwen_models)]
        
        if len(df_qwen) == 0:
            ax.text(0.5, 0.5, 'No qwen model data available',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.2, '(f)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Focus features for x-axis
        focus_feature = 'base_sample_total_entropy'  # Use one feature as example
        
        if focus_feature not in df_qwen.columns or 'is_finally_correct' not in df_qwen.columns:
            ax.text(0.5, 0.5, f'Required columns not found in data',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.2, '(f)', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Group by architecture and calculate mean accuracy per feature bin
        architectures = ['centralized', 'debate', 'hybrid', 'sequential', 'single']
        available_archs = [a for a in architectures if a in df_qwen['architecture'].unique()]
        
        # Create bins for the feature
        df_qwen['feature_bin'] = pd.qcut(df_qwen[focus_feature], q=10, duplicates='drop')
        
        # Calculate accuracy for each architecture
        for arch in available_archs:
            arch_df = df_qwen[df_qwen['architecture'] == arch]
            grouped = arch_df.groupby('feature_bin')['is_finally_correct'].mean()
            
            # Get bin centers for x-axis
            bin_centers = []
            for interval in grouped.index:
                bin_centers.append((interval.left + interval.right) / 2)
            
            # Plot line
            ax.plot(
                bin_centers,
                grouped.values*100,
                marker='o',
                linewidth=2,
                markersize=6,
                color=self.color_map.get(arch, '#999999'),
                label=arch,
                alpha=0.8
            )

        ax.set_xscale('symlog', linthresh=100) 
        
        # Styling
        ax.set_xlabel(f'Base Model Entropy', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.text(0.5, -0.2, '(f)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.legend(loc='best', frameon=False, fontsize=13, ncol=2)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        # use both major and minor grids
        ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)    

    def generate_comprehensive_figure(self):
        """
        Generate the comprehensive four-subplot figure.
        """
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        print("Generating subplot 1: Feature Importance...")
        self.plot_feature_importance(axes[0])
        
        print("Generating subplot 2: SHAP Scatter Plots...")
        self.plot_shap_scatter(axes[1])
        
        print("Generating subplot 3: Accuracy Trends...")
        self.plot_accuracy_trends(axes[2])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "base_model_analysis.pdf"
        plt.savefig(output_path, dpi=1800, bbox_inches='tight', format='pdf')
        print(f"\nComprehensive figure saved to: {output_path}")
        
        plt.close()


def main():
    """Main function to run the comprehensive visualization."""
    # Define paths
    feature_importance_dir = "data_mining/results_qwen/results/exclude_base_model_wo_entropy/classification"
    shap_dir = "data_mining/results_qwen/results/exclude_base_model_wo_entropy/shap"
    merged_data_path = "data_mining/data/merged_datasets.csv"
    combined_summary_path = "evaluation/results_qwen/combined_summary_data.csv"
    output_dir = "plot/base_model"
    
    # Initialize visualizer
    visualizer = BaseModelVisualizer(
        feature_importance_dir=feature_importance_dir,
        shap_dir=shap_dir,
        merged_data_path=merged_data_path,
        combined_summary_path=combined_summary_path,
        output_dir=output_dir
    )
    
    # Generate comprehensive figure
    visualizer.generate_comprehensive_figure()


if __name__ == "__main__":
    main()
