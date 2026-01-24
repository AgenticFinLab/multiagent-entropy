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
    
    def plot_feature_importance(self, ax, top_n=20):
        """
        Plot feature importance bar chart (Subplot 1).
        
        Args:
            ax: Matplotlib axis object
            top_n: Number of top features to display
        """
        # Load feature importance data
        fi_df = self.load_feature_importance(model_name='LightGBM')
        
        # Sort and get top N features
        fi_df = fi_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create horizontal bar plot
        ax.barh(
            range(len(fi_df)),
            fi_df['Importance'].values,
            color='#4575B4',
            edgecolor='white',
            linewidth=0.8
        )
        
        # Set labels
        ax.set_yticks(range(len(fi_df)))
        ax.set_yticklabels(fi_df['Feature'].values, fontsize=11)
        ax.invert_yaxis()
        
        # Styling
        ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
        ax.set_title('(a)', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
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
            'base_model_is_finally_correct',
            'base_sample_total_entropy',
            'base_sample_token_count'
        ]
        
        # Check if features exist
        available_features = [f for f in focus_features if f in shap_df.columns and f in x_test_df.columns]
        
        if not available_features:
            ax.text(0.5, 0.5, 'No base model features available in SHAP data',
                   ha='center', va='center', fontsize=12)
            ax.set_title('(b)', fontsize=16, fontweight='bold', pad=15)
            return
        
        # Plot scatter for each feature
        colors = ['#D73027', '#4575B4', '#91BFD8']
        markers = ['o', 's', '^']
        
        for idx, feature in enumerate(available_features):
            feature_values = x_test_df[feature].values
            shap_values = shap_df[feature].values
            
            # Calculate correlation
            corr, _ = pearsonr(feature_values, shap_values)
            
            # Plot scatter
            ax.scatter(
                feature_values,
                shap_values,
                alpha=0.6,
                s=30,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                label=f'{feature}\n(r={corr:.3f})',
                edgecolors='white',
                linewidth=0.5
            )
        
        # Styling
        ax.set_xlabel('Feature Value', fontsize=14, fontweight='bold')
        ax.set_ylabel('SHAP Value', fontsize=14, fontweight='bold')
        ax.set_title('(b)', fontsize=16, fontweight='bold', pad=15)
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
            ax.set_title('(c)', fontsize=16, fontweight='bold', pad=15)
            return
        
        # Focus features for x-axis
        focus_feature = 'base_sample_total_entropy'  # Use one feature as example
        
        if focus_feature not in df_qwen.columns or 'is_finally_correct' not in df_qwen.columns:
            ax.text(0.5, 0.5, f'Required columns not found in data',
                   ha='center', va='center', fontsize=12)
            ax.set_title('(c)', fontsize=16, fontweight='bold', pad=15)
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
        
        # Styling
        ax.set_xlabel(f'Base Model Entropy', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('(c)', fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=False, fontsize=13, ncol=2)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
    
    def load_combined_summary(self):
        """
        Load combined summary data.
            
        Returns:
            DataFrame containing summary data
        """
        if not self.combined_summary_path.exists():
            raise FileNotFoundError(f"Combined summary file not found: {self.combined_summary_path}")
            
        df = pd.read_csv(self.combined_summary_path)
        print(f"Loaded combined summary data: {len(df)} samples")
        return df
        
    def plot_base_model_accuracy_improvement(self, ax):
        """
        Plot absolute accuracy of different architectures and base models (Subplot 4).
        Aggregated across all datasets for each model size.
        
        Args:
            ax: Matplotlib axis object
        """
        # Load summary data
        df = self.load_combined_summary()
        
        # Map model names to display names
        model_name_map = {
            'qwen3_0_6b': 'Qwen3-0.6B',
            'qwen3_4b': 'Qwen3-4B',
            'qwen3_8b': 'Qwen3-8B'
        }
        df['model_display'] = df['model'].map(model_name_map)
        
        # Ensure we only have the three models we care about
        df = df[df['model_display'].notna()]
        
        # Convert architecture accuracy to 0-100 scale
        df['arch_accuracy_pct'] = df['accuracy'] * 100
        
        # 1. Prepare Base Model data
        base_df = df.groupby('model_display')['base model accuracy'].mean().reset_index()
        base_df['architecture'] = 'Base'
        base_df.rename(columns={'base model accuracy': 'accuracy_value'}, inplace=True)
        
        # 2. Prepare Architecture data
        arch_df = df.groupby(['model_display', 'architecture'])['arch_accuracy_pct'].mean().reset_index()
        arch_df.rename(columns={'arch_accuracy_pct': 'accuracy_value'}, inplace=True)
        
        # 3. Combine for plotting
        plot_df = pd.concat([base_df, arch_df], ignore_index=True)
        
        # Filter for standard architectures + Base
        architectures = ['Base', 'centralized', 'debate', 'hybrid', 'sequential', 'single']
        plot_df = plot_df[plot_df['architecture'].isin(architectures)]
        
        # Define extended color map
        full_color_map = self.color_map.copy()
        full_color_map['Base'] = '#D3D3D3'  # Light Gray for Base
        
        # Plotting
        models_order = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-8B']
        sns.barplot(
            data=plot_df,
            x='model_display',
            y='accuracy_value',
            hue='architecture',
            hue_order=architectures,
            order=models_order,
            palette=full_color_map,
            ax=ax,
            edgecolor='white',
            linewidth=0.8
        )
        
        # Styling
        ax.set_xlabel('Model Size', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('(d)', fontsize=16, fontweight='bold', pad=15)
        
        # Legend: No border
        ax.legend(loc='upper left', frameon=False, fontsize=13, ncol=2)
        
        # Set y-axis limits to zoom in on the data range but keep it readable
        # This satisfies the requirement of not forcing a Y=0 baseline if it squashes things
        min_acc = plot_df['accuracy_value'].min()
        max_acc = plot_df['accuracy_value'].max()
        ax.set_ylim(max(0, min_acc - 10), min(100, max_acc + 15))
        
        ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)

    def generate_comprehensive_figure(self):
        """
        Generate the comprehensive four-subplot figure.
        """
        # Create figure with 4 subplots
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        print("Generating subplot 1: Feature Importance...")
        self.plot_feature_importance(axes[0])
        
        print("Generating subplot 2: SHAP Scatter Plots...")
        self.plot_shap_scatter(axes[1])
        
        print("Generating subplot 3: Accuracy Trends...")
        self.plot_accuracy_trends(axes[2])
        
        print("Generating subplot 4: Accuracy Improvement...")
        self.plot_base_model_accuracy_improvement(axes[3])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "base_model_analysis.png"
        plt.savefig(output_path, dpi=900, bbox_inches='tight')
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
