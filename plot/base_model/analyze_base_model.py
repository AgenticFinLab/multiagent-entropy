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
        output_dir: str
    ):
        """
        Initialize the visualizer.
        
        Args:
            feature_importance_dir: Directory containing feature importance CSVs
            shap_dir: Directory containing SHAP analysis results
            merged_data_path: Path to merged_datasets.csv
            output_dir: Directory to save output figures
        """
        self.feature_importance_dir = Path(feature_importance_dir)
        self.shap_dir = Path(shap_dir)
        self.merged_data_path = Path(merged_data_path)
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

        xgb_min = fi_df['xgboost_importance'].min()
        xgb_max = fi_df['xgboost_importance'].max()
        if xgb_max > xgb_min:
            fi_df['xgboost_importance_normalized'] = (fi_df['xgboost_importance'] - xgb_min) / (xgb_max - xgb_min)
        else:
            fi_df['xgboost_importance_normalized'] = fi_df['xgboost_importance']
        
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
            label='LightGBM'
        )
        
        ax.barh(
            y_pos,
            fi_df['xgboost_importance_normalized'].values,
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
            label='Mean'
        )
        
        # Set labels and ticks
        ax.set_yticks(y_pos)
        
        # Define feature name mapping
        feature_mapping = {
            'base_model_answer_token_count': 'base model answer token count',
            'base_sample_total_entropy': 'base model entropy',
            'base_sample_token_count': 'base model total token count'
        }
        
        # Auto-wrap text to prevent overlap, applying feature name mapping
        labels = []
        for text in fi_df['feature'].values:
            # Apply feature name mapping if available
            mapped_text = feature_mapping.get(text, text)
            # Replace underscores with spaces and wrap text
            wrapped_text = textwrap.fill(mapped_text.replace('_', ' '), width=24)
            labels.append(wrapped_text)

        ax.set_yticklabels(labels, fontsize=11)
        
        ax.invert_yaxis()
        
        # Styling
        ax.set_xlabel('Feature Importance', fontsize=14)
        ax.text(0.5, -0.2, '(c)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.grid(True, axis='x', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='lower right', frameon=False, fontsize=12)
        
        # Remove top and right spines for a cleaner look
        sns.despine(ax=ax, top=True, right=True)
    
    def plot_shap_with_importance_inset(self, ax):
        """
        Plot SHAP value scatter plots with embedded feature importance inset.
        Combines original subplot 1 (feature importance) and subplot 2 (SHAP scatter).
        
        Args:
            ax: Matplotlib axis object
        """
        # Load SHAP data
        shap_df, x_test_df = self.load_shap_data(model_name='LightGBM')
        
        # Focus features
        focus_features = [
            'base_model_answer_token_count',
            'base_sample_total_entropy',
        ]
        
        # Check if features exist
        available_features = [f for f in focus_features if f in shap_df.columns and f in x_test_df.columns]
        
        if not available_features:
            ax.text(0.5, 0.5, 'No base model features available in SHAP data',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.2, '(c)', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Plot scatter for each feature
        colors = ['#4575B4', '#D73027']
        markers = ['o', 's', '^']
        
        for idx, feature in enumerate(available_features):
            feature_values = x_test_df[feature].values.copy()
            shap_values = shap_df[feature].values
            
            # Normalize feature values to [0, 1] for consistent scale
            fv_min, fv_max = np.nanmin(feature_values), np.nanmax(feature_values)
            if fv_max > fv_min:
                feature_values_norm = (feature_values - fv_min) / (fv_max - fv_min)
            else:
                feature_values_norm = np.full_like(feature_values, 0.5)
            
            # Calculate correlation using original values
            corr, _ = pearsonr(feature_values, shap_values)
            
            feature_map = {
                'base_model_answer_token_count': 'Base Model Answer Token',
                'base_sample_total_entropy': 'Base Model Entropy',
                'base_sample_token_count': 'Base Model Total Token'
            }

            # Plot scatter with normalized feature values
            ax.scatter(
                feature_values_norm,
                shap_values,
                alpha=0.6,
                s=30,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                label=f'{feature_map.get(feature, feature)}\n(Pearson Correlation={corr:.3f})',
                edgecolors='white',
                linewidth=0.5
            )
        
        # Main plot styling
        ax.set_xlabel('Normalized Feature Value', fontsize=14)
        ax.set_ylabel('SHAP Value', fontsize=14)
        ax.text(0.5, -0.2, '(c)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', frameon=False, fontsize=11)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
        
        # ========== Add inset axes for feature importance ==========
        inset_ax = ax.inset_axes([0.5, 0.72, 0.40, 0.25])  # [x, y, width, height]
        self._plot_importance_inset(inset_ax)
    
    def _plot_importance_inset(self, ax, top_n=5):
        """
        Plot simplified feature importance bar chart as inset.
        
        Args:
            ax: Matplotlib axis object (inset)
            top_n: Number of top features to display
        """
        # Load feature importance data
        csv_path = Path('data_mining/results_qwen/results_aggregated/exclude_base_model_wo_entropy.csv')
        if not csv_path.exists():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=8)
            return
        
        fi_df = pd.read_csv(csv_path)
        
        # Apply min-max normalization
        lightgbm_min = fi_df['lightgbm_importance'].min()
        lightgbm_max = fi_df['lightgbm_importance'].max()
        if lightgbm_max > lightgbm_min:
            fi_df['lightgbm_importance_normalized'] = (fi_df['lightgbm_importance'] - lightgbm_min) / (lightgbm_max - lightgbm_min)
        else:
            fi_df['lightgbm_importance_normalized'] = fi_df['lightgbm_importance']

        xgb_min = fi_df['xgboost_importance'].min()
        xgb_max = fi_df['xgboost_importance'].max()
        if xgb_max > xgb_min:
            fi_df['xgboost_importance_normalized'] = (fi_df['xgboost_importance'] - xgb_min) / (xgb_max - xgb_min)
        else:
            fi_df['xgboost_importance_normalized'] = fi_df['xgboost_importance']
        
        # Sort and get top N features
        fi_df = fi_df.sort_values('mean_importance_normalized', ascending=False).head(top_n)
        
        # Calculate bar positions
        y_pos = np.arange(len(fi_df))
        bar_height = 0.25
        
        # Colors
        colors = {
            'lightgbm': '#91BFD8',
            'xgboost': '#4575B4',
            'mean': '#D73027'
        }
        
        # Create horizontal bars
        ax.barh(y_pos - bar_height, fi_df['lightgbm_importance_normalized'].values,
               height=bar_height, color=colors['lightgbm'], edgecolor='white',
               linewidth=0.5, label='LightGBM')
        ax.barh(y_pos, fi_df['xgboost_importance_normalized'].values,
               height=bar_height, color=colors['xgboost'], edgecolor='white',
               linewidth=0.5, label='XGBoost')
        ax.barh(y_pos + bar_height, fi_df['mean_importance_normalized'].values,
               height=bar_height, color=colors['mean'], edgecolor='white',
               linewidth=0.5, label='Mean')
        
        # Feature name mapping (shortened)
        feature_mapping = {
            'base_model_answer_token_count': 'base model answer token',
            'base_sample_total_entropy': 'base model entropy',
            'base_sample_token_count': 'base model total token',
            'base_sample_avg_entropy_per_token': 'base model avg entropy/token'
        }
        
        # Set labels
        ax.set_yticks(y_pos)
        labels = []
        for f in fi_df['feature'].values:
            label = feature_mapping.get(f, f.replace('_', ' '))
            if len(label) > 24:
                label = label[:24] + '...'
            labels.append(label)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        
        # Styling for inset
        ax.set_xlabel('Importance', fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.legend(loc='lower right', frameon=False, fontsize=7, ncol=1)
        ax.set_facecolor('white')
        ax.patch.set_alpha(0.9)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')
            spine.set_linewidth(0.8)
        
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.4)
    
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
        
        # Calculate average base model stats for marking
        base_stats = []
        for model in qwen_models:
            model_df = df[df['model_name'] == model]
            if not model_df.empty:
                avg_entropy = model_df['base_sample_total_entropy'].mean()
                avg_acc = model_df['base_model_accuracy'].mean() * 100  # Convert to percentage
                base_stats.append({
                    'model': model,
                    'avg_entropy': avg_entropy,
                    'avg_acc': avg_acc
                })

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

        models_map = {'qwen3_0_6b': 'Qwen3-0.6B', 'qwen3_4b': 'Qwen3-4B', 'qwen3_8b': 'Qwen3-8B'}
        # Plot base model points with special markers
        markers = ['X', '^', 's']
        colors = ['#D73027', '#56B4E9', '#FEE090'] # Distinct colors
        for i, stat in enumerate(base_stats):
            ax.scatter(
                stat['avg_entropy'], 
                stat['avg_acc'],
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                s=120,  # Large size
                label=f'{models_map[stat["model"]]} (Base)',
                edgecolors='white',
                linewidths=1.5,
                zorder=5
            )

        ax.set_xscale('symlog', linthresh=100) 
        
        # Styling
        ax.set_xlabel(f'Base Model Entropy', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.text(0.5, -0.2, '(d)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.legend(loc='best', frameon=False, fontsize=11, ncol=2)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        # use both major and minor grids
        ax.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)        

    def generate_comprehensive_figure(self):
        """
        Generate the comprehensive two-subplot figure.
        Subplot 1: SHAP scatter with embedded feature importance inset
        Subplot 2: Accuracy trends
        """
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        print("Generating subplot 1: SHAP Scatter with Feature Importance Inset...")
        self.plot_shap_with_importance_inset(axes[0])
        
        print("Generating subplot 2: Accuracy Trends...")
        self.plot_accuracy_trends(axes[1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "base_model_analysis.pdf"
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='pdf')
        print(f"\nComprehensive figure saved to: {output_path}")
        
        plt.close()


def main():
    """Main function to run the comprehensive visualization."""
    # Define paths
    feature_importance_dir = "data_mining/results_qwen/results/exclude_base_model_wo_entropy/classification"
    shap_dir = "data_mining/results_qwen/results/exclude_base_model_wo_entropy/shap"
    merged_data_path = "data_mining/data/merged_datasets.csv"
    output_dir = "plot/base_model"
    
    # Initialize visualizer
    visualizer = BaseModelVisualizer(
        feature_importance_dir=feature_importance_dir,
        shap_dir=shap_dir,
        merged_data_path=merged_data_path,
        output_dir=output_dir
    )
    
    # Generate comprehensive figure
    visualizer.generate_comprehensive_figure()


if __name__ == "__main__":
    main()
