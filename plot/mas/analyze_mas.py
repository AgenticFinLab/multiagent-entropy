#!/usr/bin/env python3
"""
Comprehensive Visualization for Multi-Agent System Analysis
This module creates a three-subplot visualization maintaining ICML style consistency
"""

import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import pearsonr

from pathlib import Path
class MASVisualizer:
    """Visualizer for multi-agent system comprehensive analysis with ICML-style formatting"""
    
    def __init__(self, base_dir='/home/yuxuanzhao/multiagent-entropy'):
        """Initialize paths to data files"""
        self.base_dir = base_dir
        
        # Define file paths
        self.feature_importance_path = os.path.join(
            base_dir, 
            'data_mining/results_aggregated/exclude_base_model_all_metrics.csv'
        )
        self.x_test_path = os.path.join(
            base_dir,
            'data_mining/results/exclude_base_model_all_metrics/shap/X_test_LightGBM_classification.csv'
        )
        self.shap_values_path = os.path.join(
            base_dir,
            'data_mining/results/exclude_base_model_all_metrics/shap/shap_values_LightGBM_classification.csv'
        )
        self.lightgbm_pred_path = os.path.join(
            base_dir,
            'data_mining/results/exclude_base_model_all_metrics/shap/shap_prediction_probabilities_LightGBM_classification.csv'
        )
        self.xgboost_pred_path = os.path.join(
            base_dir,
            'data_mining/results/exclude_base_model_all_metrics/shap/shap_prediction_probabilities_XGBoost_classification.csv'
        )
        
        # ICML color palette (from analyze_base_model.py)
        self.colors = {
            'lightgbm': '#D73027',
            'xgboost': '#4575B4',
            'mean': '#FEE090'
        }
        
        # Configure ICML-style plotting
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Configure matplotlib with ICML-style settings"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['legend.title_fontsize'] = 14
        plt.rcParams['legend.fontsize'] = 13
    
    def load_feature_importance(self):
        """Load and process feature importance data"""
        fi_df = pd.read_csv(self.feature_importance_path)
        # Sort by mean_importance_normalized in descending order
        fi_df = fi_df.sort_values('mean_importance_normalized', ascending=False).reset_index(drop=True)
        return fi_df
    
    def load_shap_data(self, model_name='LightGBM'):
        """
        Load SHAP values and X_test data.
        
        Args:
            model_name: Model name
            
        Returns:
            Tuple of (shap_values_df, X_test_df)
        """
        shap_csv_path = Path(self.shap_values_path)
        x_test_csv_path = Path(self.x_test_path)
        
        if not shap_csv_path.exists():
            raise FileNotFoundError(f"SHAP values file not found: {shap_csv_path}")
        if not x_test_csv_path.exists():
            raise FileNotFoundError(f"X_test file not found: {x_test_csv_path}")
        
        shap_df = pd.read_csv(shap_csv_path, index_col='sample_index')
        x_test_df = pd.read_csv(x_test_csv_path, index_col=0)
        
        print(f"Loaded SHAP data: {len(shap_df)} samples, {len(shap_df.columns)} features")
        return shap_df, x_test_df

    
    def plot_feature_importance(self, ax, top_n=10):
        """
        Plot feature importance bar chart (Subplot 1).
        
        Args:
            ax: Matplotlib axis object
            top_n: Number of top features to display
        """
        # Load feature importance data from the new CSV file
        csv_path = Path(self.feature_importance_path)
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
            'base_sample_token_count': 'base model total token count',
            'base_sample_avg_entropy_per_token': 'base model avg entropy per token'
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
            'sample_variance_entropy',
            # 'sample_answer_token_count',
            'sample_round_1_q3_agent_variance_entropy',
            # "sample_round_1_q3_agent_std_entropy"
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
        colors = ['#4575B4', '#D73027', '#91BFD8']
        markers = ['o', 's', '^']
        
        for idx, feature in enumerate(available_features):

            if feature == 'sample_answer_token_count':
                x_test_df[feature] /= 100.0

            feature_values = x_test_df[feature].values


            shap_values = shap_df[feature].values
            
            # Calculate correlation
            corr, _ = pearsonr(feature_values, shap_values)
            
            feature_map = {
                'sample_variance_entropy': 'Sample Variance Entropy',
                'sample_round_1_q3_agent_variance_entropy': 'Sample Round 1 Q3 Agent Variance Entropy',
                # 'sample_answer_token_count': 'Sample Answer Token Count',
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
    
    def plot_top2_entropy_scatter(self, ax, feature1='sample_variance_entropy', feature2='sample_round_1_q3_agent_variance_entropy'):
        """
        Plot scatter of two entropy features with prediction probability encoding (Subplot 3)
        This is the most important subplot
        """
        # Load feature values from X_test
        x_test = pd.read_csv(self.x_test_path)
        
        # Check if the required features exist in the dataset
        if feature1 not in x_test.columns or feature2 not in x_test.columns:
            print(f"Warning: Required features not found. Available features: {list(x_test.columns[:10])}...")
            return
        
        # Load prediction probabilities for both classes
        lightgbm_df = pd.read_csv(self.lightgbm_pred_path)
        xgboost_df = pd.read_csv(self.xgboost_pred_path)
        
        # Load prediction probabilities for both classes
        lgbm_prob0 = pd.to_numeric(lightgbm_df['prob_class_0'].values, errors='coerce')
        lgbm_prob1 = pd.to_numeric(lightgbm_df['prob_class_1'].values, errors='coerce')
        xgb_prob0 = pd.to_numeric(xgboost_df['prob_class_0'].values, errors='coerce')
        xgb_prob1 = pd.to_numeric(xgboost_df['prob_class_1'].values, errors='coerce')
        
        # Get feature values
        x1 = x_test[feature1].values
        x2 = x_test[feature2].values
        
        # Align dimensions
        min_len = min(len(lgbm_prob0), len(xgb_prob0), len(x1), len(x2))
        lgbm_prob0, lgbm_prob1 = lgbm_prob0[:min_len], lgbm_prob1[:min_len]
        xgb_prob0, xgb_prob1 = xgb_prob0[:min_len], xgb_prob1[:min_len]
        x1, x2 = x1[:min_len], x2[:min_len]
        
        # Calculate average prediction probabilities for each class
        mean_prob0 = (lgbm_prob0 + xgb_prob0) / 2.0
        mean_prob1 = (lgbm_prob1 + xgb_prob1) / 2.0
        
        # Handle any NaN values
        mask_valid = ~(np.isnan(mean_prob0) | np.isnan(mean_prob1) | np.isnan(x1) | np.isnan(x2))
        x1, x2 = x1[mask_valid], x2[mask_valid]
        mean_prob0, mean_prob1 = mean_prob0[mask_valid], mean_prob1[mask_valid]
        
        # Separate positive and negative samples by comparing probabilities
        positive_mask = mean_prob1 > mean_prob0
        negative_mask = mean_prob0 >= mean_prob1
        
        # Scale point sizes based on the respective class probability
        size_scale = 100
        base_size = 10
        sizes_pos = mean_prob1 * size_scale + base_size
        sizes_neg = mean_prob0 * size_scale + base_size
        
        # Plot negative samples (class 0)
        if np.any(negative_mask):
            ax.scatter(x1[negative_mask], x2[negative_mask],
                      s=sizes_neg[negative_mask], c='#4575B4', alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label=f'Negative (prob0 > prob1)', marker='o')
        
        # Plot positive samples (class 1)
        if np.any(positive_mask):
            ax.scatter(x1[positive_mask], x2[positive_mask],
                      s=sizes_pos[positive_mask], c='#D73027', alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label=f'Positive (prob1 > prob0)', marker='^')
        
        ax.set_xlabel(f'{feature1}', fontsize=15, fontweight='bold')
        ax.set_ylabel(f'{feature2}', fontsize=15, fontweight='bold')
        ax.set_title(f'(c) {feature1} vs {feature2}', 
                    fontsize=16, fontweight='bold', pad=15)
        
        # Add grid
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='best', frameon=True, fancybox=False,
                 edgecolor='black', framealpha=0.95)
        
        # Adjust spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add size legend annotation
        ax.text(0.02, 0.98, 'Point size ∝ Class Probability',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def create_visualization(self, output_path=None):
        """
        Create the comprehensive 3-subplot visualization
        """
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        
        # Adjust layout
        plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.3)
        
        # Plot each subplot
        print("Generating subplot 1: Feature Importance...")
        self.plot_feature_importance(axes[0])
        
        print("Generating subplot 2: SHAP Value Scatter...")
        self.plot_shap_scatter(axes[1])
        
        print("Generating subplot 3: Top 2 Entropy Features Scatter...")
        self.plot_top2_entropy_scatter(axes[2])
        
        # Save figure
        if output_path is None:
            output_path = os.path.join(self.base_dir, 'plot/mas/mas_analysis.pdf')
        
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='pdf')
        print(f"Visualization saved to: {output_path}")
        
        return fig


def main():
    """Main function to generate the comprehensive visualization"""
    visualizer = MASVisualizer()
   
    print("\nGenerating comprehensive visualization...")
    visualizer.create_visualization()
    print("Done!")


if __name__ == "__main__":
    main()
