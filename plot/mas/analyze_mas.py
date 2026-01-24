#!/usr/bin/env python3
"""
Comprehensive Visualization for Multi-Agent System Analysis
This module creates a three-subplot visualization maintaining ICML style consistency
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


class MASVisualizer:
    """Visualizer for multi-agent system comprehensive analysis with ICML-style formatting."""
    
    def __init__(
        self,
        feature_importance_path: str,
        shap_x_test_path: str,
        shap_values_path: str,
        lightgbm_pred_path: str,
        xgboost_pred_path: str,
        output_dir: str,
        top_features=None
    ):
        """
        Initialize paths to data files.
        
        Args:
            feature_importance_path: Path to aggregated feature importance CSV
            shap_x_test_path: Path to X_test CSV for SHAP
            shap_values_path: Path to SHAP values CSV
            lightgbm_pred_path: Path to LightGBM prediction probabilities
            xgboost_pred_path: Path to XGBoost prediction probabilities
            output_dir: Directory to save output figures
            top_features: List of feature names to focus on (e.g., ['sample_variance_entropy', ...])
        """
        self.feature_importance_path = Path(feature_importance_path)
        self.shap_x_test_path = Path(shap_x_test_path)
        self.shap_values_path = Path(shap_values_path)
        self.lightgbm_pred_path = Path(lightgbm_pred_path)
        self.xgboost_pred_path = Path(xgboost_pred_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define target features as instance variables
        if top_features is None:
            # Default fallback if not provided
            self.feature1 = 'sample_variance_entropy'
            self.feature2 = 'sample_round_1_q3_agent_variance_entropy'
            self.focus_features = [self.feature1, self.feature2]
        else:
            self.focus_features = top_features
            self.feature1 = top_features[0]
            self.feature2 = top_features[1] if len(top_features) > 1 else top_features[0]
        
        # ICML color palette (consistent with analyze_base_model.py)
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
        """Configure matplotlib with ICML-style settings."""
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
        """Load and process feature importance data."""
        fi_df = pd.read_csv(self.feature_importance_path)
        # Sort by mean_importance_normalized in descending order
        fi_df = fi_df.sort_values('mean_importance_normalized', ascending=False).reset_index(drop=True)
        return fi_df
    
    def load_shap_data(self, model_name='LightGBM'):
        """
        Load SHAP values and X_test data.
        
        Args:
            model_name: Model name (unused here as paths are explicit, kept for signature consistency)
            
        Returns:
            Tuple of (shap_values_df, X_test_df)
        """
        if not self.shap_values_path.exists():
            raise FileNotFoundError(f"SHAP values file not found: {self.shap_values_path}")
        if not self.shap_x_test_path.exists():
            raise FileNotFoundError(f"X_test file not found: {self.shap_x_test_path}")
        
        shap_df = pd.read_csv(self.shap_values_path, index_col='sample_index')
        x_test_df = pd.read_csv(self.shap_x_test_path, index_col=0)
        
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
        if not self.feature_importance_path.exists():
            raise FileNotFoundError(f"Feature importance file not found: {self.feature_importance_path}")
        
        fi_df = pd.read_csv(self.feature_importance_path)
        
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
        shap_df, x_test_df = self.load_shap_data()
        
        # Check if features exist in the loaded data
        # Filter self.focus_features to only include those present in the dataframe
        available_features = [f for f in self.focus_features if f in shap_df.columns and f in x_test_df.columns]
        
        if not available_features:
            ax.text(0.5, 0.5, 'No specified features available in SHAP data',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.2, '(e)', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Plot scatter for each feature
        colors = ['#4575B4', '#D73027', '#91BFD8']
        markers = ['o', 's', '^']
        
        for idx, feature in enumerate(available_features):
            # Retain specific preprocessing logic if it matches specific known features
            if feature == 'sample_answer_token_count':
                x_test_df[feature] /= 100.0

            feature_values = x_test_df[feature].values
            shap_values = shap_df[feature].values
            
            # Calculate correlation
            corr, _ = pearsonr(feature_values, shap_values)
            
            feature_map = {
                'sample_variance_entropy': 'Sample Variance Entropy',
                'sample_round_1_q3_agent_variance_entropy': 'Round 1 Q3 Variance Entropy',
                'sample_answer_token_count': 'Sample Answer Token Count',
            }

            # Get display name
            display_name = feature_map.get(feature, feature.replace('_', ' ').title())

            # Plot scatter
            ax.scatter(
                feature_values,
                shap_values,
                alpha=0.6,
                s=30,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                label=f'{display_name}\n(Pearson Correlation ={corr:.3f})',
                edgecolors='white',
                linewidth=0.5
            )
        
        # Styling
        ax.set_xlabel('Feature Value', fontsize=14)
        ax.set_ylabel('SHAP Value', fontsize=14)
        ax.text(0.5, -0.2, '(e)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', frameon=False, fontsize=13)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
    
    def plot_top2_entropy_scatter(self, ax):
        """
        Plot scatter of two entropy features with prediction probability encoding (Subplot 3).
        Uses feature1 and feature2 defined in __init__.
        """
        # Load feature values from X_test
        x_test = pd.read_csv(self.shap_x_test_path)
        
        # Use instance variables feature1 and feature2
        f1, f2 = self.feature1, self.feature2
        
        # Check if the required features exist in the dataset
        if f1 not in x_test.columns or f2 not in x_test.columns:
            print(f"Warning: Required features not found ({f1}, {f2}). Available features: {list(x_test.columns[:10])}...")
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
        x1 = x_test[f1].values
        x2 = x_test[f2].values
        
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
                      edgecolors='white',
                      label=f'Positive (prob1 > prob0)', marker='^')
        
        # Format labels for display
        def format_label(name):
            return name.replace('_', ' ').title()

        ax.set_xlabel(format_label(f1), fontsize=15)
        ax.set_ylabel("Round 1 Q3 Variance Entropy", fontsize=15)
        ax.text(0.5, -0.2, '(f)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16, fontweight='bold')
        
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
               verticalalignment='top')

    def generate_comprehensive_figure(self):
        """
        Generate the comprehensive three-subplot figure.
        """
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        print("Generating subplot 1: Feature Importance...")
        self.plot_feature_importance(axes[0])
        
        print("Generating subplot 2: SHAP Value Scatter...")
        self.plot_shap_scatter(axes[1])
        
        print("Generating subplot 3: Top 2 Entropy Features Scatter...")
        self.plot_top2_entropy_scatter(axes[2])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "mas_analysis.pdf"
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='pdf')
        print(f"\nComprehensive figure saved to: {output_path}")
        
        plt.close()


def main():
    """Main function to generate the comprehensive visualization."""
    
    # Define paths
    base_dir = '/home/yuxuanzhao/multiagent-entropy'
    feature_importance_path = f"{base_dir}/data_mining/results_aggregated/exclude_base_model_all_metrics.csv"
    shap_x_test_path = f"{base_dir}/data_mining/results/exclude_base_model_all_metrics/shap/X_test_LightGBM_classification.csv"
    shap_values_path = f"{base_dir}/data_mining/results/exclude_base_model_all_metrics/shap/shap_values_LightGBM_classification.csv"
    lightgbm_pred_path = f"{base_dir}/data_mining/results/exclude_base_model_all_metrics/shap/shap_prediction_probabilities_LightGBM_classification.csv"
    xgboost_pred_path = f"{base_dir}/data_mining/results/exclude_base_model_all_metrics/shap/shap_prediction_probabilities_XGBoost_classification.csv"
    output_dir = f"{base_dir}/plot/mas"
    
    # Define the top features here to easily change them
    top_features = [
        'sample_variance_entropy', 
        'sample_round_1_q3_agent_variance_entropy',
        # 'sample_answer_token_count'
    ]
    
    # Initialize visualizer
    visualizer = MASVisualizer(
        feature_importance_path=feature_importance_path,
        shap_x_test_path=shap_x_test_path,
        shap_values_path=shap_values_path,
        lightgbm_pred_path=lightgbm_pred_path,
        xgboost_pred_path=xgboost_pred_path,
        output_dir=output_dir,
        top_features=top_features
    )
    
    # Generate comprehensive figure
    visualizer.generate_comprehensive_figure()


if __name__ == "__main__":
    main()