#!/usr/bin/env python3
"""
Comprehensive Visualization for RL Model Analysis
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
import math

warnings.filterwarnings("ignore")


class RLModelVisualizer:
    """Visualizer for RL model comprehensive analysis with ICML-style formatting."""
    
    def __init__(
        self,
        combined_summary_path: str,
        shap_dir: str,
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
            combined_summary_path: Path to combined_summary_data.csv for subplot 1
            shap_dir: Directory containing SHAP analysis results for subplot 2
            feature_importance_path: Path to aggregated feature importance CSV for subplot 3
            shap_x_test_path: Path to X_test CSV for subplot 3
            shap_values_path: Path to SHAP values CSV for subplot 3
            lightgbm_pred_path: Path to LightGBM prediction probabilities for subplot 3
            xgboost_pred_path: Path to XGBoost prediction probabilities for subplot 3
            output_dir: Directory to save output figures
            top_features: List of feature names to focus on for subplot 3
        """
        self.combined_summary_path = Path(combined_summary_path)
        self.shap_dir = Path(shap_dir)
        self.feature_importance_path = Path(feature_importance_path)
        self.shap_x_test_path = Path(shap_x_test_path)
        self.shap_values_path = Path(shap_values_path)
        self.lightgbm_pred_path = Path(lightgbm_pred_path)
        self.xgboost_pred_path = Path(xgboost_pred_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define target features for subplot 3
        if top_features is None:
            self.feature1 = 'sample_round_1_median_agent_total_entropy'
            self.feature2 = 'sample_round_1_q3_agent_total_entropy'
            self.focus_features = [self.feature1, self.feature2]
        else:
            self.focus_features = top_features
            self.feature1 = top_features[0]
            self.feature2 = top_features[1] if len(top_features) > 1 else top_features[0]
        
        # ICML color palette (consistent with other analysis files)
        self.color_map = {
            'base':        '#D3D3D3',
            'centralized': '#D73027',
            'debate':      '#FC8D59',
            'hybrid':      '#FEE090',
            'sequential':  '#4575B4',
            'single':      '#91BFD8'
        }
        
        # Configure ICML-style plotting
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Configure matplotlib with ICML-style settings."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.rcParams['axes.labelsize'] = 17
        plt.rcParams['legend.title_fontsize'] = 16
        plt.rcParams['legend.fontsize'] = 15
    
    def plot_accuracy_bar(self, ax):
        """
        Plot accuracy bar chart for RL models (Subplot 1).
        Reference: analyze_accuracy.py
        
        Args:
            ax: Matplotlib axis object
        """
        # Read data
        if not self.combined_summary_path.exists():
            raise FileNotFoundError(f"Combined summary file not found: {self.combined_summary_path}")
        
        df = pd.read_csv(self.combined_summary_path)
        
        # Get unique datasets with custom order (GSM8K at the end)
        all_datasets = df['dataset'].unique()
        # Define custom order: put gsm8k at the end
        dataset_order = []
        gsm8k_dataset = None
        for dataset in sorted(all_datasets):
            if 'gsm8k' in dataset.lower():
                gsm8k_dataset = dataset
            else:
                dataset_order.append(dataset)
        if gsm8k_dataset:
            dataset_order.append(gsm8k_dataset)
        datasets = dataset_order
        
        # Since we have multiple datasets, we'll plot all in one axis
        # We need to prepare data similar to analyze_accuracy.py
        
        # Add base model data as a new row for each dataset
        base_df_list = []
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            for model in dataset_df['model'].unique():
                model_data = dataset_df[dataset_df['model'] == model].iloc[0]
                base_row = {
                    'dataset': dataset,
                    'model': model,
                    'architecture': 'base',
                    'accuracy': model_data['base model accuracy'] / 100.0
                }
                base_df_list.append(base_row)
        
        base_df = pd.DataFrame(base_df_list)
        combined_df = pd.concat([base_df, df], ignore_index=True)
        
        # For simplicity, let's create a combined x-axis label
        combined_df['dataset_model'] = combined_df['dataset']

        dataset_name_map = {
            'math500': 'Math500',
            'aime2024_16384': 'AIME24',
            'aime2025_16384': 'AIME25',
            'gsm8k': 'GSM8K',
            'humaneval': 'HE',
            'mmlu': 'MMLU',
        }
        
        combined_df['dataset_model'] = combined_df['dataset_model'].map(dataset_name_map)
        
        # Define display order for datasets (GSM8K at the end)
        dataset_display_order = ['AIME24', 'AIME25', 'HE', 'Math500', 'MMLU', 'GSM8K']
        
        # change to percentage
        combined_df['accuracy'] = combined_df['accuracy'] * 100
        # Use seaborn barplot
        with sns.plotting_context("paper", font_scale=1.4):
            sns.barplot(
                data=combined_df,
                x='dataset_model',
                y='accuracy',
                hue='architecture',
                hue_order=['base', 'centralized', 'debate', 'hybrid', 'sequential', 'single'],
                order=dataset_display_order,
                ax=ax,
                palette=self.color_map,
                edgecolor='white',
                linewidth=0.8,
                saturation=0.9
            )
        
        # Styling
        ax.set_xlabel('Dataset', fontsize=17)
        ax.set_ylabel('Accuracy (%)', fontsize=17)
        ax.text(0.5, -0.25, '(a)', transform=ax.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
        ax.set_ylim(bottom=0)
        
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', frameon=False, fontsize=14, ncol=2)
    
    def plot_accuracy_trends(self, ax):
        """
        Plot accuracy trends across architectures (Subplot 2).
        Reference: analyze_base_model.py plot_accuracy_trends
        
        Args:
            ax: Matplotlib axis object
        """
        # Load SHAP data from shap directory
        merged_data_path = Path("plot/rl_model/merged_datasets.csv")
        
        if not merged_data_path.exists():
            ax.text(0.5, 0.5, f'X_test file not found in SHAP directory',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(b)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        df = pd.read_csv(merged_data_path)
        
        # Check required columns
        if 'is_finally_correct' not in df.columns:
            ax.text(0.5, 0.5, 'Required column not found in data',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(b)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Use base model entropy as focus feature (if available)
        focus_features = ['base_sample_total_entropy']
        focus_feature = None
        for feat in focus_features:
            if feat in df.columns:
                focus_feature = feat
                break
        
        if focus_feature is None:
            ax.text(0.5, 0.5, 'No suitable feature found for trends',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(b)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Group by architecture
        architectures = ['centralized', 'debate', 'hybrid', 'sequential', 'single']
        if 'architecture' not in df.columns:
            ax.text(0.5, 0.5, 'Architecture column not found',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(b)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        available_archs = [a for a in architectures if a in df['architecture'].unique()]
        
        # Calculate average base model stats for marking
        base_stats = []
        if 'model_name' in df.columns and 'base_model_accuracy' in df.columns:
            unique_models = df['model_name'].unique()
            for model in unique_models:
                model_df = df[df['model_name'] == model]
                if not model_df.empty and focus_feature in model_df.columns:
                    avg_entropy = model_df[focus_feature].mean()
                    avg_acc = model_df['base_model_accuracy'].mean() * 100  # Convert to percentage
                    base_stats.append({
                        'model': model,
                        'avg_entropy': avg_entropy,
                        'avg_acc': avg_acc
                    })
        
        # Create bins for the feature
        try:
            df['feature_bin'] = pd.qcut(df[focus_feature], q=10, duplicates='drop')
        except:
            # If qcut fails, use regular cut
            df['feature_bin'] = pd.cut(df[focus_feature], bins=10)
        
        # Calculate accuracy for each architecture
        for arch in available_archs:
            arch_df = df[df['architecture'] == arch]
            grouped = arch_df.groupby('feature_bin')['is_finally_correct'].mean()
            
            # Get bin centers for x-axis
            bin_centers = []
            for interval in grouped.index:
                bin_centers.append((interval.left + interval.right) / 2)
            
            # Plot line
            ax.plot(
                bin_centers,
                grouped.values * 100,
                marker='o',
                linewidth=2,
                markersize=6,
                color=self.color_map.get(arch, '#999999'),
                label=arch,
                alpha=0.8
            )
        
        model_name_map = {
            'qwen_2_5_7b_simplerl_zoo': 'Qwen-2.5-7B-RL'
        }

        # Plot base model points with special markers
        if base_stats:
            markers = ['X', '^', 's']
            colors = ['#D73027', '#56B4E9', '#FEE090']  # Distinct colors
            for i, stat in enumerate(base_stats):
                if 'model' in stat and stat['model'] in model_name_map:
                    stat['model'] = model_name_map[stat['model']]
                ax.scatter(
                    stat['avg_entropy'],
                    stat['avg_acc'],
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    s=120,  # Large size
                    label=f'{stat["model"]} (Base)',
                    edgecolors='white',
                    linewidths=1.5,
                    zorder=5
                )
        
        # Styling
        feature_map = {
            'base_sample_total_entropy': 'Base Model Entropy'
        }
        feature_label = feature_map.get(focus_feature)
        ax.set_xlabel(f'{feature_label}', fontsize=16)
        ax.set_ylabel('Accuracy (%)', fontsize=16)
        ax.text(0.5, -0.25, '(b)', transform=ax.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold')
        ax.legend(loc='best', frameon=False, fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
    
    def plot_top2_entropy_scatter(self, ax):
        """
        Plot scatter of two specified entropy features (Subplot 3).
        Reference: analyze_mas.py plot_top2_entropy_scatter
        Focuses on sample_round_1_median_agent_total_entropy and sample_round_1_q3_agent_total_entropy
        
        Args:
            ax: Matplotlib axis object
        """
        # Load feature values from X_test
        if not self.shap_x_test_path.exists():
            ax.text(0.5, 0.5, 'X_test file not found',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(c)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        x_test = pd.read_csv(self.shap_x_test_path)
        
        # Use instance variables feature1 and feature2
        f1, f2 = self.feature1, self.feature2
        
        # Check if the required features exist in the dataset
        if f1 not in x_test.columns or f2 not in x_test.columns:
            print(f"Warning: Required features not found ({f1}, {f2}).")
            ax.text(0.5, 0.5, f'Required features not found\n({f1}, {f2})',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(c)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Check if architecture column exists
        if 'architecture' not in x_test.columns:
            print(f"Warning: 'architecture' column not found in X_test data")
            ax.text(0.5, 0.5, 'Architecture column not found',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(c)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        # Load prediction probabilities
        if not self.lightgbm_pred_path.exists() or not self.xgboost_pred_path.exists():
            print("Warning: Prediction probability files not found")
            ax.text(0.5, 0.5, 'Prediction files not found',
                   ha='center', va='center', fontsize=12)
            ax.text(0.5, -0.25, '(c)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16, fontweight='bold')
            return
        
        lightgbm_df = pd.read_csv(self.lightgbm_pred_path)
        xgboost_df = pd.read_csv(self.xgboost_pred_path)
        
        # Load prediction probabilities for both classes
        lgbm_prob0 = pd.to_numeric(lightgbm_df['prob_class_0'].values, errors='coerce')
        lgbm_prob1 = pd.to_numeric(lightgbm_df['prob_class_1'].values, errors='coerce')
        xgb_prob0 = pd.to_numeric(xgboost_df['prob_class_0'].values, errors='coerce')
        xgb_prob1 = pd.to_numeric(xgboost_df['prob_class_1'].values, errors='coerce')
        
        # Get feature values and architecture
        x1 = x_test[f1].values
        x2 = x_test[f2].values
        architecture = x_test['architecture'].values

        if f2 == 'round_2_total_entropy':
            x2 = x_test[f2].values / 1000.0
        elif f1 == 'round_2_total_entropy':
            x1 = x_test[f1].values / 1000.0
        
        # Align dimensions
        min_len = min(len(lgbm_prob0), len(xgb_prob0), len(x1), len(x2), len(architecture))
        lgbm_prob0, lgbm_prob1 = lgbm_prob0[:min_len], lgbm_prob1[:min_len]
        xgb_prob0, xgb_prob1 = xgb_prob0[:min_len], xgb_prob1[:min_len]
        x1, x2, architecture = x1[:min_len], x2[:min_len], architecture[:min_len]
        
        # Calculate average prediction probabilities for each class
        mean_prob0 = (lgbm_prob0 + xgb_prob0) / 2.0
        mean_prob1 = (lgbm_prob1 + xgb_prob1) / 2.0
        
        # Handle any NaN values
        mask_valid = ~(np.isnan(mean_prob0) | np.isnan(mean_prob1) | np.isnan(x1) | np.isnan(x2))
        x1, x2 = x1[mask_valid], x2[mask_valid]
        mean_prob0, mean_prob1 = mean_prob0[mask_valid], mean_prob1[mask_valid]
        architecture = architecture[mask_valid]
        
        # Separate positive and negative samples by comparing probabilities
        positive_mask = mean_prob1 > mean_prob0
        negative_mask = mean_prob0 >= mean_prob1
        
        # Separate SAS and MAS
        # SAS: architecture == 4 (single)
        # MAS: architecture in [0, 1, 2, 3] (centralized, debate, hybrid, sequential)
        sas_mask = architecture == 4
        mas_mask = architecture < 4
        
        # Scale point sizes based on the respective class probability
        size_scale = 100
        base_size = 10
        sizes_pos = mean_prob1 * size_scale + base_size
        sizes_neg = mean_prob0 * size_scale + base_size
        
        # Plot negative samples (class 0)
        # SAS negative samples (blue circles)
        mask_sas_neg = negative_mask & sas_mask
        if np.any(mask_sas_neg):
            ax.scatter(x1[mask_sas_neg], x2[mask_sas_neg],
                      s=sizes_neg[mask_sas_neg], c='#4575B4', alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label='SAS Negative', marker='s')
        
        # MAS negative samples (blue triangles)
        mask_mas_neg = negative_mask & mas_mask
        if np.any(mask_mas_neg):
            ax.scatter(x1[mask_mas_neg], x2[mask_mas_neg],
                      s=sizes_neg[mask_mas_neg], c='#91BFD8', alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label='MAS Negative', marker='^')
        
        # Plot positive samples (class 1)
        # SAS positive samples (red circles)
        mask_sas_pos = positive_mask & sas_mask
        if np.any(mask_sas_pos):
            ax.scatter(x1[mask_sas_pos], x2[mask_sas_pos],
                      s=sizes_pos[mask_sas_pos], c='#D73027', alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label='SAS Positive', marker='o')
        
        # MAS positive samples (red triangles)
        mask_mas_pos = positive_mask & mas_mask
        if np.any(mask_mas_pos):
            ax.scatter(x1[mask_mas_pos], x2[mask_mas_pos],
                      s=sizes_pos[mask_mas_pos], c='#FC8D59', alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label='MAS Positive', marker='*')
        
        feature_map = {
            'sample_round_1_median_agent_total_entropy': 'Round 1 Median Agent Entropy',
            'sample_round_1_q3_agent_total_entropy': 'Round 1 Q3 Total Entropy',
            'round_2_total_entropy': 'Round 2 Total Entropy',
            'round_1_2_change_entropy': 'Round 1-2 Change Entropy',
            'base_sample_total_entropy': 'Base Sample Total Entropy'
        }
        
        ax.set_xlabel(f'{feature_map[self.feature1]}', fontsize=17)
        ax.set_ylabel(f'{feature_map[self.feature2]}', fontsize=17)
        ax.text(0.5, -0.25, '(c)', transform=ax.transAxes,
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Add grid
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Add legend
        ax.legend(loc='upper center', frameon=True, fancybox=False,
                 edgecolor='black', framealpha=0.95, fontsize=13)
        
        # Adjust spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def generate_comprehensive_figure(self):
        """
        Generate the comprehensive three-subplot figure.
        """
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        print("Generating subplot 1: Accuracy Bar Chart...")
        self.plot_accuracy_bar(axes[0])
        
        print("Generating subplot 2: Accuracy Trends...")
        self.plot_accuracy_trends(axes[1])
        
        print("Generating subplot 3: Top 2 Entropy Features Scatter...")
        self.plot_top2_entropy_scatter(axes[2])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "rl_model_analysis.pdf"
        plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='pdf')
        print(f"\nComprehensive figure saved to: {output_path}")
        
        plt.close()


def main():
    """Main function to generate the comprehensive visualization."""
    
    # Define paths
    base_dir = '/home/yuxuanzhao/multiagent-entropy'
    
    # Subplot 1 data
    combined_summary_path = f"{base_dir}/evaluation/results_rl/combined_summary_data.csv"
    
    # Subplot 2 data
    shap_dir = f"{base_dir}/data_mining/results/exclude_base_model_wo_entropy/shap"
    
    # Subplot 3 data
    feature_importance_path = f"{base_dir}/data_mining/results_rl/results_aggregated/exclude_base_model_all_metrics.csv"
    shap_x_test_path = f"{base_dir}/data_mining/results_rl/results/exclude_base_model_wo_entropy/shap/X_test_LightGBM_classification.csv"
    shap_values_path = f"{base_dir}/data_mining/results_rl/results/exclude_base_model_wo_entropy/shap/shap_values_LightGBM_classification.csv"
    lightgbm_pred_path = f"{base_dir}/data_mining/results_rl/results/exclude_base_model_wo_entropy/shap/shap_prediction_probabilities_LightGBM_classification.csv"
    xgboost_pred_path = f"{base_dir}/data_mining/results_rl/results/exclude_base_model_wo_entropy/shap/shap_prediction_probabilities_XGBoost_classification.csv"
    
    output_dir = f"{base_dir}/plot/rl_model"
    
    # Define the target features for subplot 3
    top_features = [
        # 'round_1_2_change_entropy',
        'round_2_total_entropy',
        'sample_round_1_median_agent_total_entropy',
        # 'base_sample_total_entropy',
    ]
    
    # Initialize visualizer
    visualizer = RLModelVisualizer(
        combined_summary_path=combined_summary_path,
        shap_dir=shap_dir,
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
