#!/usr/bin/env python3
"""
Dataset Comprehensive Visualization for Multi-Agent System Analysis
This module creates a four-subplot visualization maintaining ICML style consistency.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")


class DatasetVisualizer:
    """Visualizer for cross-dataset analysis with ICML-style formatting."""
    
    # Dataset list
    DATASETS = [
         'aime2025_16384','aime2024_16384', 
         'math500','gsm8k', 'humaneval', 'mmlu', 
    ]
    
    # Dataset display names (shorter for plotting)
    DATASET_DISPLAY_NAMES = {
        'aime2025_16384': 'AIME25',
        'aime2024_16384': 'AIME24',
        'humaneval': 'HE',
        'math500': 'MATH500',
        'mmlu': 'MMLU',
        'gsm8k': 'GSM8K',
    }
    
    # Target features for analysis
    FEATURES = [
        # 'sample_variance_entropy', 
        'sample_round_1_max_agent_total_entropy',
        'exp_infer_average_entropy',
    ]
    
    # Feature display names
    FEATURE_DISPLAY_NAMES = {
        'sample_variance_entropy': 'Variance Entropy',
        'exp_infer_average_entropy': "Average Agent Entropy",
        'sample_round_1_max_agent_total_entropy': "Round 1 Max Agent Entropy",
    }
    
    def __init__(
        self,
        shap_data_root: str,
        accuracy_data_path: str,
        output_dir: str
    ):
        """
        Initialize paths to data files.
        
        Args:
            shap_data_root: Root path to SHAP data directories
            accuracy_data_path: Path to combined_summary_data.csv
            output_dir: Directory to save output figures
        """
        self.shap_data_root = Path(shap_data_root)
        self.accuracy_data_path = Path(accuracy_data_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ICML color palette for datasets
        self.dataset_colors = {
            'aime2024_16384': '#D73027',   # Red
            'aime2025_16384': '#FC8D59',   # Orange
            'gsm8k': '#FEE090',            # Yellow
            'humaneval': '#4575B4',        # Blue
            'math500': '#91BFD8',          # Light Blue
            'mmlu': '#313695'              # Dark Blue
        }
        
        # Markers for different features
        self.feature_markers = {
            # 'sample_variance_entropy': 'o',
            'sample_round_1_max_agent_total_entropy': 'o',
            'exp_infer_average_entropy': 's'
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
        plt.rcParams['legend.fontsize'] = 12
    
    def load_shap_data_for_dataset(self, dataset_name: str):
        """
        Load SHAP values and X_test data for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (shap_values_df, X_test_df) or (None, None) if not found
        """
        shap_dir = self.shap_data_root / f"dataset_{dataset_name}_exclude_base_model_all_metrics" / "shap"
        
        shap_values_path = shap_dir / "shap_values_LightGBM_classification.csv"
        x_test_path = shap_dir / "X_test_LightGBM_classification.csv"
        
        if not shap_values_path.exists() or not x_test_path.exists():
            print(f"Warning: SHAP data not found for {dataset_name}")
            return None, None
        
        try:
            shap_df = pd.read_csv(shap_values_path, index_col='sample_index')
            x_test_df = pd.read_csv(x_test_path, index_col=0)
            return shap_df, x_test_df
        except Exception as e:
            print(f"Error loading SHAP data for {dataset_name}: {e}")
            return None, None
    
    def load_accuracy_data(self):
        """Load accuracy data from combined_summary_data.csv."""
        if not self.accuracy_data_path.exists():
            raise FileNotFoundError(f"Accuracy data not found: {self.accuracy_data_path}")
        
        return pd.read_csv(self.accuracy_data_path)
    
    def plot_shap_scatter(self, ax):
        """
        Plot SHAP value scatter plots for all datasets (Subplot 1).
        
        Colors represent datasets, markers represent features.
        Feature values are normalized to [0, 1] for consistent scale.
        
        Args:
            ax: Matplotlib axis object
        """
        legend_handles = []
        
        # Collect all data points
        all_data = []
        
        for dataset_name in self.DATASETS:
            shap_df, x_test_df = self.load_shap_data_for_dataset(dataset_name)
            
            if shap_df is None or x_test_df is None:
                continue
            
            for feature in self.FEATURES:
                if feature not in shap_df.columns or feature not in x_test_df.columns:
                    print(f"Warning: Feature {feature} not found in {dataset_name}")
                    continue
                
                feature_values = x_test_df[feature].values
                shap_values = shap_df[feature].values
                
                # Store data
                for fv, sv in zip(feature_values, shap_values):
                    all_data.append({
                        'dataset': dataset_name,
                        'feature': feature,
                        'feature_value': fv,
                        'shap_value': sv
                    })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(all_data)
        
        if plot_df.empty:
            ax.text(0.5, 0.5, 'No SHAP data available', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Normalize feature values per feature to [0, 1] for consistent scale
        for feature in self.FEATURES:
            mask = plot_df['feature'] == feature
            if mask.any():
                fv_min = plot_df.loc[mask, 'feature_value'].min()
                fv_max = plot_df.loc[mask, 'feature_value'].max()
                if fv_max > fv_min:
                    plot_df.loc[mask, 'feature_value_norm'] = (
                        plot_df.loc[mask, 'feature_value'] - fv_min
                    ) / (fv_max - fv_min)
                else:
                    plot_df.loc[mask, 'feature_value_norm'] = 0.5
        
        # Create legend handles
        dataset_handles = []
        feature_handles = []
        
        # Plot scatter points using normalized feature values
        for dataset_name in self.DATASETS:
            for feature in self.FEATURES:
                mask = (plot_df['dataset'] == dataset_name) & (plot_df['feature'] == feature)
                subset = plot_df[mask]
                
                if subset.empty:
                    continue
                
                ax.scatter(
                    subset['feature_value_norm'],
                    subset['shap_value'],
                    alpha=0.5,
                    s=25,
                    color=self.dataset_colors[dataset_name],
                    marker=self.feature_markers[feature],
                    edgecolors='white',
                    linewidth=0.3
                )
        
        # Create dataset legend handles
        for dataset_name in self.DATASETS:
            handle = plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=self.dataset_colors[dataset_name],
                               markersize=8, label=self.DATASET_DISPLAY_NAMES[dataset_name])
            dataset_handles.append(handle)
        
        # Create feature legend handles
        for feature in self.FEATURES:
            handle = plt.Line2D([0], [0], marker=self.feature_markers[feature], 
                               color='w', markerfacecolor='gray',
                               markersize=8, label=self.FEATURE_DISPLAY_NAMES[feature])
            feature_handles.append(handle)
        
        # Combine and add single legend (no title)
        all_handles = dataset_handles + feature_handles
        ax.legend(handles=all_handles, loc='lower right', 
                 frameon=False, fontsize=10, ncol=2)
        
        # Styling
        ax.set_xlabel('Normalized Feature Value', fontsize=12)
        ax.set_ylabel('SHAP Value', fontsize=12)
        ax.text(0.5, -0.2, '(a)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
    
    def plot_combined_violin(self, ax):
        """
        Plot combined box plot for both features with MAS accuracy annotation (Subplot b).
        Different colors represent different features.
        
        Args:
            ax: Matplotlib axis object
        """
        # Load accuracy data
        acc_df = self.load_accuracy_data()
        
        # Colors for different features
        feature_colors = {
            'sample_round_1_max_agent_total_entropy': '#91BFD8',      # Blue
            'exp_infer_average_entropy': '#FEE090'     # Red
        }
        
        # Collect data for all features and datasets
        all_data = []
        dataset_accuracies = {}
        
        for dataset_name in self.DATASETS:
            shap_df, x_test_df = self.load_shap_data_for_dataset(dataset_name)
            
            if x_test_df is None:
                continue
            
            # Calculate MAS average accuracy (exclude 'single' architecture)
            dataset_acc = acc_df[
                (acc_df['dataset'] == dataset_name) & 
                (acc_df['architecture'] != 'single')
            ]['accuracy'].mean()
            dataset_accuracies[dataset_name] = dataset_acc
            
            for feature in self.FEATURES:
                if feature not in x_test_df.columns:
                    continue
                
                feature_values = x_test_df[feature].values
                
                # Normalize feature values to [0, 1] for comparison
                fv_min, fv_max = np.nanmin(feature_values), np.nanmax(feature_values)
                if fv_max > fv_min:
                    normalized_values = (feature_values - fv_min) / (fv_max - fv_min)
                else:
                    normalized_values = np.full_like(feature_values, 0.5)
                
                for val in normalized_values:
                    if not np.isnan(val):
                        all_data.append({
                            'dataset': dataset_name,
                            'feature': feature,
                            'value': val
                        })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(all_data)
        
        if plot_df.empty:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Get dataset order
        order = [ds for ds in self.DATASETS if ds in plot_df['dataset'].values]
        
        # Plot box plots for each feature side by side
        width = 0.35
        positions_f1 = np.arange(len(order)) - width/2
        positions_f2 = np.arange(len(order)) + width/2
        
        for fidx, feature in enumerate(self.FEATURES):
            positions = positions_f1 if fidx == 0 else positions_f2
            feature_data = plot_df[plot_df['feature'] == feature]
            
            data_list = []
            valid_positions = []
            for idx, ds in enumerate(order):
                ds_data = feature_data[feature_data['dataset'] == ds]['value'].values
                if len(ds_data) > 0:
                    data_list.append(ds_data)
                    valid_positions.append(positions[idx])
            
            if data_list:
                # Use boxplot with more statistical info
                bp = ax.boxplot(
                    data_list,
                    positions=valid_positions,
                    widths=width * 0.8,
                    patch_artist=True,
                    showfliers=True,      # Show outliers
                    showmeans=True,       # Show mean marker
                    meanprops=dict(marker='D', markerfacecolor='white', 
                                  markeredgecolor='#333333', markersize=4),
                    flierprops=dict(marker='o', markerfacecolor='none',
                                   markeredgecolor=feature_colors[feature],
                                   markersize=3, alpha=0.5)
                )
                
                # Color the boxes
                for box in bp['boxes']:
                    box.set_facecolor(feature_colors[feature])
                    box.set_alpha(0.7)
                    box.set_edgecolor('#333333')
                
                # Style other elements
                for element in ['whiskers', 'caps']:
                    for item in bp[element]:
                        item.set_color('#333333')
                        item.set_linewidth(1)
                
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(1.0)
        
        # Add accuracy annotations above each dataset group (no shadow)
        for idx, ds in enumerate(order):
            acc = dataset_accuracies.get(ds, 0)
            ax.text(idx, 1.05, f'{acc:.0%}', 
                   ha='center', va='bottom', fontsize=9, 
                   fontweight='bold', color='#1a1a1a')
        
        # Set x-axis labels (no rotation)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([self.DATASET_DISPLAY_NAMES[ds] for ds in order], 
                          rotation=0, ha='center', fontsize=11)
        
        # Create feature legend (no title) - placed at lower left to avoid overlap with accuracy labels
        feature_handles = [
            plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=feature_colors[f], markersize=10,
                      label=self.FEATURE_DISPLAY_NAMES[f])
            for f in self.FEATURES
        ]
        # Add mean marker explanation
        mean_handle = plt.Line2D([0], [0], marker='D', color='w',
                                markerfacecolor='white', markeredgecolor='#333333',
                                markersize=5, label='Mean')
        feature_handles.append(mean_handle)
        
        ax.legend(handles=feature_handles, loc='upper left', bbox_to_anchor=(0.28, 0.88), frameon=False, fontsize=9)
        
        # Styling
        ax.set_ylabel('Normalized Feature Value', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.text(0.5, -0.2, '(b)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
    
    def plot_dual_feature_scatter(self, ax):
        """
        Plot scatter of two features with dataset and sample type encoding (Subplot 4).
        
        Args:
            ax: Matplotlib axis object
        """
        # Collect all data
        all_data = []
        
        for dataset_name in self.DATASETS:
            shap_dir = self.shap_data_root / f"dataset_{dataset_name}_exclude_base_model_all_metrics"
            
            # Load X_test
            x_test_path = shap_dir / "shap" / "X_test_LightGBM_classification.csv"
            
            if not x_test_path.exists():
                continue
            
            try:
                x_test_df = pd.read_csv(x_test_path, index_col=0)
            except Exception as e:
                print(f"Error loading X_test for {dataset_name}: {e}")
                continue
            
            f1, f2 = self.FEATURES[0], self.FEATURES[1]
            
            if f1 not in x_test_df.columns or f2 not in x_test_df.columns:
                continue
            
            # Load prediction probabilities
            lgbm_pred_path = shap_dir / "shap" / "lightgbm_predictions.csv"
            
            if lgbm_pred_path.exists():
                try:
                    lgbm_df = pd.read_csv(lgbm_pred_path)
                    prob0 = lgbm_df['prob_class_0'].values
                    prob1 = lgbm_df['prob_class_1'].values
                    positive_mask = prob1 > prob0
                except:
                    # Default: all positive
                    positive_mask = np.ones(len(x_test_df), dtype=bool)
            else:
                # Default: all positive
                positive_mask = np.ones(len(x_test_df), dtype=bool)
            
            # Add to data
            for idx in range(min(len(x_test_df), len(positive_mask))):
                all_data.append({
                    'dataset': dataset_name,
                    'f1': x_test_df[f1].iloc[idx],
                    'f2': x_test_df[f2].iloc[idx],
                    'is_positive': positive_mask[idx]
                })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(all_data)
        
        if plot_df.empty:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', fontsize=12)
            return
        
        # Plot scatter points
        for dataset_name in self.DATASETS:
            subset = plot_df[plot_df['dataset'] == dataset_name]
            
            if subset.empty:
                continue
            
            # Positive samples (circles)
            pos_subset = subset[subset['is_positive']]
            if not pos_subset.empty:
                ax.scatter(
                    pos_subset['f1'], pos_subset['f2'],
                    color=self.dataset_colors[dataset_name],
                    marker='o', s=30, alpha=0.6,
                    edgecolors='white', linewidth=0.5,
                    label=f'{self.DATASET_DISPLAY_NAMES[dataset_name]} (Pos)'
                )
            
            # Negative samples (squares)
            neg_subset = subset[~subset['is_positive']]
            if not neg_subset.empty:
                ax.scatter(
                    neg_subset['f1'], neg_subset['f2'],
                    color=self.dataset_colors[dataset_name],
                    marker='s', s=30, alpha=0.6,
                    edgecolors='white', linewidth=0.5,
                    label=f'{self.DATASET_DISPLAY_NAMES[dataset_name]} (Neg)'
                )
        
        # Create custom legend
        # Dataset legend
        dataset_handles = []
        for ds in self.DATASETS:
            if ds in plot_df['dataset'].values:
                handle = plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=self.dataset_colors[ds],
                                   markersize=8, label=self.DATASET_DISPLAY_NAMES[ds])
                dataset_handles.append(handle)
        
        # Sample type legend
        type_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='gray', markersize=8, label='Positive'),
            plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor='gray', markersize=8, label='Negative')
        ]
        
        # Combine and add single legend (no title)
        all_handles = dataset_handles + type_handles
        ax.legend(handles=all_handles, loc='center right', 
                 frameon=False, fontsize=9, ncol=1,
                 bbox_to_anchor=(1.0, 0.5))
        
        # Styling
        ax.set_xlabel(self.FEATURE_DISPLAY_NAMES[self.FEATURES[0]], fontsize=12)
        ax.set_ylabel(self.FEATURE_DISPLAY_NAMES[self.FEATURES[1]], fontsize=12)
        ax.text(0.5, -0.2, '(c)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        sns.despine(ax=ax, top=True, right=True)
    
    def create_comprehensive_figure(self, filename='dataset_analysis.pdf'):
        """
        Create the comprehensive three-subplot figure.
        
        Args:
            filename: Output filename
        """
        # Create figure with 1x3 layout (all subplots in one row)
        fig = plt.figure(figsize=(6, 10))
        
        # Create grid spec for custom layout (1 row, 3 columns)
        gs = fig.add_gridspec(2, 1, hspace=0.25)
        
        # Create subplots in a single row
        ax1 = fig.add_subplot(gs[0, 0])  # SHAP scatter
        ax2 = fig.add_subplot(gs[1, 0])  # Combined violin for both features
        # ax3 = fig.add_subplot(gs[0, 2])  # Dual feature scatter
        
        # Plot each subplot
        print("Plotting SHAP scatter (a)...")
        self.plot_shap_scatter(ax1)
        
        print("Plotting combined violin (b)...")
        self.plot_combined_violin(ax2)
        
        # print("Plotting dual feature scatter (c)...")
        # self.plot_dual_feature_scatter(ax3)
        
        # Save figure
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nFigure saved to: {output_path}")
        
        # Also save high-resolution PDF
        pdf_path = self.output_dir / filename.replace('.pdf', '_hires.pdf')
        fig.savefig(pdf_path, dpi=1200, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"High-resolution PDF saved to: {pdf_path}")
        
        plt.close(fig)
        
        return output_path


def main():
    """Main function to generate the visualization."""
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    
    shap_data_root = base_dir / "data_mining" / "results_all" / "results"
    accuracy_data_path = base_dir / "evaluation" / "results_all" / "combined_summary_data.csv"
    output_dir = base_dir / "plot" / "datasets" / "output"
    
    # Create visualizer
    visualizer = DatasetVisualizer(
        shap_data_root=str(shap_data_root),
        accuracy_data_path=str(accuracy_data_path),
        output_dir=str(output_dir)
    )
    
    # Generate comprehensive figure
    output_path = visualizer.create_comprehensive_figure('dataset_comprehensive_analysis.pdf')
    
    print(f"\nVisualization complete!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
