"""Exploratory Data Analysis Module for Multi-Agent Entropy Research.

This module provides comprehensive EDA including:
- Entropy dynamics across different multi-agent architectures (single, centralized, debate)
- Correlation analysis between entropy and system performance metrics
- Statistical summaries and visualizations (box plots, scatter plots, heatmaps)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class ExploratoryDataAnalyzer:
    """Class for performing exploratory data analysis on multi-agent entropy experiments."""

    def __init__(self, data_path: str, output_dir: str):
        """Initialize the ExploratoryDataAnalyzer.

        Args:
            data_path: Path to the CSV data file.
            output_dir: Directory to save output files and figures.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.numeric_columns = None
        self.categorical_columns = None

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'results', 'eda_results')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file.

        Returns:
            Loaded DataFrame.
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def identify_column_types(self) -> None:
        """Identify numeric and categorical columns."""
        self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include=['object']).columns.tolist()

    def analyze_architecture_distribution(self) -> None:
        """Analyze distribution of different multi-agent architectures."""
        if 'architecture' not in self.df.columns:
            print("Warning: 'architecture' column not found in dataset")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Architecture count distribution
        ax1 = axes[0, 0]
        arch_counts = self.df['architecture'].value_counts()
        ax1.bar(arch_counts.index, arch_counts.values, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Architecture Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Multi-Agent Architectures')
        ax1.tick_params(axis='x', rotation=45)

        # Architecture percentage
        ax2 = axes[0, 1]
        arch_pct = self.df['architecture'].value_counts(normalize=True) * 100
        colors = plt.cm.Set3(np.linspace(0, 1, len(arch_pct)))
        ax2.pie(arch_pct.values, labels=arch_pct.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax2.set_title('Architecture Distribution Percentage')

        # Accuracy by architecture
        ax3 = axes[1, 0]
        if 'is_correct' in self.df.columns:
            arch_accuracy = self.df.groupby('architecture')['is_correct'].mean() * 100
            ax3.bar(arch_accuracy.index, arch_accuracy.values, color='coral', alpha=0.7)
            ax3.set_xlabel('Architecture Type')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_title('Agent-level Task Completion Rate by Architecture')
            ax3.tick_params(axis='x', rotation=45)
            for i, v in enumerate(arch_accuracy.values):
                ax3.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

        # Final accuracy by architecture
        ax4 = axes[1, 1]
        if 'is_finally_correct' in self.df.columns:
            arch_final_accuracy = self.df.groupby('architecture')['is_finally_correct'].mean() * 100
            ax4.bar(arch_final_accuracy.index, arch_final_accuracy.values, color='lightgreen', alpha=0.7)
            ax4.set_xlabel('Architecture Type')
            ax4.set_ylabel('Final Accuracy (%)')
            ax4.set_title('Final Task Completion Rate by Architecture')
            ax4.tick_params(axis='x', rotation=45)
            for i, v in enumerate(arch_final_accuracy.values):
                ax4.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
        elif 'time_cost' in self.df.columns:
            arch_time = self.df.groupby('architecture')['time_cost'].mean()
            ax4.bar(arch_time.index, arch_time.values, color='lightgreen', alpha=0.7)
            ax4.set_xlabel('Architecture Type')
            ax4.set_ylabel('Average Time Cost (seconds)')
            ax4.set_title('Average Response Time by Architecture')
            ax4.tick_params(axis='x', rotation=45)
            for i, v in enumerate(arch_time.values):
                ax4.text(i, v + 0.5, f'{v:.1f}s', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'architecture_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save statistics
        arch_stats = pd.DataFrame({
            'count': arch_counts,
            'percentage': arch_pct
        })
        arch_stats.to_csv(os.path.join(self.results_dir, 'architecture_statistics.csv'))

        print(f"\nArchitecture Distribution:")
        print(arch_stats.to_string())

    def analyze_entropy_by_architecture(self) -> None:
        """Analyze entropy dynamics across different architectures."""
        entropy_cols = [col for col in self.df.columns if 'entropy' in col.lower()]

        if not entropy_cols:
            print("Warning: No entropy columns found in dataset")
            return

        if 'architecture' not in self.df.columns:
            print("Warning: 'architecture' column not found in dataset")
            return

        # Select key entropy columns
        key_entropy_cols = ['sample_mean_entropy', 'sample_total_entropy',
                           'agent_mean_entropy', 'exp_avg_entropy']
        key_entropy_cols = [col for col in key_entropy_cols if col in self.df.columns]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, col in enumerate(key_entropy_cols):
            if idx >= len(axes):
                break

            ax = axes[idx]
            for arch in self.df['architecture'].unique():
                arch_data = self.df[self.df['architecture'] == arch][col].dropna()
                ax.hist(arch_data, bins=50, alpha=0.5, label=arch, density=True)

            ax.set_xlabel('Entropy Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {col} by Architecture')
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'entropy_distribution_by_architecture.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Box plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, col in enumerate(key_entropy_cols):
            if idx >= len(axes):
                break

            ax = axes[idx]
            self.df.boxplot(column=col, by='architecture', ax=ax)
            ax.set_title(f'{col} by Architecture')
            ax.set_xlabel('Architecture')
            ax.set_ylabel('Entropy Value')

        plt.suptitle('Box Plots of Entropy Metrics by Architecture', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'entropy_boxplot_by_architecture.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Statistical summary
        entropy_stats = []
        for arch in self.df['architecture'].unique():
            arch_df = self.df[self.df['architecture'] == arch]
            for col in key_entropy_cols:
                if col in arch_df.columns:
                    stats_dict = {
                        'architecture': arch,
                        'entropy_metric': col,
                        'mean': arch_df[col].mean(),
                        'std': arch_df[col].std(),
                        'median': arch_df[col].median(),
                        'min': arch_df[col].min(),
                        'max': arch_df[col].max(),
                        'count': arch_df[col].count()
                    }
                    entropy_stats.append(stats_dict)

        entropy_stats_df = pd.DataFrame(entropy_stats)
        entropy_stats_df.to_csv(os.path.join(self.results_dir, 'entropy_statistics_by_architecture.csv'),
                                index=False)

        print(f"\nEntropy Statistics by Architecture:")
        print(entropy_stats_df.to_string(index=False))

    def analyze_entropy_performance_correlation(self) -> None:
        """Analyze correlation between entropy and performance metrics."""
        entropy_cols = [col for col in self.df.columns if 'entropy' in col.lower()]
        performance_cols = ['is_correct', 'is_finally_correct', 'time_cost', 'ground_truth']

        entropy_cols = [col for col in entropy_cols if col in self.df.columns]
        performance_cols = [col for col in performance_cols if col in self.df.columns]

        if not entropy_cols or not performance_cols:
            print("Warning: Insufficient columns for correlation analysis")
            return

        # Calculate correlation matrix
        correlation_cols = entropy_cols + performance_cols
        corr_matrix = self.df[correlation_cols].corr()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix: Entropy vs Performance Metrics', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'entropy_performance_correlation.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save correlation matrix
        corr_matrix.to_csv(os.path.join(self.results_dir, 'correlation_matrix.csv'))

        print(f"\nCorrelation Matrix:")
        print(corr_matrix.to_string())

        # Scatter plots for key correlations
        key_entropy_cols = ['sample_mean_entropy', 'sample_total_entropy']
        key_entropy_cols = [col for col in key_entropy_cols if col in self.df.columns]

        fig, axes = plt.subplots(len(key_entropy_cols), 3, figsize=(20, 6 * len(key_entropy_cols)))

        if len(key_entropy_cols) == 1:
            axes = axes.reshape(1, -1)

        for idx, entropy_col in enumerate(key_entropy_cols):
            # Entropy vs Agent-level Accuracy
            ax1 = axes[idx, 0]
            if 'is_correct' in self.df.columns:
                for arch in self.df['architecture'].unique():
                    arch_df = self.df[self.df['architecture'] == arch]
                    ax1.scatter(arch_df[entropy_col], arch_df['is_correct'],
                               alpha=0.5, label=arch, s=20)
                ax1.set_xlabel(entropy_col)
                ax1.set_ylabel('Is Correct (Agent-level)')
                ax1.set_title(f'{entropy_col} vs Agent-level Accuracy')
                ax1.legend()

            # Entropy vs Final Accuracy
            ax2 = axes[idx, 1]
            if 'is_finally_correct' in self.df.columns:
                for arch in self.df['architecture'].unique():
                    arch_df = self.df[self.df['architecture'] == arch]
                    ax2.scatter(arch_df[entropy_col], arch_df['is_finally_correct'],
                               alpha=0.5, label=arch, s=20)
                ax2.set_xlabel(entropy_col)
                ax2.set_ylabel('Is Finally Correct')
                ax2.set_title(f'{entropy_col} vs Final Accuracy')
                ax2.legend()

            # Entropy vs Time Cost
            ax3 = axes[idx, 2]
            if 'time_cost' in self.df.columns:
                for arch in self.df['architecture'].unique():
                    arch_df = self.df[self.df['architecture'] == arch]
                    ax3.scatter(arch_df[entropy_col], arch_df['time_cost'],
                               alpha=0.5, label=arch, s=20)
                ax3.set_xlabel(entropy_col)
                ax3.set_ylabel('Time Cost (seconds)')
                ax3.set_title(f'{entropy_col} vs Response Time')
                ax3.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'entropy_performance_scatter.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_correctness_comparison(self) -> None:
        """Analyze the relationship between agent-level and final correctness."""
        if 'is_correct' not in self.df.columns or 'is_finally_correct' not in self.df.columns:
            print("Warning: 'is_correct' or 'is_finally_correct' columns not found")
            return

        print("\n" + "=" * 80)
        print("ANALYZING AGENT-LEVEL VS FINAL CORRECTNESS")
        print("=" * 80)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # Overall correctness rates
        ax1 = axes[0, 0]
        rates = [self.df['is_correct'].mean(), self.df['is_finally_correct'].mean()]
        ax1.bar(['Agent-level', 'Final'], rates, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax1.set_ylabel('Correct Rate')
        ax1.set_title('Overall Correctness Rate Comparison')
        ax1.set_ylim([0, 1])
        for i, v in enumerate(rates):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Correctness by architecture
        ax2 = axes[0, 1]
        arch_comparison = self.df.groupby('architecture').agg({
            'is_correct': 'mean',
            'is_finally_correct': 'mean'
        }) * 100
        x = np.arange(len(arch_comparison))
        width = 0.35
        ax2.bar(x - width/2, arch_comparison['is_correct'], width, label='Agent-level', alpha=0.8)
        ax2.bar(x + width/2, arch_comparison['is_finally_correct'], width, label='Final', alpha=0.8)
        ax2.set_xlabel('Architecture')
        ax2.set_ylabel('Correct Rate (%)')
        ax2.set_title('Correctness Rate by Architecture')
        ax2.set_xticks(x)
        ax2.set_xticklabels(arch_comparison.index, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Confusion matrix-like visualization
        ax3 = axes[0, 2]
        cm_data = np.zeros((2, 2))
        cm_data[0, 0] = ((self.df['is_correct'] == False) & 
                         (self.df['is_finally_correct'] == False)).sum()
        cm_data[0, 1] = ((self.df['is_correct'] == False) & 
                         (self.df['is_finally_correct'] == True)).sum()
        cm_data[1, 0] = ((self.df['is_correct'] == True) & 
                         (self.df['is_finally_correct'] == False)).sum()
        cm_data[1, 1] = ((self.df['is_correct'] == True) & 
                         (self.df['is_finally_correct'] == True)).sum()

        sns.heatmap(cm_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax3,
                   xticklabels=['Final Wrong', 'Final Correct'],
                   yticklabels=['Agent Wrong', 'Agent Correct'])
        ax3.set_xlabel('Final Correctness')
        ax3.set_ylabel('Agent Correctness')
        ax3.set_title('Agent vs Final Correctness Matrix')

        # Correctness vs entropy
        ax4 = axes[1, 0]
        if 'sample_mean_entropy' in self.df.columns:
            for correctness in [False, True]:
                data = self.df[self.df['is_finally_correct'] == correctness]['sample_mean_entropy']
                ax4.hist(data, bins=50, alpha=0.5, label='Final Correct' if correctness else 'Final Wrong',
                        density=True)
            ax4.set_xlabel('Sample Mean Entropy')
            ax4.set_ylabel('Density')
            ax4.set_title('Entropy Distribution by Final Correctness')
            ax4.legend()

        # Improvement analysis
        ax5 = axes[1, 1]
        improvement_counts = {
            'Agent Correct -> Final Correct': cm_data[1, 1],
            'Agent Wrong -> Final Correct': cm_data[0, 1],
            'Agent Correct -> Final Wrong': cm_data[1, 0],
            'Agent Wrong -> Final Wrong': cm_data[0, 0]
        }
        ax5.barh(list(improvement_counts.keys()), list(improvement_counts.values()),
                color=['#2ecc71', '#27ae60', '#e74c3c', '#c0392b'], alpha=0.8)
        ax5.set_xlabel('Count')
        ax5.set_title('Correctness Transition Counts')
        ax5.grid(True, alpha=0.3)

        # Agreement rate by architecture
        ax6 = axes[1, 2]
        if 'architecture' in self.df.columns:
            agreement = self.df.groupby('architecture').apply(
                lambda x: (x['is_correct'] == x['is_finally_correct']).mean()
            )
            ax6.bar(agreement.index, agreement.values, color='steelblue', alpha=0.7)
            ax6.set_xlabel('Architecture')
            ax6.set_ylabel('Agreement Rate')
            ax6.set_title('Agent-Final Agreement Rate by Architecture')
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correctness_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Save statistics
        comparison_stats = pd.DataFrame({
            'Metric': [
                'Agent-level Correct Rate',
                'Final Correct Rate',
                'Agreement Rate',
                'Agent Correct / Final Wrong Count',
                'Agent Wrong / Final Correct Count',
                'Total Samples'
            ],
            'Value': [
                self.df['is_correct'].mean(),
                self.df['is_finally_correct'].mean(),
                (self.df['is_correct'] == self.df['is_finally_correct']).mean(),
                cm_data[1, 0],
                cm_data[0, 1],
                len(self.df)
            ]
        })
        comparison_stats.to_csv(os.path.join(self.results_dir, 'correctness_comparison.csv'),
                               index=False)

        print(f"\nCorrectness Comparison Statistics:")
        print(comparison_stats.to_string(index=False))

    def analyze_entropy_dynamics_over_rounds(self) -> None:
        """Analyze entropy dynamics across execution rounds."""
        if 'execution_order' not in self.df.columns:
            print("Warning: 'execution_order' column not found in dataset")
            return

        entropy_cols = ['sample_mean_entropy', 'sample_total_entropy']
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]

        if not entropy_cols:
            print("Warning: No entropy columns found for round analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, entropy_col in enumerate(entropy_cols):
            if idx >= len(axes):
                break

            ax = axes[idx]
            for arch in self.df['architecture'].unique():
                arch_df = self.df[self.df['architecture'] == arch]
                round_stats = arch_df.groupby('execution_order')[entropy_col].agg(['mean', 'std'])
                ax.errorbar(round_stats.index, round_stats['mean'], yerr=round_stats['std'],
                           label=arch, marker='o', capsize=5, linewidth=2)

            ax.set_xlabel('Execution Round')
            ax.set_ylabel('Entropy Value')
            ax.set_title(f'{entropy_col} Dynamics Across Rounds')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Line plots for individual samples
        for idx, entropy_col in enumerate(entropy_cols):
            if idx + 2 >= len(axes):
                break

            ax = axes[idx + 2]
            sample_ids = self.df['sample_id'].unique()[:20]

            for sample_id in sample_ids:
                sample_df = self.df[self.df['sample_id'] == sample_id]
                if len(sample_df) > 1:
                    ax.plot(sample_df['execution_order'], sample_df[entropy_col],
                           alpha=0.3, linewidth=1)

            ax.set_xlabel('Execution Round')
            ax.set_ylabel('Entropy Value')
            ax.set_title(f'{entropy_col} Dynamics for Sample Trajectories (n=20)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'entropy_dynamics_over_rounds.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_agent_entropy_patterns(self) -> None:
        """Analyze entropy patterns across different agents."""
        if 'agent_name' not in self.df.columns:
            print("Warning: 'agent_name' column not found in dataset")
            return

        entropy_cols = ['agent_mean_entropy', 'agent_total_entropy']
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]

        if not entropy_cols:
            print("Warning: No agent entropy columns found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, entropy_col in enumerate(entropy_cols):
            if idx >= len(axes):
                break

            ax = axes[idx]
            agent_stats = self.df.groupby('agent_name')[entropy_col].mean().sort_values(ascending=False)
            ax.barh(range(len(agent_stats)), agent_stats.values, color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(agent_stats)))
            ax.set_yticklabels(agent_stats.index, fontsize=8)
            ax.set_xlabel('Mean Entropy Value')
            ax.set_title(f'Average {entropy_col} by Agent')
            ax.invert_yaxis()

        # Box plots by agent
        for idx, entropy_col in enumerate(entropy_cols):
            if idx + 2 >= len(axes):
                break

            ax = axes[idx + 2]
            self.df.boxplot(column=entropy_col, by='agent_name', ax=ax, rot=90)
            ax.set_title(f'{entropy_col} Distribution by Agent')
            ax.set_xlabel('Agent Name')
            ax.set_ylabel('Entropy Value')

        plt.suptitle('Agent Entropy Patterns', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'agent_entropy_patterns.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Agent statistics
        agent_stats = []
        for agent in self.df['agent_name'].unique():
            agent_df = self.df[self.df['agent_name'] == agent]
            for col in entropy_cols:
                if col in agent_df.columns:
                    stats_dict = {
                        'agent_name': agent,
                        'entropy_metric': col,
                        'mean': agent_df[col].mean(),
                        'std': agent_df[col].std(),
                        'count': agent_df[col].count()
                    }
                    agent_stats.append(stats_dict)

        agent_stats_df = pd.DataFrame(agent_stats)
        agent_stats_df.to_csv(os.path.join(self.results_dir, 'agent_entropy_statistics.csv'),
                              index=False)

        print(f"\nAgent Entropy Statistics:")
        print(agent_stats_df.to_string(index=False))

    def generate_statistical_summary(self) -> None:
        """Generate comprehensive statistical summary."""
        summary_path = os.path.join(self.results_dir, 'eda_summary.txt')

        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total Rows: {self.df.shape[0]}\n")
            f.write(f"Total Columns: {self.df.shape[1]}\n\n")

            f.write("-" * 80 + "\n")
            f.write("ARCHITECTURE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            if 'architecture' in self.df.columns:
                arch_counts = self.df['architecture'].value_counts()
                f.write(f"Number of architectures: {len(arch_counts)}\n")
                f.write(f"Architecture distribution:\n")
                for arch, count in arch_counts.items():
                    pct = (count / len(self.df)) * 100
                    f.write(f"  {arch}: {count} ({pct:.2f}%)\n")

                if 'is_correct' in self.df.columns:
                    f.write(f"\nAccuracy by architecture:\n")
                    arch_accuracy = self.df.groupby('architecture')['is_correct'].mean() * 100
                    for arch, acc in arch_accuracy.items():
                        f.write(f"  {arch}: {acc:.2f}%\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("ENTROPY ANALYSIS\n")
            f.write("-" * 80 + "\n")
            entropy_cols = [col for col in self.df.columns if 'entropy' in col.lower()]
            if entropy_cols:
                f.write(f"Number of entropy columns: {len(entropy_cols)}\n")
                f.write(f"Entropy columns: {', '.join(entropy_cols)}\n")

                if 'sample_mean_entropy' in self.df.columns:
                    f.write(f"\nSample Mean Entropy Statistics:\n")
                    f.write(f"  Mean: {self.df['sample_mean_entropy'].mean():.4f}\n")
                    f.write(f"  Std: {self.df['sample_mean_entropy'].std():.4f}\n")
                    f.write(f"  Min: {self.df['sample_mean_entropy'].min():.4f}\n")
                    f.write(f"  Max: {self.df['sample_mean_entropy'].max():.4f}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            if 'is_correct' in self.df.columns:
                f.write(f"Agent-level Accuracy: {self.df['is_correct'].mean() * 100:.2f}%\n")
            if 'is_finally_correct' in self.df.columns:
                f.write(f"Final Accuracy: {self.df['is_finally_correct'].mean() * 100:.2f}%\n")
                if 'is_correct' in self.df.columns:
                    agreement = (self.df['is_correct'] == self.df['is_finally_correct']).mean()
                    f.write(f"Agent-Final Agreement Rate: {agreement * 100:.2f}%\n")
                    improvement = ((self.df['is_correct'] == False) & 
                                 (self.df['is_finally_correct'] == True)).sum()
                    f.write(f"Improvement Cases (Wrong -> Correct): {improvement}\n")
                    degradation = ((self.df['is_correct'] == True) & 
                                 (self.df['is_finally_correct'] == False)).sum()
                    f.write(f"Degradation Cases (Correct -> Wrong): {degradation}\n")
            if 'time_cost' in self.df.columns:
                f.write(f"Average Time Cost: {self.df['time_cost'].mean():.2f} seconds\n")
                f.write(f"  Std: {self.df['time_cost'].std():.2f} seconds\n")
                f.write(f"  Min: {self.df['time_cost'].min():.2f} seconds\n")
                f.write(f"  Max: {self.df['time_cost'].max():.2f} seconds\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\nEDA summary report saved to: {summary_path}")

    def run_full_eda(self) -> None:
        """Run complete exploratory data analysis."""
        print("=" * 80)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("=" * 80)

        self.load_data()
        self.identify_column_types()

        print("\n" + "=" * 80)
        print("ARCHITECTURE DISTRIBUTION ANALYSIS")
        print("=" * 80)
        self.analyze_architecture_distribution()

        print("\n" + "=" * 80)
        print("ENTROPY ANALYSIS BY ARCHITECTURE")
        print("=" * 80)
        self.analyze_entropy_by_architecture()

        print("\n" + "=" * 80)
        print("ENTROPY-PERFORMANCE CORRELATION ANALYSIS")
        print("=" * 80)
        self.analyze_entropy_performance_correlation()

        print("\n" + "=" * 80)
        print("CORRECTNESS COMPARISON ANALYSIS")
        print("=" * 80)
        self.analyze_correctness_comparison()

        print("\n" + "=" * 80)
        print("ENTROPY DYNAMICS OVER ROUNDS")
        print("=" * 80)
        self.analyze_entropy_dynamics_over_rounds()

        print("\n" + "=" * 80)
        print("AGENT ENTROPY PATTERNS")
        print("=" * 80)
        self.analyze_agent_entropy_patterns()

        print("\n" + "=" * 80)
        print("GENERATING STATISTICAL SUMMARY")
        print("=" * 80)
        self.generate_statistical_summary()

        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")


def main():
    """Main function to run exploratory data analysis."""
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    analyzer = ExploratoryDataAnalyzer(data_path, output_dir)
    analyzer.run_full_eda()


if __name__ == '__main__':
    main()
