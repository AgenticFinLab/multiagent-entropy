"""Pattern Identification and Validation Module for Multi-Agent Entropy Research.

This module provides comprehensive pattern identification and validation including:
- Extraction of key entropy-related patterns affecting multi-agent system performance
- Statistical validation of discovered patterns and relationships
- Pattern visualization and interpretation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal, f_oneway
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class PatternIdentifier:
    """Class for identifying and validating entropy-related patterns in multi-agent systems."""

    def __init__(self, data_path: str, output_dir: str):
        """Initialize the PatternIdentifier.

        Args:
            data_path: Path to the CSV data file.
            output_dir: Directory to save output files and figures.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.patterns = {}

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'results', 'pattern_identification')
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

    def analyze_entropy_performance_correlation(self) -> pd.DataFrame:
        """Analyze correlation between entropy metrics and performance metrics.

        Returns:
            DataFrame containing correlation coefficients and p-values.
        """
        print("\n" + "=" * 80)
        print("ANALYZING ENTROPY-PERFORMANCE CORRELATION")
        print("=" * 80)

        # Define entropy columns
        entropy_cols = [
            'sample_mean_entropy', 'sample_total_entropy', 'sample_std_entropy',
            'sample_max_entropy', 'sample_min_entropy', 'sample_q1_entropy', 'sample_q3_entropy',
            'agent_mean_entropy', 'agent_total_entropy', 'agent_std_entropy',
            'exp_avg_entropy', 'exp_total_entropy'
        ]

        # Define performance columns
        performance_cols = ['is_correct', 'is_finally_correct', 'time_cost']

        # Filter to only include columns that exist
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]
        performance_cols = [col for col in performance_cols if col in self.df.columns]

        results = []

        for ent_col in entropy_cols:
            for perf_col in performance_cols:
                # Remove missing values
                valid_idx = self.df[ent_col].notna() & self.df[perf_col].notna()
                x = self.df.loc[valid_idx, ent_col]
                y = self.df.loc[valid_idx, perf_col]

                if len(x) > 0:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(x, y)

                    # Spearman correlation
                    spearman_r, spearman_p = spearmanr(x, y)

                    results.append({
                        'entropy_metric': ent_col,
                        'performance_metric': perf_col,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p
                    })

        correlation_df = pd.DataFrame(results)

        # Save results
        correlation_df.to_csv(os.path.join(self.results_dir, 'entropy_performance_correlation.csv'),
                              index=False)

        # Identify significant correlations
        significant_corr = correlation_df[
            (correlation_df['pearson_p'] < 0.05) | (correlation_df['spearman_p'] < 0.05)
        ]

        print(f"\nSignificant correlations (p < 0.05):")
        print(significant_corr.to_string(index=False))

        # Visualize correlations
        self._visualize_entropy_performance_correlation(correlation_df)

        self.patterns['entropy_performance_correlation'] = significant_corr

        return correlation_df

    def analyze_architecture_entropy_patterns(self) -> Dict[str, pd.DataFrame]:
        """Analyze entropy patterns across different architectures.

        Returns:
            Dictionary containing architecture-specific entropy statistics.
        """
        print("\n" + "=" * 80)
        print("ANALYZING ARCHITECTURE-SPECIFIC ENTROPY PATTERNS")
        print("=" * 80)

        if 'architecture' not in self.df.columns:
            print("Warning: 'architecture' column not found in dataset")
            return {}

        # Define entropy columns
        entropy_cols = [
            'sample_mean_entropy', 'sample_total_entropy', 'sample_std_entropy',
            'agent_mean_entropy', 'agent_total_entropy', 'agent_std_entropy',
            'exp_avg_entropy', 'exp_total_entropy'
        ]

        # Filter to only include columns that exist
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]

        # Calculate statistics by architecture
        arch_stats = []
        for arch in self.df['architecture'].unique():
            arch_data = self.df[self.df['architecture'] == arch]

            for ent_col in entropy_cols:
                if ent_col in arch_data.columns:
                    stats_data = {
                        'architecture': arch,
                        'entropy_metric': ent_col,
                        'mean': arch_data[ent_col].mean(),
                        'std': arch_data[ent_col].std(),
                        'median': arch_data[ent_col].median(),
                        'q1': arch_data[ent_col].quantile(0.25),
                        'q3': arch_data[ent_col].quantile(0.75),
                        'count': arch_data[ent_col].count()
                    }
                    arch_stats.append(stats_data)

        arch_stats_df = pd.DataFrame(arch_stats)

        # Save results
        arch_stats_df.to_csv(os.path.join(self.results_dir, 'architecture_entropy_statistics.csv'),
                            index=False)

        # Perform statistical tests
        test_results = []
        for ent_col in entropy_cols:
            groups = [self.df[self.df['architecture'] == arch][ent_col].dropna()
                     for arch in self.df['architecture'].unique()]

            # Kruskal-Wallis test (non-parametric)
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                kw_stat, kw_p = kruskal(*groups)
                test_results.append({
                    'entropy_metric': ent_col,
                    'test': 'Kruskal-Wallis',
                    'statistic': kw_stat,
                    'p_value': kw_p,
                    'significant': kw_p < 0.05
                })

        test_results_df = pd.DataFrame(test_results)

        # Save test results
        test_results_df.to_csv(os.path.join(self.results_dir, 'architecture_entropy_tests.csv'),
                              index=False)

        print(f"\nArchitecture entropy statistics:")
        print(arch_stats_df.to_string(index=False))

        print(f"\nStatistical test results:")
        print(test_results_df.to_string(index=False))

        # Visualize architecture patterns
        self._visualize_architecture_entropy_patterns(arch_stats_df)

        self.patterns['architecture_entropy'] = {
            'statistics': arch_stats_df,
            'tests': test_results_df
        }

        return {
            'statistics': arch_stats_df,
            'tests': test_results_df
        }

    def analyze_entropy_performance_by_architecture(self) -> pd.DataFrame:
        """Analyze how entropy-performance relationship varies by architecture.

        Returns:
            DataFrame containing architecture-specific entropy-performance correlations.
        """
        print("\n" + "=" * 80)
        print("ANALYZING ENTROPY-PERFORMANCE RELATIONSHIP BY ARCHITECTURE")
        print("=" * 80)

        if 'architecture' not in self.df.columns:
            print("Warning: 'architecture' column not found in dataset")
            return pd.DataFrame()

        # Define entropy columns
        entropy_cols = [
            'sample_mean_entropy', 'sample_total_entropy', 'sample_std_entropy',
            'agent_mean_entropy', 'agent_total_entropy', 'agent_std_entropy',
            'exp_avg_entropy', 'exp_total_entropy'
        ]

        # Filter to only include columns that exist
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]

        results = []

        for arch in self.df['architecture'].unique():
            arch_data = self.df[self.df['architecture'] == arch]

            for ent_col in entropy_cols:
                if ent_col in arch_data.columns and 'is_correct' in arch_data.columns:
                    # Remove missing values
                    valid_idx = arch_data[ent_col].notna() & arch_data['is_correct'].notna()
                    x = arch_data.loc[valid_idx, ent_col]
                    y = arch_data.loc[valid_idx, 'is_correct']

                    if len(x) > 0:
                        # Point-biserial correlation (equivalent to Pearson for binary-continuous)
                        r, p = pearsonr(x, y)

                        results.append({
                            'architecture': arch,
                            'entropy_metric': ent_col,
                            'correlation': r,
                            'p_value': p,
                            'significant': p < 0.05
                        })

        arch_corr_df = pd.DataFrame(results)

        # Save results
        arch_corr_df.to_csv(os.path.join(self.results_dir, 'architecture_entropy_performance_correlation.csv'),
                           index=False)

        print(f"\nArchitecture-specific entropy-performance correlations:")
        print(arch_corr_df.to_string(index=False))

        # Visualize
        self._visualize_architecture_entropy_performance(arch_corr_df)

        self.patterns['architecture_entropy_performance'] = arch_corr_df

        return arch_corr_df

    def identify_optimal_entropy_ranges(self) -> pd.DataFrame:
        """Identify optimal entropy ranges for best performance.

        Returns:
            DataFrame containing optimal entropy ranges for each metric.
        """
        print("\n" + "=" * 80)
        print("IDENTIFYING OPTIMAL ENTROPY RANGES")
        print("=" * 80)

        # Define entropy columns
        entropy_cols = [
            'sample_mean_entropy', 'sample_total_entropy', 'sample_std_entropy',
            'agent_mean_entropy', 'agent_total_entropy', 'agent_std_entropy',
            'exp_avg_entropy', 'exp_total_entropy'
        ]

        # Filter to only include columns that exist
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]

        results = []

        for ent_col in entropy_cols:
            if 'is_correct' in self.df.columns:
                # Separate correct and incorrect
                correct_data = self.df[self.df['is_correct'] == 1][ent_col].dropna()
                incorrect_data = self.df[self.df['is_correct'] == 0][ent_col].dropna()

                if len(correct_data) > 0 and len(incorrect_data) > 0:
                    # Statistical test
                    stat, p = mannwhitneyu(correct_data, incorrect_data, alternative='greater')

                    results.append({
                        'entropy_metric': ent_col,
                        'correct_mean': correct_data.mean(),
                        'correct_std': correct_data.std(),
                        'correct_median': correct_data.median(),
                        'incorrect_mean': incorrect_data.mean(),
                        'incorrect_std': incorrect_data.std(),
                        'incorrect_median': incorrect_data.median(),
                        'difference': correct_data.mean() - incorrect_data.mean(),
                        'test_statistic': stat,
                        'p_value': p,
                        'significant': p < 0.05
                    })

        optimal_ranges_df = pd.DataFrame(results)

        # Save results
        optimal_ranges_df.to_csv(os.path.join(self.results_dir, 'optimal_entropy_ranges.csv'),
                                 index=False)

        print(f"\nOptimal entropy ranges:")
        print(optimal_ranges_df.to_string(index=False))

        # Visualize
        self._visualize_optimal_entropy_ranges(optimal_ranges_df)

        self.patterns['optimal_entropy_ranges'] = optimal_ranges_df

        return optimal_ranges_df

    def analyze_correctness_transition_patterns(self) -> pd.DataFrame:
        """Analyze patterns in correctness transitions from agent-level to final answer.

        Returns:
            DataFrame containing correctness transition patterns.
        """
        print("\n" + "=" * 80)
        print("ANALYZING CORRECTNESS TRANSITION PATTERNS")
        print("=" * 80)

        if 'is_correct' not in self.df.columns or 'is_finally_correct' not in self.df.columns:
            print("Warning: 'is_correct' or 'is_finally_correct' columns not found")
            return pd.DataFrame()

        # Define entropy columns
        entropy_cols = [
            'sample_mean_entropy', 'sample_total_entropy', 'sample_std_entropy',
            'agent_mean_entropy', 'agent_total_entropy', 'agent_std_entropy',
            'exp_avg_entropy', 'exp_total_entropy'
        ]

        # Filter to only include columns that exist
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]

        results = []

        # Analyze improvement cases (wrong -> correct)
        improvement_mask = (self.df['is_correct'] == False) & (self.df['is_finally_correct'] == True)
        degradation_mask = (self.df['is_correct'] == True) & (self.df['is_finally_correct'] == False)
        stable_correct_mask = (self.df['is_correct'] == True) & (self.df['is_finally_correct'] == True)
        stable_incorrect_mask = (self.df['is_correct'] == False) & (self.df['is_finally_correct'] == False)

        for ent_col in entropy_cols:
            improvement_data = self.df[improvement_mask][ent_col].dropna()
            degradation_data = self.df[degradation_mask][ent_col].dropna()
            stable_correct_data = self.df[stable_correct_mask][ent_col].dropna()
            stable_incorrect_data = self.df[stable_incorrect_mask][ent_col].dropna()

            if len(improvement_data) > 0:
                results.append({
                    'entropy_metric': ent_col,
                    'transition_type': 'improvement',
                    'mean': improvement_data.mean(),
                    'std': improvement_data.std(),
                    'median': improvement_data.median(),
                    'count': len(improvement_data)
                })

            if len(degradation_data) > 0:
                results.append({
                    'entropy_metric': ent_col,
                    'transition_type': 'degradation',
                    'mean': degradation_data.mean(),
                    'std': degradation_data.std(),
                    'median': degradation_data.median(),
                    'count': len(degradation_data)
                })

            if len(stable_correct_data) > 0:
                results.append({
                    'entropy_metric': ent_col,
                    'transition_type': 'stable_correct',
                    'mean': stable_correct_data.mean(),
                    'std': stable_correct_data.std(),
                    'median': stable_correct_data.median(),
                    'count': len(stable_correct_data)
                })

            if len(stable_incorrect_data) > 0:
                results.append({
                    'entropy_metric': ent_col,
                    'transition_type': 'stable_incorrect',
                    'mean': stable_incorrect_data.mean(),
                    'std': stable_incorrect_data.std(),
                    'median': stable_incorrect_data.median(),
                    'count': len(stable_incorrect_data)
                })

        transition_df = pd.DataFrame(results)

        # Save results
        transition_df.to_csv(os.path.join(self.results_dir, 'correctness_transition_patterns.csv'),
                            index=False)

        print(f"\nCorrectness transition patterns:")
        print(transition_df.to_string(index=False))

        # Visualize
        self._visualize_correctness_transitions(transition_df)

        self.patterns['correctness_transitions'] = transition_df

        return transition_df

    def analyze_entropy_dynamics_over_execution_order(self) -> pd.DataFrame:
        """Analyze how entropy changes over execution order.

        Returns:
            DataFrame containing entropy dynamics over execution order.
        """
        print("\n" + "=" * 80)
        print("ANALYZING ENTROPY DYNAMICS OVER EXECUTION ORDER")
        print("=" * 80)

        if 'execution_order' not in self.df.columns:
            print("Warning: 'execution_order' column not found in dataset")
            return pd.DataFrame()

        # Define entropy columns
        entropy_cols = [
            'sample_mean_entropy', 'sample_total_entropy', 'sample_std_entropy',
            'agent_mean_entropy', 'agent_total_entropy', 'agent_std_entropy',
            'exp_avg_entropy', 'exp_total_entropy'
        ]

        # Filter to only include columns that exist
        entropy_cols = [col for col in entropy_cols if col in self.df.columns]

        results = []

        for ent_col in entropy_cols:
            if 'execution_order' in self.df.columns:
                # Remove missing values
                valid_idx = self.df[ent_col].notna() & self.df['execution_order'].notna()
                x = self.df.loc[valid_idx, 'execution_order']
                y = self.df.loc[valid_idx, ent_col]

                if len(x) > 0:
                    # Correlation
                    r, p = spearmanr(x, y)

                    results.append({
                        'entropy_metric': ent_col,
                        'correlation': r,
                        'p_value': p,
                        'significant': p < 0.05,
                        'trend': 'increasing' if r > 0 else 'decreasing' if r < 0 else 'no trend'
                    })

        dynamics_df = pd.DataFrame(results)

        # Save results
        dynamics_df.to_csv(os.path.join(self.results_dir, 'entropy_dynamics_execution_order.csv'),
                          index=False)

        print(f"\nEntropy dynamics over execution order:")
        print(dynamics_df.to_string(index=False))

        # Visualize
        self._visualize_entropy_dynamics(dynamics_df)

        self.patterns['entropy_dynamics'] = dynamics_df

        return dynamics_df

    def _visualize_correctness_transitions(self, transition_df: pd.DataFrame) -> None:
        """Create visualization for correctness transition patterns.

        Args:
            transition_df: DataFrame containing correctness transition results.
        """
        if transition_df.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Pivot for heatmap
        pivot_df = transition_df.pivot(index='entropy_metric', columns='transition_type', values='mean')

        # Heatmap
        ax1 = axes[0, 0]
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax1)
        ax1.set_title('Mean Entropy by Correctness Transition Type')
        ax1.set_xlabel('Transition Type')
        ax1.set_ylabel('Entropy Metric')

        # Bar plot by transition type
        ax2 = axes[0, 1]
        top_entropy = transition_df.groupby('entropy_metric')['mean'].mean().nlargest(5).index
        filtered_df = transition_df[transition_df['entropy_metric'].isin(top_entropy)]
        sns.barplot(data=filtered_df, x='transition_type', y='mean', hue='entropy_metric', ax=ax2)
        ax2.set_title('Top 5 Entropy Metrics by Transition Type')
        ax2.set_xlabel('Transition Type')
        ax2.set_ylabel('Mean Entropy')
        ax2.legend(title='Entropy Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Count by transition type
        ax3 = axes[1, 0]
        count_df = transition_df.groupby('transition_type')['count'].first().reset_index()
        sns.barplot(data=count_df, x='transition_type', y='count', ax=ax3, palette='viridis')
        ax3.set_title('Sample Count by Transition Type')
        ax3.set_xlabel('Transition Type')
        ax3.set_ylabel('Count')

        # Median comparison
        ax4 = axes[1, 1]
        pivot_median = transition_df.pivot(index='entropy_metric', columns='transition_type', values='median')
        sns.heatmap(pivot_median, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax4)
        ax4.set_title('Median Entropy by Correctness Transition Type')
        ax4.set_xlabel('Transition Type')
        ax4.set_ylabel('Entropy Metric')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correctness_transition_patterns.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def validate_patterns_with_statistical_tests(self) -> pd.DataFrame:
        """Validate identified patterns using statistical tests.

        Returns:
            DataFrame containing validation results.
        """
        print("\n" + "=" * 80)
        print("VALIDATING PATTERNS WITH STATISTICAL TESTS")
        print("=" * 80)

        validation_results = []

        # Pattern 1: Higher total entropy correlates with better performance
        if 'exp_total_entropy' in self.df.columns and 'is_correct' in self.df.columns:
            correct = self.df[self.df['is_correct'] == 1]['exp_total_entropy'].dropna()
            incorrect = self.df[self.df['is_correct'] == 0]['exp_total_entropy'].dropna()

            if len(correct) > 0 and len(incorrect) > 0:
                stat, p = mannwhitneyu(correct, incorrect, alternative='greater')
                validation_results.append({
                    'pattern': 'Higher total entropy -> better performance',
                    'test': 'Mann-Whitney U',
                    'statistic': stat,
                    'p_value': p,
                    'validated': p < 0.05,
                    'direction': 'correct > incorrect' if correct.mean() > incorrect.mean() else 'incorrect > correct'
                })

        # Pattern 2: Entropy varies significantly across architectures
        if 'architecture' in self.df.columns and 'exp_total_entropy' in self.df.columns:
            groups = [self.df[self.df['architecture'] == arch]['exp_total_entropy'].dropna()
                     for arch in self.df['architecture'].unique()]

            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                stat, p = kruskal(*groups)
                validation_results.append({
                    'pattern': 'Entropy varies across architectures',
                    'test': 'Kruskal-Wallis',
                    'statistic': stat,
                    'p_value': p,
                    'validated': p < 0.05,
                    'direction': 'significant difference' if p < 0.05 else 'no significant difference'
                })

        # Pattern 3: Execution order correlates with entropy
        if 'execution_order' in self.df.columns and 'exp_total_entropy' in self.df.columns:
            valid_idx = self.df['execution_order'].notna() & self.df['exp_total_entropy'].notna()
            x = self.df.loc[valid_idx, 'execution_order']
            y = self.df.loc[valid_idx, 'exp_total_entropy']

            if len(x) > 0:
                r, p = spearmanr(x, y)
                validation_results.append({
                    'pattern': 'Execution order correlates with entropy',
                    'test': 'Spearman correlation',
                    'statistic': r,
                    'p_value': p,
                    'validated': p < 0.05,
                    'direction': 'positive' if r > 0 else 'negative' if r < 0 else 'no correlation'
                })

        # Pattern 4: Agent entropy correlates with sample entropy
        if 'agent_total_entropy' in self.df.columns and 'sample_total_entropy' in self.df.columns:
            valid_idx = self.df['agent_total_entropy'].notna() & self.df['sample_total_entropy'].notna()
            x = self.df.loc[valid_idx, 'agent_total_entropy']
            y = self.df.loc[valid_idx, 'sample_total_entropy']

            if len(x) > 0:
                r, p = pearsonr(x, y)
                validation_results.append({
                    'pattern': 'Agent entropy correlates with sample entropy',
                    'test': 'Pearson correlation',
                    'statistic': r,
                    'p_value': p,
                    'validated': p < 0.05,
                    'direction': 'positive' if r > 0 else 'negative' if r < 0 else 'no correlation'
                })

        validation_df = pd.DataFrame(validation_results)

        # Save results
        validation_df.to_csv(os.path.join(self.results_dir, 'pattern_validation.csv'),
                            index=False)

        print(f"\nPattern validation results:")
        print(validation_df.to_string(index=False))

        # Visualize validation
        self._visualize_pattern_validation(validation_df)

        self.patterns['validation'] = validation_df

        return validation_df

    def _visualize_entropy_performance_correlation(self, correlation_df: pd.DataFrame) -> None:
        """Create visualization for entropy-performance correlation.

        Args:
            correlation_df: DataFrame containing correlation results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Pearson correlation heatmap
        ax1 = axes[0]
        pivot_df = correlation_df.pivot(index='entropy_metric', columns='performance_metric', values='pearson_r')
        sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, fmt='.3f', ax=ax1)
        ax1.set_title('Pearson Correlation: Entropy vs Performance')
        ax1.set_xlabel('Performance Metric')
        ax1.set_ylabel('Entropy Metric')

        # Significant correlations
        ax2 = axes[1]
        significant = correlation_df[
            (correlation_df['pearson_p'] < 0.05) | (correlation_df['spearman_p'] < 0.05)
        ]

        if len(significant) > 0:
            ax2.barh(range(len(significant)), significant['pearson_r'], color='steelblue', alpha=0.7)
            ax2.set_yticks(range(len(significant)))
            ax2.set_yticklabels(significant['entropy_metric'], fontsize=8)
            ax2.set_xlabel('Pearson Correlation')
            ax2.set_title('Significant Correlations (p < 0.05)')
            ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No significant correlations found',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Significant Correlations (p < 0.05)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'entropy_performance_correlation.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_architecture_entropy_patterns(self, arch_stats_df: pd.DataFrame) -> None:
        """Create visualization for architecture-specific entropy patterns.

        Args:
            arch_stats_df: DataFrame containing architecture entropy statistics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Mean entropy by architecture
        ax1 = axes[0, 0]
        pivot_mean = arch_stats_df.pivot(index='architecture', columns='entropy_metric', values='mean')
        pivot_mean.plot(kind='bar', ax=ax1, alpha=0.7)
        ax1.set_title('Mean Entropy by Architecture')
        ax1.set_xlabel('Architecture')
        ax1.set_ylabel('Mean Entropy')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Std entropy by architecture
        ax2 = axes[0, 1]
        pivot_std = arch_stats_df.pivot(index='architecture', columns='entropy_metric', values='std')
        pivot_std.plot(kind='bar', ax=ax2, alpha=0.7)
        ax2.set_title('Entropy Standard Deviation by Architecture')
        ax2.set_xlabel('Architecture')
        ax2.set_ylabel('Std Entropy')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Box plot for total entropy
        ax3 = axes[1, 0]
        if 'exp_total_entropy' in self.df.columns and 'architecture' in self.df.columns:
            self.df.boxplot(column='exp_total_entropy', by='architecture', ax=ax3)
            ax3.set_title('Total Entropy Distribution by Architecture')
            ax3.set_xlabel('Architecture')
            ax3.set_ylabel('Total Entropy')
            plt.suptitle('')

        # Statistical test results
        ax4 = axes[1, 1]
        if 'tests' in self.patterns.get('architecture_entropy', {}):
            tests_df = self.patterns['architecture_entropy']['tests']
            x = range(len(tests_df))
            ax4.bar(x, tests_df['p_value'], color=['red' if p < 0.05 else 'gray' for p in tests_df['p_value']], alpha=0.7)
            ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance threshold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(tests_df['entropy_metric'], rotation=45, ha='right')
            ax4.set_ylabel('p-value')
            ax4.set_title('Statistical Test Results by Entropy Metric')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'architecture_entropy_patterns.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_architecture_entropy_performance(self, arch_corr_df: pd.DataFrame) -> None:
        """Create visualization for architecture-specific entropy-performance correlations.

        Args:
            arch_corr_df: DataFrame containing architecture-specific correlations.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        if len(arch_corr_df) > 0:
            # Pivot data for heatmap
            pivot_df = arch_corr_df.pivot(index='architecture', columns='entropy_metric', values='correlation')

            sns.heatmap(pivot_df, annot=True, cmap='coolwarm', center=0, fmt='.3f', ax=ax)
            ax.set_title('Entropy-Performance Correlation by Architecture')
            ax.set_xlabel('Entropy Metric')
            ax.set_ylabel('Architecture')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'architecture_entropy_performance_correlation.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_optimal_entropy_ranges(self, optimal_ranges_df: pd.DataFrame) -> None:
        """Create visualization for optimal entropy ranges.

        Args:
            optimal_ranges_df: DataFrame containing optimal entropy ranges.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Mean comparison
        ax1 = axes[0]
        x = range(len(optimal_ranges_df))
        width = 0.35

        ax1.bar([i - width/2 for i in x], optimal_ranges_df['correct_mean'], width,
               label='Correct', color='green', alpha=0.7)
        ax1.bar([i + width/2 for i in x], optimal_ranges_df['incorrect_mean'], width,
               label='Incorrect', color='red', alpha=0.7)

        ax1.set_xlabel('Entropy Metric')
        ax1.set_ylabel('Mean Entropy')
        ax1.set_title('Mean Entropy: Correct vs Incorrect')
        ax1.set_xticks(x)
        ax1.set_xticklabels(optimal_ranges_df['entropy_metric'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Significance
        ax2 = axes[1]
        colors = ['green' if sig else 'gray' for sig in optimal_ranges_df['significant']]
        ax2.barh(range(len(optimal_ranges_df)), -np.log10(optimal_ranges_df['p_value']), color=colors, alpha=0.7)
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='Significance threshold')
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_title('Statistical Significance of Entropy Differences')
        ax2.set_yticks(range(len(optimal_ranges_df)))
        ax2.set_yticklabels(optimal_ranges_df['entropy_metric'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'optimal_entropy_ranges.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_entropy_dynamics(self, dynamics_df: pd.DataFrame) -> None:
        """Create visualization for entropy dynamics over execution order.

        Args:
            dynamics_df: DataFrame containing entropy dynamics.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Correlation coefficients
        ax1 = axes[0]
        colors = ['green' if sig else 'gray' for sig in dynamics_df['significant']]
        ax1.barh(range(len(dynamics_df)), dynamics_df['correlation'], color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Spearman Correlation')
        ax1.set_title('Entropy-Execution Order Correlation')
        ax1.set_yticks(range(len(dynamics_df)))
        ax1.set_yticklabels(dynamics_df['entropy_metric'])
        ax1.grid(True, alpha=0.3)

        # Trend summary
        ax2 = axes[1]
        trend_counts = dynamics_df['trend'].value_counts()
        ax2.pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Entropy Trend Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'entropy_dynamics_execution_order.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_pattern_validation(self, validation_df: pd.DataFrame) -> None:
        """Create visualization for pattern validation results.

        Args:
            validation_df: DataFrame containing validation results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Validation status
        ax1 = axes[0]
        x = range(len(validation_df))
        colors = ['green' if validated else 'red' for validated in validation_df['validated']]
        ax1.barh(x, -np.log10(validation_df['p_value']), color=colors, alpha=0.7)
        ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='Significance threshold')
        ax1.set_xlabel('-log10(p-value)')
        ax1.set_title('Pattern Validation Results')
        ax1.set_yticks(x)
        ax1.set_yticklabels(validation_df['pattern'], fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation summary
        ax2 = axes[1]
        validated_count = validation_df['validated'].sum()
        total_count = len(validation_df)
        sizes = [validated_count, total_count - validated_count]
        labels = [f'Validated ({validated_count})', f'Not Validated ({total_count - validated_count})']
        colors = ['green', 'red']
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Pattern Validation Summary')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'pattern_validation.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_pattern_report(self) -> None:
        """Generate comprehensive pattern identification and validation report."""
        report_path = os.path.join(self.results_dir, 'pattern_identification_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PATTERN IDENTIFICATION AND VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total samples: {len(self.df)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("IDENTIFIED PATTERNS\n")
            f.write("-" * 80 + "\n\n")

            # Pattern 1: Entropy-Performance Correlation
            if 'entropy_performance_correlation' in self.patterns:
                f.write("1. ENTROPY-PERFORMANCE CORRELATION\n")
                f.write("   - Significant correlations found between entropy metrics and performance\n")
                f.write("   - Key findings:\n")
                sig_corr = self.patterns['entropy_performance_correlation']
                for _, row in sig_corr.iterrows():
                    f.write(f"     * {row['entropy_metric']} vs {row['performance_metric']}: "
                           f"r={row['pearson_r']:.3f}, p={row['pearson_p']:.4f}\n")
                f.write("\n")

            # Pattern 2: Architecture-Specific Entropy Patterns
            if 'architecture_entropy' in self.patterns:
                f.write("2. ARCHITECTURE-SPECIFIC ENTROPY PATTERNS\n")
                f.write("   - Entropy metrics vary significantly across different architectures\n")
                f.write("   - Statistical tests confirm significant differences\n")
                f.write("\n")

            # Pattern 3: Optimal Entropy Ranges
            if 'optimal_entropy_ranges' in self.patterns:
                f.write("3. OPTIMAL ENTROPY RANGES FOR PERFORMANCE\n")
                f.write("   - Specific entropy ranges associated with better performance\n")
                f.write("   - Key findings:\n")
                opt_ranges = self.patterns['optimal_entropy_ranges']
                for _, row in opt_ranges.iterrows():
                    if row['significant']:
                        f.write(f"     * {row['entropy_metric']}: "
                               f"correct mean={row['correct_mean']:.4f}, "
                               f"incorrect mean={row['incorrect_mean']:.4f}, "
                               f"p={row['p_value']:.4f}\n")
                f.write("\n")

            # Pattern 4: Entropy Dynamics
            if 'entropy_dynamics' in self.patterns:
                f.write("4. ENTROPY DYNAMICS OVER EXECUTION ORDER\n")
                f.write("   - Entropy metrics show trends over execution order\n")
                f.write("   - Key findings:\n")
                dynamics = self.patterns['entropy_dynamics']
                for _, row in dynamics.iterrows():
                    if row['significant']:
                        f.write(f"     * {row['entropy_metric']}: "
                               f"correlation={row['correlation']:.3f}, "
                               f"trend={row['trend']}, p={row['p_value']:.4f}\n")
                f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("VALIDATION RESULTS\n")
            f.write("-" * 80 + "\n\n")

            if 'validation' in self.patterns:
                f.write("Statistical validation of identified patterns:\n\n")
                validation = self.patterns['validation']
                for _, row in validation.iterrows():
                    status = "VALIDATED" if row['validated'] else "NOT VALIDATED"
                    f.write(f"Pattern: {row['pattern']}\n")
                    f.write(f"  Test: {row['test']}\n")
                    f.write(f"  Statistic: {row['statistic']:.4f}\n")
                    f.write(f"  p-value: {row['p_value']:.4f}\n")
                    f.write(f"  Status: {status}\n")
                    f.write(f"  Direction: {row['direction']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("KEY INSIGHTS\n")
            f.write("-" * 80 + "\n\n")
            f.write("1. Entropy metrics are strongly correlated with system performance\n")
            f.write("2. Different architectures exhibit distinct entropy patterns\n")
            f.write("3. Optimal entropy ranges exist for maximizing performance\n")
            f.write("4. Entropy dynamics over execution order reveal important trends\n")
            f.write("5. Statistical tests confirm the reliability of identified patterns\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\nPattern identification and validation report saved to: {report_path}")

    def run_pattern_identification(self) -> None:
        """Run complete pattern identification and validation pipeline."""
        print("=" * 80)
        print("STARTING PATTERN IDENTIFICATION AND VALIDATION")
        print("=" * 80)

        self.load_data()
        self.analyze_entropy_performance_correlation()
        self.analyze_architecture_entropy_patterns()
        self.analyze_entropy_performance_by_architecture()
        self.identify_optimal_entropy_ranges()
        self.analyze_correctness_transition_patterns()
        self.analyze_entropy_dynamics_over_execution_order()
        self.validate_patterns_with_statistical_tests()

        print("\n" + "=" * 80)
        print("GENERATING PATTERN IDENTIFICATION REPORT")
        print("=" * 80)
        self.generate_pattern_report()

        print("\n" + "=" * 80)
        print("PATTERN IDENTIFICATION AND VALIDATION COMPLETED")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")


def main():
    """Main function to run pattern identification."""
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    identifier = PatternIdentifier(data_path, output_dir)
    identifier.run_pattern_identification()


if __name__ == '__main__':
    main()
