"""Data Quality Assessment Module for Multi-Agent Entropy Research.

This module provides comprehensive data quality assessment including:
- Missing value detection and analysis
- Outlier detection using statistical methods (Z-score, IQR)
- Data distribution analysis (central tendency, dispersion, shape)
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DataQualityAssessor:
    """Class for assessing data quality of multi-agent entropy experiments."""

    def __init__(self, data_path: str, output_dir: str):
        """Initialize the DataQualityAssessor.

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
        self.results_dir = os.path.join(output_dir, 'results', 'data_quality_summary')
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
        print(f"Numeric columns: {len(self.numeric_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")

    def analyze_missing_values(self) -> pd.DataFrame:
        """Analyze missing values in the dataset.

        Returns:
            DataFrame containing missing value statistics.
        """
        missing_stats = pd.DataFrame({
            'column': self.df.columns,
            'missing_count': self.df.isnull().sum().values,
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).values,
            'data_type': self.df.dtypes.values
        })
        missing_stats = missing_stats[missing_stats['missing_count'] > 0]
        missing_stats = missing_stats.sort_values('missing_percentage', ascending=False)

        # Save missing value report
        report_path = os.path.join(self.results_dir, 'missing_values_report.csv')
        missing_stats.to_csv(report_path, index=False)

        # Visualize missing values
        self._visualize_missing_values(missing_stats)

        print(f"\nMissing Value Analysis:")
        print(f"Columns with missing values: {len(missing_stats)}")
        if len(missing_stats) > 0:
            print("\nTop 10 columns with highest missing percentage:")
            print(missing_stats.head(10).to_string(index=False))

        return missing_stats

    def _visualize_missing_values(self, missing_stats: pd.DataFrame) -> None:
        """Create visualization for missing values.

        Args:
            missing_stats: DataFrame containing missing value statistics.
        """
        if len(missing_stats) == 0:
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Bar plot of missing percentages
        ax1 = axes[0]
        top_missing = missing_stats.head(20)
        ax1.barh(range(len(top_missing)), top_missing['missing_percentage'])
        ax1.set_yticks(range(len(top_missing)))
        ax1.set_yticklabels(top_missing['column'], fontsize=8)
        ax1.set_xlabel('Missing Percentage (%)')
        ax1.set_title('Top 20 Columns by Missing Percentage')
        ax1.invert_yaxis()

        # Heatmap of missing values
        ax2 = axes[1]
        missing_data = self.df.isnull()
        if missing_data.sum().sum() > 0:
            sns.heatmap(missing_data.iloc[:, :50], cbar=False, cmap='viridis', ax=ax2)
            ax2.set_title('Missing Value Heatmap (First 50 Columns)')
        else:
            ax2.text(0.5, 0.5, 'No missing values', ha='center', va='center')
            ax2.set_title('Missing Value Heatmap')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'missing_values_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def detect_outliers_zscore(self, threshold: float = 3.0) -> Dict[str, Dict]:
        """Detect outliers using Z-score method.

        Args:
            threshold: Z-score threshold for outlier detection.

        Returns:
            Dictionary containing outlier information for each column.
        """
        outlier_info = {}

        for col in self.numeric_columns:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers = z_scores > threshold
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(self.df[col].dropna())) * 100

            outlier_info[col] = {
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage,
                'outlier_indices': self.df[col].dropna().index[outliers].tolist(),
                'outlier_values': self.df[col].dropna()[outliers].tolist()
            }

        # Save outlier report
        outlier_df = pd.DataFrame({
            'column': list(outlier_info.keys()),
            'outlier_count': [info['outlier_count'] for info in outlier_info.values()],
            'outlier_percentage': [info['outlier_percentage'] for info in outlier_info.values()]
        })
        outlier_df = outlier_df.sort_values('outlier_percentage', ascending=False)
        outlier_df.to_csv(os.path.join(self.results_dir, 'outliers_zscore_report.csv'),
                          index=False)

        print(f"\nZ-Score Outlier Detection (threshold={threshold}):")
        print(f"Columns with outliers: {len(outlier_df[outlier_df['outlier_count'] > 0])}")
        if len(outlier_df[outlier_df['outlier_count'] > 0]) > 0:
            print("\nTop 10 columns by outlier percentage:")
            print(outlier_df.head(10).to_string(index=False))

        return outlier_info

    def detect_outliers_iqr(self, multiplier: float = 1.5) -> Dict[str, Dict]:
        """Detect outliers using IQR method.

        Args:
            multiplier: IQR multiplier for outlier detection.

        Returns:
            Dictionary containing outlier information for each column.
        """
        outlier_info = {}

        for col in self.numeric_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(self.df[col].dropna())) * 100

            outlier_info[col] = {
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_indices': self.df[col].index[outliers].tolist(),
                'outlier_values': self.df[col][outliers].tolist()
            }

        # Save outlier report
        outlier_df = pd.DataFrame({
            'column': list(outlier_info.keys()),
            'outlier_count': [info['outlier_count'] for info in outlier_info.values()],
            'outlier_percentage': [info['outlier_percentage'] for info in outlier_info.values()],
            'lower_bound': [info['lower_bound'] for info in outlier_info.values()],
            'upper_bound': [info['upper_bound'] for info in outlier_info.values()]
        })
        outlier_df = outlier_df.sort_values('outlier_percentage', ascending=False)
        outlier_df.to_csv(os.path.join(self.results_dir, 'outliers_iqr_report.csv'),
                          index=False)

        print(f"\nIQR Outlier Detection (multiplier={multiplier}):")
        print(f"Columns with outliers: {len(outlier_df[outlier_df['outlier_count'] > 0])}")
        if len(outlier_df[outlier_df['outlier_count'] > 0]) > 0:
            print("\nTop 10 columns by outlier percentage:")
            print(outlier_df.head(10).to_string(index=False))

        return outlier_info

    def visualize_outliers(self, outlier_info_zscore: Dict, outlier_info_iqr: Dict) -> None:
        """Create visualizations for outliers.

        Args:
            outlier_info_zscore: Outlier information from Z-score method.
            outlier_info_iqr: Outlier information from IQR method.
        """
        # Select top columns with most outliers
        zscore_df = pd.DataFrame({
            'column': list(outlier_info_zscore.keys()),
            'outlier_percentage': [info['outlier_percentage'] for info in outlier_info_zscore.values()]
        }).sort_values('outlier_percentage', ascending=False)

        top_columns = zscore_df.head(12)['column'].tolist()

        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        axes = axes.flatten()

        for idx, col in enumerate(top_columns):
            if idx >= len(axes):
                break

            data = self.df[col].dropna()
            ax = axes[idx]

            # Box plot
            bp = ax.boxplot(data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            ax.set_title(f'{col}\nOutliers: {outlier_info_zscore[col]["outlier_count"]} '
                        f'({outlier_info_zscore[col]["outlier_percentage"]:.2f}%)',
                        fontsize=9)
            ax.set_ylabel('Value', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)

        # Hide empty subplots
        for idx in range(len(top_columns), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Box Plots for Top Columns with Most Outliers', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'outliers_boxplots.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_distribution(self) -> pd.DataFrame:
        """Analyze distribution characteristics of numeric columns.

        Returns:
            DataFrame containing distribution statistics.
        """
        distribution_stats = []

        for col in self.numeric_columns:
            data = self.df[col].dropna()

            stats_dict = {
                'column': col,
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'variance': data.var(),
                'min': data.min(),
                'max': data.max(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75),
                'iqr': data.quantile(0.75) - data.quantile(0.25),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'range': data.max() - data.min(),
                'coefficient_of_variation': (data.std() / data.mean()) if data.mean() != 0 else np.nan
            }
            distribution_stats.append(stats_dict)

        dist_df = pd.DataFrame(distribution_stats)
        dist_df.to_csv(os.path.join(self.results_dir, 'distribution_statistics.csv'),
                      index=False)

        print(f"\nDistribution Analysis:")
        print(f"Analyzed {len(dist_df)} numeric columns")

        return dist_df

    def visualize_distributions(self, n_cols: int = 15) -> None:
        """Create distribution visualizations for key numeric columns.

        Args:
            n_cols: Number of columns to visualize.
        """
        # Select columns with highest variance
        variance_df = pd.DataFrame({
            'column': self.numeric_columns,
            'variance': [self.df[col].var() for col in self.numeric_columns]
        }).sort_values('variance', ascending=False)

        top_columns = variance_df.head(n_cols)['column'].tolist()

        fig, axes = plt.subplots(5, 3, figsize=(18, 20))
        axes = axes.flatten()

        for idx, col in enumerate(top_columns):
            if idx >= len(axes):
                break

            data = self.df[col].dropna()
            ax = axes[idx]

            # Histogram with KDE
            ax.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
            ax2 = ax.twinx()

            # KDE plot
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax2.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            ax2.set_ylabel('Density', fontsize=8)

            # Add statistics
            ax.axvline(data.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(data.median(), color='orange', linestyle='--', linewidth=2, label='Median')

            ax.set_title(f'{col}\nSkewness: {data.skew():.2f}, Kurtosis: {data.kurtosis():.2f}',
                        fontsize=9)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.legend(fontsize=6, loc='upper right')
            ax2.legend(fontsize=6, loc='upper left')
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax2.tick_params(axis='both', which='major', labelsize=7)

        # Hide empty subplots
        for idx in range(len(top_columns), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Distribution Analysis of Key Numeric Features', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'distributions_histogram.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_report(self) -> None:
        """Generate a comprehensive data quality summary report."""
        report_path = os.path.join(self.results_dir, 'data_quality_summary.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA QUALITY ASSESSMENT SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total Rows: {self.df.shape[0]}\n")
            f.write(f"Total Columns: {self.df.shape[1]}\n")
            f.write(f"Numeric Columns: {len(self.numeric_columns)}\n")
            f.write(f"Categorical Columns: {len(self.categorical_columns)}\n\n")

            f.write("-" * 80 + "\n")
            f.write("MISSING VALUES ANALYSIS\n")
            f.write("-" * 80 + "\n")
            missing_stats = self.analyze_missing_values()
            if len(missing_stats) > 0:
                f.write(f"Columns with missing values: {len(missing_stats)}\n")
                f.write(f"Total missing values: {self.df.isnull().sum().sum()}\n")
            else:
                f.write("No missing values found.\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("OUTLIER ANALYSIS\n")
            f.write("-" * 80 + "\n")
            outlier_info_zscore = self.detect_outliers_zscore()
            outlier_info_iqr = self.detect_outliers_iqr()

            f.write(f"\nZ-Score Method (threshold=3.0):\n")
            zscore_outliers = sum(1 for info in outlier_info_zscore.values() if info['outlier_count'] > 0)
            f.write(f"Columns with outliers: {zscore_outliers}\n")

            f.write(f"\nIQR Method (multiplier=1.5):\n")
            iqr_outliers = sum(1 for info in outlier_info_iqr.values() if info['outlier_count'] > 0)
            f.write(f"Columns with outliers: {iqr_outliers}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("DISTRIBUTION ANALYSIS\n")
            f.write("-" * 80 + "\n")
            dist_stats = self.analyze_distribution()

            f.write(f"\nColumns analyzed: {len(dist_stats)}\n")
            f.write("\nColumns with highest skewness (absolute value):\n")
            top_skew = dist_stats.nlargest(5, 'skewness')[['column', 'skewness']]
            f.write(top_skew.to_string(index=False))

            f.write("\n\nColumns with highest kurtosis:\n")
            top_kurt = dist_stats.nlargest(5, 'kurtosis')[['column', 'kurtosis']]
            f.write(top_kurt.to_string(index=False))

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\nSummary report saved to: {report_path}")

    def run_full_assessment(self) -> None:
        """Run complete data quality assessment."""
        print("=" * 80)
        print("STARTING DATA QUALITY ASSESSMENT")
        print("=" * 80)

        self.load_data()
        self.identify_column_types()

        print("\n" + "=" * 80)
        print("MISSING VALUE ANALYSIS")
        print("=" * 80)
        self.analyze_missing_values()

        print("\n" + "=" * 80)
        print("OUTLIER DETECTION")
        print("=" * 80)
        outlier_info_zscore = self.detect_outliers_zscore()
        outlier_info_iqr = self.detect_outliers_iqr()
        self.visualize_outliers(outlier_info_zscore, outlier_info_iqr)

        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)
        self.analyze_distribution()
        self.visualize_distributions()

        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        self.generate_summary_report()

        print("\n" + "=" * 80)
        print("DATA QUALITY ASSESSMENT COMPLETED")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")


def main():
    """Main function to run data quality assessment."""
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    assessor = DataQualityAssessor(data_path, output_dir)
    assessor.run_full_assessment()


if __name__ == '__main__':
    main()
