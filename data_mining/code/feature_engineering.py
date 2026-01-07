"""Feature Engineering Module for Multi-Agent Entropy Research.

This module provides comprehensive feature engineering including:
- Feature extraction based on domain knowledge and EDA results
- Feature transformation, standardization/normalization
- Feature importance evaluation and selection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class FeatureEngineer:
    """Class for performing feature engineering on multi-agent entropy experiments."""

    def __init__(self, data_path: str, output_dir: str):
        """Initialize the FeatureEngineer.

        Args:
            data_path: Path to the CSV data file.
            output_dir: Directory to save output files and figures.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.feature_df = None
        self.target_col = None
        self.scaler = None
        self.selected_features = None

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'results', 'feature_engineering')
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

    def extract_entropy_features(self) -> pd.DataFrame:
        """Extract entropy-related features based on domain knowledge.

        Returns:
            DataFrame with extracted features.
        """
        feature_dict = {}

        # Sample-level entropy features
        if 'sample_mean_entropy' in self.df.columns:
            feature_dict['sample_mean_entropy'] = self.df['sample_mean_entropy']
        if 'sample_total_entropy' in self.df.columns:
            feature_dict['sample_total_entropy'] = self.df['sample_total_entropy']
        if 'sample_std_entropy' in self.df.columns:
            feature_dict['sample_std_entropy'] = self.df['sample_std_entropy']
        if 'sample_max_entropy' in self.df.columns:
            feature_dict['sample_max_entropy'] = self.df['sample_max_entropy']
        if 'sample_min_entropy' in self.df.columns:
            feature_dict['sample_min_entropy'] = self.df['sample_min_entropy']
        if 'sample_q1_entropy' in self.df.columns:
            feature_dict['sample_q1_entropy'] = self.df['sample_q1_entropy']
        if 'sample_q3_entropy' in self.df.columns:
            feature_dict['sample_q3_entropy'] = self.df['sample_q3_entropy']

        # Agent-level entropy features
        if 'agent_mean_entropy' in self.df.columns:
            feature_dict['agent_mean_entropy'] = self.df['agent_mean_entropy']
        if 'agent_total_entropy' in self.df.columns:
            feature_dict['agent_total_entropy'] = self.df['agent_total_entropy']
        if 'agent_std_entropy' in self.df.columns:
            feature_dict['agent_std_entropy'] = self.df['agent_std_entropy']

        # Experiment-level entropy features
        if 'exp_avg_entropy' in self.df.columns:
            feature_dict['exp_avg_entropy'] = self.df['exp_avg_entropy']
        if 'exp_total_entropy' in self.df.columns:
            feature_dict['exp_total_entropy'] = self.df['exp_total_entropy']

        # Derived entropy features
        if 'sample_total_entropy' in self.df.columns and 'sample_token_count' in self.df.columns:
            feature_dict['entropy_per_token'] = (
                self.df['sample_total_entropy'] / (self.df['sample_token_count'] + 1e-6)
            )

        if 'sample_mean_entropy' in self.df.columns and 'sample_std_entropy' in self.df.columns:
            feature_dict['entropy_coefficient_of_variation'] = (
                self.df['sample_std_entropy'] / (self.df['sample_mean_entropy'] + 1e-6)
            )

        if 'sample_q3_entropy' in self.df.columns and 'sample_q1_entropy' in self.df.columns:
            feature_dict['entropy_iqr'] = (
                self.df['sample_q3_entropy'] - self.df['sample_q1_entropy']
            )

        # Performance-related features
        if 'time_cost' in self.df.columns:
            feature_dict['time_cost'] = self.df['time_cost']

        if 'ground_truth' in self.df.columns:
            feature_dict['ground_truth'] = self.df['ground_truth']

        # Execution order feature
        if 'execution_order' in self.df.columns:
            feature_dict['execution_order'] = self.df['execution_order']

        # Token count features
        if 'sample_token_count' in self.df.columns:
            feature_dict['sample_token_count'] = self.df['sample_token_count']

        # Correctness features
        if 'is_correct' in self.df.columns:
            feature_dict['is_correct'] = self.df['is_correct']
        if 'is_finally_correct' in self.df.columns:
            feature_dict['is_finally_correct'] = self.df['is_finally_correct']

        # Create feature DataFrame
        self.feature_df = pd.DataFrame(feature_dict)

        print(f"Extracted {len(feature_dict)} features")
        print(f"Feature DataFrame shape: {self.feature_df.shape}")

        return self.feature_df

    def extract_correctness_features(self) -> pd.DataFrame:
        """Extract correctness-related features based on is_correct and is_finally_correct.

        Returns:
            DataFrame with extracted correctness features.
        """
        if 'is_correct' not in self.df.columns or 'is_finally_correct' not in self.df.columns:
            print("Warning: 'is_correct' or 'is_finally_correct' columns not found")
            return self.feature_df

        print("\nExtracting correctness-related features...")

        # Create correctness comparison features
        self.feature_df['correctness_agreement'] = (
            (self.df['is_correct'] == self.df['is_finally_correct']).astype(int)
        )

        # Create improvement/degradation indicators
        self.feature_df['correctness_improvement'] = (
            ((self.df['is_correct'] == False) & (self.df['is_finally_correct'] == True)).astype(int)
        )

        self.feature_df['correctness_degradation'] = (
            ((self.df['is_correct'] == True) & (self.df['is_finally_correct'] == False)).astype(int)
        )

        # Create architecture-specific correctness features
        if 'architecture' in self.df.columns:
            arch_correctness = self.df.groupby('architecture').agg({
                'is_correct': 'mean',
                'is_finally_correct': 'mean'
            })
            
            for arch in arch_correctness.index:
                arch_mask = self.df['architecture'] == arch
                self.feature_df[f'{arch}_agent_correct_rate'] = arch_correctness.loc[arch, 'is_correct']
                self.feature_df[f'{arch}_final_correct_rate'] = arch_correctness.loc[arch, 'is_finally_correct']

        print(f"Added correctness-related features")
        print(f"Feature DataFrame shape: {self.feature_df.shape}")

        return self.feature_df

    def handle_missing_values(self, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in features.

        Args:
            strategy: Strategy for handling missing values ('median', 'mean', 'drop').

        Returns:
            DataFrame with missing values handled.
        """
        print(f"\nHandling missing values with strategy: {strategy}")

        if strategy == 'drop':
            self.feature_df = self.feature_df.dropna()
        elif strategy == 'median':
            for col in self.feature_df.columns:
                if self.feature_df[col].isnull().any():
                    median_val = self.feature_df[col].median()
                    self.feature_df[col].fillna(median_val, inplace=True)
        elif strategy == 'mean':
            for col in self.feature_df.columns:
                if self.feature_df[col].isnull().any():
                    mean_val = self.feature_df[col].mean()
                    self.feature_df[col].fillna(mean_val, inplace=True)

        print(f"Missing values after handling: {self.feature_df.isnull().sum().sum()}")
        print(f"Feature DataFrame shape after handling: {self.feature_df.shape}")

        return self.feature_df

    def normalize_features(self, method: str = 'standard') -> pd.DataFrame:
        """Normalize features using specified method.

        Args:
            method: Normalization method ('standard', 'minmax', 'robust').

        Returns:
            DataFrame with normalized features.
        """
        print(f"\nNormalizing features using method: {method}")

        numeric_cols = self.feature_df.select_dtypes(include=[np.number]).columns.tolist()

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()

        self.feature_df[numeric_cols] = self.scaler.fit_transform(self.feature_df[numeric_cols])

        print(f"Features normalized using {method} scaling")
        print(f"Feature statistics after normalization:")
        print(self.feature_df[numeric_cols].describe().loc[['mean', 'std', 'min', 'max']])

        return self.feature_df

    def evaluate_feature_importance(self, target_col: str = 'is_correct',
                                    method: str = 'random_forest') -> pd.DataFrame:
        """Evaluate feature importance for predicting target variable.

        Args:
            target_col: Target column name.
            method: Method for feature importance ('random_forest', 'mutual_info', 'f_score').

        Returns:
            DataFrame with feature importance scores.
        """
        if target_col not in self.df.columns:
            print(f"Warning: Target column '{target_col}' not found in dataset")
            return pd.DataFrame()

        self.target_col = target_col

        # Prepare data
        X = self.feature_df
        y = self.df[target_col]

        # Remove rows with missing target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"\nEvaluating feature importance using method: {method}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

        importance_scores = {}

        if method == 'random_forest':
            if y.dtype == 'object' or y.nunique() <= 2:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            model.fit(X, y)
            importance_scores = dict(zip(X.columns, model.feature_importances_))

        elif method == 'mutual_info':
            if y.dtype == 'object' or y.nunique() <= 2:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X, y, random_state=42)
            importance_scores = dict(zip(X.columns, mi_scores))

        elif method == 'f_score':
            if y.dtype == 'object' or y.nunique() <= 2:
                f_scores, _ = f_classif(X, y)
            else:
                f_scores, _ = f_regression(X, y)
            importance_scores = dict(zip(X.columns, f_scores))

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        }).sort_values('importance', ascending=False)

        # Save importance results
        importance_df.to_csv(os.path.join(self.results_dir, f'feature_importance_{method}.csv'),
                             index=False)

        # Visualize feature importance
        self._visualize_feature_importance(importance_df, method)

        print(f"\nTop 10 most important features:")
        print(importance_df.head(10).to_string(index=False))

        return importance_df

    def _visualize_feature_importance(self, importance_df: pd.DataFrame, method: str) -> None:
        """Create visualization for feature importance.

        Args:
            importance_df: DataFrame containing feature importance scores.
            method: Method used for importance calculation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Bar plot of top features
        ax1 = axes[0]
        top_features = importance_df.head(20)
        ax1.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'], fontsize=9)
        ax1.set_xlabel('Importance Score')
        ax1.set_title(f'Top 20 Feature Importance ({method})')
        ax1.invert_yaxis()

        # Cumulative importance
        ax2 = axes[1]
        importance_df_sorted = importance_df.sort_values('importance', ascending=False)
        cumulative_importance = np.cumsum(importance_df_sorted['importance'])
        cumulative_importance = cumulative_importance / cumulative_importance.iloc[-1]

        ax2.plot(range(1, len(cumulative_importance) + 1), cumulative_importance,
                marker='o', linewidth=2, markersize=4)
        ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% threshold')
        ax2.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% threshold')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'feature_importance_{method}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def select_features(self, importance_df: pd.DataFrame, n_features: int = 10,
                       threshold: float = None) -> List[str]:
        """Select top features based on importance scores.

        Args:
            importance_df: DataFrame containing feature importance scores.
            n_features: Number of top features to select.
            threshold: Importance threshold for feature selection.

        Returns:
            List of selected feature names.
        """
        if threshold is not None:
            selected_features = importance_df[importance_df['importance'] >= threshold]['feature'].tolist()
        else:
            selected_features = importance_df.head(n_features)['feature'].tolist()

        self.selected_features = selected_features

        print(f"\nSelected {len(selected_features)} features:")
        for i, feat in enumerate(selected_features, 1):
            print(f"  {i}. {feat}")

        # Save selected features
        pd.DataFrame({'feature': selected_features}).to_csv(
            os.path.join(self.results_dir, 'selected_features.csv'), index=False
        )

        return selected_features

    def apply_pca(self, n_components: int = 5) -> Tuple[np.ndarray, pd.DataFrame]:
        """Apply Principal Component Analysis for dimensionality reduction.

        Args:
            n_components: Number of principal components to retain.

        Returns:
            Tuple of (transformed data, explained variance DataFrame).
        """
        print(f"\nApplying PCA with {n_components} components")

        # Prepare data
        X = self.feature_df.select_dtypes(include=[np.number])

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        # Create explained variance DataFrame
        variance_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(n_components)],
            'explained_variance': pca.explained_variance_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
        })

        # Save results
        variance_df.to_csv(os.path.join(self.results_dir, 'pca_variance_explained.csv'),
                          index=False)

        # Visualize PCA results
        self._visualize_pca_results(pca, variance_df)

        print(f"\nPCA Results:")
        print(variance_df.to_string(index=False))
        print(f"\nTotal variance explained: {variance_df['cumulative_variance_ratio'].iloc[-1]:.4f}")

        return X_pca, variance_df

    def _visualize_pca_results(self, pca: PCA, variance_df: pd.DataFrame) -> None:
        """Create visualization for PCA results.

        Args:
            pca: Fitted PCA model.
            variance_df: DataFrame containing variance information.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scree plot
        ax1 = axes[0]
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
               pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
        ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                np.cumsum(pca.explained_variance_ratio_),
                marker='o', linewidth=2, color='red', label='Cumulative')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2D scatter plot of first two components
        ax2 = axes[1]
        X_pca = pca.transform(self.feature_df.select_dtypes(include=[np.number]))

        if self.target_col and self.target_col in self.df.columns:
            # Color by target if available
            valid_idx = self.df[self.target_col].notna()
            scatter = ax2.scatter(X_pca[valid_idx, 0], X_pca[valid_idx, 1],
                                 c=self.df[self.target_col][valid_idx], cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax2, label=self.target_col)
        else:
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)

        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('PCA: First Two Principal Components')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'pca_results.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_feature_report(self) -> None:
        """Generate comprehensive feature engineering report."""
        report_path = os.path.join(self.results_dir, 'feature_engineering_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE ENGINEERING REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Original DataFrame shape: {self.df.shape}\n")
            f.write(f"Feature DataFrame shape: {self.feature_df.shape}\n\n")

            f.write("-" * 80 + "\n")
            f.write("EXTRACTED FEATURES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of features: {len(self.feature_df.columns)}\n")
            f.write(f"Feature list:\n")
            for i, col in enumerate(self.feature_df.columns, 1):
                f.write(f"  {i}. {col}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("FEATURE STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(self.feature_df.describe().to_string())

            if self.selected_features:
                f.write("\n\n" + "-" * 80 + "\n")
                f.write("SELECTED FEATURES\n")
                f.write("-" * 80 + "\n")
                f.write(f"Number of selected features: {len(self.selected_features)}\n")
                for i, feat in enumerate(self.selected_features, 1):
                    f.write(f"  {i}. {feat}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\nFeature engineering report saved to: {report_path}")

    def run_feature_engineering(self, target_col: str = 'is_correct',
                               normalize_method: str = 'standard',
                               importance_method: str = 'random_forest',
                               n_features: int = 10) -> pd.DataFrame:
        """Run complete feature engineering pipeline.

        Args:
            target_col: Target column for importance evaluation (e.g., 'is_correct', 'is_finally_correct').
            normalize_method: Method for feature normalization.
            importance_method: Method for feature importance evaluation.
            n_features: Number of top features to select.

        Returns:
            DataFrame with selected features.
        """
        print("=" * 80)
        print("STARTING FEATURE ENGINEERING")
        print("=" * 80)

        self.load_data()
        self.extract_entropy_features()
        self.extract_correctness_features()
        self.handle_missing_values(strategy='median')
        self.normalize_features(method=normalize_method)

        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE EVALUATION")
        print("=" * 80)
        importance_df = self.evaluate_feature_importance(target_col=target_col,
                                                        method=importance_method)

        print("\n" + "=" * 80)
        print("FEATURE SELECTION")
        print("=" * 80)
        self.select_features(importance_df, n_features=n_features)

        print("\n" + "=" * 80)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 80)
        self.apply_pca(n_components=5)

        print("\n" + "=" * 80)
        print("GENERATING FEATURE ENGINEERING REPORT")
        print("=" * 80)
        self.generate_feature_report()

        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING COMPLETED")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")

        # Return DataFrame with selected features
        return self.feature_df[self.selected_features]


def main():
    """Main function to run feature engineering."""
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    engineer = FeatureEngineer(data_path, output_dir)
    selected_features_df = engineer.run_feature_engineering(
        target_col='is_correct',
        normalize_method='standard',
        importance_method='random_forest',
        n_features=10
    )

    print(f"\nSelected features DataFrame shape: {selected_features_df.shape}")


if __name__ == '__main__':
    main()
