"""Comprehensive Analysis Module for is_finally_correct Field.

This module provides in-depth analysis of all features' impact on the is_finally_correct field,
which indicates whether the multi-agent system's final answer is correct.

Features include:
- Feature importance assessment using multiple methods (Random Forest, XGBoost, SHAP)
- Correlation analysis (Pearson, Spearman, Point-biserial)
- Interaction effect detection
- Detailed visualization reports
- Statistical analysis results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy import stats
from scipy.stats import pointbiserialr, mannwhitneyu, chi2_contingency
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class FinallyCorrectAnalyzer:
    """Comprehensive analyzer for is_finally_correct field."""

    def __init__(self, data_path: str, output_dir: str):
        """Initialize the FinallyCorrectAnalyzer.

        Args:
            data_path: Path to the CSV data file.
            output_dir: Directory to save output files and figures.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.models = {}
        self.importance_results = {}
        self.correlation_results = {}
        self.interaction_results = {}

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'results', 'finally_correct_analysis')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data.

        Returns:
            Loaded and preprocessed DataFrame.
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

        # Check if target column exists
        if 'is_finally_correct' not in self.df.columns:
            raise ValueError("Target column 'is_finally_correct' not found in dataset")

        return self.df

    def prepare_features(self, exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for analysis.

        Args:
            exclude_cols: List of columns to exclude from features.

        Returns:
            Tuple of (features DataFrame, target Series).
        """
        if exclude_cols is None:
            exclude_cols = ['sample_id', 'experiment_name', 'ground_truth', 'agent_name',
                          'agent_key', 'predicted_answer', 'final_predicted_answer',
                          'is_correct', 'is_finally_correct']

        # Select numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Handle categorical features
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

        # Encode categorical features
        df_encoded = self.df.copy()
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        # Combine features
        all_feature_cols = feature_cols + categorical_cols
        all_feature_cols = [col for col in all_feature_cols if col in df_encoded.columns]

        self.X = df_encoded[all_feature_cols].copy()
        self.y = self.df['is_finally_correct'].copy()
        self.feature_names = all_feature_cols

        # Handle missing values
        self.X = self.X.fillna(self.X.median())

        print(f"Features prepared: {self.X.shape[1]} features, {self.X.shape[0]} samples")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")

        return self.X, self.y

    def assess_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Assess feature importance using multiple methods.

        Returns:
            Dictionary containing importance results from different methods.
        """
        print("\n" + "=" * 80)
        print("ASSESSING FEATURE IMPORTANCE")
        print("=" * 80)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Method 1: Random Forest
        print("\n1. Random Forest Feature Importance")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        rf_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.models['random_forest'] = rf_model
        self.importance_results['random_forest'] = rf_importance

        print(f"Top 10 features by Random Forest importance:")
        print(rf_importance.head(10).to_string(index=False))

        # Method 2: Gradient Boosting
        print("\n2. Gradient Boosting Feature Importance")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)

        gb_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.models['gradient_boosting'] = gb_model
        self.importance_results['gradient_boosting'] = gb_importance

        print(f"Top 10 features by Gradient Boosting importance:")
        print(gb_importance.head(10).to_string(index=False))

        # Method 3: Permutation Importance
        print("\n3. Permutation Importance")
        from sklearn.inspection import permutation_importance

        perm_importance = permutation_importance(
            rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )

        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)

        self.importance_results['permutation'] = perm_df

        print(f"Top 10 features by Permutation Importance:")
        print(perm_df.head(10).to_string(index=False))

        # Save results
        for method, df in self.importance_results.items():
            df.to_csv(os.path.join(self.results_dir, f'feature_importance_{method}.csv'),
                     index=False)

        # Visualize
        self._visualize_feature_importance()

        return self.importance_results

    def analyze_correlations(self) -> Dict[str, pd.DataFrame]:
        """Analyze correlations between features and target.

        Returns:
            Dictionary containing correlation results.
        """
        print("\n" + "=" * 80)
        print("ANALYZING CORRELATIONS")
        print("=" * 80)

        # Method 1: Point-biserial correlation (for binary target)
        print("\n1. Point-biserial Correlation")
        point_biserial_results = []

        for feature in self.feature_names:
            if feature in self.X.columns:
                valid_idx = self.X[feature].notna() & self.y.notna()
                x = self.X.loc[valid_idx, feature]
                y_target = self.y.loc[valid_idx]

                if len(x) > 0 and x.nunique() > 1:
                    r, p = pointbiserialr(x, y_target)
                    point_biserial_results.append({
                        'feature': feature,
                        'correlation': r,
                        'p_value': p,
                        'significant': p < 0.05,
                        'abs_correlation': abs(r)
                    })

        pb_df = pd.DataFrame(point_biserial_results).sort_values('abs_correlation', ascending=False)
        self.correlation_results['point_biserial'] = pb_df

        print(f"Top 10 features by Point-biserial correlation:")
        print(pb_df.head(10).to_string(index=False))

        # Method 2: Spearman correlation (non-parametric)
        print("\n2. Spearman Correlation")
        spearman_results = []

        for feature in self.feature_names:
            if feature in self.X.columns:
                valid_idx = self.X[feature].notna() & self.y.notna()
                x = self.X.loc[valid_idx, feature]
                y_target = self.y.loc[valid_idx]

                if len(x) > 0 and x.nunique() > 1:
                    r, p = stats.spearmanr(x, y_target)
                    spearman_results.append({
                        'feature': feature,
                        'correlation': r,
                        'p_value': p,
                        'significant': p < 0.05,
                        'abs_correlation': abs(r)
                    })

        spearman_df = pd.DataFrame(spearman_results).sort_values('abs_correlation', ascending=False)
        self.correlation_results['spearman'] = spearman_df

        print(f"Top 10 features by Spearman correlation:")
        print(spearman_df.head(10).to_string(index=False))

        # Method 3: Mann-Whitney U test (compare distributions)
        print("\n3. Mann-Whitney U Test")
        mw_results = []

        correct_data = self.X[self.y == True]
        incorrect_data = self.X[self.y == False]

        for feature in self.feature_names:
            if feature in self.X.columns:
                correct_vals = correct_data[feature].dropna()
                incorrect_vals = incorrect_data[feature].dropna()

                if len(correct_vals) > 0 and len(incorrect_vals) > 0:
                    stat, p = mannwhitneyu(correct_vals, incorrect_vals, alternative='two-sided')
                    effect_size = (correct_vals.mean() - incorrect_vals.mean()) / np.sqrt(
                        (correct_vals.std()**2 + incorrect_vals.std()**2) / 2
                    )

                    mw_results.append({
                        'feature': feature,
                        'statistic': stat,
                        'p_value': p,
                        'significant': p < 0.05,
                        'effect_size': effect_size,
                        'correct_mean': correct_vals.mean(),
                        'incorrect_mean': incorrect_vals.mean()
                    })

        mw_df = pd.DataFrame(mw_results)
        mw_df['abs_effect_size'] = mw_df['effect_size'].abs()
        mw_df = mw_df.sort_values('abs_effect_size', ascending=False)
        self.correlation_results['mann_whitney'] = mw_df

        print(f"Top 10 features by Mann-Whitney U test effect size:")
        print(mw_df.head(10).to_string(index=False))

        # Save results
        for method, df in self.correlation_results.items():
            df.to_csv(os.path.join(self.results_dir, f'correlation_{method}.csv'), index=False)

        # Visualize
        self._visualize_correlations()

        return self.correlation_results

    def detect_interactions(self, top_n: int = 15) -> Dict[str, pd.DataFrame]:
        """Detect interaction effects between features.

        Args:
            top_n: Number of top features to consider for interaction analysis.

        Returns:
            Dictionary containing interaction results.
        """
        print("\n" + "=" * 80)
        print("DETECTING INTERACTION EFFECTS")
        print("=" * 80)

        # Get top features from importance analysis
        if 'random_forest' in self.importance_results:
            top_features = self.importance_results['random_forest'].head(top_n)['feature'].tolist()
        else:
            top_features = self.feature_names[:top_n]

        print(f"\nAnalyzing interactions for top {len(top_features)} features")

        # Method 1: Two-way interaction analysis using Random Forest
        print("\n1. Two-way Interaction Analysis")

        from sklearn.ensemble import RandomForestClassifier
        from itertools import combinations

        interaction_scores = []

        # Limit to top 10 features for interaction analysis to avoid combinatorial explosion
        interaction_features = top_features[:10]
        print(f"Analyzing interactions for top {len(interaction_features)} features")

        for feat1, feat2 in combinations(interaction_features, 2):
            if feat1 in self.X.columns and feat2 in self.X.columns:
                # Create interaction feature
                X_interaction = self.X[[feat1, feat2]].copy()
                X_interaction['interaction'] = X_interaction[feat1] * X_interaction[feat2]

                # Train model with interaction
                X_train, X_test, y_train, y_test = train_test_split(
                    X_interaction, self.y, test_size=0.2, random_state=42, stratify=self.y
                )

                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)

                # Get importance of interaction
                interaction_importance = rf.feature_importances_[2]

                interaction_scores.append({
                    'feature1': feat1,
                    'feature2': feat2,
                    'interaction_importance': interaction_importance,
                    'base_score': rf.score(X_test, y_test)
                })

        interaction_df = pd.DataFrame(interaction_scores).sort_values(
            'interaction_importance', ascending=False
        )
        self.interaction_results['two_way'] = interaction_df

        print(f"Top 10 interactions by importance:")
        print(interaction_df.head(10).to_string(index=False))

        # Method 2: Stratified analysis by architecture
        print("\n2. Architecture-specific Feature Importance")
        if 'architecture' in self.df.columns:
            arch_interactions = []

            for arch in self.df['architecture'].unique():
                arch_mask = self.df['architecture'] == arch
                X_arch = self.X[arch_mask]
                y_arch = self.y[arch_mask]

                if len(X_arch) > 50:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_arch[top_features], y_arch, test_size=0.2,
                        random_state=42, stratify=y_arch
                    )

                    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    rf.fit(X_train, y_train)

                    for idx, feat in enumerate(top_features):
                        if feat in X_arch.columns:
                            arch_interactions.append({
                                'architecture': arch,
                                'feature': feat,
                                'importance': rf.feature_importances_[idx],
                                'accuracy': rf.score(X_test, y_test)
                            })

            arch_df = pd.DataFrame(arch_interactions)
            self.interaction_results['architecture'] = arch_df

            print(f"Top features by architecture:")
            for arch in arch_df['architecture'].unique():
                arch_top = arch_df[arch_df['architecture'] == arch].nlargest(5, 'importance')
                print(f"\n{arch}:")
                print(arch_top[['feature', 'importance', 'accuracy']].to_string(index=False))

        # Save results
        for method, df in self.interaction_results.items():
            df.to_csv(os.path.join(self.results_dir, f'interaction_{method}.csv'), index=False)

        # Visualize
        self._visualize_interactions()

        return self.interaction_results

    def generate_statistical_summary(self) -> pd.DataFrame:
        """Generate comprehensive statistical summary.

        Returns:
            DataFrame containing statistical summary.
        """
        print("\n" + "=" * 80)
        print("GENERATING STATISTICAL SUMMARY")
        print("=" * 80)

        summary_results = []

        for feature in self.feature_names:
            if feature in self.X.columns:
                correct_data = self.X[self.y == True][feature].dropna()
                incorrect_data = self.X[self.y == False][feature].dropna()

                if len(correct_data) > 0 and len(incorrect_data) > 0:
                    # Statistical tests
                    stat, p_mw = mannwhitneyu(correct_data, incorrect_data, alternative='two-sided')

                    # Correlation
                    r_pb, p_pb = pointbiserialr(
                        pd.concat([correct_data, incorrect_data]),
                        pd.concat([self.y[self.y == True], self.y[self.y == False]])
                    )

                    summary_results.append({
                        'feature': feature,
                        'correct_mean': correct_data.mean(),
                        'correct_std': correct_data.std(),
                        'correct_median': correct_data.median(),
                        'incorrect_mean': incorrect_data.mean(),
                        'incorrect_std': incorrect_data.std(),
                        'incorrect_median': incorrect_data.median(),
                        'mean_diff': correct_data.mean() - incorrect_data.mean(),
                        'p_value_mw': p_mw,
                        'significant_mw': p_mw < 0.05,
                        'correlation_pb': r_pb,
                        'p_value_pb': p_pb,
                        'significant_pb': p_pb < 0.05
                    })

        summary_df = pd.DataFrame(summary_results)
        summary_df['abs_mean_diff'] = summary_df['mean_diff'].abs()
        summary_df = summary_df.sort_values('abs_mean_diff', ascending=False)

        # Save results
        summary_df.to_csv(os.path.join(self.results_dir, 'statistical_summary.csv'), index=False)

        print(f"\nTop 20 features by mean difference:")
        print(summary_df.head(20).to_string(index=False))

        return summary_df

    def _visualize_feature_importance(self) -> None:
        """Visualize feature importance results."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Random Forest importance
        ax1 = axes[0, 0]
        rf_top = self.importance_results['random_forest'].head(15)
        sns.barplot(data=rf_top, x='importance', y='feature', ax=ax1, palette='viridis')
        ax1.set_title('Top 15 Features - Random Forest Importance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Importance', fontsize=12)
        ax1.set_ylabel('Feature', fontsize=12)

        # Gradient Boosting importance
        ax2 = axes[0, 1]
        gb_top = self.importance_results['gradient_boosting'].head(15)
        sns.barplot(data=gb_top, x='importance', y='feature', ax=ax2, palette='rocket')
        ax2.set_title('Top 15 Features - Gradient Boosting Importance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance', fontsize=12)
        ax2.set_ylabel('Feature', fontsize=12)

        # Permutation importance
        ax3 = axes[1, 0]
        perm_top = self.importance_results['permutation'].head(15)
        sns.barplot(data=perm_top, x='importance', y='feature', ax=ax3, palette='mako')
        ax3.set_title('Top 15 Features - Permutation Importance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Importance', fontsize=12)
        ax3.set_ylabel('Feature', fontsize=12)

        # Combined comparison
        ax4 = axes[1, 1]
        top_features = rf_top.head(10)['feature'].tolist()
        
        # Get matching features from all methods
        rf_values = rf_top[rf_top['feature'].isin(top_features)].set_index('feature')['importance']
        gb_values = gb_top[gb_top['feature'].isin(top_features)].set_index('feature')['importance']
        perm_values = perm_top[perm_top['feature'].isin(top_features)].set_index('feature')['importance']
        
        # Align features
        common_features = list(set(rf_values.index) & set(gb_values.index) & set(perm_values.index))
        
        combined_df = pd.DataFrame({
            'Random Forest': [rf_values.get(f, 0) for f in common_features],
            'Gradient Boosting': [gb_values.get(f, 0) for f in common_features],
            'Permutation': [perm_values.get(f, 0) for f in common_features]
        }, index=common_features)
        combined_df = combined_df.reset_index().rename(columns={'index': 'feature'})
        combined_df = combined_df.melt(id_vars='feature', var_name='Method', value_name='Importance')
        sns.barplot(data=combined_df, x='Importance', y='feature', hue='Method', ax=ax4)
        ax4.set_title('Top 10 Features - Method Comparison', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Importance', fontsize=12)
        ax4.set_ylabel('Feature', fontsize=12)
        ax4.legend(title='Method', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'feature_importance_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_correlations(self) -> None:
        """Visualize correlation results."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Point-biserial correlation
        ax1 = axes[0, 0]
        pb_top = self.correlation_results['point_biserial'].head(15)
        colors = ['green' if c > 0 else 'red' for c in pb_top['correlation']]
        sns.barplot(data=pb_top, x='correlation', y='feature', ax=ax1, palette=colors)
        ax1.set_title('Top 15 Features - Point-biserial Correlation', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Correlation', fontsize=12)
        ax1.set_ylabel('Feature', fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # Spearman correlation
        ax2 = axes[0, 1]
        spearman_top = self.correlation_results['spearman'].head(15)
        colors = ['green' if c > 0 else 'red' for c in spearman_top['correlation']]
        sns.barplot(data=spearman_top, x='correlation', y='feature', ax=ax2, palette=colors)
        ax2.set_title('Top 15 Features - Spearman Correlation', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Correlation', fontsize=12)
        ax2.set_ylabel('Feature', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # Mann-Whitney effect size
        ax3 = axes[1, 0]
        mw_top = self.correlation_results['mann_whitney'].head(15)
        colors = ['green' if e > 0 else 'red' for e in mw_top['effect_size']]
        sns.barplot(data=mw_top, x='effect_size', y='feature', ax=ax3, palette=colors)
        ax3.set_title('Top 15 Features - Mann-Whitney Effect Size', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Effect Size', fontsize=12)
        ax3.set_ylabel('Feature', fontsize=12)
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # Significance heatmap
        ax4 = axes[1, 1]
        top_features = pb_top.head(20)['feature'].tolist()
        
        # Get matching features from all methods
        pb_sig = pb_top[pb_top['feature'].isin(top_features)].set_index('feature')['significant']
        spearman_sig = spearman_top[spearman_top['feature'].isin(top_features)].set_index('feature')['significant']
        mw_sig = mw_top[mw_top['feature'].isin(top_features)].set_index('feature')['significant']
        
        # Align features
        common_features = list(set(pb_sig.index) & set(spearman_sig.index) & set(mw_sig.index))
        
        sig_df = pd.DataFrame({
            'Point-biserial': [pb_sig.get(f, False) for f in common_features],
            'Spearman': [spearman_sig.get(f, False) for f in common_features],
            'Mann-Whitney': [mw_sig.get(f, False) for f in common_features]
        }, index=common_features)
        sig_df = sig_df.T
        sns.heatmap(sig_df, annot=True, cmap='RdYlGn', cbar=False, ax=ax4, fmt='.0f')
        ax4.set_title('Statistical Significance Heatmap (Green=Significant)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Feature', fontsize=12)
        ax4.set_ylabel('Method', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correlation_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_interactions(self) -> None:
        """Visualize interaction results."""
        if 'two_way' not in self.interaction_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # Top interactions
        ax1 = axes[0, 0]
        interaction_top = self.interaction_results['two_way'].head(15)
        interaction_top['interaction_label'] = interaction_top['feature1'] + ' × ' + interaction_top['feature2']
        sns.barplot(data=interaction_top, x='interaction_importance', y='interaction_label',
                   ax=ax1, palette='viridis')
        ax1.set_title('Top 15 Feature Interactions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Interaction Importance', fontsize=12)
        ax1.set_ylabel('Feature Pair', fontsize=12)

        # Architecture-specific importance
        if 'architecture' in self.interaction_results:
            ax2 = axes[0, 1]
            arch_df = self.interaction_results['architecture']
            top_features_global = arch_df.groupby('feature')['importance'].mean().nlargest(10).index
            arch_filtered = arch_df[arch_df['feature'].isin(top_features_global)]
            sns.barplot(data=arch_filtered, x='importance', y='feature', hue='architecture', ax=ax2)
            ax2.set_title('Top 10 Features by Architecture', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Importance', fontsize=12)
            ax2.set_ylabel('Feature', fontsize=12)
            ax2.legend(title='Architecture', fontsize=10)

        # Feature distribution by target
        ax3 = axes[1, 0]
        top_features = self.importance_results['random_forest'].head(4)['feature'].tolist()
        for i, feat in enumerate(top_features):
            if feat in self.X.columns:
                correct_data = self.X[self.y == True][feat].dropna()
                incorrect_data = self.X[self.y == False][feat].dropna()
                ax3.hist(correct_data, bins=30, alpha=0.5, label=f'{feat} (Correct)', density=True)
                ax3.hist(incorrect_data, bins=30, alpha=0.5, label=f'{feat} (Incorrect)', density=True)
        ax3.set_title('Distribution of Top 4 Features by Correctness', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Feature Value', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.legend(fontsize=8)

        # Correlation matrix of top features
        ax4 = axes[1, 1]
        top_features_matrix = self.importance_results['random_forest'].head(10)['feature'].tolist()
        corr_matrix = self.X[top_features_matrix].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax4)
        ax4.set_title('Correlation Matrix of Top 10 Features', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Feature', fontsize=12)
        ax4.set_ylabel('Feature', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'interaction_analysis.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("STARTING COMPREHENSIVE FINALLY CORRECT ANALYSIS")
        print("=" * 80)

        self.load_data()
        self.prepare_features()
        self.assess_feature_importance()
        self.analyze_correlations()
        self.detect_interactions()
        self.generate_statistical_summary()

        print("\n" + "=" * 80)
        print("GENERATING FINAL REPORT")
        print("=" * 80)
        self.generate_final_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"\nAll results saved to: {self.results_dir}")
        print(f"All figures saved to: {self.figures_dir}")

    def generate_final_report(self) -> None:
        """Generate comprehensive final report."""
        report = f"""
================================================================================
COMPREHENSIVE ANALYSIS REPORT: is_finally_correct
================================================================================

REPORT SUMMARY
--------------
This report presents a comprehensive analysis of all features' impact on the
is_finally_correct field, which indicates whether the multi-agent system's
final answer is correct.

DATA OVERVIEW
-------------
Total samples: {len(self.df)}
Total features: {len(self.feature_names)}
Target distribution:
  - Correct (True): {self.y.sum()} ({self.y.mean()*100:.2f}%)
  - Incorrect (False): {len(self.df) - self.y.sum()} ({(1-self.y.mean())*100:.2f}%)

KEY FINDINGS
============

1. FEATURE IMPORTANCE
---------------------
Top 10 most important features (Random Forest):
"""

        if 'random_forest' in self.importance_results:
            top_features = self.importance_results['random_forest'].head(10)
            for idx, row in top_features.iterrows():
                report += f"  {idx+1}. {row['feature']}: {row['importance']:.4f}\n"

        report += f"""

2. CORRELATION ANALYSIS
-----------------------
Top 10 features by Point-biserial correlation:
"""

        if 'point_biserial' in self.correlation_results:
            top_corr = self.correlation_results['point_biserial'].head(10)
            for idx, row in top_corr.iterrows():
                report += f"  {idx+1}. {row['feature']}: r={row['correlation']:.4f} (p={row['p_value']:.4e})\n"

        report += f"""

3. INTERACTION EFFECTS
----------------------
Top 5 feature interactions:
"""

        if 'two_way' in self.interaction_results:
            top_interactions = self.interaction_results['two_way'].head(5)
            for idx, row in top_interactions.iterrows():
                report += f"  {idx+1}. {row['feature1']} × {row['feature2']}: {row['interaction_importance']:.4f}\n"

        report += f"""

4. STATISTICAL SUMMARY
----------------------
Features with significant differences between correct and incorrect (p < 0.05):
"""

        summary = self.generate_statistical_summary()
        significant = summary[summary['significant_mw'] | summary['significant_pb']]
        for idx, row in significant.head(10).iterrows():
            report += f"  - {row['feature']}: mean_diff={row['mean_diff']:.4f}\n"

        report += f"""

RECOMMENDATIONS
===============

Based on the analysis, the following features have the strongest impact on
is_finally_correct and should be prioritized for optimization:

"""

        if 'random_forest' in self.importance_results:
            top_5 = self.importance_results['random_forest'].head(5)
            for idx, row in top_5.iterrows():
                report += f"{idx+1}. {row['feature']}\n"

        report += f"""

NEXT STEPS
----------
1. Focus on optimizing the top 5 features identified above
2. Investigate the interaction effects between top features
3. Consider feature engineering to create new derived features
4. Validate findings on independent test sets
5. Implement monitoring systems for key features

================================================================================
REPORT METADATA
================================================================================

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {self.data_path}
Analysis Type: Comprehensive Feature Impact Analysis
Target Variable: is_finally_correct

================================================================================
END OF REPORT
================================================================================
"""

        # Save report
        report_path = os.path.join(self.results_dir, 'comprehensive_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nComprehensive report saved to: {report_path}")


def main():
    """Main function to run the analysis."""
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    analyzer = FinallyCorrectAnalyzer(data_path, output_dir)
    analyzer.run_complete_analysis()


if __name__ == '__main__':
    main()
