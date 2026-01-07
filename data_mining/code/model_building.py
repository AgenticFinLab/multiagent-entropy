"""Data Mining Model Building Module for Multi-Agent Entropy Research.

This module provides comprehensive model building including:
- Selection of appropriate ML/statistical models for entropy-performance relationship
- Train-test splitting, model training, and hyperparameter tuning
- Model performance evaluation using appropriate metrics (R², MAE, RMSE, etc.)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class ModelBuilder:
    """Class for building and evaluating data mining models for multi-agent entropy research."""

    def __init__(self, data_path: str, output_dir: str):
        """Initialize the ModelBuilder.

        Args:
            data_path: Path to the CSV data file.
            output_dir: Directory to save output files and figures.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        self.results_dir = os.path.join(output_dir, 'results', 'model_building')
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        self.models_dir = os.path.join(self.results_dir, 'models')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file.

        Returns:
            Loaded DataFrame.
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def prepare_features_and_target(self, target_col: str = 'is_correct',
                                    feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling.

        Args:
            target_col: Target column name (e.g., 'is_correct', 'is_finally_correct').
            feature_cols: List of feature column names. If None, use all numeric columns except target.

        Returns:
            Tuple of (features DataFrame, target Series).
        """
        if feature_cols is None:
            # Use entropy-related features
            feature_cols = [
                'sample_mean_entropy', 'sample_total_entropy', 'sample_std_entropy',
                'sample_max_entropy', 'sample_min_entropy', 'sample_q1_entropy', 'sample_q3_entropy',
                'agent_mean_entropy', 'agent_total_entropy', 'agent_std_entropy',
                'exp_avg_entropy', 'exp_total_entropy', 'time_cost', 'execution_order',
                'sample_token_count'
            ]

        # Filter to only include columns that exist in the dataframe
        feature_cols = [col for col in feature_cols if col in self.df.columns]

        X = self.df[feature_cols].copy()
        y = self.df[target_col].copy()

        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median() if y.dtype != 'object' else y.mode()[0])

        # Remove rows with missing target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Target column: {target_col}")

        return X, y

    def compare_correctness_targets(self) -> pd.DataFrame:
        """Compare agent-level correctness (is_correct) with final correctness (is_finally_correct).

        This method analyzes the relationship between individual agent correctness
        and the final answer correctness after multi-agent collaboration.

        Returns:
            DataFrame containing comparison statistics.
        """
        print("\n" + "=" * 80)
        print("COMPARING AGENT-LEVEL VS FINAL CORRECTNESS")
        print("=" * 80)

        if 'is_correct' not in self.df.columns or 'is_finally_correct' not in self.df.columns:
            print("Warning: 'is_correct' or 'is_finally_correct' columns not found in data")
            return None

        # Create comparison DataFrame
        comparison_df = self.df[['is_correct', 'is_finally_correct']].copy()

        # Calculate statistics
        agent_correct_rate = comparison_df['is_correct'].mean()
        final_correct_rate = comparison_df['is_finally_correct'].mean()

        # Calculate cases where agent was correct but final was wrong
        agent_correct_final_wrong = ((comparison_df['is_correct'] == True) & 
                                    (comparison_df['is_finally_correct'] == False)).sum()

        # Calculate cases where agent was wrong but final was correct
        agent_wrong_final_correct = ((comparison_df['is_correct'] == False) & 
                                    (comparison_df['is_finally_correct'] == True)).sum()

        # Calculate agreement rate
        agreement_rate = (comparison_df['is_correct'] == comparison_df['is_finally_correct']).mean()

        # Create summary statistics
        stats_df = pd.DataFrame({
            'Metric': [
                'Agent-level Correct Rate',
                'Final Correct Rate',
                'Agreement Rate',
                'Agent Correct / Final Wrong Count',
                'Agent Wrong / Final Correct Count',
                'Total Samples'
            ],
            'Value': [
                agent_correct_rate,
                final_correct_rate,
                agreement_rate,
                agent_correct_final_wrong,
                agent_wrong_final_correct,
                len(comparison_df)
            ]
        })

        print(f"\nAgent-level Correct Rate: {agent_correct_rate:.4f}")
        print(f"Final Correct Rate: {final_correct_rate:.4f}")
        print(f"Agreement Rate: {agreement_rate:.4f}")
        print(f"Agent Correct / Final Wrong: {agent_correct_final_wrong}")
        print(f"Agent Wrong / Final Correct: {agent_wrong_final_correct}")

        # Create visualization
        self._visualize_correctness_comparison(comparison_df)

        # Save results
        stats_df.to_csv(os.path.join(self.results_dir, 'correctness_comparison.csv'),
                       index=False)

        return stats_df

    def _visualize_correctness_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Create visualization for correctness comparison.

        Args:
            comparison_df: DataFrame containing is_correct and is_finally_correct columns.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Correctness rate comparison
        ax1 = axes[0, 0]
        rates = [comparison_df['is_correct'].mean(), comparison_df['is_finally_correct'].mean()]
        ax1.bar(['Agent-level', 'Final'], rates, color=['#3498db', '#e74c3c'], alpha=0.8)
        ax1.set_ylabel('Correct Rate')
        ax1.set_title('Correctness Rate Comparison')
        ax1.set_ylim([0, 1])
        for i, v in enumerate(rates):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Confusion matrix-like visualization
        ax2 = axes[0, 1]
        cm_data = np.zeros((2, 2))
        cm_data[0, 0] = ((comparison_df['is_correct'] == False) & 
                         (comparison_df['is_finally_correct'] == False)).sum()
        cm_data[0, 1] = ((comparison_df['is_correct'] == False) & 
                         (comparison_df['is_finally_correct'] == True)).sum()
        cm_data[1, 0] = ((comparison_df['is_correct'] == True) & 
                         (comparison_df['is_finally_correct'] == False)).sum()
        cm_data[1, 1] = ((comparison_df['is_correct'] == True) & 
                         (comparison_df['is_finally_correct'] == True)).sum()

        sns.heatmap(cm_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax2,
                   xticklabels=['Final Wrong', 'Final Correct'],
                   yticklabels=['Agent Wrong', 'Agent Correct'])
        ax2.set_xlabel('Final Correctness')
        ax2.set_ylabel('Agent Correctness')
        ax2.set_title('Agent vs Final Correctness Matrix')

        # Distribution of correctness by sample
        ax3 = axes[1, 0]
        sample_stats = comparison_df.groupby('is_finally_correct')['is_correct'].mean()
        sample_stats.plot(kind='bar', ax=ax3, color=['#e74c3c', '#2ecc71'], alpha=0.8)
        ax3.set_ylabel('Agent Correct Rate')
        ax3.set_xlabel('Final Correctness')
        ax3.set_title('Agent Correct Rate by Final Correctness')
        ax3.set_xticklabels(['Wrong', 'Correct'], rotation=0)
        ax3.grid(True, alpha=0.3)

        # Stacked bar chart
        ax4 = axes[1, 1]
        stacked_data = pd.DataFrame({
            'Agent Wrong': [cm_data[0, 0], cm_data[0, 1]],
            'Agent Correct': [cm_data[1, 0], cm_data[1, 1]]
        }, index=['Final Wrong', 'Final Correct'])
        stacked_data.plot(kind='bar', stacked=True, ax=ax4, color=['#e74c3c', '#2ecc71'], alpha=0.8)
        ax4.set_ylabel('Count')
        ax4.set_xlabel('Final Correctness')
        ax4.set_title('Stacked Agent Correctness by Final Correctness')
        ax4.legend(title='Agent Correctness')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'correctness_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                  test_size: float = 0.2, random_state: int = 42) -> None:
        """Split data into training and testing sets.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == 'object' or y.nunique() <= 2 else None
        )

        print(f"\nData split:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Testing set: {self.X_test.shape[0]} samples")

    def scale_features(self) -> None:
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        print("Features scaled using StandardScaler")

    def build_regression_models(self) -> Dict[str, Any]:
        """Build and train regression models.

        Returns:
            Dictionary of trained models.
        """
        print("\n" + "=" * 80)
        print("BUILDING REGRESSION MODELS")
        print("=" * 80)

        regression_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=1.0, random_state=42),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0),
            'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }

        for name, model in regression_models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                print(f"  {name} trained successfully")
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")

        return self.models

    def build_classification_models(self) -> Dict[str, Any]:
        """Build and train classification models.

        Returns:
            Dictionary of trained models.
        """
        print("\n" + "=" * 80)
        print("BUILDING CLASSIFICATION MODELS")
        print("=" * 80)

        classification_models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVC': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            'MLPClassifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }

        for name, model in classification_models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                print(f"  {name} trained successfully")
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")

        return self.models

    def evaluate_regression_models(self) -> pd.DataFrame:
        """Evaluate regression models.

        Returns:
            DataFrame containing evaluation metrics for all models.
        """
        print("\n" + "=" * 80)
        print("EVALUATING REGRESSION MODELS")
        print("=" * 80)

        regression_model_names = ['LinearRegression', 'Ridge', 'Lasso', 'RandomForestRegressor', 
                                   'GradientBoostingRegressor', 'SVR', 'MLPRegressor']
        results = []

        for name, model in self.models.items():
            if name not in regression_model_names:
                continue
                
            print(f"\nEvaluating {name}...")

            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

            # Metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)

            # Cross-validation
            try:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = np.nan
                cv_std = np.nan

            results.append({
                'Model': name,
                'Train_MSE': train_mse,
                'Test_MSE': test_mse,
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'CV_R2_Mean': cv_mean,
                'CV_R2_Std': cv_std
            })

            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            print(f"  CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

        results_df = pd.DataFrame(results)
        self.results['regression'] = results_df

        # Save results
        results_df.to_csv(os.path.join(self.results_dir, 'regression_evaluation.csv'),
                         index=False)

        # Find best model
        best_idx = results_df['Test_R2'].idxmax()
        self.best_model = self.models[results_df.loc[best_idx, 'Model']]
        self.best_model_name = results_df.loc[best_idx, 'Model']

        print(f"\nBest regression model: {self.best_model_name} (Test R²: {results_df.loc[best_idx, 'Test_R2']:.4f})")

        # Visualize results
        self._visualize_regression_results(results_df)

        return results_df

    def evaluate_classification_models(self) -> pd.DataFrame:
        """Evaluate classification models.

        Returns:
            DataFrame containing evaluation metrics for all models.
        """
        print("\n" + "=" * 80)
        print("EVALUATING CLASSIFICATION MODELS")
        print("=" * 80)

        classification_model_names = ['LogisticRegression', 'RandomForestClassifier', 
                                        'GradientBoostingClassifier', 'SVC', 'MLPClassifier']
        results = []

        for name, model in self.models.items():
            if name not in classification_model_names:
                continue
                
            print(f"\nEvaluating {name}...")

            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

            # Metrics
            train_acc = accuracy_score(self.y_train, y_train_pred)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            test_precision = precision_score(self.y_test, y_test_pred, average='weighted')
            test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
            test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')

            # Cross-validation
            try:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = np.nan
                cv_std = np.nan

            # ROC AUC (if binary classification)
            try:
                if hasattr(model, 'predict_proba'):
                    y_test_proba = model.predict_proba(self.X_test)[:, 1]
                    test_roc_auc = roc_auc_score(self.y_test, y_test_proba)
                else:
                    test_roc_auc = np.nan
            except:
                test_roc_auc = np.nan

            results.append({
                'Model': name,
                'Train_Accuracy': train_acc,
                'Test_Accuracy': test_acc,
                'Test_Precision': test_precision,
                'Test_Recall': test_recall,
                'Test_F1': test_f1,
                'CV_Accuracy_Mean': cv_mean,
                'CV_Accuracy_Std': cv_std,
                'Test_ROC_AUC': test_roc_auc
            })

            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

        results_df = pd.DataFrame(results)
        self.results['classification'] = results_df

        # Save results
        results_df.to_csv(os.path.join(self.results_dir, 'classification_evaluation.csv'),
                         index=False)

        # Find best model
        best_idx = results_df['Test_Accuracy'].idxmax()
        self.best_model = self.models[results_df.loc[best_idx, 'Model']]
        self.best_model_name = results_df.loc[best_idx, 'Model']

        print(f"\nBest classification model: {self.best_model_name} (Test Accuracy: {results_df.loc[best_idx, 'Test_Accuracy']:.4f})")

        # Visualize results
        self._visualize_classification_results(results_df)

        return results_df

    def _visualize_regression_results(self, results_df: pd.DataFrame) -> None:
        """Create visualization for regression model results.

        Args:
            results_df: DataFrame containing evaluation metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # R² comparison
        ax1 = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35
        ax1.bar(x - width/2, results_df['Train_R2'], width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, results_df['Test_R2'], width, label='Test', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('R² Score')
        ax1.set_title('R² Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, results_df['Train_MAE'], width, label='Train', alpha=0.8)
        ax2.bar(x + width/2, results_df['Test_MAE'], width, label='Test', alpha=0.8)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('MAE')
        ax2.set_title('MAE Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Predicted vs Actual for best model
        ax3 = axes[1, 0]
        y_pred = self.best_model.predict(self.X_test)
        ax3.scatter(self.y_test, y_pred, alpha=0.6, s=20)
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values')
        ax3.set_title(f'Predicted vs Actual ({self.best_model_name})')
        ax3.grid(True, alpha=0.3)

        # Residual plot
        ax4 = axes[1, 1]
        residuals = self.y_test - y_pred
        ax4.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax4.axhline(y=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title(f'Residual Plot ({self.best_model_name})')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'regression_results.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_classification_results(self, results_df: pd.DataFrame) -> None:
        """Create visualization for classification model results.

        Args:
            results_df: DataFrame containing evaluation metrics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Accuracy comparison
        ax1 = axes[0, 0]
        x = np.arange(len(results_df))
        width = 0.35
        ax1.bar(x - width/2, results_df['Train_Accuracy'], width, label='Train', alpha=0.8)
        ax1.bar(x + width/2, results_df['Test_Accuracy'], width, label='Test', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1, Precision, Recall comparison
        ax2 = axes[0, 1]
        x = np.arange(len(results_df))
        width = 0.25
        ax2.bar(x - width, results_df['Test_Precision'], width, label='Precision', alpha=0.8)
        ax2.bar(x, results_df['Test_Recall'], width, label='Recall', alpha=0.8)
        ax2.bar(x + width, results_df['Test_F1'], width, label='F1', alpha=0.8)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision, Recall, F1 Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Confusion matrix for best model
        ax3 = axes[1, 0]
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title(f'Confusion Matrix ({self.best_model_name})')

        # ROC curve for best model (if binary classification)
        ax4 = axes[1, 1]
        if hasattr(self.best_model, 'predict_proba') and self.y_test.nunique() == 2:
            y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = results_df.loc[results_df['Model'] == self.best_model_name, 'Test_ROC_AUC'].values[0]
            ax4.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.4f})')
            ax4.plot([0, 1], [0, 1], 'k--', linewidth=2)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title(f'ROC Curve ({self.best_model_name})')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'ROC curve not available\nfor multi-class or non-probabilistic models',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'ROC Curve ({self.best_model_name})')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'classification_results.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def hyperparameter_tuning(self, model_name: str = 'RandomForestRegressor',
                             n_iter: int = 50) -> Dict[str, Any]:
        """Perform hyperparameter tuning for a specific model.

        Args:
            model_name: Name of the model to tune.
            n_iter: Number of iterations for RandomizedSearchCV.

        Returns:
            Dictionary containing best parameters and best score.
        """
        print(f"\n" + "=" * 80)
        print(f"HYPERPARAMETER TUNING FOR {model_name}")
        print("=" * 80)

        if model_name == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            scoring = 'r2'
        elif model_name == 'RandomForestClassifier':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            scoring = 'accuracy'
        elif model_name == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(random_state=42)
            param_dist = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            scoring = 'r2'
        elif model_name == 'GradientBoostingClassifier':
            model = GradientBoostingClassifier(random_state=42)
            param_dist = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            scoring = 'accuracy'
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return {}

        random_search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=n_iter,
            cv=5, scoring=scoring, n_jobs=-1, random_state=42
        )

        print(f"Running RandomizedSearchCV with {n_iter} iterations...")
        random_search.fit(self.X_train, self.y_train)

        print(f"\nBest parameters: {random_search.best_params_}")
        print(f"Best {scoring} score: {random_search.best_score_:.4f}")

        # Update best model
        self.best_model = random_search.best_estimator_
        self.best_model_name = model_name

        # Save results
        tuning_results = {
            'model_name': model_name,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_
        }

        pd.DataFrame([tuning_results]).to_csv(
            os.path.join(self.results_dir, f'hyperparameter_tuning_{model_name}.csv'),
            index=False
        )

        return tuning_results

    def feature_importance_analysis(self) -> pd.DataFrame:
        """Analyze feature importance from the best model.

        Returns:
            DataFrame containing feature importance scores.
        """
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.X_train.columns

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Save results
            importance_df.to_csv(
                os.path.join(self.results_dir, 'best_model_feature_importance.csv'),
                index=False
            )

            # Visualize
            self._visualize_feature_importance(importance_df)

            print("\nTop 10 most important features:")
            print(importance_df.head(10).to_string(index=False))

            return importance_df
        else:
            print(f"Feature importance not available for {self.best_model_name}")
            return pd.DataFrame()

    def _visualize_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """Create visualization for feature importance.

        Args:
            importance_df: DataFrame containing feature importance scores.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        top_features = importance_df.head(15)
        ax.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance ({self.best_model_name})')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'best_model_feature_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_model_report(self) -> None:
        """Generate comprehensive model building report."""
        report_path = os.path.join(self.results_dir, 'model_building_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL BUILDING REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Training samples: {self.X_train.shape[0]}\n")
            f.write(f"Testing samples: {self.X_test.shape[0]}\n")
            f.write(f"Features: {self.X_train.shape[1]}\n\n")

            f.write("-" * 80 + "\n")
            f.write("BEST MODEL\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model: {self.best_model_name}\n")
            f.write(f"Type: {type(self.best_model).__name__}\n\n")

            if 'regression' in self.results:
                f.write("-" * 80 + "\n")
                f.write("REGRESSION MODEL RESULTS\n")
                f.write("-" * 80 + "\n")
                f.write(self.results['regression'].to_string(index=False))

            if 'classification' in self.results:
                f.write("\n\n" + "-" * 80 + "\n")
                f.write("CLASSIFICATION MODEL RESULTS\n")
                f.write("-" * 80 + "\n")
                f.write(self.results['classification'].to_string(index=False))

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\nModel building report saved to: {report_path}")

    def run_model_building(self, target_col: str = 'is_correct',
                          task_type: str = 'auto',
                          tune_hyperparameters: bool = False) -> None:
        """Run complete model building pipeline.

        Args:
            target_col: Target column name.
            task_type: Type of task ('regression', 'classification', 'auto').
            tune_hyperparameters: Whether to perform hyperparameter tuning.
        """
        print("=" * 80)
        print("STARTING MODEL BUILDING")
        print("=" * 80)

        self.load_data()
        X, y = self.prepare_features_and_target(target_col=target_col)
        self.split_data(X, y)
        self.scale_features()

        # Determine task type
        if task_type == 'auto':
            if y.dtype == 'object' or y.nunique() <= 2:
                task_type = 'classification'
            else:
                task_type = 'regression'

        print(f"\nTask type: {task_type}")

        if task_type == 'regression':
            self.build_regression_models()
            self.evaluate_regression_models()
        else:
            self.build_classification_models()
            self.evaluate_classification_models()

        # Hyperparameter tuning
        if tune_hyperparameters:
            self.hyperparameter_tuning(model_name=self.best_model_name)

        # Feature importance analysis
        self.feature_importance_analysis()

        # Generate report
        self.generate_model_report()

        print("\n" + "=" * 80)
        print("MODEL BUILDING COMPLETED")
        print("=" * 80)
        print(f"\nAll results saved to: {self.output_dir}")


def main():
    """Main function to run model building."""
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    builder = ModelBuilder(data_path, output_dir)
    builder.run_model_building(
        target_col='is_correct',
        task_type='auto',
        tune_hyperparameters=True
    )


if __name__ == '__main__':
    main()
