"""
SHAP Analyzer for Multi-Agent Entropy Analysis

This module performs SHAP (SHapley Additive exPlanations) analysis to interpret 
machine learning model predictions for both regression and classification tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import utilities
from utils import (
    setup_visualization_style,
    load_data_from_path,
    encode_categorical_features,
    prepare_features,
    create_output_directory,
    determine_output_directory,
    get_default_data_path
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ShapAnalyzer:
    """Performs SHAP analysis to interpret model predictions for multi-agent entropy data."""

    def __init__(
        self,
        data_path: str = None,
        output_dir: str = None,
        target_dataset: str = None,
    ):
        """
        Initialize the ShapAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file
            output_dir: Directory to save SHAP analysis results
            target_dataset: Target dataset name for determining output directory
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Please install it with: pip install shap")
        
        if data_path is None:
            data_path = get_default_data_path()

        # Determine output directory based on target_dataset
        if output_dir is None:
            output_dir = determine_output_directory(
                "data_mining/results", target_dataset, "shap"
            )

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.target_dataset = target_dataset
        self.df = None
        self.results = {}

        # Create output directory if it doesn't exist
        create_output_directory(self.output_dir)

        # Set style for better visualizations
        setup_visualization_style()

    def load_data(self) -> pd.DataFrame:
        """
        Load the merged dataset.

        Returns:
            Loaded DataFrame
        """
        self.df = load_data_from_path(self.data_path)
        
        return self.df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode non-numeric features to numeric values (0, 1, 2, 3, 4, ...).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with encoded categorical features
        """
        return encode_categorical_features(df)

    def prepare_features(
        self,
        target_column: str = "exp_accuracy",
        exclude_columns: List[str] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for SHAP analysis.

        Args:
            target_column: Name of the target column 
            exclude_columns: List of columns to exclude from features

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.df is None:
            self.load_data()

        # Encode categorical features
        self.df = self.encode_categorical_features(self.df)

        if exclude_columns is None:
            # Use the same exclude columns as other analyzers
            exclude_columns = [
                "dataset", "model_name", "sample_id"
            ]

        # Use utility function to prepare features
        X, y = prepare_features(self.df, target_column, exclude_columns)

        return X, y

    def explain_model(
        self, 
        model, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        model_name: str,
        task_type: str = "regression"
    ) -> Dict:
        """
        Generate SHAP explanations for a trained model.

        Args:
            model: Trained model to explain
            X_train: Training features
            X_test: Test features
            model_name: Name of the model
            task_type: Type of task ('regression' or 'classification')

        Returns:
            Dictionary containing SHAP explanations and plots
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP is not available")
            return {}

        logger.info(f"Generating SHAP explanations for {model_name} ({task_type})...")

        # Create SHAP explainer based on model type
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # For binary classification, TreeExplainer returns a list of arrays
            if task_type == "classification" and isinstance(shap_values, list):
                # For binary classification, take the positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                # Ensure shap_values is a numpy array after extraction
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    shap_values = np.asarray(shap_values[0])
        else:
            # Use KernelExplainer as fallback for other model types
            explainer = shap.KernelExplainer(model.predict, X_train.sample(min(100, len(X_train))))
            shap_values = explainer.shap_values(X_test.sample(min(100, len(X_test))))

        # Generate SHAP plots
        plots_info = self._generate_shap_plots(
            shap_values, X_test, model_name, task_type
        )

        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "plots_info": plots_info,
            "feature_names": list(X_test.columns)
        }

    def _generate_shap_plots(
        self, 
        shap_values, 
        X_test: pd.DataFrame, 
        model_name: str, 
        task_type: str
    ) -> Dict:
        """
        Generate various SHAP plots for model interpretation.

        Args:
            shap_values: SHAP values computed by the explainer
            X_test: Test features
            model_name: Name of the model
            task_type: Type of task ('regression' or 'classification')

        Returns:
            Dictionary with paths to saved plots
        """
        plots_info = {}

        # 1. SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_test, 
            plot_type="bar", 
            show=False,
            max_display=20
        )
        summary_plot_path = self.output_dir / f"shap_summary_{model_name}_{task_type}.png"
        plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots_info["summary_plot"] = str(summary_plot_path)
        logger.info(f"SHAP summary plot saved: {summary_plot_path}")

        # 2. SHAP Feature Importance Plot (dot plot)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_test, 
            plot_type="dot", 
            show=False,
            max_display=20
        )
        importance_plot_path = self.output_dir / f"shap_importance_{model_name}_{task_type}.png"
        plt.savefig(importance_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots_info["importance_plot"] = str(importance_plot_path)
        logger.info(f"SHAP importance plot saved: {importance_plot_path}")

        # 3. SHAP Dependence Plots for top 5 important features
        # Calculate mean absolute SHAP values for feature importance
        if isinstance(shap_values, list):
            # For multi-output models (e.g., multi-class classification)
            # Use the first output for feature importance calculation
            abs_shap_values = np.abs(shap_values[0]) if len(shap_values) > 0 else np.abs(shap_values)
        else:
            abs_shap_values = np.abs(shap_values)
        
        # Ensure abs_shap_values is a numpy array
        if isinstance(abs_shap_values, list):
            abs_shap_values = np.asarray(abs_shap_values)
        
        # Calculate mean absolute SHAP values for each feature
        if abs_shap_values.ndim > 1:
            feature_importance = abs_shap_values.mean(0)
        else:
            feature_importance = abs_shap_values
        
        # Get top 5 features based on mean absolute SHAP value
        if isinstance(feature_importance, np.ndarray) and len(feature_importance) > 0:
            top_indices = np.argsort(feature_importance)[-5:][::-1]
            # Ensure top_indices is a list of scalars
            if hasattr(top_indices, '__iter__') and not isinstance(top_indices, (str, bytes)):
                top_indices = [int(i) for i in top_indices.flatten()]
            else:
                top_indices = [int(top_indices)]
        else:
            # If we can't determine importance, just take the first few features
            top_indices = range(min(5, len(X_test.columns)))
        
        dependence_plots = []
        for idx in top_indices:
            # Convert idx to integer if it's an array-like object
            if hasattr(idx, 'item'):
                idx_scalar = int(idx.item())
            elif hasattr(idx, '__int__'):
                idx_scalar = int(idx)
            else:
                idx_scalar = int(idx)
            if idx_scalar < len(X_test.columns):
                feature_name = X_test.columns[idx_scalar]
                
                plt.figure(figsize=(10, 6))
                # Get index of the feature
                feature_idx = list(X_test.columns).index(feature_name)
                
                # Create a temporary explainer for the dependence plot
                if isinstance(shap_values, list):
                    # Handle multi-output case
                    temp_shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
                    # Ensure it's an array
                    if isinstance(temp_shap_values, list):
                        temp_shap_values = np.asarray(temp_shap_values)
                    # Extract values for the specific feature
                    feature_shap_values = temp_shap_values[:, feature_idx] if temp_shap_values.ndim > 1 else temp_shap_values
                    # Create Explanation object for shap.plots.scatter
                    explanation = shap.Explanation(
                        values=feature_shap_values,
                        base_values=np.zeros(len(feature_shap_values)),
                        data=X_test[feature_name].values,
                        feature_names=[feature_name]
                    )
                    shap.plots.scatter(explanation, show=False)
                else:
                    # Extract values for the specific feature
                    feature_shap_values = shap_values[:, feature_idx] if shap_values.ndim > 1 else shap_values
                    # Create Explanation object for shap.plots.scatter
                    explanation = shap.Explanation(
                        values=feature_shap_values,
                        base_values=np.zeros(len(feature_shap_values)),
                        data=X_test[feature_name].values,
                        feature_names=[feature_name]
                    )
                    shap.plots.scatter(explanation, show=False)
                
                # Create subdirectory for dependence plots
                (self.output_dir / "shap_dependence_plots").mkdir(exist_ok=True)
                dep_plot_path = self.output_dir / "shap_dependence_plots" / f"shap_dependence_{feature_name}_{model_name}_{task_type}.png"
                plt.savefig(dep_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                dependence_plots.append(str(dep_plot_path))
                logger.info(f"SHAP dependence plot saved: {dep_plot_path}")
        
        plots_info["dependence_plots"] = dependence_plots

        # 4. SHAP Waterfall Plot for a sample prediction (first test instance)
        plt.figure(figsize=(12, 8))
        try:
            if isinstance(shap_values, list):
                # Handle multi-output case (e.g., classification)
                shap_values_for_plot = shap_values[0] if len(shap_values) > 0 else shap_values
            else:
                shap_values_for_plot = shap_values
            
            # Ensure shap_values_for_plot is a numpy array
            if isinstance(shap_values_for_plot, list):
                shap_values_for_plot = np.asarray(shap_values_for_plot)
            
            # Check dimensions and create appropriate explanation object
            if shap_values_for_plot.ndim == 1:
                # Single instance, single output
                explanation = shap.Explanation(
                    values=shap_values_for_plot,
                    base_values=0,  # This might need adjustment based on model
                    data=X_test.iloc[0].values,
                    feature_names=list(X_test.columns)
                )
            else:
                # Multiple instances - take first instance
                explanation = shap.Explanation(
                    values=shap_values_for_plot[0],
                    base_values=0,  # This might need adjustment based on model
                    data=X_test.iloc[0].values,
                    feature_names=list(X_test.columns)
                )
            
            shap.waterfall_plot(explanation, max_display=15, show=False)
        except Exception as e:
            logger.warning(f"Could not create waterfall plot: {str(e)}")
            # Create a fallback plot
            plt.text(0.5, 0.5, f"Waterfall plot not available:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
        
        waterfall_plot_path = self.output_dir / f"shap_waterfall_sample_{model_name}_{task_type}.png"
        plt.savefig(waterfall_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        plots_info["waterfall_plot"] = str(waterfall_plot_path)
        logger.info(f"SHAP waterfall plot saved: {waterfall_plot_path}")

        return plots_info

    def analyze_regression_models(
        self, 
        regression_results: Dict,
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        target_column: str = "exp_accuracy"
    ) -> Dict:
        """
        Perform SHAP analysis on regression models.

        Args:
            regression_results: Results dictionary from RegressionAnalyzer
            X_train: Training features (if None, will be extracted from regression_results)
            X_test: Test features (if None, will be extracted from regression_results)
            target_column: Target column name (only used if X_train/X_test are None)

        Returns:
            Dictionary containing SHAP analysis results for regression
        """
        logger.info("Starting SHAP analysis for regression models...")
        
        shap_results = {}
        
        # Check if regression_results is the full results dict or just the models dict
        models_dict = None
        if "regression_results" in regression_results and "models" in regression_results["regression_results"]:
            # Case: full results dict passed from RegressionAnalyzer.run_full_pipeline()
            models_dict = regression_results["regression_results"]["models"]
            # Extract X_train and X_test if not provided
            if X_train is None or X_test is None:
                X_train = regression_results["regression_results"].get("X_train")
                X_test = regression_results["regression_results"].get("X_test")
        elif "models" in regression_results:
            # Case: just the models dict passed directly
            models_dict = regression_results["models"]
            # Extract X_train and X_test if not provided
            if X_train is None or X_test is None:
                X_train = regression_results.get("X_train")
                X_test = regression_results.get("X_test")
        else:
            logger.warning("No models found in regression_results for SHAP analysis")
            return shap_results
        
        # Fallback: prepare features if X_train/X_test are still None
        if X_train is None or X_test is None:
            logger.warning("X_train or X_test not provided, preparing features from scratch")
            X, y = self.prepare_features(target_column=target_column)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        for model_name, model in models_dict.items():
            shap_result = self.explain_model(
                model, X_train, X_test, model_name, "regression"
            )
            shap_results[model_name] = shap_result
            
        self.results["regression_shap"] = shap_results
        logger.info("SHAP analysis completed for regression models")
        
        return shap_results

    def analyze_classification_models(
        self, 
        classification_results: Dict,
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None,
        target_column: str = "is_finally_correct"
    ) -> Dict:
        """
        Perform SHAP analysis on classification models.

        Args:
            classification_results: Results dictionary from ClassificationAnalyzer
            X_train: Training features (if None, will be extracted from classification_results)
            X_test: Test features (if None, will be extracted from classification_results)
            target_column: Target column name (only used if X_train/X_test are None)

        Returns:
            Dictionary containing SHAP analysis results for classification
        """
        logger.info("Starting SHAP analysis for classification models...")
        
        shap_results = {}
        
        # Check if classification_results is the full results dict or just the models dict
        models_dict = None
        if "classification_results" in classification_results and "models" in classification_results["classification_results"]:
            # Case: full results dict passed from ClassificationAnalyzer.run_full_pipeline()
            models_dict = classification_results["classification_results"]["models"]
            # Extract X_train and X_test if not provided
            if X_train is None or X_test is None:
                X_train = classification_results["classification_results"].get("X_train")
                X_test = classification_results["classification_results"].get("X_test")
        elif "models" in classification_results:
            # Case: just the models dict passed directly
            models_dict = classification_results["models"]
            # Extract X_train and X_test if not provided
            if X_train is None or X_test is None:
                X_train = classification_results.get("X_train")
                X_test = classification_results.get("X_test")
        else:
            logger.warning("No models found in classification_results for SHAP analysis")
            return shap_results
        
        # Fallback: prepare features if X_train/X_test are still None
        if X_train is None or X_test is None:
            logger.warning("X_train or X_test not provided, preparing features from scratch")
            X, y = self.prepare_features(target_column=target_column)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        for model_name, model in models_dict.items():
            shap_result = self.explain_model(
                model, X_train, X_test, model_name, "classification"
            )
            shap_results[model_name] = shap_result
            
        self.results["classification_shap"] = shap_results
        logger.info("SHAP analysis completed for classification models")
        
        return shap_results

    def generate_report(self):
        """
        Generate a comprehensive SHAP analysis report.
        """
        logger.info("Generating SHAP analysis report...")

        report_path = self.output_dir / "shap_analysis_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SHAP ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            if "regression_shap" in self.results:
                f.write("SHAP ANALYSIS FOR REGRESSION MODELS\n")
                f.write("-" * 80 + "\n\n")
                
                for model_name, shap_result in self.results["regression_shap"].items():
                    f.write(f"{model_name} (Regression):\n")
                    f.write(f"  Summary Plot: {shap_result['plots_info']['summary_plot']}\n")
                    f.write(f"  Importance Plot: {shap_result['plots_info']['importance_plot']}\n")
                    f.write(f"  Waterfall Plot: {shap_result['plots_info']['waterfall_plot']}\n")
                    f.write(f"  Dependence Plots: {len(shap_result['plots_info']['dependence_plots'])} plots\n\n")

            if "classification_shap" in self.results:
                f.write("\n" + "=" * 80 + "\n")
                f.write("SHAP ANALYSIS FOR CLASSIFICATION MODELS\n")
                f.write("-" * 80 + "\n\n")
                
                for model_name, shap_result in self.results["classification_shap"].items():
                    f.write(f"{model_name} (Classification):\n")
                    f.write(f"  Summary Plot: {shap_result['plots_info']['summary_plot']}\n")
                    f.write(f"  Importance Plot: {shap_result['plots_info']['importance_plot']}\n")
                    f.write(f"  Waterfall Plot: {shap_result['plots_info']['waterfall_plot']}\n")
                    f.write(f"  Dependence Plots: {len(shap_result['plots_info']['dependence_plots'])} plots\n\n")

        logger.info(f"SHAP analysis report saved: {report_path}")

        return report_path

    def run_full_analysis(
        self, 
        regression_results: Optional[Dict] = None, 
        classification_results: Optional[Dict] = None
    ):
        """
        Run complete SHAP analysis pipeline.

        Args:
            regression_results: Results from RegressionAnalyzer (should include X_train, X_test)
            classification_results: Results from ClassificationAnalyzer (should include X_train, X_test)

        Returns:
            Tuple of (results, report_path)
        """
        logger.info("Starting SHAP analysis pipeline...")

        # Perform SHAP analysis if results are provided
        # Note: X_train and X_test will be extracted from regression_results/classification_results
        if regression_results is not None:
            self.analyze_regression_models(regression_results)

        if classification_results is not None:
            self.analyze_classification_models(classification_results)

        # Generate report
        report_path = self.generate_report()

        logger.info("SHAP analysis pipeline completed successfully!")

        return self.results, report_path


def main():
    """Main function to execute the SHAP analysis."""
    logger.info("Initializing SHAP Analyzer...")
    
    # Note: This is mainly for testing - in practice, SHAP analyzer 
    # would be used with results from other analyzers
    analyzer = ShapAnalyzer()

    logger.info("SHAP Analyzer initialized successfully!")

    return analyzer


if __name__ == "__main__":
    analyzer = main()