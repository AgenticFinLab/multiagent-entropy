"""
Data Mining Analyzer Module for Multi-Agent Entropy Analysis

This module serves as a unified entry point for comprehensive data mining analysis:
1. Experiment-level analysis: Regression models to predict exp_accuracy (via RegressionAnalyzer)
2. Sample-level analysis: Classification models to predict is_finally_correct (via ClassificationAnalyzer)

Delegates to specialized analyzers while maintaining backward compatibility.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional

# Import utilities
from utils import (
    create_output_directory,
    determine_output_directory,
    get_default_data_path,
    generate_filter_suffix,
)

from shap_analyzer import ShapAnalyzer
from regression_analyzer import RegressionAnalyzer
from classification_analyzer import ClassificationAnalyzer

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataMiningAnalyzer:
    """Unified entry point for comprehensive data mining analysis.
    
    Delegates to RegressionAnalyzer and ClassificationAnalyzer for specialized tasks.
    Supports both programmatic usage and command-line interface.
    """

    def __init__(
        self,
        data_path: str = None,
        output_dir: str = None,
        target_dataset: str = None,
        skip_collection: bool = False,
        run_shap: bool = True,
        model_names: List[str] = None,
        architectures: List[str] = None,
        datasets: List[str] = None,
    ):
        """
        Initialize the DataMiningAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file
            output_dir: Directory to save analysis results
            target_dataset: Target dataset name for determining output directory
            skip_collection: Whether to skip data collection and use existing data
            run_shap: Whether to run SHAP analysis
            model_names: List of model names to filter (None or ['*'] for all)
            architectures: List of architectures to filter (None or ['*'] for all)
            datasets: List of datasets to filter (None or ['*'] for all)
        """
        if data_path is None:
            data_path = get_default_data_path()

        # Determine output directory based on target_dataset and filters
        if output_dir is None:
            base_output_dir = determine_output_directory(
                "data_mining/results", target_dataset
            )
            # Add filter suffix to output directory
            filter_suffix = generate_filter_suffix(
                model_names=model_names,
                architectures=architectures,
                datasets=datasets,
            )
            if filter_suffix:
                output_dir = f"{base_output_dir}/{filter_suffix}"
            else:
                output_dir = base_output_dir

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.target_dataset = target_dataset
        self.skip_collection = skip_collection
        self.model_names = model_names
        self.architectures = architectures
        self.datasets = datasets
        self.results = {}
        self.run_shap = run_shap

        # Create output directory if it doesn't exist
        create_output_directory(self.output_dir)
        
        # Initialize specialized analyzers
        self.regression_analyzer = RegressionAnalyzer(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir / "regression"),
            target_dataset=target_dataset,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
        )
        
        self.classification_analyzer = ClassificationAnalyzer(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir / "classification"),
            target_dataset=target_dataset,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
        )
        
        # Initialize SHAP analyzer if available
        if self.run_shap:
            try:
                self.shap_analyzer = ShapAnalyzer(
                    data_path=str(self.data_path),
                    output_dir=str(self.output_dir / "shap"),
                    target_dataset=target_dataset,
                )
            except ImportError:
                self.shap_analyzer = None
                print("Warning: SHAP not available. Install with: pip install shap")
                self.run_shap = False



    def run_experiment_level_analysis(self):
        """
        Perform experiment-level analysis (regression on exp_accuracy).
        
        Delegates to RegressionAnalyzer.
        """
        regression_results, _ = self.regression_analyzer.run_full_pipeline()
        self.results["experiment_level"] = regression_results
        return self.results["experiment_level"]

    def run_sample_level_analysis(self):
        """
        Perform sample-level analysis (classification on is_finally_correct).
        
        Delegates to ClassificationAnalyzer.
        """
        classification_results, _ = self.classification_analyzer.run_full_pipeline()
        self.results["sample_level"] = classification_results
        return self.results["sample_level"]

    def run_shap_analysis(self, include_regression=True, include_classification=True):
        """
        Perform SHAP analysis for interpretability of models.
        
        Args:
            include_regression: Whether to run SHAP analysis for regression models
            include_classification: Whether to run SHAP analysis for classification models
        
        Note:
            This method passes the full results dictionaries to ShapAnalyzer,
            which will automatically extract X_train, X_test, and trained models.
            This ensures SHAP analysis uses the same data splits as model training.
        """            
        if self.shap_analyzer is None:
            print("SHAP analyzer not initialized.")
            return None
            
        logger.info("Running SHAP analysis...")
        
        # Get results from previous analyses
        # These contain models, X_train, X_test, y_train, y_test
        regression_results = self.results.get("experiment_level")
        classification_results = self.results.get("sample_level")
        
        # Run SHAP analysis - data splits will be extracted from results
        shap_results, shap_report_path = self.shap_analyzer.run_full_analysis(
            regression_results=regression_results if include_regression else None,
            classification_results=classification_results if include_classification else None
        )
        
        self.results["shap"] = shap_results
        
        logger.info("SHAP analysis completed.")
        
        return self.results["shap"]

    def generate_report(self):
        """
        Generate a unified comprehensive analysis report.
        
        Combines results from regression, classification, and SHAP analyzers.
        """
        logger.info("Generating comprehensive analysis report...")

        report_path = self.output_dir / "unified_analysis_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED DATA MINING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("This report consolidates results from:\n")
            f.write(f"  - Regression Analysis: {self.output_dir / 'regression' / 'regression_report.txt'}\n")
            f.write(f"  - Classification Analysis: {self.output_dir / 'classification' / 'classification_report.txt'}\n")
            if "shap" in self.results:
                f.write(f"  - SHAP Analysis: {self.output_dir / 'shap' / 'shap_analysis_report.txt'}\n\n")
            else:
                f.write("\n")
            
            # Experiment-level results
            if "experiment_level" in self.results:
                f.write("EXPERIMENT-LEVEL ANALYSIS (Regression on exp_accuracy)\n")
                f.write("-" * 80 + "\n\n")

                for model_name, metrics in self.results["experiment_level"][
                    "regression_results"
                ]["metrics"].items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Mean Squared Error (MSE): {metrics['MSE']:.6f}\n")
                    f.write(f"  Mean Absolute Error (MAE): {metrics['MAE']:.6f}\n")
                    f.write(f"  R-squared (R2): {metrics['R2']:.6f}\n\n")

                # Top 10 important features for each model
                f.write("TOP 10 IMPORTANT FEATURES (Experiment Level):\n")
                f.write("-" * 80 + "\n\n")

                for model_name, importance_df in self.results["experiment_level"][
                    "importance_results"
                ].items():
                    f.write(f"{model_name}:\n")
                    for idx, row in importance_df.head(10).iterrows():
                        f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                    f.write("\n")

            # Sample-level results
            if "sample_level" in self.results:
                f.write("\n" + "=" * 80 + "\n")
                f.write(
                    "SAMPLE-LEVEL ANALYSIS (Classification on is_finally_correct)\n"
                )
                f.write("-" * 80 + "\n\n")

                for model_name, metrics in self.results["sample_level"][
                    "classification_results"
                ]["metrics"].items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Accuracy: {metrics['Accuracy']:.6f}\n")
                    f.write(f"  Precision: {metrics['Precision']:.6f}\n")
                    f.write(f"  Recall: {metrics['Recall']:.6f}\n")
                    f.write(f"  F1-Score: {metrics['F1']:.6f}\n\n")

                # Top 10 important features for each model
                f.write("TOP 10 IMPORTANT FEATURES (Sample Level):\n")
                f.write("-" * 80 + "\n\n")

                for model_name, importance_df in self.results["sample_level"][
                    "importance_results"
                ].items():
                    f.write(f"{model_name}:\n")
                    for idx, row in importance_df.head(10).iterrows():
                        f.write(f"  {row['Feature']}: {row['Importance']:.6f}\n")
                    f.write("\n")
            
            # SHAP Analysis results
            if "shap" in self.results:
                f.write("\n" + "=" * 80 + "\n")
                f.write("SHAP ANALYSIS RESULTS\n")
                f.write("-" * 80 + "\n\n")
                
                if "regression_shap" in self.results["shap"]:
                    f.write("SHAP ANALYSIS FOR REGRESSION MODELS:\n\n")
                    for model_name, shap_result in self.results["shap"]["regression_shap"].items():
                        f.write(f"{model_name} (Regression):\n")
                        f.write(f"  Summary Plot: {shap_result['plots_info']['summary_plot']}\n")
                        f.write(f"  Importance Plot: {shap_result['plots_info']['importance_plot']}\n")
                        f.write(f"  Waterfall Plot: {shap_result['plots_info']['waterfall_plot']}\n")
                        f.write(f"  Dependence Plots: {len(shap_result['plots_info']['dependence_plots'])} plots\n\n")
                
                if "classification_shap" in self.results["shap"]:
                    f.write("SHAP ANALYSIS FOR CLASSIFICATION MODELS:\n\n")
                    for model_name, shap_result in self.results["shap"]["classification_shap"].items():
                        f.write(f"{model_name} (Classification):\n")
                        f.write(f"  Summary Plot: {shap_result['plots_info']['summary_plot']}\n")
                        f.write(f"  Importance Plot: {shap_result['plots_info']['importance_plot']}\n")
                        f.write(f"  Waterfall Plot: {shap_result['plots_info']['waterfall_plot']}\n")
                        f.write(f"  Dependence Plots: {len(shap_result['plots_info']['dependence_plots'])} plots\n\n")

        logger.info(f"Unified analysis report saved: {report_path}")

        return report_path

    def run_data_collection(self, target_datasets: Optional[List[str]] = None):
        """
        Run data collection and merging.
        
        Args:
            target_datasets: List of datasets to collect (None for all)
        """
        from data_collector import DataCollector  # Import here to avoid circular imports
        
        logger.info("[STEP 1] Data Collection and Merging")
        logger.info("-" * 80)

        collector = DataCollector(target_datasets=target_datasets)

        # Discover datasets
        datasets = collector.discover_datasets()
        logger.info(f"Discovered {len(datasets)} datasets: {', '.join(datasets)}")

        # Merge datasets
        merged_data = collector.merge_datasets()
        logger.info(f"Merged {len(merged_data)} records from {len(datasets)} datasets")

        # Save merged data
        merged_data_path = collector.save_merged_data()
        logger.info(f"Merged data saved to: {merged_data_path}")

        # Print data summary
        summary = collector.get_data_summary()
        logger.info(f"Data Summary:")
        logger.info(f"  Total Records: {summary['total_records']}")
        logger.info(f"  Total Columns: {summary['total_columns']}")
        logger.info(f"  Datasets: {summary['datasets']}")
        logger.info(f"  Records per Dataset: {summary['dataset_counts']}")
        
        return merged_data_path

    def run_full_analysis(
        self, 
        analysis_type: str = "all", 
        target_datasets: Optional[List[str]] = None):
        """
        Run complete analysis pipeline.
        
        Executes both regression and classification analyses via specialized analyzers.
        
        Args:
            analysis_type: Type of analysis to run ('all', 'regression', or 'classification')
            target_datasets: List of datasets to collect (None for all)
            run_shap: Whether to run SHAP analysis after regression/classification
        """
        logger.info("Starting full data mining analysis pipeline...")
        
        # Run data collection if not skipping
        if not self.skip_collection:
            merged_data_path = self.run_data_collection(target_datasets)
            # Update analyzers with the new path - convert to Path object to match analyzer expectations
            self.regression_analyzer.data_path = Path(str(merged_data_path))
            self.classification_analyzer.data_path = Path(str(merged_data_path))
        else:
            logger.info("Skipping data collection step")
            merged_data_path = self.data_path

        # Run analysis based on type
        if analysis_type == "all":
            # Run experiment-level analysis (regression)
            self.run_experiment_level_analysis()

            # Run sample-level analysis (classification)
            self.run_sample_level_analysis()

        elif analysis_type == "regression":
            self.run_experiment_level_analysis()

        elif analysis_type == "classification":
            self.run_sample_level_analysis()

        # Run SHAP analysis if requested and available
        if self.run_shap and hasattr(self, 'shap_analyzer') and self.shap_analyzer is not None:
            # Determine which SHAP analyses to run based on what was run above
            include_regression = analysis_type in ["all", "regression"]
            include_classification = analysis_type in ["all", "classification"]
            
            self.run_shap_analysis(
                include_regression=include_regression,
                include_classification=include_classification
            )

        # Generate unified report
        report_path = self.generate_report()

        logger.info("Full analysis pipeline completed successfully!")

        return self.results, report_path


def main():
    """Main function to execute the data mining analysis."""
    logger.info("Initializing Data Mining Analyzer...")

    # Initialize analyzer with default settings
    analyzer = DataMiningAnalyzer()

    # Run full analysis with defaults
    results, report_path = analyzer.run_full_analysis(analysis_type="all")

    logger.info(f"Analysis complete. Report saved to: {report_path}")

    return results, report_path


if __name__ == "__main__":
    results, report_path = main()