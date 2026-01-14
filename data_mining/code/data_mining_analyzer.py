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
    ):
        """
        Initialize the DataMiningAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file
            output_dir: Directory to save analysis results
            target_dataset: Target dataset name for determining output directory
            skip_collection: Whether to skip data collection and use existing data
        """
        if data_path is None:
            data_path = "data_mining/data/merged_datasets.csv"

        # Determine output directory based on target_dataset
        if output_dir is None:
            if target_dataset:
                output_dir = f"data_mining/results/{target_dataset}"
            else:
                output_dir = "data_mining/results"

        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.target_dataset = target_dataset
        self.skip_collection = skip_collection
        self.results = {}

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize specialized analyzers
        self.regression_analyzer = RegressionAnalyzer(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir / "regression"),
            target_dataset=target_dataset,
        )
        
        self.classification_analyzer = ClassificationAnalyzer(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir / "classification"),
            target_dataset=target_dataset,
        )


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

    def generate_report(self):
        """
        Generate a unified comprehensive analysis report.
        
        Combines results from both regression and classification analyzers.
        """
        logger.info("Generating comprehensive analysis report...")

        report_path = self.output_dir / "unified_analysis_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED DATA MINING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("This report consolidates results from:\n")
            f.write(f"  - Regression Analysis: {self.output_dir / 'regression' / 'regression_report.txt'}\n")
            f.write(f"  - Classification Analysis: {self.output_dir / 'classification' / 'classification_report.txt'}\n\n")
            
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

    def run_full_analysis(self, analysis_type: str = "all", target_datasets: Optional[List[str]] = None):
        """
        Run complete analysis pipeline.
        
        Executes both regression and classification analyses via specialized analyzers.
        
        Args:
            analysis_type: Type of analysis to run ('all', 'regression', or 'classification')
            target_datasets: List of datasets to collect (None for all)
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