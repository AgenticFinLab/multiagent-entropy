"""
Data Mining Analyzer Module for Multi-Agent Entropy Analysis

This module serves as a unified entry point for comprehensive data mining analysis:
1. Experiment-level analysis: Regression models to predict exp_accuracy (via RegressionAnalyzer)
2. Sample-level analysis: Classification models to predict is_finally_correct (via ClassificationAnalyzer)

Delegates to specialized analyzers while maintaining backward compatibility.
"""

import logging
import os
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
from pca_analyzer import PCAAnalysis
from feature_ablation_analyzer import FeatureAblationAnalyzer
from calibration_analyzer import CalibrationAnalyzer

from features import (
    FINAGENT_EVALUATION_FEATURES,
    FINAGENT_STEP_ENTROPY_FEATURES,
    discover_step_entropy_features,
)

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
        base_dir: str = None,
        data_path: str = None,
        output_dir: str = None,
        target_dataset: str = None,
        skip_collection: bool = False,
        run_shap: bool = True,
        model_names: List[str] = None,
        architectures: List[str] = None,
        datasets: List[str] = None,
        exclude_features: str = "default",
        dataset_type: str = "standard",
    ):
        """
        Initialize the DataMiningAnalyzer.

        Args:
            data_path: Path to the merged dataset CSV file
            output_dir: Directory to save analysis results
            target_dataset: Target dataset name for determining output directory
            skip_collection: Whether to skip data collection and use existing data
            run_shap: Whether to run SHAP analysis
            model_names: List of model names to filter (None or ['all'] for all)
            architectures: List of architectures to filter (None or ['all'] for all)
            datasets: List of datasets to filter (None or ['all'] for all)
            exclude_features: Feature exclusion configuration ('all', 'default', or feature group names)
            dataset_type: Type of dataset ('standard' or 'finagent')
        """
        self.dataset_type = dataset_type
        if data_path is None:
            data_path = get_default_data_path()

        # Determine output directory based on target_dataset, filters, and dataset_type
        if output_dir is None:
            # Use appropriate base directory based on dataset_type
            base_results_dir = (
                "data_mining/results_finagent"
                if dataset_type == "finagent"
                else "data_mining/results"
            )
            base_output_dir = determine_output_directory(
                base_results_dir, target_dataset, dataset_type=dataset_type
            )
            # Add filter suffix to output directory
            filter_suffix = generate_filter_suffix(
                model_names=model_names,
                architectures=architectures,
                datasets=datasets,
            )

            # Generate exclude_features suffix
            exclude_features_suffix = self._generate_exclude_features_suffix(
                exclude_features
            )

            # Combine all suffixes
            all_suffixes = []
            if filter_suffix:
                all_suffixes.append(filter_suffix)
            if exclude_features_suffix:
                all_suffixes.append(exclude_features_suffix)

            if all_suffixes:
                output_dir = f"{base_output_dir}/{'_'.join(all_suffixes)}"
            else:
                output_dir = base_output_dir

        self.base_dir = base_dir
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.target_dataset = target_dataset
        self.skip_collection = skip_collection
        self.model_names = model_names
        self.architectures = architectures
        self.datasets = datasets
        self.exclude_features = exclude_features
        self.results = {}
        self.run_shap = run_shap

        # Store finagent-specific features if applicable
        self.finagent_features = []
        if dataset_type == "finagent":
            self.finagent_features = (
                FINAGENT_EVALUATION_FEATURES + FINAGENT_STEP_ENTROPY_FEATURES
            )
            logger.info(
                f"Finagent mode: Added {len(self.finagent_features)} finagent-specific features"
            )

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
            exclude_features=exclude_features,
        )

        self.classification_analyzer = ClassificationAnalyzer(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir / "classification"),
            target_dataset=target_dataset,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
            exclude_features=exclude_features,
        )

        self.pca_analyzer_instance = PCAAnalysis(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir / "pca"),
            target_dataset=target_dataset,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
            exclude_features=exclude_features,
        )

        self.feature_ablation_analyzer = FeatureAblationAnalyzer(
            data_path=str(self.data_path),
            output_dir=str(self.output_dir / "feature_ablation"),
            target_dataset=target_dataset,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
            exclude_features=exclude_features,
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

    def _generate_exclude_features_suffix(self, exclude_features: str) -> str:
        """
        Generate a suffix for output directory based on exclude_features parameter.

        Args:
            exclude_features: Feature exclusion configuration ('all', 'default', or feature group names)

        Returns:
            Directory suffix string (e.g., "exclude_default", "exclude_base_model_metrics")
        """
        if not exclude_features:
            return ""

        # Normalize the exclude_features string to create a clean suffix
        suffix = (
            exclude_features.replace("+", "_")
            .replace(",", "_")
            .replace(" ", "_")
            .strip()
        )

        # Ensure the suffix doesn't start with underscore if it's empty after processing
        if suffix:
            return f"exclude_{suffix}"

        return ""

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
            classification_results=(
                classification_results if include_classification else None
            ),
        )

        self.results["shap"] = shap_results

        logger.info("SHAP analysis completed.")

        return self.results["shap"]

    def run_pca_analysis(self):
        """
        Perform PCA analysis on classification features.
        Delegates to PCAAnalysis.
        """
        pca_results, _ = self.pca_analyzer_instance.run_full_pipeline()
        self.results["pca"] = pca_results
        return self.results["pca"]

    def run_feature_ablation_analysis(self):
        """
        Perform feature ablation analysis on classification features.
        Delegates to FeatureAblationAnalyzer.
        """
        ablation_results, _ = self.feature_ablation_analyzer.run_full_pipeline()
        self.results["feature_ablation"] = ablation_results
        return self.results["feature_ablation"]

    def run_calibration_analysis(self):
        """Run model calibration analysis (ECE, reliability diagrams, quadrant analysis)."""
        logger.info("Running calibration analysis...")
        output_dir = os.path.join(self.output_dir, "calibration")
        analyzer = CalibrationAnalyzer(
            data_path=self.data_path,
            output_dir=output_dir,
            n_bins=10,
            entropy_metrics=["sample_mean_entropy", "sample_mean_answer_token_entropy"],
        )
        analyzer.run()
        logger.info("Calibration analysis complete.")

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
            f.write(
                f"  - Regression Analysis: {self.output_dir / 'regression' / 'regression_report.txt'}\n"
            )
            f.write(
                f"  - Classification Analysis: {self.output_dir / 'classification' / 'classification_report.txt'}\n"
            )
            if "shap" in self.results:
                f.write(
                    f"  - SHAP Analysis: {self.output_dir / 'shap' / 'shap_analysis_report.txt'}\n\n"
                )
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
                    for model_name, shap_result in self.results["shap"][
                        "regression_shap"
                    ].items():
                        f.write(f"{model_name} (Regression):\n")
                        f.write(
                            f"  Summary Plot: {shap_result['plots_info']['summary_plot']}\n"
                        )
                        f.write(
                            f"  Importance Plot: {shap_result['plots_info']['importance_plot']}\n"
                        )
                        f.write(
                            f"  Waterfall Plot: {shap_result['plots_info']['waterfall_plot']}\n"
                        )
                        f.write(
                            f"  Dependence Plots: {len(shap_result['plots_info']['dependence_plots'])} plots\n\n"
                        )

                if "classification_shap" in self.results["shap"]:
                    f.write("SHAP ANALYSIS FOR CLASSIFICATION MODELS:\n\n")
                    for model_name, shap_result in self.results["shap"][
                        "classification_shap"
                    ].items():
                        f.write(f"{model_name} (Classification):\n")
                        f.write(
                            f"  Summary Plot: {shap_result['plots_info']['summary_plot']}\n"
                        )
                        f.write(
                            f"  Importance Plot: {shap_result['plots_info']['importance_plot']}\n"
                        )
                        f.write(
                            f"  Waterfall Plot: {shap_result['plots_info']['waterfall_plot']}\n"
                        )
                        f.write(
                            f"  Dependence Plots: {len(shap_result['plots_info']['dependence_plots'])} plots\n\n"
                        )

            # PCA Analysis results
            if "pca" in self.results:
                f.write("\n" + "=" * 80 + "\n")
                f.write("PCA ANALYSIS RESULTS\n")
                f.write("-" * 80 + "\n\n")
                pca_res = self.results["pca"]
                if "optimal_components" in pca_res:
                    opt_comp = pca_res["optimal_components"]
                    if (
                        isinstance(opt_comp, dict)
                        and "recommended_components" in opt_comp
                    ):
                        f.write(
                            f"Optimal number of components: {opt_comp['recommended_components']}\n\n"
                        )
                    else:
                        f.write(f"Optimal number of components: {opt_comp}\n\n")
                if "comparison" in pca_res:
                    f.write("Original vs PCA Performance Comparison:\n")
                    comp = pca_res["comparison"]
                    if "original" in comp and "pca" in comp:
                        for model_name in comp["original"].keys():
                            if model_name in comp["pca"]:
                                f.write(f"  {model_name}:\n")
                                f.write(
                                    f"    Original Accuracy: {comp['original'][model_name]['Accuracy']:.6f}\n"
                                )
                                f.write(
                                    f"    PCA Accuracy: {comp['pca'][model_name]['Accuracy']:.6f}\n"
                                )
                                f.write(
                                    f"    Original F1: {comp['original'][model_name]['F1']:.6f}\n"
                                )
                                f.write(
                                    f"    PCA F1: {comp['pca'][model_name]['F1']:.6f}\n\n"
                                )
                f.write(
                    f"  Detailed report: {self.output_dir / 'pca' / 'pca_analysis_report.txt'}\n\n"
                )

            # Feature Ablation results
            if "feature_ablation" in self.results:
                f.write("\n" + "=" * 80 + "\n")
                f.write("FEATURE ABLATION ANALYSIS RESULTS\n")
                f.write("-" * 80 + "\n\n")
                abl_res = self.results["feature_ablation"]
                if "feature_rankings" in abl_res:
                    f.write("Top 10 Features (Combined Ranking):\n")
                    ranking = abl_res["feature_rankings"]
                    if hasattr(ranking, "head"):
                        for idx, row in ranking.head(10).iterrows():
                            f.write(f"  {row['Feature']}: {row['Avg_Rank']:.6f}\n")
                    f.write("\n")
                f.write(
                    f"  Detailed report: {self.output_dir / 'feature_ablation' / 'feature_ablation_report.txt'}\n\n"
                )

        logger.info(f"Unified analysis report saved: {report_path}")

        return report_path

    def run_data_collection(self, target_datasets: Optional[List[str]] = None):
        """
        Run data collection and merging.

        Args:
            target_datasets: List of datasets to collect (None for all)
        """
        from data_collector import (
            DataCollector,
        )  # Import here to avoid circular imports

        logger.info("[STEP 1] Data Collection and Merging")
        logger.info("-" * 80)

        collector = DataCollector(
            base_dir=self.base_dir,
            target_datasets=target_datasets,
            dataset_type=self.dataset_type,
        )

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
        self, analysis_type: str = "all", target_datasets: Optional[List[str]] = None
    ):
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
            # Skip regression for finagent dataset as it doesn't need regression analysis
            if self.dataset_type == "finagent":
                logger.info(
                    "Skipping regression analysis for finagent dataset: "
                    "finagent uses evaluation_score as target which is not suitable for regression."
                )
            else:
                self.run_experiment_level_analysis()

            # Run sample-level analysis (classification)
            self.run_sample_level_analysis()

        elif analysis_type == "regression":
            # Skip regression for finagent dataset
            if self.dataset_type == "finagent":
                logger.warning(
                    "Regression analysis is not supported for finagent dataset. "
                    "Skipping regression analysis."
                )
            else:
                self.run_experiment_level_analysis()

        elif analysis_type == "classification":
            self.run_sample_level_analysis()

        elif analysis_type == "pca":
            self.run_pca_analysis()

        elif analysis_type == "feature_ablation":
            self.run_feature_ablation_analysis()

        elif analysis_type == "ablation_all":
            self.run_pca_analysis()
            self.run_feature_ablation_analysis()

        elif analysis_type == "calibration":
            self.run_calibration_analysis()

        # For 'all', also run calibration analysis
        if analysis_type == "all":
            self.run_calibration_analysis()

        # Run SHAP analysis if requested and available
        if (
            self.run_shap
            and hasattr(self, "shap_analyzer")
            and self.shap_analyzer is not None
        ):
            # Determine which SHAP analyses to run based on what was run above
            # Only run SHAP for all/regression/classification, not for pca/feature_ablation/ablation_all
            if analysis_type in ["all", "regression", "classification"]:
                include_regression = analysis_type in ["all", "regression"]
                include_classification = analysis_type in ["all", "classification"]

                self.run_shap_analysis(
                    include_regression=include_regression,
                    include_classification=include_classification,
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
