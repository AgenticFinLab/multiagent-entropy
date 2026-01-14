#!/usr/bin/env python3
"""
Main Entry Point for Multi-Agent Entropy Data Mining Analysis

This script orchestrates the complete data mining workflow:
1. Data Collection and Merging
2. Experiment-Level Analysis (Regression) - Optional
3. Sample-Level Analysis (Classification) - Optional
4. Report Generation

Supports running both analyses, or individual regression/classification analysis.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import DataCollector
from data_mining_analyzer import DataMiningAnalyzer
from regression_analyzer import RegressionAnalyzer
from classification_analyzer import ClassificationAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_mining_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """
    Execute the data mining analysis pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Multi-Agent Entropy Data Mining Analysis"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["all", "regression", "classification"],
        default="all",
        help="Type of analysis to run: all (default), regression, or classification",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["aime2025"],
        help="Target datasets to analyze (default: aime2025)",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection step (use existing merged data)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MULTI-AGENT ENTROPY DATA MINING ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Analysis Type: {args.analysis_type}")
    logger.info(f"Target Datasets: {', '.join(args.datasets)}")

    try:
        merged_data_path = None
        
        # Step 1: Data Collection and Merging (if not skipped)
        if not args.skip_collection:
            logger.info("\n[STEP 1] Data Collection and Merging")
            logger.info("-" * 80)

            collector = DataCollector(target_datasets=args.datasets)

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
        else:
            logger.info("\n[STEP 1] Skipping Data Collection (using existing data)")
            logger.info("-" * 80)
            merged_data_path = "data_mining/data/merged_datasets.csv"
            logger.info(f"Using existing merged data: {merged_data_path}")

        # Step 2: Data Mining Analysis
        logger.info("\n[STEP 2] Data Mining Analysis")
        logger.info("-" * 80)

        # Determine target_dataset for output directory
        target_dataset = None
        if args.datasets and len(args.datasets) == 1:
            target_dataset = args.datasets[0]

        results = {}
        report_paths = []

        if args.analysis_type == "all":
            # Run full analysis (both regression and classification)
            analyzer = DataMiningAnalyzer(
                data_path=merged_data_path, target_dataset=target_dataset
            )
            results, report_path = analyzer.run_full_analysis()
            report_paths.append(report_path)
            
        elif args.analysis_type == "regression":
            # Run only regression analysis
            logger.info("Running Experiment-Level Regression Analysis...")
            regression_analyzer = RegressionAnalyzer(
                data_path=merged_data_path, target_dataset=target_dataset
            )
            regression_results, regression_report = regression_analyzer.run_full_pipeline()
            results["experiment_level"] = regression_results
            report_paths.append(regression_report)
            
        elif args.analysis_type == "classification":
            # Run only classification analysis
            logger.info("Running Sample-Level Classification Analysis...")
            classification_analyzer = ClassificationAnalyzer(
                data_path=merged_data_path, target_dataset=target_dataset
            )
            classification_results, classification_report = classification_analyzer.run_full_pipeline()
            results["sample_level"] = classification_results
            report_paths.append(classification_report)

        # Step 3: Summary
        logger.info("\n[STEP 3] Analysis Summary")
        logger.info("-" * 80)

        if "experiment_level" in results:
            logger.info("Experiment-Level Analysis (Regression on exp_accuracy):")
            for model_name, metrics in results["experiment_level"]["regression_results"][
                "metrics"
            ].items():
                logger.info(
                    f"  {model_name}: R2 = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}"
                )

        if "sample_level" in results:
            logger.info("\nSample-Level Analysis (Classification on is_finally_correct):")
            for model_name, metrics in results["sample_level"]["classification_results"][
                "metrics"
            ].items():
                logger.info(
                    f"  {model_name}: Accuracy = {metrics['Accuracy']:.4f}, F1 = {metrics['F1']:.4f}"
                )

        logger.info("\nAnalysis reports saved to:")
        for report_path in report_paths:
            logger.info(f"  - {report_path}")

        logger.info("\n" + "=" * 80)
        logger.info("DATA MINING ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        return results, report_paths

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    results, report_paths = main()
