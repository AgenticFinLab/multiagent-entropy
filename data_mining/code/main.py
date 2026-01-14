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
import logging
import argparse
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_mining_analyzer import DataMiningAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_mining_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """
    Execute the data mining analysis pipeline via command-line interface.
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
    parser.add_argument(
        "--data-path",
        type=str,
        default="data_mining/data/merged_datasets.csv",
        help="Path to merged data file (used when skip-collection is True)",
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MULTI-AGENT ENTROPY DATA MINING ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Analysis Type: {args.analysis_type}")
    logger.info(f"Target Datasets: {', '.join(args.datasets)}")
    logger.info(f"Skip Collection: {args.skip_collection}")

    try:
        # Determine target_dataset for output directory
        target_dataset = None
        if args.datasets and len(args.datasets) == 1:
            target_dataset = args.datasets[0]

        # Initialize the analyzer with appropriate parameters
        analyzer = DataMiningAnalyzer(
            data_path=args.data_path,
            target_dataset=target_dataset,
            skip_collection=args.skip_collection,
        )

        # Run the analysis based on the specified type
        results, report_paths = analyzer.run_full_analysis(
            analysis_type=args.analysis_type,
            target_datasets=args.datasets if not args.skip_collection else None
        )

        # Print summary of results
        logger.info("\n[ANALYSIS SUMMARY]")
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
        if isinstance(report_paths, list):
            for report_path in report_paths:
                logger.info(f"  - {report_path}")
        else:
            logger.info(f"  - {report_paths}")

        logger.info("\n" + "=" * 80)
        logger.info("DATA MINING ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        return results, report_paths

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    results, report_paths = main()
