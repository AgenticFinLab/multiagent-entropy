#!/usr/bin/env python3
"""
Main Entry Point for Multi-Agent Entropy Data Mining Analysis

This script orchestrates the complete data mining workflow:
1. Data Collection and Merging
2. Experiment-Level Analysis (Regression)
3. Sample-Level Analysis (Classification)
4. Report Generation
"""

import sys
import logging
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collector import DataCollector
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
    Execute the complete data mining analysis pipeline.
    """
    logger.info("=" * 80)
    logger.info("MULTI-AGENT ENTROPY DATA MINING ANALYSIS")
    logger.info("=" * 80)

    try:
        # Step 1: Data Collection and Merging
        logger.info("\n[STEP 1] Data Collection and Merging")
        logger.info("-" * 80)

        # Configure target datasets here (None for all datasets, or specify a list)
        # e.g., ["gsm8k", "aime2024"] or None for all
        target_datasets = ["gsm8k"]

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

        # Step 2: Data Mining Analysis
        logger.info("\n[STEP 2] Data Mining Analysis")
        logger.info("-" * 80)

        # Determine target_dataset for output directory
        target_dataset = None
        if target_datasets and len(target_datasets) == 1:
            target_dataset = target_datasets[0]

        analyzer = DataMiningAnalyzer(
            data_path=merged_data_path, target_dataset=target_dataset
        )

        # Run full analysis
        results, report_path = analyzer.run_full_analysis()

        # Step 3: Summary
        logger.info("\n[STEP 3] Analysis Summary")
        logger.info("-" * 80)

        logger.info("Experiment-Level Analysis (Regression on exp_accuracy):")
        for model_name, metrics in results["experiment_level"]["regression_results"][
            "metrics"
        ].items():
            logger.info(
                f"  {model_name}: R2 = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}"
            )

        logger.info("\nSample-Level Analysis (Classification on is_finally_correct):")
        for model_name, metrics in results["sample_level"]["classification_results"][
            "metrics"
        ].items():
            logger.info(
                f"  {model_name}: Accuracy = {metrics['Accuracy']:.4f}, F1 = {metrics['F1']:.4f}"
            )

        logger.info(f"\nAnalysis report saved to: {report_path}")
        logger.info(f"All visualizations saved to: {analyzer.output_dir}")

        logger.info("\n" + "=" * 80)
        logger.info("DATA MINING ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

        return results, report_path

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    results, report_path = main()
