#!/usr/bin/env python3
"""
Main Entry Point for Multi-Agent Entropy Data Mining Analysis

Command-line interface for the DataMiningAnalyzer.
Supports running full analysis, regression-only, or classification-only workflows.
"""
import sys
import argparse
import logging
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_mining_analyzer import DataMiningAnalyzer
from aggregator import ExperimentAggregator
from visualizer import AggregatedResultsVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_mining_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """Execute the data mining analysis pipeline via command-line interface."""
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
        "--merged-datasets",
        type=str,
        nargs="*",
        default=["all"],
        help="Datasets to merge during data collection (use 'all' for all available datasets). Only effective when --skip-collection is not used.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        nargs="*",
        default=["all"],
        help="Filter by specific model name(s) for analysis (use 'all' for all models)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        nargs="*",
        default=["all"],
        help="Filter by specific architecture(s) for analysis (use 'all' for all architectures)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=["all"],
        help="Filter by specific dataset(s) for analysis (use 'all' for all datasets)",
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
    parser.add_argument(
        "--run-shap",
        action="store_true",
        default=True,
        help="Run SHAP analysis (default: True)",
    )
    parser.add_argument(
        "--no-shap",
        dest="run_shap",
        action="store_false",
        help="Skip SHAP analysis",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default="default",
        help="""Feature exclusion configuration. 
            Options:
            'all' - Use all features (no exclusions)
            'default' - Use default exclusions (recommended)
            Feature group name(s) - Specify groups from features.py (comma-separated)
            Available groups: 
            base_model_metrics, unseen_features, experiment_identifier, sample_identifier, experiment_statistics, unseen_features, round_statistics, sample_statistics, sample_distribution_shape, sample_baseline_entropy, aggregation_over_agents, sample_round_wise_aggregated, cross_round_aggregated, intra_round_agent_distribution, cross_round_agent_spread_change, sample_round1_agent_statistics, sample_round2_agent_statistics
            Examples:
                --exclude-features 'all' (use all features)
                --exclude-features 'default' (default exclusions)
                --exclude-features 'base_model_metrics' (exclude base model metrics only)
                --exclude-features 'base_model_metrics,experiment_identifier' (exclude multiple groups)
                --exclude-features 'default+base_model_metrics' (combine default with additional exclusions)
        """,
    )
    parser.add_argument(
        "--run-aggregation",
        default=True,
        help="Run experiment results aggregation after analysis (default: True)",
    )
    parser.add_argument(
        "--run-visualization",
        default=True,
        help="Run visualization of aggregated results after aggregation (default: True)",
    )
    args = parser.parse_args()

    # Handle the case where user specifies 'all' to collect all available datasets
    if args.merged_datasets == ["all"]:
        from data_collector import DataCollector

        collector = DataCollector()
        discovered_datasets = collector.discover_datasets()
        logger.info(f"Auto-discovered datasets: {discovered_datasets}")
        merged_datasets = discovered_datasets
    else:
        # Filter out empty strings and handle default case
        filtered_datasets = [
            ds for ds in args.merged_datasets if ds
        ]  # Remove empty strings
        if not filtered_datasets:
            filtered_datasets = ["aime2025"]  # Default to aime2025 if none provided
        merged_datasets = filtered_datasets

    # Process filter arguments
    model_names = None if args.model_name == ["all"] else args.model_name
    architectures = None if args.architecture == ["all"] else args.architecture
    datasets = None if args.dataset == ["all"] else args.dataset

    logger.info("=" * 80)
    logger.info("MULTI-AGENT ENTROPY DATA MINING ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Analysis Type: {args.analysis_type}")
    logger.info(
        f"Merged Datasets (for collection): {', '.join(merged_datasets) if merged_datasets else 'None (will auto-discover)'}"
    )
    logger.info(f"Skip Collection: {args.skip_collection}")
    logger.info(f"Filter - Model Names: {model_names if model_names else 'All'}")
    logger.info(f"Filter - Architectures: {architectures if architectures else 'All'}")
    logger.info(f"Filter - Datasets: {datasets if datasets else 'All'}")
    logger.info(f"Exclude Features: {args.exclude_features}")

    try:
        # Determine target_dataset for output directory
        target_dataset = None
        if merged_datasets and len(merged_datasets) == 1:
            target_dataset = merged_datasets[0]

        # Initialize the analyzer with appropriate parameters
        analyzer = DataMiningAnalyzer(
            data_path=args.data_path,
            target_dataset=target_dataset,
            skip_collection=args.skip_collection,
            run_shap=args.run_shap,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
            exclude_features=args.exclude_features,
        )

        # Run the analysis based on the specified type
        results, report_paths = analyzer.run_full_analysis(
            analysis_type=args.analysis_type,
            target_datasets=merged_datasets if not args.skip_collection else None,
        )

        # Print summary of results
        logger.info("\n[ANALYSIS SUMMARY]")
        logger.info("-" * 80)

        if "experiment_level" in results:
            logger.info("Experiment-Level Analysis (Regression on exp_accuracy):")
            for model_name, metrics in results["experiment_level"][
                "regression_results"
            ]["metrics"].items():
                logger.info(
                    f"  {model_name}: R2 = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.4f}"
                )

        if "sample_level" in results:
            logger.info(
                "\nSample-Level Analysis (Classification on is_finally_correct):"
            )
            for model_name, metrics in results["sample_level"][
                "classification_results"
            ]["metrics"].items():
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

        # Run aggregation if requested
        if args.run_aggregation:
            logger.info("\nStarting experiment results aggregation...")
            try:
                script_dir = Path(__file__).parent
                project_root = script_dir.parent
                results_dir = project_root / "results"
                output_dir = project_root / "results_aggregated"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                aggregator = ExperimentAggregator(str(results_dir), str(output_dir))
                aggregator.aggregate_all_experiments()
                logger.info("Experiment results aggregation completed!")
            except Exception as e:
                logger.error(f"Error during aggregation: {str(e)}", exc_info=True)
                raise
        
        # Run visualization if requested
        if args.run_visualization:
            logger.info("\nStarting visualization of aggregated results...")
            try:
                script_dir = Path(__file__).parent
                project_root = script_dir.parent
                input_dir = project_root / "results_aggregated"
                output_dir = project_root / "results_visualizations"
                
                visualizer = AggregatedResultsVisualizer(
                    str(input_dir),
                    str(output_dir),
                    n_features=20,
                    feature_importance_from="mean_importance_normalized",
                )
                visualizer.visualize_all_experiments()
                logger.info("Visualization of aggregated results completed!")
            except Exception as e:
                logger.error(f"Error during visualization: {str(e)}", exc_info=True)
                raise

        return results, report_paths

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    results, report_paths = main()
