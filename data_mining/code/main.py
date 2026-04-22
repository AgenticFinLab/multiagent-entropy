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
        choices=[
            "all",
            "regression",
            "classification",
            "pca",
            "feature_ablation",
            "ablation_all",
            "calibration",
        ],
        default="classification",
        help="Type of analysis to run: all (default), regression, classification, pca, feature_ablation, ablation_all, or calibration",
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
        default=False,
        help="Skip data collection step (use existing merged data)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data_mining/data/merged_datasets_qwen3_14b.csv",
        help="Path to merged data file (used when skip-collection is True)",
    )
    parser.add_argument(
        "--run-shap",
        default=True,
        help="Run SHAP analysis (default: True)",
    )
    parser.add_argument(
        "--exclude-features",
        type=str,
        default="base_model_all_metrics",
        help="""Feature exclusion configuration. 
            Options:
            'all' - Use all features (no exclusions)
            'default' - Use default exclusions (recommended)
            Feature group name(s) - Specify groups from features.py (comma-separated)
            Available groups: 
                "base_model_wo_entropy": BASE_MODEL_WO_ENTROPY,
                "base_model_all_metrics": BASE_MODEL_ALL_METRICS,
                ...
            Examples:
                --exclude-features 'all' (use all features)
                --exclude-features 'default' (default exclusions)
                --exclude-features 'base_model_all_metrics' (exclude base model metrics only)
                --exclude-features 'base_model_metrics,experiment_identifier' (exclude multiple groups)
                --exclude-features 'default+base_model_metrics' (combine default with additional exclusions)
        """,
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="standard",
        choices=["standard", "finagent", "gaia"],
        help="dataset type: standard (gsm8k/humaneval etc.), finagent, or gaia",
    )
    args = parser.parse_args()

    # Determine paths based on dataset type
    if args.dataset_type == "finagent":
        base_dir = "evaluation/results_finagent"
        output_dir = f"data_mining/results_finagent/{args.exclude_features}"
        data_path = "data_mining/data/merged_datasets_finagent.csv"
        merged_datasets = ["finagent"]
        logger.info(
            f"Using finagent mode: base_dir={base_dir}, output_dir={output_dir}"
        )
    elif args.dataset_type == "gaia":
        base_dir = "evaluation/results_gaia"
        output_dir = f"data_mining/results_gaia/{args.exclude_features}"
        data_path = "data_mining/data/merged_datasets_gaia.csv"
        merged_datasets = ["gaia"]
        logger.info(
            f"Using gaia mode: base_dir={base_dir}, output_dir={output_dir}"
        )
    else:
        base_dir = "evaluation/results_qwen3_14b"
        output_dir = f"data_mining/results_qwen3_14b/{args.exclude_features}"
        data_path = args.data_path
        # Handle the case where user specifies 'all' to collect all available datasets
        if args.merged_datasets == ["all"]:
            from data_collector import DataCollector

            collector = DataCollector(base_dir=base_dir)
            discovered_datasets = collector.discover_datasets()
            logger.info(f"Auto-discovered datasets: {discovered_datasets}")
            merged_datasets = discovered_datasets
        else:
            # Filter out empty strings and handle default case
            filtered_datasets = [
                ds for ds in args.merged_datasets if ds
            ]  # Remove empty strings
            if not filtered_datasets:
                filtered_datasets = ["gsm8k"]  # Default to aime2025 if none provided
            merged_datasets = filtered_datasets

    # Process filter arguments
    model_names = None if args.model_name == ["all"] else args.model_name
    architectures = None if args.architecture == ["all"] else args.architecture
    datasets = None if args.dataset == ["all"] else args.dataset

    logger.info("=" * 80)
    logger.info("MULTI-AGENT ENTROPY DATA MINING ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Dataset Type: {args.dataset_type}")
    logger.info(f"Analysis Type: {args.analysis_type}")
    logger.info(
        f"Merged Datasets (for collection): {', '.join(merged_datasets) if merged_datasets else 'None (will auto-discover)'}"
    )
    logger.info(f"Skip Collection: {args.skip_collection}")
    logger.info(f"Filter - Model Names: {model_names if model_names else 'All'}")
    logger.info(f"Filter - Architectures: {architectures if architectures else 'All'}")
    logger.info(f"Filter - Datasets: {datasets if datasets else 'All'}")
    logger.info(f"Exclude Features: {args.exclude_features}")

    # Check for incompatible combination: finagent + regression
    if args.dataset_type == "finagent" and args.analysis_type in ["regression", "all"]:
        if args.analysis_type == "regression":
            logger.warning(
                "Warning: Regression analysis is not supported for finagent dataset. "
                "finagent uses evaluation_score as target which is not suitable for regression. "
                "The regression analysis will be skipped."
            )
        elif args.analysis_type == "all":
            logger.info(
                "Note: For finagent dataset, regression analysis will be skipped. "
                "Only classification analysis will be performed."
            )

    try:
        # Determine target_dataset for output directory
        target_dataset = None
        if merged_datasets and len(merged_datasets) == 1:
            target_dataset = merged_datasets[0]

        # Initialize the analyzer with appropriate parameters
        analyzer = DataMiningAnalyzer(
            base_dir=base_dir,
            data_path=data_path,
            output_dir=output_dir,
            target_dataset=target_dataset,
            skip_collection=args.skip_collection,
            run_shap=args.run_shap,
            model_names=model_names,
            architectures=architectures,
            datasets=datasets,
            exclude_features=args.exclude_features,
            dataset_type=args.dataset_type,
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

        if "pca" in results:
            logger.info("\nPCA Analysis:")
            if "optimal_components" in results["pca"]:
                opt_comp = results["pca"]["optimal_components"]
                if isinstance(opt_comp, dict) and "recommended_components" in opt_comp:
                    logger.info(
                        f"  Optimal components: {opt_comp['recommended_components']}"
                    )
                else:
                    logger.info(f"  Optimal components: {opt_comp}")
            if "comparison" in results["pca"]:
                comp = results["pca"]["comparison"]
                if "original" in comp and "pca" in comp:
                    for model_name in comp["original"].keys():
                        if model_name in comp["pca"]:
                            logger.info(
                                f"  {model_name}: Original Acc={comp['original'][model_name]['Accuracy']:.4f}, "
                                f"PCA Acc={comp['pca'][model_name]['Accuracy']:.4f}"
                            )

        if "feature_ablation" in results:
            logger.info("\nFeature Ablation Analysis:")
            logger.info("  Feature ablation analysis completed. See detailed report.")

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
