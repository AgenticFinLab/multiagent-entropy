"""Main execution script for multi-agent system entropy analysis.

This script serves as the entry point for the comprehensive entropy analysis
of multi-agent system performance. It orchestrates the entire analysis pipeline
including data loading, preprocessing, entropy analysis, and visualization
generation.

The analysis aims to explore when different architectures of multi-agent
systems outperform or underperform single-agent systems from an entropy
perspective.

Usage:
    python main.py --dataset <dataset_name>
    python main.py --dataset gsm8k
    python main.py --dataset aime2024

The script will:
1. Load data from evaluation/results/<dataset_name>/all_aggregated_data.csv
2. Store results in entropy_analysis/results/<dataset_name>/
3. Store visualizations in entropy_analysis/visualizations/<dataset_name>/

For model-specific analysis:
    python main.py --dataset <dataset_name> --model <model_name>
    python main.py --dataset aime2024 --model qwen3_4b

The script will:
1. Load data from evaluation/results/<dataset_name>/<model_name>/aggregated_data.csv
2. Store results in entropy_analysis/results/<dataset_name>/<model_name>/
3. Store visualizations in entropy_analysis/visualizations/<dataset_name>/<model_name>/
"""

import os

# Set OpenBLAS thread limit to prevent memory allocation errors
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import sys
import warnings
import argparse
from pathlib import Path
from typing import Dict, Optional

from data_loader import DataLoader
from error_handling import ErrorHandler
from visualizer import EntropyVisualizer
from entropy_analyzer import EntropyAnalyzer


# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path object pointing to the multiagent-entropy project root.
    """
    return Path(__file__).parent.parent.parent


def get_paths(dataset_name: str, model_name: Optional[str] = None) -> Dict[str, Path]:
    """Get all required paths for a given dataset and model.

    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k', 'aime2024').
        model_name: Name of the model (e.g., 'qwen3_4b'). If None, uses dataset-level path.

    Returns:
        Dictionary containing all required paths.
    """
    project_root = get_project_root()

    if model_name:
        data_path = (
            project_root
            / "evaluation"
            / "results"
            / dataset_name
            / model_name
            / "aggregated_data.csv"
        )
        results_path = (
            project_root / "entropy_analysis" / "results" / dataset_name / model_name
        )
        visualizations_path = (
            project_root
            / "entropy_analysis"
            / "visualizations"
            / dataset_name
            / model_name
        )
    else:
        data_path = (
            project_root
            / "evaluation"
            / "results"
            / dataset_name
            / "all_aggregated_data.csv"
        )
        results_path = project_root / "entropy_analysis" / "results" / dataset_name
        visualizations_path = (
            project_root / "entropy_analysis" / "visualizations" / dataset_name
        )

    return {
        "data": data_path,
        "results": results_path,
        "visualizations": visualizations_path,
    }


def main() -> None:
    """Execute the complete entropy analysis pipeline.

    This function orchestrates the entire analysis process:
    1. Load and preprocess experimental data
    2. Generate architecture comparison summary
    3. Save processed data
    4. Perform comprehensive entropy analysis
    5. Save analysis results
    6. Generate visualizations
    7. Display key findings

    The function uses command line arguments to specify the dataset and model,
    allowing flexible analysis of different datasets and models.
    """
    parser = argparse.ArgumentParser(
        description="Analyze multi-agent system entropy for different datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aime2024",
        help="Name of the dataset to analyze, options: gsm8k, aime2024, aime2025, math500, mmlu, humaneval.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name of the model to analyze. If not provided, performs dataset-level analysis",
    )
    parser.add_argument(
        "--multi-level",
        action="store_true",
        help="Enable multi-level analysis across datasets, models, and experiments",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Multi-Agent System Entropy Analysis Project")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    if args.model:
        print(f"Model: {args.model}")
    if args.multi_level:
        print("Multi-level analysis: Enabled")
    print()

    error_handler = ErrorHandler(raise_exceptions=True)

    try:
        if args.multi_level:
            print("Performing Multi-Level Analysis")
            print("-" * 80)

            base_path = get_project_root() / "evaluation" / "results"
            loader = DataLoader(str(base_path), base_path=str(base_path))

            if args.model:
                print(f"Analyzing model-level data for {args.dataset}/{args.model}...")
                report = loader.analyze_model_level(args.dataset, args.model)
            else:
                print(f"Analyzing dataset-level data for {args.dataset}...")
                report = loader.analyze_dataset_level(args.dataset)

            results_path = (
                get_project_root() / "entropy_analysis" / "results" / args.dataset
            )
            if args.model:
                results_path = results_path / args.model
            loader.save_results(str(results_path))

            print("\nMulti-level analysis completed!")
            print(f"Results saved to: {results_path}")
            return

        get_paths(args.dataset, args.model)

        paths = get_paths(args.dataset, args.model)

        if not paths["data"].exists():
            print(f"Error: Data file not found at {paths['data']}")
            print("Please ensure the dataset has been aggregated first.")
            sys.exit(1)

        paths["results"].mkdir(parents=True, exist_ok=True)
        paths["visualizations"].mkdir(parents=True, exist_ok=True)

        print("Step 1: Data Loading and Preprocessing")
        print("-" * 80)
        print(f"Loading data from: {paths['data']}")
        loader = DataLoader(str(paths["data"]))
        processed_data = loader.preprocess_data()

        summary = loader.get_summary_statistics()
        print(f"Total rows: {summary['total_rows']}")
        print(f"Unique samples: {summary['unique_samples']}")
        print(f"Unique experiments: {summary['unique_experiments']}")
        print(f"Architecture types: {', '.join(summary['architectures'])}")
        print(f"Round range: {summary['rounds']}")
        print(
            f"Accuracy range: {summary['accuracy_range'][0]:.3f} - "
            f"{summary['accuracy_range'][1]:.3f}"
        )
        print()

        print("Step 2: Generate Architecture Comparison Summary")
        print("-" * 80)
        comparison = loader.get_architecture_comparison()
        print(comparison)
        print()

        print("Step 3: Save Processed Data")
        print("-" * 80)
        output_file = paths["results"] / "processed_data.csv"
        loader.save_processed_data(output_file)
        print()

        print("Step 4: Execute Entropy Analysis")
        print("-" * 80)
        analyzer = EntropyAnalyzer(processed_data)
        report = analyzer.generate_comprehensive_report()

        print("Analysis Results Summary:")
        print(
            f"- Architecture differences: "
            f"{len(report['architecture_differences']['anova'])} features show significant differences"
        )
        print(
            f"- Entropy-accuracy correlation: "
            f"{len(report['entropy_accuracy_correlation']['significant_features'])} "
            f"significantly correlated features"
        )
        print(
            f"- Round entropy evolution: "
            f"analyzed {len(report['round_entropy_evolution']['overall_stats'])} rounds of data"
        )
        print()

        print("Step 5: Save Analysis Results")
        print("-" * 80)
        analyzer.save_results(str(paths["results"]))
        print()

        print("Step 6: Generate Visualization Charts")
        print("-" * 80)
        analysis_level = "dataset" if args.model is None else "model"

        # Determine number of models for visualization purposes
        num_models = 1
        if analysis_level == "dataset":
            # Try to infer number of models from the data if at dataset level
            if "model_name" in processed_data.columns:
                num_models = processed_data["model_name"].nunique()

        visualizer = EntropyVisualizer(
            processed_data,
            str(paths["visualizations"]),
            analysis_level=analysis_level,
            num_models=num_models,
        )
        visualizer.generate_all_visualizations()
        print()

        print("=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print()
        print("Key Findings:")
        print()

        print("1. Architecture Differences Analysis:")
        arch_diff = report["architecture_differences"]["statistics"]
        entropy_features = arch_diff.index.tolist()
        print(f"   - Analyzed {len(entropy_features)} entropy features")
        print(
            f"   - Architecture with highest mean entropy: "
            f"{arch_diff.mean().idxmax()} ({arch_diff.mean().max():.4f})"
        )
        print(
            f"   - Architecture with lowest mean entropy: "
            f"{arch_diff.mean().idxmin()} ({arch_diff.mean().min():.4f})"
        )

        if "sample_mean_entropy" in arch_diff.index:
            print(
                f"   - Architecture with highest sample mean entropy: "
                f"{arch_diff.loc['sample_mean_entropy'].idxmax()} "
                f"({arch_diff.loc['sample_mean_entropy'].max():.4f})"
            )
            print(
                f"   - Architecture with lowest sample mean entropy: "
                f"{arch_diff.loc['sample_mean_entropy'].idxmin()} "
                f"({arch_diff.loc['sample_mean_entropy'].min():.4f})"
            )
        print()

        print("2. Entropy-Accuracy Correlation:")
        corr = report["entropy_accuracy_correlation"]["correlations"]
        top_positive = corr["correlation"].nlargest(3)
        top_negative = corr["correlation"].nsmallest(3)
        print(f"   - Features with strongest positive correlation:")
        for idx, (feature, value) in enumerate(top_positive.items(), 1):
            print(f"     {idx}. {feature}: {value:.3f}")
        print(f"   - Features with strongest negative correlation:")
        for idx, (feature, value) in enumerate(top_negative.items(), 1):
            print(f"     {idx}. {feature}: {value:.3f}")
        print()

        print("3. Collaboration Pattern Comparison:")
        collab = report["collaboration_patterns"]["arch_comparison"]
        print(
            f"   - Architecture with highest accuracy: "
            f"{collab['accuracy'].idxmax()} ({collab['accuracy'].max():.3f})"
        )
        print(
            f"   - Architecture with lowest accuracy: "
            f"{collab['accuracy'].idxmin()} ({collab['accuracy'].min():.3f})"
        )
        print(
            f"   - Architecture with lowest entropy: "
            f"{collab['mean_entropy'].idxmin()} ({collab['mean_entropy'].min():.3f})"
        )
        print()

        print("4. Output Files:")
        print(f"   - Processed data: {paths['results']}/processed_data.csv")
        print(f"   - Analysis results: {paths['results']}/")
        print(f"   - Visualization charts: {paths['visualizations']}/")
        print()
        print("=" * 80)

    except Exception as e:
        error_handler.handle_error(
            e,
            {
                "dataset": args.dataset,
                "model": args.model,
                "multi_level": args.multi_level,
            },
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
