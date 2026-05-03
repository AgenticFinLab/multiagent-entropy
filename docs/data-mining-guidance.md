# Multi-Agent Entropy Data Mining Analysis

## Project Overview

This project performs comprehensive data mining analysis on multi-agent experiment results, focusing on understanding how various features influence experiment accuracy and sample-level correctness.

## Project Structure

```
data_mining/
├── data/                                # Data storage
│   └── merged_datasets.csv              # Merged dataset from all sources
├── code/                                # Source code
│   ├── base/                            # Shared base-class subpackage
│   │   ├── __init__.py                  # Public re-exports
│   │   ├── analyzer.py                  # BaseAnalyzer — template-method pipeline
│   │   ├── constants.py                 # MODEL_NAMES, default hyperparams, plot defaults
│   │   ├── feature_manager.py           # FeatureManager (FinAgent / standard modes)
│   │   ├── model_factory.py             # ModelFactory.regressor / classifier / feature_importance
│   │   ├── io_utils.py                  # OutputManager, save_plot, load_dataset_csv
│   │   ├── cli.py                       # Shared argparse builder helpers
│   │   └── post_processor.py            # BasePostProcessor (experiment-iteration helper)
│   ├── causal_analysis/                 # Causal inference subpackage
│   │   ├── causal_discovery.py          # PC/FCI causal graph learning (causal-learn)
│   │   ├── causal_effect_estimator.py   # ATE/CATE estimation with DoWhy
│   │   ├── causal_mediation_analyzer.py # Mediation analysis (direct vs indirect effects)
│   │   ├── causal_report_generator.py   # Unified causal analysis report generator
│   │   ├── feature_selection_crossval.py # Cross-validated feature selection for causal pipeline
│   │   └── run_causal_on_correlation_results.py # Orchestrator: runs 4-stage causal pipeline on existing correlation slices
│   ├── regression_analyzer.py           # RegressionAnalyzer(BaseAnalyzer) — predicts exp_accuracy
│   ├── classification_analyzer.py       # ClassificationAnalyzer(BaseAnalyzer) — predicts is_finally_correct
│   ├── shap_analyzer.py                 # ShapAnalyzer(BaseAnalyzer) — SHAP interpretability
│   ├── pca_analyzer.py                  # PCAAnalysis(BaseAnalyzer) — feature redundancy
│   ├── feature_ablation_analyzer.py     # FeatureAblationAnalyzer(BaseAnalyzer) — ablation study
│   ├── calibration_analyzer.py          # CalibrationAnalyzer — probability calibration
│   ├── data_mining_analyzer.py          # Unified orchestrator (delegates to specialized analyzers)
│   ├── mas_causal_analysis.py           # SAS vs MAS separation-control causal analysis
│   ├── aggregator.py                    # Experiment result aggregation
│   ├── visualizer.py                    # Aggregated result visualization
│   ├── summarizer.py                    # Statistical summarization
│   ├── data_collector.py                # Data collection and merging
│   ├── features.py                      # Feature group definitions
│   ├── utils.py                         # Shared utility functions (thin shims over base/)
│   ├── main.py                          # Command-line interface
│   ├── run_experiments.py               # Automated batch experiment runner
│   └── data_mining_analysis.log         # Execution log
└── results/                             # Analysis outputs
    └── {dataset}/
        ├── unified_analysis_report.txt
        ├── regression/
        │   ├── regression_report.txt
        │   ├── Feature_Correlation_Heatmap_-_Experiment_Level_Regression.png
        │   ├── Feature_Importance_-_RandomForest_(Regression).png
        │   ├── Feature_Importance_-_XGBoost_(Regression).png
        │   └── Feature_Importance_-_LightGBM_(Regression).png
        ├── classification/
        │   ├── classification_report.txt
        │   ├── Feature_Correlation_Heatmap_-_Sample_Level_Classification.png
        │   ├── Feature_Importance_-_RandomForest_(Classification).png
        │   ├── Feature_Importance_-_XGBoost_(Classification).png
        │   └── Feature_Importance_-_LightGBM_(Classification).png
        └── shap/
            ├── shap_analysis_report.txt
            ├── shap_summary_RandomForest_regression.png
            ├── shap_importance_XGBoost_classification.png
            └── [other SHAP visualization files]
```

## Data Sources

The analysis uses data from:
- `multiagent-entropy/evaluation/results/{dataset_name}/all_aggregated_data_exclude_agent.csv`

## Code Architecture

The `data_mining/code/` package is organized around a **base-class template** to eliminate duplicated load → encode → train → evaluate scaffolding that previously appeared in every analyzer.

### `base/` Subpackage

| Module | Class / Function | Purpose |
| --- | --- | --- |
| `base/analyzer.py` | `BaseAnalyzer` | Template-method base class shared by all analyzers |
| `base/constants.py` | `MODEL_NAMES`, `PLOT_DEFAULTS`, … | Single source for model names, default hyperparameters, plot config |
| `base/feature_manager.py` | `FeatureManager` | Encapsulates FinAgent step-entropy discovery and feature exclusion |
| `base/model_factory.py` | `ModelFactory` | Constructs RF / XGBoost / LightGBM estimators; extracts `feature_importances_` |
| `base/io_utils.py` | `OutputManager`, `save_plot`, `load_dataset_csv` | Output-path resolution, figure saving, CSV loading |
| `base/cli.py` | `add_filter_args`, `add_io_args`, … | Shared argparse builder helpers used by `main.py` and `run_experiments.py` |
| `base/post_processor.py` | `BasePostProcessor` | Walks `results/` trees and yields `ExperimentContext` for aggregator / visualizer |

### `BaseAnalyzer` Pipeline

All specialized analyzers (`RegressionAnalyzer`, `ClassificationAnalyzer`, `PCAAnalysis`, `FeatureAblationAnalyzer`, `ShapAnalyzer`) inherit from `BaseAnalyzer` and share this pipeline:

```
load_data() → encode_categorical_features() → prepare_features()
    → split() → train_models() → run_analysis() → generate_report()
```

Subclasses configure behavior via class attributes and hook overrides:

| Attribute / Hook | Purpose |
| --- | --- |
| `target_column` | Target variable (e.g. `"exp_accuracy"`, `"is_finally_correct"`) |
| `analyzer_type` | Output subdirectory label (e.g. `"regression"`) |
| `is_classification` | Enables stratified split and classifier factory |
| `_metrics(y_true, y_pred)` | Returns task-specific metric dict (MSE/R² or Accuracy/F1) |
| `_postprocess_model(name, model, X_test)` | Per-model post-processing (e.g. `predict_proba` capture) |
| `run_analysis()` | Full orchestration after data is loaded |
| `generate_report()` | Writes the text report and returns its path |

Analyzers with fully custom pipelines (PCA, feature ablation, SHAP) override `run_analysis` and `generate_report` while still inheriting the data-loading and encoding infrastructure.

## Key Features

### Data Collection
- Automatically discovers dataset folders
- Merges CSV files from multiple datasets
- Adds dataset identification column
- Handles missing values

### Experiment-Level Analysis (Regression)
- **Target Variable**: `exp_accuracy` (experiment accuracy)
- **Algorithms**: Random Forest, XGBoost, LightGBM
- **Excluded Features**: `is_finally_correct` (used to calculate target), `dataset`
- **Metrics**: MSE, MAE, R^2
- **Module**: [regression_analyzer.py](../data_mining/code/regression_analyzer.py)

### Sample-Level Analysis (Classification)
- **Target Variable**: `is_finally_correct` (sample correctness)
- **Algorithms**: Random Forest, XGBoost, LightGBM
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Module**: [classification_analyzer.py](../data_mining/code/classification_analyzer.py)

### SHAP Analysis (Model Interpretability)
- **Purpose**: Provides SHAP (SHapley Additive exPlanations) analysis for model interpretability
- **Support**: Works with tree-based models (Random Forest, XGBoost, LightGBM)
- **Visualizations**: Summary plots, importance plots, dependence plots, waterfall plots
- **Module**: [shap_analyzer.py](../data_mining/code/shap_analyzer.py)

### Unified Analysis
- **Module**: [data_mining_analyzer.py](../data_mining/code/data_mining_analyzer.py) - serves as a unified entry point that delegates to specialized analyzers
- **Features**: Backward compatibility with existing code, support for both programmatic and CLI usage, optional SHAP analysis integration

### Automated Experiment Runner
- **Module**: [run_experiments.py](../data_mining/code/run_experiments.py) - enables batch execution of multiple experiment configurations
- **Features**: 
  - Define multiple experiment configurations with different parameter sets
  - Execute experiments serially or in parallel
  - Track and log results from each experiment
  - Generate comprehensive reports in JSON, CSV, and text formats
  - Handle errors gracefully to ensure one failed experiment doesn't stop the batch
  - Timeout protection to prevent hung processes
  - Automatic experiment result aggregation after analysis
  - Visualization of aggregated results
  - Hierarchical statistical analysis and summarization

### Utilities Module
- **Module**: [utils.py](../data_mining/code/utils.py) - contains shared utility functions and constants
- **Components**: EXCLUDE_COLUMNS, data loading, categorical encoding, feature preparation, directory management, visualization setup, filter functions
- **Purpose**: Eliminates code duplication across analyzer modules

### Feature Groups

The analysis supports flexible feature exclusion through predefined feature groups defined in [features.py](../data_mining/code/features.py). Use the `--exclude-features` parameter to specify which feature groups to exclude.

#### Available Feature Groups

| Group Name | Description | Example Features |
|------------|-------------|------------------|
| `default` | Default exclusions (recommended) | EXPERIMENT_IDENTIFIER, SAMPLE_IDENTIFIER, EXPERIMENT_METRICS |
| `base_model_wo_entropy` | Base model metrics without entropy | base_model_accuracy, base_model_is_finally_correct |
| `base_model_all_metrics` | All base model metrics | All base model metrics including entropy features |
| `experiment_identifier` | Experiment identification columns | model_name, sample_id, num_rounds |
| `sample_identifier` | Sample identification columns | is_finally_correct, final_format_compliance |
| `experiment_statistics` | Experiment-level statistics | exp_total_entropy, exp_accuracy, exp_total_time |
| `round_statistics` | Round-level statistics | round_1_total_entropy, round_1_2_change_tokens |
| `sample_statistics` | Sample-level statistics | sample_total_entropy, sample_mean_entropy, sample_std_entropy |
| `sample_distribution_shape` | Distribution shape features | sample_entropy_range, sample_entropy_skewness, sample_entropy_cv |
| `sample_baseline_entropy` | Baseline entropy features | base_sample_total_entropy, sample_entropy_ratio_vs_base_total |
| `aggregation_over_agents` | Agent aggregation features | sample_avg_entropy_per_agent |
| `sample_round_wise_aggregated` | Round-wise aggregated features | sample_round_1_all_agents_total_entropy, cross-round changes |
| `cross_round_aggregated` | Cross-round aggregated features | sample_round_mean_agent_mean_entropy_first_last_diff |
| `intra_round_agent_distribution` | Intra-round agent distribution | sample_round_1_agent_mean_entropy_spread, cv, skewness |
| `cross_round_agent_spread_change` | Cross-round spread changes | sample_round_agent_mean_entropy_spread_first_last_diff |
| `sample_round1_agent_statistics` | Round 1 agent statistics | sample_round_1_mean_agent_mean_entropy, sample_round_1_max_agent_total_entropy |
| `sample_round2_agent_statistics` | Round 2 agent statistics | sample_round_2_mean_agent_mean_entropy, sample_round_2_max_agent_total_entropy |

#### Feature Exclusion Syntax

```bash
# Use all features (no exclusions)
python main.py --exclude-features all

# Use default exclusions (recommended)
python main.py --exclude-features default

# Exclude specific feature group
python main.py --exclude-features base_model_metrics

# Exclude multiple feature groups
python main.py --exclude-features "base_model_metrics,experiment_identifier"

# Combine default with additional exclusions
python main.py --exclude-features "default+base_model_metrics"
```

### Command-Line Interface
- **Module**: [main.py](../data_mining/code/main.py) - provides pure command-line interface for the unified analyzer
- **Features**: Eliminates duplicate functionality, acts as wrapper for data_mining_analyzer

### Visualization
- Feature importance rankings (top 20 features)
- Correlation heatmap (lower triangle with values)
- SHAP summary and dependence plots for model interpretability
- High-resolution plots (300 DPI)

## Usage

### Prerequisites

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap
```

### Running the Analysis

Navigate to the code directory and run:

```bash
cd multiagent-entropy/data_mining/code
python main.py
```

### Command Line Arguments

The main.py script supports the following command-line arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--analysis-type` | str | `all` | Type of analysis to run: `all`, `regression`, or `classification` |
| `--merged-datasets` | str[] | `["all"]` | Datasets to merge during data collection. Use `"all"` to auto-discover all available datasets |
| `--model-name` | str[] | `["all"]` | Filter by specific model name(s). Use `"all"` for all models |
| `--architecture` | str[] | `["all"]` | Filter by specific architecture(s). Use `"all"` for all architectures |
| `--dataset` | str[] | `["all"]` | Filter by specific dataset(s) for analysis. Use `"all"` for all datasets |
| `--skip-collection` | flag | `False` | Skip data collection step (use existing merged data) |
| `--data-path` | str | `data_mining/data/merged_datasets.csv` | Path to merged data file (used when skip-collection is True) |
| `--run-shap` | flag | `True` | Run SHAP analysis (default: True) |
| `--exclude-features` | str | `default` | Feature exclusion configuration (see Feature Groups section below) |

### Running Individual Analysis Types

```bash
# Run full analysis with SHAP (default)
python main.py --analysis-type all --merged-datasets aime2025

# Run only regression analysis
python main.py --analysis-type regression --merged-datasets aime2025

# Run only classification analysis
python main.py --analysis-type classification --merged-datasets aime2025

# Skip data collection step (use existing merged data)
python main.py --skip-collection --analysis-type regression

# Specify multiple datasets
python main.py --analysis-type all --merged-datasets aime2025 gsm8k

# Automatically discover and use all available datasets
python main.py --analysis-type all --merged-datasets all

# Specify custom data path when skipping collection
python main.py --skip-collection --data-path /custom/path/data.csv

# Filter by specific model, architecture, and dataset
python main.py --model-name qwen3_4b --architecture centralized --dataset aime2025

# Exclude specific feature groups
python main.py --exclude-features base_model_metrics
python main.py --exclude-features "base_model_metrics,experiment_identifier"
python main.py --exclude-features "default+base_model_metrics"
```

### Running Batch Experiments with the Automated Runner

The experiment runner allows you to automate multiple experiments with different parameter combinations:

```bash
# Run with default configurations (auto-generates configs from default parameter lists)
python run_experiments.py

# Run with custom configuration file
python run_experiments.py --config-file my_configs.json

# Run experiments in parallel
python run_experiments.py --parallel --config-file my_configs.json

# Limit parallel workers
python run_experiments.py --parallel --max-workers 4 --config-file my_configs.json

# Dry run to see what would be executed
python run_experiments.py --dry-run --config-file my_configs.json

# Generate configuration file only without running experiments
python run_experiments.py --generate-config-only

# Run with custom parameter lists
python run_experiments.py --dataset-list aime2024 aime2025 gsm8k --model-list qwen3_4b qwen3_8b

# Run without aggregation, visualization, or summarization
python run_experiments.py --run-aggregation False --run-visualization False --run-summarization False

# Specify number of top features for summarization
python run_experiments.py --n-top-analysis 10
```

#### Command-Line Arguments for the Experiment Runner

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config-file` | str | `None` | Path to JSON file containing experiment configurations |
| `--parallel` | flag | `False` | Run experiments in parallel mode |
| `--max-workers` | int | `None` | Maximum number of parallel workers (default: CPU count) |
| `--output-dir` | str | `data_mining/experiment_reports` | Directory to save experiment reports |
| `--dry-run` | flag | `False` | Show what experiments would be run without executing them |
| `--dataset-list` | str[] | `["math500", "aime2024_8192", "aime2025_8192", "gsm8k", "humaneval", "all"]` | List of dataset names to use |
| `--model-list` | str[] | `["qwen3_4b", "qwen3_8b", "all"]` | List of model names to use |
| `--arch-list` | str[] | `["all"]` | List of architecture types to use |
| `--exclude-feature-list` | str[] | `["base_model_wo_entropy", "base_model_all_metrics", "default"]` | List of exclude feature options |
| `--generate-config-only` | flag | `False` | Generate configuration file only without running experiments |
| `--run-aggregation` | bool | `True` | Run experiment results aggregation after analysis |
| `--run-visualization` | bool | `True` | Run visualization of aggregated results |
| `--run-summarization` | bool | `True` | Run summarization of generated images |
| `--n-top-analysis` | int | `5` | Number of top features to summarize |

#### Configuration File Format

The configuration file can be in two formats:

**Direct Array Format**
```json
[
  {
    "name": "experiment_name",
    "params": {
      "dataset": ["dataset1", "dataset2"],
      "model_name": ["model1", "model2"],
      "architecture": ["arch1"],
      "exclude_features": "default"
    },
    "timeout": 3600
  }
]
```

**Object with Key Format**
```json
{
  "experiment_configs": [
    {
      "name": "experiment_name",
      "params": {
        "dataset": ["dataset1", "dataset2"],
        "model_name": ["model1", "model2"],
        "architecture": ["arch1"],
        "exclude_features": "default"
      },
      "timeout": 3600
    }
  ]
}
```

#### Parameter Details

- `name`: Human-readable name for the experiment
- `params`: Dictionary of parameters to pass to main.py
  - `dataset`: List of dataset names to use
  - `model_name`: List of model names to train/test
  - `architecture`: List of architecture types
  - `exclude_features`: Feature exclusion strategy ('default', 'all', or specific groups)
- `timeout`: Maximum time (in seconds) to allow the experiment to run

### Running Individual Modules

#### Data Collection Only

```bash
python data_collector.py
```

#### Regression Analysis Only (requires merged data)

```bash
python regression_analyzer.py
```

#### Classification Analysis Only (requires merged data)

```bash
python classification_analyzer.py
```

#### Unified Analysis (via delegation)

```bash
python data_mining_analyzer.py
```

### Output Files

#### Data Files
- `merged_datasets.csv`: Combined dataset from multiple sources
- `data_mining/results/{dataset}/merged_datasets.csv`: Dataset-specific merged data (when single dataset specified)

#### Report Files
- `unified_analysis_report.txt`: Comprehensive text report with all metrics and rankings
- `regression/regression_report.txt`: Regression-specific analysis report
- `classification/classification_report.txt`: Classification-specific analysis report
- `shap/shap_analysis_report.txt`: SHAP analysis report

#### Experiment Runner Output Files
The experiment runner generates several output files in the specified output directory (`data_mining/experiment_reports/` by default):

- `experiment_results_<timestamp>.json`: Detailed results in JSON format
- `experiment_summary_<timestamp>.txt`: Summary report with key metrics
- `experiment_results_<timestamp>.csv`: Results in CSV format for analysis

Additionally, when running with aggregation/visualization/summarization:
- `data_mining/results_aggregated/`: Aggregated experiment results
- `data_mining/results_visualizations/`: Visualization plots from aggregated results
- `data_mining/results_summaries/`: Statistical summaries and analysis

#### Visualization Files

**Regression Analysis:**
- `regression/Feature_Correlation_Heatmap_-_Experiment_Level_Regression.png`: Correlation matrix for regression
- `regression/Feature_Correlation_Heatmap_-_Experiment_Level_Regression.csv`: Correlation matrix data
- `regression/Feature_Importance_-_RandomForest_(Regression).png`: Random Forest feature importance
- `regression/Feature_Importance_-_XGBoost_(Regression).png`: XGBoost feature importance
- `regression/Feature_Importance_-_LightGBM_(Regression).png`: LightGBM feature importance

**Classification Analysis:**
- `classification/Feature_Correlation_Heatmap_-_Sample_Level_Classification.png`: Correlation matrix for classification
- `classification/Feature_Correlation_Heatmap_-_Sample_Level_Classification.csv`: Correlation matrix data
- `classification/Feature_Importance_-_RandomForest_(Classification).png`: Random Forest feature importance
- `classification/Feature_Importance_-_XGBoost_(Classification).png`: XGBoost feature importance
- `classification/Feature_Importance_-_LightGBM_(Classification).png`: LightGBM feature importance
- `classification/prediction_probabilities_{model}.csv`: Prediction probabilities for each model

**SHAP Analysis:**
- `shap/shap_summary_{model}_{task_type}.png`: SHAP summary plots (bar type)
- `shap/shap_importance_{model}_{task_type}.png`: SHAP importance plots (dot type)
- `shap/shap_waterfall_sample_{model}_{task_type}.png`: SHAP waterfall plots for sample predictions
- `shap/shap_dependence_plots/shap_dependence_{feature}_{model}_{task_type}.png`: SHAP dependence plots for top 5 features
- `shap/shap_values_{model}_{task_type}.csv`: SHAP values for each sample
- `shap/X_test_{model}_{task_type}.csv`: Test features used for SHAP analysis
- `shap/shap_feature_importance_{model}_{task_type}.csv`: Mean absolute SHAP values per feature
- `shap/shap_prediction_probabilities_{model}_{task_type}.csv`: Prediction probabilities (classification only)

---

## Causal Analysis

The `causal_analysis/` subpackage and `mas_causal_analysis.py` provide two complementary causal analysis workflows.

### 4-Stage Causal Pipeline (`causal_analysis/`)

A modular pipeline for discovering and quantifying causal relationships between entropy features and MAS correctness.

| Module | Purpose |
| --- | --- |
| `feature_selection_crossval.py` | Cross-validated feature selection; produces `selected_features.csv` as pipeline input |
| `causal_discovery.py` | Learns causal graph structure using PC and FCI algorithms (via `causal-learn`); enforces domain-knowledge constraints (temporal ordering, forbidden edges) |
| `causal_effect_estimator.py` | Estimates ATE/CATE for selected features using DoWhy |
| `causal_mediation_analyzer.py` | Decomposes total causal effect into direct and indirect (mediated) components |
| `causal_report_generator.py` | Aggregates all stage outputs into a unified causal analysis report |
| `run_causal_on_correlation_results.py` | Orchestrator: runs the full 4-stage pipeline against every correlation-analysis slice in `data_mining/exp_*/results_aggregated/` |

#### Running the 4-Stage Pipeline

```bash
cd data_mining/code
# Run against all existing correlation result slices
python causal_analysis/run_causal_on_correlation_results.py
```

Output per slice `<slice_id>` is written to:

```
data_mining/exp_<NAME>/results_causal/<slice_id>/
    feature_selection/selected_features.csv
    causal_discovery/
    causal_effects/
    causal_mediation/
    causal_report/
```

### SAS vs MAS Separation-Control Analysis (`mas_causal_analysis.py`)

Pairs matched single-agent (SAS) and multi-agent (MAS) samples on the same question to isolate the causal effect of agent topology on accuracy, controlling for question difficulty via entropy.

Key analyses:

- **SAS vs MAS Round-1 entropy comparison** — tests whether entropy in the first round predicts the outcome gap
- **Entropy-change direction vs accuracy** — classifies samples by whether MAS entropy increases or decreases relative to SAS and measures the accuracy difference
- **Three-way comparison plots** — visualizes (SAS correct, MAS wrong), (both correct), (MAS correct, SAS wrong) cases
- **Paired scatter plots** — per-model/dataset scatter of SAS entropy vs MAS entropy coloured by outcome

#### Running the SAS vs MAS Analysis

```bash
cd data_mining/code
python mas_causal_analysis.py \
  --data-path ../../evaluation/results/{dataset}/all_aggregated_data_exclude_agent.csv \
  --output-dir ../../data_mining/results/{dataset}/causal_sas_mas/
```

Output includes PNG plots and a text report in the specified output directory.
