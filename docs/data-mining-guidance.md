# Multi-Agent Entropy Data Mining Analysis

## Project Overview

This project performs comprehensive data mining analysis on multi-agent experiment results, focusing on understanding how various features influence experiment accuracy and sample-level correctness.

## Project Structure

```
data_mining/
├── data/                           # Data storage
│   └── merged_datasets.csv         # Merged dataset from all sources
├── code/                           # Source code
│   ├── data_collector.py           # Data collection and merging module
│   ├── regression_analyzer.py      # Experiment-level regression analysis module
│   ├── classification_analyzer.py  # Sample-level classification analysis module
│   ├── shap_analyzer.py            # SHAP analysis module for model interpretability
│   ├── data_mining_analyzer.py     # Unified entry point (delegates to specialized analyzers)
│   ├── features.py                 # Includes all features for analysis
│   ├── utils.py                    # Shared utility functions and constants
│   ├── main.py                     # Command-line interface (uses data_mining_analyzer)
│   ├── run_experiments.py          # Automated experiment runner for batch processing
│   └── data_mining_analysis.log    # Execution log
└── results/                        # Analysis outputs
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
