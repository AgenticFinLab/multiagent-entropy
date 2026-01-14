# Multi-Agent Entropy Data Mining Analysis

## Project Overview

This project performs comprehensive data mining analysis on multi-agent experiment results, focusing on understanding how various features influence experiment accuracy and sample-level correctness. The analysis now includes SHAP interpretability features and a refactored codebase with shared utilities. The project also includes an automated experiment runner to facilitate batch execution of multiple experiment configurations.

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

### Utilities Module
- **Module**: [utils.py](../data_mining/code/utils.py) - contains shared utility functions and constants
- **Components**: EXCLUDE_COLUMNS, data loading, categorical encoding, feature preparation, directory management, visualization setup
- **Purpose**: Eliminates code duplication across analyzer modules

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

### Running Individual Analysis Types

The main.py script now supports command-line arguments:

```bash
# Run full analysis with SHAP (default)
python main.py --analysis-type all --datasets aime2025

# Run only regression analysis
python main.py --analysis-type regression --datasets aime2025

# Run only classification analysis
python main.py --analysis-type classification --datasets aime2025

# Skip data collection step (use existing merged data)
python main.py --skip-collection --analysis-type regression

# Skip SHAP analysis (faster execution)
python main.py --analysis-type all --run-shap false

# Specify multiple datasets
python main.py --analysis-type all --datasets aime2025 gsm8k

# Automatically discover and use all available datasets
python main.py --analysis-type all --datasets '*'

# Specify custom data path when skipping collection
python main.py --skip-collection --data-path /custom/path/data.csv
```

### Running Batch Experiments with the Automated Runner

The experiment runner allows you to automate multiple experiments with different parameter combinations:

```bash
# Run with default configurations
python3 run_experiments.py

# Run with custom configuration file
python3 run_experiments.py --config-file my_configs.json

# Run experiments in parallel
python3 run_experiments.py --parallel --config-file my_configs.json

# Limit parallel workers
python3 run_experiments.py --parallel --max-workers 4 --config-file my_configs.json

# Dry run to see what would be executed
python3 run_experiments.py --dry-run --config-file my_configs.json
```

#### Command-Line Arguments for the Experiment Runner

- `--config-file`: Path to JSON file containing experiment configurations
- `--parallel`: Run experiments in parallel mode (default: serial)
- `--max-workers`: Maximum number of parallel workers (default: CPU count)
- `--output-dir`: Directory to save experiment reports (default: experiment_reports)
- `--dry-run`: Show what experiments would be run without executing them

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
  - `exclude_features`: Feature exclusion strategy ('default', '*', or specific groups)
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

## Output Files

### Data Files
- `merged_datasets.csv`: Combined dataset with 1,300 records from 2 datasets (aime2024, gsm8k)

### Report Files
- `unified_analysis_report.txt`: Comprehensive text report with all metrics and rankings
- `regression/regression_report.txt`: Regression-specific analysis report
- `classification/classification_report.txt`: Classification-specific analysis report

### Experiment Runner Output Files
The experiment runner generates several output files in the specified output directory:

- `experiment_results_<timestamp>.json`: Detailed results in JSON format
- `experiment_summary_<timestamp>.txt`: Summary report with key metrics
- `experiment_results_<timestamp>.csv`: Results in CSV format for analysis

### Visualization Files
- `regression/Feature_Correlation_Heatmap_-_Experiment_Level_Regression.png`: Correlation matrix for regression
- `classification/Feature_Correlation_Heatmap_-_Sample_Level_Classification.png`: Correlation matrix for classification
- `regression/Feature_Importance_*.png`: Feature importance plots for regression models
- `classification/Feature_Importance_*.png`: Feature importance plots for classification models
- `shap/shap_summary_*.png`: SHAP summary plots showing feature importance
- `shap/shap_importance_*.png`: SHAP importance plots with detailed feature impacts
- `shap/shap_dependence_*.png`: SHAP dependence plots showing feature interactions
- `shap/shap_waterfall_*.png`: SHAP waterfall plots for individual prediction explanations
