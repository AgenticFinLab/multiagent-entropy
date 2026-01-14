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
│   ├── data_mining_analyzer.py     # Unified entry point (delegates to specialized analyzers)
│   ├── main.py                     # Entry point with command-line interface
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
        └── classification/
            ├── classification_report.txt
            ├── Feature_Correlation_Heatmap_-_Sample_Level_Classification.png
            ├── Feature_Importance_-_RandomForest_(Classification).png
            ├── Feature_Importance_-_XGBoost_(Classification).png
            └── Feature_Importance_-_LightGBM_(Classification).png
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

### Unified Analysis
- **Module**: [data_mining_analyzer.py](../data_mining/code/data_mining_analyzer.py) - serves as a unified entry point that delegates to specialized analyzers
- **Features**: Backward compatibility with existing code

### Visualization
- Feature importance rankings (top 20 features)
- Correlation heatmap (lower triangle with values)
- High-resolution plots (300 DPI)

## Usage

### Prerequisites

Install required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
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
# Run full analysis (default)
python main.py --analysis-type all --datasets aime2025

# Run only regression analysis
python main.py --analysis-type regression --datasets aime2025

# Run only classification analysis
python main.py --analysis-type classification --datasets aime2025

# Skip data collection step (use existing merged data)
python main.py --skip-collection --analysis-type regression

# Specify multiple datasets
python main.py --analysis-type all --datasets aime2025 gsm8k
```

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

### Visualization Files
- `regression/Feature_Correlation_Heatmap_-_Experiment_Level_Regression.png`: Correlation matrix for regression
- `classification/Feature_Correlation_Heatmap_-_Sample_Level_Classification.png`: Correlation matrix for classification
- `regression/Feature_Importance_*.png`: Feature importance plots for regression models
- `classification/Feature_Importance_*.png`: Feature importance plots for classification models

## Code Quality

- Follows Google Python Style Guide
- Comprehensive docstrings and comments
- Modular design with clear separation of concerns
- Error handling and logging
- Type hints for better code clarity

## Future Enhancements

- Add cross-validation for more robust evaluation
- Implement hyperparameter tuning
- Add feature selection methods
- Include SHAP values for model interpretability
- Add time-series analysis for round-by-round dynamics
- Implement ensemble methods combining multiple models