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
│   ├── data_mining_analyzer.py     # Main analysis module
│   ├── main.py                     # Entry point
│   └── data_mining_analysis.log    # Execution log
└── results/                        # Analysis outputs
    ├── analysis_report.txt         # Comprehensive analysis report
    ├── Feature_Correlation_Heatmap_-_Experiment_Level.png
    ├── Feature_Importance_-_RandomForest_(Regression).png
    ├── Feature_Importance_-_XGBoost_(Regression).png
    ├── Feature_Importance_-_LightGBM_(Regression).png
    ├── Feature_Importance_-_RandomForest_(Classification).png
    ├── Feature_Importance_-_XGBoost_(Classification).png
    └── Feature_Importance_-_LightGBM_(Classification).png
```

## Data Sources

The analysis uses data from:
- `/home/yuxuanzhao/multiagent-entropy/evaluation/results/aime2024/all_aggregated_data_exclude_agent.csv`
- `/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/all_aggregated_data_exclude_agent.csv`

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
- **Metrics**: MSE, MAE, R²

### Sample-Level Analysis (Classification)
- **Target Variable**: `is_finally_correct` (sample correctness)
- **Algorithms**: Random Forest, XGBoost, LightGBM
- **Metrics**: Accuracy, Precision, Recall, F1-Score

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
cd /home/yuxuanzhao/multiagent-entropy/data_mining/code
python3 main.py
```

### Running Individual Modules

#### Data Collection Only

```bash
python3 data_collector.py
```

#### Analysis Only (requires merged data)

```bash
python3 data_mining_analyzer.py
```

## Analysis Results

### Experiment-Level Findings

All three regression models achieved near-perfect performance (R² = 1.0000), indicating that the features can perfectly predict experiment accuracy.

**Top Important Features (Random Forest)**:
1. `round_1_total_time` (75.78%)
2. `round_2_total_time` (16.72%)
3. `round_1_2_change_entropy` (1.19%)
4. `round_1_2_change_tokens` (0.83%)
5. `exp_total_time` (0.82%)

### Sample-Level Findings

Classification models achieved excellent performance:
- Random Forest: Accuracy = 91.92%, F1 = 95.52%
- XGBoost: Accuracy = 90.38%, F1 = 94.65%
- LightGBM: Accuracy = 91.92%, F1 = 95.52%

**Top Important Features (Random Forest)**:
1. `sample_total_entropy` (10.68%)
2. `sample_entropy_stability_index` (10.45%)
3. `sample_variance_entropy` (10.15%)
4. `sample_avg_entropy_per_token` (8.15%)
5. `sample_avg_entropy_per_agent` (7.83%)

## Key Insights

1. **Time-based features** (round total time, experiment total time) are most influential for experiment-level accuracy
2. **Entropy-based features** (sample total entropy, entropy stability index) are most influential for sample-level correctness
3. **Entropy dynamics** (change in entropy between rounds) plays a significant role
4. **Token generation patterns** influence both experiment and sample performance

## Output Files

### Data Files
- `merged_datasets.csv`: Combined dataset with 1,300 records from 2 datasets (aime2024, gsm8k)

### Report Files
- `analysis_report.txt`: Comprehensive text report with all metrics and rankings

### Visualization Files
- `Feature_Correlation_Heatmap_-_Experiment_Level.png`: Correlation matrix
- `Feature_Importance_*.png`: Feature importance plots for each model and analysis type

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

## Contact

For questions or issues, please refer to the project documentation or contact the development team.