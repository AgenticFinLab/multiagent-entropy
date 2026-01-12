# Entropy Analysis for Multi-Agent Systems

## Overview

This project provides comprehensive entropy analysis tools for evaluating multi-agent system performance. The analysis explores when different architectures of multi-agent systems outperform or underperform single-agent systems from an entropy perspective.

## Project Structure

```
entropy_analysis/
├── code/
│   ├── main.py                  # Main execution script
│   ├── data_loader.py           # Hierarchical data loading and preprocessing
│   ├── entropy_analyzer.py      # Core entropy analysis algorithms
│   ├── visualizer.py            # Visualization generation with multi-model support
│   ├── utils.py                 # Utility functions for data processing
│   ├── constants.py             # Shared constants and configurations
│   └── error_handling.py        # Error handling utilities
├── results/
│   └── <dataset_name>/          # Analysis results by dataset
│       └── <model_name>/        # Optional: Model-specific results
│           ├── processed_data.csv
│           └── *.csv            # Various analysis outputs
└── visualizations/
    └── <dataset_name>/          # Visualization plots by dataset
        └── <model_name>/        # Optional: Model-specific visualizations
            └── *.png            # Generated plots
```

## Features

### Hierarchical Data Loading
- Load experimental data at multiple levels: dataset, model, and experiment
- Support for dataset-level aggregation across all models
- Model-specific analysis for detailed comparisons
- Automatic discovery of available datasets, models, and experiments
- Cross-level data aggregation and comparison

### Data Loading and Preprocessing
- Load experimental data from aggregated CSV files
- Support multiple granularity levels: experiment, sample, agent, and round
- Generate summary statistics and architecture comparisons
- Filter and preprocess data for analysis
- Feature grouping and categorization

### Entropy Analysis
- **Architecture Differences**: ANOVA tests to identify significant entropy feature differences across architectures
- **Correlation Analysis**: Examine relationships between entropy features and accuracy
- **Evolution Tracking**: Analyze entropy changes across processing rounds
- **Collaboration Patterns**: Compare entropy characteristics in multi-agent vs single-agent systems
- **Sample-Architecture Interaction**: Study how sample properties interact with different architectures
- **Advanced Analysis**: Principal Component Analysis (PCA) and clustering for deeper insights

### Visualization
- Architecture entropy comparison plots (with multi-model support)
- Entropy-accuracy correlation scatter plots (with dual classification for model and architecture)
- Round entropy evolution plots
- Collaboration pattern comparisons
- Entropy feature heatmaps
- Distribution comparison plots
- Comprehensive summary dashboard

### Error Handling
- Custom exception classes for hierarchical data operations
- Centralized error handling with context tracking
- Graceful error recovery and reporting
- Validation utilities for data integrity

## Usage

### Basic Usage

Analyze a specific dataset:

```bash
cd /home/yuxuanzhao/multiagent-entropy/entropy_analysis/code
python main.py --dataset gsm8k
```

### Model-Specific Analysis

Analyze a specific model within a dataset:

```bash
python main.py --dataset aime2024 --model qwen3_4b
```

### Multi-Level Analysis

Perform multi-level analysis across datasets, models, and experiments:

```bash
python main.py --dataset aime2025 --multi-level
```

### Command Line Arguments

- `--dataset`: Name of the dataset to analyze (default: `aime2025`)
  - Options: `gsm8k`, `aime2024`, `aime2025`, `math500`, `mmlu`, `humaneval`
  - Example: `--dataset aime2024`
- `--model`: Name of the model to analyze (optional)
  - If not provided, performs dataset-level analysis across all models
  - Example: `--model qwen3_4b`
- `--multi-level`: Enable multi-level analysis across datasets, models, and experiments
  - Analyzes hierarchical data structure
  - Generates cross-level comparisons and aggregations

### Input Data Format

The script expects aggregated data in CSV format at:

**Dataset-level analysis** (all models):
```
/home/yuxuanzhao/multiagent-entropy/evaluation/results/<dataset_name>/all_aggregated_data.csv
```

**Model-level analysis** (specific model):
```
/home/yuxuanzhao/multiagent-entropy/evaluation/results/<dataset_name>/<model_name>/aggregated_data.csv
```

### Output Organization

Results are organized by dataset and model:

**Dataset-level analysis**:
- **Processed Data**: `entropy_analysis/results/<dataset_name>/processed_data.csv`
- **Analysis Results**: `entropy_analysis/results/<dataset_name>/`
- **Visualizations**: `entropy_analysis/visualizations/<dataset_name>/`

**Model-level analysis**:
- **Processed Data**: `entropy_analysis/results/<dataset_name>/<model_name>/processed_data.csv`
- **Analysis Results**: `entropy_analysis/results/<dataset_name>/<model_name>/`
- **Visualizations**: `entropy_analysis/visualizations/<dataset_name>/<model_name>/`

## Supported Architectures

The analysis supports five multi-agent system architectures:

1. **centralized**: Centralized coordination architecture
2. **debate**: Debate-based architecture (orchestrator excluded from analysis)
3. **hybrid**: Hybrid collaboration architecture
4. **sequential**: Sequential processing architecture
5. **single**: Single-agent baseline

## Analysis Pipeline

The analysis follows these steps:

1. **Data Loading**: Load and preprocess experimental data at the specified level (dataset, model, or experiment)
2. **Summary Generation**: Create architecture comparison summaries
3. **Data Processing**: Save processed data for further analysis
4. **Entropy Analysis**: Execute comprehensive entropy analysis including:
   - Architecture differences analysis (ANOVA tests)
   - Entropy-accuracy correlation analysis
   - Round entropy evolution tracking
   - Collaboration pattern comparison
   - Sample-architecture interaction analysis
   - Principal Component Analysis (PCA)
   - K-means clustering analysis
5. **Result Storage**: Save analysis results to CSV files
6. **Visualization**: Generate all visualization plots with support for multi-model comparison
7. **Reporting**: Display key findings and insights

## Key Findings

The analysis provides insights into:

- Which architectures show the highest/lowest entropy values
- How entropy features correlate with prediction accuracy
- How entropy evolves across processing rounds
- Differences between multi-agent and single-agent systems
- Optimal architecture choices for different scenarios
- Model-specific entropy patterns and performance characteristics
- Cross-dataset comparisons and generalizability of findings

## Hierarchical Data Loading System

The entropy analysis system supports a hierarchical data structure that enables flexible multi-level analysis:

### Data Hierarchy

```
evaluation/results/
├── <dataset_name>/
│   ├── all_aggregated_data.csv    # Dataset-level aggregation (all models)
│   ├── all_metrics.json           # Dataset-level metrics
│   ├── all_entropy_results.json   # Dataset-level entropy results
│   └── <model_name>/
│       ├── aggregated_data.csv    # Model-level aggregation
│       ├── metrics.json           # Model-level metrics
│       └── <experiment_name>.csv  # Individual experiment data
```

### Data Loading Methods

The `DataLoader` class provides multiple methods for accessing data at different levels:

- **`load_dataset_level_data(dataset)`**: Load aggregated data for all models in a dataset
- **`load_model_level_data(dataset, model)`**: Load aggregated data for a specific model
- **`load_experiment_level_data(dataset, model, experiment)`**: Load data for a specific experiment
- **`load_all_levels(dataset, model, experiment)`**: Load data at multiple levels based on parameters
- **`aggregate_across_models(dataset, models)`**: Aggregate data across multiple models
- **`aggregate_across_experiments(dataset, model, experiments)`**: Aggregate data across multiple experiments

### Discovery Methods

- **`get_available_datasets()`**: Discover all available datasets
- **`get_available_models(dataset)`**: Get list of models for a specific dataset
- **`get_available_experiments(dataset, model)`**: Get list of experiments for a specific model
- **`get_hierarchy_info(dataset)`**: Get complete hierarchy information

### Multi-Level Analysis

The `--multi-level` flag enables comprehensive analysis across the entire hierarchy:

- Analyzes data at dataset, model, and experiment levels
- Generates cross-level comparisons and aggregations
- Provides insights into how patterns vary across different levels of granularity
- Supports both dataset-level and model-specific reporting

## Example Output

After running the analysis, you will find:

### Results Directory
- `processed_data.csv`: Preprocessed experimental data
- `architecture_differences_*.csv`: Statistical tests and comparisons
- `entropy_accuracy_correlation_*.csv`: Correlation analysis results
- `round_entropy_evolution_*.csv`: Evolution tracking data
- `collaboration_patterns_*.csv`: Multi-agent vs single-agent comparisons
- `sample_architecture_interaction_*.csv`: Interaction analysis results
- `pca_analysis_*.csv`: Principal component analysis outputs
- `clustering_analysis_*.csv`: Clustering analysis results

### Visualizations Directory
- `architecture_entropy_comparison.png`: Box plots comparing entropy across architectures
  - For dataset-level analysis: Creates a 2×N grid where N is the number of models
  - Each column represents a different model
  - Each row shows a different entropy feature
- `entropy_accuracy_correlation.png`: Scatter plots with regression lines
  - For dataset-level analysis: Uses dual classification (model=color, architecture=marker)
  - Shows correlation between entropy features and accuracy
- `round_entropy_evolution.png`: Entropy changes across rounds
  - Total and average entropy evolution with error bars
  - Comparison between correct and incorrect samples
  - Architecture-specific evolution patterns
- `collaboration_comparison.png`: Multi-agent vs single-agent comparisons
- `entropy_heatmap.png`: Correlation heatmap of entropy features
- `accuracy_entropy_scatter.png`: Accuracy vs entropy scatter plots
- `distribution_comparison.png`: Distribution plots for key features
- `summary_dashboard.png`: Comprehensive dashboard of key findings

## Visualization Features

### Multi-Model Support

When analyzing at the dataset level with multiple models, the visualizations provide:

1. **Dual Classification**: Points are classified by both model (color) and architecture (marker)
2. **Grid Layout**: Architecture comparison plots use a 2×N grid layout for easy model comparison
3. **Separate Analysis**: Each model can be analyzed independently while maintaining cross-model comparisons
4. **Aggregated Trends**: Overall trend lines show patterns across all models

### Visualization Types

1. **Architecture Entropy Comparison**
   - Box plots showing distribution of entropy features across architectures
   - Color-coded by model for dataset-level analysis
   - Supports up to 10 models with distinct colors

2. **Entropy-Accuracy Correlation**
   - 3×3 grid of scatter plots for different entropy features
   - Regression lines showing overall trends
   - Dual classification for model and architecture
   - Correlation coefficients displayed for each feature

3. **Round Entropy Evolution**
   - Line plots showing entropy changes across rounds
   - Error bars showing standard deviation
   - Separate analysis for correct vs incorrect samples
   - Architecture-specific evolution patterns

4. **Collaboration Pattern Comparison**
   - Bar charts comparing multi-agent vs single-agent systems
   - Architecture-specific performance metrics
   - Entropy and accuracy comparisons

5. **Entropy Heatmap**
   - Correlation matrix of entropy features
   - Color-coded correlation coefficients
   - Hierarchical clustering of features

6. **Distribution Comparison**
   - Histograms and KDE plots for key features
   - Comparison across architectures
   - Statistical annotations (mean, median, std)

7. **Summary Dashboard**
   - Comprehensive overview of key findings
   - Multiple subplots in a single figure
   - Executive summary of analysis results

## Notes

- The debate architecture excludes orchestrator data from analysis as it is a voting mechanism, not a true agent
- All paths are computed dynamically to support different datasets and models
- Output directories are created automatically if they don't exist
- Multi-model analysis requires dataset-level aggregated data (`all_aggregated_data.csv`)
- The system uses OpenBLAS thread limit (4 threads) to prevent memory allocation errors
- Warnings are suppressed for cleaner output during analysis

## Module Details

### Constants Module ([constants.py](file:///home/yuxuanzhao/multiagent-entropy/entropy_analysis/code/constants.py))

Defines shared constants used across the entropy analysis system:

- **ARCHITECTURES**: List of all supported architecture types
  - `centralized`, `debate`, `hybrid`, `sequential`, `single`
- **MULTI_AGENT_ARCHITECTURES**: Subset of architectures that are multi-agent
- **SINGLE_AGENT_ARCHITECTURES**: Subset of architectures that are single-agent
- **FEATURE_GROUPS**: Categorized feature groups (sample, agent, round, experiment, metadata)
- **ENTROPY_FEATURES**: List of all entropy-related features
- **METADATA_COLUMNS**: List of metadata column names
- **BASE_MODEL_COLUMNS**: List of base model comparison columns

### Error Handling Module ([error_handling.py](file:///home/yuxuanzhao/multiagent-entropy/entropy_analysis/code/error_handling.py))

Provides comprehensive error handling utilities:

- **Custom Exceptions**:
  - `HierarchicalDataError`: Base exception for hierarchical data operations
  - `DatasetNotFoundError`: Raised when a dataset is not found
  - `ModelNotFoundError`: Raised when a model is not found
  - `ExperimentNotFoundError`: Raised when an experiment is not found
  - `FileNotFoundError`: Raised when a required file is not found
  - `DataFormatError`: Raised when data format is invalid
  - `MissingColumnError`: Raised when required columns are missing
  - `InvalidDataError`: Raised when data is invalid or corrupted
  - `AnalysisError`: Raised when an analysis operation fails

- **Validation Functions**:
  - `validate_file_exists()`: Validate that a file exists
  - `validate_directory_exists()`: Validate that a directory exists
  - `validate_columns()`: Validate that required columns are present
  - `safe_load_json()`: Safely load a JSON file with error handling
  - `safe_load_csv()`: Safely load a CSV file with error handling

- **ErrorHandler Class**:
  - Centralized error handling with context tracking
  - Error counting and summary generation
  - Configurable exception raising behavior
  - Optional logger integration

### Utils Module ([utils.py](file:///home/yuxuanzhao/multiagent-entropy/entropy_analysis/code/utils.py))

Provides shared utility functions:

- **Feature Grouping**:
  - `get_feature_groups()`: Categorize features based on prefixes

- **Data Preprocessing**:
  - `preprocess_data()`: Convert data types and handle missing values

- **Statistics**:
  - `get_summary_statistics()`: Generate comprehensive summary statistics
  - `get_architecture_comparison()`: Compare metrics across architectures
  - `calculate_metrics_from_data()`: Calculate performance metrics

## Technical Implementation

### Data Preprocessing Pipeline

1. **Type Conversion**: Convert columns to appropriate data types
   - Architecture: categorical
   - Boolean flags: boolean
   - Entropy features: numeric

2. **Missing Value Handling**: Convert invalid values to NaN
   - Use `pd.to_numeric()` with `errors='coerce'` for entropy features

3. **Feature Extraction**: Identify and categorize features
   - Sample-level features (prefix: `sample_`)
   - Agent-level features (prefix: `agent_`)
   - Round-level features (prefix: `round_`)
   - Experiment-level features (prefix: `exp_`)
   - Metadata columns

### Statistical Analysis Methods

1. **ANOVA Tests**: Identify significant differences across architectures
   - One-way ANOVA for each entropy feature
   - F-statistic and p-value calculation

2. **Correlation Analysis**: Examine relationships between variables
   - Pearson correlation coefficient
   - Significance threshold: |r| > 0.1

3. **T-Tests**: Compare multi-agent vs single-agent systems
   - Independent samples t-test
   - Mean difference and significance testing

4. **PCA**: Dimensionality reduction for pattern discovery
   - StandardScaler normalization
   - Retain components explaining 95% variance
   - Loadings analysis for feature importance

5. **K-Means Clustering**: Group samples by entropy characteristics
   - StandardScaler normalization
   - Default: 3 clusters
   - Cluster statistics and architecture distribution

### Visualization Techniques

1. **Box Plots**: Distribution comparison across architectures
   - Color-coded boxes
   - Outlier detection
   - Grid layout for multi-model comparison

2. **Scatter Plots**: Correlation visualization
   - Regression lines with confidence intervals
   - Dual classification (model + architecture)
   - Error handling for insufficient data

3. **Line Plots**: Evolution tracking
   - Error bars for uncertainty
   - Multiple series comparison
   - Correct vs incorrect sample analysis

4. **Heatmaps**: Correlation matrix visualization
   - Color-coded correlation coefficients
   - Hierarchical clustering
   - Annotated values

5. **Bar Charts**: Categorical comparison
   - Multi-agent vs single-agent
   - Architecture-specific metrics
   - Performance comparisons

## Performance Considerations

- **Memory Management**: OpenBLAS thread limit set to 4 to prevent memory allocation errors
- **Data Loading**: Efficient CSV reading with pandas
- **Visualization**: High DPI (300) for publication-quality plots
- **Error Handling**: Graceful degradation for missing or invalid data
- **Batch Processing**: Support for analyzing multiple models and experiments
