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
cd entropy_analysis/code
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

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `aime2024` | Name of the dataset to analyze |
| `--model` | str | `None` | Name of the model to analyze (optional). If not provided, performs dataset-level analysis |
| `--multi-level` | flag | `False` | Enable multi-level analysis across datasets, models, and experiments |

#### Supported Datasets
- `gsm8k`: GSM8K math dataset
- `aime2024`: AIME 2024 competition problems
- `aime2025`: AIME 2025 competition problems
- `math500`: MATH-500 dataset
- `mmlu`: Massive Multitask Language Understanding dataset
- `humaneval`: HumanEval code generation dataset

#### Usage Examples

```bash
# Analyze a specific dataset
cd entropy_analysis/code
python main.py --dataset gsm8k

# Analyze a specific model within a dataset
python main.py --dataset aime2024 --model qwen3_4b

# Perform multi-level analysis
python main.py --dataset aime2025 --multi-level

# Analyze specific dataset and model with multi-level analysis
python main.py --dataset aime2024 --model qwen3_4b --multi-level
```

### Input Data Format

The script expects aggregated data in CSV format at:

**Dataset-level analysis** (all models):
```
evaluation/results/<dataset_name>/all_aggregated_data.csv
```

**Model-level analysis** (specific model):
```
evaluation/results/<dataset_name>/<model_name>/aggregated_data.csv
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

The analysis supports seven multi-agent system architectures defined in [constants.py](../entropy_analysis/code/constants.py):

| Architecture | Type | Description |
|--------------|------|-------------|
| **single** | Single-agent | Single-agent baseline for comparison |
| **sequential** | Multi-agent | Sequential processing architecture with agents in a pipeline |
| **centralized** | Multi-agent | Centralized coordination with domain agents and central orchestrator |
| **decentralized** | Multi-agent | Decentralized architecture with loopback mechanism |
| **full_decentralized** | Multi-agent | Fully decentralized where each agent can communicate with all others |
| **debate** | Multi-agent | Debate-based architecture with majority voting (orchestrator uses voting, not LLM) |
| **hybrid** | Multi-agent | Hybrid collaboration with enhanced context sharing |

### Architecture Categories

- **Single-agent architectures**: `single`
- **Multi-agent architectures**: `centralized`, `debate`, `hybrid`, `sequential`, `decentralized`, `full_decentralized`

## Analysis Pipeline

The analysis follows these steps:

1. **Data Loading**: Load and preprocess experimental data at the specified level (dataset, model, or experiment)
2. **Summary Generation**: Create architecture comparison summaries
3. **Data Processing**: Save processed data for further analysis
4. **Entropy Analysis**: Execute comprehensive entropy analysis using EntropyAnalyzer class
5. **Result Storage**: Save analysis results to CSV files
6. **Visualization**: Generate all visualization plots with support for multi-model comparison
7. **Reporting**: Display key findings and insights

### EntropyAnalyzer Class Methods

The [EntropyAnalyzer](file:///d:/GitHub/multiagent-entropy/entropy_analysis/code/entropy_analyzer.py) class provides comprehensive analysis capabilities:

| Method | Description | Analysis Type |
|--------|-------------|---------------|
| `analyze_architecture_differences()` | Performs ANOVA tests to identify significant entropy feature differences across architectures. Calculates mean, std, and median for each architecture. | Statistical Testing |
| `analyze_entropy_accuracy_correlation()` | Calculates Pearson correlation coefficients between entropy features and accuracy. Identifies significant features (\|r\| > 0.1). | Correlation Analysis |
| `analyze_round_entropy_evolution()` | Tracks entropy changes across processing rounds. Compares evolution patterns between correct and incorrect samples. | Temporal Analysis |
| `analyze_collaboration_patterns()` | Compares multi-agent vs single-agent systems using t-tests. Analyzes architecture-specific performance metrics. | Comparative Analysis |
| `analyze_sample_architecture_interaction()` | Examines entropy-accuracy correlations by architecture. Compares high vs low entropy sample performance. | Interaction Analysis |
| `perform_pca_analysis()` | Performs Principal Component Analysis on entropy features. Retains components explaining 95% variance. Uses StandardScaler normalization. | Dimensionality Reduction |
| `perform_clustering_analysis(n_clusters=3)` | Performs K-means clustering on entropy features. Groups samples by entropy characteristics. Default: 3 clusters. | Clustering |
| `generate_comprehensive_report()` | Executes all analysis methods and combines results into a single comprehensive report. | Report Generation |
| `save_results(output_dir)` | Saves all analysis results to CSV files in the specified directory. | Output |

#### Analysis Method Details

**Architecture Differences Analysis**
- Calculates descriptive statistics (mean, std, median) for each entropy feature by architecture
- Performs one-way ANOVA for each entropy feature across all architectures
- Returns F-statistic and p-value for each feature

**Entropy-Accuracy Correlation**
- Computes Pearson correlation between all entropy features and `exp_accuracy`
- Sorts features by correlation strength
- Identifies significant features with \|correlation\| > 0.1

**Round Entropy Evolution**
- Groups data by sample_id and agent_round_number
- Tracks: round_total_entropy, round_infer_avg_entropy, round_total_time, round_total_token
- Separates analysis for correct vs incorrect samples
- Calculates mean statistics per round

**Collaboration Pattern Analysis**
- Compares MULTI_AGENT_ARCHITECTURES vs SINGLE_AGENT_ARCHITECTURES
- Performs independent samples t-test for each entropy feature
- Calculates architecture-specific metrics: mean_entropy, accuracy, avg_tokens

**Sample-Architecture Interaction**
- Analyzes sample-level features (prefix: `sample_`) by architecture
- Identifies high entropy samples (>75th percentile) vs low entropy samples (<25th percentile)
- Compares accuracy between entropy levels across architectures

**PCA Analysis**
- Standardizes entropy features using StandardScaler
- Retains principal components explaining 95% of variance
- Returns loadings matrix and explained variance ratios

**Clustering Analysis**
- Standardizes entropy features using StandardScaler
- Performs K-means clustering (default k=3, random_state=42)
- Analyzes cluster statistics and architecture distribution within clusters

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

#### Level-Specific Loading Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `load_dataset_level_data(dataset, include_models=None)` | Load aggregated data for all models in a dataset | `dataset`: dataset name; `include_models`: optional list of models to include |
| `load_model_level_data(dataset, model)` | Load aggregated data for a specific model | `dataset`: dataset name; `model`: model name |
| `load_experiment_level_data(dataset, model, experiment)` | Load data for a specific experiment | `dataset`: dataset name; `model`: model name; `experiment`: experiment name |
| `load_all_levels(dataset, model=None, experiment=None)` | Load data at multiple levels based on parameters | Returns dict with `data`, `metrics`, `entropy_results` |

#### Aggregation Methods

| Method | Description |
|--------|-------------|
| `aggregate_across_models(dataset, models=None)` | Aggregate data across multiple models |
| `aggregate_across_experiments(dataset, model, experiments=None)` | Aggregate data across multiple experiments |

#### Discovery Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_available_datasets()` | Discover all available datasets | List of dataset names |
| `get_available_models(dataset)` | Get list of models for a specific dataset | List of model names |
| `get_available_experiments(dataset, model)` | Get list of experiments for a specific model | List of experiment names |
| `get_hierarchy_info(dataset=None)` | Get complete hierarchy information | Dict with hierarchy structure |

#### Analysis Methods

| Method | Description |
|--------|-------------|
| `analyze_dataset_level(dataset, include_models=None)` | Analyze data at dataset level with model comparison |
| `analyze_model_level(dataset, model)` | Analyze data at model level with architecture comparison |
| `analyze_experiment_level(dataset, model, experiment)` | Analyze data at experiment level with sample/agent statistics |
| `compare_across_datasets(datasets, metric='exp_accuracy')` | Compare performance across multiple datasets |
| `compare_across_models(dataset, models=None)` | Compare performance across models within a dataset |
| `compare_architectures_across_models(dataset, models=None)` | Compare architecture performance across models |

### Multi-Level Analysis

The `--multi-level` flag enables comprehensive analysis across the entire hierarchy:

- Analyzes data at dataset, model, and experiment levels
- Generates cross-level comparisons and aggregations
- Provides insights into how patterns vary across different levels of granularity
- Supports both dataset-level and model-specific reporting

## Output Files and Formats

### Results Directory Structure

```
entropy_analysis/results/
└── <dataset_name>/
    ├── processed_data.csv                    # Preprocessed experimental data
    ├── architecture_differences_statistics.csv    # Mean/Std/Median by architecture
    ├── architecture_differences_anova.csv         # ANOVA F-statistic and p-value
    ├── entropy_accuracy_correlation_correlations.csv  # Pearson correlations
    ├── entropy_accuracy_correlation_significant_features.csv  # Significant features (|r|>0.1)
    ├── round_entropy_evolution_overall_stats.csv   # Overall round statistics
    ├── round_entropy_evolution_correct_samples.csv # Correct sample evolution
    ├── round_entropy_evolution_incorrect_samples.csv # Incorrect sample evolution
    ├── collaboration_patterns_multi_vs_single.csv  # Multi-agent vs single-agent comparison
    ├── collaboration_patterns_arch_comparison.csv  # Architecture-specific metrics
    ├── sample_architecture_interaction_correlation_by_arch.csv  # Correlation by architecture
    ├── sample_architecture_interaction_entropy_level_comparison.csv  # High vs low entropy accuracy
    ├── pca_analysis_loadings.csv            # PCA component loadings
    ├── pca_analysis_explained_variance.csv  # Explained variance ratios
    ├── clustering_analysis_cluster_stats.csv       # Cluster statistics
    ├── clustering_analysis_cluster_arch_distribution.csv  # Architecture distribution per cluster
    └── <model_name>/                        # Model-specific results (if model-level analysis)
        └── [same file structure as above]
```

### CSV File Formats

**architecture_differences_statistics.csv**
```csv
,single,sequential,centralized,decentralized,full_decentralized,debate,hybrid
sample_mean_entropy,0.234,0.189,0.212,0.198,0.205,0.221,0.215
sample_std_entropy,0.156,0.134,0.145,0.138,0.142,0.151,0.148
...
```

**entropy_accuracy_correlation_correlations.csv**
```csv
,correlation
sample_mean_entropy,-0.342
sample_std_entropy,-0.287
...
```

**pca_analysis_explained_variance.csv**
```csv
PC,Explained_Variance_Ratio,Cumulative_Variance
PC1,0.4523,0.4523
PC2,0.2314,0.6837
...
```

### Visualizations Directory

```
entropy_analysis/visualizations/
└── <dataset_name>/
    ├── architecture_entropy_comparison.png        # Box plots by architecture
    ├── entropy_accuracy_correlation.png           # Scatter plots with regression
    ├── round_entropy_evolution.png                # Line plots across rounds
    ├── collaboration_comparison.png               # Multi-agent vs single-agent bar charts
    ├── entropy_heatmap.png                        # Correlation heatmap
    ├── accuracy_entropy_scatter.png               # 2x2 scatter plot grid
    ├── distribution_comparison.png                # Distribution histograms
    ├── cross_model_architecture_comparison.png    # Cross-model comparison (dataset-level)
    └── <model_name>/                              # Model-specific visualizations
        └── [same file structure as above]
```

## Visualization Features

### Multi-Model Support

When analyzing at the dataset level with multiple models, the visualizations provide:

1. **Dual Classification**: Points are classified by both model (color) and architecture (marker)
2. **Grid Layout**: Architecture comparison plots use a 2×N grid layout for easy model comparison
3. **Separate Analysis**: Each model can be analyzed independently while maintaining cross-model comparisons
4. **Aggregated Trends**: Overall trend lines show patterns across all models

### Visualization Types

The [EntropyVisualizer](file:///d:/GitHub/multiagent-entropy/entropy_analysis/code/visualizer.py) class generates the following visualizations:

| Method | Output File | Description |
|--------|-------------|-------------|
| `plot_architecture_entropy_comparison()` | `architecture_entropy_comparison.png` | Box plots comparing entropy features across architectures. Dataset-level: 2×N grid by model. |
| `plot_entropy_accuracy_correlation()` | `entropy_accuracy_correlation.png` | 3×3 grid of scatter plots with regression lines. Dual classification (model=color, architecture=marker). |
| `plot_round_entropy_evolution()` | `round_entropy_evolution.png` | 2×2 grid showing entropy changes across rounds. Includes correct vs incorrect comparison. |
| `plot_collaboration_comparison()` | `collaboration_comparison.png` | 2×2 grid comparing multi-agent vs single-agent systems. Bar charts for accuracy and entropy. |
| `plot_entropy_heatmap()` | `entropy_heatmap.png` | Correlation heatmap of entropy features. Dataset-level: separate heatmap per model. |
| `plot_accuracy_entropy_scatter()` | `accuracy_entropy_scatter.png` | 2×2 grid: sample_mean_entropy, sample_std_entropy, sample_max_entropy, token_count vs accuracy. |
| `plot_distribution_comparison()` | `distribution_comparison.png` | 2×3 grid of histograms for 6 entropy features. Dual classification by model and architecture. |
| `plot_cross_model_architecture_comparison()` | `cross_model_architecture_comparison.png` | Bar charts comparing architecture performance across models (dataset-level only). |
| `plot_base_model_comparison()` | `base_model_comparison.png` | Compares multi-agent system performance with base model performance. |
| `generate_all_visualizations()` | All above files | Generates all visualization plots in sequence. |

#### Visualization Details

**Architecture Entropy Comparison**
- Features: entropy features containing "mean" in name
- Colors: #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7
- Dataset-level: 2 rows × N columns grid (N = number of models)

**Entropy-Accuracy Correlation**
- 3×3 grid layout (up to 9 entropy features)
- Markers: o, s, ^, v, <, >, D, p, *, h
- Colors: tab10 colormap
- Shows correlation coefficient in title
- Includes regression line with error handling

**Round Entropy Evolution**
- 4 subplots: Total entropy, Average entropy, Correct vs Incorrect, By model/architecture
- Error bars showing standard deviation
- Separate lines for correct (green) and incorrect (red) samples

**Collaboration Comparison**
- 4 subplots: Accuracy comparison, Entropy comparison, Architecture accuracy ranking, Architecture entropy ranking
- Dataset-level: grouped by model with side-by-side bars
- Colors: #4ECDC4 (multi-agent), #FF6B6B (single-agent)

**Entropy Heatmap**
- Uses coolwarm colormap with center=0
- Upper triangle masked
- Annotated with correlation values (2 decimal places)
- Dataset-level: separate subplot per model with consistent color scale

**Distribution Comparison**
- 6 features: sample_mean_entropy, sample_std_entropy, sample_max_entropy, sample_min_entropy, sample_median_entropy, sample_q3_entropy
- 30 bins per histogram
- Density normalization
- Dataset-level: tab20 colormap for model-architecture combinations

**Cross-Model Architecture Comparison**
- 4 subplots: Accuracy by architecture, Mean entropy, Std entropy, Stacked comparison
- Grouped bar charts by architecture
- Legend shows models

**Base Model Comparison**
- 4 subplots: Accuracy comparison, Format compliance, Improvement heatmap, Architecture-level comparison
- Requires base_model columns in data
- Shows multi-agent vs base model performance

## Notes

- The debate architecture excludes orchestrator data from analysis as it is a voting mechanism, not a true agent
- All paths are computed dynamically to support different datasets and models
- Output directories are created automatically if they don't exist
- Multi-model analysis requires dataset-level aggregated data (`all_aggregated_data.csv`)
- The system uses OpenBLAS thread limit (4 threads) to prevent memory allocation errors
- Warnings are suppressed for cleaner output during analysis

## Module Details

### Constants Module ([constants.py](entropy_analysis/code/constants.py))

Defines shared constants used across the entropy analysis system:

- **ARCHITECTURES**: List of all supported architecture types
  - `centralized`, `debate`, `hybrid`, `sequential`, `single`
- **MULTI_AGENT_ARCHITECTURES**: Subset of architectures that are multi-agent
- **SINGLE_AGENT_ARCHITECTURES**: Subset of architectures that are single-agent
- **FEATURE_GROUPS**: Categorized feature groups (sample, agent, round, experiment, metadata)
- **ENTROPY_FEATURES**: List of all entropy-related features
- **METADATA_COLUMNS**: List of metadata column names
- **BASE_MODEL_COLUMNS**: List of base model comparison columns

### Error Handling Module ([error_handling.py](entropy_analysis/code/error_handling.py))

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

### Utils Module ([utils.py](entropy_analysis/code/utils.py))

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
