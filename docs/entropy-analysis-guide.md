# Entropy Analysis for Multi-Agent Systems

## Overview

This project provides comprehensive entropy analysis tools for evaluating multi-agent system performance. The analysis explores when different architectures of multi-agent systems outperform or underperform single-agent systems from an entropy perspective.

## Project Structure

```
entropy_analysis/
├── code/
│   ├── main.py                  # Main execution script
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── entropy_analyzer.py      # Core entropy analysis algorithms
│   └── visualizer.py            # Visualization generation
├── results/
│   └── <dataset_name>/          # Analysis results by dataset
│       ├── processed_data.csv
│       └── *.csv                # Various analysis outputs
└── visualizations/
    └── <dataset_name>/          # Visualization plots by dataset
        └── *.png                # Generated plots
```

## Features

### Data Loading and Preprocessing
- Load experimental data from aggregated CSV files
- Support multiple granularity levels: experiment, sample, agent, and round
- Generate summary statistics and architecture comparisons
- Filter and preprocess data for analysis

### Entropy Analysis
- **Architecture Differences**: ANOVA tests to identify significant entropy feature differences across architectures
- **Correlation Analysis**: Examine relationships between entropy features and accuracy
- **Evolution Tracking**: Analyze entropy changes across processing rounds
- **Collaboration Patterns**: Compare entropy characteristics in multi-agent vs single-agent systems
- **Sample-Architecture Interaction**: Study how sample properties interact with different architectures
- **Advanced Analysis**: Principal Component Analysis (PCA) and clustering for deeper insights

### Visualization
- Architecture entropy comparison plots
- Entropy-accuracy correlation scatter plots
- Round entropy evolution plots
- Collaboration pattern comparisons
- Entropy feature heatmaps
- Distribution comparison plots
- Comprehensive summary dashboard

## Usage

### Basic Usage

Analyze a specific dataset:

```bash
cd /home/yuxuanzhao/multiagent-entropy/entropy_analysis/code
python main.py --dataset gsm8k
```

### Command Line Arguments

- `--dataset`: Name of the dataset to analyze (default: `gsm8k`)
  - Example: `--dataset aime2024`
  - Example: `--dataset gsm8k`

### Input Data Format

The script expects aggregated data in CSV format at:
```
/home/yuxuanzhao/multiagent-entropy/evaluation/results/<dataset_name>/aggregated/aggregated_data.csv
```

### Output Organization

Results are organized by dataset:

- **Processed Data**: `entropy_analysis/results/<dataset_name>/processed_data.csv`
- **Analysis Results**: `entropy_analysis/results/<dataset_name>/`
- **Visualizations**: `entropy_analysis/visualizations/<dataset_name>/`

## Supported Architectures

The analysis supports five multi-agent system architectures:

1. **centralized**: Centralized coordination architecture
2. **debate**: Debate-based architecture (orchestrator excluded from analysis)
3. **hybrid**: Hybrid collaboration architecture
4. **sequential**: Sequential processing architecture
5. **single**: Single-agent baseline

## Analysis Pipeline

The analysis follows these steps:

1. **Data Loading**: Load and preprocess experimental data
2. **Summary Generation**: Create architecture comparison summaries
3. **Data Processing**: Save processed data for further analysis
4. **Entropy Analysis**: Execute comprehensive entropy analysis
5. **Result Storage**: Save analysis results to CSV files
6. **Visualization**: Generate all visualization plots
7. **Reporting**: Display key findings and insights

## Key Findings

The analysis provides insights into:

- Which architectures show the highest/lowest entropy values
- How entropy features correlate with prediction accuracy
- How entropy evolves across processing rounds
- Differences between multi-agent and single-agent systems
- Optimal architecture choices for different scenarios

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
- `entropy_accuracy_correlation.png`: Scatter plots with regression lines
- `round_entropy_evolution.png`: Entropy changes across rounds
- `collaboration_comparison.png`: Multi-agent vs single-agent comparisons
- `entropy_heatmap.png`: Correlation heatmap of entropy features
- `accuracy_entropy_scatter.png`: Accuracy vs entropy scatter plots
- `distribution_comparison.png`: Distribution plots for key features
- `summary_dashboard.png`: Comprehensive dashboard of key findings

## Notes

- The debate architecture excludes orchestrator data from analysis as it is a voting mechanism, not a true agent
- All paths are computed dynamically to support different datasets
- Output directories are created automatically if they don't exist
