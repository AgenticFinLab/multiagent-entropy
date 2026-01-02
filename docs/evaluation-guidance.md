# Evaluation Package

This package provides comprehensive evaluation tools for MultiAgent-Entropy experiments.

## Components

### 1. Task Evaluators

- **MathEvaluator**: Evaluates math tasks with support for:
  - Extracting answers from `\boxed{}` format
  - Normalizing numeric answers (fractions, mixed fractions, decimals)
  - Comparing answers with tolerance for floating-point precision

- **CodeEvaluator**: Evaluates code tasks with support for:
  - Extracting code from markdown code blocks
  - Validating Python syntax
  - Executing code against test cases
  - Comparing function implementations

- **OptionEvaluator**: Evaluates multiple-choice tasks with support for:
  - Extracting answers from various formats (boxed, brackets, plain text)
  - Normalizing option letters (A, B, C, D)
  - Confidence analysis

### 2. Entropy Analyzer

Analyzes entropy values from multi-agent experiments:
- Loads entropy data from experiment results
- Calculates statistics (mean, std, min, max, median, quartiles)
- Visualizes entropy change curves per agent
- Visualizes entropy distribution
- Analyzes entropy-accuracy correlation (Pearson and Spearman)
- Compares agents using statistical tests (t-test, KS-test)

### 3. Report Generator

Generates comprehensive evaluation reports:
- Text-based reports with detailed metrics
- HTML reports with embedded visualizations
- Summary tables for multiple experiments
- Comparison plots across experiments

## Usage

### Basic Evaluation

```bash
# Evaluate a single experiment
python evaluation/evaluate.py -i experiments/results/raw/gsm8k/experiment_name -o evaluation/results

# Evaluate all experiments in a directory
python evaluation/evaluate.py -i experiments/results/raw/gsm8k -o evaluation/results --task-type math

# Auto-detect task type
python evaluation/evaluate.py -i experiments/results/raw -o evaluation/results --task-type auto
```

### Advanced Options

```bash
# Generate comparison across multiple experiments
python evaluation/evaluate.py -i experiments/results/raw -o evaluation/results --compare

# Skip entropy analysis
python evaluation/evaluate.py -i experiments/results/raw -o evaluation/results --no-entropy

# Skip visualizations
python evaluation/evaluate.py -i experiments/results/raw -o evaluation/results --no-visualization

# Specify report format
python evaluation/evaluate.py -i experiments/results/raw -o evaluation/results --format html

# Evaluate specific experiments
python evaluation/evaluate.py -i experiments/results/raw -o evaluation/results --experiment-names exp1 exp2 exp3
```

## Output Structure

```
evaluation/results/
├── experiment_name_report_20260102_120000.txt
├── experiment_name_report_20260102_120000.html
├── experiment_name_visualizations/
│   ├── entropy_curves.png
│   ├── entropy_distribution.png
│   ├── entropy_accuracy_correlation.png
│   └── agent_comparison.png
├── experiments_summary.csv
├── experiments_comparison.png
└── evaluation_summary.json
```

## Extensibility

The evaluation system is designed to be easily extensible:

1. **Adding New Task Types**: Create a new evaluator class inheriting from `BaseEvaluator`
2. **Custom Metrics**: Extend the evaluator classes to add custom metrics
3. **Custom Visualizations**: Add new visualization methods to `EntropyAnalyzer`
4. **Custom Reports**: Extend `ReportGenerator` to support new report formats

## Requirements

- Python 3.7+
- numpy
- matplotlib
- seaborn
- pandas
- scipy
