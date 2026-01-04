## Evaluation Guidance

### Overview

The evaluation module provides comprehensive tools for analyzing multi-agent experiment results, including performance metrics, entropy statistics, and cross-architecture comparisons.

### Module Structure

#### Core Components

- **[DataLoader](../evaluation/data_loader.py)**: Loads experiment data, configurations, and results
- **[MetricsCalculator](../evaluation/metrics_calculator.py)**: Calculates performance metrics
- **[ExperimentAnalyzer](../evaluation/experiment_analyzer.py)**: Analyzes experiment results
- **[EntropyAnalyzer](../evaluation/entropy_analyzer.py)**: Analyzes entropy statistics
- **[evaluator.py](../evaluation/evaluator.py)**: Main entry point for evaluation
- **[utils.py](../evaluation/utils.py)**: Utility functions for saving results

### Usage

#### Command Line Interface

Run evaluation from the project root:

```bash
python -m evaluation.evaluator --dataset gsm8k --task-type math
```

#### Available Arguments

| Argument              | Type | Default | Description                                 |
| --------------------- | ---- | ------- | ------------------------------------------- |
| `--dataset`           | str  | gsm8k   | Dataset to analyze (gsm8k, humaneval, mmlu) |
| `--task-type`         | str  | math    | Task type (math, code, option)              |
| `--experiment`        | str  | None    | Specific experiment to analyze              |
| `--output`            | str  | None    | Output file path                            |
| `--compare`           | flag | False   | Compare experiments by architecture         |
| `--save-csv`          | bool | True    | Save summary to CSV                         |
| `--analyze-entropy`   | bool | True    | Perform entropy analysis                    |
| `--entropy-compare`   | bool | True    | Compare entropy across architectures        |
| `--save-entropy-json` | bool | True    | Save entropy results to JSON                |

#### Examples

**Analyze a single experiment:**
```bash
python -m evaluation.evaluator --dataset gsm8k --experiment single_gsm8k
```

**Compare all experiments:**
```bash
python -m evaluation.evaluator --dataset gsm8k --compare
```

**Analyze with custom output:**
```bash
python -m evaluation.evaluator --dataset humaneval --output results/custom.json
```

### Metrics

#### Performance Metrics

- **Accuracy**: Correct answer rate
- **Format Compliance Rate**: Percentage of valid answer formats
- **Time Cost**: Average execution time
- **Average Entropy**: Mean entropy value

#### Entropy Statistics

**Macro Statistics:**
- Total entropy per experiment
- Average entropy per round
- Sample-level statistics

**Micro Statistics:**
- Agent-level entropy (max, mean, variance, median, Q1, Q3)
- Change rate statistics
- Token position analysis

### Agent Architectures

Supported architectures:
- **single**: Single solver agent
- **sequential**: Sequential multi-agent system
- **centralized**: Centralized orchestration
- **debate**: Debate-based multi-agent system
- **hybrid**: Hybrid multi-agent system

### Output Structure

Results are saved to `evaluation/results/{dataset}/`:

```
evaluation/results/
├── {dataset}/
│   ├── {experiment_name}_metrics.json
│   ├── all_metrics.json
│   ├── comparison.json
│   ├── all_metrics_summary.csv
│   └── entropy/
│       ├── {experiment_name}_entropy.json
│       ├── all_entropy_results.json
│       ├── macro_statistics.csv
│       ├── micro_statistics.csv
│       ├── token_position_statistics.csv
│       └── architecture_comparison.csv
```

### Programmatic Usage

```python
from evaluation import ExperimentAnalyzer, EntropyAnalyzer

base_path = "/path/to/project"

## Analyze experiments
analyzer = ExperimentAnalyzer(base_path)
metrics = analyzer.analyze_experiment("gsm8k", "single_gsm8k", "math")

## Analyze entropy
entropy_analyzer = EntropyAnalyzer(base_path)
entropy_results = entropy_analyzer.analyze_experiment_entropy("gsm8k", "single_gsm8k")
```

### Data Loading

The [DataLoader](file:///home/yuxuanzhao/multiagent-entropy/evaluation/data_loader.py) class handles:

- Ground truth data from `experiments/data/{dataset}/train-all-samples.json`
- Experiment configs from `experiments/configs_exp/{experiment}.yml`
- Results from `experiments/results/raw/{dataset}/{experiment}/traces/`
- Entropy tensors from `traces/tensors/{result_id}_extras_entropy.pt`
