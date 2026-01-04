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
- **experiment_level**: Total entropy, average entropy, total samples, total results
- **round_level**: Total entropy and average entropy per round
- **agent_level**: Agent-level entropy statistics including:
  - total_entropy: Sum of all entropy values for the agent
  - average_entropy: Average entropy per sample for the agent
  - mean_entropy: Mean of all entropy values
  - max_entropy: Maximum entropy value
  - min_entropy: Minimum entropy value
  - median_entropy: Median entropy value
  - std_entropy: Standard deviation of entropy values
  - variance_entropy: Variance of entropy values
  - q1_entropy: First quartile (25th percentile)
  - q3_entropy: Third quartile (75th percentile)
  - average_tokens_per_sample: Average number of tokens per sample
- **architecture_comparison**: Architecture comparison data

**Micro Statistics:**
- **sequence_level**: Sequence-level entropy statistics with composite keys in the format `{main_id}-{agent_type}-{execution_order}`. Each key uniquely identifies a specific sequence and contains:
  - total_entropy: Sum of all entropy values for the sequence
  - max_entropy: Maximum entropy value
  - mean_entropy: Mean entropy value
  - variance_entropy: Variance of entropy values
  - median_entropy: Median entropy value
  - q1_entropy: First quartile (25th percentile)
  - q3_entropy: Third quartile (75th percentile)
  - std_entropy: Standard deviation of entropy values
  - min_entropy: Minimum entropy value
  - token_count: Total number of tokens
  - sample_count: Number of samples
  - average_entropy_per_token: Average entropy per token
  - average_tokens_per_sample: Average number of tokens per sample

  Example structure for single agent architecture:
  ```json
  "sequence_level": {
    "ID1-SingleSolver-1": {
      "total_entropy": 123.45,
      "max_entropy": 1.123,
      "mean_entropy": 0.062,
      "variance_entropy": 0.031,
      "median_entropy": 0.0,
      "q1_entropy": 0.0,
      "q3_entropy": 0.012,
      "std_entropy": 0.176,
      "min_entropy": 0.0,
      "token_count": 2000,
      "sample_count": 1,
      "average_entropy_per_token": 0.062,
    },
    "ID1-SingleSolver-2": {
      ...
    },
    ...
  }
  ```

  The total number of sequence entries follows the formula: 100 (samples) × number of agents in the architecture × number of rounds.
- **token_position_level**: Token position analysis including:
  - mean: Mean entropy at each token position
  - std: Standard deviation at each token position
  - median: Median entropy at each token position
  - min: Minimum entropy at each token position
  - max: Maximum entropy at each token position
  - q1: First quartile at each token position
  - q3: Third quartile at each token position
  - count: Number of samples at each token position

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

### JSON Result File Structure

The entropy analysis generates JSON files with the following structure:

#### all_entropy_results.json

```json
{
  "dataset": "gsm8k",
  "architectures": {
    "centralized": ["experiment_name_1"],
    "debate": ["experiment_name_2"],
    ...
  },
  "experiments": {
    "experiment_name": {
      "experiment_name": "...",
      "dataset": "gsm8k",
      "agent_architecture": "centralized",
      "num_rounds": 2,
      "num_samples": 100,
      "macro_statistics": {
        "experiment_level": {
          "total_entropy": 37961.39,
          "average_entropy": 47.45,
          "total_samples": 100,
          "total_results": 800
        },
        "round_level": {
          "1": {
            "total_entropy": 28156.25,
            "count": 400,
            "average_entropy": 70.39
          },
          "2": {
            "total_entropy": 9805.14,
            "count": 400,
            "average_entropy": 24.51
          }
        },
        "agent_level": {
          "MathAgent": {
            "total_entropy": 12345.67,
            "average_entropy": 61.73,
            "mean_entropy": 0.051,
            "max_entropy": 1.123,
            "min_entropy": 0.0,
            "median_entropy": 0.0,
            "std_entropy": 0.195,
            "variance_entropy": 0.028,
            "q1_entropy": 0.0,
            "q3_entropy": 0.005,
            "average_tokens_per_sample": 690.94
          },
          ...
        },
        "architecture_comparison": {}
      },
      "micro_statistics": {
        "sequence_level": {
          "ID1-MathAgent-1": {
            "total_entropy": 123.45,
            "max_entropy": 1.123,
            "mean_entropy": 0.062,
            "variance_entropy": 0.031,
            "median_entropy": 0.0,
            "q1_entropy": 0.0,
            "q3_entropy": 0.012,
            "std_entropy": 0.176,
            "min_entropy": 0.0,
            "token_count": 2000,
            "sample_count": 1,
            "average_entropy_per_token": 0.062,
          },
          "ID1-MathAgent-2": {
            ...
          },
          ...
        },
        "token_position_level": {
          "0": {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "count": 800
          },
          "1": {
            "mean": 0.044,
            "std": 0.151,
            "median": 0.0,
            "min": 0.0,
            "max": 1.123,
            "q1": 0.0,
            "q3": 0.0,
            "count": 800
          },
          ...
        }
      }
    }
  }
}
```

#### Key Changes in Structure

**Macro Statistics** now includes:
- `experiment_level`: High-level experiment summary
- `round_level`: Per-round entropy statistics
- `agent_level`: Per-agent entropy statistics (moved from micro_statistics)
- `architecture_comparison`: Cross-architecture comparison data

**Micro Statistics** now includes:
- `sequence_level`: Per-sequence entropy statistics (new, replaces agent_level)
- `token_position_level`: Per-token-position entropy statistics (enhanced with min, max, q1, q3)

#### CSV Output Files

**macro_statistics.csv**:
- experiment_name, agent_architecture, num_rounds, num_samples, total_results, total_entropy, average_entropy, level, round_number, count

**micro_statistics.csv**:
- experiment_name, agent_architecture, sequence_id (format: {main_id}-{agent_type}-{execution_order}), total_entropy, max_entropy, mean_entropy, variance_entropy, median_entropy, q1_entropy, q3_entropy, std_entropy, min_entropy, token_count, sample_count, average_entropy_per_token, average_tokens_per_sample

**token_position_statistics.csv**:
- experiment_name, agent_architecture, token_position, mean_entropy, std_entropy, median_entropy, min_entropy, max_entropy, q1_entropy, q3_entropy, count

**architecture_comparison.csv**:
- agent_architecture, experiment_name, total_entropy, average_entropy, num_samples

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

The [DataLoader](../evaluation/data_loader.py) class handles:

- Ground truth data from `experiments/data/{dataset}/train-all-samples.json`
- Experiment configs from `experiments/configs_exp/{experiment}.yml`
- Results from `experiments/results/raw/{dataset}/{experiment}/traces/`
- Entropy tensors from `traces/tensors/{result_id}_extras_entropy.pt`
