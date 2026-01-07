## Evaluation Guidance

### Overview

The evaluation module provides comprehensive tools for analyzing multi-agent experiment results, including performance metrics, entropy statistics, entropy change trends, cross-architecture comparisons, and data aggregation for machine learning analysis.

### Module Structure

#### Core Components

- **[DataLoader](../evaluation/data_loader.py)**: Loads experiment data, configurations, and results
- **[MetricsCalculator](../evaluation/metrics_calculator.py)**: Calculates performance metrics
- **[ExperimentAnalyzer](../evaluation/experiment_analyzer.py)**: Analyzes experiment results
- **[EntropyAnalyzer](../evaluation/entropy_analyzer.py)**: Analyzes entropy statistics and change trends
- **[Aggregator](../evaluation/aggregator.py)**: Aggregates metrics and entropy data into unified CSV format
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
| `--dataset`           | str  | gsm8k   | Dataset to analyze (gsm8k, humaneval, mmlu, aime2024, math500) |
| `--task-type`         | str  | math    | Task type (math, code, option)              |
| `--experiment`        | str  | None    | Specific experiment to analyze              |
| `--output`            | str  | None    | Output file path                            |
| `--compare`           | flag | False   | Compare experiments by architecture         |
| `--analyze-entropy`   | bool | True    | Perform entropy statistical analysis       |
| `--save-entropy-json` | bool | True    | Save entropy results to JSON                |
| `--aggregate`         | bool | True    | Aggregate results from metrics files        |
| `--analyze-trends`    | bool | True    | Analyze entropy change trends between agents across rounds |
| `--save-trends-json`  | bool | True    | Save detailed trend results to JSON file    |
| `--run-aggregator`    | bool | True    | Run results aggregator to combine metrics and entropy for data mining |
| `--aggregate-all`     | bool | False   | Aggregate results from all datasets        |

#### Examples

**Analyze a single experiment with entropy and trend analysis:**
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

**Analyze all datasets and aggregate results:**
```bash
python -m evaluation.evaluator --aggregate-all
```

**Analyze without running aggregator:**
```bash
python -m evaluation.evaluator --dataset gsm8k --run-aggregator False
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
- **sample_level**: Sample-level entropy statistics with the same metrics as agent_level but aggregated per sample
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
      "average_entropy_per_token": 0.062
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

#### Entropy Change Trends

**Intra-round Trends:**
- **trends**: Ranking of agents by entropy within each round
- **differences**: Pairwise differences between agents in the same round
- **summary**: Summary of intra-round patterns

**Inter-round Trends:**
- **agent_trends**: Per-agent entropy changes across rounds
- **round_to_round_changes**: Round-to-round entropy changes for each agent
- **summary**: Overall trend summary including:
  - total_change: Total change from first to last round
  - average_change: Average change across rounds
  - total_percentage_change: Total percentage change
  - average_percentage_change: Average percentage change
  - trend_direction: "increasing" or "decreasing"
  - volatility: Standard deviation of changes

**Trend Statistics:**
- **intra_round_stats**: Statistics on agent differences within rounds
- **inter_round_stats**: Statistics on round-to-round changes
- **overall_summary**: Overall summary of trend patterns across all agents

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
│   ├── aggregated_data.csv
│   └── entropy/
│       ├── {experiment_name}_entropy.json
│       ├── all_entropy_results.json
│       ├── {experiment_name}_trends.json
```

#### Key Output Files

**all_metrics.json**: Contains performance metrics for all experiments
- experiment_name, dataset, task_type, agent_architecture, num_rounds, num_samples
- samples: Detailed metrics for each sample including agent-level performance
  - main_id, ground_truth, final_predicted_answer, is_finally_correct, agents
  - agents: Per-agent metrics including agent_type, execution_order, time_cost, average_entropy, predicted_answer, is_correct
- summary: Aggregated statistics across all samples

**all_entropy_results.json**: Contains entropy analysis for all experiments
- dataset, architectures (mapping architecture type to experiment names)
- experiments: Per-experiment entropy analysis including:
  - macro_statistics: experiment_level, round_level, agent_level, architecture_comparison
  - micro_statistics: sample_level, sequence_level, token_position_level
  - trend_analysis: (optional) entropy change trends if analyzed

**aggregated_data.csv**: Unified CSV file combining metrics and entropy data for data mining
- sample_id, experiment_name, architecture, ground_truth
- agent_name, agent_key, execution_order, time_cost
- predicted_answer, is_correct, final_predicted_answer, is_finally_correct
- sample-level entropy statistics (total, max, min, mean, median, std, variance, q1, q3)
- sample token count and average entropy per token
- agent-level entropy statistics (total, sample_count, total_tokens, avg, mean, max, min, median, std, variance, q1, q3)
- experiment-level statistics (total_entropy, avg_entropy, total_samples, accuracy, total_time, avg_time)

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
        "sample_level": {
          "ID1": {
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
            "sample_count": 8,
            "average_entropy_per_token": 0.062
          },
          ...
        },
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
            "average_entropy_per_token": 0.062
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
      },
      "trend_analysis": {
        "experiment_name": "...",
        "dataset": "gsm8k",
        "agent_architecture": "centralized",
        "num_rounds": 2,
        "entropy_by_round_agent": {
          "1": {
            "MathAgent": {
              "mean_entropy": 0.051,
              "std_entropy": 0.195,
              "median_entropy": 0.0,
              "min_entropy": 0.0,
              "max_entropy": 1.123,
              "total_entropy": 12345.67,
              "sample_count": 100
            },
            ...
          },
          "2": {
            ...
          }
        },
        "intra_round_trends": {
          "1": {
            "trends": {
              "ranking": "MathAgent(0.051) -> ScienceAgent(0.062) -> CodeAgent(0.058) -> OrchestratorAgent(0.071)",
              "highest_entropy_agent": "OrchestratorAgent",
              "lowest_entropy_agent": "MathAgent",
              "entropy_range": 0.020
            },
            "differences": {
              "MathAgent_vs_ScienceAgent": {
                "absolute_difference": -0.011,
                "percentage_difference": -17.74,
                "agent1_entropy": 0.051,
                "agent2_entropy": 0.062,
                "agent1_std": 0.195,
                "agent2_std": 0.206
              },
              ...
            },
            "summary": ""
          },
          ...
        },
        "inter_round_trends": {
          "agent_trends": {
            "MathAgent": {
              "mean_entropies": [0.051, 0.042],
              "rounds": [1, 2],
              "changes": [-0.009],
              "percentage_changes": [-17.65]
            },
            ...
          },
          "round_to_round_changes": {
            "1_to_2": {
              "MathAgent": {
                "change": -0.009,
                "percentage_change": -17.65,
                "from_entropy": 0.051,
                "to_entropy": 0.042
              },
              ...
            },
            ...
          },
          "summary": {
            "MathAgent": {
              "total_change": -0.009,
              "average_change": -0.009,
              "total_percentage_change": -17.65,
              "average_percentage_change": -17.65,
              "trend_direction": "decreasing",
              "volatility": 0.0
            },
            ...
          }
        },
        "trend_statistics": {
          "intra_round_stats": {
            "mean_agent_difference": 0.015,
            "max_agent_difference": 0.025,
            "min_agent_difference": 0.005,
            "std_agent_difference": 0.008
          },
          "inter_round_stats": {
            "mean_round_to_round_change": -0.007,
            "max_round_to_round_change": 0.002,
            "min_round_to_round_change": -0.015,
            "std_round_to_round_change": 0.006
          },
          "overall_summary": {
            "total_agents_analyzed": 4,
            "agents_with_increasing_trend": 1,
            "agents_with_decreasing_trend": 3,
            "dominant_trend": "decreasing"
          }
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
- `agent_level`: Per-agent entropy statistics
- `architecture_comparison`: Cross-architecture comparison data

**Micro Statistics** now includes:
- `sample_level`: Per-sample entropy statistics (new)
- `sequence_level`: Per-sequence entropy statistics
- `token_position_level`: Per-token-position entropy statistics (enhanced with min, max, q1, q3)

**Trend Analysis** (optional, added when `--analyze-trends` is enabled):
- `entropy_by_round_agent`: Entropy values organized by round and agent
- `intra_round_trends`: Trends between agents within the same round
- `inter_round_trends`: Trends of individual agents across rounds
- `trend_statistics`: Overall trend statistics across all agents and rounds

### Programmatic Usage

#### Analyzing Experiments

```python
from evaluation.experiment_analyzer import ExperimentAnalyzer
from evaluation.entropy_analyzer import EntropyAnalyzer
from evaluation.aggregator import Aggregator

base_path = "/path/to/project"

# Analyze experiment metrics
analyzer = ExperimentAnalyzer(base_path)
results = analyzer.analyze_experiment(
    dataset="gsm8k",
    experiment_name="single_gsm8k",
    task_type="math"
)

# Analyze entropy statistics
entropy_analyzer = EntropyAnalyzer(base_path)
entropy_results = entropy_analyzer.analyze_experiment_entropy(
    dataset="gsm8k",
    experiment_name="single_gsm8k"
)

# Analyze entropy change trends
trend_results = entropy_analyzer.analyze_entropy_trends(
    dataset="gsm8k",
    experiment_name="single_gsm8k"
)

# Aggregate results for data mining
aggregator = Aggregator(base_path)
aggregator.aggregate_results(dataset="gsm8k")
```

#### Comparing Architectures

```python
from evaluation.experiment_analyzer import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(base_path)
comparison = analyzer.compare_architectures(
    dataset="gsm8k",
    task_type="math"
)
```

#### Loading Results

```python
from evaluation.data_loader import DataLoader

loader = DataLoader(base_path)

# Load experiment configuration
config = loader.load_experiment_config("gsm8k", "single_gsm8k")

# Load ground truth
ground_truths = loader.load_ground_truth("gsm8k")

# Load all results
results = loader.load_all_results("gsm8k", "single_gsm8k")

# Load result store info
info = loader.load_result_store_info("gsm8k", "single_gsm8k")

# Parse result ID
parsed = loader.parse_result_id("gsm8k_single_gsm8k_ID1_SingleSolver_1")
# Returns: {'dataset': 'gsm8k', 'experiment': 'single_gsm8k', 'main_id': 'ID1', 'agent_type': 'SingleSolver', 'execution_order': 1}
```

#### Calculating Metrics

```python
from evaluation.metrics_calculator import MetricsCalculator

# Check answer correctness
is_correct = MetricsCalculator.is_answer_correct(
    predicted="42",
    ground_truth="42"
)

# Normalize answer
normalized = MetricsCalculator.normalize_answer("The answer is 42.")
# Returns: "42"
```

### Data Loading

The [DataLoader](../evaluation/data_loader.py) class handles:

- Ground truth data from `experiments/data/{dataset}/train-all-samples.json`
- Experiment configs from `experiments/configs_exp/{experiment}.yml`
- Results from `experiments/results/raw/{dataset}/{experiment}/traces/`
- Entropy tensors from `traces/tensors/{result_id}_extras_entropy.pt`
- Result store info from `experiments/results/raw/{dataset}/{experiment}/result_store_info.json`

#### Key Methods

- `load_experiment_config(dataset, experiment_name)`: Load experiment configuration
- `load_ground_truth(dataset)`: Load ground truth answers
- `load_all_results(dataset, experiment_name)`: Load all experiment results
- `load_result_store_info(dataset, experiment_name)`: Load result store information
- `parse_result_id(result_id)`: Parse result ID into components

### Results Aggregator

The results aggregator provides functionality to aggregate metrics and entropy data from JSON files into unified CSV format for machine learning analysis.

#### Core Components

- **[Aggregator](../evaluation/aggregator.py)**: Main aggregation module

#### Usage

Run the aggregator from the project root:

```bash
python -m evaluation.evaluator --dataset gsm8k --run-aggregator True
```

Or aggregate all datasets:

```bash
python -m evaluation.evaluator --aggregate-all
```

#### Aggregation Process

The aggregator:

1. Loads metrics from `evaluation/results/{dataset}/all_metrics.json`
2. Loads entropy data from `evaluation/results/{dataset}/entropy/all_entropy_results.json`
3. Merges metrics and entropy data for each sample
4. Generates unified CSV file with comprehensive features

#### Output File

The aggregator generates a single CSV file in `evaluation/results/{dataset}/aggregated_data.csv`:

**Columns:**
- sample_id, experiment_name, architecture, ground_truth
- agent_name, agent_key, execution_order, time_cost
- predicted_answer, is_correct
- sample-level entropy statistics (total, max, min, mean, median, std, variance, q1, q3)
- sample token count and average entropy per token
- agent-level entropy statistics (total, sample_count, total_tokens, avg, mean, max, min, median, std, variance, q1, q3)
- experiment-level statistics (total_entropy, avg_entropy, total_samples, accuracy, total_time, avg_time)

#### CSV Structure

Each row represents a unique combination of sample and agent sequence, with the following structure:

```csv
sample_id,experiment_name,architecture,ground_truth,agent_name,agent_key,execution_order,time_cost,predicted_answer,is_correct,sample_entropy_total,sample_entropy_max,sample_entropy_min,sample_entropy_mean,sample_entropy_median,sample_entropy_std,sample_entropy_variance,sample_entropy_q1,sample_entropy_q3,sample_token_count,sample_avg_entropy_per_token,agent_entropy_total,agent_sample_count,agent_total_tokens,agent_entropy_avg,agent_entropy_mean,agent_entropy_max,agent_entropy_min,agent_entropy_median,agent_entropy_std,agent_entropy_variance,agent_entropy_q1,agent_entropy_q3,experiment_total_entropy,experiment_avg_entropy,experiment_total_samples,experiment_accuracy,experiment_total_time,experiment_avg_time
ID1,single_gsm8k,single,42,SingleSolver,ID1-SingleSolver-1,1,2.5,42,True,123.45,1.123,0.0,0.062,0.0,0.176,0.031,0.0,0.012,2000,0.062,12345.67,100,69094,0.051,0.051,1.123,0.0,0.0,0.195,0.028,0.0,0.005,37961.39,47.45,100,0.75,250.0,2.5
```

#### Examples

**Aggregate GSM8K results:**
```bash
python -m evaluation.evaluator --dataset gsm8k --run-aggregator True
```

**Aggregate all datasets:**
```bash
python -m evaluation.evaluator --aggregate-all
```

**Analyze without running aggregator:**
```bash
python -m evaluation.evaluator --dataset gsm8k --run-aggregator False
```
