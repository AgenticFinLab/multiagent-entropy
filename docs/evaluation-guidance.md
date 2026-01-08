## Evaluation Guidance

### Overview

The evaluation module provides comprehensive tools for analyzing multi-agent experiment results, including performance metrics, entropy statistics, entropy change trends, cross-architecture comparisons, and data aggregation for machine learning analysis.

### Module Structure

#### Core Components

- **[DataLoader](../evaluation/data_loader.py)**: Loads experiment data, configurations, and results
- **[MetricsCalculator](../evaluation/metrics_calculator.py)**: Calculates performance metrics
- **[ExperimentAnalyzer](../evaluation/experiment_analyzer.py)**: Analyzes experiment results
- **[EntropyStatistic](../evaluation/entropy_statistic.py)**: Analyzes entropy statistics and change trends
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
| `--dataset`           | str  | aime2024| Dataset to analyze (gsm8k, humaneval, mmlu, aime2024, math500) |
| `--model`             | str  | qwen3_4b| Model name (required when analyzing specific experiment) |
| `--task-type`         | str  | auto    | Task type (math, code, option, auto to infer from dataset) |
| `--experiment`        | str  | None    | Specific experiment to analyze (if not provided, analyze all) |
| `--output`            | str  | None    | Output file path (if not provided, save to evaluation/results/) |
| `--analyze-entropy`   | bool | True    | Perform entropy statistical analysis       |
| `--save-entropy-json` | bool | True    | Save detailed entropy results to JSON file |
| `--aggregate`         | bool | True    | Aggregate results from metrics files        |
| `--analyze-trends`    | bool | True    | Analyze entropy change trends between agents across rounds |
| `--save-trends-json`  | bool | True    | Save detailed trend results to JSON file    |
| `--run-aggregator`    | bool | True    | Run results aggregator to combine metrics and entropy for data mining |
| `--aggregate-all`     | bool | False   | Aggregate results from all datasets        |

#### Examples

**Analyze all experiments for a dataset:**
```bash
python -m evaluation.evaluator --dataset aime2024
```

**Analyze a specific experiment:**
```bash
python -m evaluation.evaluator --dataset aime2024 --model qwen3_4b --experiment single_agent
```

**Analyze with custom output path:**
```bash
python -m evaluation.evaluator --dataset humaneval --output results/custom.json
```

**Analyze all datasets and aggregate results:**
```bash
python -m evaluation.evaluator --aggregate-all
```

**Analyze without running aggregator:**
```bash
python -m evaluation.evaluator --dataset aime2024 --run-aggregator False
```

**Analyze with specific task type:**
```bash
python -m evaluation.evaluator --dataset mmlu --task-type option
```

### Metrics

#### Performance Metrics

The evaluation module calculates the following performance metrics:

- **Accuracy**: Percentage of correct answers across all samples
  - Calculated at both agent-level and experiment-level
  - Agent-level: Accuracy of individual agent predictions
  - Experiment-level: Final answer accuracy based on the final agent's output

- **Time Cost**: Execution time for each agent's inference
  - Measured in seconds
  - Aggregated at sample, agent, and experiment levels
  - Used to analyze computational efficiency

- **Average Entropy**: Mean entropy value across all tokens
  - Indicates model confidence and uncertainty
  - Lower entropy typically indicates higher confidence
  - Calculated per agent, per sample, and per experiment

- **Format Compliance Rate**: Percentage of valid answer formats
  - Checks for boxed format in math tasks: `\boxed{answer}`
  - Checks for code blocks in code tasks: ```python ... ```
  - Checks for single uppercase letter in option tasks

#### Entropy Statistics

The entropy analysis provides comprehensive statistics at multiple levels:

**Macro Statistics:**

1. **Experiment Level** (`experiment_level`)
   - `total_entropy`: Sum of all entropy values in the experiment
   - `infer_average_entropy`: Average entropy per inference
   - `num_inferences`: Total number of inferences performed

2. **Round Level** (`round_level`)
   - `total_entropy`: Sum of entropy values for each round
   - `num_inferences`: Number of inferences in each round
   - `infer_average_entropy`: Average entropy per inference in the round

3. **Agent Level** (`agent_level`)
   - `total_entropy`: Sum of all entropy values for the agent
   - `num_inferences`: Number of inferences by the agent
   - `total_tokens`: Total number of tokens generated
   - `infer_average_entropy`: Average entropy per inference
   - `mean_entropy`: Mean of all entropy values
   - `max_entropy`: Maximum entropy value
   - `min_entropy`: Minimum entropy value
   - `median_entropy`: Median entropy value
   - `std_entropy`: Standard deviation of entropy values
   - `variance_entropy`: Variance of entropy values
   - `q1_entropy`: First quartile (25th percentile)
   - `q3_entropy`: Third quartile (75th percentile)

**Micro Statistics:**

1. **Sample Level** (`sample_level`)
   - Same metrics as agent-level but aggregated per sample
   - `total_entropy`: Total entropy for the sample
   - `max_entropy`, `min_entropy`, `mean_entropy`, `median_entropy`
   - `std_entropy`, `variance_entropy`, `q1_entropy`, `q3_entropy`
   - `all_agents_token_count`: Total tokens across all agents
   - `num_agents`: Number of agents in the sample
   - `average_entropy_per_token`: Average entropy per token
   - `agents`: Dictionary of agent-level statistics per agent

2. **Sequence Level** (`sequence_level`)
   - Composite keys in format: `{main_id}-{agent_type}-{execution_order}`
   - Each key uniquely identifies a specific execution sequence
   - Metrics include:
     - `total_entropy`: Sum of entropy values for the sequence
     - `max_entropy`, `min_entropy`, `mean_entropy`, `median_entropy`
     - `variance_entropy`, `q1_entropy`, `q3_entropy`, `std_entropy`
     - `token_count`: Total number of tokens
     - `sample_count`: Number of samples in the sequence
     - `average_entropy_per_token`: Average entropy per token

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
     }
   }
   ```

   The total number of sequence entries follows: 100 (samples) × number of agents × number of rounds

3. **Token Position Level** (`token_position_level`)
   - Analysis of entropy distribution across token positions
   - Metrics for each position:
     - `mean`: Mean entropy at the position
     - `std`: Standard deviation at the position
     - `median`: Median entropy at the position
     - `min`: Minimum entropy at the position
     - `max`: Maximum entropy at the position
     - `q1`: First quartile at the position
     - `q3`: Third quartile at the position
     - `count`: Number of samples at the position

#### Entropy Change Trends

The trend analysis provides insights into entropy dynamics:

**Intra-round Trends** (`intra_round_trends`)
- **trends**: Ranking of agents by entropy within each round
  - `ranking`: Ordered list of agents with their entropy values
  - `highest_entropy_agent`: Agent with highest entropy
  - `lowest_entropy_agent`: Agent with lowest entropy
  - `entropy_range`: Difference between highest and lowest entropy

- **differences**: Pairwise differences between agents in the same round
  - `absolute_difference`: Absolute difference in mean entropy
  - `percentage_difference`: Percentage difference
  - `agent1_entropy`, `agent2_entropy`: Individual agent entropies
  - `agent1_std`, `agent2_std`: Standard deviations

- **summary**: Summary of intra-round patterns

**Inter-round Trends** (`inter_round_trends`)
- **agent_trends**: Per-agent entropy changes across rounds
  - `mean_entropies`: List of mean entropy values per round
  - `rounds`: List of round numbers
  - `changes`: List of changes between consecutive rounds
  - `percentage_changes`: List of percentage changes

- **round_to_round_changes**: Round-to-round entropy changes for each agent
  - `change`: Absolute change between rounds
  - `percentage_change`: Percentage change between rounds
  - `from_entropy`: Entropy in the starting round
  - `to_entropy`: Entropy in the ending round

- **summary**: Overall trend summary per agent
  - `total_change`: Total change from first to last round
  - `average_change`: Average change across rounds
  - `total_percentage_change`: Total percentage change
  - `average_percentage_change`: Average percentage change
  - `trend_direction`: "increasing" or "decreasing"
  - `volatility`: Standard deviation of changes

**Trend Statistics** (`trend_statistics`)
- **intra_round_stats**: Statistics on agent differences within rounds
  - `mean_agent_difference`: Mean of absolute differences
  - `max_agent_difference`: Maximum absolute difference
  - `min_agent_difference`: Minimum absolute difference
  - `std_agent_difference`: Standard deviation of differences

- **inter_round_stats**: Statistics on round-to-round changes
  - `mean_round_to_round_change`: Mean of round-to-round changes
  - `max_round_to_round_change`: Maximum round-to-round change
  - `min_round_to_round_change`: Minimum round-to-round change
  - `std_round_to_round_change`: Standard deviation of changes

- **overall_summary**: Overall summary of trend patterns
  - `total_agents_analyzed`: Total number of agents
  - `agents_with_increasing_trend`: Number of agents with increasing entropy
  - `agents_with_decreasing_trend`: Number of agents with decreasing entropy
  - `dominant_trend`: Overall dominant trend direction

### Agent Architectures

Supported architectures:
- **single**: Single solver agent
- **sequential**: Sequential multi-agent system
- **centralized**: Centralized orchestration
- **debate**: Debate-based multi-agent system
- **hybrid**: Hybrid multi-agent system

### Output Structure

Results are saved to `evaluation/results/{dataset}/` with a hierarchical organization:

```
evaluation/results/
├── {dataset}/                          # Dataset-level directory
│   ├── all_metrics.json              # All models' performance metrics for this dataset
│   ├── all_entropy_results.json      # All models' entropy analysis for this dataset
│   ├── all_aggregated_data.csv       # Combined aggregated data for all models
│   └── {model_name}/                 # Model-level directory
│       ├── aggregated_data.csv       # Model-level aggregated data (all experiments)
│       ├── aggregated_data_{model}_{dataset}_{architecture}_{timestamp}.csv  # Single experiment aggregated data
│       ├── {experiment_name}_metrics.json
│       ├── {experiment_name}_entropy.json
│       └── {experiment_name}_trends.json
```

**Example Structure:**

```
evaluation/results/
├── aime2024/
│   ├── all_metrics.json
│   ├── all_entropy_results.json
│   ├── all_aggregated_data.csv
│   └── qwen3_4b/
│       ├── aggregated_data.csv
│       ├── aggregated_data_qwen3-4b_aime2024_single_agent_20260106_061544_711_3447086.csv
│       ├── aggregated_data_qwen3-4b_aime2024_sequential_agent_20260107_085953_585_481922.csv
│       ├── aggregated_data_qwen3-4b_aime2024_hybrid_agent_20260106_044602_425_3328646.csv
│       ├── aggregated_data_qwen3-4b_aime2024_debate_agent_20260106_123550_587_3941124.csv
│       ├── aggregated_data_qwen3-4b_aime2024_centralized_agent_20260107_144400_708_785617.csv
│       └── ...
├── gsm8k/
│   ├── all_metrics.json
│   ├── all_entropy_results.json
│   ├── all_aggregated_data.csv
│   └── qwen3_4b/
│       ├── aggregated_data.csv
│       └── aggregated_data_qwen3-4b_gsm8k_single_agent_20260106_061544_711_3447086.csv
```

### Hierarchical Result Storage Structure

The evaluation system now supports a hierarchical result storage structure that enables multi-level analysis across datasets, models, and experiments. This structure provides better organization and facilitates comprehensive analysis at different granularities.

#### Directory Structure

```
evaluation/results/
└── {dataset}/                          # Dataset-level directory
    ├── all_metrics.json              # All models' performance metrics for this dataset
    ├── all_entropy_results.json      # All models' entropy analysis for this dataset
    ├── all_aggregated_data.csv       # Combined aggregated data for all models
    └── {model_name}/                 # Model-level directory
        ├── aggregated_data.csv       # Model-level aggregated data (all experiments)
        └── aggregated_data_{model}_{dataset}_{architecture}_{timestamp}.csv  # Single experiment aggregated data
```

**Example Structure:**

```
evaluation/results/
├── aime2024/
│   ├── all_metrics.json
│   ├── all_entropy_results.json
│   ├── all_aggregated_data.csv
│   └── qwen3_4b/
│       ├── aggregated_data.csv
│       ├── aggregated_data_qwen3-4b_aime2024_single_agent_20260106_061544_711_3447086.csv
│       ├── aggregated_data_qwen3-4b_aime2024_sequential_agent_20260107_085953_585_481922.csv
│       ├── aggregated_data_qwen3-4b_aime2024_hybrid_agent_20260106_044602_425_3328646.csv
│       ├── aggregated_data_qwen3-4b_aime2024_debate_agent_20260106_123550_587_3941124.csv
│       └── aggregated_data_qwen3-4b_aime2024_centralized_agent_20260107_144400_708_785617.csv
├── gsm8k/
│   ├── all_metrics.json
│   ├── all_entropy_results.json
│   ├── all_aggregated_data.csv
│   └── qwen3_4b/
│       ├── aggregated_data.csv
│       └── aggregated_data_qwen3-4b_gsm8k_single_agent_20260106_061544_711_3447086.csv
```

#### Organization Logic

**1. Dataset Level**
- **Directory**: `evaluation/results/{dataset}/`
- **Purpose**: Contains all experimental results for a specific dataset
- **Contents**:
  - `all_metrics.json`: Aggregated performance metrics for all models and experiments on this dataset
  - `all_entropy_results.json`: Aggregated entropy analysis for all models and experiments on this dataset
  - `all_aggregated_data.csv`: Combined CSV data for all models on this dataset
- **Analysis Scope**: Enables cross-model comparison and dataset-level performance analysis

**2. Model Level**
- **Directory**: `evaluation/results/{dataset}/{model_name}/`
- **Purpose**: Contains all experimental results for a specific model on a specific dataset
- **Contents**:
  - `aggregated_data.csv`: Model-level aggregated data combining all experiments for this model
  - Individual experiment CSV files (one per experiment/agent architecture)
- **Analysis Scope**: Enables comparison of different agent architectures within the same model
- **Naming Convention**: Model names follow the format `{model}_{size}` (e.g., `qwen3_4b`, `gpt4_32k`)

**3. Experiment Level**
- **Files**: `aggregated_data_{model}_{dataset}_{architecture}_{timestamp}.csv`
- **Purpose**: Contains detailed results for a single experiment with a specific agent architecture
- **Analysis Scope**: Enables fine-grained analysis of individual experiments and agent architectures
- **Naming Convention**: `{aggregated_data}_{model}_{dataset}_{architecture}_{timestamp}.csv`
  - `model`: Model identifier (e.g., `qwen3-4b`)
  - `dataset`: Dataset name (e.g., `aime2024`)
  - `architecture`: Agent architecture type (e.g., `single_agent`, `centralized_agent`, `sequential_agent`, `debate_agent`, `hybrid_agent`)
  - `timestamp`: Unique timestamp identifier (e.g., `20260107_144400_708_785617`)

#### File Naming Conventions

**Dataset-Level Files:**
- `all_metrics.json`: All models' performance metrics
- `all_entropy_results.json`: All models' entropy analysis results
- `all_aggregated_data.csv`: Combined data for all models

**Model-Level Files:**
- `aggregated_data.csv`: Model-level aggregated data (all experiments combined)

**Experiment-Level Files:**
- `aggregated_data_{model}_{dataset}_{architecture}_{timestamp}.csv`: Single experiment data
  - Example: `aggregated_data_qwen3-4b_aime2024_centralized_agent_20260107_144400_708_785617.csv`

#### File Content Specifications

**Dataset-Level JSON Files:**

`all_metrics.json` structure:
```json
{
  "dataset": "aime2024",
  "task_type": "math",
  "models": {
    "qwen3_4b": {
      "experiments": {
        "experiment_name_1": {
          "experiment_name": "...",
          "dataset": "aime2024",
          "model_name": "qwen3_4b",
          "agent_architecture": "centralized",
          "num_rounds": 2,
          "num_samples": 30,
          "samples": { ... },
          "summary": { ... }
        }
      }
    }
  }
}
```

`all_entropy_results.json` structure:
```json
{
  "dataset": "aime2024",
  "models": {
    "qwen3_4b": {
      "experiments": {
        "experiment_name_1": {
          "experiment_name": "...",
          "dataset": "aime2024",
          "model_name": "qwen3_4b",
          "agent_architecture": "centralized",
          "num_rounds": 2,
          "num_samples": 30,
          "macro_statistics": { ... },
          "micro_statistics": { ... },
          "trend_analysis": { ... }
        }
      }
    }
  }
}
```

**CSV File Structure:**

All CSV files (dataset-level, model-level, and experiment-level) share the same column structure:

```csv
model_name,sample_id,experiment_name,architecture,num_rounds,ground_truth,agent_name,agent_key,execution_order,time_cost,final_predicted_answer,is_finally_correct,sample_total_entropy,sample_max_entropy,sample_min_entropy,sample_mean_entropy,sample_median_entropy,sample_std_entropy,sample_variance_entropy,sample_q1_entropy,sample_q3_entropy,sample_num_agents,sample_all_agents_token_count,sample_avg_entropy_per_token,agent_total_entropy,agent_max_entropy,agent_min_entropy,agent_mean_entropy,agent_median_entropy,agent_std_entropy,agent_variance_entropy,agent_q1_entropy,agent_q3_entropy,agent_token_count,agent_avg_entropy_per_token,agent_round_number,agent_avg_entropy,round_total_entropy,round_num_inferences,round_avg_entropy,exp_total_entropy,exp_infer_average_entropy,exp_num_inferences,exp_accuracy,exp_total_time
```

**Column Groups:**
- **Metadata**: `model_name`, `sample_id`, `experiment_name`, `architecture`, `num_rounds`, `ground_truth`
- **Agent Information**: `agent_name`, `agent_key`, `execution_order`, `time_cost`
- **Performance**: `final_predicted_answer`, `is_finally_correct`
- **Sample-level Entropy**: `sample_total_entropy`, `sample_max_entropy`, `sample_min_entropy`, `sample_mean_entropy`, `sample_median_entropy`, `sample_std_entropy`, `sample_variance_entropy`, `sample_q1_entropy`, `sample_q3_entropy`, `sample_num_agents`, `sample_all_agents_token_count`, `sample_avg_entropy_per_token`
- **Agent-level Entropy**: `agent_total_entropy`, `agent_max_entropy`, `agent_min_entropy`, `agent_mean_entropy`, `agent_median_entropy`, `agent_std_entropy`, `agent_variance_entropy`, `agent_q1_entropy`, `agent_q3_entropy`, `agent_token_count`, `agent_avg_entropy_per_token`, `agent_round_number`, `agent_avg_entropy`
- **Round-level Entropy**: `round_total_entropy`, `round_num_inferences`, `round_avg_entropy`
- **Experiment-level Statistics**: `exp_total_entropy`, `exp_infer_average_entropy`, `exp_num_inferences`, `exp_accuracy`, `exp_total_time`

#### Multi-Level Analysis Capabilities

This hierarchical structure enables analysis at multiple levels:

1. **Dataset-Level Analysis**: Compare performance across different models on the same dataset
2. **Model-Level Analysis**: Compare different agent architectures within the same model
3. **Experiment-Level Analysis**: Analyze detailed results of individual experiments

The structure supports:
- Cross-dataset comparisons
- Cross-model comparisons
- Architecture-specific analysis
- Fine-grained experiment analysis
- Aggregated statistics at any level

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


### Data Loading

The [DataLoader](../evaluation/data_loader.py) class handles:

- Ground truth data from `experiments/data/{dataset}/train-all-samples.json`
- Experiment configs from `experiments/configs_exp/{experiment}.yml`
- Results from `experiments/results/raw/{dataset}/{experiment}/traces/`
- Entropy tensors from `traces/tensors/{result_id}_extras_entropy.pt`
- Result store info from `experiments/results/raw/{dataset}/{experiment}/result_store_info.json`

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
2. Loads entropy data from `evaluation/results/{dataset}/all_entropy_results.json`
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
ID1,single_agent,single,42,SingleSolver,ID1-SingleSolver-1,1,2.5,42,True,123.45,1.123,0.0,0.062,0.0,0.176,0.031,0.0,0.012,2000,0.062,12345.67,100,69094,0.051,0.051,1.123,0.0,0.0,0.195,0.028,0.0,0.005,37961.39,47.45,100,0.75,250.0,2.5
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
