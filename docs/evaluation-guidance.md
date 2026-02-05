## Evaluation Guidance

### Overview

The evaluation module provides comprehensive tools for analyzing multi-agent experiment results, including performance metrics, entropy statistics, entropy change trends, cross-architecture comparisons, and data aggregation for machine learning analysis.

### Module Structure

#### Core Components

- **[DataLoader](../evaluation/data_loader.py)**: Loads experiment data, configurations, and results
- **[MetricsCalculator](../evaluation/metrics_calculator.py)**: Calculates performance metrics
- **[ExperimentAnalyzer](../evaluation/experiment_analyzer.py)**: Analyzes experiment results
- **[EntropyStatistic](../evaluation/entropy_statistic.py)**: Analyzes entropy statistics and change trends
- **[FeatureEnhancer](../evaluation/feature_enhancer.py)**: Applies advanced feature engineering on aggregated metrics and entropy data (sample-level entropy features, cross-round dynamics, intra-round agent distribution, etc.)
- **[Aggregator](../evaluation/aggregator.py)**: Loads JSON results, performs basic data aggregation, and delegates advanced feature computation to `FeatureEnhancer` before writing unified CSVs
- **[evaluator.py](../evaluation/evaluator.py)**: Main entry point for evaluation
- **[utils.py](../evaluation/utils.py)**: Utility functions for saving results

### Usage

#### Command Line Interface

Run evaluation from the project root:

```bash
python -m evaluation.evaluator --dataset aime2025 --task-type math
```

#### Available Arguments

| Argument              | Type | Default | Description                                 |
| --------------------- | ---- | ------- | ------------------------------------------- |
| `--datasets`          | str  | All datasets | Datasets to analyze (space-separated list). Choices: gsm8k, humaneval, mmlu, math500, aime2024_16384, aime2025_16384 |
| `--all-datasets`      | flag | False   | Analyze all available datasets             |
| `--model`             | str  | qwen3_0_6b, qwen3_4b, qwen3_8b | Model names (space-separated list). If not provided, analyze all models |
| `--task-type`         | str  | auto    | Task type (math, code, option, auto to infer from dataset) |
| `--experiment`        | str  | None    | Specific experiment to analyze (if not provided, analyze all) |
| `--output`            | str  | None    | Output file path (if not provided, save to evaluation/results/) |
| `--run-aggregator`    | bool | True    | Run results aggregator to combine metrics and entropy for data mining |
| `--aggregate-all`     | bool | False   | Aggregate results from all datasets        |
| `--generate-summary`  | bool | True    | Generate summary CSV from aggregated data  |
| `--timeout`           | int  | 10      | Maximum time in seconds to execute code for code tasks |

#### Supported Datasets

The evaluation module supports the following datasets defined in [evaluator.py](../evaluation/evaluator.py):

| Dataset | Description |
|---------|-------------|
| `gsm8k` | GSM8K math dataset |
| `humaneval` | HumanEval code generation dataset |
| `mmlu` | Massive Multitask Language Understanding dataset |
| `math500` | MATH-500 dataset |
| `aime2024_16384` | AIME 2024 competition problems (16384 token context) |
| `aime2025_16384` | AIME 2025 competition problems (16384 token context) |

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

#### Base Model Metrics

The evaluation module also calculates base model performance metrics for comparison:

Base model is defined as the first round agent in single agent architecture. These metrics provide a baseline to compare multi-agent system performance against single-agent performance.

- **base_model_predicted_answer**: The answer predicted by the base model
  - Extracted from the first round agent in single agent architecture
  - Used for comparison with multi-agent final answers
  - Shared across all architectures for the same dataset and model

- **base_model_is_finally_correct**: Whether the base model's prediction is correct
  - Boolean value indicating correctness
  - Compared against the ground truth answer
  - Used to calculate base model accuracy

- **base_model_format_compliance**: Whether the base model's answer follows the required format
  - Boolean value indicating format compliance
  - Checks for task-specific format requirements
  - Used to calculate base model format compliance rate

- **base_model_accuracy**: Overall accuracy of the base model across all samples
  - Calculated as: (number of correct predictions) / (total predictions)
  - Provides baseline performance metric
  - Used to compare multi-agent system improvements

- **base_model_format_compliance_rate**: Percentage of base model answers with valid format
  - Calculated as: (number of format-compliant answers) / (total predictions)
  - Indicates base model's ability to follow output format requirements
  - Used for comparison with multi-agent format compliance

These base model metrics are automatically calculated and included in the aggregated data CSV files, enabling direct comparison between single-agent (base model) and multi-agent system performance.

#### Entropy Statistics

The entropy analysis provides comprehensive statistics at multiple levels:

**Macro Statistics:**

1. **Experiment Level** (`experiment_level`)
   - `total_entropy`: Sum of all entropy values in the experiment
   - `infer_average_entropy`: Average entropy per inference
   - `num_inferences`: Total number of inferences performed
   - `total_time`: Total execution time for the entire experiment (sum of all agent time costs)
   - `total_token`: Total number of tokens generated across all agents in the experiment

2. **Round Level** (`round_level`)
   - `total_entropy`: Sum of entropy values for each round
   - `num_inferences`: Number of inferences in each round
   - `infer_average_entropy`: Average entropy per inference in the round
   - `round_infer_average_entropy`: Average entropy per inference in the round (same as infer_average_entropy)
   - `total_time`: Total execution time for the round (sum of all agent time costs in the round)
   - `total_token`: Total number of tokens generated by all agents in the round

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
   - `sample_entropy_stability_index`: Stability index calculated as 1 - (std_entropy / mean_entropy), indicates consistency of entropy values
   - `sample_entropy_range`, `sample_entropy_iqr`: Range and interquartile range of entropy values
   - `sample_entropy_relative_iqr_mean`, `sample_entropy_relative_iqr_range`: Relative IQR w.r.t. mean entropy and overall range
   - `sample_entropy_bowley_skewness`: Quantile-based skewness of the entropy distribution, indicating whether high-entropy or low-entropy tails dominate
   - `sample_entropy_median_over_mean`: Ratio between median and mean entropy, measures how much the mean is affected by high-entropy outliers
   - `sample_entropy_tail_weight`: Relative weight of the high-entropy tail compared to the main body (IQR)
   - `sample_entropy_cv`: Coefficient of variation of entropy values (std / mean)
   - `agent_entropy_contribution`: Proportion of total entropy contributed by the current agent, calculated as agent_total_entropy / sample_total_entropy
   - `sample_avg_entropy_per_agent`: Average entropy per agent in the sample, calculated as sample_total_entropy / sample_num_agents
   - `base_sample_total_entropy`, `base_sample_token_count`, `base_sample_avg_entropy_per_token`: Entropy statistics of the single-agent baseline (first-round agent in the single architecture) for the same model and sample
   - `sample_entropy_ratio_vs_base_total`, `sample_entropy_reduction_vs_base_total`: Relative and absolute change of total entropy compared to the single-agent baseline
   - `sample_avg_entropy_per_token_ratio_vs_base`, `sample_avg_entropy_per_token_diff_vs_base`: Relative and absolute change of per-token entropy compared to the baseline
   - Round-wise sample features (aggregated over all agents in the sample):
     - `sample_round_{r}_all_agents_total_entropy`: Total entropy of all agents in round r
     - `sample_round_{r}_all_agents_total_token`: Total tokens of all agents in round r
     - `sample_round_{r}_all_agents_entropy_per_token`: Entropy per token in round r
   - Cross-round sample dynamics features:
     - `sample_round_all_agents_total_entropy_first_last_diff`, `sample_round_all_agents_total_entropy_first_last_ratio`: Change and relative change of total entropy between first and last rounds
     - `sample_round_all_agents_total_token_first_last_diff`, `sample_round_all_agents_total_token_first_last_ratio`: Change and relative change of total tokens between first and last rounds
     - `sample_round_all_agents_entropy_per_token_first_last_diff`, `sample_round_all_agents_entropy_per_token_first_last_ratio`: Change and relative change of per-token entropy between first and last rounds
     - `sample_round_all_agents_entropy_per_token_slope_per_round`, `sample_round_all_agents_entropy_per_token_volatility`: Time-series features describing per-round entropy-per-token trend and volatility
     - `sample_round_1_2_change_tokens`, `sample_round_1_2_change_entropy`: Sample-level token and entropy change from round 1 to round 2 (when both rounds exist)
   - Cross-round features based on aggregated agent entropy:
     - `sample_round_mean_agent_mean_entropy_first_last_diff`, `sample_round_mean_agent_mean_entropy_first_last_ratio`: Change and relative change of per-round mean agent entropy between first and last rounds
     - `sample_round_mean_agent_mean_entropy_slope_per_round`, `sample_round_mean_agent_mean_entropy_volatility`, `sample_round_mean_agent_mean_entropy_trend_sign`: Trend and volatility of mean agent entropy across rounds
     - `sample_round_mean_agent_total_entropy_first_last_diff`, `sample_round_mean_agent_total_entropy_first_last_ratio`: Change and relative change of per-round mean agent total entropy
     - `sample_round_mean_agent_total_entropy_slope_per_round`, `sample_round_mean_agent_total_entropy_volatility`: Trend and volatility of mean agent total entropy across rounds
   - Intra-round agent distribution features:
     - `sample_round_{r}_agent_mean_entropy_spread`, `sample_round_{r}_agent_mean_entropy_cv`, `sample_round_{r}_agent_mean_entropy_bowley_skewness`: Spread, coefficient of variation, and skewness of agents' mean entropy within round r
     - `sample_round_{r}_agent_total_entropy_spread`: Spread of agents' total entropy within round r
     - `sample_round_agent_mean_entropy_spread_first_last_diff`, `sample_round_agent_total_entropy_spread_first_last_diff`: Change of agent-level entropy spread between first and last rounds
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

**Dynamic Round Comparison Features**

The aggregator generates dynamic features that compare different rounds within an experiment:

- **round_{x}_{y}_change_tokens**: Change in total token count from round x to round y
  - Calculated as: `round_y_total_token - round_x_total_token`
  - Positive values indicate increased token generation in later rounds
  - Useful for analyzing how discussion length evolves across rounds
  - Example: `round_1_2_change_tokens` shows token change from round 1 to round 2

- **round_{x}_{y}_change_entropy**: Change in total entropy from round x to round y
  - Calculated as: `round_y_total_entropy - round_x_total_entropy`
  - Positive values indicate increased entropy (uncertainty) in later rounds
  - Useful for analyzing how model confidence changes across rounds
  - Example: `round_1_2_change_entropy` shows entropy change from round 1 to round 2

These features are generated for all valid round pairs within each experiment, enabling comprehensive analysis of how the multi-agent system evolves over multiple rounds of interaction.

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

The evaluation module supports all seven multi-agent system architectures:

| Architecture | Type | Description |
|--------------|------|-------------|
| **single** | Single-agent | Single solver agent baseline |
| **sequential** | Multi-agent | Sequential multi-agent system with pipeline processing |
| **centralized** | Multi-agent | Centralized orchestration with domain agents and central coordinator |
| **decentralized** | Multi-agent | Decentralized architecture with loopback mechanism |
| **full_decentralized** | Multi-agent | Fully decentralized where each agent can communicate with all others |
| **debate** | Multi-agent | Debate-based multi-agent system with majority voting |
| **hybrid** | Multi-agent | Hybrid multi-agent system with enhanced context sharing |

**Special Handling:**
- **Code tasks** (`task_type == "code"`): Debate architecture is automatically excluded from aggregation since it doesn't support code generation
- **Debate architecture**: Orchestrator agent data is excluded from entropy analysis (uses voting mechanism, not LLM)

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
├── aime2025/
│   ├── all_metrics.json
│   ├── all_entropy_results.json
│   ├── all_aggregated_data.csv
│   └── qwen3_4b/
│       ├── aggregated_data.csv
│       └── aggregated_data_qwen3-4b_aime2025_single_agent_20260106_061544_711_3447086.csv
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
├── aime2025/
│   ├── all_metrics.json
│   ├── all_entropy_results.json
│   ├── all_aggregated_data.csv
│   └── qwen3_4b/
│       ├── aggregated_data.csv
│       └── aggregated_data_qwen3-4b_aime2025_single_agent_20260106_061544_711_3447086.csv
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
model_name,sample_id,experiment_name,architecture,num_rounds,ground_truth,agent_name,agent_key,execution_order,agent_time_cost,final_predicted_answer,is_finally_correct,final_format_compliance,base_model_predicted_answer,base_model_is_finally_correct,base_model_format_compliance,sample_total_entropy,sample_max_entropy,sample_min_entropy,sample_mean_entropy,sample_median_entropy,sample_std_entropy,sample_variance_entropy,sample_q1_entropy,sample_q3_entropy,sample_num_agents,sample_all_agents_token_count,sample_avg_entropy_per_token,agent_total_entropy,agent_max_entropy,agent_min_entropy,agent_mean_entropy,agent_median_entropy,agent_std_entropy,agent_variance_entropy,agent_q1_entropy,agent_q3_entropy,agent_token_count,agent_avg_entropy_per_token,agent_round_number,agent_avg_entropy,round_total_entropy,round_num_inferences,round_avg_entropy,round_total_time,round_total_token,exp_total_entropy,exp_infer_average_entropy,exp_num_inferences,exp_accuracy,exp_format_compliance_rate,exp_total_time,exp_total_token,base_model_accuracy,base_model_format_compliance_rate
```

**Column Groups:**
- **Metadata**: `model_name`, `sample_id`, `experiment_name`, `architecture`, `num_rounds`, `ground_truth`
- **Agent Information**: `agent_name`, `agent_key`, `execution_order`, `agent_time_cost`
- **Performance**: `final_predicted_answer`, `is_finally_correct`, `final_format_compliance`
- **Sample-level Entropy**: `sample_total_entropy`, `sample_max_entropy`, `sample_min_entropy`, `sample_mean_entropy`, `sample_median_entropy`, `sample_std_entropy`, `sample_variance_entropy`, `sample_q1_entropy`, `sample_q3_entropy`, `sample_num_agents`, `sample_all_agents_token_count`, `sample_avg_entropy_per_token`
- **Agent-level Entropy**: `agent_total_entropy`, `agent_max_entropy`, `agent_min_entropy`, `agent_mean_entropy`, `agent_median_entropy`, `agent_std_entropy`, `agent_variance_entropy`, `agent_q1_entropy`, `agent_q3_entropy`, `agent_token_count`, `agent_avg_entropy_per_token`, `agent_round_number`, `agent_avg_entropy`
- **Round-level Entropy**: `round_total_entropy`, `round_num_inferences`, `round_avg_entropy`, `round_total_time`, `round_total_token`
- **Experiment-level Statistics**: `exp_total_entropy`, `exp_infer_average_entropy`, `exp_num_inferences`, `exp_accuracy`, `exp_format_compliance_rate`, `exp_total_time`, `exp_total_token`
- **Base Model Metrics**: `base_model_predicted_answer`, `base_model_is_finally_correct`, `base_model_format_compliance`, `base_model_accuracy`, `base_model_format_compliance_rate`

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
  - agents: Per-agent metrics including agent_type, execution_order, agent_time_cost, average_entropy, predicted_answer, is_correct
- summary: Aggregated statistics across all samples

**all_entropy_results.json**: Contains entropy analysis for all experiments
- dataset, architectures (mapping architecture type to experiment names)
- experiments: Per-experiment entropy analysis including:
  - macro_statistics: experiment_level, round_level, agent_level, architecture_comparison
  - micro_statistics: sample_level, sequence_level, token_position_level
  - trend_analysis: (optional) entropy change trends if analyzed

**aggregated_data.csv**: Unified CSV file combining metrics and entropy data for data mining
- sample_id, experiment_name, architecture, ground_truth
- agent_name, agent_key, execution_order, agent_time_cost
- final_predicted_answer, is_finally_correct, final_format_compliance
- sample-level entropy statistics (total, max, min, mean, median, std, variance, q1, q3)
- sample token count and average entropy per token
- agent-level entropy statistics (total, max, min, mean, median, std, variance, q1, q3, token_count, avg_entropy_per_token, round_number, avg_entropy)
- round-level statistics (total_entropy, num_inferences, avg_entropy, total_time, total_token)
- experiment-level statistics (total_entropy, infer_average_entropy, num_inferences, accuracy, format_compliance_rate, total_time, total_token)
- base model metrics (base_model_predicted_answer, base_model_is_finally_correct, base_model_format_compliance, base_model_accuracy, base_model_format_compliance_rate)

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
- **[FeatureEnhancer](../evaluation/feature_enhancer.py)**: Advanced feature engineering for entropy-related features

#### Aggregator Class Methods

| Method | Description |
|--------|-------------|
| `load_json_files()` | Load entropy and metrics JSON files |
| `extract_sample_level_data()` | Extract sample-level data using FeatureEnhancer |
| `extract_round_level_data()` | Extract round-level statistics with task type filtering |
| `extract_agent_level_data()` | Extract agent-level statistics |
| `extract_experiment_level_data()` | Extract experiment-level statistics with base model comparison |
| `merge_all_data()` | Merge all levels of data into comprehensive records |
| `add_dynamic_round_features()` | Add dynamic round comparison features (delegates to FeatureEnhancer) |
| `generate_exclude_agent_csv()` | Generate CSV excluding agent-specific columns, merging by sample |
| `generate_aggregated_csvs()` | Main method to generate all aggregated CSV files |

#### FeatureEnhancer Class Methods

| Method | Description |
|--------|-------------|
| `build_sample_records()` | Build enriched sample-level records with advanced entropy features |
| `add_dynamic_round_features()` | Add experiment-level dynamic round comparison features |
| `_extract_base_model_data()` | Extract base model data from single agent architecture |
| `_extract_base_model_entropy()` | Extract base model entropy statistics |

#### Usage

Run the aggregator from the project root:

```bash
python -m evaluation.evaluator --dataset gsm8k --run-aggregator
```

Or aggregate all datasets:

```bash
python -m evaluation.evaluator --aggregate-all
```

#### Aggregation Process

The aggregator:

1. Loads metrics from `evaluation/results/{dataset}/all_metrics.json`
2. Loads entropy data from `evaluation/results/{dataset}/all_entropy_results.json`
3. Extracts base model data from single agent architecture experiments:
   - Base model is defined as the first round agent in single agent architecture
   - Extracts predicted answer, correctness, and format compliance
   - Calculates base model accuracy and format compliance rate
   - Base model data is shared across all architectures for the same dataset and model
4. Delegates sample-level feature enhancement to `FeatureEnhancer.build_sample_records()`
5. Merges metrics and entropy data at multiple levels (sample, round, agent, experiment)
6. Adds dynamic round comparison features
7. Generates unified CSV files with comprehensive features

#### Output File

The aggregator generates a single CSV file in `evaluation/results/{dataset}/aggregated_data.csv`:

**Columns:**
- sample_id, experiment_name, architecture, ground_truth
- agent_name, agent_key, execution_order, agent_time_cost
- final_predicted_answer, is_finally_correct, final_format_compliance
- sample-level entropy statistics (total, max, min, mean, median, std, variance, q1, q3)
- sample token count and average entropy per token
- agent-level entropy statistics (total, max, min, mean, median, std, variance, q1, q3, token_count, avg_entropy_per_token, round_number, avg_entropy)
- round-level statistics (total_entropy, num_inferences, avg_entropy, total_time, total_token)
- experiment-level statistics (total_entropy, infer_average_entropy, num_inferences, accuracy, format_compliance_rate, total_time, total_token)
- base model metrics (base_model_predicted_answer, base_model_is_finally_correct, base_model_format_compliance, base_model_accuracy, base_model_format_compliance_rate)

#### CSV Structure

Each row represents a unique combination of sample and agent sequence, with the following structure:

```csv
model_name,sample_id,experiment_name,architecture,num_rounds,ground_truth,agent_name,agent_key,execution_order,agent_time_cost,final_predicted_answer,is_finally_correct,final_format_compliance,base_model_predicted_answer,base_model_is_finally_correct,base_model_format_compliance,sample_total_entropy,sample_max_entropy,sample_min_entropy,sample_mean_entropy,sample_median_entropy,sample_std_entropy,sample_variance_entropy,sample_q1_entropy,sample_q3_entropy,sample_num_agents,sample_all_agents_token_count,sample_avg_entropy_per_token,agent_total_entropy,agent_max_entropy,agent_min_entropy,agent_mean_entropy,agent_median_entropy,agent_std_entropy,agent_variance_entropy,agent_q1_entropy,agent_q3_entropy,agent_token_count,agent_avg_entropy_per_token,agent_round_number,agent_avg_entropy,round_total_entropy,round_num_inferences,round_avg_entropy,round_total_time,round_total_token,exp_total_entropy,exp_infer_average_entropy,exp_num_inferences,exp_accuracy,exp_format_compliance_rate,exp_total_time,exp_total_token,base_model_accuracy,base_model_format_compliance_rate
```

**Base Model Columns:**
- `base_model_predicted_answer`: Answer predicted by the base model
- `base_model_is_finally_correct`: Whether base model prediction is correct
- `base_model_format_compliance`: Whether base model answer follows required format
- `base_model_accuracy`: Base model accuracy across all samples
- `base_model_format_compliance_rate`: Base model format compliance rate

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
