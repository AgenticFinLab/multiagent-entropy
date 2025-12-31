# MultiAgent-Entropy Experiments

This directory contains the organized structure for running large-scale experiments with the MultiAgent-Entropy framework.

## Directory Structure

```
experiments/
├── configs/                    # Configuration files for experiments
│   ├── base_config.yml         # Base configuration with common settings
│   ├── batch_example.yml       # Example batch configuration file
│   ├── dataset_specific/       # Dataset-specific configuration files
│   │   └── gsm8k.yml          # GSM8K dataset configuration
│   ├── entropy_configs/        # Entropy calculation configuration files
│   │   ├── standard.yml       # Standard entropy configuration
│   │   └── no_entropy.yml     # No entropy calculation configuration
│   ├── agent_specific/         # Agent-specific configuration files
│   │   ├── single_agents.yml   # Single agent configuration
│   │   ├── fan_agents.yml      # Fan agent configuration (two-layer)
│   │   └── sequential_agents.yml # Sequential agent configuration
│   └── model_specific/         # Model-specific configuration files
│       ├── qwen3-1.7b.yml      # Qwen3-1.7B model configuration
│       └── qwen3-0.6b.yml     # Qwen3-0.6B model configuration
├── configs_exp/                    # Generated experiment configuration files
├── logs/                       # Log files from experiments
├── results/                    # Experiment results
│   ├── aggregated/            # Aggregated results across experiments
│   │   └── {dataset_name}/    # Results organized by dataset
│   └── raw/                   # Raw results from individual experiments
│       └── {dataset_name}/    # Results organized by dataset
└── scripts/                    # Utility scripts
    ├── config_loader.py       # Configuration loading and merging utilities
    ├── result_aggregator.py   # Result aggregation and visualization utilities
    └── run_experiment.py      # Experiment runner script
```

## Configuration System

### Base Configuration
The `base_config.yml` file contains common settings shared across all experiments, such as:
- Environment settings
- Graph configuration
- Inference base configuration
- Generation base configuration
- Default agent type (single, fan, or sequential)

### Model-Specific Configuration
Model-specific configurations (`model_specific/`) define settings for each model, including:
- Model name/path
- Model-specific inference settings (device type, precision, etc.)

### Dataset-Specific Configuration
Dataset-specific configurations (`dataset_specific/`) define settings for each dataset, including:
- Dataset name
- Dataset path
- Number of samples to process
- Batch size

### Entropy Configuration
Entropy configurations (`entropy_configs/`) define settings for entropy calculation, including:
- Whether to calculate entropy
- Type of entropy to calculate
- Entropy-related parameters

### Agent-Specific Configuration
Agent-specific configurations (`agent_specific/`) define the structure and parameters for each agent mode:

#### Single Agent Mode
- **Configuration file**: `agent_specific/single_agents.yml`
- **Description**: Single linear agent topology
- **Agents**: Contains one agent (SingleSolver)

#### Fan Agent Mode (Two-Layer)
- **Configuration file**: `agent_specific/fan_agents.yml`
- **Description**: Two-layer fan topology with multiple agents feeding into a summarizer
- **Agents**: MathAgent, ScienceAgent, CodeAgent, SummarizerAgent
- **Structure**: Layer 1 (Math/Science/Code) → Layer 2 (Summarizer)

#### Sequential Agent Mode
- **Configuration file**: `agent_specific/sequential_agents.yml`
- **Description**: Sequential pipeline topology with agents in a series
- **Agents**: planner, solver, critic, judger
- **Structure**: planner → solver → critic → judger

## Running Experiments

### Single Experiment
To run a single experiment, use the `run_experiment.py` script with the following command:

#### Single Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --model-config experiments/configs/model_specific/qwen3-0.6b.yml \
  --dataset-config experiments/configs/dataset_specific/gsm8k.yml \
  --entropy-config experiments/configs/entropy_configs/standard.yml \
  --experiment-name qwen3-0.6b_gsm8k_single \
  --agent-type single \
  --save-config
```

#### Fan Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --model-config experiments/configs/model_specific/qwen3-0.6b.yml \
  --dataset-config experiments/configs/dataset_specific/gsm8k.yml \
  --entropy-config experiments/configs/entropy_configs/standard.yml \
  --experiment-name qwen3-0.6b_gsm8k_fan \
  --agent-type fan \
  --save-config
```

#### Sequential Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --model-config experiments/configs/model_specific/qwen3-0.6b.yml \
  --dataset-config experiments/configs/dataset_specific/gsm8k.yml \
  --entropy-config experiments/configs/entropy_configs/standard.yml \
  --experiment-name qwen3-0.6b_gsm8k_sequential \
  --agent-type sequential \
  --save-config
```

### Batch Experiments
To run multiple experiments in batch mode, create a batch configuration file like `batch_example.yml` and use:

```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --batch-config experiments/configs/batch_example.yml \
  --save-config
```

### Command-Line Options

- `-b, --base-config`: Path to base configuration file (default: experiments/configs/base_config.yml)
- `-m, --model-config`: Path to model-specific configuration file (required for single experiment)
- `-d, --dataset-config`: Path to dataset-specific configuration file (required for single experiment)
- `-e, --entropy-config`: Path to entropy configuration file (required for single experiment)
- `-n, --experiment-name`: Name of the experiment (required for single experiment)
- `--agent-type`: Type of agent configuration to use (single, fan, sequential)
- `--batch-config`: Path to batch configuration file (for batch experiments)
- `--dry-run`: Only prepare configurations without running experiments
- `--save-config`: Save merged configuration to file (saved in experiments/configs/experiment_configs/ directory)

## Results Management

### Raw Results
Raw experiment results are saved in `experiments/results/raw/<dataset_name>/<experiment_name>_<timestamp>/` directory, including:
- Individual batch results (JSON format)
- Combined final results (JSON format)
- Tensor files (in traces/tensors directory)
- Configuration files

### Aggregated Results
Aggregated results are saved in `experiments/results/aggregated/<dataset_name>/` directory, including:
- Experiment summaries (CSV/JSON format)
- Batch results summaries (CSV/JSON format)
- Visualizations (PNG files) with timestamps in filenames

## Result Aggregation and Visualization

The `result_aggregator.py` script provides functionality to:
1. Load raw experiment results
2. Extract key metrics (accuracy, entropy, etc.)
3. Generate aggregated reports in CSV/JSON format
4. Create visualizations (accuracy comparison, entropy comparison, etc.)

### Usage Example

```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/result_aggregator.py \
  --input-dir experiments/results/raw \
  --output-dir experiments/results/aggregated \
  --format all \
  --visualize \
  --metrics accuracy entropy_mean response_length_mean
```

### Command-Line Options

- `-i, --input-dir`: Directory containing raw experiment results (default: experiments/results/raw)
- `-o, --output-dir`: Directory to save aggregated results (default: experiments/results/aggregated)
- `--format`: Output format for aggregated results (csv, json, or all)
- `--visualize`: Generate visualizations of the results
- `--metrics`: Metrics to extract and visualize (default: accuracy entropy_mean)
- `--experiment-names`: Specific experiment names to process (default: all experiments)

## Adding New Configurations

### Adding a New Model
1. Create a new YAML file in `experiments/configs/model_specific/`
2. Define model-specific settings (lm_name, inference_config, etc.)

### Adding a New Dataset
1. Create a new YAML file in `experiments/configs/dataset_specific/`
2. Define dataset-specific settings (data_name, data_path, data_num, batch_size)

### Adding a New Entropy Configuration
1. Create a new YAML file in `experiments/configs/entropy_configs/`
2. Define entropy calculation settings (calculate_entropy, entropy_type, etc.)

## Best Practices

1. **Configuration Reuse**: Leverage the base configuration to avoid duplication
2. **Experiment Naming**: Use descriptive experiment names that reflect the configuration (e.g., "model_dataset_entropy_setting")
3. **Logging**: Check `experiments/logs/experiment_runner.log` for detailed experiment logs
4. **Dry Runs**: Use `--dry-run` to validate configurations before running expensive experiments
5. **Batch Processing**: Use batch mode for running multiple experiments efficiently

## Example Workflow

1. **Prepare Configurations**: Create or modify configuration files as needed
2. **Test with Dry Run**: Validate configurations with `--dry-run`
3. **Run Experiments**: Execute single or batch experiments
4. **Monitor Progress**: Check logs for experiment progress
5. **Analyze Results**: Examine raw and aggregated results
