# MultiAgent-Entropy Experiments

This directory contains the organized structure for running large-scale experiments with the MultiAgent-Entropy framework.

## Directory Structure

```
experiments/
├── configs/                                # Configuration files for experiments
│   ├── base_config.yml                     # Base configuration with common settings
│   ├── batch_example_qwen3_4b_gsm8k.yml    # Example batch configuration file
│   ├── batch_example_qwen3_8b_gsm8k.yml    # Example batch configuration file
│   ├── dataset_specific/                   # Dataset-specific configuration files
│   │   ├── gsm8k.yml                       # GSM8K dataset configuration
│   │   ├── aime2024.yml                    # AIME2024 dataset configuration
│   │   ├── mmlu.yml                        # MMLU dataset configuration
│   │   └── humaneval.yml                   # HumanEval dataset configuration
│   ├── agent_specific/                     # Agent-specific configuration files
│   │   ├── single_agents.yml               # Single agent configuration
│   │   ├── sequential_agents.yml           # Sequential agent configuration
│   │   ├── centralized_agents.yml          # Centralized agent configuration
│   │   ├── decentralized_agents.yml        # Decentralized agent configuration
│   │   ├── full_decentralized_agents.yml   # Full decentralized agent configuration
│   │   ├── debate_agents.yml               # Debate agent configuration
│   │   └── hybrid_agents.yml               # Hybrid agent configuration
│   └── model_specific/                     # Model-specific configuration files
│       ├── qwen3-0.6b.yml                  # Qwen3-0.6B model configuration
│       ├── qwen3-1.7b.yml                  # Qwen3-1.7B model configuration
│       ├── qwen3-4b.yml                    # Qwen3-4B model configuration
│       └── qwen3-8b.yml                    # Qwen3-8B model configuration
├── configs_exp/                            # Generated experiment configuration files
├── logs/                                   # Log files from experiments
├── results/                                # Experiment results
│   ├── aggregated/                         # Aggregated results across experiments
│   │   └── {dataset_name}/                 # Results organized by dataset
│   └── raw/                                # Raw results from individual experiments
│       └── {dataset_name}/                 # Results organized by dataset
└── scripts/                                # Utility scripts
    ├── config_loader.py                    # Configuration loading and merging utilities
    ├── result_aggregator.py                # Result aggregation and visualization utilities
    └── run_experiment.py                   # Experiment runner script
```

## Configuration System

### Base Configuration
The `base_config.yml` file contains common settings shared across all experiments, including:
- Environment settings (dotenv_path)
- Graph configuration (recursion_limit)
- Generation base configuration (max_new_tokens, do_sample, temperature, top_p)
- Inference configuration (device, torch_dtype, device_map)
- Entropy configuration (calculate_entropy, entropy_type, etc.)
- Agent round settings (round, aggregate_history, max_history_chars, max_history_rounds)
- Save folder configuration

The `inference_config` and `entropy_config` sections in `base_config.yml` provide default settings for all experiments. These can be overridden by model-specific configurations if needed.

### Model-Specific Configuration
Model-specific configurations (`model_specific/`) define settings for each model, including:
- Model name/path
- Optional model-specific inference settings (can override base_config's inference_config)

### Dataset-Specific Configuration
Dataset-specific configurations (`dataset_specific/`) define settings for each dataset, including:
- Dataset name
- Dataset path
- Number of samples to process
- Batch size
- Optional generation_config overrides (can override base_config's generation_config)

### Configuration Priority and Override Mechanism
The configuration system supports a hierarchical priority mechanism that allows dataset-specific configurations to override base configuration settings. This enables flexible customization for different datasets while maintaining a common baseline.

#### Priority Order (Highest to Lowest)
1. **Dataset-specific configuration** (`dataset_specific/*.yml`)
2. **Base configuration** (`base_config.yml`)

#### Generation Configuration Override
The `generation_config` section supports parameter-level overrides. When a dataset-specific configuration defines `generation_config`, it is merged with the base configuration using the following rules:

- **Override**: Parameters defined in the dataset-specific configuration replace those in the base configuration
- **Inheritance**: Parameters not defined in the dataset-specific configuration are inherited from the base configuration
- **Supported parameters**: `max_new_tokens`, `do_sample`, `temperature`, `top_p`

#### Example: max_new_tokens Override

**Base Configuration** (`base_config.yml`):
```yaml
generation_config:
  max_new_tokens: 2000
  do_sample: true
  temperature: 0.6
  top_p: 0.95
```

**Dataset-Specific Configuration** (`dataset_specific/aime2024.yml`):
```yaml
generation_config:
  max_new_tokens: 8192
```

**Result**: When running experiments with AIME2024 dataset:
- `max_new_tokens`: 8192 (overridden from dataset config)
- `do_sample`: true (inherited from base config)
- `temperature`: 0.6 (inherited from base config)
- `top_p`: 0.95 (inherited from base config)

#### Implementation Details
The override mechanism is implemented in `experiments/scripts/config_loader.py` in the `resolve_agent_placeholders()` function. The function:
1. Checks if `dataset_config` contains a `generation_config` section
2. If present, merges it with `base_config["generation_config"]` using `merge_dicts()`
3. The merged configuration ensures all required parameters are present while respecting dataset-specific overrides

This design ensures that:
- New datasets can easily customize generation parameters without modifying the base configuration
- Missing parameters in dataset-specific configurations are safely inherited from the base configuration
- The system remains flexible and maintainable as new datasets are added

### Agent-Specific Configuration
Agent-specific configurations (`agent_specific/`) define the structure and parameters for each agent mode:

#### Single Agent Mode
- **Configuration file**: `agent_specific/single_agents.yml`
- **Description**: Single linear agent topology
- **Agents**: Contains one agent (SingleSolver)

#### Sequential Agent Mode
- **Configuration file**: `agent_specific/sequential_agents.yml`
- **Description**: Sequential pipeline topology with agents in a series
- **Agents**: planner, solver, critic, judger
- **Structure**: planner -> solver -> critic -> judger -> (Loop)

#### Centralized Agent Mode (Two-Layer)
- **Configuration file**: `agent_specific/centralized_agents.yml`
- **Description**: Two-layer centralized topology with domain-specific agents and central orchestrator
- **Agents**: MathAgent, ScienceAgent, CodeAgent, OrchestratorAgent
- **Structure**: Layer 1 (Math/Science/Code) -> Layer 2 (Orchestrator) -> (Loop)

#### Decentralized Agent Mode (Loop + Orchestrator)
- **Configuration file**: `agent_specific/decentralized_agents.yml`
- **Description**: Sequential agents with loopback mechanism before final orchestration
- **Agents**: MathAgent, ScienceAgent, CodeAgent, OrchestratorAgent
- **Structure**: Math -> Science -> Code -> (Loop) -> Orchestrator

#### Full Decentralized Agent Mode (Loop + Orchestrator)
- **Configuration file**: `agent_specific/full_decentralized_agents.yml`
- **Description**: Sequential agents with loopback mechanism before final orchestration, each agent can communicate with all other agents
- **Agents**: MathAgent, ScienceAgent, CodeAgent, OrchestratorAgent
- **Structure**: Math -> Science -> Code -> (Loop) -> Orchestrator

#### Debate Agent Mode (Multi-Agent with Voting)
- **Configuration file**: `agent_specific/debate_agents.yml`
- **Description**: Multi-agent debate system with majority voting mechanism
- **Agents**: agent1, agent2, agent3 (no orchestrator agent - uses majority voting instead)
- **Structure**: Sequential agents in loop -> Majority voting -> Output
- **Note**: Unlike other agent modes, Debate mode uses majority voting instead of an orchestrator agent. The orchestrator does NOT use LLM inference; it extracts answers wrapped in \\boxed{} from each agent's response and selects the most frequent one as the final result.

#### Hybrid Agent Mode (Enhanced Context Sharing)
- **Configuration file**: `agent_specific/hybrid_agents.yml`
- **Description**: Two-layer hybrid topology with enhanced context sharing and feedback
- **Agents**: MathAgent, ScienceAgent, CodeAgent, OrchestratorAgent
- **Structure**: Layer 1 (Math/Science/Code with communication) -> Layer 2 (Orchestrator with feedback) -> (Loop)

## Running Experiments

### Single Experiment
To run a single experiment, use the `run_experiment.py` script with the following command:

#### Single Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_aime2024_single_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "single"
```

#### Sequential Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_sequential_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "sequential"
```

#### Centralized Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_centralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "centralized"
```

#### Decentralized Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_decentralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "decentralized"
```

#### Full Decentralized Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_full_decentralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "full_decentralized"
```

#### Debate Agent Mode
```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_debate_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "debate"
```

#### Hybrid Agent Mode
```bash
conda activate maep
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2024_hybrid_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2024.yml" \
  --agent-type "hybrid"
```

### Batch Experiments
To run multiple experiments in batch mode, create a batch configuration file like `batch_example_qwen3_4b_gsm8k.yml` and use:

```bash
cd /home/yuxuanzhao/multiagent-entropy
python experiments/scripts/run_experiment.py \
  --batch-config experiments/configs/batch_example_qwen3_4b_gsm8k.yml
```

### Command-Line Options

- `-b, --base-config`: Path to base configuration file (default: experiments/configs/base_config.yml)
- `-m, --model-config`: Path to model-specific configuration file (required for single experiment)
- `-d, --dataset-config`: Path to dataset-specific configuration file (required for single experiment)
- `-n, --experiment-name`: Name of the experiment (required for single experiment)
- `--agent-type`: Type of agent configuration to use (single, sequential, centralized, decentralized, full_decentralized, debate, hybrid)
- `--batch-config`: Path to batch configuration file (for batch experiments)
- `--dry-run`: Only prepare configurations without running experiments
- `--save-config`: Save merged configuration to file (default: True)

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
- `--format`: Output format for aggregated results (csv, json, or all; default: all)
- `--visualize`: Generate visualizations of the results
- `--metrics`: Metrics to extract and visualize (default: accuracy entropy_mean round)
- `--experiment-names`: Specific experiment names to process (default: all experiments)

## Adding New Configurations

### Adding a New Model
1. Create a new YAML file in `experiments/configs/model_specific/`
2. Define model-specific settings (lm_name, inference_config, etc.)

### Adding a New Dataset
1. Create a new YAML file in `experiments/configs/dataset_specific/`
2. Define dataset-specific settings (data_name, data_path, data_num, batch_size)
3. Optionally add `generation_config` section to override base generation parameters:
   ```yaml
   generation_config:
     max_new_tokens: 8192  # Override default max_new_tokens
     # Other generation parameters (do_sample, temperature, top_p) will be inherited from base_config.yml
   ```

## Best Practices

1. **Configuration Reuse**: Leverage the base configuration to avoid duplication
2. **Experiment Naming**: Use descriptive experiment names that reflect the configuration (e.g., "model_dataset_entropy_setting")
3. **Logging**: Check `experiments/logs/experiment_runner.log` for detailed experiment logs
4. **Dry Runs**: Use `--dry-run` to validate configurations before running expensive experiments
5. **Batch Processing**: Use batch mode for running multiple experiments efficiently

## Special Notes and Limitations

### Agent Mode Specific Notes
- **Debate Mode**: Unlike other agent modes, Debate mode uses majority voting instead of an orchestrator agent. The orchestrator does NOT use LLM inference; it extracts answers wrapped in \\boxed{} from each agent's response and selects the most frequent one as the final result.
- **History Aggregation**: When `aggregate_history: true`, all previous round interactions, outputs, and context are aggregated into the next round's prompt. Use `max_history_chars` and `max_history_rounds` to limit the history size and manage memory usage.

### Environment Configuration
- **dotenv_path**: The base configuration specifies `dotenv_path: .env` for loading environment variables. Ensure a `.env` file exists in the project root directory with necessary API keys and configuration values.

### Result Storage
- **Raw Results**: Individual batch results are saved as `Batch_{N}_State.json` files, and combined results are saved as `Combined_FinalState.json`.
- **Aggregated Results**: Batch experiment summaries are saved with timestamps (e.g., `batch_results_20250104_123456.yml`), and single experiment results are saved as `{experiment_name}_results_{timestamp}.yml`.

## Example Workflow

1. **Prepare Configurations**: Create or modify configuration files as needed
2. **Test with Dry Run**: Validate configurations with `--dry-run`
3. **Run Experiments**: Execute single or batch experiments
4. **Monitor Progress**: Check logs for experiment progress
5. **Analyze Results**: Examine raw and aggregated results
