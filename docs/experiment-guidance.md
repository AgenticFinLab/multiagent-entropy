# Multi-Agent Entropy Experiments

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
│   │   ├── aime2025.yml                    # AIME2025 dataset configuration
│   │   ├── math500.yml                     # Math500 dataset configuration
│   │   ├── mmlu.yml                        # MMLU dataset configuration
│   │   ├── humaneval.yml                   # HumanEval dataset configuration
│   │   ├── finagent.yml                    # FinAgent finance benchmark configuration
│   │   └── gaia.yml                        # GAIA benchmark configuration
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
│       ├── qwen3-8b.yml                    # Qwen3-8B model configuration
│       ├── llama-3.1-8b-instruct.yml       # LLaMA-3.1-8B-Instruct model configuration
│       ├── llama-3.2-3b-instruct.yml       # LLaMA-3.2-3B-Instruct model configuration
│       └── qwen-2.5-7b-simplerl-zoo.yml    # Qwen2.5-7B-SimpleRL-Zoo model configuration
├── configs_exp/                            # Generated experiment configuration files
├── data/                                   # Dataset storage
├── results/                                # Experiment results
│   ├── aggregated/                         # Aggregated results across experiments
│   │   └── {dataset_name}/                 # Results organized by dataset
│   │       └── {model_name}/               # Results organized by model
│   └── raw/                                # Raw results from individual experiments
│       └── {dataset_name}/                 # Results organized by dataset
│           └── {model_name}/               # Results organized by model
├── temp/                                   # Temporary files
└── scripts/                                # Utility and experiment runner scripts
    ├── config_loader.py                    # Configuration loading and merging utilities
    ├── run_experiment.py                   # Standard experiment runner
    ├── run_finagent_experiment.py          # FinAgent experiment runner
    ├── run_gaia_experiment.py              # GAIA benchmark experiment runner
    ├── regenerate_gaia_evaluation.py       # Re-evaluates existing GAIA results
    ├── download_gaia_attachments.py        # Downloads GAIA task attachments
    ├── finagent_experiment/                # FinAgent-specific agent modules
    └── gaia_experiment/                    # GAIA-specific agent modules
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

### Configuration System

The configuration system is implemented in [config_loader.py](file:///d:/GitHub/multiagent-entropy/experiments/scripts/config_loader.py) and supports hierarchical configuration merging with priority-based overrides.

#### Configuration Priority Order (Highest to Lowest)

1. **Agent-specific configuration** (`agent_specific/{type}_agents.yml`)
2. **Experiment configuration** (auto-generated with save_folder, experiment_name, agent_type)
3. **Dataset-specific configuration** (`dataset_specific/*.yml`)
4. **Model-specific configuration** (`model_specific/*.yml`)
5. **Base configuration** (`base_config.yml`)

#### ConfigLoader Functions

| Function | Description |
|----------|-------------|
| `load_config(config_path)` | Load a YAML configuration file |
| `merge_dicts(dict1, dict2)` | Recursively merge two dictionaries |
| `merge_configs(configs)` | Merge multiple configuration dictionaries |
| `resolve_agent_placeholders()` | Resolve placeholders in agent templates with actual values |
| `load_experiment_config()` | Load and merge all configuration files for an experiment |
| `generate_batch_configs()` | Generate multiple experiment configurations from batch file |
| `save_config()` | Save configuration dictionary to YAML file |
| `is_aime25_all_subset()` | Check if dataset is AIME2025 with subset='all' |
| `map_aime25_subset()` | Map AIME2025 subset values to expected format |
| `prepare_aime25_merged_dataset()` | Merge AIME2025-I and AIME2025-II subsets |

#### Configuration Override Rules

**Generation Configuration** (`generation_config`):
- Priority: `dataset_config` > `base_config`
- Uses `merge_dicts()` for deep merging
- Supported parameters: `max_new_tokens`, `do_sample`, `temperature`, `top_p`

**Inference Configuration** (`inference_config`):
- Priority: `model_config` > `base_config`
- Falls back to defaults if not found: `device: cuda`, `torch_dtype: float16`, `device_map: auto`

**Agent Configuration**:
- Placeholders resolved in `resolve_agent_placeholders()`:
  - `lm_name`: Replaced with model_config's lm_name
  - `inference_config`: From model_config or base_config
  - `generation_config`: Merged from dataset_config and base_config

#### AIME2025 Special Handling

When `subset: all` is specified for AIME2025 dataset:
1. `is_aime25_all_subset()` detects this configuration
2. `prepare_aime25_merged_dataset()` is called to merge AIME2025-I and AIME2025-II
3. Samples from AIME2025-II are renumbered (ID{last_id_num + 1}, etc.)
4. Merged dataset is saved to `experiments/data/AIME2025/{split}-all-samples.json`
5. `map_aime25_subset()` handles subset value mapping ('i' → 'AIME2025-I', 'ii' → 'AIME2025-II')

#### Example: max_new_tokens Override

**Base Configuration** (`base_config.yml`):
```yaml
generation_config:
  max_new_tokens: 2000
  do_sample: true
  temperature: 0.6
  top_p: 0.95
```

**Dataset-Specific Configuration** (`dataset_specific/aime2025.yml`):
```yaml
data:
  data_name: AIME2025
  data_path: experiments/data/AIME2025
  subset: all  # 'all' for both AIME2025-I and AIME2025-II
  split: test
  data_num: -1
  batch_size: 1

task_type: "math"

generation_config:
  max_new_tokens: 8192  # Override for longer reasoning chains
```

**Result**: When running experiments with AIME2025 dataset:
- `max_new_tokens`: 8192 (overridden from dataset config)
- `do_sample`: true (inherited from base config)
- `temperature`: 0.6 (inherited from base config)
- `top_p`: 0.95 (inherited from base config)

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
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name qwen3-4b_aime2025_single_agent \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "single"
```

#### Sequential Agent Mode
```bash
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_sequential_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "sequential"
```

#### Centralized Agent Mode
```bash
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_centralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "centralized"
```

#### Decentralized Agent Mode
```bash
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_decentralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "decentralized"
```

#### Full Decentralized Agent Mode
```bash
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_full_decentralized_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "full_decentralized"
```

#### Debate Agent Mode
```bash
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_debate_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "debate"
```

#### Hybrid Agent Mode
```bash
conda activate maep
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --experiment-name "qwen3-4b_aime2025_hybrid_agent" \
  --base-config "experiments/configs/base_config.yml" \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/aime2025.yml" \
  --agent-type "hybrid"
```

### Batch Experiments

To run multiple experiments in batch mode, create a batch configuration file and use:

```bash
cd multiagent-entropy
python experiments/scripts/run_experiment.py \
  --batch-config experiments/configs/batch_example_qwen3_4b_gsm8k.yml
```

#### Batch Configuration Format

```yaml
experiments:
  - name: experiment_name_1
    base_config: experiments/configs/base_config.yml
    model_config: experiments/configs/model_specific/qwen3-4b.yml
    dataset_config: experiments/configs/dataset_specific/gsm8k.yml
    agent_type: centralized
  
  - name: experiment_name_2
    base_config: experiments/configs/base_config.yml
    model_config: experiments/configs/model_specific/qwen3-4b.yml
    dataset_config: experiments/configs/dataset_specific/gsm8k.yml
    agent_type: debate
  
  - name: experiment_name_3
    base_config: experiments/configs/base_config.yml
    model_config: experiments/configs/model_specific/qwen3-8b.yml
    dataset_config: experiments/configs/dataset_specific/aime2025.yml
    agent_type: hybrid
```

#### Running Specific Experiment from Batch

To run only a specific experiment from a batch configuration:

```bash
python experiments/scripts/run_experiment.py \
  --batch-config experiments/configs/batch_example.yml \
  --experiment-name experiment_name_1
```

### Command-Line Options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--base-config` | `-b` | No | `experiments/configs/base_config.yml` | Path to base configuration file |
| `--model-config` | `-m` | Yes (single mode) | None | Path to model-specific configuration file |
| `--dataset-config` | `-d` | Yes (single mode) | None | Path to dataset-specific configuration file |
| `--experiment-name` | `-n` | Yes (single mode) | None | Name of the experiment |
| `--agent-type` | - | Yes (single mode) | None | Agent type: single, sequential, centralized, decentralized, full_decentralized, debate, hybrid |
| `--batch-config` | `-bc` | Yes (batch mode) | None | Path to batch configuration file |
| `--dry-run` | - | No | False | Only prepare configurations without running experiments |
| `--save-config` | - | No | True | Save merged configuration to file |

#### Supported Agent Types

The experiment runner supports all seven multi-agent system architectures:

| Agent Type | Description |
|------------|-------------|
| `single` | Single solver agent baseline |
| `sequential` | Sequential pipeline with planner → solver → critic → judger |
| `centralized` | Two-layer: domain agents (Math/Science/Code) → central orchestrator |
| `decentralized` | Sequential agents with loopback before orchestration |
| `full_decentralized` | Each agent can communicate with all other agents |
| `debate` | Multi-agent debate with majority voting (no LLM orchestrator) |
| `hybrid` | Two-layer with enhanced context sharing and feedback |

### FinAgent Experiments

FinAgent is the Finance Agent Benchmark. It uses rubric-based evaluation via financial tools (web search, EDGAR SEC search, HTML parsing).

#### Running a FinAgent Experiment

```bash
cd multiagent-entropy
python experiments/scripts/run_finagent_experiment.py \
  --experiment-name qwen3-4b_finagent_single_agent \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/finagent.yml" \
  --agent-type "single"
```

Run as a batch:

```bash
python experiments/scripts/run_finagent_experiment.py \
  --batch-config my_finagent_batch.yml
```

#### FinAgent-Specific CLI Options

In addition to all standard options above, `run_finagent_experiment.py` accepts:

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-config` | `experiments/configs/dataset_specific/finagent.yml` | Defaults to FinAgent config |
| `--skip-evaluation` | False | Skip rubric-based evaluation; run inference only |

#### FinAgent Package (`finagent_experiment/`)

| Module | Purpose |
|--------|---------|
| `runner.py` | Main experiment orchestration |
| `tools.py` | Financial tools: `GoogleWebSearch`, `EDGARSearch`, `ParseHtmlPage`, `RetrieveInformation` |
| `evaluation.py` | Rubric-based scoring via `calculate_finagent_accuracy` |
| `answer_extraction.py` | Extract answers by identifier |
| `sec_cache.py` | SEC query caching |
| `tool_logger.py` | Tool call logging |
| `prompts.py` | Prompt templates |
| `checkpoint.py` | Checkpoint / resume support |
| `constants.py` | `MAX_END_DATE`, `FINAGENT_TASK_TYPE` |

Results are saved to `experiments/results_finagent/raw/`.

### GAIA Experiments

GAIA is a general-purpose AI assistant benchmark with exact-match evaluation. It includes multi-modal tasks that may require web search and file-attachment processing.

#### Downloading GAIA Attachments

Before running GAIA experiments, download task attachments:

```bash
python experiments/scripts/download_gaia_attachments.py
```

#### Running a GAIA Experiment

```bash
cd multiagent-entropy
python experiments/scripts/run_gaia_experiment.py \
  --experiment-name qwen3-4b_gaia_single_agent \
  --model-config "experiments/configs/model_specific/qwen3-4b.yml" \
  --dataset-config "experiments/configs/dataset_specific/gaia.yml" \
  --agent-type "single"
```

#### GAIA-Specific CLI Options

In addition to all standard options above, `run_gaia_experiment.py` accepts:

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-config` | `experiments/configs/dataset_specific/gaia.yml` | Defaults to GAIA config |
| `--skip-evaluation` | False | Skip evaluation; run inference only |

#### Re-evaluating Existing GAIA Results

To re-score already-generated GAIA outputs without re-running inference:

```bash
python experiments/scripts/regenerate_gaia_evaluation.py
```

#### GAIA Package (`gaia_experiment/`)

| Module | Purpose |
|--------|---------|
| `runner.py` | Main experiment orchestration |
| `tools.py` | GAIA tools (web search, file reading, etc.) |
| `evaluation.py` | Exact-match evaluation |
| `answer_extraction.py` | Extract final answers |
| `prompts.py` | Prompt templates |
| `checkpoint.py` | Checkpoint / resume support |
| `constants.py` | GAIA-specific constants |

#### Raw Results

Raw experiment results are saved in:
```
experiments/results/raw/<dataset_name>/<model_name>/<experiment_name>_<timestamp>_<ms>_<pid>/
```

Contents:
- `Batch_{N}_State.json`: Individual batch results
- `Combined_FinalState.json`: Combined final results
- `traces/tensors/`: Tensor files with entropy data
- Configuration files

#### Aggregated Results

Aggregated results are saved in:
```
experiments/results/aggregated/<dataset_name>/<model_name>/
```

Contents:
- `{experiment_name}_results_{timestamp}.yml`: Individual experiment summaries
- `batch_results_{timestamp}.yml`: Batch experiment summaries

#### Generated Configurations

Merged configurations are saved to:
```
experiments/configs_exp/<dataset_name>/<model_name>/<experiment_name>_<timestamp>_<ms>_<pid>.yml
```

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
