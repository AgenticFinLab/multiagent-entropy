#!/usr/bin/env python3
"""
Configuration loader for MultiAgent-Entropy experiments.

This script provides functionality to:
1. Load and merge configuration files
2. Support multiple agent modes: single, sequential, centralized, decentralized, full_decentralized, debate, hybrid
3. Load agent-specific configurations from agent_specific directory
4. Merge all configurations into a complete experiment configuration

Usage:
    from config_loader import load_experiment_config, generate_batch_configs

    # Load a single experiment config
    config = load_experiment_config(
        base_config_path='experiments/configs/base_config.yml',
        model_config_path='experiments/configs/model_specific/qwen3-0.6b.yml',
        dataset_config_path='experiments/configs/dataset_specific/gsm8k.yml',
        experiment_name='test_experiment',
        agent_type='centralized'
    )

    # Generate configs from batch file
    batch_configs = generate_batch_configs('experiments/configs/batch_example.yml')
"""

import os
import yaml
import time
import json
import logging
from typing import Dict, Any, List

from lmbase.dataset import registry as data_registry

# Ensure log directory exists
os.makedirs("experiments/logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiments/logs/config_loader.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges two dictionaries.

    Args:
        dict1 (Dict[str, Any]): First dictionary.
        dict2 (Dict[str, Any]): Second dictionary.

    Returns:
        Dict[str, Any]: Merged dictionary.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merges multiple configuration dictionaries into one.

    Args:
        configs (List[Dict[str, Any]]): List of configuration dictionaries to merge.

    Returns:
        Dict[str, Any]: Merged configuration dictionary.
    """
    merged = {}
    for config in configs:
        merged = merge_dicts(merged, config)
    return merged


def resolve_agent_placeholders(
    agent_template: Dict[str, Any],
    model_config: Dict[str, Any],
    base_config: Dict[str, Any],
    dataset_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Resolve placeholders in agent configuration template with actual values.

    Args:
        agent_template (Dict[str, Any]): Agent configuration template
        model_config (Dict[str, Any]): Model configuration
        base_config (Dict[str, Any]): Base configuration
        dataset_config (Dict[str, Any]): Dataset-specific configuration (optional, overrides base_config)

    Returns:
        Dict[str, Any]: Resolved agent configuration
    """
    resolved_agent = agent_template.copy()

    # Replace placeholders with actual values
    resolved_agent["lm_name"] = model_config["lm_name"]

    # Determine inference_config priority: model_config > base_config
    if "inference_config" in model_config:
        resolved_agent["inference_config"] = model_config["inference_config"]
    elif "inference_config" in base_config:
        resolved_agent["inference_config"] = base_config["inference_config"]
    else:
        # Provide default inference_config if not found in any config
        resolved_agent["inference_config"] = {
            "device": "cuda",
            "torch_dtype": "float16",
            "device_map": "auto",
        }
    # Determine generation_config priority: dataset_config > base_config
    # If dataset_config has generation_config, merge it with base_config to ensure all fields are present
    if dataset_config and "generation_config" in dataset_config:
        resolved_agent["generation_config"] = merge_dicts(
            base_config["generation_config"], dataset_config["generation_config"]
        )
    else:
        resolved_agent["generation_config"] = base_config["generation_config"]

    return resolved_agent


def load_experiment_config(
    base_config_path: str,
    model_config_path: str,
    dataset_config_path: str,
    experiment_name: str,
    agent_type: str = None,
) -> Dict[str, Any]:
    """Loads and merges all configuration files for an experiment.

    Args:
        base_config_path (str): Path to base configuration file.
        model_config_path (str): Path to model-specific configuration file.
        dataset_config_path (str): Path to dataset-specific configuration file.
        experiment_name (str): Name of the experiment.
        agent_type (str): Type of agent configuration to load (single/sequential/centralized/decentralized/full_decentralized/debate/hybrid).

    Returns:
        Dict[str, Any]: Complete merged configuration for the experiment.
    """
    try:
        # Load individual configs
        base_config = load_config(base_config_path)
        model_config = load_config(model_config_path)
        dataset_config = load_config(dataset_config_path)

        # Get dataset name and model name for path creation
        dataset_name = dataset_config["data"]["data_name"].lower()
        model_name = (
            model_config["lm_name"]
            .split("/")[-1]
            .lower()
            .replace("-", "_")
            .replace(".", "_")
        )

        # Create experiment-specific config with timestamp and process ID to avoid conflicts
        import os as os_module

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        timestamp_ms = int(time.time() * 1000) % 1000
        pid = os_module.getpid()
        experiment_config = {
            "save_folder": f"experiments/results/raw/{dataset_name}/{model_name}/{experiment_name}_{timestamp}_{timestamp_ms}_{pid}"
        }

        # Get agent type - prioritize passed parameter, then base config, then default to single
        agent_type = agent_type or base_config.get("agent_type", "single")

        # Validate agent type
        valid_agent_types = [
            "single",
            "sequential",
            "centralized",
            "decentralized",
            "full_decentralized",
            "debate",
            "hybrid",
        ]
        if agent_type not in valid_agent_types:
            raise ValueError(
                f"Invalid agent_type: {agent_type}. Valid types are: {valid_agent_types}"
            )

        # Load agent-specific configuration
        agent_config_path = (
            f"experiments/configs/agent_specific/{agent_type}_agents.yml"
        )
        agent_specific_config = load_config(agent_config_path)

        # Create agents config with actual parameter values
        agents_config = {"agents": {}}
        for agent_name, agent_template in agent_specific_config["agents"].items():
            resolved_agent = resolve_agent_placeholders(
                agent_template=agent_template,
                model_config=model_config,
                base_config=base_config,
                dataset_config=dataset_config,
            )
            agents_config["agents"][agent_name] = resolved_agent

        # Merge all configs
        all_configs = [
            base_config,
            model_config,
            dataset_config,
            experiment_config,
            agents_config,
        ]

        merged_config = merge_configs(all_configs)

        # Add experiment name and agent type for reference
        merged_config["experiment_name"] = experiment_name
        merged_config["agent_type"] = agent_type

        logger.info(
            f"Generated experiment configuration for: {experiment_name} (Agent type: {agent_type})"
        )
        return merged_config

    except Exception as e:
        logger.error(f"Error loading experiment configuration: {str(e)}")
        raise


def generate_batch_configs(batch_config_path: str) -> Dict[str, Any]:
    """Generate multiple experiment configurations from a batch configuration file.

    Args:
        batch_config_path (str): Path to batch configuration file.

    Returns:
        Dict[str, Any]: Dictionary of experiment configurations, keyed by experiment name.
    """
    try:
        batch_config = load_config(batch_config_path)
        experiment_configs = {}

        for experiment in batch_config["experiments"]:
            name = experiment["name"]
            base_config_path = experiment.get(
                "base_config", "experiments/configs/base_config.yml"
            )
            model_config_path = experiment["model_config"]
            dataset_config_path = experiment["dataset_config"]
            agent_type = experiment.get("agent_type")

            # Load and merge configurations
            experiment_config = load_experiment_config(
                base_config_path=base_config_path,
                model_config_path=model_config_path,
                dataset_config_path=dataset_config_path,
                experiment_name=name,
                agent_type=agent_type,
            )

            experiment_configs[name] = experiment_config

        logger.info(
            f"Generated {len(experiment_configs)} experiment configurations from batch file: {batch_config_path}"
        )
        return experiment_configs

    except Exception as e:
        logger.error(f"Error generating batch configurations: {str(e)}")
        raise


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Saves a configuration dictionary to a YAML file.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        save_path (str): Path to save the YAML file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Saved configuration to: {save_path}")


def is_aime25_all_subset(dataset_config: Dict[str, Any]) -> bool:
    """Check if the dataset is AIME2025 with subset='all'.

    Args:
        dataset_config (Dict[str, Any]): Dataset configuration dictionary.

    Returns:
        bool: True if dataset is AIME2025 with subset='all', False otherwise.
    """
    data_name = dataset_config.get("data", {}).get("data_name", "")
    subset = dataset_config.get("data", {}).get("subset", "")
    return data_name.lower() == "aime2025" and subset.lower() == "all"


def map_aime25_subset(subset: str) -> str:
    """Map AIME2025 subset values to the format expected by data_registry.

    Args:
        subset (str): Subset value from config ('all', 'i', 'ii', 'AIME2025-I', 'AIME2025-II')

    Returns:
        str: Mapped subset value ('AIME2025-I', 'AIME2025-II', or original value)
    """
    subset_lower = subset.lower()
    if subset_lower == "i":
        return "AIME2025-I"
    elif subset_lower == "ii":
        return "AIME2025-II"
    return subset


def prepare_aime25_merged_dataset(dataset_config: Dict[str, Any]) -> str:
    """Prepare merged AIME2025 dataset by combining AIME2025-I and AIME2025-II subsets.

    Args:
        dataset_config (Dict[str, Any]): Dataset configuration dictionary.

    Returns:
        str: Path to the saved merged dataset file.
    """

    data_cfg = dataset_config["data"]
    data_path = data_cfg["data_path"]
    split = data_cfg["split"]

    # Load AIME2025-I subset
    ds_i = data_registry.get(
        {
            "data_name": "aime2025",
            "data_path": data_path,
            "subset": "AIME2025-I",
        },
        split,
    )

    # Load AIME2025-II subset
    ds_ii = data_registry.get(
        {
            "data_name": "aime2025",
            "data_path": data_path,
            "subset": "AIME2025-II",
        },
        split,
    )

    # Merge the two subsets
    merged_samples = []

    # Add samples from AIME2025-I
    for i in range(len(ds_i)):
        merged_samples.append(ds_i[i])

    # Find the last main_id from AIME2025-I
    last_main_id = None
    if len(merged_samples) > 0:
        last_main_id = merged_samples[-1].get("main_id", "")

    # Extract the numeric part from the last main_id
    last_id_num = 0
    if last_main_id and last_main_id.startswith("ID"):
        try:
            last_id_num = int(last_main_id[2:])
        except ValueError:
            last_id_num = len(merged_samples)

    # Add samples from AIME2025-II with renumbered main_id
    for i in range(len(ds_ii)):
        sample = ds_ii[i]
        last_id_num += 1
        sample["main_id"] = f"ID{last_id_num}"
        merged_samples.append(sample)

    logger.info(
        f"Merged {len(ds_i)} samples from AIME2025-I and {len(ds_ii)} samples from AIME2025-II"
    )
    logger.info(
        f"Renumbered AIME2025-II samples starting from ID{last_id_num - len(ds_ii) + 1}"
    )

    # Convert list of dicts to dict of lists for consistency with HuggingFace Dataset format
    if len(merged_samples) > 0:
        merged_samples_dict = {
            key: [sample[key] for sample in merged_samples]
            for key in merged_samples[0].keys()
        }
    else:
        merged_samples_dict = {}

    # Save merged dataset
    save_dir = f"experiments/data/AIME2025"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split}-all-samples.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged_samples_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved merged AIME2025 dataset to: {save_path}")
    return save_path


if __name__ == "__main__":
    """Test the configuration loader functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="Test configuration loader")
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=[
            "single",
            "sequential",
            "centralized",
            "decentralized",
            "full_decentralized",
            "debate",
            "hybrid",
        ],
        default="single",
        help="Agent type to test",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Test batch configuration generation"
    )

    args = parser.parse_args()

    if args.batch:
        # Test batch configuration generation
        try:
            batch_configs = generate_batch_configs(
                "experiments/configs/batch_example.yml"
            )

            print(f"\nGenerated {len(batch_configs)} batch configurations:")
            for exp_name, config in batch_configs.items():
                print(
                    f"- {exp_name}: Agent type = {config.get('agent_type')}, Agents = {list(config['agents'].keys())}"
                )

        except Exception as e:
            print(f"Error generating batch configurations: {e}")
    else:
        # Test single experiment configuration
        try:
            config = load_experiment_config(
                base_config_path="experiments/configs/base_config.yml",
                model_config_path="experiments/configs/model_specific/qwen3-0.6b.yml",
                dataset_config_path="experiments/configs/dataset_specific/gsm8k.yml",
                experiment_name="test_experiment",
                agent_type=args.agent_type,
            )

            print(f"\nGenerated experiment config for agent type: {args.agent_type}")
            print(f"Experiment name: {config['experiment_name']}")
            print(f"Agent type: {config['agent_type']}")
            print(f"Agents configured: {list(config['agents'].keys())}")
            print(f"Data name: {config['data']['data_name']}")
            print(
                f"Model name: {config['agents'][list(config['agents'].keys())[0]]['lm_name']}"
            )
            print(f"Save folder: {config['save_folder']}")

            # Test saving the config
            save_config(
                config, f"experiments/configs/test_{args.agent_type}_config.yml"
            )
            print(
                f"\nSaved test configuration to: experiments/configs/test_{args.agent_type}_config.yml"
            )

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
