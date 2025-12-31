#!/usr/bin/env python3
"""
Configuration loader for MultiAgent-Entropy experiments.

This script provides functionality to:
1. Load and merge configuration files
2. Support three agent modes: single, fan, and sequential
3. Load agent-specific configurations from agent_specific directory
4. Merge all configurations into a complete experiment configuration

Usage:
    from config_loader import load_experiment_config, generate_batch_configs

    # Load a single experiment config
    config = load_experiment_config(
        base_config_path='experiments/configs/base_config.yml',
        model_config_path='experiments/configs/model_specific/qwen3-0.6b.yml',
        dataset_config_path='experiments/configs/dataset_specific/gsm8k.yml',
        entropy_config_path='experiments/configs/entropy_configs/standard.yml',
        experiment_name='test_experiment',
        agent_type='fan'
    )

    # Generate configs from batch file
    batch_configs = generate_batch_configs('experiments/configs/batch_example.yml')
"""

import os
import yaml
import time
import logging
from typing import Dict, Any, List

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
    entropy_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve placeholders in agent configuration template with actual values.

    Args:
        agent_template (Dict[str, Any]): Agent configuration template
        model_config (Dict[str, Any]): Model configuration
        base_config (Dict[str, Any]): Base configuration
        entropy_config (Dict[str, Any]): Entropy configuration

    Returns:
        Dict[str, Any]: Resolved agent configuration
    """
    resolved_agent = agent_template.copy()

    # Replace placeholders with actual values
    resolved_agent["lm_name"] = model_config["lm_name"]
    resolved_agent["inference_config"] = model_config.get(
        "inference_config", base_config["inference_config"]
    )
    resolved_agent["entropy_config"] = entropy_config["entropy_config"]
    resolved_agent["generation_config"] = base_config["generation_config"]

    return resolved_agent


def load_experiment_config(
    base_config_path: str,
    model_config_path: str,
    dataset_config_path: str,
    entropy_config_path: str,
    experiment_name: str,
    agent_type: str = None,
) -> Dict[str, Any]:
    """Loads and merges all configuration files for an experiment.

    Args:
        base_config_path (str): Path to base configuration file.
        model_config_path (str): Path to model-specific configuration file.
        dataset_config_path (str): Path to dataset-specific configuration file.
        entropy_config_path (str): Path to entropy configuration file.
        experiment_name (str): Name of the experiment.
        agent_type (str): Type of agent configuration to load (single/fan/sequential).

    Returns:
        Dict[str, Any]: Complete merged configuration for the experiment.
    """
    try:
        # Load individual configs
        base_config = load_config(base_config_path)
        model_config = load_config(model_config_path)
        dataset_config = load_config(dataset_config_path)
        entropy_config = load_config(entropy_config_path)

        # Get dataset name for path creation
        dataset_name = dataset_config['data']['data_name'].lower()
        
        # Create experiment-specific config with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        experiment_config = {
            "save_folder": f"experiments/results/raw/{dataset_name}/{experiment_name}_{timestamp}"
        }

        # Get agent type - prioritize passed parameter, then base config, then default to single
        agent_type = agent_type or base_config.get("agent_type", "single")

        # Validate agent type
        valid_agent_types = ["single", "fan", "sequential"]
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
                entropy_config=entropy_config,
            )
            agents_config["agents"][agent_name] = resolved_agent

        # Merge all configs
        all_configs = [
            base_config,
            model_config,
            dataset_config,
            entropy_config,
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
            entropy_config_path = experiment["entropy_config"]
            agent_type = experiment.get("agent_type")

            # Load and merge configurations
            experiment_config = load_experiment_config(
                base_config_path=base_config_path,
                model_config_path=model_config_path,
                dataset_config_path=dataset_config_path,
                entropy_config_path=entropy_config_path,
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


if __name__ == "__main__":
    """Test the configuration loader functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="Test configuration loader")
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["single", "fan", "sequential"],
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
                entropy_config_path="experiments/configs/entropy_configs/standard.yml",
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
