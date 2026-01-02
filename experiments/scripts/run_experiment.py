#!/usr/bin/env python3
"""
Experiment runner script for large-scale batch processing.

This script provides a command-line interface to run experiments with different configurations, supporting both single experiment runs and batch processing of multiple configurations.

It supports multiple agent modes: single, sequential, centralized, decentralized, full_decentralized, debate, hybrid.
"""

import os
import json
import yaml
import time
import logging
import argparse
from typing import Dict, Any, List

from maep.language.single import SingleAgent
from maep.language.sequential import SequentialAgents
from maep.language.centralized import OrchestratorCentralized
from maep.language.decentralized import OrchestratorDecentralized
from maep.language.full_decentralized import OrchestratorFullDecentralized
from maep.language.debate import DebateMAS
from maep.language.hybrid import OrchestratorHybrid
from lmbase.dataset import registry as data_registry
from config_loader import load_experiment_config, save_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiments/logs/experiment_runner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run experiments with specified configurations"
    )

    # Single experiment mode
    parser.add_argument(
        "-b",
        "--base-config",
        type=str,
        default="experiments/configs/base_config.yml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "-m",
        "--model-config",
        type=str,
        help="Path to model-specific configuration file",
    )
    parser.add_argument(
        "-d",
        "--dataset-config",
        type=str,
        help="Path to dataset-specific configuration file",
    )
    parser.add_argument(
        "-e", "--entropy-config", type=str, help="Path to entropy configuration file"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, help="Name of the experiment"
    )
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
        help="Type of agent configuration to use (single/sequential/centralized/decentralized/full_decentralized/debate/hybrid)",
    )

    # Batch experiment mode
    parser.add_argument(
        "--batch-config",
        "-bc",
        type=str,
        help="Path to batch configuration file defining multiple experiments",
    )

    # General options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare configurations without running experiments",
    )
    parser.add_argument(
        "--save-config", default=True, help="Save merged configuration to file"
    )

    args = parser.parse_args()

    # Validate arguments: either provide batch-config OR all single experiment parameters
    if not args.batch_config:
        # Check for required parameters in single experiment mode
        if not args.model_config:
            parser.error("--model-config is required when not using --batch-config")
        if not args.dataset_config:
            parser.error("--dataset-config is required when not using --batch-config")
        if not args.entropy_config:
            parser.error("--entropy-config is required when not using --batch-config")
        if not args.experiment_name:
            parser.error("--experiment-name is required when not using --batch-config")

    return args


def run_single_experiment(
    config: Dict[str, Any], dry_run: bool = False
) -> Dict[str, Any]:
    """Run a single experiment with the given configuration.

    Args:
        config (Dict[str, Any]): Experiment configuration
        dry_run (bool): If True, only prepare but don't run the experiment

    Returns:
        Dict[str, Any]: Experiment results or status
    """
    experiment_name = config.get("experiment_name", "unnamed_experiment")
    agent_type = config.get("agent_type", "single")
    logger.info(f"Starting experiment: {experiment_name} (Agent type: {agent_type})")

    if dry_run:
        logger.info(f"Dry run mode - skipping experiment execution")
        return {
            "status": "dry_run",
            "experiment_name": experiment_name,
            "agent_type": agent_type,
            "config": config,
        }

    try:
        # Initialize agent based on agent_type
        if agent_type == "single":
            agent = SingleAgent(run_config=config)
        elif agent_type == "sequential":
            agent = SequentialAgents(run_config=config)
        elif agent_type == "centralized":
            agent = OrchestratorCentralized(run_config=config)
        elif agent_type == "decentralized":
            agent = OrchestratorDecentralized(run_config=config)
        elif agent_type == "full_decentralized":
            agent = OrchestratorFullDecentralized(run_config=config)
        elif agent_type == "debate":
            agent = DebateMAS(run_config=config)
        elif agent_type == "hybrid":
            agent = OrchestratorHybrid(run_config=config)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        # Load dataset
        data_cfg = config["data"]
        dataset = data_registry.get(config=data_cfg, split=data_cfg["split"])

        # Save all samples to local disk
        data_save_dir = f"experiments/data/{data_cfg['data_name']}"
        os.makedirs(data_save_dir, exist_ok=True)
        dataset_save_path = os.path.join(
            data_save_dir,
            f"{data_cfg['split']}-all-samples.json",
        )
        all_samples = [dataset[i] for i in range(len(dataset))]
        with open(dataset_save_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        # Determine total samples to process
        total_samples = (
            len(dataset)
            if data_cfg["data_num"] == -1
            else min(data_cfg["data_num"], len(dataset))
        )

        batch_size = data_cfg["batch_size"]

        logger.info(f"Processing {total_samples} samples in batches of {batch_size}")

        # Initialize results storage
        all_final_states = []
        start_time = time.time()

        # Process in batches
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_samples = dataset[start_idx:end_idx]
            batch_num = start_idx // batch_size + 1

            logger.info(
                f"Processing batch {batch_num} (samples {start_idx}-{end_idx-1})"
            )

            # Run agent on current batch
            result = agent.run(batch_samples)
            final_state = result.final_state

            # Store batch results
            if "agent_results" in final_state:
                all_final_states.extend(final_state["agent_results"])
            elif "merged_results" in final_state:  # For sequential agents
                all_final_states.append(final_state["merged_results"])

            # Save intermediate batch results
            agent.store_manager.save(
                savename=f"Batch_{batch_num}_State",
                data=final_state,
            )

        # Save combined final state
        if all_final_states:
            combined_state = {"agent_results": all_final_states}
            agent.store_manager.save(
                savename="Combined_FinalState",
                data=combined_state,
            )

        # Calculate experiment duration
        duration = time.time() - start_time

        logger.info(f"Experiment {experiment_name} completed in {duration:.2f} seconds")
        logger.info(
            f"Processed {total_samples} samples across {len(range(0, total_samples, batch_size))} batches"
        )

        return {
            "status": "completed",
            "experiment_name": experiment_name,
            "agent_type": agent_type,
            "samples_processed": total_samples,
            "batches": len(range(0, total_samples, batch_size)),
            "duration_seconds": duration,
            "results_path": config.get("save_folder", ""),
        }

    except Exception as e:
        logger.error(f"Experiment {experiment_name} failed with error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return {
            "status": "failed",
            "experiment_name": experiment_name,
            "agent_type": agent_type,
            "error": str(e),
        }


def run_batch_experiments(
    batch_config_path: str, dry_run: bool = False, save_config_flag: bool = True
) -> List[Dict[str, Any]]:
    """Run multiple experiments defined in a batch configuration file.

    Args:
        batch_config_path (str): Path to batch configuration file
        dry_run (bool): If True, only prepare but don't run experiments
        save_config_flag (bool): If True, save merged configurations to files

    Returns:
        List[Dict[str, Any]]: List of experiment results or statuses
    """
    logger.info(f"Loading batch configuration from: {batch_config_path}")

    with open(batch_config_path, "r", encoding="utf-8") as f:
        batch_config = yaml.safe_load(f)

    experiments = batch_config.get("experiments", [])
    logger.info(f"Found {len(experiments)} experiments in batch configuration")

    results = []
    for exp_idx, exp in enumerate(experiments):
        logger.info(
            f"Processing experiment {exp_idx + 1}/{len(experiments)}: {exp.get('name', 'unnamed')}"
        )

        # Load and merge configuration
        merged_config = load_experiment_config(
            base_config_path=exp.get(
                "base_config", "experiments/configs/base_config.yml"
            ),
            model_config_path=exp["model_config"],
            dataset_config_path=exp["dataset_config"],
            entropy_config_path=exp["entropy_config"],
            experiment_name=exp["name"],
            agent_type=exp.get("agent_type"),
        )

        # Save merged configuration if requested
        if save_config_flag:
            config_save_path = f"experiments/configs_exp/{exp['name']}_{time.strftime('%Y%m%d_%H%M%S')}.yml"
            save_config(merged_config, config_save_path)
            logger.info(f"Saved merged configuration to: {config_save_path}")

        # Run experiment
        result = run_single_experiment(merged_config, dry_run=dry_run)
        results.append(result)

    return results


def main():
    """Main function to run the experiment runner."""
    args = parse_args()

    # Ensure log directory exists
    os.makedirs("experiments/logs", exist_ok=True)

    # Ensure results directories exist
    os.makedirs("experiments/results/raw", exist_ok=True)
    os.makedirs("experiments/results/aggregated", exist_ok=True)

    if args.batch_config:
        # Batch experiment mode
        results = run_batch_experiments(
            args.batch_config, dry_run=args.dry_run, save_config_flag=args.save_config
        )

        # Get dataset name from the first experiment's config
        if results and not args.dry_run:
            with open(args.batch_config, "r", encoding="utf-8") as f:
                batch_config = yaml.safe_load(f)
            first_exp = batch_config.get("experiments", [])[0]
            with open(first_exp["dataset_config"], "r", encoding="utf-8") as f:
                dataset_config = yaml.safe_load(f)
            dataset_name = dataset_config["data"]["data_name"].lower()

            # Create dataset directory if it doesn't exist
            os.makedirs(f"experiments/results/aggregated/{dataset_name}", exist_ok=True)

            # Save batch results summary
            summary_path = f"experiments/results/aggregated/{dataset_name}/batch_results_{time.strftime('%Y%m%d_%H%M%S')}.yml"
            with open(summary_path, "w", encoding="utf-8") as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Batch experiment summary saved to: {summary_path}")

    else:
        # Single experiment mode
        # Load and merge configuration
        merged_config = load_experiment_config(
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            dataset_config_path=args.dataset_config,
            entropy_config_path=args.entropy_config,
            experiment_name=args.experiment_name,
            agent_type=args.agent_type,
        )

        # Save merged configuration if requested
        if args.save_config:
            config_save_path = f"experiments/configs_exp/{args.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}.yml"
            save_config(merged_config, config_save_path)
            logger.info(f"Saved merged configuration to: {config_save_path}")

        # Run experiment
        result = run_single_experiment(merged_config, dry_run=args.dry_run)

        # Save single experiment results summary
        if not args.dry_run:
            # Get dataset name from merged config
            dataset_name = merged_config["data"]["data_name"].lower()

            # Create dataset directory if it doesn't exist
            os.makedirs(f"experiments/results/aggregated/{dataset_name}", exist_ok=True)

            summary_path = f"experiments/results/aggregated/{dataset_name}/{args.experiment_name}_results_{time.strftime('%Y%m%d_%H%M%S')}.yml"
            with open(summary_path, "w", encoding="utf-8") as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Experiment results saved to: {summary_path}")


if __name__ == "__main__":
    main()
