#!/usr/bin/env python3
"""
GAIA Experiment Runner - GAIA Benchmark Integration.

This script integrates the GAIA benchmark dataset with the multiagent-entropy
experiment framework. It supports:
- All existing multi-agent architectures (single, debate, centralized, etc.)
- Exact-match evaluation with numeric normalization
- Level-based metrics (Level 1 / 2 / 3)
- Checkpoint/resume functionality
- Batch experiment processing

Dataset: gaia-benchmark/GAIA (pre-downloaded to experiments/data/GAIA)
- 165 validation samples
- 3 difficulty levels (1 = easiest, 3 = hardest)
- 38 samples include attached files

Usage:
    # Single experiment
    python run_gaia_experiment.py -m model_config.yml -n exp_name --agent-type single

    # Batch experiments
    python run_gaia_experiment.py --batch-config batch_gaia.yml
"""

import argparse
import logging
import os
import time

import yaml

from config_loader import load_experiment_config, save_config
from gaia_experiment import run_gaia_experiment, run_batch_gaia_experiments

os.makedirs("experiments/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiments/logs/gaia_experiment_runner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GAIA benchmark experiments")

    parser.add_argument("-b", "--base-config", type=str, default="experiments/configs/base_config.yml")
    parser.add_argument("-m", "--model-config", type=str, help="Path to model-specific configuration file")
    parser.add_argument(
        "-d", "--dataset-config", type=str,
        default="experiments/configs/dataset_specific/gaia.yml",
        help="Path to dataset-specific configuration file (default: gaia.yml)",
    )
    parser.add_argument("-n", "--experiment-name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--agent-type", type=str,
        choices=["single", "sequential", "centralized", "decentralized", "full_decentralized", "debate", "hybrid"],
        default="single",
    )
    parser.add_argument("--batch-config", "-bc", type=str, help="Path to batch configuration file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-config", default=True)
    parser.add_argument("--skip-evaluation", action="store_true")

    args = parser.parse_args()

    if not args.batch_config:
        if not args.model_config:
            parser.error("--model-config is required when not using --batch-config")
        if not args.experiment_name:
            parser.error("--experiment-name is required when not using --batch-config")

    return args


def main():
    args = parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    save_folder = base_config.get("save_folder", "experiments/results")
    if save_folder.startswith("/"):
        save_folder = save_folder[1:]

    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs(f"{save_folder}/raw", exist_ok=True)
    os.makedirs(f"{save_folder}/aggregated", exist_ok=True)

    if args.batch_config:
        results = run_batch_gaia_experiments(
            args.batch_config,
            dry_run=args.dry_run,
            save_config_flag=args.save_config,
            save_folder=save_folder,
            skip_evaluation=args.skip_evaluation,
        )

        if results and not args.dry_run:
            with open(args.batch_config, "r", encoding="utf-8") as f:
                batch_config = yaml.safe_load(f)
            first_exp = batch_config.get("experiments", [])[0]

            with open(first_exp.get("dataset_config", "experiments/configs/dataset_specific/gaia.yml"), "r") as f:
                dataset_config = yaml.safe_load(f)
            dataset_name = dataset_config["data"]["data_name"].lower()

            with open(first_exp["model_config"], "r") as f:
                model_config = yaml.safe_load(f)
            model_name = model_config["lm_name"].split("/")[-1].lower().replace("-", "_").replace(".", "_")

            os.makedirs(f"{save_folder}/aggregated/{dataset_name}/{model_name}", exist_ok=True)
            batch_summary_path = (
                f"{save_folder}/aggregated/{dataset_name}/{model_name}/"
                f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.yml"
            )
            with open(batch_summary_path, "w", encoding="utf-8") as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Batch experiment summary saved to: {batch_summary_path}")

    else:
        merged_config = load_experiment_config(
            base_config_path=args.base_config,
            model_config_path=args.model_config,
            dataset_config_path=args.dataset_config,
            experiment_name=args.experiment_name,
            agent_type=args.agent_type,
        )

        if args.save_config:
            dataset_name = merged_config["data"]["data_name"].lower()
            model_name = (
                merged_config["agents"][list(merged_config["agents"].keys())[0]]["lm_name"]
                .split("/")[-1].lower().replace("-", "_").replace(".", "_")
            )
            config_save_path = (
                f"experiments/configs_exp/{dataset_name}/{model_name}/"
                f"{args.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}.yml"
            )
            save_config(merged_config, config_save_path)
            logger.info(f"Saved merged configuration to: {config_save_path}")

        result = run_gaia_experiment(
            merged_config,
            dry_run=args.dry_run,
            save_folder=save_folder,
            skip_evaluation=args.skip_evaluation,
        )

        if not args.dry_run:
            dataset_name = merged_config["data"]["data_name"].lower()
            model_name = (
                merged_config["agents"][list(merged_config["agents"].keys())[0]]["lm_name"]
                .split("/")[-1].lower().replace("-", "_").replace(".", "_")
            )

            save_folder_from_config = merged_config.get("save_folder", "")
            original_timestamp = time.strftime("%Y%m%d_%H%M%S")
            if save_folder_from_config:
                folder_name = os.path.basename(save_folder_from_config)
                parts = folder_name.split("_")
                for i in range(len(parts) - 1):
                    try:
                        if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                            original_timestamp = f"{parts[i]}_{parts[i + 1]}"
                            break
                    except (ValueError, IndexError):
                        continue

            os.makedirs(f"{save_folder}/aggregated/{dataset_name}/{model_name}", exist_ok=True)
            summary_path = (
                f"{save_folder}/aggregated/{dataset_name}/{model_name}/"
                f"{args.experiment_name}_results_{original_timestamp}.yml"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Experiment results saved to: {summary_path}")


if __name__ == "__main__":
    main()
