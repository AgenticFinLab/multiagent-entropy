#!/usr/bin/env python3
"""
FinAgent Experiment Runner - Finance Agent Benchmark Integration.

This script integrates the Finance Agent Benchmark (FinAgent) dataset with the
multiagent-entropy experiment framework. It supports:
- All existing multi-agent architectures (single, debate, centralized, etc.)
- FinAgent's rubric-based evaluation method with real financial tools
- Checkpoint/resume functionality
- Batch experiment processing

Dataset: vals-ai/finance_agent_benchmark
- 537 expert-authored financial research questions
- 9 task categories
- Rubric-based evaluation with 'correctness' and 'contradiction' operators

Usage:
    # Single experiment with multi-agent architecture
    python run_finagent_experiment.py -m model_config.yml -d finagent.yml -n exp_name --agent-type debate

    # Batch experiments
    python run_finagent_experiment.py --batch-config batch_finagent.yml
"""

import argparse
import logging
import os
import time

import yaml

from config_loader import load_experiment_config, save_config
from finagent_experiment import run_finagent_experiment, run_batch_finagent_experiments

# Ensure log directory exists
os.makedirs("experiments/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiments/logs/finagent_experiment_runner.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinAgent (Finance Agent Benchmark) experiments")

    parser.add_argument("-b", "--base-config", type=str, default="experiments/configs/base_config.yml", help="Path to base configuration file")
    parser.add_argument("-m", "--model-config", type=str, help="Path to model-specific configuration file")
    parser.add_argument("-d", "--dataset-config", type=str, default="experiments/configs/dataset_specific/finagent.yml", help="Path to dataset-specific configuration file (default: finagent.yml)")
    parser.add_argument("-n", "--experiment-name", type=str, help="Name of the experiment")
    parser.add_argument(
        "--agent-type", type=str,
        choices=["single", "sequential", "centralized", "decentralized", "full_decentralized", "debate", "hybrid"],
        default="single", help="Type of agent configuration to use",
    )
    parser.add_argument("--batch-config", "-bc", type=str, help="Path to batch configuration file defining multiple experiments")
    parser.add_argument("--dry-run", action="store_true", help="Only prepare configurations without running experiments")
    parser.add_argument("--save-config", default=True, help="Save merged configuration to file")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip FinAgent rubric-based evaluation (only run inference)")

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
        results = run_batch_finagent_experiments(
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

            with open(first_exp.get("dataset_config", "experiments/configs/dataset_specific/finagent.yml"), "r") as f:
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

        result = run_finagent_experiment(
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
                if len(parts) >= 4:
                    for i in range(len(parts) - 3):
                        try:
                            date_part = parts[i]
                            if len(date_part) == 8 and date_part.isdigit():
                                time_part = parts[i + 1]
                                if len(time_part) == 6 and time_part.isdigit():
                                    original_timestamp = f"{date_part}_{time_part}"
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
