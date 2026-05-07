"""
Wrapper around experiments/scripts/run_experiment.py that:
  1. Overrides save_folder → experiments/results_entropy_judger/raw/
  2. Supports --data-num to cap sample count
  3. Implements checkpoint/resume: finds the most recent existing directory
     for this experiment name, counts completed batches, and skips them.

Usage (called by run.sh):
    python entropy_judger/run_single.py \
        -b experiments/configs/base_config.yml \
        -m experiments/configs/model_specific/qwen3-4b.yml \
        -d experiments/configs/dataset_specific/gsm8k.yml \
        -n run3_gsm8k_qwen3-4b_single \
        --agent-type single [--data-num 100]
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "experiments" / "scripts"))

from config_loader import load_experiment_config

from lmbase.dataset import registry as data_registry
from maep.language.single import SingleAgent
from maep.language.sequential import SequentialAgents
from maep.language.centralized import OrchestratorCentralized
from maep.language.decentralized import OrchestratorDecentralized
from maep.language.full_decentralized import OrchestratorFullDecentralized
from maep.language.debate import DebateMAS
from maep.language.hybrid import OrchestratorHybrid
from config_loader import map_aime25_subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RESULTS_ROOT = "experiments/results_entropy_judger/raw"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_existing_dir(results_root: str, dataset_name: str, model_name: str, exp_name: str) -> str:
    """Return the most recent directory matching exp_name_* (ignoring timestamp suffix)."""
    pattern = f"{results_root}/{dataset_name}/{model_name}/{exp_name}_*"
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if not dirs:
        return ""
    dirs.sort(reverse=True)  # lexicographic desc ≈ newest timestamp first
    return dirs[0]


def _count_completed_batches(exp_dir: str) -> int:
    """Count completed batches from Batch-store-information.json."""
    info_path = os.path.join(exp_dir, "traces", "Batch-store-information.json")
    if not os.path.exists(info_path):
        return 0
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        return sum(
            block["count"]
            for block in info.values()
            if isinstance(block, dict) and "count" in block
        )
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not read batch info from {info_path}: {e}")
        return 0


# ---------------------------------------------------------------------------
# Dataset loading — local files take priority over HuggingFace
# ---------------------------------------------------------------------------

def _load_dataset(config: dict) -> object:
    """
    Load dataset, preferring local JSON files in data_path over HuggingFace.

    Looks for {data_path}/{split}-all-samples.json first.  This file is
    written by run_experiment.py on every standard run, so it will exist
    after any previous experiment on the same dataset.  Falling back to
    data_registry.get() only when the file is absent.
    """
    data_cfg = config["data"]
    split = data_cfg["split"]
    local_path = os.path.join(data_cfg["data_path"], f"{split}-all-samples.json")

    if os.path.exists(local_path):
        logger.info(f"Loading dataset from local file: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Convert list-of-dicts → dict-of-lists (consistent with HF Dataset slicing)
        if isinstance(raw, list) and raw:
            return {k: [s[k] for s in raw] for k in raw[0].keys()}
        return raw

    # AIME2025 subset mapping before hitting HuggingFace
    if data_cfg.get("data_name", "").lower() == "aime2025":
        orig = data_cfg.get("subset", "")
        mapped = map_aime25_subset(orig)
        if mapped != orig:
            logger.info(f"Mapped AIME2025 subset '{orig}' → '{mapped}'")
            data_cfg["subset"] = mapped

    logger.info(f"Local file not found, downloading from HuggingFace: {data_cfg['data_name']}")
    return data_registry.get(config=data_cfg, split=split)


# ---------------------------------------------------------------------------
# Resume-aware inference loop (mirrors run_single_experiment internals)
# ---------------------------------------------------------------------------

def _run_with_resume(config: dict, start_batch: int) -> dict:
    """
    Run inference starting from start_batch, appending results into the
    already-existing save_folder. Mirrors the loop in run_single_experiment
    but skips the first start_batch batches.
    """
    experiment_name = config["experiment_name"]
    agent_type = config["agent_type"]
    data_cfg = config["data"]

    # Load dataset first (cheap) to determine total_batches before loading the model
    dataset = _load_dataset(config)

    dataset_len = (
        len(next(iter(dataset.values()))) if isinstance(dataset, dict) else len(dataset)
    )
    total_samples = (
        dataset_len if data_cfg["data_num"] == -1 else min(data_cfg["data_num"], dataset_len)
    )
    batch_size = data_cfg["batch_size"]
    total_batches = (total_samples + batch_size - 1) // batch_size

    if start_batch >= total_batches:
        logger.info(f"All {total_batches} batches already completed, skipping model load.")
        return {"status": "completed", "experiment_name": experiment_name,
                "samples_processed": total_samples, "resumed_from_batch": start_batch}

    # Only load the model if there is actual work to do
    agent_map = {
        "single": SingleAgent,
        "sequential": SequentialAgents,
        "centralized": OrchestratorCentralized,
        "decentralized": OrchestratorDecentralized,
        "full_decentralized": OrchestratorFullDecentralized,
        "debate": DebateMAS,
        "hybrid": OrchestratorHybrid,
    }
    if agent_type not in agent_map:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    agent = agent_map[agent_type](run_config=config)

    logger.info(
        f"Resuming {experiment_name}: skipping {start_batch} completed batches, "
        f"{total_batches - start_batch} remaining."
    )

    all_final_states = []
    start_time = time.time()

    for batch_idx in range(start_batch, total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_num = batch_idx + 1  # keep 1-based numbering consistent with original

        if isinstance(dataset, dict):
            batch_samples = {k: v[start_idx:end_idx] for k, v in dataset.items()}
        else:
            batch_samples = dataset[start_idx:end_idx]

        logger.info(f"Processing batch {batch_num}/{total_batches} (samples {start_idx}-{end_idx-1})")
        result = agent.run(batch_samples)
        final_state = result.final_state

        if "agent_results" in final_state:
            all_final_states.extend(final_state["agent_results"])
        elif "merged_results" in final_state:
            all_final_states.append(final_state["merged_results"])

        agent.store_manager.save(savename=f"Batch_{batch_num}_State", data=final_state)

    if all_final_states:
        agent.store_manager.save(
            savename="Combined_FinalState", data={"agent_results": all_final_states}
        )

    duration = time.time() - start_time
    logger.info(f"{experiment_name} finished in {duration:.2f}s")
    return {
        "status": "completed",
        "experiment_name": experiment_name,
        "samples_processed": total_samples - start_batch * batch_size,
        "resumed_from_batch": start_batch,
        "duration_seconds": duration,
        "results_path": config["save_folder"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Entropy Judger single-run wrapper")
    p.add_argument("-b", "--base-config", default="experiments/configs/base_config.yml")
    p.add_argument("-m", "--model-config", required=True)
    p.add_argument("-d", "--dataset-config", required=True)
    p.add_argument("-n", "--experiment-name", required=True)
    p.add_argument("--agent-type", required=True)
    p.add_argument(
        "--data-num", type=int, default=None,
        help="Number of samples to run (-1 for all, overrides dataset config)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    config = load_experiment_config(
        base_config_path=args.base_config,
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        experiment_name=args.experiment_name,
        agent_type=args.agent_type,
    )

    dataset_name = config.get("data", {}).get("data_name", "unknown").lower()
    model_name = (
        config.get("lm_name", "unknown")
        .split("/")[-1].lower().replace("-", "_").replace(".", "_")
    )

    if args.data_num is not None:
        config.setdefault("data", {})["data_num"] = args.data_num

    # Check for an existing checkpoint directory
    existing_dir = _find_existing_dir(_RESULTS_ROOT, dataset_name, model_name, args.experiment_name)
    completed_batches = _count_completed_batches(existing_dir) if existing_dir else 0

    if existing_dir and completed_batches > 0:
        logger.info(
            f"Checkpoint found: {existing_dir} ({completed_batches} batches done), resuming."
        )
        config["save_folder"] = existing_dir
    else:
        # Fresh run — generate a new timestamped directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        timestamp_ms = int(time.time() * 1000) % 1000
        pid = os.getpid()
        config["save_folder"] = (
            f"{_RESULTS_ROOT}/{dataset_name}/{model_name}/"
            f"{args.experiment_name}_{timestamp}_{timestamp_ms}_{pid}"
        )
        completed_batches = 0

    if args.dry_run:
        print(f"[dry-run] save_folder → {config['save_folder']}")
        print(f"[dry-run] completed_batches = {completed_batches}")
        return

    if completed_batches > 0:
        _run_with_resume(config, start_batch=completed_batches)
    else:
        _run_with_resume(config, start_batch=0)


if __name__ == "__main__":
    main()
