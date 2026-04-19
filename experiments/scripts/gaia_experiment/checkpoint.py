import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


def find_existing_experiment_dir(
    save_folder: str, dataset_name: str, model_name: str, experiment_name: str
) -> str:
    """Find the most recent existing experiment directory matching the given prefix."""
    search_pattern = f"{save_folder}/raw/{dataset_name}/{model_name}/{experiment_name}_*"
    matching_dirs = glob.glob(search_pattern)
    matching_dirs = [d for d in matching_dirs if os.path.isdir(d)]
    if not matching_dirs:
        return ""
    matching_dirs.sort(reverse=True)
    return matching_dirs[0]


def get_completed_batches(experiment_dir: str) -> int:
    """Return the number of completed batches recorded in the experiment directory."""
    batch_info_path = os.path.join(experiment_dir, "traces", "Batch-store-information.json")
    if not os.path.exists(batch_info_path):
        return 0
    try:
        with open(batch_info_path, "r", encoding="utf-8") as f:
            batch_info = json.load(f)
        total_count = 0
        for block_data in batch_info.values():
            if isinstance(block_data, dict) and "count" in block_data:
                total_count += block_data["count"]
        return total_count
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to read batch info from {batch_info_path}: {e}")
        return 0
