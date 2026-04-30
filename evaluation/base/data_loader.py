"""Abstract base for experiment data loaders.

Holds the path layout and all I/O methods that are independent of which
results tree (`results/raw` vs `results_temp/raw`) the subclass targets.
Subclasses customize storage layout via `_init_paths()`.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml


def _extract_experiment_base(name: str) -> str:
    """Strip trailing `_YYYYMMDD_<more>` segments from an experiment name."""
    match = re.match(r"^(.+?)_\d{8}_\d+", name)
    if match:
        return match.group(1)
    return name


class BaseDataLoader:
    """Abstract loader of experiment configs, ground truth, results, and tensors.

    Subclasses must implement `_init_paths()` to set:
      - self.results_path
      - self.configs_path
      - self.data_path
      - self.results_finagent_path
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self._init_paths()

    # --- subclass hook ---------------------------------------------------

    def _init_paths(self) -> None:
        raise NotImplementedError(
            "BaseDataLoader subclasses must override _init_paths()"
        )

    # --- shared loaders --------------------------------------------------

    def _get_traces_path(
        self, dataset: str, model_name: str, experiment_name: str
    ) -> Path:
        if dataset.lower() == "finagent":
            return (
                self.results_finagent_path
                / "finagent"
                / model_name
                / experiment_name
                / "traces"
            )
        if dataset.lower() == "gaia":
            return (
                self.results_gaia_path
                / "gaia"
                / model_name
                / experiment_name
                / "traces"
            )
        return (
            self.results_path
            / dataset.lower()
            / model_name
            / experiment_name
            / "traces"
        )

    def load_ground_truth(self, dataset: str) -> Dict[str, Any]:
        dataset_map = {
            "gsm8k": "GSM8K",
            "humaneval": "HumanEval",
            "mmlu": "MMLU",
            "aime2024_16384": "AIME2024",
            "aime2025_16384": "AIME2025",
            "math500": "Math500",
            "aime2024_8192": "AIME2024",
            "aime2025_8192": "AIME2025",
        }
        dataset_folder = dataset_map.get(dataset.lower(), dataset)
        dataset_path = self.data_path / dataset_folder
        data_files = list(dataset_path.glob("*-all-samples.json"))
        if not data_files:
            raise FileNotFoundError(f"Ground truth file not found in: {dataset_path}")

        with open(data_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            num_samples = len(data.get("main_id", []))
            ground_truth_dict: Dict[str, Any] = {}
            for i in range(num_samples):
                item = {
                    key: data[key][i]
                    for key in data
                    if isinstance(data[key], list) and i < len(data[key])
                }
                ground_truth_dict[str(item["main_id"])] = item
            return ground_truth_dict
        return {str(item["main_id"]): item for item in data}

    def load_experiment_config(
        self, dataset: str, experiment_name: str, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        config_file = self.configs_path / f"{dataset}" / f"{experiment_name}.yml"

        if not config_file.exists():
            config_file = None
            search_paths: List[Path] = []
            if model_name:
                search_paths.append(self.configs_path / f"{dataset}" / model_name)
            search_paths.append(self.configs_path / f"{dataset}")
            search_paths.append(self.configs_path)

            for search_path in search_paths:
                if config_file is not None:
                    break
                if not (search_path.exists() and search_path.is_dir()):
                    continue
                for f in search_path.glob("*.yml"):
                    if experiment_name.startswith(f.stem):
                        config_file = f
                        break
                    exp_base = _extract_experiment_base(experiment_name)
                    config_base = _extract_experiment_base(f.stem)
                    if exp_base and config_base and exp_base == config_base:
                        config_file = f
                        break

            if config_file is None:
                raise FileNotFoundError(
                    f"Config file not found for experiment: {experiment_name}"
                )

        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_experiments_by_dataset(self, dataset: str) -> Dict[str, List[str]]:
        if dataset.lower() == "finagent":
            dataset_path = self.results_finagent_path / "finagent"
        elif dataset.lower() == "gaia":
            dataset_path = self.results_gaia_path / "gaia"
        else:
            dataset_path = self.results_path / dataset.lower()

        if not dataset_path.exists():
            return {}

        experiments_by_model: Dict[str, List[str]] = {}
        for model_dir in dataset_path.iterdir():
            if not model_dir.is_dir():
                continue
            experiments = [
                exp_dir.name for exp_dir in model_dir.iterdir() if exp_dir.is_dir()
            ]
            if experiments:
                experiments_by_model[model_dir.name] = sorted(experiments)
        return experiments_by_model

    def load_result_store_info(
        self, dataset: str, model_name: str, experiment_name: str
    ) -> Dict[str, Any]:
        traces_path = self._get_traces_path(dataset, model_name, experiment_name)
        info_file = traces_path / "Result-store-information.json"
        if not info_file.exists():
            raise FileNotFoundError(f"Result store info not found: {info_file}")
        with open(info_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_result_block(
        self, dataset: str, model_name: str, experiment_name: str, block_name: str
    ) -> Dict[str, Any]:
        traces_path = self._get_traces_path(dataset, model_name, experiment_name)
        block_file = traces_path / block_name
        if not block_file.exists():
            raise FileNotFoundError(f"Result block not found: {block_file}")
        with open(block_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_entropy_tensor(
        self, dataset: str, model_name: str, experiment_name: str, result_id: str
    ) -> Optional[torch.Tensor]:
        traces_path = self._get_traces_path(dataset, model_name, experiment_name)

        subdir_path = traces_path / "tensors" / result_id / "extras_entropy.pt"
        if subdir_path.exists():
            return torch.load(subdir_path, weights_only=True)

        flat_path = traces_path / "tensors" / f"{result_id}_extras_entropy.pt"
        if flat_path.exists():
            return torch.load(flat_path, weights_only=True)

        return None

    def load_all_results(
        self, dataset: str, model_name: str, experiment_name: str
    ) -> Dict[str, Any]:
        info = self.load_result_store_info(dataset, model_name, experiment_name)
        all_results: Dict[str, Any] = {}
        for block_name in info.keys():
            block_data = self.load_result_block(
                dataset, model_name, experiment_name, block_name
            )
            all_results.update(block_data)
        return all_results

    def parse_result_id(self, result_id: str) -> Dict[str, Any]:
        # Format: Result_<main_id>-<agent_type>-<order>_sample_<num>
        # main_id may be a plain id or a UUID (containing hyphens), so match from the right.
        m = re.match(
            r"^Result_(.+)-([^-]+)-(\d+)_sample_(\d+)$", result_id
        )
        if m:
            main_id = m.group(1)
            agent_type = m.group(2)
            execution_order = int(m.group(3))
            sample_number = int(m.group(4))
        else:
            # Fallback: legacy split (no UUID hyphens)
            parts = result_id.replace("Result_", "").split("-")
            main_id = parts[0]
            agent_type = parts[1]
            execution_order_sample = parts[2]
            execution_order = int(execution_order_sample.split("_")[0])
            sample_number = int(execution_order_sample.split("_")[2])
        return {
            "main_id": main_id,
            "agent_type": agent_type,
            "execution_order": execution_order,
            "sample_number": sample_number,
        }

    def load_step_entropy_tensors(
        self, dataset: str, model_name: str, experiment_name: str, result_id: str
    ) -> List[Tuple[int, torch.Tensor]]:
        traces_path = self._get_traces_path(dataset, model_name, experiment_name)
        tensor_dir = traces_path / "tensors" / result_id
        if not tensor_dir.exists() or not tensor_dir.is_dir():
            return []

        step_tensors: List[Tuple[int, torch.Tensor]] = []
        pattern = re.compile(r"extras_react_steps_(\d+)__entropy\.pt")
        for pt_file in tensor_dir.iterdir():
            match = pattern.match(pt_file.name)
            if match:
                step_idx = int(match.group(1))
                tensor = torch.load(pt_file, weights_only=True)
                step_tensors.append((step_idx, tensor))
        step_tensors.sort(key=lambda x: x[0])
        return step_tensors

    def load_finagent_evaluation_results(
        self, model_name: str, experiment_name: str
    ) -> Dict[str, Any]:
        eval_file = (
            self.results_finagent_path
            / "finagent"
            / model_name
            / experiment_name
            / "finagent_evaluation_results.json"
        )
        if not eval_file.exists():
            raise FileNotFoundError(
                f"FinAgent evaluation results not found: {eval_file}"
            )
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results_by_id: Dict[str, Any] = {}
        for item in data.get("individual_results", []):
            qid = item.get("question_id", "")
            results_by_id[qid] = {
                "question_type": item.get("question_type", ""),
                "evaluation_result": item.get("evaluation_result", False),
                "evaluation_score": item.get("evaluation_score", 0.0),
                "expected_answer": item.get("expected_answer", ""),
            }
        results_by_id["_aggregate"] = data.get("aggregate_metrics", {})
        return results_by_id

    def load_gaia_evaluation_results(
        self, model_name: str, experiment_name: str
    ) -> Dict[str, Any]:
        eval_file = (
            self.results_gaia_path
            / "gaia"
            / model_name
            / experiment_name
            / "gaia_evaluation_results.json"
        )
        if not eval_file.exists():
            raise FileNotFoundError(
                f"GAIA evaluation results not found: {eval_file}"
            )
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results_by_id: Dict[str, Any] = {}
        for item in data.get("individual_results", []):
            qid = item.get("question_id", "")
            results_by_id[qid] = {
                "level": item.get("level", ""),
                "evaluation_result": item.get("evaluation_result", False),
                "evaluation_score": item.get("evaluation_score", 0.0),
                "groundtruth": item.get("groundtruth", ""),
                "generated_answer": item.get("generated_answer", ""),
                "round1_evaluation_result": item.get(
                    "round1_evaluation_result", None
                ),
                "round1_evaluation_score": item.get(
                    "round1_evaluation_score", None
                ),
                "round1_generated_answer": item.get(
                    "round1_generated_answer", ""
                ),
            }
        results_by_id["_aggregate"] = data.get("aggregate_metrics", {})
        return results_by_id
