"""Data loader for multi-agent experiment results.

This module provides utilities for loading experiment data,
configurations, ground truths, and entropy tensors from storage.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch


class DataLoader:
    """Loader for experiment data and configurations.

    Provides methods to load ground truths, experiment configs,
    results, and entropy tensors from the file system.
    """

    def __init__(self, base_path: str):
        """Initialize the data loader with base path.

        Args:
            base_path: Base path to the project directory.
        """
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "experiments" / "results" / "raw"
        self.configs_path = self.base_path / "experiments" / "configs_exp"
        self.data_path = self.base_path / "experiments" / "data"

    def load_ground_truth(self, dataset: str) -> Dict[str, Any]:
        """Load ground truth data for a given dataset.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").

        Returns:
            Dictionary mapping main_id to ground truth data.

        Raises:
            FileNotFoundError: If ground truth file is not found.
        """
        dataset_map = {
            "gsm8k": "GSM8K",
            "humaneval": "HumanEval",
            "mmlu": "MMLU",
            "aime2024": "AIME2024",
        }
        dataset_folder = dataset_map.get(dataset.lower(), dataset)
        dataset_path = self.data_path / dataset_folder
        data_files = list(dataset_path.glob("*-all-samples.json"))

        if not data_files:
            raise FileNotFoundError(f"Ground truth file not found in: {dataset_path}")

        data_file = data_files[0]

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {item["main_id"]: item for item in data}

    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment configuration from YAML file.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing experiment configuration.

        Raises:
            FileNotFoundError: If config file is not found.
        """
        config_file = self.configs_path / f"{experiment_name}.yml"

        if not config_file.exists():
            config_file = None
            for f in self.configs_path.glob("*.yml"):
                if experiment_name.startswith(f.stem):
                    config_file = f
                    break

            if config_file is None:
                raise FileNotFoundError(
                    f"Config file not found for experiment: {experiment_name}"
                )

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def get_experiments_by_dataset(self, dataset: str) -> List[str]:
        """Get list of experiment names for a given dataset.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").

        Returns:
            Sorted list of experiment names.
        """
        dataset_path = self.results_path / dataset.lower()

        if not dataset_path.exists():
            return []

        experiments = []
        for exp_dir in dataset_path.iterdir():
            if exp_dir.is_dir():
                experiments.append(exp_dir.name)

        return sorted(experiments)

    def load_result_store_info(
        self, dataset: str, experiment_name: str
    ) -> Dict[str, Any]:
        """Load result store information for an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing result store information.

        Raises:
            FileNotFoundError: If result store info is not found.
        """
        traces_path = self.results_path / dataset.lower() / experiment_name / "traces"
        info_file = traces_path / "Result-store-information.json"

        if not info_file.exists():
            raise FileNotFoundError(f"Result store info not found: {info_file}")

        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        return info

    def load_result_block(
        self, dataset: str, experiment_name: str, block_name: str
    ) -> Dict[str, Any]:
        """Load a specific result block for an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment.
            block_name: Name of the result block.

        Returns:
            Dictionary containing result block data.

        Raises:
            FileNotFoundError: If result block is not found.
        """
        traces_path = self.results_path / dataset.lower() / experiment_name / "traces"
        block_file = traces_path / block_name

        if not block_file.exists():
            raise FileNotFoundError(f"Result block not found: {block_file}")

        with open(block_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def load_entropy_tensor(
        self, dataset: str, experiment_name: str, result_id: str
    ) -> Optional[torch.Tensor]:
        """Load entropy tensor for a specific result.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment.
            result_id: ID of the result.

        Returns:
            Entropy tensor or None if not found.
        """
        traces_path = self.results_path / dataset.lower() / experiment_name / "traces"
        tensor_path = traces_path / "tensors" / f"{result_id}_extras_entropy.pt"

        if not tensor_path.exists():
            return None

        return torch.load(tensor_path)

    def load_all_results(self, dataset: str, experiment_name: str) -> Dict[str, Any]:
        """Load all results for an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing all experiment results.
        """
        info = self.load_result_store_info(dataset, experiment_name)
        all_results = {}

        for block_name, block_info in info.items():
            block_data = self.load_result_block(dataset, experiment_name, block_name)
            all_results.update(block_data)

        return all_results

    def parse_result_id(self, result_id: str) -> Dict[str, Any]:
        """Parse result ID to extract components.

        Args:
            result_id: Result ID string to parse.

        Returns:
            Dictionary containing parsed components:
                - main_id: Main sample identifier
                - agent_type: Type of agent
                - execution_order: Order of execution
                - sample_number: Sample number
        """
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
