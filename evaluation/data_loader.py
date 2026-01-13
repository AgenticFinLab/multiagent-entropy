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
        # Convert base path to Path object for consistent path handling
        self.base_path = Path(base_path)
        # Set path to raw experiment results
        self.results_path = self.base_path / "experiments" / "results" / "raw"
        # Set path to experiment configuration files
        self.configs_path = self.base_path / "experiments" / "configs_exp"
        # Set path to dataset data files
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
        # Map dataset names to their corresponding folder names
        dataset_map = {
            "gsm8k": "GSM8K",
            "humaneval": "HumanEval",
            "mmlu": "MMLU",
            "aime2024": "AIME2024",
            "aime2025": "AIME2025",
            "math500": "Math500",
            "aime2024_8192": "AIME2024",
            "aime2025_8192": "AIME2025",
        }
        # Get the folder name for this dataset
        dataset_folder = dataset_map.get(dataset.lower(), dataset)
        # Construct path to dataset directory
        dataset_path = self.data_path / dataset_folder
        # Find all data files matching the pattern
        data_files = list(dataset_path.glob("*-all-samples.json"))

        # Raise error if no data files found
        if not data_files:
            raise FileNotFoundError(f"Ground truth file not found in: {dataset_path}")

        # Use the first matching data file
        data_file = data_files[0]

        # Load JSON data from file
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle dictionary format data
        if isinstance(data, dict):
            # Get number of samples from main_id list
            num_samples = len(data.get("main_id", []))
            # Initialize dictionary to store ground truth data
            ground_truth_dict = {}
            # Iterate through samples and create ground truth entries
            for i in range(num_samples):
                # Create item dictionary with all fields for this sample
                item = {key: data[key][i] for key in data if isinstance(data[key], list) and i < len(data[key])}
                # Store item in dictionary keyed by main_id
                ground_truth_dict[str(item["main_id"])] = item
            return ground_truth_dict
        # Handle list format data
        else:
            # Create dictionary mapping main_id to item
            return {str(item["main_id"]): item for item in data}

    def load_experiment_config(
        self, dataset: str, experiment_name: str, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load experiment configuration from YAML file.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            experiment_name: Name of the experiment.
            model_name: Model name (e.g., "qwen3_4b"). Optional but helps with config lookup.

        Returns:
            Dictionary containing experiment configuration.

        Raises:
            FileNotFoundError: If config file is not found.
        """
        # Try to load config file from standard location
        config_file = self.configs_path / f"{dataset}" / f"{experiment_name}.yml"

        # If standard location doesn't exist, search for config file
        if not config_file.exists():
            config_file = None
            
            # Initialize list of search paths
            search_paths = []
            
            # Add model-specific path if model name is provided
            if model_name:
                search_paths.append(self.configs_path / f"{dataset}" / model_name)
            
            # Add dataset-specific path
            search_paths.append(self.configs_path / f"{dataset}")
            # Add base configs path
            search_paths.append(self.configs_path)

            # Search through all paths for matching config file
            for search_path in search_paths:
                # Stop if config file was already found
                if config_file is not None:
                    break
                    
                # Check if search path exists and is a directory
                if search_path.exists() and search_path.is_dir():
                    # Iterate through all YAML files in the path
                    for f in search_path.glob("*.yml"):
                        # Check if experiment name starts with file stem
                        if experiment_name.startswith(f.stem):
                            config_file = f
                            break

            # Raise error if no config file was found
            if config_file is None:
                raise FileNotFoundError(
                    f"Config file not found for experiment: {experiment_name}"
                )

        # Load YAML configuration from file
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def get_experiments_by_dataset(self, dataset: str) -> Dict[str, List[str]]:
        """Get list of experiment names for a given dataset, grouped by model.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").

        Returns:
            Dictionary mapping model names to sorted lists of experiment names.
        """
        # Construct path to dataset directory
        dataset_path = self.results_path / dataset.lower()

        # Return empty dictionary if dataset path doesn't exist
        if not dataset_path.exists():
            return {}

        # Initialize dictionary to store experiments by model
        experiments_by_model = {}
        
        # Iterate through model directories
        for model_dir in dataset_path.iterdir():
            # Process only directories
            if model_dir.is_dir():
                # Get model name from directory name
                model_name = model_dir.name
                # Initialize list for this model's experiments
                experiments = []
                # Iterate through experiment directories
                for exp_dir in model_dir.iterdir():
                    # Process only directories
                    if exp_dir.is_dir():
                        # Add experiment name to list
                        experiments.append(exp_dir.name)
                # Store sorted list of experiments for this model
                if experiments:
                    experiments_by_model[model_name] = sorted(experiments)

        return experiments_by_model

    def load_result_store_info(
        self, dataset: str, model_name: str, experiment_name: str
    ) -> Dict[str, Any]:
        """Load result store information for an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing result store information.

        Raises:
            FileNotFoundError: If result store info is not found.
        """
        # Construct path to traces directory
        traces_path = self.results_path / dataset.lower() / model_name / experiment_name / "traces"
        # Construct path to result store information file
        info_file = traces_path / "Result-store-information.json"

        # Raise error if info file doesn't exist
        if not info_file.exists():
            raise FileNotFoundError(f"Result store info not found: {info_file}")

        # Load JSON data from info file
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        return info

    def load_result_block(
        self, dataset: str, model_name: str, experiment_name: str, block_name: str
    ) -> Dict[str, Any]:
        """Load a specific result block for an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.
            block_name: Name of the result block.

        Returns:
            Dictionary containing result block data.

        Raises:
            FileNotFoundError: If result block is not found.
        """
        # Construct path to traces directory
        traces_path = self.results_path / dataset.lower() / model_name / experiment_name / "traces"
        # Construct path to result block file
        block_file = traces_path / block_name

        # Raise error if block file doesn't exist
        if not block_file.exists():
            raise FileNotFoundError(f"Result block not found: {block_file}")

        # Load JSON data from block file
        with open(block_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def load_entropy_tensor(
        self, dataset: str, model_name: str, experiment_name: str, result_id: str
    ) -> Optional[torch.Tensor]:
        """Load entropy tensor for a specific result.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.
            result_id: ID of the result.

        Returns:
            Entropy tensor or None if not found.
        """
        # Construct path to traces directory
        traces_path = self.results_path / dataset.lower() / model_name / experiment_name / "traces"
        # Construct path to entropy tensor file
        tensor_path = traces_path / "tensors" / f"{result_id}_extras_entropy.pt"

        # Return None if tensor file doesn't exist
        if not tensor_path.exists():
            return None

        # Load and return the entropy tensor
        return torch.load(tensor_path)

    def load_all_results(self, dataset: str, model_name: str, experiment_name: str) -> Dict[str, Any]:
        """Load all results for an experiment.

        Args:
            dataset: Dataset name (e.g., "gsm8k", "humaneval").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.

        Returns:
            Dictionary containing all experiment results.
        """
        # Load result store information to get block names
        info = self.load_result_store_info(dataset, model_name, experiment_name)
        # Initialize dictionary to store all results
        all_results = {}

        # Iterate through all result blocks
        for block_name, block_info in info.items():
            # Load data for each block
            block_data = self.load_result_block(dataset, model_name, experiment_name, block_name)
            # Merge block data into all results dictionary
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
        # Remove "Result_" prefix and split by hyphen
        parts = result_id.replace("Result_", "").split("-")

        # Extract main_id from first part
        main_id = parts[0]
        # Extract agent_type from second part
        agent_type = parts[1]
        # Extract execution order and sample number from third part
        execution_order_sample = parts[2]
        # Parse execution order (first number before underscore)
        execution_order = int(execution_order_sample.split("_")[0])
        # Parse sample number (second number after underscore)
        sample_number = int(execution_order_sample.split("_")[2])

        # Return dictionary with all parsed components
        return {
            "main_id": main_id,
            "agent_type": agent_type,
            "execution_order": execution_order,
            "sample_number": sample_number,
        }
