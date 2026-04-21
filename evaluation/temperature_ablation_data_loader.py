"""Temperature ablation experiment data loader.

This module provides a specialized data loader for temperature ablation
experiments, extending the base DataLoader with temperature-specific
functionality.
"""

import logging
import re
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from .data_loader import DataLoader


# Configure module logger
logger = logging.getLogger(__name__)


class TempDataLoader(DataLoader):
    """Data loader specialized for temperature ablation experiments.

    Extends DataLoader to handle temperature-specific experiment data,
    including parsing temperature values from experiment names and
    grouping experiments by temperature.
    """

    def _init_paths(self) -> None:
        """Set storage layout for temperature ablation experiments."""
        super()._init_paths()
        # Override results to point at the temperature tree
        self.results_path = self.base_path / "experiments" / "results_temp" / "raw"
        # Aggregated results path specific to temperature experiments
        self.aggregated_path = (
            self.base_path / "experiments" / "results_temp" / "aggregated"
        )

    @staticmethod
    def get_temperature_from_experiment_name(experiment_name: str) -> Optional[float]:
        """Parse temperature value from experiment name.

        Extracts temperature from experiment names following the pattern
        't_{integer}_{decimal}_...' (e.g., 't_0_4_qwen3-4b_math500_...' -> 0.4).

        Args:
            experiment_name: Name of the experiment directory.

        Returns:
            Parsed temperature value as float, or None if pattern not matched.
        """
        # Quick check for common temperature prefixes
        if experiment_name.startswith("t_0_4_"):
            return 0.4
        if experiment_name.startswith("t_0_8_"):
            return 0.8

        # General pattern matching for t_{integer}_{decimal}_
        pattern = r"^t_(\d+)_(\d+)_"
        match = re.match(pattern, experiment_name)

        if match:
            integer_part = match.group(1)
            decimal_part = match.group(2)
            return float(f"{integer_part}.{decimal_part}")

        return None

    def get_experiments_by_temperature(
        self, dataset: str
    ) -> Dict[float, Dict[str, List[str]]]:
        """Get experiments grouped by temperature and model.

        Args:
            dataset: Dataset name (e.g., "math500").

        Returns:
            Dictionary mapping temperature to model-experiment mappings:
            {temperature: {model_name: [experiment_names]}}
        """
        # Get all experiments using parent class method
        experiments_by_model = self.get_experiments_by_dataset(dataset)

        # Initialize result dictionary
        experiments_by_temp: Dict[float, Dict[str, List[str]]] = {}

        # Group experiments by temperature
        for model_name, experiment_names in experiments_by_model.items():
            for exp_name in experiment_names:
                # Parse temperature from experiment name
                temperature = self.get_temperature_from_experiment_name(exp_name)

                if temperature is not None:
                    # Initialize temperature group if not exists
                    if temperature not in experiments_by_temp:
                        experiments_by_temp[temperature] = {}

                    # Initialize model list if not exists
                    if model_name not in experiments_by_temp[temperature]:
                        experiments_by_temp[temperature][model_name] = []

                    # Add experiment to the appropriate group
                    experiments_by_temp[temperature][model_name].append(exp_name)

        # Sort experiment lists within each group
        for temp in experiments_by_temp:
            for model_name in experiments_by_temp[temp]:
                experiments_by_temp[temp][model_name].sort()

        return experiments_by_temp

    def get_completed_experiments(self, dataset: str) -> Dict[str, List[str]]:
        """Get completed experiments from aggregated results.

        Reads aggregated YML files and filters for experiments with
        status: completed.

        Args:
            dataset: Dataset name (e.g., "math500").

        Returns:
            Dictionary mapping model names to lists of completed experiment names.
        """
        # Construct path to aggregated results for this dataset
        aggregated_dataset_path = self.aggregated_path / dataset.lower()

        # Return empty dict if path doesn't exist
        if not aggregated_dataset_path.exists():
            logger.warning(f"Aggregated path not found: {aggregated_dataset_path}")
            return {}

        # Initialize result dictionary
        completed_experiments: Dict[str, List[str]] = {}

        # Iterate through model directories
        for model_dir in aggregated_dataset_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            completed_list: List[str] = []

            # Iterate through YML files in model directory
            for yml_file in model_dir.glob("*.yml"):
                try:
                    # Load YML content
                    with open(yml_file, "r", encoding="utf-8") as f:
                        yml_data = yaml.safe_load(f)

                    # Check if experiment is completed
                    if yml_data and yml_data.get("status") == "completed":
                        # Extract experiment directory name from results_path
                        results_path = yml_data.get("results_path", "")
                        if results_path:
                            # Get the last part of the path as experiment name
                            exp_name = Path(results_path).name
                            completed_list.append(exp_name)
                        else:
                            logger.warning(
                                f"No results_path in completed experiment: {yml_file}"
                            )

                except yaml.YAMLError as e:
                    logger.error(f"Failed to parse YML file {yml_file}: {e}")
                except Exception as e:
                    logger.error(f"Error reading {yml_file}: {e}")

            # Store completed experiments for this model
            if completed_list:
                completed_experiments[model_name] = sorted(completed_list)

        return completed_experiments

    def load_experiment_config(
        self, dataset: str, experiment_name: str, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load experiment configuration with fallback for temperature experiments.

        Attempts to load config using parent class logic. If that fails,
        provides a default configuration based on aggregated YML data.

        Args:
            dataset: Dataset name (e.g., "math500").
            experiment_name: Name of the experiment.
            model_name: Model name (e.g., "qwen3_4b"). Optional.

        Returns:
            Dictionary containing experiment configuration.
        """
        # Try parent class method first
        try:
            return super().load_experiment_config(dataset, experiment_name, model_name)
        except FileNotFoundError:
            logger.debug(
                f"Config not found via parent method for {experiment_name}, "
                "attempting fallback"
            )

        # Fallback: try to get agent_type from aggregated YML
        agent_type = self._parse_agent_type_from_aggregated(
            dataset, experiment_name, model_name
        )

        # Return default configuration
        default_config = {
            "agent_type": agent_type,
            "round": 2,
            "lm_name": "Qwen/Qwen3-4B",
            "task_type": "math",
        }

        logger.info(f"Using fallback config for {experiment_name}: {default_config}")

        return default_config

    def load_entropy_tensor(
        self, dataset: str, model_name: str, experiment_name: str, result_id: str
    ) -> Optional["torch.Tensor"]:
        """Load entropy tensor for temperature ablation experiments.

        Temperature experiments store entropy files in subdirectories:
        traces/tensors/{result_id}/extras_entropy.pt
        Unlike standard experiments which use flat files:
        traces/tensors/{result_id}_extras_entropy.pt

        Falls back to standard format if subdirectory format not found.

        Args:
            dataset: Dataset name (e.g., "math500").
            model_name: Model name (e.g., "qwen3_4b").
            experiment_name: Name of the experiment.
            result_id: ID of the result.

        Returns:
            Entropy tensor or None if not found.
        """
        # Construct path to traces directory
        traces_path = (
            self.results_path
            / dataset.lower()
            / model_name
            / experiment_name
            / "traces"
        )

        # Try subdirectory format first (temperature experiment structure)
        tensor_path = traces_path / "tensors" / result_id / "extras_entropy.pt"
        if tensor_path.exists():
            return torch.load(tensor_path)

        # Fall back to standard flat file format
        return super().load_entropy_tensor(
            dataset, model_name, experiment_name, result_id
        )

    def _parse_agent_type_from_aggregated(
        self, dataset: str, experiment_name: str, model_name: Optional[str] = None
    ) -> str:
        """Parse agent type from aggregated YML file.

        Args:
            dataset: Dataset name.
            experiment_name: Name of the experiment.
            model_name: Model name. If None, searches all model directories.

        Returns:
            Agent type string, or "single_agent" as default.
        """
        aggregated_dataset_path = self.aggregated_path / dataset.lower()

        if not aggregated_dataset_path.exists():
            return "single_agent"

        # Determine which model directories to search
        if model_name:
            model_dirs = [aggregated_dataset_path / model_name]
        else:
            model_dirs = [d for d in aggregated_dataset_path.iterdir() if d.is_dir()]

        # Search for matching YML file
        for model_dir in model_dirs:
            if not model_dir.exists() or not model_dir.is_dir():
                continue

            for yml_file in model_dir.glob("*.yml"):
                try:
                    with open(yml_file, "r", encoding="utf-8") as f:
                        yml_data = yaml.safe_load(f)

                    if not yml_data:
                        continue

                    # Check if this YML corresponds to the experiment
                    results_path = yml_data.get("results_path", "")
                    if results_path and Path(results_path).name == experiment_name:
                        # Found matching YML, extract agent_type
                        return yml_data.get("agent_type", "single_agent")

                    # Also check if experiment name starts with yml file stem
                    if experiment_name.startswith(yml_file.stem):
                        return yml_data.get("agent_type", "single_agent")

                except Exception as e:
                    logger.debug(f"Error reading {yml_file}: {e}")
                    continue

        # Default fallback
        return "single_agent"
