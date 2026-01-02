"""Data loader for groundtruth from dataset files."""

import json
import os
from typing import Dict, Optional
from pathlib import Path


class GroundtruthLoader:
    """Load groundtruth from dataset files."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize groundtruth loader.
        
        Args:
            data_dir: Directory containing dataset files (default: experiments/data)
        """
        self.data_dir = Path(data_dir) if data_dir is not None else Path("experiments/data")
        self._cache: Dict[str, Dict[str, str]] = {}
    
    def _get_data_file_path(self, task_type: str) -> Optional[Path]:
        """Get the data file path for a given task type.
        
        Args:
            task_type: Task type (math, code, option)
            
        Returns:
            Path to the data file or None if not found
        """
        task_to_file = {
            "math": "GSM8K/train-all-samples.json",
            "code": "HumanEval/test-all-samples.json",
            "option": "MMLU/test-all-samples.json",
        }
        
        if task_type not in task_to_file:
            return None
        
        file_path = self.data_dir / task_to_file[task_type]
        return file_path if file_path.exists() else None
    
    def _load_data_file(self, task_type: str) -> Dict[str, str]:
        """Load groundtruth data from file.
        
        Args:
            task_type: Task type (math, code, option)
            
        Returns:
            Dictionary mapping main_id to groundtruth
        """
        if task_type in self._cache:
            return self._cache[task_type]
        
        file_path = self._get_data_file_path(task_type)
        if file_path is None:
            self._cache[task_type] = {}
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            groundtruth_map = {}
            for sample in data:
                main_id = sample.get('main_id')
                groundtruth = sample.get('groundtruth')
                if main_id is not None and groundtruth is not None:
                    groundtruth_map[str(main_id)] = str(groundtruth)
            
            self._cache[task_type] = groundtruth_map
            return groundtruth_map
        except Exception as e:
            print(f"Warning: Failed to load data file {file_path}: {e}")
            self._cache[task_type] = {}
            return {}
    
    def get_groundtruth(self, task_type: str, sample_id: str) -> Optional[str]:
        """Get groundtruth for a sample.
        
        Args:
            task_type: Task type (math, code, option)
            sample_id: Sample ID (e.g., "Result_ID1-SingleSolver-1_sample_0")
            
        Returns:
            Groundtruth string or None if not found
        """
        groundtruth_map = self._load_data_file(task_type)
        
        main_id = self._extract_main_id(sample_id)
        if main_id is None:
            return None
        
        return groundtruth_map.get(main_id)
    
    def _extract_main_id(self, sample_id: str) -> Optional[str]:
        """Extract main_id from sample ID.
        
        Args:
            sample_id: Sample ID (e.g., "Result_ID1-SingleSolver-1_sample_0")
            
        Returns:
            main_id string or None if not found
        """
        if sample_id.startswith("Result_"):
            parts = sample_id[len("Result_"):].split("-")
            if parts:
                return parts[0]
        
        return None
    
    def get_all_groundtruths(self, task_type: str) -> Dict[str, str]:
        """Get all groundtruths for a task type.
        
        Args:
            task_type: Task type (math, code, option)
            
        Returns:
            Dictionary mapping main_id to groundtruth
        """
        return self._load_data_file(task_type)
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
