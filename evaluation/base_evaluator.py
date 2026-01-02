"""
Base evaluator class for task evaluation.

This module provides the abstract base class for all task-specific evaluators.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import re
from pathlib import Path

from .groundtruth_loader import GroundtruthLoader


logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Abstract base class for task evaluators."""

    def __init__(self, task_type: str, data_dir: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            task_type: Type of task (math, code, option)
            data_dir: Directory containing dataset files (default: experiments/data)
        """
        self.task_type = task_type
        self.groundtruth_loader = GroundtruthLoader(data_dir)

    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """
        Extract the final answer from the model response.

        Args:
            response: The model's response text

        Returns:
            The extracted answer string
        """
        pass

    @abstractmethod
    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted answer with ground truth.

        Args:
            predicted: The predicted answer
            ground_truth: The ground truth answer

        Returns:
            True if answers match, False otherwise
        """
        pass

    def evaluate_sample(
        self, response: str, ground_truth: str, sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        Args:
            response: The model's response
            ground_truth: The ground truth answer
            sample_id: Optional sample ID for tracking

        Returns:
            Dictionary containing evaluation results
        """
        predicted_answer = self.extract_answer(response)
        is_correct = self.compare_answers(predicted_answer, ground_truth)

        result = {
            "predicted": predicted_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
        }

        if sample_id is not None:
            result["sample_id"] = sample_id

        return result

    def evaluate_batch(
        self, responses: List[str], ground_truths: List[str], sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of samples.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth answers
            sample_ids: Optional list of sample IDs

        Returns:
            Dictionary containing batch evaluation results
        """
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Number of responses ({len(responses)}) "
                f"does not match number of ground truths ({len(ground_truths)})"
            )

        if sample_ids is not None and len(sample_ids) != len(responses):
            raise ValueError(
                f"Number of sample_ids ({len(sample_ids)}) "
                f"does not match number of responses ({len(responses)})"
            )

        results = []
        correct_count = 0

        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            result = self.evaluate_sample(response, ground_truth)
            if sample_ids is not None and i < len(sample_ids):
                result["sample_id"] = sample_ids[i]
            results.append(result)
            if result["is_correct"]:
                correct_count += 1

        accuracy = correct_count / len(responses) if responses else 0.0

        return {
            "task_type": self.task_type,
            "total_samples": len(responses),
            "correct_samples": correct_count,
            "accuracy": accuracy,
            "sample_results": results,
        }

    def load_experiment_data(self, experiment_dir: str) -> Dict[str, Any]:
        """
        Load experiment data from directory.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary containing experiment data
        """
        data = {"experiment_dir": experiment_dir}

        traces_dir = os.path.join(experiment_dir, "traces")
        if not os.path.exists(traces_dir):
            logger.warning(f"Traces directory not found: {traces_dir}")
            return data

        combined_block_path = os.path.join(traces_dir, "Combined_block_0.json")
        if os.path.exists(combined_block_path):
            with open(combined_block_path, "r", encoding="utf-8") as f:
                combined_data = json.load(f)
                data["combined_results"] = combined_data

        result_block_path = os.path.join(traces_dir, "Result_block_0.json")
        if os.path.exists(result_block_path):
            with open(result_block_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                data["result_data"] = result_data

        return data

    def _detect_agent_type(self, experiment_dir: str) -> str:
        """
        Detect the agent type from experiment directory name.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Agent type (sequential, centralized, decentralized, full_decentralized, hybrid, debate, single_agent)
        """
        dir_name = os.path.basename(experiment_dir).lower()
        
        if "sequential" in dir_name:
            return "sequential"
        elif "centralized" in dir_name:
            return "centralized"
        elif "decentralized" in dir_name:
            if "full_decentralized" in dir_name:
                return "full_decentralized"
            return "decentralized"
        elif "hybrid" in dir_name:
            return "hybrid"
        elif "debate" in dir_name:
            return "debate"
        elif "single" in dir_name:
            return "single_agent"
        else:
            return "single_agent"
    
    def _extract_answer_for_agent_type(
        self, agent_outputs: List[str], agent_type: str
    ) -> str:
        """
        Extract the appropriate answer based on agent type.
        
        Args:
            agent_outputs: List of agent outputs
            agent_type: Type of agent architecture
            
        Returns:
            The extracted answer
        """
        if not agent_outputs:
            return ""
        
        if agent_type == "sequential":
            return agent_outputs[-1] if agent_outputs else ""
        elif agent_type in ["centralized", "decentralized", "full_decentralized", "hybrid"]:
            return agent_outputs[-1] if agent_outputs else ""
        elif agent_type == "debate":
            last_output = agent_outputs[-1] if agent_outputs else ""
            try:
                import json
                parsed = json.loads(last_output)
                if isinstance(parsed, dict) and "final_answer" in parsed:
                    return str(parsed["final_answer"])
            except (json.JSONDecodeError, TypeError):
                pass
            return last_output
        else:
            return agent_outputs[-1] if agent_outputs else ""

    def extract_responses_and_ground_truths(
        self, experiment_data: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Extract responses and ground truths from experiment data.

        Args:
            experiment_data: Dictionary containing experiment data

        Returns:
            Tuple of (responses list, ground_truths list, sample_ids list)
        """
        responses = []
        ground_truths = []
        sample_ids = []

        if "result_data" not in experiment_data:
            logger.warning("No result_data found in experiment data")
            return responses, ground_truths, sample_ids

        result_data = experiment_data["result_data"]
        agent_type = self._detect_agent_type(experiment_data.get("experiment_dir", ""))

        if "combined_results" not in experiment_data:
            logger.warning("No combined results found in experiment data")
            return responses, ground_truths, sample_ids

        combined_results = experiment_data["combined_results"]
        if "Combined_FinalState" not in combined_results:
            logger.warning("Combined_FinalState not found")
            return responses, ground_truths, sample_ids

        final_state = combined_results["Combined_FinalState"]
        if "agent_results" not in final_state:
            logger.warning("agent_results not found in final state")
            return responses, ground_truths, sample_ids

        agent_results = final_state["agent_results"]

        if agent_type == "centralized":
            samples = self._group_centralized_samples(result_data)
            for sample_info in samples:
                sample_id = sample_info["sample_id"]
                response = self._extract_centralized_answer(sample_info, agent_results)
                
                sample_ids.append(sample_id)
                responses.append(response)
                
                ground_truth = self.groundtruth_loader.get_groundtruth(
                    self.task_type, sample_id
                )
                ground_truths.append(ground_truth)
        elif agent_type == "sequential":
            samples = self._group_sequential_samples(result_data)
            for sample_info in samples:
                sample_id = sample_info["sample_id"]
                response = self._extract_sequential_answer(sample_info, agent_results)
                
                sample_ids.append(sample_id)
                responses.append(response)
                
                ground_truth = self.groundtruth_loader.get_groundtruth(
                    self.task_type, sample_id
                )
                ground_truths.append(ground_truth)
        elif agent_type == "single_agent":
            samples = self._group_single_agent_samples(result_data)
            for sample_info in samples:
                sample_id = sample_info["sample_id"]
                response = self._extract_single_agent_answer(sample_info, agent_results)
                
                sample_ids.append(sample_id)
                responses.append(response)
                
                ground_truth = self.groundtruth_loader.get_groundtruth(
                    self.task_type, sample_id
                )
                ground_truths.append(ground_truth)
        else:
            logger.warning(f"Unknown agent type: {agent_type}")

        return responses, ground_truths, sample_ids

    def _group_centralized_samples(
        self, result_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Group result data by sample for centralized agent architecture.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            List of sample information dictionaries
        """
        samples = {}
        
        for key in result_data.keys():
            parts = key.split("-")
            if len(parts) >= 2:
                main_id = parts[0].replace("Result_", "")
                agent_name = parts[1]
                
                if main_id not in samples:
                    samples[main_id] = {"main_id": main_id, "agents": {}}
                
                samples[main_id]["agents"][agent_name] = key
        
        result = []
        for main_id, sample_info in samples.items():
            sample_id = f"Result_{main_id}-OrchestratorAgent_sample_0"
            result.append({
                "main_id": main_id,
                "sample_id": sample_id,
                "agents": sample_info["agents"]
            })
        
        return result

    def _group_sequential_samples(
        self, result_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Group result data by sample for sequential agent architecture.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            List of sample information dictionaries
        """
        samples = {}
        
        for key in result_data.keys():
            parts = key.split("-")
            if len(parts) >= 2:
                main_id = parts[0].replace("Result_", "")
                agent_name = parts[1]
                
                if main_id not in samples:
                    samples[main_id] = {"main_id": main_id, "agents": {}}
                
                samples[main_id]["agents"][agent_name] = key
        
        result = []
        for main_id, sample_info in samples.items():
            sample_id = f"Result_{main_id}-judger_sample_0"
            result.append({
                "main_id": main_id,
                "sample_id": sample_id,
                "agents": sample_info["agents"]
            })
        
        return result

    def _group_single_agent_samples(
        self, result_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Group result data by sample for single agent architecture.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            List of sample information dictionaries
        """
        samples = {}
        
        for key in result_data.keys():
            parts = key.split("-")
            if len(parts) >= 2:
                main_id = parts[0].replace("Result_", "")
                agent_name = parts[1]
                
                if main_id not in samples:
                    samples[main_id] = {"main_id": main_id, "agents": {}}
                
                samples[main_id]["agents"][agent_name] = key
        
        result = []
        for main_id, sample_info in samples.items():
            agent_name = list(sample_info["agents"].keys())[0]
            sample_id = f"Result_{main_id}-{agent_name}_sample_0"
            result.append({
                "main_id": main_id,
                "sample_id": sample_id,
                "agents": sample_info["agents"]
            })
        
        return result

    def _extract_centralized_answer(
        self, sample_info: Dict[str, Any], agent_results: List[Dict[str, Any]]
    ) -> str:
        """
        Extract answer for centralized agent architecture.
        
        For centralized, we need to find the OrchestratorAgent output for this sample.
        
        Args:
            sample_info: Sample information dictionary
            agent_results: List of agent results from combined_data
            
        Returns:
            The extracted answer
        """
        try:
            agents = sample_info.get("agents", {})
            orchestrator_key = agents.get("OrchestratorAgent")
            
            if orchestrator_key:
                for result in agent_results:
                    if "OrchestratorAgent" in result:
                        outputs = result["OrchestratorAgent"]
                        if isinstance(outputs, list) and len(outputs) > 0:
                            return outputs[-1]
                        elif isinstance(outputs, str):
                            return outputs
        except Exception as e:
            logger.warning(f"Failed to extract centralized answer: {e}")
        
        return ""

    def _extract_sequential_answer(
        self, sample_info: Dict[str, Any], agent_results: List[Dict[str, Any]]
    ) -> str:
        """
        Extract answer for sequential agent architecture.
        
        For sequential, we need to find the judger agent output for this sample.
        
        Args:
            sample_info: Sample information dictionary
            agent_results: List of agent results from combined_data
            
        Returns:
            The extracted answer
        """
        try:
            agents = sample_info.get("agents", {})
            judger_key = agents.get("judger")
            
            if judger_key:
                for result in agent_results:
                    if "judger" in result:
                        outputs = result["judger"]
                        if isinstance(outputs, list) and len(outputs) > 0:
                            return outputs[-1]
                        elif isinstance(outputs, str):
                            return outputs
        except Exception as e:
            logger.warning(f"Failed to extract sequential answer: {e}")
        
        return ""

    def _extract_single_agent_answer(
        self, sample_info: Dict[str, Any], agent_results: List[Dict[str, Any]]
    ) -> str:
        """
        Extract answer for single agent architecture.
        
        Args:
            sample_info: Sample information dictionary
            agent_results: List of agent results from combined_data
            
        Returns:
            The extracted answer
        """
        try:
            agents = sample_info.get("agents", {})
            if agents:
                agent_name = list(agents.keys())[0]
                agent_key = agents[agent_name]
                
                for result in agent_results:
                    if agent_name in result:
                        outputs = result[agent_name]
                        if isinstance(outputs, list) and len(outputs) > 0:
                            return outputs[-1]
                        elif isinstance(outputs, str):
                            return outputs
        except Exception as e:
            logger.warning(f"Failed to extract single agent answer: {e}")
        
        return ""

    def evaluate_experiment(self, experiment_dir: str) -> Dict[str, Any]:
        """
        Evaluate an entire experiment.

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            Dictionary containing evaluation results
        """
        experiment_data = self.load_experiment_data(experiment_dir)
        responses, ground_truths, sample_ids = self.extract_responses_and_ground_truths(
            experiment_data
        )

        if not responses:
            logger.warning(f"No responses found in experiment: {experiment_dir}")
            return {
                "experiment_dir": experiment_dir,
                "task_type": self.task_type,
                "error": "No responses found",
            }

        valid_samples = 0
        valid_responses = []
        valid_ground_truths = []
        valid_sample_ids = []

        for response, ground_truth, sample_id in zip(responses, ground_truths, sample_ids):
            if ground_truth is not None:
                valid_samples += 1
                valid_responses.append(response)
                valid_ground_truths.append(ground_truth)
                valid_sample_ids.append(sample_id)
            else:
                logger.warning(f"No groundtruth found for sample: {sample_id}")

        if valid_samples == 0:
            logger.warning(
                f"No valid ground truths found in experiment: {experiment_dir}. "
                "Cannot evaluate accuracy."
            )
            return {
                "experiment_dir": experiment_dir,
                "task_type": self.task_type,
                "error": "No valid ground truths found",
            }

        batch_results = self.evaluate_batch(valid_responses, valid_ground_truths)
        batch_results["experiment_dir"] = experiment_dir
        batch_results["sample_ids"] = valid_sample_ids

        return batch_results
