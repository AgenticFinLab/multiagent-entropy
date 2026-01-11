"""Metrics calculator for evaluating multi-agent system performance.

This module provides utilities for calculating various metrics including
accuracy, time cost, and entropy from experiment results.
"""

import re
import torch
from typing import Dict, Any, List, Optional

from math_verify import parse, verify


class MetricsCalculator:
    """Calculator for various evaluation metrics.

    Provides static methods to extract answers, verify correctness,
    and calculate performance metrics from experiment results.
    """

    @staticmethod
    def extract_boxed_answer(text: str) -> tuple[Optional[str], bool]:
        """Extract answer from boxed format in text.

        Args:
            text: Input text containing potential boxed answer.

        Returns:
            Tuple of (extracted_answer_string, format_compliance_bool).
            Returns (None, False) if no valid format found.
        """
        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, text)
        if matches:
            if matches[-1].startswith("{"):
                return matches[-1][1:], True
            else:
                return matches[-1], True

        return None, False

    @staticmethod
    def has_valid_format(text: str) -> bool:
        """Check if text contains valid boxed answer format.

        Args:
            text: Input text to check.

        Returns:
            True if valid format found, False otherwise.
        """
        _, format_compliance = MetricsCalculator.extract_boxed_answer(text)
        return format_compliance

    @staticmethod
    def extract_code_answer(text: str) -> tuple[Optional[str], bool]:
        """Extract Python code block from text.

        Args:
            text: Input text containing potential code block.

        Returns:
            Tuple of (extracted_code_string, format_compliance_bool).
            Returns (None, False) if no valid format found.
        """
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1], True
        return None, False

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer string by stripping and collapsing whitespace.

        Args:
            answer: Raw answer string.

        Returns:
            Normalized answer string.
        """
        answer = answer.strip()
        answer = re.sub(r"\s+", " ", answer)
        return answer

    @staticmethod
    def is_single_uppercase_letter(text: str) -> bool:
        """Check if text is a single uppercase letter.

        Args:
            text: Text to check.

        Returns:
            True if single uppercase letter, False otherwise.
        """
        text = text.strip()
        return len(text) == 1 and text.isupper()

    @staticmethod
    def is_answer_correct(predicted: Optional[str], ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth.

        Args:
            predicted: Predicted answer string.
            ground_truth: Ground truth answer string.

        Returns:
            True if answers match, False otherwise.
        """
        if predicted is None:
            return False

        predicted_norm = MetricsCalculator.normalize_answer(predicted)
        ground_truth_norm = MetricsCalculator.normalize_answer(ground_truth)

        if MetricsCalculator.is_single_uppercase_letter(ground_truth_norm):
            return predicted_norm == ground_truth_norm

        try:
            gold_parsed = parse(ground_truth_norm)
            answer_parsed = parse(predicted_norm)

            if gold_parsed is None or answer_parsed is None:
                return False

            is_correct = verify(gold_parsed, answer_parsed)
            return is_correct
        except Exception as e:
            return False

    @staticmethod
    def is_code_correct(
        predicted_code: Optional[str],
        ground_truth_code: str,
        test_cases: Optional[str] = None,
    ) -> bool:
        """Check if predicted code passes all test cases.

        Args:
            predicted_code: Predicted Python code string.
            ground_truth_code: Ground truth code (for reference).
            test_cases: Test cases to validate the code against.

        Returns:
            True if code passes all tests, False otherwise.
        """
        if not predicted_code:
            return False

        if not test_cases:
            return False

        try:
            local_namespace = {}
            exec(predicted_code, {}, local_namespace)

            test_namespace = local_namespace.copy()
            exec(test_cases, {}, test_namespace)

            if 'check' in test_namespace:
                for func_name, func in local_namespace.items():
                    if callable(func) and not func_name.startswith('_'):
                        try:
                            test_namespace['check'](func)
                        except AssertionError:
                            return False
                        except Exception:
                            continue

            return True
        except Exception as e:
            return False

    @staticmethod
    def is_answer_correct_by_task_type(
        predicted: Optional[str],
        ground_truth: str,
        task_type: str = "math",
        test_cases: Optional[str] = None,
    ) -> bool:
        """Check if predicted answer matches ground truth based on task type.

        Args:
            predicted: Predicted answer string.
            ground_truth: Ground truth answer string.
            task_type: Type of task ("math", "code", "option").
            test_cases: Test cases for code tasks.

        Returns:
            True if answers match, False otherwise.
        """
        if task_type == "code":
            return MetricsCalculator.is_code_correct(predicted, ground_truth, test_cases)
        else:
            return MetricsCalculator.is_answer_correct(predicted, ground_truth)

    @staticmethod
    def calculate_time_cost(result: Dict[str, Any]) -> float:
        """Extract time cost from result dictionary.

        Args:
            result: Result dictionary containing cost information.

        Returns:
            Time cost value or 0.0 if not found.
        """
        if "cost" in result and "time" in result["cost"]:
            return result["cost"]["time"]
        return 0.0

    @staticmethod
    def calculate_average_entropy(entropy_tensor: torch.Tensor) -> float:
        """Calculate average entropy from tensor.

        Args:
            entropy_tensor: Tensor containing entropy values.

        Returns:
            Average entropy value or 0.0 if tensor is None.
        """
        if entropy_tensor is None:
            return 0.0

        if isinstance(entropy_tensor, torch.Tensor):
            return float(entropy_tensor.mean().item())

        return 0.0

    @staticmethod
    def _calculate_agent_accuracy_base(
        results: Dict[str, Any],
        ground_truths: Dict[str, Any],
        sample_ids: List[str],
        agent_filter: Optional[List[str]] = None,
        use_final_answer: bool = False,
        key_suffix: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calculate agent accuracy base implementation.

        Args:
            results: Dictionary of experiment results.
            ground_truths: Dictionary of ground truth answers.
            sample_ids: List of sample IDs to analyze.
            agent_filter: Optional list of agent types to filter.
            use_final_answer: Whether to use final answer field.
            key_suffix: Optional suffix for result keys.

        Returns:
            Dictionary mapping agent keys to accuracy values.
        """
        accuracies = {}

        for sample_id in sample_ids:
            result = results.get(sample_id)
            if not result:
                continue

            parsed_id = sample_id.replace("Result_", "").split("-")
            main_id = parsed_id[0]
            agent_type = parsed_id[1]

            ground_truth = ground_truths.get(main_id)
            if not ground_truth:
                continue

            if use_final_answer and "final_answer" in result:
                predicted = result["final_answer"]
            else:
                predicted, _ = MetricsCalculator.extract_boxed_answer(
                    result.get("response", "")
                )

            is_correct = MetricsCalculator.is_answer_correct(
                predicted, ground_truth["groundtruth"]
            )

            if agent_filter is None or agent_type in agent_filter:
                key = f"{main_id}_{key_suffix if key_suffix else agent_type}"
                if key not in accuracies:
                    accuracies[key] = []
                accuracies[key].append(is_correct)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in accuracies.items()}

    @staticmethod
    def get_agent_accuracy_for_single(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy for single agent architecture.

        Args:
            results: Dictionary of experiment results.
            ground_truths: Dictionary of ground truth answers.
            sample_ids: List of sample IDs to analyze.

        Returns:
            Dictionary mapping agent keys to accuracy values.
        """
        return MetricsCalculator._calculate_agent_accuracy_base(
            results, ground_truths, sample_ids
        )

    @staticmethod
    def get_agent_accuracy_for_sequential(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy for sequential agent architecture.

        Args:
            results: Dictionary of experiment results.
            ground_truths: Dictionary of ground truth answers.
            sample_ids: List of sample IDs to analyze.

        Returns:
            Dictionary mapping agent keys to accuracy values.
        """
        return MetricsCalculator._calculate_agent_accuracy_base(
            results, ground_truths, sample_ids, agent_filter=["judger"]
        )

    @staticmethod
    def get_agent_accuracy_for_centralized(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy for centralized agent architecture.

        Args:
            results: Dictionary of experiment results.
            ground_truths: Dictionary of ground truth answers.
            sample_ids: List of sample IDs to analyze.

        Returns:
            Dictionary mapping agent keys to accuracy values.
        """
        return MetricsCalculator._calculate_agent_accuracy_base(
            results, ground_truths, sample_ids, agent_filter=["OrchestratorAgent"]
        )

    @staticmethod
    def get_agent_accuracy_for_debate(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy for debate agent architecture.

        Args:
            results: Dictionary of experiment results.
            ground_truths: Dictionary of ground truth answers.
            sample_ids: List of sample IDs to analyze.

        Returns:
            Dictionary mapping agent keys to accuracy values.
        """
        return MetricsCalculator._calculate_agent_accuracy_base(
            results,
            ground_truths,
            sample_ids,
            agent_filter=["OrchestratorAgent"],
            use_final_answer=True,
            key_suffix="final",
        )

    @staticmethod
    def get_agent_accuracy_for_hybrid(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
        """Calculate accuracy for hybrid agent architecture.

        Args:
            results: Dictionary of experiment results.
            ground_truths: Dictionary of ground truth answers.
            sample_ids: List of sample IDs to analyze.

        Returns:
            Dictionary mapping agent keys to accuracy values.
        """
        return MetricsCalculator._calculate_agent_accuracy_base(
            results, ground_truths, sample_ids, agent_filter=["OrchestratorAgent"]
        )

    @staticmethod
    def calculate_agent_accuracy(
        agent_architecture: str,
        results: Dict[str, Any],
        ground_truths: Dict[str, Any],
        sample_ids: List[str],
    ) -> Dict[str, float]:
        """Calculate agent accuracy based on architecture type.

        Args:
            agent_architecture: Type of agent architecture.
            results: Dictionary of experiment results.
            ground_truths: Dictionary of ground truth answers.
            sample_ids: List of sample IDs to analyze.

        Returns:
            Dictionary mapping agent keys to accuracy values.
        """
        architecture_map = {
            "single": MetricsCalculator.get_agent_accuracy_for_single,
            "sequential": MetricsCalculator.get_agent_accuracy_for_sequential,
            "centralized": MetricsCalculator.get_agent_accuracy_for_centralized,
            "debate": MetricsCalculator.get_agent_accuracy_for_debate,
            "hybrid": MetricsCalculator.get_agent_accuracy_for_hybrid,
        }

        calculator = architecture_map.get(agent_architecture.lower())
        if calculator:
            return calculator(results, ground_truths, sample_ids)

        return {}
