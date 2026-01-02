"""
Option task evaluator.

This module provides evaluation functionality for multiple-choice (option) tasks.
"""

import re
from typing import Dict, Any, Optional
from .base_evaluator import BaseEvaluator


class OptionEvaluator(BaseEvaluator):
    """Evaluator for multiple-choice (option) tasks."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the option evaluator.
        
        Args:
            data_dir: Directory containing dataset files (default: experiments/data)
        """
        super().__init__(task_type="option", data_dir=data_dir)

    def extract_answer(self, response: str) -> str:
        """
        Extract the final answer from the model response.

        For option tasks, answers are typically in \\boxed{} format or
        as a single letter (A, B, C, D, etc.).

        Args:
            response: The model's response text

        Returns:
            The extracted answer string
        """
        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, response)

        if matches:
            answer = matches[-1].strip()
            if len(answer) == 1 and answer.isalpha():
                return answer.upper()
            return answer

        pattern2 = r"\[([^\]]*)\]"
        matches2 = re.findall(pattern2, response)
        if matches2:
            answer = matches2[-1].strip()
            if len(answer) == 1 and answer.isalpha():
                return answer.upper()
            return answer

        pattern3 = r"(?:answer|Answer|ANSWER|option|Option|OPTION)\s*[:=]\s*([A-Da-d])"
        matches3 = re.findall(pattern3, response)
        if matches3:
            return matches3[-1].upper()

        pattern4 = r"\b([A-Da-d])\b"
        matches4 = re.findall(pattern4, response)
        if matches4:
            return matches4[-1].upper()

        lines = response.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and len(line) == 1 and line.isalpha():
                return line.upper()

        return ""

    def normalize_option(self, answer: str) -> str:
        """
        Normalize an option answer.

        Args:
            answer: The answer string

        Returns:
            Normalized option (A, B, C, or D)
        """
        answer = answer.strip().upper()

        if len(answer) == 1 and answer in "ABCD":
            return answer

        mapping = {
            "FIRST": "A",
            "SECOND": "B",
            "THIRD": "C",
            "FOURTH": "D",
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
        }

        if answer in mapping:
            return mapping[answer]

        return ""

    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted answer with ground truth.

        For option tasks, we compare option letters.

        Args:
            predicted: The predicted answer
            ground_truth: The ground truth answer

        Returns:
            True if answers match, False otherwise
        """
        if not predicted or not ground_truth:
            return False

        predicted_norm = self.normalize_option(predicted)
        ground_truth_norm = self.normalize_option(ground_truth)

        return predicted_norm == ground_truth_norm

    def evaluate_sample(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate a single option sample.

        Args:
            response: The model's response
            ground_truth: The ground truth answer

        Returns:
            Dictionary containing evaluation results
        """
        predicted_answer = self.extract_answer(response)
        is_correct = self.compare_answers(predicted_answer, ground_truth)

        predicted_norm = self.normalize_option(predicted_answer)
        ground_truth_norm = self.normalize_option(ground_truth)

        return {
            "predicted": predicted_answer,
            "predicted_normalized": predicted_norm,
            "ground_truth": ground_truth,
            "ground_truth_normalized": ground_truth_norm,
            "is_correct": is_correct,
        }

    def evaluate_confidence(
        self, response: str, correct_option: str
    ) -> Dict[str, Any]:
        """
        Evaluate confidence in the predicted answer.

        Args:
            response: The model's response
            correct_option: The correct option

        Returns:
            Dictionary containing confidence metrics
        """
        predicted = self.extract_answer(response)
        predicted_norm = self.normalize_option(predicted)
        correct_norm = self.normalize_option(correct_option)

        is_correct = predicted_norm == correct_norm

        confidence_indicators = [
            "definitely",
            "certainly",
            "surely",
            "clearly",
            "obviously",
            "undoubtedly",
            "without doubt",
        ]

        response_lower = response.lower()
        has_confidence = any(
            indicator in response_lower for indicator in confidence_indicators
        )

        uncertainty_indicators = [
            "maybe",
            "perhaps",
            "possibly",
            "might be",
            "could be",
            "not sure",
            "uncertain",
        ]

        has_uncertainty = any(
            indicator in response_lower for indicator in uncertainty_indicators
        )

        return {
            "predicted": predicted,
            "predicted_normalized": predicted_norm,
            "correct_normalized": correct_norm,
            "is_correct": is_correct,
            "has_confidence": has_confidence,
            "has_uncertainty": has_uncertainty,
        }
