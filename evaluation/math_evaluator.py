"""
Math task evaluator.

This module provides evaluation functionality for math tasks.
"""

import re
from typing import Dict, Any, Optional

from math_verify import parse, verify

from .base_evaluator import BaseEvaluator


class MathEvaluator(BaseEvaluator):
    """Evaluator for math tasks."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the math evaluator.

        Args:
            data_dir: Directory containing dataset files (default: experiments/data)
        """
        super().__init__(task_type="math", data_dir=data_dir)

    def is_single_uppercase_letter(self, text: str) -> bool:
        """
        Check if the text is a single uppercase letter.

        Args:
            text: The text to check

        Returns:
            True if text is a single uppercase letter, False otherwise
        """
        text = text.strip()
        return len(text) == 1 and text.isupper() and text.isalpha()

    def has_valid_boxed_format(self, response: str) -> bool:
        """
        Check if the response contains a valid \\boxed{} format.

        Args:
            response: The model's response text

        Returns:
            True if response contains a valid \\boxed{} format, False otherwise
        """
        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, response)
        
        if matches:
            last_match = matches[-1].strip()
            return len(last_match) > 0
        
        return False

    def extract_answer(self, response: str) -> str:
        """
        Extract the final answer from the model response.

        For math tasks, answers are typically in \\boxed{} format.

        Args:
            response: The model's response text

        Returns:
            The extracted answer string
        """
        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, response)

        if matches:
            return matches[-1].strip()

        pattern2 = r"\[([^\]]*)\]"
        matches2 = re.findall(pattern2, response)
        if matches2:
            return matches2[-1].strip()

        lines = response.split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("%"):
                return line

        return ""

    def normalize_number(self, answer: str) -> float:
        """
        Normalize a numeric answer.

        Args:
            answer: The answer string

        Returns:
            Normalized float value
        """
        answer = answer.strip()

        fraction_pattern = r"(-?\d+)\s*/\s*(\d+)"
        fraction_match = re.search(fraction_pattern, answer)
        if fraction_match:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            return numerator / denominator

        mixed_fraction_pattern = r"(-?\d+)\s+(\d+)\s*/\s*(\d+)"
        mixed_match = re.search(mixed_fraction_pattern, answer)
        if mixed_match:
            whole = float(mixed_match.group(1))
            numerator = float(mixed_match.group(2))
            denominator = float(mixed_match.group(3))
            return whole + (numerator / denominator)

        answer = re.sub(r"[^\d.-]", "", answer)

        try:
            return float(answer)
        except ValueError:
            return 0.0

    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted answer with ground truth using mathverify.

        For math tasks, we use the mathverify library to check correctness.

        Args:
            predicted: The predicted answer
            ground_truth: The ground truth answer

        Returns:
            True if answers match, False otherwise
        """
        if not predicted or not ground_truth:
            return False

        predicted = predicted.strip()
        ground_truth = ground_truth.strip()

        if predicted == ground_truth:
            return True

        if self.is_single_uppercase_letter(ground_truth):
            return predicted.strip() == ground_truth.strip()
        else:
            try:
                gold_parsed = parse(ground_truth)
                answer_parsed = parse(predicted)

                if gold_parsed is None or answer_parsed is None:
                    return False

                is_correct = verify(gold_parsed, answer_parsed)
                return is_correct

            except Exception as e:
                return False

    def evaluate_sample(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate a single math sample.

        Args:
            response: The model's response
            ground_truth: The ground truth answer

        Returns:
            Dictionary containing evaluation results
        """
        predicted_answer = self.extract_answer(response)
        is_correct = self.compare_answers(predicted_answer, ground_truth)
        has_valid_format = self.has_valid_boxed_format(response)

        try:
            pred_num = self.normalize_number(predicted_answer)
            gt_num = self.normalize_number(ground_truth)
            absolute_error = abs(pred_num - gt_num)
            relative_error = (
                absolute_error / abs(gt_num) if gt_num != 0 else float("inf")
            )
        except (ValueError, ZeroDivisionError):
            absolute_error = float("inf")
            relative_error = float("inf")

        return {
            "predicted": predicted_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "has_valid_format": has_valid_format,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
        }

    def evaluate_batch(
        self,
        responses: List[str],
        ground_truths: List[str],
        sample_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of math samples.

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
        valid_format_count = 0

        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            result = self.evaluate_sample(response, ground_truth)
            if sample_ids is not None and i < len(sample_ids):
                result["sample_id"] = sample_ids[i]
            results.append(result)
            if result["is_correct"]:
                correct_count += 1
            if result["has_valid_format"]:
                valid_format_count += 1

        accuracy = correct_count / len(responses) if responses else 0.0
        format_validity_rate = valid_format_count / len(responses) if responses else 0.0

        return {
            "task_type": self.task_type,
            "total_samples": len(responses),
            "correct_samples": correct_count,
            "accuracy": accuracy,
            "valid_format_samples": valid_format_count,
            "format_validity_rate": format_validity_rate,
            "sample_results": results,
        }

