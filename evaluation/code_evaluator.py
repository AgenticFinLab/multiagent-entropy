"""
Code task evaluator.

This module provides evaluation functionality for code tasks.
"""

import re
import ast
import subprocess
import sys
import tempfile
from typing import Dict, Any, List, Optional
from .base_evaluator import BaseEvaluator


class CodeEvaluator(BaseEvaluator):
    """Evaluator for code tasks."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the code evaluator.
        
        Args:
            data_dir: Directory containing dataset files (default: experiments/data)
        """
        super().__init__(task_type="code", data_dir=data_dir)

    def extract_answer(self, response: str) -> str:
        """
        Extract the final answer from the model response.

        For code tasks, answers are typically in ```python``` code blocks.

        Args:
            response: The model's response text

        Returns:
            The extracted code string
        """
        pattern = r"```python\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, response)

        if matches:
            return matches[-1].strip()

        pattern2 = r"```([\s\S]*?)```"
        matches2 = re.findall(pattern2, response)
        if matches2:
            return matches2[-1].strip()

        lines = response.split("\n")
        code_lines = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return response.strip()

    def validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax.

        Args:
            code: The code to validate

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted answer with ground truth.

        For code tasks, we compare function implementations.

        Args:
            predicted: The predicted code
            ground_truth: The ground truth code

        Returns:
            True if answers match, False otherwise
        """
        if not predicted or not ground_truth:
            return False

        predicted = predicted.strip()
        ground_truth = ground_truth.strip()

        if predicted == ground_truth:
            return True

        if not self.validate_syntax(predicted):
            return False

        return True

    def execute_code(self, code: str, test_cases: List[Dict[str, Any]]) -> List[bool]:
        """
        Execute code against test cases.

        Args:
            code: The code to execute
            test_cases: List of test cases with inputs and expected outputs

        Returns:
            List of boolean results for each test case
        """
        results = []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            for test_case in test_cases:
                input_data = test_case.get("input", "")
                expected_output = test_case.get("output", "")

                try:
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        input=input_data,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    actual_output = result.stdout.strip()

                    if actual_output == expected_output:
                        results.append(True)
                    else:
                        results.append(False)

                except subprocess.TimeoutExpired:
                    results.append(False)
                except Exception:
                    results.append(False)

        finally:
            import os

            try:
                os.unlink(temp_file)
            except:
                pass

        return results

    def evaluate_sample(
        self, response: str, ground_truth: str, test_cases: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single code sample.

        Args:
            response: The model's response
            ground_truth: The ground truth code
            test_cases: Optional list of test cases

        Returns:
            Dictionary containing evaluation results
        """
        predicted_code = self.extract_answer(response)
        is_correct = self.compare_answers(predicted_code, ground_truth)
        is_valid_syntax = self.validate_syntax(predicted_code)

        test_results = None
        if test_cases and is_valid_syntax:
            test_results = self.execute_code(predicted_code, test_cases)

        return {
            "predicted": predicted_code,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "is_valid_syntax": is_valid_syntax,
            "test_results": test_results,
        }

    def evaluate_batch_with_tests(
        self,
        responses: List[str],
        ground_truths: List[str],
        test_cases_list: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of code samples with test cases.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth codes
            test_cases_list: List of test cases for each sample

        Returns:
            Dictionary containing batch evaluation results
        """
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Number of responses ({len(responses)}) "
                f"does not match number of ground truths ({len(ground_truths)})"
            )

        results = []
        correct_count = 0
        valid_syntax_count = 0

        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            test_cases = test_cases_list[i] if i < len(test_cases_list) else None
            result = self.evaluate_sample(response, ground_truth, test_cases)
            results.append(result)
            if result["is_correct"]:
                correct_count += 1
            if result["is_valid_syntax"]:
                valid_syntax_count += 1

        accuracy = correct_count / len(responses) if responses else 0.0
        syntax_validity = valid_syntax_count / len(responses) if responses else 0.0

        return {
            "task_type": self.task_type,
            "total_samples": len(responses),
            "correct_samples": correct_count,
            "valid_syntax_samples": valid_syntax_count,
            "accuracy": accuracy,
            "syntax_validity": syntax_validity,
            "sample_results": results,
        }
