"""
Code task evaluator.

This module provides evaluation functionality for code tasks using test case execution.
"""

import re
import os
import ast
import sys
import json
import logging
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple

from .base_evaluator import BaseEvaluator


logger = logging.getLogger(__name__)


class CodeEvaluator(BaseEvaluator):
    """Evaluator for code tasks using test case execution."""

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the code evaluator.

        Args:
            data_dir: Directory containing dataset files (default: experiments/data)
        """
        super().__init__(task_type="code", data_dir=data_dir)
        self.data_dir = data_dir or "experiments/data"
        self.test_cache = {}

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

    def normalize_code_indentation(self, code: str) -> str:
        """
        Normalize code indentation by removing common leading whitespace.

        Args:
            code: The code to normalize

        Returns:
            Normalized code
        """
        lines = code.split("\n")
        if not lines:
            return code

        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            return code

        leading_whitespace = min(
            len(line) - len(line.lstrip()) for line in non_empty_lines
        )

        normalized_lines = []
        for line in lines:
            if line.strip():
                normalized_lines.append(line[leading_whitespace:])
            else:
                normalized_lines.append("")

        return "\n".join(normalized_lines)

    def compare_answers(self, predicted: str, ground_truth: str) -> bool:
        """
        Compare predicted answer with ground truth using test case execution.

        For code tasks, we execute both implementations against test cases
        and compare their outputs.

        Args:
            predicted: The predicted code
            ground_truth: The ground truth code

        Returns:
            True if both implementations produce same outputs on all test cases
        """
        if not predicted or not ground_truth:
            return False

        predicted = predicted.strip()
        ground_truth = ground_truth.strip()

        if not self.validate_syntax(predicted):
            return False

        if not self.validate_syntax(ground_truth):
            return False

        return True

    def load_test_cases(self, sample_id: str) -> Optional[str]:
        """
        Load test cases for a given sample from the dataset.

        Args:
            sample_id: The sample ID (e.g., "ID1", "ID2")

        Returns:
            Test case code string or None if not found
        """
        if sample_id in self.test_cache:
            return self.test_cache[sample_id]

        data_dir = self.data_dir or "experiments/data"
        dataset_path = os.path.join(data_dir, "HumanEval", "test-all-samples.json")

        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return None

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                samples = json.load(f)

            for sample in samples:
                if sample.get("main_id") == sample_id:
                    test_cases = sample.get("test_cases")
                    if test_cases:
                        self.test_cache[sample_id] = test_cases
                        return test_cases

            logger.warning(f"Sample {sample_id} not found in dataset")
            return None

        except Exception as e:
            logger.error(f"Error loading test cases for {sample_id}: {e}")
            return None

    def extract_function_signature(self, question: str) -> Optional[str]:
        """
        Extract function signature from the question.

        Args:
            question: The question text containing function definition

        Returns:
            Function signature string or None if not found
        """
        pattern = r"def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[\w\[\],\s]+)?:"
        match = re.search(pattern, question)
        if match:
            return match.group(0)
        return None

    def construct_executable_code(
        self, question: str, implementation: str
    ) -> Optional[str]:
        """
        Construct executable code by combining question and implementation.

        Args:
            question: The question containing function signature
            implementation: The function implementation

        Returns:
            Complete executable code or None if construction fails
        """
        try:
            lines = question.strip().split("\n")

            imports = []
            signature = None

            for i, line in enumerate(lines):
                stripped = line.strip()

                if stripped.startswith("import ") or stripped.startswith("from "):
                    imports.append(line)

                elif stripped.startswith("def "):
                    signature = self.extract_function_signature(question)
                    break

            if signature is None:
                logger.warning("Could not extract function signature from question")
                return None

            normalized_implementation = self.normalize_code_indentation(implementation)

            indented_implementation = "\n".join(
                "    " + line if line.strip() else ""
                for line in normalized_implementation.split("\n")
            )

            code_parts = []
            if imports:
                code_parts.extend(imports)
            code_parts.append(signature)
            code_parts.append(indented_implementation)

            full_code = "\n".join(code_parts)

            if self.validate_syntax(full_code):
                return full_code

            logger.warning("Constructed code has invalid syntax")
            return None

        except Exception as e:
            logger.error(f"Error constructing executable code: {e}")
            return None

    def execute_code_with_tests(
        self, code: str, test_cases: str, timeout: float = 5.0
    ) -> Tuple[bool, str, List[bool]]:
        """
        Execute code against test cases and capture results.

        Args:
            code: The complete executable code
            test_cases: The test case code (check function)
            timeout: Execution timeout in seconds

        Returns:
            Tuple of (success, error_message, test_results)
            - success: True if execution succeeded
            - error_message: Error message if execution failed
            - test_results: List of boolean results for each assertion
        """
        import re

        pattern = r"def\s+(\w+)\s*\("
        match = re.search(pattern, code)

        if match:
            original_name = match.group(1)
            code = code.replace(f"def {original_name}(", "def candidate(")

        full_code = code + "\n\n" + test_cases + "\n\n"

        test_runner_code = """
def run_tests():
    try:
        check(candidate)
        return True, []
    except AssertionError as e:
        return False, [False]
    except Exception as e:
        return False, [False]

success, results = run_tests()
print(f"SUCCESS:{success}")
if results:
    print(f"RESULTS:{results}")
"""

        full_code += test_runner_code

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout + result.stderr

            if "SUCCESS:True" in output:
                return True, "", [True]
            elif "SUCCESS:False" in output:
                return False, "Test assertions failed", [False]
            else:
                error_msg = output.strip()
                if result.returncode != 0:
                    return False, error_msg, [False]
                return False, "Unknown error during execution", [False]

        except subprocess.TimeoutExpired:
            return False, "Execution timeout", [False]
        except Exception as e:
            return False, str(e), [False]
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    def evaluate_sample(
        self,
        response: str,
        ground_truth: str,
        sample_id: Optional[str] = None,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single code sample using test case execution.

        Args:
            response: The model's response
            ground_truth: The ground truth code
            sample_id: Optional sample ID for loading test cases
            question: Optional question text for extracting function signature

        Returns:
            Dictionary containing evaluation results
        """
        predicted_code = self.extract_answer(response)

        is_valid_syntax = self.validate_syntax(predicted_code)

        test_passed = False
        test_error = ""
        execution_success = False

        if sample_id:
            test_cases = self.load_test_cases(sample_id)

            if test_cases and question:
                executable_code = self.construct_executable_code(
                    question, predicted_code
                )

                if executable_code:
                    execution_success, test_error, test_results = (
                        self.execute_code_with_tests(executable_code, test_cases)
                    )
                    test_passed = execution_success and (
                        len(test_results) > 0 and all(test_results)
                    )
                    is_valid_syntax = True
                else:
                    test_error = "Failed to construct executable code"
                    is_valid_syntax = False
            elif test_cases:
                test_error = "Question not provided for signature extraction"
            else:
                test_error = "No test cases found"

        return {
            "predicted": predicted_code,
            "ground_truth": ground_truth,
            "is_valid_syntax": is_valid_syntax,
            "test_passed": test_passed,
            "test_error": test_error,
            "execution_success": execution_success,
        }

    def evaluate_batch_with_tests(
        self,
        responses: List[str],
        ground_truths: List[str],
        sample_ids: List[str],
        questions: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of code samples with test cases.

        Args:
            responses: List of model responses
            ground_truths: List of ground truth codes
            sample_ids: List of sample IDs for loading test cases
            questions: List of question texts for extracting function signatures

        Returns:
            Dictionary containing batch evaluation results
        """
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Number of responses ({len(responses)}) "
                f"does not match number of ground truths ({len(ground_truths)})"
            )

        if len(responses) != len(sample_ids):
            raise ValueError(
                f"Number of responses ({len(responses)}) "
                f"does not match number of sample_ids ({len(sample_ids)})"
            )

        results = []
        test_passed_count = 0
        valid_syntax_count = 0
        execution_success_count = 0

        for i, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
            sample_id = sample_ids[i] if i < len(sample_ids) else None
            question = questions[i] if i < len(questions) else None

            result = self.evaluate_sample(response, ground_truth, sample_id, question)
            results.append(result)

            if result["test_passed"]:
                test_passed_count += 1
            if result["is_valid_syntax"]:
                valid_syntax_count += 1
            if result["execution_success"]:
                execution_success_count += 1

        total_samples = len(responses)
        test_pass_rate = test_passed_count / total_samples if total_samples > 0 else 0.0
        syntax_validity = (
            valid_syntax_count / total_samples if total_samples > 0 else 0.0
        )
        execution_success_rate = (
            execution_success_count / total_samples if total_samples > 0 else 0.0
        )

        return {
            "task_type": self.task_type,
            "total_samples": total_samples,
            "test_passed_samples": test_passed_count,
            "valid_syntax_samples": valid_syntax_count,
            "execution_success_samples": execution_success_count,
            "test_pass_rate": test_pass_rate,
            "syntax_validity": syntax_validity,
            "execution_success_rate": execution_success_rate,
            "sample_results": results,
        }
