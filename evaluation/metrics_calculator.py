"""Metrics calculator for evaluating multi-agent system performance.

This module provides utilities for calculating various metrics including
accuracy, time cost, and entropy from experiment results.
"""

import re
import multiprocessing
from typing import Optional
from typing import Dict, Any, List, Optional

import torch
from math_verify import parse, verify


class TimeoutError(Exception):
    pass


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
        # Define regex pattern to match LaTeX boxed format \boxed{...}
        pattern = r"\\boxed\{([^}]*)\}"
        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text)
        # Check if any matches were found
        if matches:
            # Handle case where answer starts with { (nested braces)
            if matches[-1].startswith("{"):
                return matches[-1][1:], True
            # Handle case where answer ends with } (nested braces)
            elif matches[-1].endswith("}"):
                return matches[-1][:-1], True
            # Handle case where answer is wrapped in parentheses
            elif matches[-1].startswith("(") and matches[-1].endswith(")"):
                return matches[-1][1:-1], True
            # Default case: return the matched answer as-is
            else:
                return matches[-1], True

        # No valid boxed format found
        return None, False

    @staticmethod
    def has_valid_format(text: str) -> bool:
        """Check if text contains valid boxed answer format.

        Args:
            text: Input text to check.

        Returns:
            True if valid format found, False otherwise.
        """
        # Extract boxed answer and return format compliance status
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
        # Define regex pattern to match Python code blocks
        pattern = r"```python\s*(.*?)\s*```"
        # Find all matches with DOTALL flag to match across newlines
        matches = re.findall(pattern, text, re.DOTALL)
        # Return the last match if any code blocks were found
        if matches:
            return matches[-1], True
        # No valid code block found
        return None, False

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer string by stripping and collapsing whitespace.

        Args:
            answer: Raw answer string.

        Returns:
            Normalized answer string.
        """
        # Remove leading and trailing whitespace
        answer = answer.strip()
        # Collapse multiple whitespace characters into single space
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
        # Remove any surrounding whitespace
        text = text.strip()
        # Check if text is exactly one character and is uppercase
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
        # Return False if no predicted answer was provided
        if predicted is None:
            return False

        # Normalize both predicted and ground truth answers
        predicted_norm = MetricsCalculator.normalize_answer(predicted)
        ground_truth_norm = MetricsCalculator.normalize_answer(ground_truth)

        # For single letter options, do direct string comparison
        if MetricsCalculator.is_single_uppercase_letter(ground_truth_norm):
            return predicted_norm == ground_truth_norm

        # For mathematical expressions, use parsing and verification
        try:
            # Parse both answers into mathematical expressions
            gold_parsed = parse(ground_truth_norm)
            answer_parsed = parse(predicted_norm)

            # Return False if parsing failed for either answer
            if gold_parsed is None or answer_parsed is None:
                return False

            # Verify if parsed answers are mathematically equivalent
            is_correct = verify(gold_parsed, answer_parsed)
            return is_correct
        # Return False if any error occurs during verification
        except Exception as e:
            return False

    @staticmethod
    def is_code_correct(
        predicted_code: Optional[str],
        ground_truth_code: str,
        test_cases: Optional[str] = None,
        timeout: int = 10,
    ) -> bool:
        """Check if predicted code passes all test cases.

        Args:
            predicted_code: Predicted Python code string.
            ground_truth_code: Ground truth code (for reference).
            test_cases: Test cases to validate the code against.
            timeout: Maximum time in seconds to execute the code.

        Returns:
            True if code passes all tests, False otherwise.
        """
        # Return False if no predicted code was provided
        if not predicted_code:
            return False

        # Return False if no test cases were provided
        if not test_cases:
            return False

        def execute_code_in_process(result_queue):
            """Execute code in a separate process and put result in queue."""
            import sys
            import io
            
            # Capture stdout to suppress print statements from predicted code
            old_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                # Execute predicted code in isolated namespace with restricted builtins
                local_namespace = {}
                restricted_builtins = {
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'sorted': sorted,
                    'reversed': reversed,
                    'map': map,
                    'filter': filter,
                    'any': any,
                    'all': all,
                    'isinstance': isinstance,
                    'type': type,
                    'Exception': Exception,
                    'ValueError': ValueError,
                    'TypeError': TypeError,
                    'IndexError': IndexError,
                    'KeyError': KeyError,
                    'AttributeError': AttributeError,
                    'AssertionError': AssertionError,
                    'RuntimeError': RuntimeError,
                }
                safe_globals = {'__builtins__': restricted_builtins}
                exec(predicted_code, safe_globals, local_namespace)

                # Execute test cases in copy of local namespace
                test_namespace = local_namespace.copy()
                exec(test_cases, safe_globals, test_namespace)
            except Exception:
                # Restore original stdout in case of exception
                sys.stdout = old_stdout
                result_queue.put(False)
                return
            finally:
                # Restore original stdout
                sys.stdout = old_stdout
            
            # Check if test cases define a check function
            success = True
            if "check" in test_namespace:
                # Run check function for each callable in predicted code
                for func_name, func in local_namespace.items():
                    # Skip private functions and non-callables
                    if callable(func) and not func_name.startswith("_"):
                        try:
                            # Run check function with the predicted function
                            test_namespace["check"](func)
                        # Return False if assertion fails
                        except AssertionError:
                            success = False
                            break
                        # Continue if other exceptions occur
                        except Exception:
                            continue

            result_queue.put(success)

        # Create a queue to get the result from the subprocess
        result_queue = multiprocessing.Queue()
        
        # Create and start the process
        process = multiprocessing.Process(target=execute_code_in_process, args=(result_queue,))
        process.start()
        try:
            # Wait for the result with timeout
            result = result_queue.get(timeout=timeout)
            process.join(timeout=timeout)
            
            # If process is still alive after timeout, terminate it
            if process.is_alive():
                process.terminate()
                process.join()
                return False
            
            return result
            
        except:
            # If anything goes wrong, make sure to terminate the process
            if process.is_alive():
                process.terminate()
                process.join()
            return False

    @staticmethod
    def is_answer_correct_by_task_type(
        predicted: Optional[str],
        ground_truth: str,
        task_type: str = "math",
        test_cases: Optional[str] = None,
        timeout: int = 10,
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
        # For code tasks, use code verification with test cases
        if task_type == "code":
            return MetricsCalculator.is_code_correct(
                predicted, ground_truth, test_cases, timeout
            )
        # For math and option tasks, use mathematical verification
        else:
            return MetricsCalculator.is_answer_correct(predicted, ground_truth)

    @staticmethod
    def calculate_agent_time_cost(result: Dict[str, Any]) -> float:
        """Extract time cost from result dictionary.

        Args:
            result: Result dictionary containing cost information.

        Returns:
            Time cost value or 0.0 if not found.
        """
        # Check if cost dictionary exists and contains time information
        if "cost" in result and "time" in result["cost"]:
            # Return the time cost value
            return result["cost"]["time"]
        # Return 0.0 if no time cost information is available
        return 0.0

    @staticmethod
    def calculate_average_entropy(entropy_tensor: torch.Tensor) -> float:
        """Calculate average entropy from tensor.

        Args:
            entropy_tensor: Tensor containing entropy values.

        Returns:
            Average entropy value or 0.0 if tensor is None.
        """
        # Return 0.0 if tensor is None
        if entropy_tensor is None:
            return 0.0

        # Check if input is a PyTorch tensor
        if isinstance(entropy_tensor, torch.Tensor):
            # Calculate mean and convert to Python float
            return float(entropy_tensor.mean().item())

        # Return 0.0 for non-tensor inputs
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
        # Initialize dictionary to store accuracy results
        accuracies = {}

        # Iterate through each sample ID
        for sample_id in sample_ids:
            # Get result data for this sample
            result = results.get(sample_id)
            # Skip if no result data exists
            if not result:
                continue

            # Parse the sample ID to extract components
            parsed_id = sample_id.replace("Result_", "").split("-")
            main_id = parsed_id[0]
            agent_type = parsed_id[1]

            # Get ground truth for this sample
            ground_truth = ground_truths.get(main_id)
            # Skip if no ground truth exists
            if not ground_truth:
                continue

            # Extract predicted answer based on configuration
            if use_final_answer and "final_answer" in result:
                # Use final answer field if available and requested
                predicted = result["final_answer"]
            else:
                # Extract answer from response text using boxed format
                predicted, _ = MetricsCalculator.extract_boxed_answer(
                    result.get("response", "")
                )

            # Check if predicted answer is correct
            is_correct = MetricsCalculator.is_answer_correct(
                predicted, ground_truth["groundtruth"]
            )

            # Only process if agent type matches filter or no filter is set
            if agent_filter is None or agent_type in agent_filter:
                # Create key for this agent result
                key = f"{main_id}_{key_suffix if key_suffix else agent_type}"
                # Initialize list for this key if not exists
                if key not in accuracies:
                    accuracies[key] = []
                # Append correctness result to list
                accuracies[key].append(is_correct)

        # Calculate accuracy for each agent by averaging correctness values
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
        # Calculate accuracy using base implementation without filtering
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
        # Calculate accuracy using base implementation with judger filter
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
        # Calculate accuracy using base implementation with OrchestratorAgent filter
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
        # Calculate accuracy using base implementation with OrchestratorAgent filter and final answer
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
        # Calculate accuracy using base implementation with OrchestratorAgent filter
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
        # Map architecture names to their corresponding calculator functions
        architecture_map = {
            "single": MetricsCalculator.get_agent_accuracy_for_single,
            "sequential": MetricsCalculator.get_agent_accuracy_for_sequential,
            "centralized": MetricsCalculator.get_agent_accuracy_for_centralized,
            "debate": MetricsCalculator.get_agent_accuracy_for_debate,
            "hybrid": MetricsCalculator.get_agent_accuracy_for_hybrid,
        }

        # Get the appropriate calculator function for the architecture
        calculator = architecture_map.get(agent_architecture.lower())
        # Call the calculator function if it exists
        if calculator:
            return calculator(results, ground_truths, sample_ids)

        # Return empty dictionary if architecture is not recognized
        return {}
