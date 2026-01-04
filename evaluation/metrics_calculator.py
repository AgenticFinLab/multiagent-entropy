import re
import torch
from typing import Dict, Any, List, Optional

from math_verify import parse, verify


class MetricsCalculator:
    @staticmethod
    def extract_boxed_answer(text: str) -> Optional[str]:
        pattern = r"\\boxed\{([^}]*)\}"
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]

        pattern = r"(?:the\s+correct\s+answer\s+is|answer\s*[:=]\s*|final\s+answer\s*[:=]\s*)\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1]

        return None

    @staticmethod
    def has_valid_format(text: str) -> bool:
        return MetricsCalculator.extract_boxed_answer(text) is not None

    @staticmethod
    def extract_code_answer(text: str) -> Optional[str]:
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1]
        return None

    @staticmethod
    def normalize_answer(answer: str) -> str:
        answer = answer.strip()
        answer = re.sub(r"\s+", " ", answer)
        return answer

    @staticmethod
    def is_single_uppercase_letter(text: str) -> bool:
        text = text.strip()
        return len(text) == 1 and text.isupper()

    @staticmethod
    def is_answer_correct(predicted: Optional[str], ground_truth: str) -> bool:
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
    def calculate_time_cost(result: Dict[str, Any]) -> float:
        if "cost" in result and "time" in result["cost"]:
            return result["cost"]["time"]
        return 0.0

    @staticmethod
    def calculate_average_entropy(entropy_tensor: torch.Tensor) -> float:
        if entropy_tensor is None:
            return 0.0

        if isinstance(entropy_tensor, torch.Tensor):
            return float(entropy_tensor.mean().item())

        return 0.0

    @staticmethod
    def get_agent_accuracy_for_single(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
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

            predicted = MetricsCalculator.extract_boxed_answer(
                result.get("response", "")
            )
            is_correct = MetricsCalculator.is_answer_correct(
                predicted, ground_truth["groundtruth"]
            )

            key = f"{main_id}_{agent_type}"
            if key not in accuracies:
                accuracies[key] = []
            accuracies[key].append(is_correct)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in accuracies.items()}

    @staticmethod
    def get_agent_accuracy_for_sequential(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
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

            predicted = MetricsCalculator.extract_boxed_answer(
                result.get("response", "")
            )
            is_correct = MetricsCalculator.is_answer_correct(
                predicted, ground_truth["groundtruth"]
            )

            if agent_type == "judger":
                key = f"{main_id}_judger"
                if key not in accuracies:
                    accuracies[key] = []
                accuracies[key].append(is_correct)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in accuracies.items()}

    @staticmethod
    def get_agent_accuracy_for_centralized(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
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

            predicted = MetricsCalculator.extract_boxed_answer(
                result.get("response", "")
            )
            is_correct = MetricsCalculator.is_answer_correct(
                predicted, ground_truth["groundtruth"]
            )

            if agent_type == "OrchestratorAgent":
                key = f"{main_id}_OrchestratorAgent"
                if key not in accuracies:
                    accuracies[key] = []
                accuracies[key].append(is_correct)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in accuracies.items()}

    @staticmethod
    def get_agent_accuracy_for_debate(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
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

            if "final_answer" in result:
                predicted = result["final_answer"]
            else:
                predicted = ""

            is_correct = MetricsCalculator.is_answer_correct(
                predicted, ground_truth["groundtruth"]
            )

            if agent_type == "OrchestratorAgent" or "final_answer" in result:
                key = f"{main_id}_final"
                if key not in accuracies:
                    accuracies[key] = []
                accuracies[key].append(is_correct)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in accuracies.items()}

    @staticmethod
    def get_agent_accuracy_for_hybrid(
        results: Dict[str, Any], ground_truths: Dict[str, Any], sample_ids: List[str]
    ) -> Dict[str, float]:
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

            predicted = MetricsCalculator.extract_boxed_answer(
                result.get("response", "")
            )
            is_correct = MetricsCalculator.is_answer_correct(
                predicted, ground_truth["groundtruth"]
            )

            if agent_type == "OrchestratorAgent":
                key = f"{main_id}_OrchestratorAgent"
                if key not in accuracies:
                    accuracies[key] = []
                accuracies[key].append(is_correct)

        return {k: sum(v) / len(v) if v else 0.0 for k, v in accuracies.items()}

    @staticmethod
    def calculate_agent_accuracy(
        agent_architecture: str,
        results: Dict[str, Any],
        ground_truths: Dict[str, Any],
        sample_ids: List[str],
    ) -> Dict[str, float]:
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
