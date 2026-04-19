import logging
import re
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    if not answer:
        return ""
    answer = answer.strip()
    # Strip trailing dict/list stringification artifacts (e.g. ']} from str(dict))
    answer = re.sub(r"['\]\}]+$", '', answer).strip()
    # Remove surrounding quotes
    if len(answer) >= 2 and answer[0] in ('"', "'") and answer[-1] == answer[0]:
        answer = answer[1:-1].strip()
    return answer.lower()


def calculate_gaia_accuracy(
    generated_answer: str,
    groundtruth: str,
) -> Dict[str, Any]:
    """
    Calculate accuracy for a GAIA sample.

    GAIA uses exact-match evaluation on the final answer after normalization.
    Numbers are compared numerically when possible.

    Args:
        generated_answer: Answer produced by the agent
        groundtruth: Ground truth answer from the dataset

    Returns:
        Dict with 'score' (0.0 or 1.0), 'binary_result' (bool), and 'details'
    """
    if not generated_answer or not groundtruth:
        return {
            "score": 0.0,
            "binary_result": False,
            "details": {
                "generated_answer_empty": not generated_answer,
                "groundtruth_empty": not groundtruth,
            },
        }

    gen_norm = normalize_answer(generated_answer)
    gt_norm = normalize_answer(groundtruth)

    # Exact match after normalization
    if gen_norm == gt_norm:
        return {
            "score": 1.0,
            "binary_result": True,
            "details": {"evaluation_method": "exact_match"},
        }

    # Numeric comparison
    try:
        gen_num = float(gen_norm.replace(",", ""))
        gt_num = float(gt_norm.replace(",", ""))
        if abs(gen_num - gt_num) < 1e-6:
            return {
                "score": 1.0,
                "binary_result": True,
                "details": {"evaluation_method": "numeric_match"},
            }
    except ValueError:
        pass

    # Substring containment as partial credit signal (score only, binary stays False)
    contains = gt_norm in gen_norm
    overlap = (
        len(set(gt_norm.split()) & set(gen_norm.split())) / max(1, len(set(gt_norm.split())))
    )

    return {
        "score": 0.0,
        "binary_result": False,
        "details": {
            "evaluation_method": "no_match",
            "gen_norm": gen_norm,
            "gt_norm": gt_norm,
            "contains_gt": contains,
            "word_overlap": round(overlap, 4),
        },
    }


def evaluate_gaia_result(
    sample: Dict[str, Any],
    generated_answer: str,
) -> Dict[str, Any]:
    """
    Evaluate a single GAIA sample.

    Args:
        sample: Original sample dict from the dataset
        generated_answer: Answer produced by the agent

    Returns:
        Dict with evaluation results
    """
    groundtruth = sample.get("groundtruth", "")
    level = sample.get("sample_info", {}).get("level", "")
    file_name = sample.get("sample_info", {}).get("file_name", "")

    accuracy_result = calculate_gaia_accuracy(generated_answer, groundtruth)

    return {
        "question_id": sample.get("main_id", ""),
        "level": level,
        "file_name": file_name,
        "groundtruth": groundtruth,
        "generated_answer": generated_answer,
        "evaluation_result": accuracy_result["binary_result"],
        "evaluation_score": accuracy_result["score"],
        "evaluation_details": accuracy_result["details"],
        "timestamp": datetime.now().isoformat(),
    }


def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate aggregate metrics from evaluation results, broken down by level.

    Args:
        results: List of per-sample evaluation dicts

    Returns:
        Dict with aggregate metrics
    """
    total = len(results)
    if total == 0:
        return {"total_questions": 0, "accuracy": 0.0, "by_level": {}}

    correct = sum(1 for r in results if r.get("evaluation_result", False))

    by_level: Dict[str, Any] = {}
    for r in results:
        lvl = str(r.get("level", "unknown"))
        if lvl not in by_level:
            by_level[lvl] = {"count": 0, "correct": 0}
        by_level[lvl]["count"] += 1
        if r.get("evaluation_result", False):
            by_level[lvl]["correct"] += 1

    for lvl in by_level:
        cnt = by_level[lvl]["count"]
        by_level[lvl]["accuracy"] = round(by_level[lvl]["correct"] / cnt, 4) if cnt > 0 else 0.0

    return {
        "total_questions": total,
        "correct_count": correct,
        "accuracy": round(correct / total, 4),
        "by_level": by_level,
    }
