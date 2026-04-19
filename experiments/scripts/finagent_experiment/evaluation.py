import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def calculate_finagent_accuracy(
    generated_answer: str, expected_answer: str, rubric: str = None
) -> Dict[str, Any]:
    if not generated_answer or not expected_answer:
        return {
            "score": 0.0,
            "binary_result": False,
            "details": {
                "generated_answer_empty": not generated_answer,
                "expected_answer_empty": not expected_answer,
            },
        }

    if rubric:
        try:
            try:
                rubric_criteria = json.loads(rubric)
            except json.JSONDecodeError:
                import ast
                rubric_criteria = ast.literal_eval(rubric)

            total_criteria = 0
            satisfied_criteria = 0
            satisfied_details = []
            failed_details = []

            for criterion in rubric_criteria:
                operator = criterion.get("operator", "").lower()
                criteria_text = criterion.get("criteria", "")
                total_criteria += 1

                if operator == "correctness":
                    if criteria_text.lower() in generated_answer.lower():
                        satisfied_criteria += 1
                        satisfied_details.append({"type": "correctness", "text": criteria_text, "found": True})
                    else:
                        failed_details.append({"type": "correctness", "text": criteria_text, "found": False})
                elif operator == "contradiction":
                    if criteria_text.lower() in generated_answer.lower():
                        failed_details.append({"type": "contradiction", "text": criteria_text, "found": True})
                    else:
                        satisfied_criteria += 1
                        satisfied_details.append({"type": "contradiction", "text": criteria_text, "found": False})

            if total_criteria > 0:
                score = max(0.0, satisfied_criteria / total_criteria)
                binary_result = satisfied_criteria >= (total_criteria / 2)
                return {
                    "score": min(1.0, score),
                    "binary_result": binary_result,
                    "details": {
                        "evaluation_method": "rubric_based",
                        "total_criteria": total_criteria,
                        "satisfied_criteria": satisfied_criteria,
                        "satisfied_details": satisfied_details,
                        "failed_details": failed_details,
                    },
                }
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    gen_lower = generated_answer.strip().lower()
    exp_lower = expected_answer.strip().lower()
    exact_match = gen_lower == exp_lower
    contains_expected = exp_lower in gen_lower
    exp_words = set(exp_lower.split())
    gen_words = set(gen_lower.split())
    overlap_ratio = len(exp_words.intersection(gen_words)) / len(exp_words) if exp_words else 0
    binary_result = exact_match or contains_expected or overlap_ratio >= 0.5

    if exact_match:
        score = 1.0
    elif contains_expected:
        score = 0.8
    elif overlap_ratio >= 0.5:
        score = min(0.7, overlap_ratio)
    else:
        score = 0.0

    return {
        "score": score,
        "binary_result": binary_result,
        "details": {
            "evaluation_method": "fallback_similarity",
            "exact_match": exact_match,
            "contains_expected": contains_expected,
            "overlap_ratio": overlap_ratio,
        },
    }


def evaluate_finagent_result(sample: Dict[str, Any], generated_answer: str) -> Dict[str, Any]:
    expected_answer = sample.get("cot_answer", "")
    rubric = sample.get("sample_info", {}).get("rubric", "")
    question_type = sample.get("sample_info", {}).get("question_type", "")
    accuracy_result = calculate_finagent_accuracy(generated_answer, expected_answer, rubric)
    return {
        "question_id": sample.get("main_id", ""),
        "question_type": question_type,
        "expected_answer": expected_answer,
        "generated_answer": generated_answer,
        "evaluation_result": accuracy_result["binary_result"],
        "evaluation_score": accuracy_result["score"],
        "evaluation_details": accuracy_result["details"],
        "rubric": rubric,
        "timestamp": datetime.now().isoformat(),
    }


def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_questions = len(results)
    if total_questions == 0:
        return {"total_questions": 0, "average_score": 0.0, "binary_accuracy": 0.0, "by_question_type": {}}

    total_score = sum(r.get("evaluation_score", 0.0) for r in results)
    binary_correct = sum(1 for r in results if r.get("evaluation_result", False))

    by_question_type = {}
    for r in results:
        qtype = r.get("question_type", "unknown")
        if qtype not in by_question_type:
            by_question_type[qtype] = {"count": 0, "total_score": 0.0, "correct": 0}
        by_question_type[qtype]["count"] += 1
        by_question_type[qtype]["total_score"] += r.get("evaluation_score", 0.0)
        if r.get("evaluation_result", False):
            by_question_type[qtype]["correct"] += 1

    for qtype in by_question_type:
        count = by_question_type[qtype]["count"]
        by_question_type[qtype]["average_score"] = by_question_type[qtype]["total_score"] / count if count > 0 else 0.0
        by_question_type[qtype]["accuracy"] = by_question_type[qtype]["correct"] / count if count > 0 else 0.0

    return {
        "total_questions": total_questions,
        "average_score": total_score / total_questions,
        "binary_accuracy": binary_correct / total_questions,
        "correct_count": binary_correct,
        "by_question_type": by_question_type,
    }
