import re
from typing import Optional

from .constants import GAIA_TASK_TYPE


def extract_final_answer_by_identifier(
    text: str,
    task_type: str = GAIA_TASK_TYPE,
) -> Optional[str]:
    """
    Extract the final answer from agent response text using the official GAIA
    answer marker "FINAL ANSWER:".

    The official GAIA system prompt instructs the model to end its response with:
        FINAL ANSWER: [YOUR FINAL ANSWER]

    This function first tries to match that marker (case-insensitive), then falls
    back to the last substantial paragraph if no marker is present.

    Args:
        text: Full response text to parse
        task_type: Unused — kept for API symmetry with finagent_experiment

    Returns:
        Extracted answer string, or None
    """
    if not text:
        return None

    # Official GAIA marker patterns (most specific first)
    patterns = [
        # "FINAL ANSWER: foo"  — captures rest of line
        r"FINAL\s+ANSWER\s*:\s*(.+?)(?:\n|$)",
        # With optional bold markers or brackets
        r"\*\*FINAL\s+ANSWER\*\*\s*:?\s*(.+?)(?:\n|$)",
        r"FINAL\s+ANSWER\s*:\s*[\[\(](.+?)[\]\)]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Strip trailing dict/list stringification artifacts (e.g. ']} from str(dict))
            answer = re.sub(r"['\]\}\*\[]+$", '', answer).strip()
            if answer:
                return answer

    # Fallback: last substantial paragraph
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    for para in reversed(paragraphs):
        if len(para) > 5 and not para.lower().startswith(('note:', 'source:', 'citation:')):
            return para

    return None


def extract_answer_from_result(
    final_state: dict,
    task_type: str = GAIA_TASK_TYPE,
    extract_by_identifier: bool = True,
) -> str:
    """
    Extract the generated answer from the agent's final state dict.

    Args:
        final_state: Final state returned by the agent
        task_type: Used for identifier-based extraction
        extract_by_identifier: Whether to attempt identifier-based extraction

    Returns:
        Extracted answer string
    """
    raw_answer = None

    if "agent_results" in final_state and len(final_state["agent_results"]) > 0:
        result = final_state["agent_results"][0]
        for key in ["final_answer", "answer", "response", "output", "result"]:
            if key in result:
                raw_answer = str(result[key])
                break
        if raw_answer is None:
            raw_answer = str(result)
    elif "merged_results" in final_state:
        raw_answer = str(final_state["merged_results"])
    else:
        raw_answer = str(final_state)

    if extract_by_identifier and raw_answer:
        extracted = extract_final_answer_by_identifier(raw_answer, task_type)
        if extracted:
            return extracted

    return raw_answer
