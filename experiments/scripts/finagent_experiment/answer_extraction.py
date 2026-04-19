import re
from typing import Optional

from .constants import FINAGENT_TASK_TYPE
from maep.prompts import TASK_IDENTIFIERS


def extract_final_answer_by_identifier(
    text: str,
    task_type: str = FINAGENT_TASK_TYPE,
) -> Optional[str]:
    if not text:
        return None

    if task_type == "finance":
        patterns = [
            r"(?:FINAL\s*ANSWER\s*:\s*)(.+?)(?:\n\n|$)",
            r"(?:FINAL\s*ANSWER\s*:?)\s*[\[\(]?(.+?)[\]\)]?\s*(?:\n\n|$)",
            r"\*\*FINAL\s*ANSWER\*\*\s*:?\s*(.+?)(?:\n\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r'\s*\*\*\s*$', '', answer)
                if answer:
                    return answer
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            for para in reversed(paragraphs):
                if len(para) > 20 and not para.lower().startswith(('note:', 'source:', 'citation:')):
                    return para

    elif task_type in ("math", "option"):
        matches = re.findall(r"\\boxed\{([^}]+)\}", text)
        if matches:
            return matches[-1].strip()

    elif task_type == "code":
        matches = re.findall(r"```python\s*\n(.+?)\n```", text, re.DOTALL)
        if matches:
            return matches[-1].strip()

    return None


def extract_answer_from_result(
    final_state: dict,
    task_type: str = FINAGENT_TASK_TYPE,
    extract_by_identifier: bool = True,
) -> str:
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
