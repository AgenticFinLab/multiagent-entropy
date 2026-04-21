"""Agent-architecture conventions shared by analyzers.

Centralizes the mapping from architecture name to "final" agent type and the
round-number formula. Previously duplicated in `experiment_analyzer.py` and
`entropy_statistic.py`.
"""

from typing import Any, Dict, List, Optional


ARCHITECTURE_FINAL_AGENT: Dict[str, str] = {
    "single": "SingleSolver",
    "sequential": "judger",
    "centralized": "OrchestratorAgent",
    "debate": "orchestrator",
    "hybrid": "OrchestratorAgent",
}


def get_final_agent_type(architecture: str) -> Optional[str]:
    """Return the agent type that produces the final answer for an architecture."""
    return ARCHITECTURE_FINAL_AGENT.get(architecture)


def get_round_number(
    execution_order: int,
    agent_type: str,
    architecture: str,
    num_rounds: int,
) -> int:
    """Compute the 1-based round number for an inference.

    Formulae preserved from `EntropyStatistic._get_round_number`:
      - single  : round == execution_order
      - debate  : orchestrator -> num_rounds; others -> (order-1)//3 + 1
      - others  : (order-1)//4 + 1
    """
    if architecture == "single":
        return execution_order
    if architecture == "debate":
        if agent_type == "orchestrator":
            return num_rounds
        return (execution_order - 1) // 3 + 1
    return (execution_order - 1) // 4 + 1


def get_final_agent_key_from_metrics(
    agents: Dict[str, Any], architecture: str
) -> Optional[str]:
    """Find the final agent's key inside an `experiment_analyzer` agents dict.

    Mirrors `ExperimentAnalyzer._get_final_agent_key`.
    """
    if not agents:
        return None
    final_agent_type = get_final_agent_type(architecture)
    if final_agent_type is None:
        return None
    if architecture == "debate":
        return "orchestrator"

    max_execution_order = -1
    final_agent_key: Optional[str] = None
    for agent_key, agent_data in agents.items():
        if agent_data["agent_type"] != final_agent_type:
            continue
        execution_order = agent_data["execution_order"]
        if execution_order > max_execution_order:
            max_execution_order = execution_order
            final_agent_key = agent_key
    return final_agent_key


def get_final_result_id_from_entropy(
    agents_data: List[Dict[str, Any]], architecture: str
) -> Optional[str]:
    """Find the final agent's result_id inside an `entropy_statistic` agents list.

    Mirrors `EntropyStatistic._get_final_agent_key`.
    """
    if not agents_data:
        return None
    final_agent_type = get_final_agent_type(architecture)
    if final_agent_type is None:
        return None

    max_execution_order = -1
    final_result_id: Optional[str] = None
    for data in agents_data:
        if data["agent_type"] != final_agent_type:
            continue
        execution_order = data["execution_order"]
        if execution_order > max_execution_order:
            max_execution_order = execution_order
            final_result_id = data["result_id"]
    return final_result_id
