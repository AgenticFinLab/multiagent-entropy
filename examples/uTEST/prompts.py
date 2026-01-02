"""
Centralized prompt definitions for uTEST examples.
"""

# Task identifiers for different task types
TASK_IDENTIFIERS = {
    "math": "\\boxed{{}}",
    "code": "```python\n```",
    "option": "\\boxed{{}}"
}

def get_identifier(task_type: str) -> str:
    """
    Get the identifier for a given task type.
    
    Args:
        task_type: One of 'math', 'code', or 'option'
        
    Returns:
        The corresponding identifier string
        
    Raises:
        ValueError: If task_type is not supported
    """
    if task_type not in TASK_IDENTIFIERS:
        raise ValueError(f"Unsupported task_type: {task_type}. Must be one of {list(TASK_IDENTIFIERS.keys())}")
    return TASK_IDENTIFIERS[task_type]

def validate_task_type(task_type: str) -> bool:
    """
    Validate that the task_type is supported.
    
    Args:
        task_type: The task type to validate
        
    Returns:
        True if valid, False otherwise
    """
    return task_type in TASK_IDENTIFIERS

# --- From maep/language/single.py ---
SINGLE_SYS = """You are a precise solver.
Solve the problem correctly and concisely. """

SINGLE_USER = """Question:
{question}
Think step by step and place the final answer in {identifier}."""

# --- From maep/language/sequential.py ---
PLANNER_SYS = """You are the planner agent. Generate plans that are the general instructions only.
Do not execute the plan, do not perform any calculations, and do not produce any answers or intermediate numerical results.
Output a structured, numbered plans."""

PLANNER_USER = """For the question: {question}
Please only generate plans that are guidances required for the subsequent reasoning for the problem-solving. Do not include any specific calculation or numerical results. Your input may include previous round outputs content. You can consider the given contents as the initial state of the problem-solving.
"""

SOLVER_SYS = """You are the solver agent. Solve strictly according to the provided plans. Execute each step precisely and produce the final result.
Output the final result into {identifier}."""

SOLVER_USER = """Question: {question}
### Plans ###
{block}
### Plans ###
Follow the plans to solve the question step by step."""

CRITIC_SYS = """You are the critic agent. Review the solver's solution in detail, re-derive independently, and correct any mistakes.
Keep the review terse."""

CRITIC_USER = """Review the solution for: {question}
### Solution ###
{block}
### Solution ###

If corrections are needed, output the mistaken steps and the analysis, otherwise output 'Correct'."""

JUDGER_SYS = """You are the final judge. Audit only the final candidate and ensure it is correct."""

JUDGER_USER = """Final check for: {question}
### Solution ###
{block}
### Solution ###

If correct, only output the final answer without words, labels, and steps, and wrapped in {identifier}."""

# --- From maep/language/centralized.py (decentralized/full_decentralized/hybrid) ---
MATH_SYS = """You are the MathAgent. Solve the given question with clear steps.
Your input may include feedback from the Orchestrator from the previous round."""
MATH_USER = """Question: {question}
Provide a concise mathematical solution, showing key steps."""

SCIENCE_SYS = """You are the ScienceAgent. Analyze and solve the given question with scientific reasoning.
Your input may include feedback from the Orchestrator from the previous round."""
SCIENCE_USER = """Question: {question}
Explain your scientific reasoning and provide a final result."""

CODE_SYS = """You are the CodeAgent. Provide a self-contained Python function that solves the problem.
Your input may include feedback from the Orchestrator from the previous round."""
CODE_USER = """Question: {question}
Write a single self-contained Python function in a markdown code block that solves the problem."""

ORCHESTRATOR_SYS = """You are the Orchestrator Agent. Your task is to aggregate the solutions provided by the first-layer agents and produce a final answer wrapped in {identifier}."""

ORCHESTRATOR_USER = """Question: {question}

Here are the solutions from the expert agents:
=== Solutions ===
{block}
=== Solutions ===

Based on these inputs, provide the final answer wrapped in {identifier}."""

ORCHESTRATOR_FEEDBACK_SYS = """You are the Orchestrator Agent. Your task is to review the solutions provided by the first-layer agents in the current round.
Analyze the provided solutions, identify any issues or areas for improvement, and provide constructive feedback.
You may rewrite content, provide specific feedback, and offer improvement suggestions as needed.
Your feedback will be used by the agents in the next round to improve their solutions."""

ORCHESTRATOR_FEEDBACK_USER = """Question: {question}

Here are the solutions from the expert agents in the current round:
=== Solutions ===
{block}
=== Solutions ===

Review these solutions and provide feedback for the next round. If corrections are needed, specify the issues and suggest improvements. If the solutions are satisfactory, acknowledge them and provide guidance for further refinement."""

# --- From maep/language/debate.py ---
DEBATE_AGENT1_SYS = """You are Agent 1 in a debate multi-agent system. Solve the given question with clear steps and wrap your final answer in {identifier}.
Your input may include previous round debate content."""
DEBATE_AGENT1_USER = """Question: {question}
Provide a concise solution, showing key steps, and wrap the final answer in {identifier}."""

DEBATE_AGENT2_SYS = """You are Agent 2 in a debate multi-agent system. Solve the given question with clear steps and wrap your final answer in {identifier}.
Your input may include previous round debate content."""
DEBATE_AGENT2_USER = """Question: {question}
Provide a concise solution, showing key steps, and wrap the final answer in {identifier}."""

DEBATE_AGENT3_SYS = """You are Agent 3 in a debate multi-agent system. Solve the given question with clear steps and wrap your final answer in {identifier}.
Your input may include previous round debate content."""
DEBATE_AGENT3_USER = """Question: {question}
Provide a concise solution, showing key steps, and wrap the final answer in {identifier}."""
