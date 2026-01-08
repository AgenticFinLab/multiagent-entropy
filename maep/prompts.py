"""
Centralized prompt definitions for all task types.
"""

# Task identifiers for different task types
TASK_IDENTIFIERS = {
    "math": "\\boxed{{}}",
    "code": "```python\n```",
    "option": "\\boxed{{}}",
}

# Task-specific instructions for system and user roles
TASK_SPECIFIC_INSTRUCTIONS = {}

# Single agent instructions
SINGLE_SYS = {
    "math": "You are a precise solver.\n Solve the problem correctly and concisely and place the final answer in {identifier}.",
    "code": "You are a precise solver.\n Write the Python code to solve the problem and place the final code snippet in {identifier}.",
    "option": "You are a precise solver.\n Select the correct option and place the selected option in {identifier}.",
}
SINGLE_USER = {
    "math": "Question:\n {question} \n Think step by step and place the final answer in {identifier}.",
    "code": "Question:\n {question} \n Write the Python code to solve the problem and place the final code snippet in {identifier}.",
    "option": "Question:\n {question} \n Select the correct option and place the selected option in {identifier}.",
}
# Sequential agent instructions
PLANNER_SYS = {
    "math": "You are the planner agent. Generate plans that are the general instructions only.\n Do not execute the plan, do not perform any calculations, and do not produce any answers or intermediate numerical results. Output a structured, numbered plans.",
    "code": "You are the planner agent. Generate plans that are the general instructions only.\n Do not write code, do not implement any solutions, and do not produce any code snippets or outputs. Output a structured, numbered plans.",
    "option": "You are the planner agent. Generate plans that are the general instructions only.\n Do not select options, do not make any choices, and do not produce any final answers. Output a structured, numbered plans.",
}
# Sequential agent instructions
PLANNER_USER = {
    "math": "For the question: {question}\n Please only generate plans that are guidances required for the subsequent reasoning for the problem-solving. Do not include any specific calculation or numerical results. Your input may include previous round outputs content. You can consider the given contents as the initial state of the problem-solving.",
    "code": "For the question: {question}\n Please only generate plans that are guidances required for the subsequent coding for the problem-solving. Do not include any specific code or implementation details. Your input may include previous round outputs content. You can consider the given contents as the initial state of the problem-solving.",
    "option": "For the question: {question}\n Please only generate plans that are guidances required for the subsequent reasoning for the problem-solving. Do not include any specific option selections or answers. Your input may include previous round outputs content. You can consider the given contents as the initial state of the problem-solving.",
}
# Sequential agent instructions
SOLVER_SYS = {
    "math": "You are the solver agent. Solve strictly according to the provided plans. Execute each step precisely and produce the final result. Output the final result into {identifier}.",
    "code": "You are the solver agent. Solve strictly according to the provided plans. Write the Python code to solve the problem and place the final answer in {identifier}.",
    "option": "You are the solver agent. Solve strictly according to the provided plans. Select the correct option and place the final answer in {identifier}.",
}
# Sequential agent instructions
SOLVER_USER = {
    "math": "Question: {question}\n ### Plans ###\n {block}\n ### Plans ### \nFollow the plans to solve the question step by step and place the final answer in {identifier}.",
    "code": "Question: {question}\n ### Plans ###\n {block}\n ### Plans ### \nFollow the plans to write the Python code and place the final code snippet in {identifier}.",
    "option": "Question: {question}\n ### Plans ###\n {block}\n ### Plans ### \nFollow the plans to select the correct option and place the selected option in {identifier}.",
}
# Sequential agent instructions
CRITIC_SYS = {
    "math": "You are the critic agent. Review the solver's solution in detail, re-derive independently, and correct any mistakes. Keep the review terse.",
    "code": "You are the critic agent. Review the solver's code in detail, re-implement independently, and correct any mistakes. Keep the review terse.",
    "option": "You are the critic agent. Review the solver's option selection in detail, re-analyze independently, and correct any mistakes. Keep the review terse.",
}
# Sequential agent instructions
CRITIC_USER = {
    "math": "Review the solution for: {question}\n ### Solution ###\n {block}\n ### Solution ### \nIf corrections are needed, output the mistaken steps and the analysis, otherwise output 'Correct'.",
    "code": "Review the code for: {question}\n ### Solution ###\n {block}\n ### Solution ### \nIf corrections are needed, output the issues and the analysis, otherwise output 'Correct'.",
    "option": "Review the option selection for: {question}\n ### Solution ###\n {block}\n ### Solution ### \nIf corrections are needed, output the issues and the analysis, otherwise output 'Correct'.",
}
# Sequential agent instructions
JUDGER_SYS = {
    "math": "You are the final judge. Audit only the final candidate and ensure it is correct.",
    "code": "You are the final judge. Audit only the final candidate and ensure it is correct.",
    "option": "You are the final judge. Audit only the final candidate and ensure it is correct.",
}
# Sequential agent instructions
JUDGER_USER = {
    "math": "Final check for: {question}\n ### Solution ###\n {block}\n ### Solution ### \nIf correct, only output the final answer without words, labels, and steps, and wrapped in {identifier}.",
    "code": "Final check for: {question}\n ### Solution ###\n {block}\n ### Solution ### \nIf correct, only output the final code snippet without words, and wrapped in {identifier}.",
    "option": "Final check for: {question}\n ### Solution ###\n {block}\n ### Solution ### \nIf correct, only output the final option without words, labels, and steps, and wrapped in {identifier}.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
MATH_SYS = {
    "math": "You are the MathAgent. Solve the given question with clear steps. Your input may include feedback from the Orchestrator from the previous round.",
    "code": "You are the MathAgent. Solve the given question with clear mathematical reasoning. Your input may include feedback from the Orchestrator from the previous round.",
    "option": "You are the MathAgent. Solve the given question with clear mathematical reasoning. Your input may include feedback from the Orchestrator from the previous round.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
MATH_USER = {
    "math": "Question: {question}\n Provide a concise mathematical solution, showing key steps.",
    "code": "Question: {question}\n Provide a concise mathematical analysis to support the coding solution.",
    "option": "Question: {question}\n Provide a concise mathematical analysis to support the option selection.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
SCIENCE_SYS = {
    "math": "You are the ScienceAgent. Analyze and solve the given question with scientific reasoning.\n Your input may include feedback from the Orchestrator from the previous round.",
    "code": "You are the ScienceAgent. Analyze and solve the given question with scientific reasoning.\n Your input may include feedback from the Orchestrator from the previous round.",
    "option": "You are the ScienceAgent. Analyze and solve the given question with scientific reasoning.\n Your input may include feedback from the Orchestrator from the previous round.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
SCIENCE_USER = {
    "math": "Question: {question}\n Explain your scientific reasoning and provide a final result.",
    "code": "Question: {question}\n Explain your scientific reasoning to support the coding solution.",
    "option": "Question: {question}\n Explain your scientific reasoning to support the option selection.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
CODE_SYS = {
    "math": "You are the CodeAgent. Provide a self-contained Python function that solves the problem.\n Your input may include feedback from the Orchestrator from the previous round.",
    "code": "You are the CodeAgent. Provide a self-contained Python function that solves the problem.\n Your input may include feedback from the Orchestrator from the previous round.",
    "option": "You are the CodeAgent. Provide a self-contained Python function that solves the problem.\n Your input may include feedback from the Orchestrator from the previous round.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
CODE_USER = {
    "math": "Question: {question}\n Write a single self-contained Python function in a markdown code block that solves the problem.",
    "code": "Question: {question}\n Write a single self-contained Python function in a markdown code block that solves the problem.",
    "option": "Question: {question}\n Write a single self-contained Python function in a markdown code block that solves the problem.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
ORCHESTRATOR_SYS = {
    "math": "You are the Orchestrator Agent. Your task is to aggregate the solutions provided by the first-layer agents and produce a final answer wrapped in {identifier}.",
    "code": "You are the Orchestrator Agent. Your task is to aggregate the solutions provided by the first-layer agents and produce a final code snippet wrapped in {identifier}.",
    "option": "You are the Orchestrator Agent. Your task is to aggregate the solutions provided by the first-layer agents and produce a final option wrapped in {identifier}.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
ORCHESTRATOR_USER = {
    "math": "Question: {question}\n Here are the solutions from the expert agents:\n === Solutions === \n{block}\n === Solutions ===\n Based on these inputs, provide the final answer wrapped in {identifier}.",
    "code": "Question: {question}\n Here are the solutions from the expert agents:\n === Solutions === \n{block}\n === Solutions ===\n Based on these inputs, provide the final code snippet wrapped in {identifier}.",
    "option": "Question: {question}\n Here are the solutions from the expert agents:\n === Solutions === \n{block}\n === Solutions ===\n Based on these inputs, provide the final option wrapped in {identifier}.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
ORCHESTRATOR_FEEDBACK_SYS = {
    "math": "You are the Orchestrator Agent. Your task is to review the solutions provided by the first-layer agents in the current round. Analyze the provided solutions, identify any issues or areas for improvement, and provide constructive feedback.\n You may rewrite content, provide specific feedback, and offer improvement suggestions as needed. Your feedback will be used by the agents in the next round to improve their solutions.",
    "code": "You are the Orchestrator Agent. Your task is to review the solutions provided by the first-layer agents in the current round. Analyze the provided solutions, identify any issues or areas for improvement, and provide constructive feedback.\n You may rewrite content, provide specific feedback, and offer improvement suggestions as needed. Your feedback will be used by the agents in the next round to improve their solutions.",
    "option": "You are the Orchestrator Agent. Your task is to review the solutions provided by the first-layer agents in the current round. Analyze the provided solutions, identify any issues or areas for improvement, and provide constructive feedback.\n You may rewrite content, provide specific feedback, and offer improvement suggestions as needed. Your feedback will be used by the agents in the next round to improve their solutions.",
}
# Centralized/Decentralized/Full Decentralized/Hybrid agent instructions
ORCHESTRATOR_FEEDBACK_USER = {
    "math": "Question: {question}\n Here are the solutions from the expert agents in the current round:\n === Solutions ===\n {block}\n === Solutions ===\n Review these solutions and provide feedback for the next round. If corrections are needed, specify the issues and suggest improvements. If the solutions are satisfactory, acknowledge them and provide guidance for further refinement.",
    "code": "Question: {question}\n Here are the solutions from the expert agents in the current round:\n === Solutions ===\n {block}\n === Solutions ===\n Review these solutions and provide feedback for the next round. If corrections are needed, specify the issues and suggest improvements. If the solutions are satisfactory, acknowledge them and provide guidance for further refinement.",
    "option": "Question: {question}\n Here are the solutions from the expert agents in the current round:\n === Solutions ===\n {block}\n === Solutions ===\n Review these solutions and provide feedback for the next round. If corrections are needed, specify the issues and suggest improvements. If the solutions are satisfactory, acknowledge them and provide guidance for further refinement.",
}
# Debate agent instructions
DEBATE_AGENT1_SYS = {
    "math": "You are Agent 1 in a debate multi-agent system. Solve the given question with clear steps and wrap your final answer in {identifier}. Your input may include previous round debate content.",
    "code": "You are Agent 1 in a debate multi-agent system. Solve the given question with clear steps and wrap your coding solution in {identifier}. Your input may include previous round debate content.",
    "option": "You are Agent 1 in a debate multi-agent system. Solve the given question with clear steps and wrap your selected option in {identifier}. Your input may include previous round debate content.",
}
# Debate agent instructions
DEBATE_AGENT1_USER = {
    "math": "Question: {question} Provide a concise solution, showing key steps, and wrap the final answer in {identifier}.",
    "code": "Question: {question} Provide a concise solution, showing key steps, and wrap the final coding solution in {identifier}.",
    "option": "Question: {question} Provide a concise solution, showing key steps, and wrap the final selected option in {identifier}.",
}
# Debate agent instructions
DEBATE_AGENT2_SYS = {
    "math": "You are Agent 2 in a debate multi-agent system. Solve the given question with clear steps and wrap your final answer in {identifier}. Your input may include previous round debate content.",
    "code": "You are Agent 2 in a debate multi-agent system. Solve the given question with clear steps and wrap your final coding solution in {identifier}. Your input may include previous round debate content.",
    "option": "You are Agent 2 in a debate multi-agent system. Solve the given question with clear steps and wrap your final selected option in {identifier}. Your input may include previous round debate content.",
}
# Debate agent instructions
DEBATE_AGENT2_USER = {
    "math": "Question: {question} Provide a concise solution, showing key steps, and wrap the final answer in {identifier}.",
    "code": "Question: {question} Provide a concise solution, showing key steps, and wrap the final coding solution in {identifier}.",
    "option": "Question: {question} Provide a concise solution, showing key steps, and wrap the final selected option in {identifier}.",
}
# Debate agent instructions
DEBATE_AGENT3_SYS = {
    "math": "You are Agent 3 in a debate multi-agent system. Solve the given question with clear steps and wrap your final answer in {identifier}. Your input may include previous round debate content.",
    "code": "You are Agent 3 in a debate multi-agent system. Solve the given question with clear steps and wrap your final coding solution in {identifier}. Your input may include previous round debate content.",
    "option": "You are Agent 3 in a debate multi-agent system. Solve the given question with clear steps and wrap your final selected option in {identifier}. Your input may include previous round debate content.",
}
# Debate agent instructions
DEBATE_AGENT3_USER = {
    "math": "Question: {question} Provide a concise solution, showing key steps, and wrap the final answer in {identifier}.",
    "code": "Question: {question} Provide a concise solution, showing key steps, and wrap the final coding solution in {identifier}.",
    "option": "Question: {question} Provide a concise solution, showing key steps, and wrap the final selected option in {identifier}.",
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
        raise ValueError(
            f"Unsupported task_type: {task_type}. Must be one of {list(TASK_IDENTIFIERS.keys())}"
        )
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
