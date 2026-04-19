"""
ReAct (Reasoning + Acting) utilities for tool-augmented language model inference.

This module provides utilities for:
1. Parsing tool calls from model outputs
2. Formatting tool results and conversation history
3. Managing ReAct iteration state and records
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MAX_REACT_ITERATIONS = 10
TOOL_RESULT_MAX_CHARS = 5000


# ============================================================================
# System Prompt Template
# ============================================================================

REACT_SYSTEM_SUFFIX = """
You have access to the following tools:

{tool_descriptions}

When you need to use a tool, respond with:
Thought: Your reasoning about what to do next
Action: tool_name
Action Input: {{"arg1": "value1", "arg2": "value2"}}

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
1. You MUST output ONLY ONE tool call per response (one Thought + one Action + one Action Input).
2. After outputting "Action Input: {{...}}", you MUST IMMEDIATELY STOP. Do NOT continue writing anything else.
3. The system will automatically execute the tool and return the "Observation:" result to you.
4. **NEVER generate Observation yourself** - it will be provided by the system after tool execution.
5. Do NOT chain multiple Thought/Action/Observation cycles in a single response.
6. Wait for the real Observation before your next reasoning step.

When you have gathered enough information and are ready to give a final answer, respond with:
Thought: Your final reasoning
Final Answer: Your comprehensive answer to the question

**IMPORTANT:** If you need to use a tool, your response must end with the Action Input line. Do not write anything after it.
"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReActStepRecord:
    """
    Record for a single step in the ReAct reasoning loop.
    
    Each react_step represents one complete cycle of: prompt → model response → optional tool call.
    
    Attributes:
        step_index: The step number (0-based).
        prompt: The full prompt used for this step.
        response: The raw model output text for this step.
        tool_calls: Tool call information if this step involves a tool call, otherwise None.
                    Structure: {"tool_name": str, "tool_arguments": str/dict, "tool_result": str}
        entropy: The entropy value for this step (torch.Tensor or None).
    """
    step_index: int
    prompt: str
    response: str
    tool_calls: Optional[dict] = None
    entropy: Any = None

    def to_dict(self) -> dict:
        """
        Convert to a dictionary for storage.
        
        Note: entropy is kept as torch.Tensor (not converted to list) so that
        BlockBasedStoreManager can automatically save it as a separate tensor file.
        
        Returns:
            A dictionary representation of this record.
        """
        return {
            "step_index": self.step_index,
            "prompt": self.prompt,
            "response": self.response,
            "tool_calls": self.tool_calls,
            "entropy": self.entropy,  # Keep as torch.Tensor for BlockBasedStoreManager
        }


@dataclass
class ReActResult:
    """
    Result container for a complete ReAct reasoning session.
    
    Attributes:
        final_response: The final answer text.
        final_entropy: The entropy value for the final step.
        steps: List of all step records in the session.
        total_iterations: Total number of iterations performed.
    """
    final_response: str
    final_entropy: Any
    steps: List[ReActStepRecord]
    total_iterations: int

    def to_dict(self) -> dict:
        """
        Convert to a JSON-serializable dictionary.
        
        Returns:
            A dictionary representation of this result.
        """
        return {
            "final_response": self.final_response,
            "total_iterations": self.total_iterations,
            "steps": [s.to_dict() for s in self.steps],
        }


# ============================================================================
# Parsing Functions
# ============================================================================

def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first complete JSON object from text using bracket counting.
    
    This function handles nested braces correctly and skips braces inside strings.
    
    Args:
        text: The text to search for a JSON object.
    
    Returns:
        The first complete JSON object string, or None if not found.
    """
    # Find the first '{'
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start_idx, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                # Found the matching closing brace
                return text[start_idx:i+1]
    
    # No matching closing brace found
    return None


def parse_tool_call(response: str) -> Optional[Tuple[str, dict]]:
    """
    Parse tool call from model output.
    
    Supports three formats in order of priority:
    1. ReAct standard format:
       Action: tool_name
       Action Input: {"arg": "value"}
    
    2. JSON code block format:
       ```json
       {"tool": "tool_name", "arguments": {"arg": "value"}}
       ```
       or
       ```json
       {"name": "tool_name", "arguments": {"arg": "value"}}
       ```
    
    3. TOOL_CALL format:
       TOOL_CALL: tool_name(arg="value")
    
    Args:
        response: The model output text to parse.
    
    Returns:
        A tuple of (tool_name, arguments_dict) if a tool call is found,
        None otherwise.
    """
    # Format 1: ReAct standard format (Action: / Action Input:) - supports multi-line JSON
    react_header_pattern = r"Action:\s*(.+?)\s*\n\s*Action Input:\s*"
    header_match = re.search(react_header_pattern, response, re.DOTALL)
    if header_match:
        tool_name = header_match.group(1).strip()
        rest = response[header_match.end():]
        # Find the first complete JSON object using bracket counting
        action_input_str = _extract_first_json_object(rest)
        if action_input_str:
            try:
                arguments = json.loads(action_input_str)
                logger.debug(f"Parsed ReAct tool call: {tool_name}")
                return (tool_name, arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse Action Input JSON: {action_input_str[:200]}...")
        else:
            # No JSON block found, try to use content up to next newline as simple value
            simple_val = rest.strip().split('\n')[0].strip()
            if simple_val:
                try:
                    arguments = json.loads(simple_val)
                    logger.debug(f"Parsed ReAct tool call (simple): {tool_name}")
                    return (tool_name, arguments)
                except json.JSONDecodeError:
                    # Treat it as a single string parameter
                    logger.debug(f"Action Input is not JSON, treating as query: {simple_val[:100]}")
                    return (tool_name, {"query": simple_val})

    # Format 2: JSON code block format
    # First find the code block, then extract JSON using bracket counting
    json_block_pattern = r'```(?:json)?\s*'
    block_match = re.search(json_block_pattern, response, re.DOTALL)
    if block_match:
        block_content = response[block_match.end():]
        # Find the closing ``` to limit our search
        closing_idx = block_content.find('```')
        if closing_idx != -1:
            block_content = block_content[:closing_idx]
        # Extract the first JSON object from the block
        json_str = _extract_first_json_object(block_content)
        if json_str:
            try:
                json_obj = json.loads(json_str)
                tool_name = json_obj.get("tool") or json_obj.get("name")
                arguments = json_obj.get("arguments", {})
                if tool_name:
                    tool_name = tool_name.strip()
                    logger.debug(f"Parsed JSON block tool call: {tool_name}")
                    return (tool_name, arguments)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON block: {json_str[:200] if json_str else 'empty'}")

    # Format 3: TOOL_CALL format
    tool_call_pattern = r'TOOL_CALL:\s*(\w+)\((.+?)\)'
    match = re.search(tool_call_pattern, response, re.DOTALL)
    if match:
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        # Parse key=value pairs
        arguments = {}
        # Match patterns like: key="value" or key='value' or key=value
        arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\s\)]+))'
        for arg_match in re.finditer(arg_pattern, args_str):
            key = arg_match.group(1)
            # Get value from whichever group matched
            value = arg_match.group(2) or arg_match.group(3) or arg_match.group(4)
            arguments[key] = value
        if arguments:
            logger.debug(f"Parsed TOOL_CALL format: {tool_name}")
            return (tool_name, arguments)

    logger.debug("No tool call found in response")
    return None


# ============================================================================
# Formatting Functions
# ============================================================================

def format_tool_result(tool_name: str, result: dict) -> str:
    """
    Format tool result as an Observation string.
    
    Args:
        tool_name: Name of the tool that was called.
        result: The result dictionary returned by the tool.
    
    Returns:
        Formatted observation string with truncation if necessary.
    """
    result_str = json.dumps(result, ensure_ascii=False, default=str)
    if len(result_str) > TOOL_RESULT_MAX_CHARS:
        result_str = result_str[:TOOL_RESULT_MAX_CHARS] + "... [truncated]"
    return f"Observation: [{tool_name}] {result_str}\n"


def format_react_history(history: List[dict]) -> str:
    """
    Format multi-turn ReAct history into a context string.
    
    Each history entry should contain:
    - "role": Either "assistant" or "tool"
    - "content": The content for that turn
    
    Args:
        history: List of history entries.
    
    Returns:
        Formatted history string.
    """
    parts = []
    for entry in history:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "assistant":
            parts.append(content)
        elif role == "tool":
            parts.append(content)
    return "\n".join(parts)


# ============================================================================
# Detection Functions
# ============================================================================

def is_final_answer(response: str) -> bool:
    """
    Check if the model output contains a final answer indicator.
    
    Args:
        response: The model output text to check.
    
    Returns:
        True if a final answer indicator is found, False otherwise.
    """
    indicators = [
        "Final Answer:",
        "FINAL ANSWER:",
        "final answer:",
        "\\boxed{",
        "\\boxed ",
    ]
    return any(indicator in response for indicator in indicators)


# ============================================================================
# Builder Functions
# ============================================================================

def build_react_system_suffix(tool_definitions: List[Dict]) -> str:
    """
    Build the ReAct system prompt suffix with tool descriptions.
    
    Args:
        tool_definitions: List of tool definition dictionaries. Each should contain:
            - "name": Tool name
            - "description": Tool description
            - "parameters" (optional): Parameter schema
    
    Returns:
        The formatted system prompt suffix with tool descriptions.
    """
    tool_descriptions_parts = []
    
    for tool_def in tool_definitions:
        # Support both function-calling nested structure and flat structure
        if "function" in tool_def:
            fn = tool_def["function"]
            name = fn.get("name", "unknown_tool")
            description = fn.get("description", "No description provided.")
            parameters = fn.get("parameters", {})
        else:
            # Flat structure
            name = tool_def.get("name", "unknown_tool")
            description = tool_def.get("description", "No description provided.")
            parameters = tool_def.get("parameters", {})
        
        tool_desc = f"- {name}: {description}"
        
        # Add parameter information if available
        if parameters:
            props = parameters.get("properties", {})
            required = parameters.get("required", [])
            if props:
                param_parts = []
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required
                    req_marker = " (required)" if is_required else " (optional)"
                    param_parts.append(f"    - {param_name} ({param_type}){req_marker}: {param_desc}")
                if param_parts:
                    tool_desc += "\n  Parameters:\n" + "\n".join(param_parts)
        
        tool_descriptions_parts.append(tool_desc)
    
    tool_descriptions = "\n".join(tool_descriptions_parts)
    
    return REACT_SYSTEM_SUFFIX.format(tool_descriptions=tool_descriptions)
