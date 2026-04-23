"""ReAct loop executor extracted from BaseAgents.react_infer_batch.

Encapsulates per-sample ReAct iteration state and the weak-model guards
(repeated tool calls, parse failures, unknown tools).
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from maep.language.react_utils import (
    MAX_REACT_ITERATIONS,
    ReActResult,
    ReActStepRecord,
    format_tool_result,
    is_final_answer,
    parse_tool_call,
)

logger = logging.getLogger(__name__)

# Threshold for weak-model guards (repeated calls / parse failures / unknown tools)
WEAK_MODEL_LIMIT = 3


@dataclass
class _ReActLoopState:
    """Per-sample mutable state of the ReAct loop."""

    sample_idx: int
    enhanced_system: str
    current_user_msg: str
    react_steps: List[ReActStepRecord] = field(default_factory=list)
    iteration: int = 0
    conversation_history: str = ""
    final_output: Any = None
    seen_tool_calls: Set[Tuple[str, str]] = field(default_factory=set)
    tool_name_counts: Dict[str, int] = field(default_factory=dict)
    parse_failure_streak: int = 0
    unknown_tool_count: int = 0
    augmented_user_msg: str = ""

    def build_augmented_user_msg(self) -> str:
        if self.conversation_history:
            self.augmented_user_msg = (
                self.current_user_msg + "\n\n" + self.conversation_history
            )
        else:
            self.augmented_user_msg = self.current_user_msg
        return self.augmented_user_msg


class ReActExecutor:
    """Run the ReAct loop for a batch of inputs, one sample at a time."""

    def __init__(
        self,
        agents_lm: Any,
        tools: Dict[str, Any],
        system_suffix: str,
        max_iterations: int = MAX_REACT_ITERATIONS,
    ):
        self.agents_lm = agents_lm
        self.tools = tools
        self.system_suffix = system_suffix
        self.max_iterations = max_iterations

    # ----- public ----------------------------------------------------------

    def run_batch(self, infer_inputs: List[Any]) -> List[Any]:
        return [
            self._run_single(idx, infer_input)
            for idx, infer_input in enumerate(infer_inputs)
        ]

    # ----- per-sample loop --------------------------------------------------

    def _run_single(self, sample_idx: int, infer_input: Any) -> Any:
        state = _ReActLoopState(
            sample_idx=sample_idx,
            enhanced_system=infer_input.system_msg + "\n\n" + self.system_suffix,
            current_user_msg=infer_input.user_msg,
        )

        while state.iteration < self.max_iterations:
            step_output = self._step_inference(state, infer_input)
            response = step_output.response
            step_entropy = self._extract_entropy(step_output)
            tool_call = parse_tool_call(response)

            if tool_call is not None:
                self._handle_tool_call(state, tool_call, response, step_entropy)
            elif is_final_answer(response):
                self._record_step(state, response, None, step_entropy)
                state.final_output = step_output
                logger.info(
                    f"[ReAct] Sample {sample_idx}: Final answer at step {state.iteration}"
                )
                break
            else:
                self._handle_parse_failure(state, response, step_entropy)

            state.iteration += 1

        if state.final_output is None:
            self._handle_max_iterations(state, infer_input)

        return self._build_final_output(state)

    # ----- inference + parsing ---------------------------------------------

    def _step_inference(self, state: _ReActLoopState, infer_input: Any) -> Any:
        augmented = state.build_augmented_user_msg()
        single_input = type(infer_input)(
            system_msg=state.enhanced_system,
            user_msg=augmented,
        )
        if hasattr(infer_input, "messages") and infer_input.messages:
            single_input.messages = infer_input.messages

        step_outputs = self.agents_lm.infer_batch([single_input])
        step_output = step_outputs[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return step_output

    @staticmethod
    def _extract_entropy(step_output: Any) -> Any:
        return (
            step_output.extras.get("entropy", None)
            if step_output.extras
            else None
        )

    # ----- branch handlers --------------------------------------------------

    def _handle_tool_call(
        self,
        state: _ReActLoopState,
        tool_call: Tuple[str, dict],
        response: str,
        step_entropy: Any,
    ) -> None:
        tool_name, tool_args = tool_call
        state.parse_failure_streak = 0  # reset on any successful parse
        state.tool_name_counts[tool_name] = (
            state.tool_name_counts.get(tool_name, 0) + 1
        )

        # weak-model guard: same tool name called too many times
        if state.tool_name_counts[tool_name] >= WEAK_MODEL_LIMIT:
            logger.warning(
                f"[ReAct] Sample {state.sample_idx}, Step {state.iteration}: "
                f"Tool '{tool_name}' called {state.tool_name_counts[tool_name]} times "
                f"(>{WEAK_MODEL_LIMIT}); forcing final answer"
            )
            redirect_msg = (
                f"Observation: [system] You have called '{tool_name}' "
                f"{WEAK_MODEL_LIMIT} times already. Stop calling tools and "
                f"provide your Final Answer now based on information gathered.\n"
            )
            self._force_final(
                state, response, step_entropy, redirect_msg,
                tool_calls={
                    "tool_name": tool_name,
                    "tool_arguments": tool_args,
                    "tool_result": redirect_msg,
                },
            )
            return

        # detect repeated tool call (same args) to break infinite loops
        try:
            call_signature = (tool_name, json.dumps(tool_args, sort_keys=True))
        except (TypeError, ValueError):
            call_signature = (tool_name, str(tool_args))

        if call_signature in state.seen_tool_calls:
            logger.warning(
                f"[ReAct] Sample {state.sample_idx}, Step {state.iteration}: "
                f"Repeated tool call detected for '{tool_name}', redirecting "
                f"model to give final answer"
            )
            redirect_msg = (
                f"Observation: [system] You have already called '{tool_name}' with the same "
                f"arguments before. Do NOT repeat the same tool call. "
                f"Based on the information already gathered, please provide your Final Answer now.\n"
            )
            self._force_final(
                state, response, step_entropy, redirect_msg,
                tool_calls={
                    "tool_name": tool_name,
                    "tool_arguments": tool_args,
                    "tool_result": redirect_msg,
                },
            )
            return

        state.seen_tool_calls.add(call_signature)

        # execute tool (or report unknown)
        if tool_name in self.tools:
            tool_result_str = self._execute_tool(state, tool_name, tool_args)
        else:
            unknown_handled = self._handle_unknown_tool(
                state, tool_name, tool_args, response, step_entropy
            )
            if unknown_handled:
                return  # forced final, control already advanced
            tool_result_str = format_tool_result(
                tool_name,
                {"success": False, "error": f"Unknown tool: {tool_name}"},
            )

        self._record_step(
            state,
            response,
            {
                "tool_name": tool_name,
                "tool_arguments": tool_args,
                "tool_result": tool_result_str,
            },
            step_entropy,
        )
        state.conversation_history += response + "\n" + tool_result_str + "\n"

    def _execute_tool(
        self, state: _ReActLoopState, tool_name: str, tool_args: dict
    ) -> str:
        tool_instance = self.tools[tool_name]
        try:
            loop = asyncio.new_event_loop()
            try:
                tool_result = loop.run_until_complete(
                    tool_instance(arguments=tool_args)
                )
            finally:
                loop.close()
            tool_result_str = format_tool_result(tool_name, tool_result)
            logger.info(
                f"[ReAct] Sample {state.sample_idx}, Step {state.iteration}: "
                f"Called tool '{tool_name}' successfully"
            )
        except Exception as e:
            tool_result = {"success": False, "error": str(e)}
            tool_result_str = format_tool_result(tool_name, tool_result)
            logger.warning(
                f"[ReAct] Sample {state.sample_idx}, Step {state.iteration}: "
                f"Tool '{tool_name}' failed: {e}"
            )
        return tool_result_str

    def _handle_unknown_tool(
        self,
        state: _ReActLoopState,
        tool_name: str,
        tool_args: dict,
        response: str,
        step_entropy: Any,
    ) -> bool:
        """Return True if a forced-final happened; caller should return."""
        state.unknown_tool_count += 1
        logger.warning(
            f"[ReAct] Sample {state.sample_idx}, Step {state.iteration}: "
            f"Unknown tool '{tool_name}' (count={state.unknown_tool_count})"
        )
        if state.unknown_tool_count >= WEAK_MODEL_LIMIT:
            available = ", ".join(self.tools.keys()) or "none"
            redirect_msg = (
                f"Observation: [system] You have called {state.unknown_tool_count} "
                f"unknown tools. Available tools: [{available}]. Stop calling "
                f"tools and provide your Final Answer now.\n"
            )
            self._force_final(
                state, response, step_entropy, redirect_msg,
                tool_calls={
                    "tool_name": tool_name,
                    "tool_arguments": tool_args,
                    "tool_result": redirect_msg,
                },
            )
            return True
        return False

    def _handle_parse_failure(
        self, state: _ReActLoopState, response: str, step_entropy: Any
    ) -> None:
        state.parse_failure_streak += 1
        logger.warning(
            f"[ReAct] Sample {state.sample_idx}: Step {state.iteration} - "
            f"tool call parse failed, not a final answer "
            f"(streak={state.parse_failure_streak})"
        )
        # Always record the step (matching legacy behavior)
        self._record_step(state, response, None, step_entropy)

        if state.parse_failure_streak >= WEAK_MODEL_LIMIT:
            logger.warning(
                f"[ReAct] Sample {state.sample_idx}: {state.parse_failure_streak} "
                f"consecutive parse failures; forcing final answer"
            )
            redirect_msg = (
                "Observation: [system] Your last responses could not be "
                "parsed as either an Action or a Final Answer. You MUST "
                "respond with 'Final Answer: <your answer>' now.\n"
            )
            state.conversation_history += response + "\n" + redirect_msg + "\n"
            return

        # let the model retry; just append the raw response
        state.conversation_history += response + "\n"

    # ----- shared helpers ---------------------------------------------------

    def _record_step(
        self,
        state: _ReActLoopState,
        response: str,
        tool_calls: Optional[dict],
        step_entropy: Any,
    ) -> None:
        state.react_steps.append(
            ReActStepRecord(
                step_index=state.iteration,
                prompt=state.augmented_user_msg,
                response=response,
                tool_calls=tool_calls,
                entropy=step_entropy,
            )
        )

    def _force_final(
        self,
        state: _ReActLoopState,
        response: str,
        step_entropy: Any,
        redirect_msg: str,
        tool_calls: Optional[dict],
    ) -> None:
        """Inject system observation forcing the model to give Final Answer.

        Note: caller returns and _run_single performs the +1 iteration increment.
        """
        self._record_step(state, response, tool_calls, step_entropy)
        state.conversation_history += response + "\n" + redirect_msg + "\n"

    # ----- max-iter fallback + finalize ------------------------------------

    def _handle_max_iterations(
        self, state: _ReActLoopState, infer_input: Any
    ) -> None:
        logger.warning(
            f"[ReAct] Sample {state.sample_idx}: "
            f"Max iterations ({self.max_iterations}) reached"
        )
        final_user_msg = state.current_user_msg + "\n\n" + state.conversation_history
        final_user_msg += (
            "\n\nYou have used all available tool calls. Please provide your "
            "Final Answer now based on the information gathered above."
        )
        final_input = type(infer_input)(
            system_msg=state.enhanced_system,
            user_msg=final_user_msg,
        )
        final_outputs = self.agents_lm.infer_batch([final_input])
        state.final_output = final_outputs[0]

        final_entropy = self._extract_entropy(state.final_output)
        # Note: legacy code records this with augmented_user_msg replaced by final_user_msg
        state.react_steps.append(
            ReActStepRecord(
                step_index=state.iteration,
                prompt=final_user_msg,
                response=state.final_output.response,
                tool_calls=None,
                entropy=final_entropy,
            )
        )

    def _build_final_output(self, state: _ReActLoopState) -> Any:
        # Build ReActResult (kept for parity though not returned directly)
        _ = ReActResult(
            final_response=state.final_output.response,
            final_entropy=self._extract_entropy(state.final_output),
            steps=state.react_steps,
            total_iterations=len(state.react_steps),
        )

        if state.final_output.extras is None:
            state.final_output.extras = {}
        state.final_output.extras["react_steps"] = [
            s.to_dict() for s in state.react_steps
        ]
        state.final_output.extras["total_react_iterations"] = len(state.react_steps)
        state.final_output.extras["has_tool_calls"] = any(
            s.tool_calls is not None for s in state.react_steps
        )
        return state.final_output
