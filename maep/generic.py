"""Multi-agent base classes (state, output container, and BaseAgents).

The ReAct loop implementation lives in `maep.language.react_loop`.
The entropy-inference base class lives in `maep.inference_base`
and is re-exported here for backward compatibility.
"""

import os
import sys
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from lmbase.inference.base import InferInput

from lmbase.utils.tools import BaseContainer, BlockBasedStoreManager
from lmbase.inference.api_call import LangChainAPIInference
from lmbase.inference.model_call import LLMInference

from maep.inference_base import BaseEntropyInference  # re-export
from maep.language.react_utils import (
    MAX_REACT_ITERATIONS,
    build_react_system_suffix,
)
from maep.language.react_loop import ReActExecutor
from maep.prompts import get_identifier

__all__ = [
    "AgentState",
    "AgentReasonOutput",
    "BaseAgents",
    "BaseEntropyInference",
]


class AgentState(MessagesState):
    """
    Shared state container for both single-agent and sequential-agent runs.

    Purpose
    - Standardize prompt sourcing, per-stage outputs, and lightweight bookkeeping across agents.

    Fields
    - init_input: List[Any], arbitrary payload to carry input context (e.g., [{"question": "..."}]).
    - agent_results: Ordered list of per-stage outputs; each item is a dict like {"name": <agent_name>, "output": <string>}.
    - agent_executed: Ordered list of agent names that have executed; mirrors the sequence without payload.
    - cost: Per-stage metrics or accounting records (e.g., {agent_name: {"latency": seconds}}); schema is project-defined.
    - agent_system_msgs: Mapping {agent_name: system_msg} passed directly to the inference backend.
    - agent_user_msgs: Mapping {agent_name: user_msg_template}; templates may include placeholders (e.g., {question} or prior outputs) and are formatted at runtime.

    Constraints
    - Prompt maps (`agent_system_msgs`, `agent_user_msgs`) should remain stable during a run.
    - agent_results and agent_executed grow monotonically as stages execute.
    """

    init_input: List[Any]
    agent_results: List[Dict[str, Any]]
    agent_executed: List[str]

    cost: List[Dict[str, Any]]

    agent_system_msgs: Dict[str, str]
    agent_user_msgs: Dict[str, str]


@dataclass
class AgentReasonOutput(BaseContainer):
    """Output format of the pipeline builder."""

    final_state: AgentState
    results: Dict[str, Any]
    logs: List[str] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseAgents(ABC):
    """Abstract base class for agents in this project.

    !!Note that the graph node name should be the same as the agent name.!!
    Because the execute_agent function relies on the 'agent_name' to determine which prompt to be used.

    Purpose
    - Define a minimal, consistent interface for both single and sequential agents.
    - Standardize execution, graph construction, and output persistence naming.

    Notes
    - Implementations must read prompts from AgentState and write outputs to AgentState.
    - Implementations should avoid side effects beyond documented persistence.
    """

    def __init__(
        self,
        run_config: dict,
    ):
        self.run_config = run_config
        self.save_dir = self.run_config["save_folder"]
        # Configs for all agents
        self.agents_config = self.run_config["agents"]

        # Task type for dynamic prompt loading
        self.task_type = self.run_config["task_type"]

        # By default, we set only one lm for agents
        # for any agent with a different generation config, we only
        # change the generation config.
        self.agents_lm = None

        # Tool storage for ReAct execution (Optional)
        self._tools = None  # Optional[Dict[str, Any]]
        self._tool_definitions = None  # Optional[List[Dict]]

        self.define_agent_models()
        os.makedirs(self.save_dir, exist_ok=True)

        self.store_folder = os.path.join(self.save_dir, "traces")
        self.store_manager = BlockBasedStoreManager(
            folder=self.store_folder,
            file_format="json",
            block_size=500,
        )

    def _load_from_module(self, import_path: str) -> Any:
        """
        Load an object (variable, class, function) from a string format "module.path:ObjectName".
        If the object is a dictionary, it will select the appropriate value based on task_type.
        If the object is a string and contains {identifier} placeholder, it will be
        formatted with the task-specific identifier.

        Args:
            import_path: String in format "module.path:ObjectName"

        Returns:
            The loaded object, with task_type selection applied if applicable.
        """
        if ":" not in import_path:
            raise ValueError(
                f"Invalid import format: {import_path}. Expected 'module.path:ObjectName'"
            )

        module_path, obj_name = import_path.split(":")

        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Try adding current directory to path if not found
            if os.getcwd() not in sys.path:
                sys.path.append(os.getcwd())
            module = importlib.import_module(module_path)

        if not hasattr(module, obj_name):
            raise ValueError(f"Object '{obj_name}' not found in module '{module_path}'")

        obj = getattr(module, obj_name)

        # If the object is a dictionary, select the appropriate value based on task_type
        if isinstance(obj, dict):
            if self.task_type not in obj:
                raise ValueError(
                    f"Task type '{self.task_type}' not found in object '{obj_name}'. "
                    f"Available task types: {list(obj.keys())}"
                )
            obj = obj[self.task_type]

        if isinstance(obj, str) and "{identifier}" in obj:
            obj = obj.replace("{identifier}", get_identifier(self.task_type))

        return obj

    def _validate_task_type(self):
        """
        Validate the task_type parameter.

        Raises:
            ValueError: If task_type is not supported.
        """
        from maep.prompts import validate_task_type

        if not validate_task_type(self.task_type):
            raise ValueError(
                f"Unsupported task_type: {self.task_type}. "
                f"Must be one of {list(TASK_IDENTIFIERS.keys()) if 'TASK_IDENTIFIERS' in dir() else ['math', 'code', 'option']}"
            )

    def define_agent_models(self):
        """Construct HF inference as the agent LM.

        Requires `run_config` with keys:
        - lm_name: HF model identifier
        - inference_config: device/dtype and backend options
        - entropy_config: entropy-related configuration
        - generation_config: decoding hyperparameters
        """
        lm_type = self.run_config["lm_type"]
        model_name = self.run_config["lm_name"]

        if "api" in lm_type.lower():
            self.agents_lm = LangChainAPIInference(
                lm_name=model_name,
                generation_config=self.run_config["generation_config"],
            )
        else:
            self.agents_lm = LLMInference(
                lm_path=model_name,
                inference_config=self.run_config["inference_config"],
                generation_config=self.run_config["generation_config"],
            )

    def get_save_name(
        self,
        agent_name: str,
        execution_idx: int,
        **kwargs,
    ) -> str:
        """Return the filename (or full path) used to persist stage outputs.

        Parameters
        - agent_name: The logical name of the agent/stage.
        - execution_idx: The execution index used for disambiguation.
        - **kwargs: Optional naming options (e.g., dir, ext, prefix, suffix, timestamp).

        Returns
        - A deterministic string suitable for saving artifacts like pickled outputs.
        """
        return f"{agent_name}-{execution_idx}"

    @abstractmethod
    def execute_agent(self, state: AgentState, agent_name: str) -> AgentState:
        """Execute exactly one stage using prompts from the provided state.

        Due to that the langgraph does not support the additional argument, i.e. agent_name, here, one need to use the partial to bind the agent_name to the function during the node creation of the graph.

        Requirements
        - Read system/user prompts from AgentState for the current agent.
        - Produce a model output and write it to state.agent_results[agent_name].
        - Update state.execute_count[agent_name] (increment by 1).
        - Optionally persist artifacts using get_save_name.

        Returns
        - The updated AgentState after this stage completes.
        """

    @abstractmethod
    def graph(self) -> CompiledStateGraph:
        """Construct and compile the LangGraph for this agent or pipeline.

        Note that the `execute_agent` function should be bound with the 'agent_name' using partial.

        Returns
        - A CompiledStateGraph with entry point and edges defined.
        """

    @abstractmethod
    def _get_agent_prompts(self):
        """Initialize system and user prompts for all agents."""

    def run(
        self, batch: List[InferInput], tools=None, tool_definitions=None, **kwargs
    ) -> AgentReasonOutput:
        """Run the agent on a batch of inputs.

        Parameters
        - batch: A list of InferInput instances.
        - tools: Optional dict mapping tool names to callable tool instances.
        - tool_definitions: Optional list of tool definition dicts for ReAct prompts.

        Returns
        - AgentReasonOutput: The structured output containing final state and results.
        """
        # Store tools (Optional)
        self._tools = tools  # Optional[Dict[str, Any]]
        self._tool_definitions = tool_definitions  # Optional[List[Dict]]

        # 1. Get prompts
        agent_system_msgs, agent_user_msgs = self._get_agent_prompts()

        # 2. Build initial state
        state = {
            "init_input": batch,
            "agent_results": [],
            "agent_executed": [],
            "cost": [],
            "agent_system_msgs": agent_system_msgs,
            "agent_user_msgs": agent_user_msgs,
        }

        # 3. Compile graph
        app = self.graph()

        # 4. Run graph
        # Increase recursion limit if needed, though default is usually 25
        # For long events, we might need more.
        config = self.run_config["graph_config"]
        final_state = app.invoke(state, config=config)

        # 5. Construct AgentReasonOutput
        return AgentReasonOutput(
            final_state=final_state,
            results=None,
            logs=None,
            extras=None,
        )

    def _has_tools(self):
        """Check if tools are available for ReAct execution."""
        return self._tools is not None and len(self._tools) > 0

    def _get_react_system_suffix(self):
        """Generate ReAct system prompt suffix with tool definitions."""
        if not self._has_tools() or not self._tool_definitions:
            return ""
        return build_react_system_suffix(self._tool_definitions)

    def react_infer_batch(self, infer_inputs, max_iterations=None):
        """Batch inference with ReAct loop, recording entropy at each step.

        If no tools are available, falls back to standard infer_batch (zero behavior change).
        Otherwise delegates to ReActExecutor.
        """
        if not self._has_tools():
            return self.agents_lm.infer_batch(infer_inputs)

        executor = ReActExecutor(
            agents_lm=self.agents_lm,
            tools=self._tools,
            system_suffix=self._get_react_system_suffix(),
            max_iterations=max_iterations or MAX_REACT_ITERATIONS,
        )
        return executor.run_batch(infer_inputs)
