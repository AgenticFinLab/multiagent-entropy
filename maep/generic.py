"""
Implementation of the general terms and abstract base classes used across the project.

This file serves as the central documentation hub for:
1. Standard Dimension Notations
2. Entropy Inference Workflow (ASCII Diagram)
3. Common Tensor Shapes and Types

Standard Dimension Notations:
-----------------------------
B       : Batch size
L       : Sequence length (generic, after padding)
L_g     : Sequence length of generated tokens
V       : Vocabulary size
D_h     : Model hidden dimension (hidden size)

ASCII Diagram (Unified Entropy Inference Workflow):
---------------------------------------------------
Read messages (List[List[Dict]])
  ↓
apply_chat_template → prompts [B]
  ↓
tokenizer(prompts, padding=True) → input_ids [B, L], attention_mask [B, L]
  ↓
infer_entropy (HF / vLLM)
  - HF: Forward pass → Logits [B, L_g, V] → Entropy Calculation
  - vLLM: Generate / Probabilities → Logits [B, L_g, V] → Entropy Calculation
  ↓
Output:
  - Token-level Entropy
  - Token-level Logits
  - Generated Text (if applicable)

Concrete Step-by-Step Workflows:
--------------------------------
- HF workflow:
  Read messages → Encode → Forward pass to obtain logits → Compute Entropy per token.
- vLLM workflow:
  Read messages → Encode → Engine Generation (with logprobs) → Extract logits/probs → Compute Entropy per token.

Common Tensor Shapes and Types:
-------------------------------
- input_ids        : [B, L] (Long)
- attention_mask   : [B, L] (Long)
- logits           : [B, L_g, V] (Float)
- entropy          : [B, L_g] (Float)
"""

import os
import sys
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from lmbase.inference.base import InferInput, InferOutput
from lmbase.utils.tools import BaseContainer, BlockBasedStoreManager

from lmbase.inference.api_call import LangChainAPIInference
from lmbase.inference.model_call import LLMInference


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

        # By default, we set only one lm for agents
        # for any agent with a different generation config, we only
        # change the generation config.
        self.agents_lm = None

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

        Args:
            import_path: String in format "module.path:ObjectName"

        Returns:
            The loaded object.
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

        return getattr(module, obj_name)

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

    def run(self, batch: List[InferInput]) -> AgentReasonOutput:
        """Run the agent on a batch of inputs.

        Parameters
        - batch: A list of InferInput instances.

        Returns
        - AgentReasonOutput: The structured output containing final state and results.
        """
        # 1. Get prompts
        agent_system_msgs, agent_user_msgs = self._get_agent_prompts()

        # 2. Build initial state
        state = {
            "input": batch,
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


class BaseEntropyInference(ABC):
    """
    Abstract base class for inference with the entropy exposure.

    Config Members
    - lm_name: Backend model identifier.
    - inference_config: Backend/runtime settings (e.g., device, tensor_parallel).
    - generation_config: Decoding hyperparameters (e.g., `max_new_tokens`, `temperature`, `top_p`).

    """

    def __init__(
        self,
        *,
        lm_name: str,
        inference_config: Dict[str, Any],
        entropy_config: Dict[str, Any],
        generation_config: Dict[str, Any],
    ):
        # Backend model identifier
        self.lm_name = lm_name
        # Backend/runtime settings (e.g., device, tensor parallel)
        self.inference_config = inference_config
        # Entropy exposure settings
        self.entropy_config = entropy_config
        # Decoding hyperparameters (max_new_tokens, temperature, top_p)
        self.generation_config = generation_config
        # Whether to append generation prompt tokens when rendering chat messages
        self.add_generation_prompt = True

        # Runtime device (mandatory)
        # - Single-GPU: set `inference_config["device"]` to "cuda" or "cuda:0"
        # - Multi-GPU: vLLM uses `tensor_parallel_size` and env `CUDA_VISIBLE_DEVICES` for distribution;
        #              HF uses `device_map="auto"` for shard placement while `self.device` is the default tensor device
        # - No protective defaults; missing key in `inference_config` raises an error
        self.device = torch.device(self.inference_config["device"])
        # Runtime torch dtype (mandatory)
        # - Set `inference_config["torch_dtype"]` to a torch dtype string, e.g., "bfloat16", "float32"
        self.torch_dtype = getattr(torch, self.inference_config["torch_dtype"])

        # Tokenizer paired with the chosen backend (assigned in `load_model`)
        self.tokenizer = None
        # Active generation backend instance (assigned in `load_model`)
        self.model = None

        # Load the backend model and the tokenizer
        self.load_tokenizer()
        self.load_model()

    @abstractmethod
    def calculate_entropy(self, logits: Any) -> Any:
        """
        Calculate entropy from logits.

        Entropy H(x) = - Σ p(x) * log(p(x))

        Args:
            logits: Input logits tensor.
                    Shape: [B, L_g, V]
                    Type: torch.Float

        Returns:
            Calculated entropy tensor.
            Shape: [B, L_g]
            Type: torch.Float

        Dimensions:
            B   : Batch size
            L_g : Generated sequence length
            V   : Vocabulary size
        """

    @abstractmethod
    def load_model(self):
        """
        Initialize and load the active backend, then set shared members.

        Contract:
        - Decide backend via `inference_config['use_vllm']`:
          - If True: initialize vLLM engine (requires `tensor_parallel_size`, `gpu_memory_utilization`).
          - If False: initialize HF `AutoModelForCausalLM` with `self.torch_dtype`.
        - Do not return anything; assign members directly.

        Side effects (subclass must perform):
        - Set `self.model` to the active generation backend instance.
        - Optionally set any other backend-specific members required by the subclass.
        """

    @abstractmethod
    def load_tokenizer(self):
        """
        Initialize and assign `self.tokenizer` compatible with the chosen backend.

        Contract:
        - Create tokenizer for `self.lm_name` using the backend library.
        - Ensure `pad_token` is set (reuse EOS or add a new pad token).
        - Do not return anything; assign to `self.tokenizer`.
        """

    @abstractmethod
    def build_messages(
        self,
        infer_inputs: List[InferInput],
    ) -> List[List[Dict]]:
        """
        Build chat messages for a batch.

        Args:
            infer_inputs: List of InferInput objects containing user/system messages.
                          Length: B

        Returns:
            A list (batch) of message lists.
            Shape: List[List[Dict]] (Length: B)
            Structure: [[{"role": "user", "content": "..."}], ...]
        """

    @abstractmethod
    def encode_messages(
        self, messages_batch: List[List[Dict]]
    ) -> Tuple[List[str], Any, Any, List[List[str]]]:
        """
        Encode a batch of messages into model inputs.

        Args:
            messages_batch: Batch of message lists.
                            Length: B

        Returns:
            Tuple containing:
            - prompts: List[str], length B.
            - input_ids: Tensor, shape [B, L]. Type: torch.Long
            - attention_mask: Tensor, shape [B, L]. Type: torch.Long
            - tokens_batch: List[List[str]], B lists of tokens, each length L.

        Dimensions:
            B : Batch size
            L : Padded input sequence length
        """

    def infer_entropy_hf(
        self,
        input_ids: Any,
        attention_mask: Any = None,
    ) -> Tuple[Any, Any]:
        """
        Execute entropy inference for HuggingFace backend.

        This method performs generation and returns the full output object and computed entropy.

        Args:
            input_ids: Token IDs tensor.
                       Shape: [B, L]
            attention_mask: Attention mask tensor (optional).
                            Shape: [B, L]

        Returns:
            Tuple containing:
            - outputs: HF ModelOutput object (contains sequences, scores/logits).
            - entropy: Calculated entropy tensor.
                       Shape: [B, L_g]

        Dimensions:
            B   : Batch size
            L   : Input sequence length
            L_g : Generated sequence length
        """
        pass

    def infer_entropy_vllm(
        self,
        input_ids: Any,
    ) -> Tuple[Any, Any]:
        """
        Execute entropy inference for vLLM backend.

        This method performs generation via vLLM engine and computes entropy from logprobs.

        Args:
            input_ids: Token IDs tensor or list.

        Returns:
            Tuple containing:
            - outputs: vLLM output objects.
            - entropy: Calculated entropy tensor.
                       Shape: [B, L_g]

        Dimensions:
            B   : Batch size
            L_g : Generated sequence length
        """
        pass

    @abstractmethod
    def infer_batch(self, infer_inputs: List[InferInput]) -> List[InferOutput]:
        """
        Orchestrate entropy inference for a batch of inputs.

        This method acts as the main controller. It must implement the following pipeline:
        1. Build and encode messages.
        2. Dispatch to `infer_entropy_hf` or `infer_entropy_vllm` based on configuration.
        3. Package results into `InferOutput`.

        Args:
            infer_inputs: List of inputs to process.
                          Length: B

        Returns:
            List[InferOutput]: List of output objects, one for each input.
                               Length: B
        """
