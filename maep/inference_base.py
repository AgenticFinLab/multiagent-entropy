"""Abstract base class for entropy-aware inference backends.

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

Common Tensor Shapes and Types:
-------------------------------
- input_ids        : [B, L] (Long)
- attention_mask   : [B, L] (Long)
- logits           : [B, L_g, V] (Float)
- entropy          : [B, L_g] (Float)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from lmbase.inference.base import InferInput, InferOutput


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
