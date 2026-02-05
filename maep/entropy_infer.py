"""
Entropy Inference Implementation for Hugging Face Models.

This module provides the `HFEntropyInference` class, which implements the
`BaseEntropyInference` interface for models loaded via the Hugging Face `transformers` library.
It allows for computing token-level logits and entropy for a batch of inputs.
"""

import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmbase.inference.base import InferInput, InferOutput
from maep.generic import BaseEntropyInference


class HFEntropyInference(BaseEntropyInference):
    """
    Hugging Face implementation of the Entropy Inference.

    This class handles the loading of HF models and tokenizers,
    and implements the pipeline to compute token-level entropy
    and logits for a given batch of inputs.

    Attributes:
        model (AutoModelForCausalLM): The loaded HF model.
        tokenizer (AutoTokenizer): The loaded HF tokenizer.
    """

    def load_model(self):
        """
        Load the Hugging Face model based on the configuration.

        This method initializes `self.model` using `AutoModelForCausalLM`.
        It respects the `inference_config` for device placement and data types.
        Supports multi-GPU via device_map configuration.
        """
        print(f"[HFEntropyInference] Loading model: {self.lm_name}")
        model_kwargs = self.inference_config.get("model_kwargs", {})

        # Check if device_map is specified in inference_config for multi-GPU support
        # device_map can be "auto", "balanced", or a custom dict mapping layers to devices
        device_map = self.inference_config.get("device_map", None)

        # Enable gradient checkpointing for memory optimization
        use_gradient_checkpointing = self.inference_config.get(
            "use_gradient_checkpointing", True
        )

        if device_map:
            # Multi-GPU mode: use device_map for parallel inference
            self.model = AutoModelForCausalLM.from_pretrained(
                self.lm_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map=device_map,
                **model_kwargs,
            )
        else:
            # Single-GPU mode: load model and move to specified device
            self.model = AutoModelForCausalLM.from_pretrained(
                self.lm_name,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                **model_kwargs,
            )
            self.model.to(self.device)

        # Enable gradient checkpointing for memory optimization
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print(f"[HFEntropyInference] Gradient checkpointing enabled")

        self.model.eval()

    def load_tokenizer(self):
        # Causality & Principle
        # - Create tokenizer bound to `lm_name`; must ensure a valid `pad_token_id` for batch padding.
        # Sizes & Types
        # - tokenizer outputs: `input_ids [B, L]`, `attention_mask [B, L]`, dtype=torch.long
        # - pad/eos tokens are strings; IDs are integers
        tokenizer = AutoTokenizer.from_pretrained(self.lm_name, use_fast=True)
        # Set padding side to 'left' for decoder-only architecture
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer = tokenizer

    def build_messages(
        self,
        infer_inputs: List[InferInput],
    ) -> List[List[Dict]]:
        # Causality & Principle
        # - Normalize inputs into a deterministic chat-format list for template rendering.
        # Sizes & Types
        # - infer_inputs: length B
        # - returns: List[List[Dict]] length B; each inner list: {"role": str, "content": str}
        batch: List[List[Dict]] = []
        for inp in infer_inputs:
            if inp.messages is not None:
                batch.append(inp.messages)
            else:
                messages = [
                    {"role": "system", "content": inp.system_msg},
                    {"role": "user", "content": inp.user_msg},
                ]
                batch.append(messages)
        return batch

    def encode_messages(
        self, messages_batch: List[List[Dict]]
    ) -> Tuple[List[str], Any, Any, List[List[str]]]:
        """
        Convert message batches into model-ready inputs.

        Returns
        - prompts: List[str], length B
        - input_ids: Tensor [B, L], dtype=torch.long
        - attention_mask: Tensor [B, L], dtype=torch.long
        - tokens_batch: List[List[str]], B lists of tokens, each length L

        Dimensions
        - B: Batch size
        - L: Padded sequence length (max over prompts)
        """
        # Render messages to text prompts (List[str] length B)
        prompts: List[str] = []
        for messages in messages_batch:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=self.add_generation_prompt,
            )
            prompts.append(prompt)

        # Encode prompts -> tensors; shapes: [B, L]
        encodings = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )
        # Encoded tensors
        # - input_ids: Tensor [B, L], dtype=torch.long
        # - attention_mask: Tensor [B, L], dtype=torch.long
        # [B, L]
        input_ids = encodings["input_ids"].to(self.device)
        # [B, L]
        attention_mask = encodings["attention_mask"].to(self.device)

        # Token strings for inspection
        # - tokens_batch: List[List[str]], each inner list length L
        tokens_batch = [
            self.tokenizer.convert_ids_to_tokens(ids.tolist()) for ids in input_ids
        ]

        return prompts, input_ids, attention_mask, tokens_batch

    def calculate_entropy(self, logits: Any) -> Any:
        """
        Compute the entropy for each token position from the logits.

        Entropy H(x) = - Σ p(x) * log(p(x))

        This method can handle two input shapes:
        - Single token: [B, V] -> returns [B]
        - Multiple tokens: [B, L_g, V] -> returns [B, L_g]

        Args:
            logits: Logits tensor.
                    Shape: [B, V] or [B, L_g, V]

        Returns:
            Entropy tensor.
            Shape: [B] (if input is [B, V]) or [B, L_g] (if input is [B, L_g, V])

        Dimensions:
            B   : Batch size
            L_g : Generated sequence length (optional)
            V   : Vocabulary size
        """
        # Softmax to get probabilities
        # Input can be [B, V] or [B, L_g, V]
        # Output will be same shape as input
        probs = torch.softmax(logits, dim=-1)

        # Log probabilities (add small epsilon to avoid log(0))
        log_probs = torch.log(probs + 1e-12)

        # Calculate entropy: -sum(p * log(p)) along vocabulary dimension
        # If input is [B, V], output is [B]
        # If input is [B, L_g, V], output is [B, L_g]
        entropy = -torch.sum(probs * log_probs, dim=-1)

        return entropy

    def infer_entropy_hf(
        self, input_ids: Any, attention_mask: Any = None
    ) -> Tuple[Any, Any]:
        """
        Perform generation and return the full output object and entropy.

        Args:
            input_ids: Token IDs tensor.
                       Shape: [B, L]
            attention_mask: Attention mask tensor.
                            Shape: [B, L]

        Returns:
            Tuple containing:
            - outputs: HF ModelOutput object.
            - entropy: Calculated entropy tensor.
                       Shape: [B, L_g]

        Dimensions:
            B   : Batch size
            L   : Input sequence length
            L_g : Generated sequence length
        """
        with torch.no_grad():
            # Ensure we get scores and dict output
            # output_scores=True -> returns tuple of scores (logits)
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # Enable logits/scores in outputs
                output_scores=True,
                # Do not need hidden states
                output_hidden_states=False,
                # Must be True
                return_dict_in_generate=True,
                # Return the unnormalized logits
                #  before softmax
                renormalize_logits=False,
                **self.generation_config,
            )

            # Extract logits from outputs and compute entropy incrementally
            # to avoid stacking all logits at once which causes OOM
            # scores: tuple of [B, V] tensors
            entropy_list = []

            for score in outputs.scores:
                # score shape: [B, V]
                # Compute entropy for this token position using calculate_entropy
                entropy = self.calculate_entropy(score)
                entropy_list.append(entropy)

            # Stack entropy tensors: [L_g, B] -> [B, L_g]
            entropy = torch.stack(entropy_list, dim=1)

            # Clear the entropy_list to free memory
            del entropy_list

        return outputs, entropy

    def infer_batch(self, infer_inputs: List[InferInput]) -> List[InferOutput]:
        """
        Orchestrate the batch inference pipeline.

        1. Build messages from inputs.
        2. Encode messages into tensors.
        3. Run inference to get outputs and entropy.
        4. Package results into InferOutput objects.

        Args:
            infer_inputs: List of inputs to process.
                          Length: B

        Returns:
            List of InferOutput objects containing the results.
            Length: B
        """
        # Record start time
        t0 = time.time()
        # 1. Build messages
        messages_batch = self.build_messages(infer_inputs)

        # 2. Encode messages
        prompts, input_ids, attention_mask, tokens_batch = self.encode_messages(
            messages_batch
        )

        # 3. Run inference (HF backend)
        hf_outputs, entropy = self.infer_entropy_hf(input_ids, attention_mask)

        # Slice to get only generated tokens
        input_len = input_ids.shape[-1]
        generated_ids = hf_outputs.sequences[:, input_len:]

        # [B, L_g]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Move results to CPU for storage
        # logits = torch.stack(hf_outputs.scores, dim=1)
        # logits_cpu = logits.detach().cpu()
        entropy_cpu = entropy.detach().cpu()

        # Clear GPU memory after moving to CPU
        del entropy
        del input_ids
        del attention_mask
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Package results
        infer_outputs = []
        for i, _ in enumerate(infer_inputs):

            infer_outputs.append(
                InferOutput(
                    prompt=prompts[i],
                    response=responses[i],
                    raw_response=None,
                    cost={"time": time.time() - t0},
                    prompt_tokens=tokens_batch[i],
                    extras={
                        # "logits": logits_cpu[i],
                        "entropy": entropy_cpu[i],
                    },
                )
            )

        return infer_outputs
