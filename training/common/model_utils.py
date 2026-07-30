"""Shared model and tokenizer utilities for NeuralAI training.

Provides helpers to:
  - Initialize LlamaForCausalLM from a local config.json with RANDOM weights
  - Load the custom 32K BPE tokenizer with verified special-token IDs
  - Count model parameters
"""

import os
from typing import Tuple

# Lazy-load heavy dependencies so the module imports cleanly even when
# PyTorch / transformers are not installed (e.g. local structure verification).
try:
    import torch
    from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedTokenizerFast
except ImportError:
    torch = None  # type: ignore
    LlamaForCausalLM = None  # type: ignore
    LlamaConfig = None  # type: ignore
    PreTrainedTokenizerFast = None  # type: ignore


def load_model_from_config(
    config_path: str,
    use_flash_attention: bool = True,
    torch_dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    torch_compile: bool = False,
):
    """Initialize a LlamaForCausalLM from config.json with RANDOM weights.

    Args:
        config_path: Directory containing config.json (e.g. ``NeuralAI-Air-135M-HF``).
        use_flash_attention: Whether to use Flash Attention 2 (requires CUDA).
        torch_dtype: PyTorch dtype name (``bfloat16``, ``float16``, ``float32``).
        gradient_checkpointing: Enable gradient checkpointing to save VRAM.
        torch_compile: Wrap the model with ``torch.compile`` (PyTorch 2.1+).

    Returns:
        An uninitialized (random-weight) LlamaForCausalLM.

    Raises:
        FileNotFoundError: If *config_path* does not contain ``config.json``.
        RuntimeError: If PyTorch / transformers are not installed.
    """
    if torch is None or LlamaForCausalLM is None:
        raise RuntimeError(
            "PyTorch and transformers are required. "
            "Install: pip install torch>=2.1.0 transformers>=4.40.0"
        )

    config_json = os.path.join(config_path, "config.json")
    if not os.path.exists(config_json):
        raise FileNotFoundError(
            f"Model config not found: {config_json}. "
            "Ensure the directory exists and contains config.json."
        )

    config = LlamaConfig.from_pretrained(config_path)
    attn = "flash_attention_2" if use_flash_attention else "eager"
    dtype = getattr(torch, torch_dtype, torch.float32)
    if dtype == torch.bfloat16 and not torch.cuda.is_available():
        # CPU does not support bfloat16 well; fall back safely.
        dtype = torch.float32

    model = LlamaForCausalLM.from_config(
        config,
        attn_implementation=attn,
        torch_dtype=dtype,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if torch_compile and hasattr(torch, "compile") and torch.cuda.is_available():
        model = torch.compile(model)

    return model


def load_tokenizer(tokenizer_path: str):
    """Load the NeuralAI custom BPE tokenizer from ``tokenizer.json``.

    Verifies and enforces the special-token IDs expected by the model config:
      - PAD = 0 (``<|endoftext|>``)
      - BOS = 1 (``<|im_start|>``)
      - EOS = 2 (``<|im_end|>``)

    Args:
        tokenizer_path: Directory containing ``tokenizer.json``.

    Returns:
        A ``PreTrainedTokenizerFast`` with aligned special tokens.

    Raises:
        FileNotFoundError: If ``tokenizer.json`` is missing.
        RuntimeError: If transformers is not installed.
    """
    if PreTrainedTokenizerFast is None:
        raise RuntimeError(
            "transformers is required. Install: pip install transformers>=4.40.0"
        )

    tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        raise FileNotFoundError(
            f"tokenizer.json not found at {tokenizer_path}. "
            "Ensure NeuralAI-Air-135M-HF/tokenizer.json exists."
        )

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)

    # Align with config.json special token IDs
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"

    return tokenizer


def count_parameters(model) -> Tuple[int, int]:
    """Return ``(total_params, trainable_params)`` for a model.

    Args:
        model: A PyTorch module (e.g. LlamaForCausalLM).

    Returns:
        A tuple of ``(total, trainable)`` parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
