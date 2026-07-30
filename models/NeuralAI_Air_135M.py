#!/usr/bin/env python3
"""
NeuralAI-Air-135M — custom from-scratch transformer architecture.

A 135M-parameter decoder-only LLM with:
  - GQA (grouped-query attention): 12 query heads, 2 KV heads
  - SwiGLU MLP
  - RMSNorm (pre-norm)
  - Rotary position embeddings (RoPE)
  - Tied word embeddings (lm_head = embed_tokens)

Architecture: vocab=32000, hidden=768, layers=15, heads=12, kv_heads=2,
              intermediate=2560, max_pos=2048, head_dim=64.

Reconstructed from bytecode analysis of the original neuralai_air_model.py
and validated against final.pt weight shapes.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class NeuralAIAir135MConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 15
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 2560
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NeuralAIRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_pos: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class NeuralAIRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------
class NeuralAIGQAAttention(nn.Module):
    def __init__(self, config: NeuralAIAir135MConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_group_size = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary_emb = NeuralAIRotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(T)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos.to(q.dtype), sin.to(q.dtype))

        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class NeuralAISwiGLU(nn.Module):
    def __init__(self, config: NeuralAIAir135MConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------
class NeuralAIDecoderLayer(nn.Module):
    def __init__(self, config: NeuralAIAir135MConfig):
        super().__init__()
        self.attn = NeuralAIGQAAttention(config)
        self.mlp = NeuralAISwiGLU(config)
        self.input_layernorm = NeuralAIRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NeuralAIRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class NeuralAIAir135MModel(nn.Module):
    def __init__(self, config: NeuralAIAir135MConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NeuralAIDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = NeuralAIRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_position_embeddings :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if next_token.item() == eos_token_id:
                break
        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def load_config_from_hf(snapshot_dir: str) -> NeuralAIAir135MConfig:
    cfg_path = Path(snapshot_dir) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    valid_keys = {f.name for f in NeuralAIAir135MConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in cfg.items() if k in valid_keys}
    return NeuralAIAir135MConfig(**filtered)


def _load_state_dict(snapshot_dir: str) -> dict:
    snapshot_dir = Path(snapshot_dir)
    safetensors_files = sorted(snapshot_dir.glob("*.safetensors"))
    if safetensors_files:
        from safetensors.torch import load_file

        state = {}
        for f in safetensors_files:
            state.update(load_file(str(f)))
        return state

    bin_files = sorted(snapshot_dir.glob("pytorch_model*.bin"))
    if bin_files:
        state = {}
        for f in bin_files:
            sd = torch.load(str(f), map_location="cpu", weights_only=False)
            if isinstance(sd, dict):
                state.update(sd)
            else:
                state.update(sd.state_dict() if hasattr(sd, "state_dict") else {})
        return state

    raise FileNotFoundError(f"No model weights found in {snapshot_dir}")


def load_sft_model(
    sft_repo: str = "Subject-Emu-5259/NeuralAI-Air-135M-SFT",
    base_repo: str = "Subject-Emu-5259/NeuralAI-Air-135M",
):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    snapshot_dir = os.path.dirname(
        hf_hub_download(repo_id=sft_repo, filename="config.json")
    )

    config = load_config_from_hf(snapshot_dir)
    model = NeuralAIAir135MModel(config)

    state = _load_state_dict(snapshot_dir)
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)

    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id

    params = model.count_parameters()
    print(f"[NeuralAI-Air] {params / 1e6:.2f}M params loaded from {sft_repo}")

    return model, tokenizer, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeuralAI-Air-135M model")
    parser.add_argument("--model-dir", type=str, help="Local model directory with config.json + weights")
    parser.add_argument("--prompt", type=str, default="Hello, who are you?")
    args = parser.parse_args()

    if args.model_dir:
        config = load_config_from_hf(args.model_dir)
        model = NeuralAIAir135MModel(config)
        state = _load_state_dict(args.model_dir)
        model.load_state_dict(state, strict=False)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        model.eval()

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

        prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        out = model.generate(ids, max_new_tokens=80, temperature=0.7, eos_token_id=config.eos_token_id)
        text = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        print(text)
    else:
        print(f"NeuralAI-Air-135M architecture ({NeuralAIAir135MConfig()})")
