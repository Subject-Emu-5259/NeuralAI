#!/usr/bin/env python3
"""
Clean Nae1 model matching the step-80000 checkpoint EXACTLY.

Checkpoint structure (100.1M params):
  model.embed_tokens.weight: [32000, 768]
  model.layers.0-11:
    .input_layernorm.weight/bias: [768]
    .attn.q_proj.weight: [768, 768]      (12 heads * 64)
    .attn.k_proj.weight: [256, 768]      (4 heads * 64)
    .attn.v_proj.weight: [256, 768]      (4 heads * 64)
    .attn.o_proj.weight: [768, 768]
    .mlp.c_fc.w_gate.weight: [2048, 768]
    .mlp.c_fc.w_up.weight: [2048, 768]
    .mlp.c_fc.w_down.weight: [768, 2048]
    .post_attention_layernorm.weight/bias: [768]
  model.final_layernorm.weight/bias: [768]
  model.rope.0-11.inv_freq: [32]  (shared per-layer RoPE)
  NO lm_head.weight (tied to embed_tokens)
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel


class Nae1Config(PretrainedConfig):
    model_type = "nae1"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        intermediate_size: int = 2048,
        layer_norm_epsilon: float = 1e-5,
        rope_theta: float = 1_000_000.0,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads
        self.n_query_groups = n_head // n_kv_heads  # 3
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    """Rotary Position Embedding matching checkpoint (model.rope.X.inv_freq)"""
    
    def __init__(self, head_dim: int, max_seq_len: int, rope_theta: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        # x: (B, n_heads, L, head_dim)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=x.device, dtype=x.dtype))
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, head_dim)
        cos = emb.cos()
        sin = emb.sin()
        return (x * cos) + (_rotate_half(x) * sin)


class MLP(nn.Module):
    """SwiGLU MLP matching checkpoint: c_fc.w_gate, c_fc.w_up, c_fc.w_down"""
    
    def __init__(self, config: Nae1Config):
        super().__init__()
        self.c_fc = nn.ModuleDict({
            "w_gate": nn.Linear(config.n_embd, config.intermediate_size, bias=False),
            "w_up": nn.Linear(config.n_embd, config.intermediate_size, bias=False),
            "w_down": nn.Linear(config.intermediate_size, config.n_embd, bias=False),
        })
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_fc.w_down(self.act(self.c_fc.w_gate(x)) * self.c_fc.w_up(x))


class GQA(nn.Module):
    """Grouped-Query Attention matching checkpoint"""
    
    def __init__(self, config: Nae1Config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_query_groups = config.n_query_groups
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, _ = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        offset = past_key_value[0].shape[2] if past_key_value is not None else 0
        
        q = rope(q, seq_len=L, offset=offset).to(dtype=q.dtype)
        k = rope(k, seq_len=L, offset=offset).to(dtype=k.dtype)
        
        # Repeat KV heads for GQA
        k = k.repeat_interleave(self.n_query_groups, dim=1)
        v = v.repeat_interleave(self.n_query_groups, dim=1)
        
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        past = (k, v)
        
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.n_head * self.head_dim)
        return self.o_proj(attn), past


class Block(nn.Module):
    """Transformer block matching checkpoint"""
    
    def __init__(self, config: Nae1Config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GQA(config)
        self.post_attention_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, past = self.attn(self.input_layernorm(x), rope, past_key_value)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, past


class Nae1Model(PreTrainedModel):
    config_class = Nae1Config
    
    def __init__(self, config: Nae1Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_layernorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # Shared RoPE buffers (matching checkpoint: model.rope.0..11)
        self.rope = nn.ModuleList([
            RoPE(config.head_dim, config.n_positions, config.rope_theta)
            for _ in range(config.n_layer)
        ])
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        past_key_values = past_key_values or [None] * self.config.n_layer
        x = self.embed_tokens(input_ids)
        new_pasts = []
        for i, layer in enumerate(self.layers):
            x, past = layer(x, self.rope[i], past_key_values[i])
            new_pasts.append(past)
        x = self.final_layernorm(x)
        return x, new_pasts


class Nae1ForCausalLM(PreTrainedModel):
    config_class = Nae1Config
    # No lm_head.weight in checkpoint - tied to embed_tokens
    _tied_weights_keys = {}
    
    def __init__(self, config: Nae1Config):
        super().__init__(config)
        self.model = Nae1Model(config)
        # NO separate lm_head - use tied embed_tokens directly
        self.post_init()
    
    def get_output_embeddings(self):
        return self.model.embed_tokens
    
    def set_output_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> dict:
        hidden, past = self.model(input_ids, past_key_values)
        # Use tied embed_tokens as lm_head
        logits = F.linear(hidden, self.model.embed_tokens.weight)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "past_key_values": past}


# Allow importing from scripts/
try:
    from nae1_config import Nae1Config as _Nae1Config
except ImportError:
    pass

if __name__ == "__main__":
    # Smoke test
    config = Nae1Config()
    model = Nae1ForCausalLM(config)
    model.tie_weights()
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model params (no tied): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward
    x = torch.randint(0, 32000, (1, 16))
    with torch.no_grad():
        out = model(x)
    print(f"Logits shape: {out['logits'].shape}")
    print("Smoke test passed!")
