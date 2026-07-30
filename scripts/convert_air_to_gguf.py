#!/usr/bin/env python3
"""
Convert NeuralAI-Air-135M-SFT state_dict (final.pt) to a llama-arch GGUF.

The Air 135M is structurally Llama (RMSNorm + RoPE + GQA + SwiGLU + tied
embeddings), so the GGUF is written with arch="llama" so llama.cpp loads it
natively. The Python architecture file is NOT used by this converter — it
maps state_dict tensors directly to Llama tensor names.

Runs on the ZO host where final.pt lives.

Usage:
  python3 convert_air_to_gguf.py --weights /path/to/final.pt \
    --config /path/to/config.json --tokenizer /path/to/tokenizer.json \
    --out /path/to/NeuralAI-Air-135M-SFT.F16.gguf --quant f16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def write_gguf(
    state_dict: dict,
    config: dict,
    tokenizer_path: str,
    out_path: str,
    quant: str = "f16",
):
    """Write a llama-arch GGUF from the Air 135M state dict."""
    import gguf
    from gguf import GGUFValueType

    # arch="llama" writes general.architecture="llama" as a STRING
    writer = gguf.GGUFWriter(out_path, "llama")

    head_dim = config["hidden_size"] // config["num_attention_heads"]

    # --- Llama metadata ---
    writer.add_uint32("llama.vocab_size", config["vocab_size"])
    writer.add_uint32("llama.context_length", config["max_position_embeddings"])
    writer.add_uint32("llama.embedding_length", config["hidden_size"])
    writer.add_uint32("llama.block_count", config["num_hidden_layers"])
    writer.add_uint32("llama.attention.head_count", config["num_attention_heads"])
    writer.add_uint32("llama.attention.head_count_kv", config["num_key_value_heads"])
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config["rms_norm_eps"])
    writer.add_uint32("llama.rope.dimension_count", head_dim)
    writer.add_uint32("llama.feed_forward_length", config["intermediate_size"])
    writer.add_bool("llama.tie_word_embeddings", bool(config.get("tie_word_embeddings", True)))

    # --- Tokenizer metadata ---
    writer.add_uint32("tokenizer.ggml.bos_token_id", config.get("bos_token_id", 1))
    writer.add_uint32("tokenizer.ggml.eos_token_id", config.get("eos_token_id", 2))
    writer.add_uint32("tokenizer.ggml.pad_token_id", config.get("pad_token_id", 0))
    writer.add_string("tokenizer.ggml.model", "gpt2")

    with open(tokenizer_path) as f:
        tok = json.load(f)
    model = tok.get("model", {})
    vocab = model.get("vocab", {})
    items = sorted(vocab.items(), key=lambda x: x[1])
    tokens = [t for t, _ in items]
    token_type = [1] * len(tokens)  # 1 = NORMAL
    scores = [0.0] * len(tokens)

    # String arrays MUST pass sub_type=STRING — otherwise the gguf writer
    # mis-infers the element type (BOOL) and llama.cpp rejects the model.
    writer.add_key_value(
        "tokenizer.ggml.tokens", tokens, GGUFValueType.ARRAY,
        sub_type=GGUFValueType.STRING,
    )
    # int / float arrays infer correctly via add_array
    writer.add_array("tokenizer.ggml.token_type", token_type)
    writer.add_array("tokenizer.ggml.scores", scores)

    # merges in tokenizer.json are a list of [a, b] pairs; llama.cpp expects
    # "a b" space-joined strings. Flatten + pass sub_type=STRING.
    mraw = model.get("merges", [])
    merges = [
        f"{a} {b}" if isinstance(m, (list, tuple)) else str(m)
        for m in mraw
    ]
    if merges:
        writer.add_key_value(
            "tokenizer.ggml.merges", merges, GGUFValueType.ARRAY,
            sub_type=GGUFValueType.STRING,
        )

    # --- Weights ---
    num_layers = config["num_hidden_layers"]
    tie = bool(config.get("tie_word_embeddings", True))
    f16 = quant == "f16"

    def add_w(name: str, t: torch.Tensor):
        t = t.float()
        if f16:
            writer.add_tensor(name, t.to(torch.float16).numpy())
        else:
            writer.add_tensor(name, t.numpy(), raw_dtype=gguf.GGUFQuantizationType.F32)

    def add_norm(name: str, t: torch.Tensor):
        # norms stay f32 for stability
        writer.add_tensor(name, t.float().numpy(), raw_dtype=gguf.GGUFQuantizationType.F32)

    add_w("token_embd.weight", state_dict["embed_tokens.weight"])
    add_norm("output_norm.weight", state_dict["norm.weight"])
    if not tie:
        add_w("output.weight", state_dict["lm_head.weight"])

    for i in range(num_layers):
        p = f"layers.{i}."
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            add_w(f"blk.{i}.attn_{proj}.weight", state_dict[f"{p}attn.{proj}.weight"])
        # SwiGLU: w1=gate, w3=up, w2=down
        for k, g in [
            ("mlp.w1.weight", f"blk.{i}.ffn_gate.weight"),
            ("mlp.w3.weight", f"blk.{i}.ffn_up.weight"),
            ("mlp.w2.weight", f"blk.{i}.ffn_down.weight"),
        ]:
            add_w(g, state_dict[f"{p}{k}"])
        add_norm(f"blk.{i}.attn_norm.weight", state_dict[f"{p}input_layernorm.weight"])
        add_norm(f"blk.{i}.ffn_norm.weight", state_dict[f"{p}post_attention_layernorm.weight"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(out_path)
    print(f"[GGUF] wrote {out_path} ({size / 1e6:.1f} MB, quant={quant})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", default="f16", choices=["f16", "q4_k_m", "q8_0"])
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"[Convert] loading weights from {args.weights} ...")
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict) and hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    for k in ["embed_tokens.weight", "norm.weight"]:
        if k not in state_dict:
            print(f"[ERROR] missing key: {k}", file=sys.stderr)
            sys.exit(1)

    total = sum(v.numel() for v in state_dict.values())
    print(f"[Convert] {total / 1e6:.2f}M params, {len(state_dict)} tensors")

    if args.quant == "f16":
        write_gguf(state_dict, config, args.tokenizer, args.out, "f16")
        return

    # Real quantization: emit f16 first, then quantize via llama.cpp if present.
    f16_path = args.out + ".f16.tmp"
    write_gguf(state_dict, config, args.tokenizer, f16_path, "f16")
    try:
        import subprocess
        r = subprocess.run(
            ["python3", "-m", "llama_cpp", "quantize", f16_path, args.out, args.quant.upper()],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            os.remove(f16_path)
            print(f"[Convert] quantized -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
            return
        print(f"[Convert] quantize failed: {r.stderr.strip()}")
    except Exception as e:
        print(f"[Convert] quantize unavailable: {e}")
    os.replace(f16_path, args.out)
    print(f"[Convert] fell back to F16 at {args.out}")


if __name__ == "__main__":
    main()