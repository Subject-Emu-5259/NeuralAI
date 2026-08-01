#!/usr/bin/env python3
"""
Merge DPO LoRA adapter into SFT base model, then convert to GGUF.

Reads HuggingFace-format safetensors (not raw state_dict), so it works
with the output of training/sft + training/dpo pipelines.

Usage:
  python3 merge_and_convert_v19.py \
    --base checkpoints/sft-v19/final \
    --adapter checkpoints/dpo-v19/final_adapter \
    --out models/NeuralAI-Air-135M-SFT-v19.gguf \
    --quant q4_k_m
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil

import torch
from safetensors.torch import load_file, save_file


def merge_adapter(base_dir: str, adapter_dir: str, out_dir: str) -> str:
    """Merge LoRA adapter into base model, save merged safetensors to out_dir."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Merge] Loading base from {base_dir} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.float32, device_map="cpu"
    )
    tok = AutoTokenizer.from_pretrained(base_dir)

    print(f"[Merge] Loading adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("[Merge] Merging adapter weights ...")
    model = model.merge_and_unload()

    os.makedirs(out_dir, exist_ok=True)
    print(f"[Merge] Saving merged model to {out_dir} ...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    config_path = os.path.join(out_dir, "config.json")
    print(f"[Merge] Done. Merged model at {out_dir}")
    return out_dir


def convert_hf_to_gguf(model_dir: str, out_path: str, quant: str = "f16"):
    """Convert HuggingFace safetensors model to GGUF (llama arch)."""
    import gguf
    from gguf import GGUFValueType

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Load all safetensors shards
    safetensors_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    )
    if not safetensors_files:
        print("[ERROR] No safetensors files found in model dir")
        sys.exit(1)

    state_dict = {}
    for sf in safetensors_files:
        path = os.path.join(model_dir, sf)
        print(f"[GGUF] Loading {sf} ...")
        sd = load_file(path)
        state_dict.update(sd)

    print(f"[GGUF] {len(state_dict)} tensors loaded")

    writer = gguf.GGUFWriter(out_path, "llama")

    head_dim = config["hidden_size"] // config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]
    tie = bool(config.get("tie_word_embeddings", True))

    # Llama metadata
    writer.add_uint32("llama.vocab_size", config["vocab_size"])
    writer.add_uint32("llama.context_length", config.get("max_position_embeddings", 2048))
    writer.add_uint32("llama.embedding_length", config["hidden_size"])
    writer.add_uint32("llama.block_count", num_layers)
    writer.add_uint32("llama.attention.head_count", config["num_attention_heads"])
    writer.add_uint32("llama.attention.head_count_kv", config["num_key_value_heads"])
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", config["rms_norm_eps"])
    writer.add_uint32("llama.rope.dimension_count", head_dim)
    writer.add_uint32("llama.feed_forward_length", config["intermediate_size"])
    writer.add_bool("llama.tie_word_embeddings", tie)

    # Tokenizer
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        # Fall back to base/sft tokenizer
        tok_path = os.path.join(model_dir, "tokenizer.json")
    with open(tok_path) as f:
        tok = json.load(f)
    model_tok = tok.get("model", {})
    vocab = model_tok.get("vocab", {})
    items = sorted(vocab.items(), key=lambda x: x[1])
    tokens = [t for t, _ in items]
    token_type = [1] * len(tokens)
    scores = [0.0] * len(tokens)

    writer.add_uint32("tokenizer.ggml.bos_token_id", config.get("bos_token_id", 1))
    writer.add_uint32("tokenizer.ggml.eos_token_id", config.get("eos_token_id", 2))
    writer.add_uint32("tokenizer.ggml.pad_token_id", config.get("pad_token_id", 0))
    writer.add_string("tokenizer.ggml.model", "gpt2")

    writer.add_key_value(
        "tokenizer.ggml.tokens", tokens, GGUFValueType.ARRAY,
        sub_type=GGUFValueType.STRING,
    )
    writer.add_array("tokenizer.ggml.token_type", token_type)
    writer.add_array("tokenizer.ggml.scores", scores)

    merges_raw = model_tok.get("merges", [])
    merges = []
    for m in merges_raw:
        if isinstance(m, (list, tuple)) and len(m) == 2:
            merges.append(f"{m[0]} {m[1]}")
        else:
            merges.append(str(m))
    if merges:
        writer.add_key_value(
            "tokenizer.ggml.merges", merges, GGUFValueType.ARRAY,
            sub_type=GGUFValueType.STRING,
        )

    # Weight helpers
    f16 = quant == "f16"

    def add_w(name: str, t: torch.Tensor):
        t = t.float()
        if f16:
            writer.add_tensor(name, t.to(torch.float16).numpy())
        else:
            writer.add_tensor(name, t.numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)

    def add_norm(name: str, t: torch.Tensor):
        writer.add_tensor(name, t.float().numpy(), raw_dtype=gguf.GGMLQuantizationType.F32)

    # Map HF names to GGUF names
    add_w("token_embd.weight", state_dict["model.embed_tokens.weight"])
    add_norm("output_norm.weight", state_dict["model.norm.weight"])
    if not tie and "lm_head.weight" in state_dict:
        add_w("output.weight", state_dict["lm_head.weight"])

    for i in range(num_layers):
        hf = f"model.layers.{i}."
        for hf_proj, gguf_proj in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "output")]:
            add_w(
                f"blk.{i}.attn_{gguf_proj}.weight",
                state_dict[f"{hf}self_attn.{hf_proj}.weight"],
            )
        for hf_k, gguf_k in [
            ("mlp.gate_proj.weight", f"blk.{i}.ffn_gate.weight"),
            ("mlp.up_proj.weight", f"blk.{i}.ffn_up.weight"),
            ("mlp.down_proj.weight", f"blk.{i}.ffn_down.weight"),
        ]:
            add_w(gguf_k, state_dict[f"{hf}{hf_k}"])
        add_norm(f"blk.{i}.attn_norm.weight", state_dict[f"{hf}input_layernorm.weight"])
        add_norm(f"blk.{i}.ffn_norm.weight", state_dict[f"{hf}post_attention_layernorm.weight"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(out_path)
    print(f"[GGUF] wrote {out_path} ({size / 1e6:.1f} MB, quant={quant})")

    # Quantize if needed
    if quant != "f16":
        try:
            import subprocess
            quant_path = out_path.replace(".gguf", f"_{quant}.gguf")
            r = subprocess.run(
                ["python3", "-m", "llama_cpp", "quantize", out_path, quant_path, quant.upper()],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                os.replace(quant_path, out_path)
                print(f"[GGUF] quantized -> {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
            else:
                print(f"[GGUF] quantize failed: {r.stderr.strip()}")
                print(f"[GGUF] keeping F16 at {out_path}")
        except Exception as e:
            print(f"[GGUF] quantize unavailable: {e}, keeping F16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SFT base model dir (safetensors)")
    ap.add_argument("--adapter", default="", help="DPO adapter dir (optional)")
    ap.add_argument("--out", required=True, help="Output GGUF path")
    ap.add_argument("--quant", default="q4_k_m", choices=["f16", "q4_k_m", "q8_0"])
    args = ap.parse_args()

    model_dir = args.base

    if args.adapter and os.path.isdir(args.adapter):
        merged_dir = os.path.join(os.path.dirname(args.base), "merged")
        model_dir = merge_adapter(args.base, args.adapter, merged_dir)

    convert_hf_to_gguf(model_dir, args.out, args.quant)


if __name__ == "__main__":
    main()
