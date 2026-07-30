#!/usr/bin/env python3
"""
Remap NeuralAI-Air-135M state_dict keys to HuggingFace LlamaForCausalLM keys.
Produces a safetensors model file that loads via AutoModelForCausalLM.

Usage (on ZO host):
  python3 remap_air_to_hf.py --weights final.pt --out NeuralAI-Air-135M-HF/
"""
import argparse, json, os, re, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REMAPPINGS = [
    # (source key pattern, target key template)
    # source uses regex, target uses str.format(i=layer_idx)
    (r"^embed_tokens\.weight$", "model.embed_tokens.weight"),
    (r"^norm\.weight$", "model.norm.weight"),
    (r"^layers\.(\d+)\.attn\.q_proj\.weight$", "model.layers.{i}.self_attn.q_proj.weight"),
    (r"^layers\.(\d+)\.attn\.k_proj\.weight$", "model.layers.{i}.self_attn.k_proj.weight"),
    (r"^layers\.(\d+)\.attn\.v_proj\.weight$", "model.layers.{i}.self_attn.v_proj.weight"),
    (r"^layers\.(\d+)\.attn\.o_proj\.weight$", "model.layers.{i}.self_attn.o_proj.weight"),
    (r"^layers\.(\d+)\.mlp\.w1\.weight$", "model.layers.{i}.mlp.gate_proj.weight"),
    (r"^layers\.(\d+)\.mlp\.w3\.weight$", "model.layers.{i}.mlp.up_proj.weight"),
    (r"^layers\.(\d+)\.mlp\.w2\.weight$", "model.layers.{i}.mlp.down_proj.weight"),
    (r"^layers\.(\d+)\.input_layernorm\.weight$", "model.layers.{i}.input_layernorm.weight"),
    (r"^layers\.(\d+)\.post_attention_layernorm\.weight$", "model.layers.{i}.post_attention_layernorm.weight"),
]

def remap(state_dict):
    """Remap state_dict keys from Air 135M format to HF Llama format."""
    new_sd = {}
    unmatched = list(state_dict.keys())
    for pattern, template in REMAPPINGS:
        for key in list(unmatched):
            m = re.match(pattern, key)
            if m:
                i = m.group(1) if m.lastindex and m.lastindex >= 1 else None
                new_key = template.format(i=i) if i is not None else template
                new_sd[new_key] = state_dict[key]
                unmatched.remove(key)
    return new_sd, unmatched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to final.pt")
    ap.add_argument("--out", required=True, help="Output HF model directory")
    args = ap.parse_args()

    print(f"[Remap] loading {args.weights} ...")
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) and hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    total_params = sum(v.numel() for v in sd.values())
    print(f"[Remap] {total_params/1e6:.1f}M params, {len(sd)} keys")

    new_sd, unmatched = remap(sd)
    print(f"[Remap] mapped {len(new_sd)} keys")
    if unmatched:
        print(f"[Remap] {len(unmatched)} unmatched keys (optimizer states / non-model):")
        for k in unmatched[:10]:
            print(f"  - {k}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(new_sd, out_dir / "model.safetensors")
    size_mb = os.path.getsize(out_dir / "model.safetensors") / 1e6
    print(f"[Remap] wrote {out_dir / 'model.safetensors'} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()