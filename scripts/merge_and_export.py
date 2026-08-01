#!/usr/bin/env python3
"""Merge a trained LoRA adapter into its base Mamba model and export to GGUF.

Usage:
  python scripts/merge_and_export.py \
    --base models/mamba-k1 \
    --adapter checkpoints/k1-lora-sft-v2/final \
    --output models/mamba-k1-merged \
    --gguf models/mamba-k1/neuralai-mamba-k1.Q4_K_M.gguf \
    --quant Q4_K_M
"""
import argparse
import os
import subprocess
import sys


def merge_adapter(base_path, adapter_path, output_path):
    print(f"[merge] base={base_path} adapter={adapter_path} -> {output_path}")
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    merged = model.merge_and_unload()
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print(f"[merge] saved merged model -> {output_path}")


def convert_gguf(merged_path, gguf_path, quant):
    print(f"[gguf] quant={quant} {gguf_path}")
    # Try the local convert script first; otherwise attempt common paths.
    candidates = [
        "llama.cpp/convert_hf_to_gguf.py",
        "convert_hf_to_gguf.py",
        "/workspace/llama.cpp/convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
    ]
    script = None
    for c in candidates:
        if os.path.exists(c):
            script = c
            break
    if not script:
        print("[gguf] ERROR: convert_hf_to_gguf.py not found. Please clone llama.cpp:")
        print("  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git")
        print("  pip install -r llama.cpp/requirements.txt")
        return 1
    cmd = [
        sys.executable, script,
        merged_path,
        "--outfile", gguf_path,
        "--outtype", quant,
    ]
    print(f"[gguf] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="HF base model path (informational; adapter stores base)")
    parser.add_argument("--adapter", required=True, help="Path to trained LoRA adapter (final/)")
    parser.add_argument("--output", required=True, help="Directory for merged HF model")
    parser.add_argument("--gguf", help="Path for output GGUF (optional)")
    parser.add_argument("--quant", default="Q4_K_M", help="GGUF quantization type")
    args = parser.parse_args()

    merge_adapter(args.base, args.adapter, args.output)
    if args.gguf:
        rc = convert_gguf(args.output, args.gguf, args.quant)
        if rc != 0:
            print("[gguf] conversion failed; merged HF model is still available")
            sys.exit(rc)
    print("[done]")


if __name__ == "__main__":
    main()
