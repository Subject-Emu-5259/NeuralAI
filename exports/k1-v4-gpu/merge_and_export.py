#!/usr/bin/env python3
"""Merge a trained LoRA adapter into its base Mamba model and export to GGUF.

Usage (Colab / GPU box):
  python merge_and_export.py \
    --base state-spaces/mamba-130m-hf \
    --adapter checkpoints/k1-lora-sft-v4/best \
    --output NeuralAI-Mamba-K1-v4 \
    --gguf neuralai-mamba-k1-v4.Q4_K_M.gguf \
    --quant Q4_K_M

If `llama.cpp/convert_hf_to_gguf.py` is missing it will be downloaded from
https://github.com/ggerganov/llama.cpp.git before conversion.
"""
import argparse
import os
import subprocess
import sys


def merge_adapter(base_path, adapter_path, output_path):
    print(f"[merge] base={base_path} adapter={adapter_path} -> {output_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="500MB")
    tokenizer.save_pretrained(output_path)
    print(f"[merge] saved merged model -> {output_path}")


def ensure_convert_script():
    """Return a usable path to llama.cpp convert_hf_to_gguf.py."""
    candidates = [
        "llama.cpp/convert_hf_to_gguf.py",
        "/workspace/llama.cpp/convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    print("[gguf] convert_hf_to_gguf.py not found. Cloning/updating llama.cpp...")
    repo = "https://github.com/ggerganov/llama.cpp.git"
    subprocess.run(["git", "clone", "--depth", "1", repo, "llama.cpp"], check=False)
    if os.path.exists("llama.cpp/convert_hf_to_gguf.py"):
        return "llama.cpp/convert_hf_to_gguf.py"
    print("[gguf] ERROR: failed to obtain convert_hf_to_gguf.py")
    sys.exit(1)


def convert_gguf(merged_path, gguf_path, quant):
    print(f"[gguf] quant={quant} {gguf_path}")
    script = ensure_convert_script()
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
    parser.add_argument("--base", required=True, help="HF base model id or local path")
    parser.add_argument("--adapter", required=True, help="Path to trained LoRA adapter")
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
