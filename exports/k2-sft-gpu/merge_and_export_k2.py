#!/usr/bin/env python3
"""Merge K2 SFT v1 LoRA into base, push to HF, and build Q4_K_M GGUF.

Usage in Colab after training:
  python exports/k2-sft-gpu/merge_and_export_k2.py \
    --adapter checkpoints/k2-sft-v1/best \
    --hf_repo Subject-Emu-5259/NeuralAI-Mamba-K2-SFT-v1
"""
import argparse, os, subprocess, sys
from pathlib import Path

BASE_ID = "state-spaces/mamba-790m-hf"
MERGED_NAME = "NeuralAI-Mamba-K2-SFT-v1-merged"
GGUF_NAME = "neuralai-mamba-k2-sft-v1.Q4_K_M.gguf"


def merge_adapter(base_id, adapter_path, output_path):
    print(f"[merge] base={base_id} adapter={adapter_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(base_id, trust_remote_code=True, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path, safe_serialization=True, max_shard_size="500MB")
    tokenizer.save_pretrained(output_path)
    print(f"[merge] saved -> {output_path}")


def push_to_hf(local_path, repo_id):
    print(f"[hf] pushing {local_path} to {repo_id}")
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id, private=False, exist_ok=True)
    api.upload_folder(folder_path=local_path, repo_id=repo_id)
    print(f"[hf] uploaded -> https://huggingface.co/{repo_id}")


def ensure_convert_script():
    for c in ["llama.cpp/convert_hf_to_gguf.py", "/workspace/llama.cpp/convert_hf_to_gguf.py", os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")]:
        if os.path.exists(c):
            return c
    print("[gguf] cloning llama.cpp...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", "llama.cpp"], check=False)
    return "llama.cpp/convert_hf_to_gguf.py" if os.path.exists("llama.cpp/convert_hf_to_gguf.py") else None


def convert_gguf(merged_path, gguf_path, quant="Q4_K_M"):
    script = ensure_convert_script()
    if not script:
        print("[gguf] could not get convert_hf_to_gguf.py")
        return 1
    cmd = [sys.executable, script, merged_path, "--outfile", gguf_path, "--outtype", quant]
    print("[gguf]", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to best adapter checkpoint")
    parser.add_argument("--hf_repo", default=f"Subject-Emu-5259/{MERGED_NAME}", help="HF repo for merged model")
    parser.add_argument("--skip_gguf", action="store_true", help="Skip GGUF conversion")
    args = parser.parse_args()

    merge_adapter(BASE_ID, args.adapter, MERGED_NAME)
    try:
        push_to_hf(MERGED_NAME, args.hf_repo)
    except Exception as e:
        print(f"[hf] upload failed: {e}")

    if not args.skip_gguf:
        rc = convert_gguf(MERGED_NAME, GGUF_NAME)
        if rc != 0:
            print("[gguf] conversion failed; merged HF model is still available")
            sys.exit(rc)
        print(f"[gguf] produced {GGUF_NAME}")
    print("[done]")


if __name__ == "__main__":
    main()
