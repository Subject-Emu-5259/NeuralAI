#!/usr/bin/env python3
"""
NeuralAI Mamba → GGUF Q4_K_M Converter
=======================================
Converts a fine-tuned Mamba model (transformers/HuggingFace format) to
GGUF Q4_K_M quantized format for fast local inference via LM Studio / llama.cpp.

Prerequisites:
  git clone https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp && make

Usage:
  python3 scripts/export_mamba_gguf.py --model models/k2/base \
    --out models/k2/gguf/NeuralAI-Mamba-K2.Q4_K_M.gguf

Requirements:
  - llama.cpp must be installed at ~/llama.cpp or specified with --llamacpp
  - Model must be in HuggingFace transformers format (with config.json, safetensors, tokenizer)
"""

import argparse
import os
import subprocess
import sys
import shutil

DEFAULT_LLAMACPP = os.path.expanduser("~/llama.cpp")


def find_llamacpp(path=None):
    paths = [
        path,
        DEFAULT_LLAMACPP,
        "/home/workspace/llama.cpp",
        "/tmp/llama.cpp",
    ]
    for p in paths:
        if p and os.path.exists(os.path.join(p, "convert_hf_to_gguf.py")):
            return p
    return None


def check_model(model_path):
    required = ["config.json", "tokenizer_config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]
    # Also check for safetensors or pytorch_model.bin
    has_weights = any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    )

    if missing:
        print(f"❌ Missing files in {model_path}: {missing}")
        return False
    if not has_weights:
        print(f"❌ No model weights (safetensors/bin) found in {model_path}")
        return False
    return True


def install_llamacpp(path):
    """Clone and build llama.cpp if not present."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  Cloning llama.cpp to {path}...")
    subprocess.run(
        ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"  Building llama.cpp...")
    subprocess.run(["make", "-j4"], cwd=path, check=True, stdout=subprocess.DEVNULL)


def convert_to_gguf(model_path, f16_path, llamacpp_path):
    """Step 1: Convert HuggingFace model → F16 GGUF"""
    convert_script = os.path.join(llamacpp_path, "convert_hf_to_gguf.py")

    print(f"\n🔄 Step 1: Convert HF → F16 GGUF")
    print(f"   Model:  {model_path}")
    print(f"   Output: {f16_path}")

    cmd = [
        sys.executable, convert_script,
        model_path,
        "--outtype", "f16",
        "--outfile", f16_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Conversion failed:")
        print(result.stderr[-500:])
        return False

    size_mb = os.path.getsize(f16_path) / (1024 * 1024)
    print(f"   ✅ F16 GGUF created: {size_mb:.0f} MB")
    return True


def quantize(f16_path, q4_path, llamacpp_path):
    """Step 2: Quantize F16 → Q4_K_M"""
    quantize_bin = os.path.join(llamacpp_path, "llama-quantize")

    print(f"\n🔧 Step 2: Quantize F16 → Q4_K_M")
    print(f"   Input:  {f16_path}")
    print(f"   Output: {q4_path}")

    cmd = [quantize_bin, f16_path, "Q4_K_M", q4_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Quantization failed:")
        print(result.stderr[-500:])
        return False

    size_mb = os.path.getsize(q4_path) / (1024 * 1024)
    orig_mb = os.path.getsize(f16_path) / (1024 * 1024)
    compression = round((1 - size_mb / orig_mb) * 100, 1)
    print(f"   ✅ Q4_K_M GGUF: {size_mb:.0f} MB ({compression}% compression)")
    return True


def test_generation(q4_path, llamacpp_path):
    """Step 3: Quick generation test"""
    main_bin = os.path.join(llamacpp_path, "llama-cli")

    if not os.path.exists(main_bin):
        main_bin = os.path.join(llamacpp_path, "main")

    if not os.path.exists(main_bin):
        print("⚠️  llama-cli/main not found, skipping generation test")
        return

    print(f"\n🧪 Step 3: Quick generation test")
    cmd = [
        main_bin,
        "-m", q4_path,
        "-p", "<|im_start|>system\nYou are NeuralAI K2.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n",
        "-n", "64",
        "--temp", "0.7",
        "--no-display-prompt",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout.strip() or result.stderr.strip()
        print(f"   Output: {output[:200]}")
    except subprocess.TimeoutExpired:
        print("   ⚠️  Generation timed out (but model loads)")
    except Exception as e:
        print(f"   ⚠️  Test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Mamba → GGUF Q4_K_M converter")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--out", default="", help="Output GGUF path (default: auto)")
    parser.add_argument("--llamacpp", default="", help="Path to llama.cpp directory")
    parser.add_argument("--skip-test", action="store_true", help="Skip generation test")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)

    if not check_model(model_path):
        sys.exit(1)

    # Find/build llama.cpp
    llamacpp = find_llamacpp(args.llamacpp)
    if not llamacpp:
        llamacpp = DEFAULT_LLAMACPP
        if not os.path.exists(llamacpp):
            print("📦 llama.cpp not found, installing...")
            install_llamacpp(llamacpp)
    else:
        print(f"🔧 Using llama.cpp at: {llamacpp}")

    # Derive output name
    if args.out:
        q4_path = os.path.abspath(args.out)
    else:
        model_name = os.path.basename(model_path.rstrip("/"))
        q4_path = os.path.join(
            os.path.dirname(model_path),
            f"NeuralAI-{model_name}-Q4_K_M.gguf",
        )

    f16_path = q4_path.replace(".gguf", "-f16.gguf").replace("Q4_K_M", "F16")

    print(f"\n🚀 NeuralAI Mamba → GGUF Converter")
    print(f"   Model:    {model_path}")
    print(f"   F16:      {f16_path}")
    print(f"   Q4_K_M:   {q4_path}")
    print(f"   llama.cpp: {llamacpp}")

    # Run pipeline
    if not convert_to_gguf(model_path, f16_path, llamacpp):
        sys.exit(1)

    if not quantize(f16_path, q4_path, llamacpp):
        sys.exit(1)

    # Clean up F16 (keep only Q4)
    if os.path.exists(f16_path):
        os.remove(f16_path)
        print(f"\n🧹 Cleaned up F16 intermediate")

    if not args.skip_test:
        test_generation(q4_path, llamacpp)

    print(f"\n✅ Done! Model ready at: {q4_path}")
    print(f"   Copy to LM Studio models directory or use with:")
    print(f"   llama-cli -m {q4_path} -p \"Hello!\"")


if __name__ == "__main__":
    main()
