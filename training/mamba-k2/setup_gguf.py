#!/usr/bin/env python3
"""
Mamba K2 GGUF Download + Setup
===============================
Downloads pre-converted Mamba-790M GGUF (Q4_K_M) from HuggingFace
and registers it with NeuralAI.

Mamba SSM models ARE compatible with recent llama.cpp builds
(built with -DLLAMA_CUBLAS=OFF for Mamba support).

Source: mradermacher/mamba-790m-hf-GGUF
"""

import os
import sys
import json
import hashlib
import shutil
from pathlib import Path

GGUF_MODEL_URL = "https://huggingface.co/mradermacher/mamba-790m-hf-GGUF/resolve/main/mamba-790m-hf.Q4_K_M.gguf"
MODELS_DIR = "/home/workspace/Projects/NeuralAI/models"
MODEL_FILENAME = "NeuralAI-Mamba-K2.Q4_K_M.gguf"
MODEL_ID = "mamba-k2"
MODEL_LABEL = "Mamba K2 (790M Q4_K_M)"


def download_gguf():
    """Download the Q4_K_M GGUF if not already present."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, MODEL_FILENAME)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"✅ {MODEL_FILENAME} already exists ({size_mb:.0f} MB)")
        return dest

    print(f"📥 Downloading Mamba-790M Q4_K_M GGUF...")
    print(f"   URL: {GGUF_MODEL_URL}")

    import subprocess
    result = subprocess.run([
        "curl", "-L", "--progress-bar", "-o", dest, GGUF_MODEL_URL
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Download failed: {result.stderr}")
        return None

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"✅ Downloaded: {dest} ({size_mb:.0f} MB)")
    return dest


def verify_model(path):
    """Basic integrity check: verify file exists and has valid size."""
    if not os.path.exists(path):
        return False, "File not found"

    size = os.path.getsize(path)
    if size < 10 * 1024 * 1024:  # Less than 10MB = corrupted
        return False, f"File too small ({size} bytes)"

    # Check GGUF magic bytes
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return False, f"Not a valid GGUF file (magic: {magic!r})"

    return True, f"{size / (1024*1024):.0f} MB"


def register_in_model_manager():
    """Register Mamba K2 GGUF in model_manager.py."""
    model_manager_path = "/home/workspace/Projects/NeuralAI/scripts/model_manager.py"

    with open(model_manager_path, "r") as f:
        content = f.read()

    # Check if already registered
    if '"mamba-k2"' in content:
        print("✅ Mamba K2 already registered in model_manager.py")
        return

    # Add entry after mamba-k1
    insert_marker = '        "label": "Mamba K1 (1st Owned Base)",\n'
    new_entry = (
        '        "label": "Mamba K1 (1st Owned Base)",\n'
        '        "gguf": "models/NeuralAI-Mamba-K1.Q4_K_M.gguf",\n'
        '    },\n'
        '    "mamba-k2": {\n'
        '        "id": "mamba-k2",\n'
        '        "label": "Mamba K2 (790M Q4_K_M)",\n'
        '        "gguf": "models/NeuralAI-Mamba-K2.Q4_K_M.gguf",\n'
    )
    content = content.replace(insert_marker, new_entry)

    with open(model_manager_path, "w") as f:
        f.write(content)

    print("✅ Registered Mamba K2 in model_manager.py")


def main():
    print("=" * 60)
    print("🧠 NeuralAI Mamba K2 GGUF Setup")
    print("=" * 60)

    model_path = download_gguf()
    if not model_path:
        print("❌ Download failed. Check network or HF availability.")
        sys.exit(1)

    valid, info = verify_model(model_path)
    if not valid:
        print(f"❌ Verification failed: {info}")
        sys.exit(1)
    print(f"✅ Verified: {info}")

    register_in_model_manager()

    print("\n" + "=" * 60)
    print("📋 Next Steps:")
    print(f"   1. Load in llama.cpp: llama-cli -m {model_path}")
    print("   2. Or select 'Mamba K2' in NeuralAI Web UI model picker")
    print("   3. The K2 model must be FINE-TUNED first (use colab_train_mamba_k2.py)")
    print("      This GGUF is the BASE Mamba-790M — not yet NeuralAI-trained.")
    print("=" * 60)


if __name__ == "__main__":
    main()
