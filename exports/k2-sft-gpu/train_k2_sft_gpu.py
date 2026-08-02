#!/usr/bin/env python3
"""K2 SFT v1 — Colab / GPU launcher.

Runs the unified Mamba LoRA SFT trainer for Mamba K2 (state-spaces/mamba-790m-hf).
Expected Colab runtime: T4 / A100 GPU.
"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "training" / "train_mamba_lora.py"

if not TRAINER.exists():
    print(f"ERROR: trainer not found at {TRAINER}", file=sys.stderr)
    sys.exit(1)

cmd = [
    sys.executable, str(TRAINER),
    "--base", "state-spaces/mamba-790m-hf",
    "--data", str(ROOT / "data" / "train_intel_ultrachat_1k_clean.jsonl"),
    "--output_dir", str(ROOT / "checkpoints"),
    "--run_name", "k2-sft-v1",
    "--max_steps", "500",
    "--batch_size", "1",
    "--grad_accum", "8",
    "--lr", "5e-5",
    "--max_length", "512",
]
print("Running:", " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
