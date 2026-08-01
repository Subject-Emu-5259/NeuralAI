#!/usr/bin/env python3
"""Merge the best K1 SFT v3 LoRA adapter into its base model and save."""
import os
import shutil
import sys

os.environ["HF_HOME"] = "/home/workspace/.cache/hf"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/home/workspace/Projects/NeuralAI/models/mamba-k1-merged-v2"
ADAPTER = "/home/workspace/Projects/NeuralAI/checkpoints/k1-lora-sft-v3/best"
OUT = "/home/workspace/Projects/NeuralAI/models/NeuralAI-Mamba-K1-v3-merged"

print(f"Loading base model from {BASE} ...")
base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, torch_dtype="float32")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

print(f"Loading adapter from {ADAPTER} ...")
peft_model = PeftModel.from_pretrained(base, ADAPTER)

print("Merging adapter weights into base...")
merged = peft_model.merge_and_unload()

print(f"Saving merged model to {OUT} ...")
os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="500MB")
tok.save_pretrained(OUT)

# Preserve chat template if present
src_chat = os.path.join(ADAPTER, "chat_template.jinja")
if os.path.exists(src_chat):
    shutil.copy(src_chat, os.path.join(OUT, "chat_template.jinja"))

params = sum(p.numel() for p in merged.parameters()) / 1e6
print(f"Done. Total params: {params:.1f}M")
