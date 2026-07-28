#!/usr/bin/env python3
"""Merge the NeuralAI v17-dpo LoRA adapter into the base SmolLM2-360M-Instruct model
and save the full merged model to disk, ready to upload to the Hub."""
import os
os.environ["HF_HOME"] = "/home/workspace/.cache/hf"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER = "/home/workspace/Projects/NeuralAI/checkpoints/v17-dpo"
OUT = "/home/workspace/Projects/NeuralAI/checkpoints/v17_merged"

print("Loading base model in fp32...")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="float32")

print("Applying LoRA adapter...")
peft_model = PeftModel.from_pretrained(base, ADAPTER)

print("Merging adapter weights into base...")
merged = peft_model.merge_and_unload()

print(f"Saving merged model to {OUT} ...")
os.makedirs(OUT, exist_ok=True)
merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="500MB")

print("Saving tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUT)

# Copy chat template if present
import shutil
src_chat = os.path.join(ADAPTER, "chat_template.jinja")
if os.path.exists(src_chat):
    shutil.copy(src_chat, os.path.join(OUT, "chat_template.jinja"))

print("Done. Total params:", sum(p.numel() for p in merged.parameters()) / 1e6, "M")
