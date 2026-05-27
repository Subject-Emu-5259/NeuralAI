#!/usr/bin/env python3
"""
NeuralAI DPO Local Training Script (CPU Optimized) - v9.3
======================================================
Aligns SmolLM2-360M with DPO preferences on CPU.
"""
import os
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

# Config
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
PREV_DPO_MODEL = "/home/workspace/Projects/NeuralAI/checkpoints/dpo_model_v9"
OUTPUT_DIR = "/home/workspace/Projects/NeuralAI/checkpoints/dpo_model_v10"
DATA_PATH = "/home/workspace/Projects/NeuralAI/data/train_dpo_v10.jsonl"

def train():
    print(f"[NeuralAI] Starting DPO alignment (v10) on CPU...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    load_path = PREV_DPO_MODEL if os.path.exists(PREV_DPO_MODEL) else BASE_MODEL
    print(f"[NeuralAI] Loading model from {load_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True
    )
    
    # Initialize NEW LoRA adapter
    print(f"[NeuralAI] Initializing LoRA adapter for DPO v9...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load data
    print(f"[NeuralAI] Loading dataset from {DATA_PATH}...")
    pairs = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    dataset = Dataset.from_list([
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])

    # DPO Config
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_length=512,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        use_cpu=True
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer, # CORRECT PARAM FOR trl 1.3.0
    )

    print("[NeuralAI] Beginning DPO v9 training run (CPU)...")
    trainer.train()
    
    print(f"[NeuralAI] Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Create symlink
    target_link = "/home/workspace/Projects/NeuralAI/checkpoints/dpo_model_latest"
    if os.path.exists(target_link):
        os.remove(target_link)
    os.symlink(OUTPUT_DIR, target_link)
    
    print("[OK] DPO v10 Training Complete.")

if __name__ == "__main__":
    train()
