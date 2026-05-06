#!/usr/bin/env python3
"""
NeuralAI Local Training Script
Fine-tunes SmolLM2 on CPU with existing training data
"""

import json
import torch
from pathlib import Path
from datetime import datetime
import sys

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"[NeuralAI] PyTorch version: {torch.__version__}")
print(f"[NeuralAI] CUDA available: {torch.cuda.is_available()}")
print(f"[NeuralAI] Training on CPU")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Configuration
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "v2_model"
TRAIN_DATA = PROJECT_ROOT / "data" / "train_v3.jsonl"
MAX_LENGTH = 512

# LoRA config (same as original)
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

def load_training_data(path: Path) -> Dataset:
    """Load training data from JSONL"""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Handle different formats
            if "messages" in data:
                # ChatML format
                text = ""
                for msg in data["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        text += f"<|im_start|>system\n{content}<|im_end|>\n"
                    elif role == "user":
                        text += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif "prompt" in data and "response" in data:
                # Prompt-response format
                text = f"<|im_start|>user\n{data['prompt']}<|im_end|>\n<|im_start|>assistant\n{data['response']}<|im_end|>\n"
            elif "instruction" in data:
                # Instruction format
                output = data.get("output", data.get("response", ""))
                text = f"<|im_start|>user\n{data['instruction']}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n"
            else:
                continue
            samples.append({"text": text})
    
    print(f"[NeuralAI] Loaded {len(samples)} training samples")
    return Dataset.from_list(samples)

def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Tokenize the dataset"""
    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    return dataset.map(tokenize, batched=False, remove_columns=["text"])

def train():
    """Main training function"""
    print("=" * 50)
    print("NeuralAI Local Training")
    print("=" * 50)
    
    # Load tokenizer
    print(f"\n[1/6] Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"\n[2/6] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    
    # Apply LoRA
    print(f"\n[3/6] Applying LoRA adapter...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    
    # Load data
    print(f"\n[4/6] Loading training data from {TRAIN_DATA}...")
    if not TRAIN_DATA.exists():
        print(f"[ERROR] Training data not found: {TRAIN_DATA}")
        print("[INFO] Generating training data...")
        import subprocess
        subprocess.run([sys.executable, str(PROJECT_ROOT / "training" / "generate_training_v3.py")], check=True)
    
    dataset = load_training_data(TRAIN_DATA)
    tokenized = tokenize_dataset(dataset, tokenizer)
    
    # Split for validation
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(eval_data)}")
    
    # Training arguments (CPU optimized)
    print(f"\n[5/6] Setting up training...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        
        # Batch settings for CPU
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        
        # Learning
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        
        # Epochs
        num_train_epochs=3,
        
        # Logging
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        
        # Performance
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        
        # Other
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer (no tokenizer arg - use processing_class instead if needed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\n[6/6] Starting training...")
    print(f"   This may take a while on CPU...")
    print()
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"\n[NeuralAI] Training completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    # Save
    print(f"\n[NeuralAI] Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training log
    log_data = {
        "base_model": BASE_MODEL,
        "training_samples": len(train_data),
        "validation_samples": len(eval_data),
        "epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "duration_seconds": duration,
        "completed": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n{'=' * 50}")
    print("✓ Training Complete!")
    print(f"{'=' * 50}")
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print(f"To use: Restart the NeuralAI service")
    
    return trainer

if __name__ == "__main__":
    train()
