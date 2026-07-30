#!/usr/bin/env python3
"""
SFT training for NeuralAI-Air-135M using HuggingFace LlamaForCausalLM.
Runs on Colab (GPU T4/V100) or any machine with >=4GB VRAM.

Prerequisites on Colab:
  !pip install transformers datasets accelerate safetensors torch

Usage:
  python3 train_sft_air135m.py \\
    --model_dir ./NeuralAI-Air-135M-HF \\
    --data_path ./data/train_sft_v19.jsonl \\
    --output_dir ./checkpoints/v19-sft \\
    --epochs 3
"""
import argparse, json, os, sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="HF model dir with config.json + model.safetensors")
    ap.add_argument("--data_path", required=True, help="JSONL file with text field in ChatML format")
    ap.add_argument("--output_dir", default="./checkpoints/sft-run", help="Where to save checkpoints")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no_fp16", dest="fp16", action="store_false")
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--push_to_hub", type=str, default=None, help="HF repo to push to")
    return ap.parse_args()

def load_chatml_dataset(path, tokenizer, max_length):
    """Load JSONL where each line has {"text": "<|im_start|>...<|im_end|>"}"""
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = Dataset.from_list(records)
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] device={device}, fp16={args.fp16}")

    # Load tokenizer from model directory (contains tokenizer.json + config)
    print(f"[Train] loading tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (weights in model.safetensors, config in config.json)
    print(f"[Train] loading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        trust_remote_code=False,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] {total_params/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")

    # Load dataset
    print(f"[Train] loading data from {args.data_path}")
    dataset = load_chatml_dataset(args.data_path, tokenizer, args.max_length)
    print(f"[Train] {len(dataset)} examples")

    # Split: 90% train, 10% eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        fp16=args.fp16,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(f"[Train] starting training ({args.epochs} epochs, {len(train_ds)} train / {len(eval_ds)} eval)")
    trainer.train()

    # Save final model
    print(f"[Train] saving to {args.output_dir}/final")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[Train] done — model at {final_dir}")

    if args.push_to_hub:
        print(f"[Train] pushing to {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

if __name__ == "__main__":
    main()