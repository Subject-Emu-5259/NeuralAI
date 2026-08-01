#!/usr/bin/env python3
"""Mamba K3 LoRA SFT Training — runs on Zo CPU or Colab GPU."""
import os, sys, json, time, argparse, math
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/k3/base")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data/train_sft_ultrachat_10k.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints/k3-lora")

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["in_proj", "dt_proj", "x_proj"]

MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 5e-5
WARMUP_STEPS = 50
MAX_STEPS = 1000
SAVE_EVERY = 100
LOG_EVERY = 10


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item["text"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": input_ids.clone(),
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print(f"Steps: {args.max_steps}, Batch: {args.batch_size}, GradAccum: {args.grad_accum}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA trainable params: {trainable / 1e6:.2f}M ({100 * trainable / total:.2f}%)")

    model.to(device)
    model.train()

    dataset = SFTDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * max(0.1, 1.0 - progress)

    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    step = 0
    total_loss = 0.0
    start_time = time.time()
    best_loss = float("inf")
    lr = args.lr
    optimizer.zero_grad()

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels[attention_mask == 0] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()

            total_loss += loss.item() * args.grad_accum
            current_step_loss = loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

            step += 1

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / step
                sps = elapsed / step
                eta_min = sps * (args.max_steps - step) / 60
                print(f"[{datetime.now().strftime('%H:%M:%S')}] step={step}/{args.max_steps} "
                      f"loss={avg_loss:.4f} last={current_step_loss:.4f} lr={lr:.2e} "
                      f"sps={sps:.1f} eta={eta_min:.0f}m")

            if step % args.save_every == 0 and step > 0:
                avg_loss = total_loss / step
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"  -> Saved checkpoint-{step} | avg_loss={avg_loss:.4f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_dir = os.path.join(output_dir, "best")
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    print(f"  -> New best! loss={best_loss:.4f} -> {best_dir}")

    avg_loss = total_loss / step
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Training complete: {step} steps in {elapsed/60:.1f}m")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Output: {final_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--log_every", type=int, default=LOG_EVERY)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("k3-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    train(args)
