#!/usr/bin/env python3
"""Unified Mamba LoRA SFT trainer for K1/K2/K3.

Usage:
  python training/train_mamba_lora.py --base models/mamba-k3-base --data data/train_sft_ultrachat_10k.jsonl --output checkpoints/k3-lora --max_steps 1000
  python training/train_mamba_lora.py --base models/mamba-k1 --data data/train_intel_ultrachat_1k.jsonl --output checkpoints/k1-lora --max_steps 500
  python training/train_mamba_lora.py --base models/mamba-k2-base --data data/train_intel_ultrachat_10k.jsonl --output checkpoints/k2-lora --max_steps 500
"""
import os, sys, json, time, argparse
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["in_proj", "x_proj", "dt_proj"]

MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 5e-5
WARMUP_STEPS = 50
SAVE_EVERY = 100
LOG_EVERY = 10


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        with open(path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = item["text"]
        prompt = item.get("prompt", "")
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Assistant-only loss: mask everything in the prompt (so the model
        # only learns to generate assistant responses).
        if prompt:
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            prompt_len = prompt_tokens["input_ids"].size(1)
            # Tokens before (and including) the last prompt token predict the
            # response, so we keep the label at prompt_len-1 as the first
            # response token and mask earlier positions.
            if prompt_len > 0:
                # If the prompt is longer than max_length, the input was
                # truncated and there may be no response labels left. Cap
                # prompt_len so at least the final token remains supervised.
                prompt_len = min(prompt_len, self.max_length - 1)
                labels[: max(1, prompt_len - 1)] = -100

        # Ignore padding positions.
        labels[attention_mask == 0] = -100

        # Guard against a sample where every label got masked (-100 would
        # make cross_entropy produce NaN).
        non_pad = attention_mask.nonzero(as_tuple=True)[0]
        if non_pad.numel() > 0 and labels[non_pad].eq(-100).all():
            labels[non_pad[-1]] = input_ids[non_pad[-1]]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training {args.run_name}")
    print(f"  Device: {device}, dtype: {dtype}")
    print(f"  Base: {args.base}")
    print(f"  Data: {args.data}")
    print(f"  Steps: {args.max_steps}, Batch: {args.batch_size}, GradAccum: {args.grad_accum}")

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("  Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s | params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

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
    print(f"  LoRA trainable: {trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    model.to(device)
    model.train()

    dataset = SFTDataset(args.data, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * max(0.1, 1.0 - progress)

    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    # Save args
    with open(os.path.join(output_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
    print(f"Final loss: {avg_loss:.4f} | Best loss: {best_loss:.4f}")
    print(f"Output: {final_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path to HF Mamba base model")
    parser.add_argument("--data", required=True, help="Path to JSONL SFT data")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--run_name", default=datetime.now().strftime("mamba-%Y%m%d-%H%M%S"))
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--log_every", type=int, default=LOG_EVERY)
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length for training")
    args = parser.parse_args()
    train(args)
