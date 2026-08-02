#!/bin/bash
# Train Mamba K1 (130M) SFT LoRA v4 on single-turn cleaned UltraChat data.
set -u
cd "$(dirname "$0")"
TRAIN_DATA="data/train_intel_ultrachat_1k_clean.jsonl"
python3 training/train_mamba_lora.py \
  --base models/k1/base \
  --data "$TRAIN_DATA" \
  --output_dir checkpoints \
  --run_name k1-lora-sft-v4 \
  --max_steps 500 \
  --warmup_steps 50 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 5e-5 \
  --save_every 50 \
  --log_every 10 \
  --max_length 512
