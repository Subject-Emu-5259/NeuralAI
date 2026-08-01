#!/bin/bash
# Train Mamba K3 (2.8B SlimPajama base, Mamba-2.8b-slimpj) SFT LoRA on 10K UltraChat samples.
set -u
cd "$(dirname "$0")"
TRAIN_DATA="data/train_intel_ultrachat_10k.jsonl"
if [ ! -f "$TRAIN_DATA" ]; then
  echo "Formatting SFT training data..."
  python3 scripts/prepare_sft_data.py \
    --input data/train_sft_ultrachat_10k.jsonl \
    --output "$TRAIN_DATA" \
    --format intel
fi
python3 training/train_mamba_lora.py \
  --base models/k3/base \
  --data "$TRAIN_DATA" \
  --output_dir checkpoints \
  --run_name k3-lora-sft \
  --max_steps 1000 \
  --warmup_steps 100 \
  --batch_size 1 \
  --grad_accum 16 \
  --lr 2e-5 \
  --save_every 100 \
  --log_every 10
