#!/bin/bash
# Train Mamba K1 (130M) SFT LoRA on 1K UltraChat samples using vocabulary-friendly intel format.
set -u
cd "$(dirname "$0")"
TRAIN_DATA="data/train_intel_ultrachat_1k.jsonl"
if [ ! -f "$TRAIN_DATA" ]; then
  echo "Formatting SFT training data..."
  python3 scripts/prepare_sft_data.py \
    --input data/train_sft_ultrachat_1k.jsonl \
    --output "$TRAIN_DATA" \
    --format intel
fi
python3 training/train_mamba_lora.py \
  --base models/k1/base \
  --data "$TRAIN_DATA" \
  --output_dir checkpoints \
  --run_name k1-lora-sft-v2 \
  --max_steps 500 \
  --warmup_steps 50 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 5e-5 \
  --save_every 50 \
  --log_every 10
