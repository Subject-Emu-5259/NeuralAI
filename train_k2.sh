#!/bin/bash
# Train Mamba K2 (790M) SFT LoRA on 10K UltraChat samples using vocabulary-friendly intel format.
set -u
cd "$(dirname "$0")"
if [ ! -d "models/mamba-k2-base" ]; then
  echo "Downloading Mamba K2 base (state-spaces/mamba-790m-hf)..."
  python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("state-spaces/mamba-790m-hf", local_dir="models/mamba-k2-base", local_dir_use_symlinks=False)
PY
fi
TRAIN_DATA="data/train_intel_ultrachat_10k.jsonl"
if [ ! -f "$TRAIN_DATA" ]; then
  echo "Formatting SFT training data..."
  python3 scripts/prepare_sft_data.py \
    --input data/train_sft_ultrachat_10k.jsonl \
    --output "$TRAIN_DATA" \
    --format intel
fi
python3 training/train_mamba_lora.py \
  --base models/mamba-k2-base \
  --data "$TRAIN_DATA" \
  --output_dir checkpoints \
  --run_name k2-lora-sft \
  --max_steps 500 \
  --warmup_steps 50 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 5e-5 \
  --save_every 100 \
  --log_every 10
