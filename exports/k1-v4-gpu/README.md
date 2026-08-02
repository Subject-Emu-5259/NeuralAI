# K1 SFT v4 — GPU Training Export

This folder contains everything needed to train Mamba K1 SFT LoRA v4 on a GPU.
Local CPU training was stopped because Mamba falls back to the slow sequential CPU implementation here; a 500-step run would take many hours with no checkpoints.

## Contents

- `train_intel_ultrachat_1k_clean.jsonl` — 1,000 single-turn SFT samples (intel format).
- `train_k1_v4_gpu.py` — training script with flushed logging and auto base-model fallback.
- `merge_and_export.py` — merges the best adapter into base and quantizes to GGUF.

## Quick Colab / GPU run

1. Upload this folder to your GPU runtime (Colab, RunPod, etc.).
2. Install deps:
   ```bash
   pip install transformers torch peft accelerate datasets
   # Optional but strongly recommended for Mamba GPU speed:
   pip install causal-conv1d>=1.2 mamba-ssm
   ```
3. Train:
   ```bash
   python train_k1_v4_gpu.py \
     --base state-spaces/mamba-130m-hf \
     --data train_intel_ultrachat_1k_clean.jsonl \
     --output_dir checkpoints \
     --run_name k1-lora-sft-v4 \
     --max_steps 500 --warmup_steps 50 \
     --batch_size 1 --grad_accum 4 --lr 5e-5 \
     --save_every 50 --log_every 10 --max_length 512
   ```
   Expected time on a T4/A100: tens of minutes, not hours.
4. Merge + quantize:
   ```bash
   python merge_and_export.py \
     --base state-spaces/mamba-130m-hf \
     --adapter checkpoints/k1-lora-sft-v4/best \
     --output merged \
     --gguf neuralai-mamba-k1-v4.Q4_K_M.gguf \
     --quant Q4_K_M
   ```
   `merge_and_export.py` auto-detects `tools/llama.cpp/convert_hf_to_gguf.py`; if it is missing, it clones llama.cpp.
5. Bring the GGUF back to this project and place it at:
   ```
   Projects/NeuralAI/models/k1/current/gguf/neuralai-mamba-k1-v4.Q4_K_M.gguf
   ```
   Then run:
   ```bash
   python3 scripts/model_manager.py set mamba-k1
   ```
   to promote K1 v4 as the active inference model.

## Why one folder per model now

`models/k1/` now holds only:

- `base/` — the original K1 base weights.
- `current/` — the active adapter, merged weights, and GGUF for the iteration being worked on.

Old broken v3 artifacts were removed from the local repository and are kept only on HuggingFace (`Subject-Emu-5259/NeuralAI-Mamba-K1`). This prevents duplicate merged/GGUF copies from piling up.
