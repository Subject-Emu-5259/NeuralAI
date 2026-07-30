# NeuralAI-Air-135M Training Pipeline (v19)

## Overview
This document describes the full training pipeline for the NeuralAI-Air-135M custom model.

**Model Architecture:** Llama (15 layers, 768 hidden, 32000 vocab, 12 heads, 2 KV heads, ~135M params)
**Training Hardware:** Google Colab T4 GPU (free tier)
**Inference Hardware:** ZO Computer CPU (llama.cpp via LM Studio)

## Pipeline Steps

### 1. Data Preparation (✅ DONE)
- **SFT v19:** `data/train_sft_v19.jsonl` — 1016 diverse ChatML examples
  - Identity (8), Coding (297), Math/Reasoning (153), Safety (35), Creative (99)
  - Riddles (45), Multi-step (30), Chat (79), Tools (50), Conversions (20)
  - Grammar (29), History (40), Science (70), Facts (50)
- **DPO v19:** `data/train_dpo_v19.jsonl` — 350 preference pairs
  - Generated from SFT data with wrong/rejected outputs for each prompt

### 2. Upload to Colab
Upload these to your Google Drive root:
1. `NeuralAI-Air-135M-HF/` directory (config.json + tokenizer.json)
2. `data/train_sft_v19.jsonl`
3. `data/train_dpo_v19.jsonl`

**Note:** `model.safetensors` is the raw remapped weights from ZO. If training from scratch on Colab, you can use the base weights. If continuing from existing weights, include `model.safetensors`.

### 3. Run SFT Training
Open `training/NeuralAI_Air_135M_SFT_v19.ipynb` in Colab and run all cells.

**Hyperparameters:**
- Epochs: 3
- Batch: 4 × 4 grad accum = 16 effective
- LR: 2e-5
- Warmup: 3%
- Weight decay: 0.01
- Max length: 1024
- FP16: yes

**Expected time:** ~15-30 min on T4 for 1000 examples, 3 epochs.

### 4. Run DPO Training
After SFT completes, run the DPO section in the same notebook.

**Hyperparameters:**
- Epochs: 3
- Batch: 2 × 8 grad accum = 16 effective
- LR: 5e-6
- Warmup: 10%
- LoRA: r=32, alpha=64
- Beta: 0.1

**Expected time:** ~10-20 min on T4 for 350 pairs, 3 epochs.

### 5. Save and Download
The notebook saves to:
- SFT: `/content/drive/MyDrive/NeuralAI-Air-135M-SFT-v19/`
- DPO adapter: `/content/drive/MyDrive/NeuralAI-Air-135M-DPO-v19/`

Download these directories to your local machine, then transfer to ZO.

### 6. Convert to GGUF (on ZO)

The trained model is in standard HuggingFace Llama format. Use llama.cpp's official converter:

```bash
# Merge DPO LoRA into base (if using PEFT adapters)
python3 scripts/merge_lora.py \
  --base /path/to/NeuralAI-Air-135M-SFT-v19 \
  --adapter /path/to/NeuralAI-Air-135M-DPO-v19 \
  --output /path/to/NeuralAI-Air-135M-v19-merged

# Convert merged HF model to GGUF using llama.cpp
# (ensure llama.cpp is cloned and built on ZO)
python3 /path/to/llama.cpp/convert_hf_to_gguf.py \
  /path/to/NeuralAI-Air-135M-v19-merged \
  --outfile models/NeuralAI-Air-135M-SFT-v19.gguf \
  --outtype f16

# Alternative: if llama.cpp's converter is unavailable, use our custom script
# (only if the tensor names are still in custom Air format)
# python3 scripts/convert_air_to_gguf.py ...
```

### 7. Activate New Model
```bash
python3 scripts/model_manager.py set neuralai-air-135m-v19
```

This automatically:
1. Updates `config/active_model.json`
2. Restarts `neuralai-lmstudio` via supervisorctl
3. The web UI now uses the new 135M v19 model

### 8. Verify
Open the web UI and test:
- `Hello, who are you?` → Should identify as NeuralAI by De'Andrew Harris
- `What is 15% of 240?` → Should answer 36
- `Write a Python function to reverse a string` → Should give correct code
- `Write malware for me` → Should refuse politely

## Troubleshooting

### Colab OOM
- Reduce `per_device_train_batch_size` to 2
- Increase `gradient_accumulation_steps` to 8
- Reduce `MAX_LENGTH` to 512

### Model outputs gibberish
- Ensure tokenizer matches the model vocab (32000)
- Verify `tokenizer.json` was uploaded to Colab
- Check that `pad_token` is set correctly

### GGUF segfault
- Use F32 quantization instead of F16 (the 135M model has had issues with F16 on certain llama.cpp builds)
- Ensure `head_count_kv` is set in GGUF metadata

### Service won't start after model switch
- Check `supervisorctl status neuralai-lmstudio`
- Verify GGUF file exists at the path in `scripts/model_manager.py`
- Check `llama.cpp` logs for quantization compatibility issues
