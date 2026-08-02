# NeuralAI Mamba K2 SFT v1 — Colab GPU Export

Everything needed to produce the first chat-aligned K2 (793M Mamba) model on a Colab GPU.

## Contents

- `colab_k2_sft.ipynb` — one-click Colab notebook.
- `train_k2_sft_gpu.py` — LoRA SFT wrapper for `state-spaces/mamba-790m-hf`.
- `merge_and_export_k2.py` — merge adapter, convert to Q4_K_M GGUF.
- `data/train_intel_ultrachat_1k_clean.jsonl` — clean single-turn SFT data (intel format).

## Run locally (CPU not recommended, use Colab)

```bash
uv run training/train_mamba_lora.py \
  --base state-spaces/mamba-790m-hf \
  --data exports/k2-sft-gpu/data/train_intel_ultrachat_1k_clean.jsonl \
  --output_dir checkpoints \
  --run_name k2-sft-v1 \
  --max_steps 500
```

## Colab flow

1. Open `colab_k2_sft.ipynb` in Google Colab.
2. Set a Hugging Face token (`HF_TOKEN`) in Colab secrets.
3. Run all cells. Training takes ~20-40 min on a T4.
4. Download `neuralai-mamba-k2-sft-v1.Q4_K_M.gguf` and replace `models/k2/gguf/mamba-790m-hf.Q4_K_M.gguf` (or place next to it).
5. Update `scripts/model_manager.py` to point the K2 entry at the new GGUF and flip status to `active`.

## Expected artifacts

- `checkpoints/k2-sft-v1/best/` — best LoRA adapter.
- `NeuralAI-Mamba-K2-SFT-v1-merged/` — merged HF model.
- `neuralai-mamba-k2-sft-v1.Q4_K_M.gguf` — quantized LM Studio model.

## Notes

- This run uses a **fresh** base (`state-spaces/mamba-790m-hf`) because the Zo workspace keeps only the quantized GGUF for K2 to save disk.
- After a successful run the local `models/k2/gguf/` can be updated and the live service switched back from the temporary OpenRouter fallback to local K2 chat.
