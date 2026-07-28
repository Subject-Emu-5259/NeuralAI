# NeuralAI Training Manifest

> Single source of truth for which dataset/training script/checkpoint was used for each model version, and which artifacts are stale.

## Model Zoo

| Model | Version | Base | Dataset | Params | Status | HF Repo | Local Path | When |
|-------|---------|------|---------|--------|--------|---------|------------|------|
| NeuralAI 360M DPO | v16 / d16 | `HuggingFaceTB/SmolLM2-360M-Instruct` | `data/train_dpo_v15.jsonl` (597 pairs) | 360M | **STABLE** | `Subject-Emu-5259/NeuralAI` | `checkpoints/v2_model` | 2026-07-14 |
| NeuralAI 360M DPO | **v17 / d17** | `HuggingFaceTB/SmolLM2-360M-Instruct` | `data/train_dpo_v16_combined.jsonl` (679 pairs) | 360M | **PUSHED** | `Subject-Emu-5259/NeuralAI` | `checkpoints/v17-dpo` | 2026-07-27 |
| NeuralAI-Air 135M SFT | **v17** | custom `neuralai-air` 135M base | `data/train_sft_v17.jsonl` (37 pairs) | 135M | **PUSHED** | `Subject-Emu-5259/NeuralAI-Air-135M-SFT` | `models/NeuralAI-Air-135M-SFT-v3.gguf` | 2026-07-26 |
| NeuralAI-Air 135M SFT | **v18** | custom `neuralai-air` 135M base | `data/train_sft_v18.jsonl` (≥500 pairs) | 135M | **COMPLETE** | `Subject-Emu-5259/NeuralAI-Air-135M-SFT-v18` | `checkpoints/v18-sft` | 2026-07-27 |

## Active Datasets (do not delete)

| File | Lines | Used By | Notes |
|------|-------|---------|-------|
| `data/train_dpo_v16_combined.jsonl` | 679 | d17 | **Current DPO dataset** = v15 (597) + v16 (64) + v16_supplement (18) |
| `data/train_sft_v17.jsonl` | 37 | SFT v17 | Used for Air-135M-SFT v17 |
| `data/train_sft_v18.jsonl` | 500 | SFT v18 | Expanded SFT dataset ready for next Colab run |

## Stale / Historical Datasets

These have been moved to `data/archive/` for provenance and are **not** the current training source:

- `data/archive/train_dpo_v2.jsonl` through `data/archive/train_dpo_v14.jsonl` — intermediate DPO iterations.
- `data/archive/train_dpo_v16.jsonl` + `data/archive/train_dpo_v16_supplement.jsonl` — components already merged into `train_dpo_v16_combined.jsonl`.
- `data/archive/train.jsonl` — legacy generic instruction dataset (347 lines).

The largest pre-combined base `data/train_dpo_v15.jsonl` (597 pairs) remains at root for provenance.

## Checkpoints

| Path | What It Is | Action Needed |
|------|------------|---------------|
| `checkpoints/v2_model` | d16 final adapter | Keep as rollback / archive |
| `checkpoints/v17-dpo` | d17 final adapter (r=32, α=64, 129 steps, 97.5% reward accuracy) | Live — restored from HF `Subject-Emu-5259/NeuralAI`; push updates here |
| `models/NeuralAI-Air-135M-SFT/` | SFT v17 full PyTorch model (133.72M params) | Keep as local source of truth |
| `models/NeuralAI-Air-135M-SFT-v3.gguf` | **SFT v17 GGUF** (Q4_K_M) currently served by LM Studio | Live inference model |
| `models/NeuralAI-Air-135M-GGUF/NeuralAI-Air-135M-SFT.Q4_K_M.gguf` | Duplicate/alternate GGUF export of SFT v17 | Review; probably safe to delete once v3 is confirmed live |
| `checkpoints/v18-sft` | SFT v18 full model snapshots (`final/`, `stabilized/`, `hyper_converged/`, `checkpoint-32/`) | Keep as local source of truth |
| `checkpoints/` loose `.pt` / config files | Orphaned intermediate artifacts | Review before deleting |

## Scripts

- `colab/colab_d17_train.ipynb` → d17 (360M DPO) — **RAN**
- `colab/train_sft_v17.py` / `colab/NeuralAI_Air_135M_SFT_v17.ipynb` → SFT v17 (135M) — **RAN**
- `colab/train_sft_v18.ipynb` → SFT v18 (135M, ≥500 pairs) — **RAN**
- `colab/v17_scale_up_1_7B.ipynb` → optional 1.7B scale-up (future)

## Next Actions

1. Push SFT v18 artifacts to HF `Subject-Emu-5259/NeuralAI-Air-135M-SFT-v18` (token refresh needed).
2. Convert SFT v18 final checkpoint to GGUF for LM Studio inference.
3. Expand SFT dataset beyond 500 pairs and plan next training run.

## Hugging Face Organization

Collection: [NeuralAI Model Family](https://huggingface.co/collections/Subject-Emu-5259/neuralai-model-family-6a66ee29c7c5f26e044dee3c)
- `Subject-Emu-5259/NeuralAI` — D17 360M DPO LoRA adapter (live)
- `Subject-Emu-5259/NeuralAI-Air-135M` — custom 135M base
- `Subject-Emu-5259/NeuralAI-Air-135M-SFT` — SFT v17 of the 135M base
- `Subject-Emu-5259/NeuralAI-Air-135M-SFT-v18` — SFT v18 (expanded ≥500 pairs, scaled weight init, precision masking)
