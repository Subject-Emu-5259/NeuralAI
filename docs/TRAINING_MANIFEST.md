# NeuralAI Training Manifest

> Single source of truth for which dataset, script, checkpoint, and artifact corresponds to each active Mamba model. Legacy DPO / Air-135M / SmolLM manifests are preserved in git history (pre-2026-08-02) and the old HF repos; they are no longer tracked here because those adapters and base models were retired.

## Active Model Family

| Model | Version | Base | Dataset | Params | Status | HF Repo | Local Path | When |
|-------|---------|------|---------|--------|--------|---------|------------|------|
| Mamba K1 | **SFT v4** | `state-spaces/mamba-130m-hf` | `data/train_intel_ultrachat_1k_clean.jsonl` (1K single-turn) | 130M | **TRAINING PAUSED — awaiting GPU** | `Subject-Emu-5259/NeuralAI-Mamba-K1` | `models/k1/current/gguf/neuralai-mamba-k1-v4.Q4_K_M.gguf` (target) | 2026-08-02 |
| Mamba K1 | v3 | `state-spaces/mamba-130m-hf` | `data/train_intel_ultrachat_1k.jsonl` (long multi-turn) | 130M | **RETIRED / OVERFIT** | `Subject-Emu-5259/NeuralAI-Mamba-K1` | removed locally; archived on HF | 2026-08-01 |
| Mamba K2 | base | `state-spaces/mamba-790m-hf` | n/a (base pretrained) | 793M | **ACTIVE INFERENCE** | `Subject-Emu-5259/NeuralAI-Mamba-K2` | `models/k2/gguf/mamba-790m-hf.Q4_K_M.gguf` | 2026-08-01 |
| Mamba K3 | base | `state-spaces/mamba-2.8b-slimpj` | n/a (base pretrained) | 2.8B | **QUEUED FOR SFT** | `Subject-Emu-5259/NeuralAI-Mamba-K3` | `models/k3/base/` | 2026-08-01 |

## Active Datasets (do not delete)

| File | Lines | Used By | Notes |
|------|-------|---------|-------|
| `data/train_intel_ultrachat_1k_clean.jsonl` | ~1K | K1 v4 | Current K1 SFT dataset — single-turn, intel format, capped length |
| `data/train_intel_ultrachat_10k.jsonl` | ~10K | K2 / K3 SFT candidate | Larger set for future K2/K3 SFT runs |
| `data/train_sft_v19.jsonl` | ~100 | historical | Last legacy single-turn SFT set; keep for provenance |

## Stale / Historical Datasets Removed

The following folders/files were removed locally because they were superseded, duplicated, or retired:

- `data/archive/` — old DPO iterations (`train_dpo_v2-v14`, `train_dpo_v16*`) and legacy `train.jsonl`. Combined into `data/train_dpo_v16_combined.jsonl` where relevant.
- `data/train_intel_ultrachat_1k.jsonl` — original K1 v3 dataset that caused overfitting; replaced by `_clean.jsonl`.
- `models/k1/sft-v3/` and `archive/k1-v3/` — broken v3 artifacts. Backed up remotely on HF and removed locally.

## Checkpoints

| Path | What It Is | Action Needed |
|------|------------|---------------|
| `checkpoints/k1-lora-sft-v3/` | Last v3 LoRA checkpoint | Keep as rollback/cautionary artifact; do not resume |
| `models/k2/gguf/mamba-790m-hf.Q4_K_M.gguf` | **Live inference model** | Keep; promote to K1 v4 when ready |
| `models/k1/current/gguf/` | Target for K1 v4 GGUF | Empty until GPU run completes |
| `models/k3/base/` | K3 base weights | Keep; run SFT after K1/K2 |

## HuggingFace Repos

- `Subject-Emu-5259/NeuralAI-Mamba-K1` — K1 v3/v4 artifacts (GGUF + safetensors)
- `Subject-Emu-5259/NeuralAI-Mamba-K2` — K2 Q4_K_M GGUF
- `Subject-Emu-5259/NeuralAI-Mamba-K3` — K3 base (future)

## Next Steps

1. Run `exports/k1-v4-gpu/` on a CUDA GPU to train K1 v4 LoRA.
2. Merge adapter into base, quantize to Q4_K_M.
3. Copy GGUF to `models/k1/current/gguf/neuralai-mamba-k1-v4.Q4_K_M.gguf`.
4. `python3 scripts/model_manager.py set mamba-k1` to promote to live inference.
