# Colab / GPU Upload Manifest

This manifest maps the local `exports/k1-v4-gpu/` package to the paths expected by the GPU training notebook.

## 📂 Model & Weights

| Local File / Folder | Colab Destination Path | Description |
| :--- | :--- | :--- |
| `models/k1/base/` | `/content/NeuralAI/models/k1/base/` | Mamba K1 130M base weights |
| `exports/k1-v4-gpu/train_k1_v4_gpu.py` | `/content/NeuralAI/train_k1_v4_gpu.py` | GPU SFT training script |
| `exports/k1-v4-gpu/merge_and_export.py` | `/content/NeuralAI/merge_and_export.py` | Merge + quantize script |

## 🧪 Training Data

| Local File | Colab Destination Path | Description |
| :--- | :--- | :--- |
| `exports/k1-v4-gpu/train_intel_ultrachat_1k_clean.jsonl` | `/content/NeuralAI/data/train_intel_ultrachat_1k_clean.jsonl` | K1 v4 SFT dataset (single-turn, intel format) |

## 💾 Outputs

| Local Output Path | Colab Output Path | Description |
| :--- | :--- | :--- |
| `models/k1/current/gguf/neuralai-mamba-k1-v4.Q4_K_M.gguf` | `/content/NeuralAI/models/k1/current/gguf/neuralai-mamba-k1-v4.Q4_K_M.gguf` | Desired quantized artifact to bring back |

## ⚠️ Retired / Removed

The following legacy entries were removed because the associated artifacts no longer exist:

- `NeuralAI-v2-merged` — K1 model layout is now `models/k1/base/` + `models/k1/current/`.
- `neuralair-135m` / v18 SFT & DPO scripts — legacy Air-135M and SmolLM pipelines retired.

---
**Note**: Ensure `HF_TOKEN` is set as an environment variable or in Colab Secrets before running scripts.
