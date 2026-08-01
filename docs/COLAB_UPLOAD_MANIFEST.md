# 🚀 Colab Upload Manifest

This manifest defines the mapping between local workspace files and their intended destinations within the Google Colab environment for NeuralAI v18 training.

## 📂 Model & Weights
| Local File / Folder | Colab Destination Path | Description |
| :--- | :--- | :--- |
| `/home/.z/workspaces/con_Be6MM5KUzfA88RWI/neuralair-135m/neuralair-135m/final.pt` | `/home/.z/workspaces/con_Be6MM5KUzfA88RWI/neuralair-135m/neuralair-135m/final.pt` | Base weights (final.pt) |
| `/home/workspace/Projects/NeuralAI/NeuralAI-v2-merged` | `/content/NeuralAI/NeuralAI-v2-merged` | Merged Model Config & Tokenizer |

## 🧪 Training Data & Scripts
| Local File | Colab Destination Path | Description |
| :--- | :--- | :--- |
| `/home/workspace/Projects/NeuralAI/data/train_sft_v18.jsonl` | `/content/train_sft_v18.jsonl` | SFT Training Dataset (v18) |
| `/home/workspace/Projects/NeuralAI/data/train_dpo_v18.jsonl` | `/content/train_dpo_v18.jsonl` | DPO Training Dataset (v18) |
| `/home/workspace/Projects/NeuralAI/training/train_sft_v18.py` | `/content/train_sft_v18.py` | SFT Training Script |
| `/home/workspace/Projects/NeuralAI/training/train_sft_v18.ipynb` | `/content/train_sft_v18.ipynb` | SFT Training Notebook |
| `/home/workspace/Projects/NeuralAI/training/train_dpo_v18.py` | `/content/train_dpo_v18.py` | DPO Training Script |
| `/home/workspace/Projects/NeuralAI/training/train_dpo_v18.ipynb` | `/content/train_dpo_v18.ipynb` | DPO Training Notebook |

## 💾 Outputs
| Local Output Path | Colab Output Path | Description |
| :--- | :--- | :--- |
| `/home/workspace/Projects/NeuralAI/checkpoints/v18-sft` | `/content/checkpoints/v18-sft` | SFT Checkpoints |
| `/home/workspace/Projects/NeuralAI/checkpoints/v18-dpo` | `/content/checkpoints/v18-dpo` | DPO Checkpoints |

---
**Note**: Ensure `HF_TOKEN` is set as an environment variable or in Colab Secrets before running scripts.
