# NeuralAI Release Notes

## v7.3 — Mamba Era (2026-08-01)

### 🧬 NeuralAI's First Owned Base Models

This release marks the launch of NeuralAI's **Mamba model family** — the first base models fully owned and trained by NeuralAI, built on the Mamba SSM architecture.

### Mamba K1 — First Owned Base Model ✅

- **130M parameters** — Mamba SSM architecture
- SFT trained on UltraChat (50 steps, LoRA rank 8)
- Merged safetensors model (493MB)
- Deployed to Hugging Face: `Subject-Emu-5259/NeuralAI-Mamba-K1`
- ~19 tok/s on CPU, proof of concept for NeuralAI-owned weights

### Mamba K2 — Scaled & Deployable ✅

- **793M parameters** — Mamba SSM architecture
- Converted to GGUF Q4_K_M (460MB)
- Ready for LM Studio / llama.cpp inference
- Deployed to Hugging Face: `Subject-Emu-5259/NeuralAI-Mamba-K2`
- Laid training infrastructure for future SFT/DPO alignment

### Mamba K3 — Full SFT Training 🔄

- **790M parameters** — same base as K2
- Training pipeline: 500-1000 SFT steps, 10K+ UltraChat samples
- LoRA fine-tuning with rank 16, targeting SSM projection layers
- Benchmarks: perplexity, generation diversity, reasoning
- Colab notebook at `colab/colab_mamba_k3_train.ipynb`
- Training script at `train_k3_lora.py`

### Web UI Upgrades

- Mamba model family cards on welcome screen and settings
- Updated model manager with `mamba-k1`, `mamba-k2`, `mamba-k3` entries
- Chat formatting and information structure enhancements
- Service version bumped to v7.3 (Mamba Era)

### Cleanup

- Removed SmolLM2-360M and Air-135M from model manager selections
- Removed orphaned Air architecture conversion scripts
- Consolidated all docs under Mamba Era

### What's Next

- Complete K3 SFT training and evaluation
- Scale to 2B/3B Mamba architectures
- Deploy K3 as production inference engine
- DPO alignment for K3

## v7.3.1 — SFT Pipeline & Chat Format Fix (2026-08-01)

### Problem

After the v7.3 Mamba launch, live tests showed that all three Mamba models produced token-soup / uncoherent output in chat:

- **K1** was undertrained (50 SFT steps on bad Llama-2/ChatML format with out-of-vocabulary special tokens).
- **K2 and K3** were still raw base models with no instruction tuning.
- The production llama.cpp server was using `llama-2` / `chatml` chat handlers, which rely on `</s>` / `<|im_start|>` tokens that do **not** exist in Mamba's GPT-NeoX tokenizer.

### Fix

- **Vocabulary-friendly "intel" prompt format**: plain-text `### System/User/Assistant:` prompts, using only tokens that exist in the Mamba tokenizer.
- **Custom `lmstudio_server.py`**: registers a `neuralai-intel` chat handler so llama.cpp serves the trained format correctly.
- **Assistant-only loss**: `training/train_mamba_lora.py` now masks the user/system portion of each example and only trains the model to predict assistant tokens.
- **CPU-safe training launcher per model**:
  - `train_k1.sh` — 500 SFT steps on 1K UltraChat (K1 130M)
  - `train_k2.sh` — SFT on 10K UltraChat (K2 793M)
  - `train_k3.sh` — SFT on 10K UltraChat (K3 2.8B)
- **Colab notebook**: `colab/NeuralAI_Mamba_SFT_Training.ipynb` with dependency-pinned cells for GPU training.
- **Docs updated**: README, landing page, privacy/terms, and HF model cards now describe the corrected Mamba state.

### Status After Fix

| Model | State |
|-------|-------|
| Mamba K1 | 🔄 SFT v2 retraining (intel format) |
| Mamba K2 | ⚠️ Base pretrained — awaiting SFT |
| Mamba K3 | ⚠️ Base pretrained — awaiting SFT |

### Next Step

Run GPU SFT on K1/K2/K3 (Colab notebook), then merge adapters, convert Q4_K_M GGUF, and republish to Hugging Face.
