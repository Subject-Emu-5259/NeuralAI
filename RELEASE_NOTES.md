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
