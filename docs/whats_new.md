# 📰 NeuralAI — What's New

_Last updated: August 13, 2026_

## ⚡ NeuralAI · Powered by SmolLM2-360M

The live chat backend is now an **awareness-tuned SmolLM2-360M-Instruct** fine-tune trained to answer as NeuralAI.

- **Base:** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Dataset v1:** 83 pairs across brand, model, site, chat, assistant, and companion categories
- **Training:** LoRA SFT, r=8 / α=16, 3 epochs, final loss **2.7186**
- **Live artifact:** `models/NeuralAI-Smol-Awareness-Q8_0.gguf`
- **HF Repo:** `Subject-Emu-5259/NeuralAI-Powered-By-SmolLM2360`
- **Dataset v2:** 506 pairs, 9 categories (in training)

## 🧬 Mamba K1 — First Owned Base Model

NeuralAI's first fully owned Mamba SSM checkpoint.

- **Architecture:** Mamba SSM — 130M parameters, d_model 768, 24 layers
- **Artifacts:** merged safetensors + Q4_K_M/F16 GGUFs
- **HF Repo:** `Subject-Emu-5259/NeuralAI-Mamba-K1`
- **Status:** R&D; chat SFT pipeline queued for GPU

## 🧹 Fleet Cleanup

- Removed Mamba K2/K3 files and references.
- Removed Air-135M and old DPO checkpoints from active code and model manager.
- Model manager now registers only the two active models.
- Updated README, model card, roadmap, release notes, training manifest, and HF repos.

## 📊 Docs & Visual Refresh

- New hero banners, architecture, comparison, and training diagrams in `assets/`.
- Hugging Face README cards rebuilt with branded graphics, full specs, usage code, and company info.
- GitHub README and model card simplified to the two-model fleet.

---

## Previous (2026-07-31)

### Mamba K1 — First Owned Base

- 130M Mamba SSM, 50 SFT steps, 1K samples, ~19 tok/s CPU
- First model trained from a base architecture instead of fine-tuning someone else's transformer

### Mamba K2 (793M) — Scaled Up

- 793M Mamba base quantized to Q4_K_M GGUF for local inference
- Later archived as project focus narrowed

### Web UI Upgrades

- Mamba model info cards
- Mamba chat template and structured output
- Model manager registered Mamba K1, K2, NeuralAI v17 DPO

### Browser Engine

- From-scratch layout engine: DOM, CSS, style, layout, paint
- render_page() returns title, text, links, headings, screenshot
