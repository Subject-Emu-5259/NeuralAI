# NeuralAI — Release Notes

---

## v7.4 — Two-Model Fleet & Visual Refresh (August 13, 2026)

### ⚡ NeuralAI · Powered by SmolLM2-360M

- **Awareness v2** is the live chat backend as of 2026-08-13
- Dataset v2: 506 prompt/response pairs across brand, model, site, chat, assistant, companion, tools, refusal
- LoRA SFT, r=16 / α=32, 5 epochs / 320 steps, final loss **0.1252**
- Merged to `checkpoints/smol-awareness-sft-v2/merged` and quantized to `models/NeuralAI-Smol-Awareness-v2-Q8_0.gguf` (~369MB)
- Active model id: `smol-awareness-v2-merged`; served by `neuralai-lmstudio` on `127.0.0.1:1234`
- v1 remains archived as the baseline adapter/merged weights and `models/NeuralAI-Smol-Awareness-Q8_0.gguf`
- HuggingFace repo: `Subject-Emu-5259/NeuralAI-Powered-By-SmolLM2360`

### 🧬 Mamba K1

- 130M Mamba SSM first owned base model
- d_model 768, 24 layers, 50280 tokenizer vocab
- Merged safetensors + Q4_K_M/F16 GGUFs hosted on HF
- Chat SFT tooling ready; awaiting GPU run
- HuggingFace repo: `Subject-Emu-5259/NeuralAI-Mamba-K1`

### 🧹 Housekeeping

- Removed Mamba K2/K3 local artifacts, training scripts, and model-manager entries
- Removed Air-135M and legacy DPO checkpoints from active repo
- Model manager now registers only `mamba-k1` and `smollm2-360m`
- README, model card, roadmap, release notes, training manifest all refreshed
- Added branded banners, architecture diagrams, and training graphics to `assets/`
- Hugging Face README cards rebuilt with visuals, specs, usage, and creator/company info

---

## v7.3 — Mamba Era (August 1, 2026)

### 🧬 Owned Base Models

- **Mamba K1** — 129M Mamba SSM, first owned merged weights on HF
- **Mamba K2** — 790M Mamba SSM Q4_K_M GGUF (later archived)
- **Mamba K3** — 790M SFT in planning (later archived)

### 🧹 Housekeeping

- SmolLM2-360M removed from model manager selections
- Web UI formatting upgrades for Mamba
- Benchmark harness added
- Model card, roadmap, release notes refreshed

---

## v17 (D17) — DPO Alignment (July 20, 2026)

- **D17:** 679 preference pairs, 3 epochs / 129 steps, reward accuracy 97.5%
- Stable entropy, no eval set collapse

---

## v7.2 — Service Hardening (July 15–17, 2026)

- Live chat stable on CPU
- llmster inference (258MB RAM)
- 10 slash commands live
- ChatML prompt template
- Image gen and TTS fallbacks

---

## v15–v16 — DPO Foundation (June–July 2026)

- DPO v15: 597 pairs, 3 epochs, 450 steps
- DPO v16 added 64 new pairs
- Apple Silicon MPS training

---

## v6–v7 — Workstation Pivot (May–June 2026)

- Workstation dashboard
- Multi-turn context
- S2S voice
- Unified service architecture

---

## v1–v5 — Foundation (April–May 2026)

- SmolLM2-360M base fine-tuned with QLoRA
- Chat streaming, web UI, tools
