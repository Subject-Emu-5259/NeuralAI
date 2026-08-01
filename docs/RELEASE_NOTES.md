# NeuralAI — Release Notes

---

## v7.3 — Mamba Era (August 1, 2026)

### 🧬 Owned Base Models

**Mamba K1 — First self-owned base model.** NeuralAI now owns full model weights, not just LoRA adapters on someone else's base. 129M Mamba SSM parameters, SFT on 1,000 UltraChat samples (50 steps), merged and published as standalone safetensors (493MB) to `Subject-Emu-5259/NeuralAI-Mamba-K1`.

**Mamba K2 — Scaled GGUF base.** 6× scale-up from K1: 790M parameters, 48 layers, hidden size 2048. Quantized to Q4_K_M GGUF (460MB), ready for LM Studio / llama.cpp. Published to `Subject-Emu-5259/NeuralAI-Mamba-K2`.

**Mamba K3 — Full SFT training.** 790M base model undergoing 500–1000 step SFT on 10K+ UltraChat samples. LoRA rank 32. Target loss < 3.0. Training on Google Colab.

### 🧹 Housekeeping

- **SmolLM2-360M** removed from model manager selections (retired in favor of Mamba K1/K2 + NeuralAI v17 DPO)
- **Web UI** formatting upgrades for Mamba models and structured info output
- **Benchmark harness** added (`benchmarks/run_evals.py`)
- **Chat format** module created (`chat/chat_format.py`)
- **Model card, roadmap, release notes** all refreshed for Mamba era
- **Company card & architecture card** updated in web UI Settings

---

## v17 (D17) — DPO Alignment (July 20, 2026)

- **D17 DPO model**: 679 preference pairs, 3 epochs / 129 steps, reward accuracy 97.5%
- Published to HF `Subject-Emu-5259/NeuralAI`
- Stable entropy, no eval set collapse
- Web UI: landing page, Model Status badge, `/api/health` all reflect v17

---

## v7.2 — Service Hardening (July 15–17, 2026)

- **Live chat stable on CPU**: model eager-loaded at startup (no cold-start 502s)
- **llmster inference**: 258MB RAM vs 5GB PyTorch, solved OOM-kill loop
- **Embedded NeuralBrowser**: shipped (AI Mirror + User Browser modes), later removed at user request
- **NL→Tool Router**: plain-English web requests auto-routed to correct tools
- **10 slash commands**: all live and verified (`/web`, `/fetch`, `/browse`, `/research`, `/img`, `/speak`, `/summarize`, `/translate`, `/news`, `/yt`)
- **ChatML prompt template**: fixed self-talk regression
- **Image gen**: inline rendering in chat, Pollinations flux + OpenRouter Gemini fallback
- **TTS**: gTTS fallback (Gemini TTS 401'd)

---

## v15–v16 — DPO Foundation (June–July 2026)

- DPO v15: 597 pairs, 3 epochs, 450 steps, loss 0.305, reward margin ~3.5
- DPO v16: 64 new pairs added (debugging, logic, multi-step reasoning)
- ChatML template adoption across training and inference
- Apple Silicon MPS training (~12 min per run on MacBook Air M4)

---

## v6–v7 — Workstation Pivot (May–June 2026)

- Multi-panel Workstation Dashboard
- Multi-turn context (10-message sliding window)
- Speech-to-speech (Gemini Live + ElevenLabs fallback)
- Unified service architecture (model + UI + terminal consolidation)

---

## v1–v5 — Foundation (April–May 2026)

- SmolLM2-360M base fine-tuned with QLoRA
- Chat streaming (SSE)
- Web UI deployed
- Code Execution Sandbox, File Manager, Web Fetcher, DB Connector, Git Assistant
- Tool detection and routing in chat
