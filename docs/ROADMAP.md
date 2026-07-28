# 🚀 NeuralAI Development Roadmap

**Version Target: 6.0 (The Workstation Pivot) + 7.3 (D17 DPO Runtime)**
**Last Updated: July 20, 2026**

---

## ✅ Completed Milestones (NeuralAI Legacy)

### Phase 0: Core System ✓

- [x] SmolLM2-360M base model fine-tuned with QLoRA

- [x] Chat streaming (SSE) working

- [x] Web UI deployed (NeuralAI → NeuralAI v1.0)

- [x] Unified Service Migration: Consolidated Model + UI + Terminal into `file neural_core_service.py`

- [x] Fixed Chat streaming and Terminal consistency

- [x] Improved SSE responsiveness with tool execution indicators and non-blocking stream chunks.

### Phase 1: Tool Ecosystem ✓

- [x] Code Execution Sandbox, File Manager, Web Fetcher, DB Connector, Git Assistant

- [x] Tool detection and routing in chat

### Phase 2: DPO Alignment ✓

- [x] DPO training pipeline implemented

- [x] Preference dataset expanded to 244 pairs (v12.0)

- [x] **D17 DPO continuation (v16 → v17):** 679 pairs, 3 epochs / 129 steps, reward accuracy 97.5%, stable entropy — no collapse (2026-07-20)

- [x] Memphis Culture & Founder Context integration

- [x] **D17 DPO continuation (v16→v17, 2026-07-20):** 679 pairs, 3 epochs / 129 steps, reward accuracy 97.5%; adapter at `checkpoints/v17-dpo` + HF `Subject-Emu-5259/NeuralAI`.

---

## 🏗️ Phase 3: NeuralAI Evolution (In Progress)

### 1. Workstation Orchestration

- [x] Establish distinction: NeuralAI (Model) vs. NeuralAI (Hub)

- [x] UI Overhaul: Added "Workstation Dashboard" tab with project/model/shell status

- [x] Robust Multi-Turn Context Support (10-message sliding window)

- [x] Integrated Multi-Modal Speech-to-Speech (Gemini Live + ElevenLabs Fallback)

- [ ] Transition UI from Chat-Only to Multi-Panel Workstation (Expand dashboard features)

- [ ] Implement System-Wide Context Layer

- [ ] Add "Vibe Stack" Workflow Registry

### 2. Neural Knowledge Graph

- [ ] Implement Persistent Memory (Graph-based)

- [ ] Automate Infrastructure Learning

- [ ] Sync with Supermemory

---

## 📊 System Status

- **Main Service:** **READY & RESILIENT** (`webui_service.py` v7.2.0 — auto-restart, memory watchdog, ZO-native inference backend)
- **Voice Service:** **READY** (ElevenLabs v2 Migrated)
- **Model:** SmolLM2-360M-Instruct + DPO v15.0 (Deployed)
- **Context:** System-wide (Expanding)

### Latest DPO Run (v15.0)

- **Training samples:** 597 (expanded from 302)
- **Epochs:** 3
- **Steps:** 450
- **Final training loss:** `0.305`
- **Reward margin:** `~0.5` → `~3.5`
- **Hardware:** Apple Silicon MPS (MacBook Air M4)
- **Duration:** `730.5s` (~12 min)
- **Completed:** `2026-07-11 20:00 UTC`
- **Adapter:** live on HF `Subject-Emu-5259/NeuralAI`

### Deployment & Inference (updated July 15, 2026)

- **Host:** ZO Computer (Free plan, 4 GB RAM) at `neuralai-deandrewharris.zocomputer.io`
- **Inference backend:** `LLM_BACKEND=zo` → ZO native `/zo/ask` using the user's own BYOK model (`byok:0d3567f7-f521-42b0-8adf-65c9b036cf89`). Uses **0 MB local RAM**.
- **Why not local:** Loading PyTorch + SmolLM2-360M on a 4 GB box used ~6.2 GB → OOM-kill loop that paused the service. The ZO backend removes that dependency entirely (see `docs/INCIDENT-2026-07-14-NEURALAI-PAUSES.md`).
- **Local PyTorch backend:** still available via `LLM_BACKEND=local` on GPU/Colab-class machines (≥8 GB RAM) for offline/private inference.
- **Resilience:** supervisor auto-restart, `/api/health` keepalive, and a memory watchdog that runs GC before any OOM. Service verified stable after the July 15 fix (free RAM returned to ~3 GB).

---

## 🎯 Next Steps (Priority Order)

### 1. Training Data Expansion

**Status:** In progress (500+ samples reached, target 1000+)

**Categories to expand:**

- Symbolic Logic & Formal Proofs: +50 samples
- Security & Vulnerability Analysis: +50 samples
- Multi-Step Algorithmic Reasoning: +50 samples
- Advanced Mathematics (Calculus/Linear Algebra): +50 samples

### 2. Evaluation Suite

**Status:** Created, pending execution

**Benchmarks:**

- Code correctness: Generated code runs
- Response helpfulness: Quality scoring
- Safety: Refuses harmful requests
- Latency: Inference speed

### 3. Scale the Base Model (optional, targeted for v17)

**Status:** Planned — config swap, not a rewrite

- Graduate the inference base from **SmolLM2-360M-Instruct** → **SmolLM2-1.7B-Instruct** to address context length + reasoning depth together.
- Backends are already pluggable (llmster/LM Studio on port 1234, `LLM_BACKEND=zo|local`), so this is a **model config swap** (update the loaded GGUF / backend model id), not a code rewrite.
- Trade-off: ~4–6× more RAM than the 360M (~258 MB) build. Validate on a GPU/Colab-class machine (≥8 GB RAM) before flipping the live Zo Free-plan service, which currently runs the 360M build for OOM safety.
- **Training path:** Use **Google Colab CLI** (`colab-cli` / `google-colab` via the `gcolab` package or `kaggle`+Colab bridge) to run DPO/training notebooks remotely — see Phase 6 note below. No local GPU required; push `NeuralAI_TPU_Training.ipynb` and `training/train_dpo.py` to a Colab runtime and pull the adapter back to HF `Subject-Emu-5259/NeuralAI`.

---

## 🚀 Future Phases (The Agentic Horizon)

### Phase 4: Agentic Autonomy & Computer Use

**Goal:** Transition from "Assistant" to "Operator"

- [x] **Browser Agent Integration**: Embedded NeuralBrowser shipped 2026-07-17 (tab strip, omni bar, AI Mirror + User Browser modes via `/api/browser/*`). **ROLLED BACK 2026-07-19** at user request — UI removed, backend endpoints orphaned. Do NOT re-add unless explicitly asked. (See AGENTS.md REMOVED note.)

- [ ] **Multi-Agent Orchestration**: Ability to spawn and manage specialized sub-agents for parallel task execution.

- [ ] **Long-Horizon Planning**: Implement hierarchical planning for tasks requiring 10+ steps.

- [ ] **Third-Party App Integration**: Direct agentic control over productivity tools (Calendar, Email, CRM).

### Phase 5: Universal Knowledge Integration (The "World-Brain" Training)

**Goal:** Massive expansion of general-world intelligence and cultural context.

- [ ] **Natural World**: Plants, animals, creatures, ecosystems, and biology.

- [ ] **Humanity & Culture**: History, religions, beliefs, sociology, and anthropology.

- [ ] **The Arts**: Music theory, cinematic history, fine arts, and literature.

- [ ] **Global Systems**: Geography, geopolitics, economics, and planetary sciences.

### Phase 6: Model Capability Upgrades

**Goal:** Integration of frontier reasoning and multimodal capabilities.

- [x] **DPO v15 Training — COMPLETE (shipped, not pending)**: 597-pair dataset (3 epochs, 450 steps, loss 0.305, margin ~3.5) trained on Apple Silicon MPS; adapter live on HF `Subject-Emu-5259/NeuralAI`. Marked done — remove from "Next Session Goals" (item 2 below is stale).

- [ ] **Deep Reasoning Integration**: Implement "Think" modes for complex mathematical and logical deduction.

- [ ] **Native Multimodal Understanding**: Unified processing of video, audio, and images in a single context window.

- [ ] **Test-Time Compute Optimization**: Optimize inference to allow the model to "think longer" for harder problems.

- [ ] **Scale the base (optional) — Graduate 360M → SmolLM2-1.7B for v17** to address context + reasoning depth together. More RAM, but backends are already pluggable, so it's a config swap (LM Studio model load + `webui_service.py` `MODEL_NAME`), not a rewrite. Train on **Google Colab CLI** (`colab-cli` / `gcolab` package or `kaggle`+Colab bridge) — no local GPU required; push `NeuralAI_TPU_Training.ipynb` + `training/train_dpo.py` to a Colab runtime and pull the adapter back to HF `Subject-Emu-5259/NeuralAI`. See Training Path note (Phase 1 line 128).

---

## 📊 Data Files

```markdown
data/
├── train.jsonl              # 347 original samples
├── train_v3.jsonl           # 404 samples (latest)
├── train_dpo.jsonl          # 13 DPO pairs
├── train_dpo_expanded.jsonl # 31 DPO pairs
└── train_expanded.jsonl     # 363 samples
```

## 📁 Project Structure

```markdown
NeuralAI/
├── checkpoints/final_model/    # LoRA adapter
├── data/                       # Training data
├── eval/benchmarks.py          # Evaluation suite
├── from-scratch/web_ui/        # Flask app + static files
│   ├── app.py                  # Main Flask server
│   ├── neuralai_engine.py      # Model + tools
│   └── neuralai_router.py      # Routing logic
├── tools/                      # Tool implementations
│   ├── code_sandbox.py
│   ├── file_manager.py
│   ├── web_fetcher.py
│   ├── db_connector.py
│   └── git_assistant.py
└── training/                   # Training scripts
    ├── train_dpo.py
    ├── generate_training_v3.py
    └── NeuralAI_TPU_Training.ipynb
```

---

## 🔗 Quick Links

- **Live Chat:** https://neuralai-deandrewharris.zocomputer.io
- **GitHub:** https://github.com/Subject-Emu-5259/NeuralAI
- **Local Dev:** http://localhost:5000

---

## 📝 Commands

```bash
# Start the service
cd /home/workspace/Projects/NeuralAI/from-scratch/web_ui
python3 app.py

# Generate v5 DPO data
python3 training/generate_dpo_v5.py

# DPO training (currently running in background)
python3 training/train_dpo.py
```

---

**Next Session Goals:**

1. Run evaluation benchmarks
2. ~~Expand training data to 1000+ samples~~ → DPO v15 already shipped (597 pairs, loss 0.305, adapter on HF `Subject-Emu-5259/NeuralAI`). Next data push should target v17 base-swap, not a v15 retry.
3. DPO training runs on **Google Colab CLI** (`colab-cli`/`gcolab` or `kaggle`+Colab bridge) — no local GPU. Push `NeuralAI_TPU_Training.ipynb` + `training/train_dpo.py`, pull adapter to HF `Subject-Emu-5259/NeuralAI`.
4. Plan v17 base graduation: 360M → SmolLM2-1.7B (config swap, pluggable backends) for context + reasoning depth.