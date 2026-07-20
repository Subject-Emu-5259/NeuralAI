---
language:
  - en
library_name: peft
license: apache-2.0
tags:
  - lora
  - conversational
  - text-generation
  - peft
  - smollm2
  - dpo
  - fine-tuned
model_id: Subject-Emu-5259/NeuralAI
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
inference: false
---

# 🧠 NeuralAI: The Generative AI Engine

<img src="neuralai_banner.svg" alt="NeuralAI - Your AI. On your hardware. In your browser." />

## 📊 Repository Composition

| Language | Percentage |
| --- | --- |
| Python | 71.1% |
| HTML | 13.0% |
| JavaScript | 12.4% |
| CSS | 2.6% |
| Shell | 0.4% |
| Jupyter Notebook | 0.3% |
| Jinja | 0.2% |

**The High-Velocity Model for Your Entire Vibe Stack**

NeuralAI is the central intelligence engine developed by **De'Andrew Preston Harris**. Conceived and engineered by **De'Andrew Preston Harris** (Founder), it is a highly tuned, DPO-aligned multimodal AI ec\[...\]

---

## 🌟 Vision & Manifesto

NeuralAI doesn't just predict text; it *operates the work*. The core mission is to create a multimodal generative system that bridges the gap between raw idea and execution. By fusing autoregressi\[...\]

Born from resilience and ambition in Memphis, Tennessee and West Memphis, Arkansas, NeuralAI represents a forward-thinking approach to personal, private AI computing.

---

## 🛠️ Tech Stack & Architecture (v7.3)

NeuralAI is built on a high-performance architecture that decouples the inference engine from the web interface, enabling lightweight cloud hosting with powerful local inference.

### Core Stack

- **Core Model**: `SmolLM2-360M-Instruct` fine-tuned with the custom **DPO v17 (D17) LoRA** at `checkpoints/v17-dpo` — aligned for logic, math, multi-step reasoning, debugging, and safe refusals
- **Inference Engine**: [llmster](https://lmstudio.ai/docs/cli) (LM Studio headless) — OpenAI-compatible API with continuous batching, running via llama.cpp
- **Vocal Identity**: Andrew (Warm/Multilingual) — Optimized for Live Speech-to-Speech (S2S)
- **Backend Framework**: Python / Flask (Core Service) — routes to llmster or local PyTorch
- **Storage & Database**: SQLite3 (Metadata) + Nextcloud Hub via NeuralCloud WebDAV Client (NeuralDrive)
- **Frontend UI**: Vanilla JS, HTML5, CSS3 with an advanced Dark Mode layout

### Pluggable LLM Backend

NeuralAI supports multiple inference backends via the `LLM_BACKEND` environment variable:

| Backend | `LLM_BACKEND` | API Endpoint | Use Case |
| --- | --- | --- | --- |
| **llmster** (recommended) | `lmstudio` | `http://localhost:1234/v1` | Headless GPU/CPU inference |
| **Ollama** | `ollama` | `http://localhost:11434/v1` | Local Ollama server |
| **OpenAI-compatible** | `openai_compatible` | Any OpenAI API URL | Remote/cloud inference |
| **Local PyTorch** | `local` | Built-in transformers | Loads BASE_MODEL + LoRA at MODEL_PATH in float16 (your own model) |
| **ZO Native (fallback)** | `zo` | `https://api.zo.computer/zo/ask` | Routes to Zo's own assistant (HY3) — **NOT** your NeuralAI model; last-resort only |
### Resilient Launcher (2026-07-15)
The service entrypoint is now `run_service.sh`, which auto-selects the backend at boot:
1. Starts **llmster** (LM Studio) headless on port 1234 and loads the `smollm2-360m-instruct` GGUF.
2. If llmster is reachable, sets `LLM_BACKEND=openai_compatible` → `http://localhost:1234/v1` (recommended; low RAM).
3. Only if llmster is unavailable does it fall back to the local PyTorch backend.

This guarantees the chat always has a live backend (no more 503/401 "model not authed" stalls) and keeps RAM low (llmster ~1.1 GB vs ~6 GB for local transformers), so the UI no longer freezes under memory pressure.

> **Hosting on ZO Computer (4 GB RAM):** set `LLM_BACKEND=local`. The service loads `BASE_MODEL`
> (default `HuggingFaceTB/SmolLM2-360M-Instruct`) and applies the LoRA at `MODEL_PATH`
> (default `checkpoints/v17-dpo`) in float16 (~720 MB), which fits the 4 GB host. **Do not use
> `LLM_BACKEND=zo` for the chat UI** — it proxies to Zo's assistant and answers as "Zo Computer's
> assistant" instead of your trained model.

```bash
# Example: start NeuralAI with llmster backend
LLM_BACKEND=lmstudio LLM_API_URL=http://localhost:1234/v1 LLM_MODEL=smollm2 \
  python3 services/neural_core_service.py
```

### Core Architectural Pillars

1. **NeuralAI Core**: Handles chat state, direct model inference, terminal session proxying, and tool orchestration.
2. **NeuralDrive (Cloud Storage)**: The intelligent data layer for all projects, featuring isolated user storage, automatic versioning, and semantic mapping.
3. **Diffusion Engine**: An integrated generative diffusion layer for producing visual branding assets, UI mockups, and visual logic maps.
4. **Agentic Orchestrator**: A high-autonomy layer enabling NeuralAI to plan, reason, and execute multi-step workflows across the OS and web, moving beyond simple chat to active goal achievement.

---
### 🆕 What's New (v7.3.3)

- **Developer / API Access (BYO API)**: Generate a personal API key from Settings to use NeuralAI as an OpenAI-compatible backend on other hosts (e.g. ZO Computer's "Bring Your Own Key"). Exposes `/v1/chat/completions` (SSE, CORS-enabled) and `/v1/models`; Base URL is `https://neuralai-web-ui-deandrewharris.zocomputer.io/v1`, model id is `neuralai`. Keys are hashed and revocable. Full ZO Computer setup walkthrough: `docs/BYOK_ZO_INTEGRATION.md`.
- **Auto Release Notes ("What's New")**: A new top-bar panel surfaces the latest features and fixes automatically. Open it anytime via the ✨ **What's New** button.
- **Generated images render in chat**: Image-generation responses are parsed as Markdown and displayed inline.
- **No more self-talk**: Chat now uses the ChatML prompt template (`apply_chat_template`), matching the model's training format.
- **NeuralDrive upload reliability**: The file list correctly handles API response shapes after an upload.
- **Dark theme by default**: UI restores your saved theme and defaults to dark mode (fixes white file-cards in light mode).
- **Phase 8 in progress**: Knowledge Graph & Agentic Autonomy — long-term cross-project memory ("Supermemory") and fully autonomous task execution.
- **The NeuralLabs Shift**: NeuralAI is evolving into a standalone, downloadable intelligence environment (NeuralLabs v1 Client → v2 Edge → v3 Eco).
- **v7.3.3 — Public site restored (service reliability)**: Fixed a 404 on the root page and guest auth. Root cause: `/api/auth/guest` is a POST route hit with GET; corrected the request method. All critical endpoints (`/`, `/api/files`, `/api/tool`, `/api/auth/guest`) now return 200 and the live service is confirmed up. See `data/release_notes.json`.
- **v7.3.2 — Backend identity fix (ZO hosting)**: The hosted service now runs `LLM_BACKEND=local` so chat uses *your* SmolLM2-360M + SFT/DPO v16 LoRA. The previous `zo` fallback proxied to Zo's assistant and answered as "I'm Zo Computer's assistant" — that was a routing bug, not your model. See `docs/INCIDENT-2026-07-14-NEURALAI-PAUSES.md`.
## 🚀 Deployment & Model Distribution

- **Source (GitHub)**: [Subject-Emu-5259/NeuralAI](https://github.com/Subject-Emu-5259/NeuralAI)
- **Model (Hugging Face)**: [Subject-Emu-5259/NeuralAI](https://huggingface.co/Subject-Emu-5259/NeuralAI) — SmolLM2-360M + DPO v17 (D17) LoRA adapter (PEFT). The adapter is in `checkpoints/v17-dpo` and also published to `Subject-Emu-5259/NeuralAI`.
- **Hosted demo**: `neuralai-web-ui-deandrewharris.zocomputer.io` (ZO Computer) — runs the local backend so chat uses the trained model directly.

To publish the LoRA to Hugging Face:

```bash
pip install huggingface_hub
HF_TOKEN=<your-write-token> python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/v17-dpo',
    repo_id='Subject-Emu-5259/NeuralAI',
    repo_type='model',
    commit_message='NeuralAI SmolLM2-360M DPO v17 (D17) LoRA',
)
"
```

## ✨ Key Features & Capabilities

### 💬 Multimodal Chat & Agentic Intelligence

- **High-Velocity Text Inference**: Fast, local inference with deep context awareness.
- **Deep Reasoning Mode**: Integration of test-time compute and chain-of-thought reasoning for complex problem decomposition and error-free logic.
- **Autonomous Agentic Workflows**: Ability to operate as an agent—interacting with the browser, terminal, and third-party apps to complete end-to-end tasks with minimal supervision.
- **Live S2S (Speech-to-Speech)**: Real-time voice interaction with an integrated microphone interface and fluid vocal responses.
- **Identity Vault & Memory**: Persistent user memory and rule constraints, ensuring NeuralAI remembers preferences, behavioral rules, and historical context.

### 💻 Developer & Engineering Tools

- **Integrated Web Terminal**: A fully functional, WebSocket-driven terminal embedded directly in the web UI for immediate environment control.
- **File Workspace**: An in-browser IDE experience allowing users to browse directories, read, and write code seamlessly.
- **Code Execution & Sandbox**: Secure environment for the model to execute and test code on the fly.

### 🔐 Authentication & Access Tiers

- **Founder Mode**: Ultimate root-level access and system control.
- **Maestro Student Portal**: Tiered access for educational and collaborative development.
- **Guest Preview**: Frictionless instant access for testing the system without an account.

---

## 🏋️ Model Training & Fine-Tuning (DPO)

NeuralAI is continuously learning and improving through rigorous **Direct Preference Optimization (DPO)**.

### Training Pipeline

```python
# Example of the DPO alignment configuration used in NeuralAI
dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=1024,
    max_prompt_length=512,
)
```

- **Dataset Expansions**: The dataset is aggressively expanded to include advanced reasoning, complex mathematics, logical deduction, creative writing, and system debugging.
- **Behavioral Alignment**: NeuralAI is aligned using Gemini-style behavioral principles—prioritizing safety, structured reasoning, helpful conversational flow, and transparent step-by-step explanations. Training enforces clear Markdown formatting, code-first responses, and rejection of boilerplate or overly verbose outputs.
- **Model Drift Monitoring**: Continuous evaluation against previous checkpoints to ensure response quality and consistency never regress.

### Latest Alignment Run: v17.0 (D17)

- **Training samples**: 679 DPO preference pairs (`data/train_dpo_v16_combined.jsonl`)
- **Epochs**: 3
- **Steps**: 129
- **Final training loss**: `~0.39` (0.692 → 0.396)
- **Reward accuracy**: 0.975 (chosen preferred over rejected)
- **Reward margin**: ~0.9 (stable, no collapse)
- **Hardware**: NVIDIA GPU (Google Colab)
- **Run duration**: `1876s` (~31m)
- **Completed**: `2026-07-20`
- **Adapter**: live on Hugging Face at [`Subject-Emu-5259/NeuralAI`](https://huggingface.co/Subject-Emu-5259/NeuralAI)

> The D17 dataset (`data/train_dpo_v16_combined.jsonl`) is a 679-pair DPO continuation of the v16 adapter, spanning code correctness, safety refusals, identity, math, debugging, code style, conciseness, tool usage, reasoning, and logic.

---

## 📸 Brand & UI Gallery

*(UI screenshots showcase the beautiful dark mode interface, the terminal integration, and the NeuralDrive file explorer.)*

```html
<!-- Example Frontend UI Component Structure -->
<div class="neural-chat-container">
  <div class="message-bubble ai-response">
    NeuralAI: System optimal. Ready for execution.
  </div>
</div>
```

---

## 🗺️ Implementation Roadmap

- ✅ **Phase 1: Alignment** - DPO training for Founder context and optimal engineering tone.
- ✅ **Phase 2: NeuralDrive** - Deployment of the Cloud Storage File Server.
- ✅ **Phase 3: Terminal UI** - Integrated command-line access within the browser.
- ✅ **Phase 4: Live S2S** - High-velocity Live Speech-to-Speech conversations.
- ✅ **Phase 5: "Founder Mode"** - Enhancements to vocal profile and streamlined UI.
- ✅ **Phase 6: Frontend Polish** - Dark themes, real-time code execution display, UI stability.
- ✅ **Phase 7: Diffusion Integration** - Implementation of Text2Img & Img2Img capabilities.
- 🚀 **Phase 8: Knowledge Graph & Agentic Autonomy** - Advanced long-term memory for cross-project context, "Supermemory" features, and fully autonomous task execution.

---

## 🎯 Future Vision: The Software Transition

NeuralAI is evolving from a workspace-bound assistant into a standalone, downloadable intelligence environment.

**Project Code Name**: `NeuralLabs` (Working Title)
**Vision**: A local-first, AI-native operating environment that integrates the Agentic Orchestrator, World-Brain, and NeuralDrive into a seamless desktop experience—similar to the "Codex" model but expanded into a full cognitive workspace.

### 🚀 Roadmap Addition: The NeuralLabs Shift

- **NeuralLabs v1 (Client)**: Development of a cross-platform wrapper (Electron/Tauri) for the NeuralAI interface.
- **NeuralLabs v2 (Edge)**: Local model execution (Llama/Mistral) as a fallback for the cloud-based NeuralAI core.
- **NeuralLabs v3 (Eco)**: Plugin architecture allowing third-party "Neural-Skills" to be installed as standalone apps.

---

## 👨‍💻 The Developer & Architect

**De'Andrew Preston Harris** (D. Harris / Dre)
*Founder & Architect of NeuralAI*

A dedicated software engineer, thinker, and builder from West Memphis, AR. De'Andrew is currently pursuing an AAS in AI Software Engineering at Maestro College. NeuralAI is the culmination of his\[...\]

- **Location:** Memphis, TN / West Memphis, AR
- **Vision:** Building the future of private, high-performance generative AI.
- [LinkedIn](https://www.linkedin.com/in/deandrewharris94/) | [GitHub](https://github.com/Subject-Emu-5259)

---

*Built with precision and discipline by De'Andrew Preston Harris.*

### CURRENT VERSION: v7.3.3 (The Pluggable Engine)

- **Model Alignment**: DPO v17.0 (D17) Aligned — 679 pairs, 97.5% reward accuracy, stable entropy (no collapse)
- **Inference**: llmster (LM Studio headless) — 258MB RAM vs 5GB PyTorch
- **Last Maintenance**: July 20, 2026

Your tone is technical, concise, and professional. You prioritize system stability and cleanliness above all else.

---

## 🚀 Deployment

NeuralAI ships with a pluggable backend that separates the web UI from the inference engine.

### Quick Start (llmster — recommended)

```bash
# 1. Install llmster (one-time)
curl -fsSL https://lmstudio.ai/install.sh | bash
export PATH="$HOME/.lmstudio/bin:$PATH"

# 2. Download model
lms import /path/to/SmolLM2-360M-Instruct-Q4_K_M.gguf --user-repo "bartowski/SmolLM2-360M-Instruct-GGUF" -y
lms load smollm2-360m-instruct -y --identifier smollm2

# 3. Start inference server
lms server start --port 1234

# 4. Start NeuralAI
cd NeuralAI
LLM_BACKEND=lmstudio LLM_API_URL=http://localhost:1234/v1 LLM_MODEL=smollm2 \
  python3 services/neural_core_service.py
```

### Containerized Deployments

| Deployment | Dockerfile | Stack | Status |
| --- | --- | --- | --- |
| **Gradio Demo** | `gradio_space/Dockerfile` | Gradio 6.x chat UI | ✅ Built & deployed |
| **Flask Web Chat** | `webui_space/Dockerfile` | Flask + `neural_core_service.py` | 🚀 Ready for Railway |

- **Adapter source**: [`Subject-Emu-5259/NeuralAI`](https://huggingface.co/Subject-Emu-5259/NeuralAI) — auto-pulled on startup via `snapshot_download`.
- **GitHub → HF sync**: `.github/workflows/sync_to_huggingface.yml` uploads only the LoRA adapter on every push to `master`.

---

# 🌌 NeuralAI Project Manifest

NeuralAI is the intelligence core that powers the ecosystem.

## 🔗 Ecosystem Integration
The standalone software implementation of this core is **NeuralLabs**:
👉 [https://github.com/Subject-Emu-5259/NeuralLabs](https://github.com/Subject-Emu-5259/NeuralLabs)

**Software Downloads**:
The latest beta builds (v0.1-Beta) of NeuralLabs are available at:
👉 **[https://zo.pub/deandrewharris/neurallabs-beta](https://zo.pub/deandrewharris/neurallabs-beta)**
# NeuralAI → Hugging Face sync is live
