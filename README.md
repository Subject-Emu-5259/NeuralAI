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

## 🛠️ Tech Stack & Architecture (v7.1-alpha)

NeuralAI is built on a high-performance, containerized architecture that marries local inference with cloud-grade storage.

### Core Stack

- **Core Model**: `SmolLM2-360M-Instruct` (DPO v15.0 Aligned for logic, math, multi-step reasoning, and debugging)
- **Vocal Identity**: Andrew (Warm/Multilingual) - Optimized for Live Speech-to-Speech (S2S)
- **Backend Framework**: Python / Flask (Core Service)
- **Storage & Database**: SQLite3 (Metadata) + Nextcloud Hub via NeuralCloud WebDAV Client (NeuralDrive)
- **Inference Engine**: PyTorch (CPU/Edge Optimized)
- **Frontend UI**: Vanilla JS, HTML5, CSS3 with an advanced Dark Mode layout

### Core Architectural Pillars

1. **NeuralAI Core**: Handles chat state, direct model inference, terminal session proxying, and tool orchestration.
2. **NeuralDrive (Cloud Storage)**: The intelligent data layer for all projects, featuring isolated user storage, automatic versioning, and semantic mapping.
3. **Diffusion Engine**: An integrated generative diffusion layer for producing visual branding assets, UI mockups, and visual logic maps.
4. **Agentic Orchestrator**: A high-autonomy layer enabling NeuralAI to plan, reason, and execute multi-step workflows across the OS and web, moving beyond simple chat to active goal achievement.

---

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

### Latest Alignment Run: v15.0

- **Training samples**: 597 (expanded DPO preference pairs)
- **Epochs**: 3
- **Steps**: 450
- **Final training loss**: `0.305`
- **Reward margin**: improved from `~0.5` → `~3.5` (model strongly prefers chosen responses)
- **Hardware**: Apple Silicon MPS (MacBook Air M4)
- **Run duration**: `730.5s` (~12m 11s)
- **Completed**: `2026-07-11 20:00 UTC`
- **Adapter**: live on Hugging Face at [`Subject-Emu-5259/NeuralAI`](https://huggingface.co/Subject-Emu-5259/NeuralAI)

> The v15 dataset (`data/train_dpo_v15.jsonl`) was generated by expanding the template pools in `training/build_dataset_v15.py` from 302 → 597 unique preference pairs covering code correctness, logic, reasoning, debugging, and multi-step tasks.

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

### CURRENT VERSION: v7.1-alpha (The Agentic Operator)

- **Model Alignment**: DPO v15.0 Aligned (597 pairs, Logic, Debugging, Reasoning)
- **Last Maintenance**: July 11, 2026

Your tone is technical, concise, and professional. You prioritize system stability and cleanliness above all else.

---

## 🚀 Deployment

NeuralAI ships two containerized deployments, both pulling the LoRA adapter from Hugging Face at runtime:

| Deployment | Dockerfile | Stack | Status |
| --- | --- | --- | --- |
| **Gradio Demo** | `gradio_space/Dockerfile` | Gradio 6.x chat UI | ✅ Built & deployed (Metal builder, healthcheck `/`) |
| **Flask Web Chat** | `webui_space/Dockerfile` | Flask + `neural_core_service.py` | 🚀 Ready for Railway (`railway.json`) |

- **Adapter source**: [`Subject-Emu-5259/NeuralAI`](https://huggingface.co/Subject-Emu-5259/NeuralAI) — auto-pulled on startup via `snapshot_download`, so retraining + pushing updates the live model on next restart.
- **8-bit quantization** (`QUANTIZE=1`) keeps the 360M model under the 512 MB free-tier RAM limit.
- **GitHub → HF sync**: `.github/workflows/sync_to_huggingface.yml` uploads only the LoRA adapter (not the 1.5 GB repo) on every push to `master`.

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
