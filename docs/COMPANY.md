# NeuralAI — Company Bio

**NeuralAI** is an AI software company founded by **De'Andrew Preston Harris (D. Harris / Dre)**, a 31-year-old AI Software Engineering student at Maestro College, builder, and father from Memphis, TN / West Memphis, AR. Born from resilience and ambition, NeuralAI's mission is to build **private, high-performance, personal generative AI** that doesn't just answer — it *operates the work*.

## What We Build

NeuralAI is the intelligence core of a growing product ecosystem. The model fleet spans three tiers:

- **Mamba K1 (130M)** — NeuralAI's first fully owned base model, built from scratch with Gemini on the `state-spaces/mamba-130m-hf` Mamba SSM architecture. SFT-trained on 1K UltraChat samples, running at ~19 tok/s on CPU.
- **Mamba K2 (793M)** — Next-generation Mamba at Q4_K_M quantization (437MB GGUF), built on `state-spaces/mamba-790m-hf`. Planned 500–1000 SFT steps on 10K+ UltraChat samples.
- **NeuralAI v17 DPO (360M)** — Production DPO-aligned SmolLM2-360M with 679 preference pairs, 97.5% reward accuracy, live on Hugging Face.

The platform wraps these models in a custom web workspace fusing chat, live terminal, file IDE, web tools, voice (S2S), and agentic orchestration into one interface.

## The Product Stack
- **NeuralAI Core** — model + backend (Flask, pluggable LLM backends: llmster, Ollama, OpenAI-compatible, local PyTorch, ZO-native fallback). Hosted live at `neuralai-web-ui-deandrewharris.zocomputer.io`.
- **NeuralDrive** — cloud data layer: isolated user storage, versioning, and semantic mapping via WebDAV.
- **NeuralLabs** — standalone downloadable intelligence environment (Client → Edge → Eco with third-party "Neural-Skills" plugins). Beta builds ship via zo.pub.
- **Agentic Orchestrator** — manager-worker system that decomposes goals into parallel worker tasks, moving NeuralAI from assistant to operator.
- **Diffusion Engine** — integrated text-to-image generation for branding and UI assets.
- **BYO API** — OpenAI-compatible `/v1/chat/completions` endpoint with hashed, revocable keys so other hosts can run on NeuralAI.

## Company Facts
- **Founder & Architect:** De'Andrew Preston Harris
- **Founded:** Conceived as a personal AI project, now evolving into a standalone software company
- **Headquarters:** Memphis, TN / West Memphis, AR (remote-first)
- **Repo:** [github.com/Subject-Emu-5259/NeuralAI](https://github.com/Subject-Emu-5259/NeuralAI)
- **Models:** [huggingface.co/Subject-Emu-5259](https://huggingface.co/Subject-Emu-5259)
- **Current Fleet:** Mamba K1 (130M Mamba SSM) · Mamba K2 (793M GGUF) · NeuralAI v17 DPO (360M)
- **Education tie-in:** Maestro Student Portal access tier for collaborative/educational use

## Vision
NeuralAI is transitioning from a workspace-bound assistant into a **local-first, AI-native computing environment** — private intelligence that runs on your hardware, in your browser, owned by you. The long-term horizon: "World-Brain" knowledge graph, full agentic autonomy, native multimodal understanding, and a plugin economy where anyone can extend the system.

**Built with precision and discipline by De'Andrew Preston Harris.**
