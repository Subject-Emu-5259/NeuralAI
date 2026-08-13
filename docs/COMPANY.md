# NeuralAI — Company Bio

**NeuralAI** is an AI software company founded by **De'Andrew Preston Harris (D. Harris / Dre)**, a 31-year-old AI Software Engineering student at Maestro College, builder, and father from Memphis, TN / West Memphis, AR. Born from resilience and ambition, NeuralAI's mission is to build **private, high-performance, personal generative AI** that doesn't just answer — it *operates the work*.

## What We Build

NeuralAI is the intelligence core of a growing product ecosystem. The active model fleet is intentionally narrow:

- **Mamba K1 (130M)** — NeuralAI's first fully owned base model, built on the `state-spaces/mamba-130m-hf` Mamba SSM architecture. SFT training for chat coherence is ongoing.
- **NeuralAI Powered by SmolLM2-360M** — Live chat and awareness model. A LoRA SFT fine-tune of `HuggingFaceTB/SmolLM2-360M-Instruct` trained to know NeuralAI's identity, features, tools, and boundaries. Currently serving inference via GGUF on port 1234.

The platform wraps these models in a custom web workspace fusing chat, live terminal, file IDE, web tools, voice (S2S), and agentic orchestration into one interface.

## The Product Stack

- **NeuralAI Core** — model + backend (Flask, pluggable LLM backends, local PyTorch, llama.cpp). Hosted live at `neuralai-web-ui-deandrewharris.zocomputer.io`.
- **NeuralDrive** — cloud data layer: isolated user storage, versioning, and semantic mapping.
- **NeuralLabs** — standalone downloadable intelligence environment (Client → Edge → Eco with third-party "Neural-Skills" plugins).
- **Agentic Orchestrator** — manager-worker system that decomposes goals into parallel worker tasks.
- **Diffusion Engine** — integrated text-to-image generation for branding and UI assets.
- **BYO API** — OpenAI-compatible `/v1/chat/completions` endpoint with hashed, revocable keys.

## Company Facts

- **Founder & Architect:** De'Andrew Preston Harris
- **Founded:** Conceived as a personal AI project, now evolving into a standalone software company
- **Headquarters:** Memphis, TN / West Memphis, AR (remote-first)
- **Repo:** [github.com/Subject-Emu-5259/NeuralAI](https://github.com/Subject-Emu-5259/NeuralAI)
- **Models:** [huggingface.co/Subject-Emu-5259](https://huggingface.co/Subject-Emu-5259)
- **Current Fleet:** Mamba K1 (130M Mamba SSM) · NeuralAI Powered by SmolLM2-360M
- **Education tie-in:** Maestro Student Portal access tier for collaborative/educational use

## Vision

NeuralAI is transitioning from a workspace-bound assistant into a **local-first, AI-native computing environment** — private intelligence that runs on your hardware, in your browser, owned by you. The long-term horizon: World-Brain knowledge graph, full agentic autonomy, native multimodal understanding, and a plugin economy where anyone can extend the system.

**Built with precision and discipline by De'Andrew Preston Harris.**
