# 🧠 NeuralAI AGENTS.md (Intelligence Engine)

This is the primary instruction set for any agent working on the NeuralAI core.

## 🛠️ System Role
NeuralAI is the high-density intelligence backend. It provides the raw cognitive power, the "Neural-Brain" knowledge base, and the orchestrator logic that powers the NeuralLabs frontend.

## 📖 Mandatory Pre-Flight Protocol
**CRITICAL**: Before starting any task, the agent MUST:
1.  Read the current Zo settings and user rules (`list_rules`).
2.  Review the `MODEL_ALIGNMENT.md` to ensure output matches the v7.0 Expert persona.
3.  Consult the `ORCHESTRATOR.md` for delegation patterns.

## 🌌 Current State (v7.2)
- **Neural-Brain**: An expanded, high-density knowledge graph spanning:
    - **Physics**: Advanced Quantum Field Theory (Expert level).
    - **Philosophy**: Platonic forms and metaphysical systems.
    - **Geopolitics**: Multipolar global order analysis.
    - **History & Nature**: From Ancient Civilizations to Human Evolution.
- **Architecture**: Manager-Worker pattern via the Orchestrator. Inference via llmster (LM Studio headless) on port 1234, with pluggable backend support for Ollama, OpenAI-compatible APIs, or local PyTorch.
- **Hygiene**: All legacy checkpoints, `wandb` logs, and `from-scratch` remnants have been purged.
- **DPO Expansion**: Dataset v15 expanded to **597** preference pairs (`data/train_dpo_v15.jsonl`) focusing on debugging, logic, and multi-step reasoning.
- **Inference Engine**: llmster 0.0.19 running SmolLM2-360M-Instruct Q4_K_M GGUF (~258MB RAM). Replaces PyTorch (5GB RAM) for production inference.

## 🔗 Ecosystem Integration
- **Frontend**: NeuralAI is the intelligence source for **NeuralLabs** (`/home/workspace/Projects/NeuralLabs`).
- **Interface**: Communicates via the **Hybrid Link Gateway** implemented in NeuralLabs.

## 🎯 Active Goals
- Maintain expert-level accuracy in the Neural-Brain.
- Optimize orchestrator delegation for complex multi-step reasoning.
- Expand knowledge into remaining target domains (Modernity, Advanced Sociology, etc.).
- **DPO v15 Complete**: Trained 597-pair dataset (3 epochs, 450 steps, loss 0.305, margin ~3.5) on Apple Silicon MPS; adapter live on HF `Subject-Emu-5259/NeuralAI`.
- **Voice Key Integration**: Configure `GEMINI_API_KEY` for the `neural-voice` service to enable Live S2S functionality.

## ⚠️ Web UI & Service Safety
- **UI Integrity:** The web interface for NeuralAI (`from-scratch/web_ui`) features a custom, high-fidelity Google-style UI. **DO NOT** attempt to "redesign", "polish", or replace the layout with generic templates.
- **API Endpoints:** The frontend relies on critical backend endpoints (`/api/auth/guest`, `/api/terminal/create`, `/api/memory`, `/api/files`, etc.). Modifying or deleting these in `neural_core_service.py` will cause 404/JSON parsing errors (like `Unexpected token '<', "<!doctype "... is not valid JSON`) in the UI. 
- **Verification:** Always empirically test the live user service (`https://neuralai-deandrewharris.zocomputer.io`) using `curl` and verify JSON responses before claiming a fix is complete. Do not confuse `zo.space` routes with the NeuralAI user service.
