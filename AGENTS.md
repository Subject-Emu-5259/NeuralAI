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
- **Hygiene**: `wandb` logs are gone. `from-scratch` is the LIVE UI (not a remnant) — see Web UI & Service Safety. A `checkpoints/` directory still exists at repo root and is NOT purged; treat it as stale model artifacts, do not assume it was cleaned.
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
- **Voice Key Status**: `GEMINI_API_KEY` IS present in the live `neuralai-web-ui` service env (and `neural_voice` falls back to ElevenLabs when Gemini is absent). The standalone `neural-voice` service (`services/neural_voice/neural_voice_service.py`) is NOT currently registered as a running Zo service — launch it separately if Live S2S is needed. Key presence ≠ service running.

## ⚠️ Web UI & Service Safety
- **UI Integrity:** The live web interface for NeuralAI lives in `from-scratch/web_ui` and features a custom, Google-style chat UI. **DO NOT** attempt to "redesign", "polish", or replace the layout with generic templates. The running Zo service is **`neuralai-web-ui`** (label `svc_1cHl6qlp4_g`, public at `https://neuralai-web-ui-deandrewharris.zocomputer.io`). Its entrypoint is `run_service.sh`, which boots `services/webui_service.py` (Flask, port 5000). The older `services/neural_core_service.py` exists but is NOT the deployed service — do not edit it expecting live changes.
- **Tool entry point:** Slash web commands in the chat input are intercepted client-side in `from-scratch/web_ui/static/js/main.js` (`runToolCommand` + the `sendMessage` interceptor) and POST to **`/api/tool`** in `services/webui_service.py` (line ~1367). That handler calls `ToolHandler.execute(tool, params)` in `tools/tool_handler.py`, whose `handlers` dict maps tool names (`web_search`, `web_fetcher`, `web_browser`, `research`, `image`, `speak`, `summarize`, `translate`, `news`, `youtube`) to methods. **All 10 slash commands are LIVE in `main.js`** (`/web`, `/fetch`, `/browse`, `/research`, `/img`, `/speak`, `/summarize`, `/translate`, `/news`, `/yt`). When adding a tool, register it in BOTH the `main.js` interceptor/branch AND `ToolHandler.execute()` handlers dict, or the command 404s.
- **API Endpoints:** The frontend relies on critical backend endpoints (`/api/auth/guest`, `/api/terminal/create`, `/api/memory`, `/api/files`, `/api/tool`, `/api/chat`, `/api/tools/*` in `tools_api.py`, etc.). Modifying or deleting these in `webui_service.py` will cause 404/JSON parsing errors (like `Unexpected token '<', "<!doctype "... is not valid JSON`) in the UI. 
- **Verification:** Always empirically test the live user service (`https://neuralai-web-ui-deandrewharris.zocomputer.io`) using `curl` and verify JSON responses before claiming a fix is complete. Do not confuse `zo.space` routes with the NeuralAI user service. Note: the stale URL `neuralai-deandrewharris.zocomputer.io` returns 404 and is NOT a live service.

## 🔌 Web Tool Chain (all 10 LIVE, verified 2026-07-16)
- `/web <query>` → `web_search` (top 5). **Backend: Bing scrape → Gemini grounding → OpenRouter `perplexity/sonar` (keyed fallback).**
- `/fetch <url>` → `web_fetcher` (single page text/markdown).
- `/browse <url>` → `web_browser` (session steps via local browser automation).
- `/research <topic>` → chain: `web_search` (top N) → `web_fetcher.extract_text` → `summarize.summarize_sources` → brief + sources.
- `/img <prompt>` → `image_generator` via **OpenRouter** `google/gemini-2.5-flash-image` (GEMINI_API_KEY is dead/401; do NOT use it for images).
- `/speak <text>` → `tts.py` → **gTTS** (Gemini TTS 401; gTTS fallback is the live path). Returns a local mp3 path.
- `/summarize <url|text>` → `summarize.summarize_sources`.
- `/translate <lang> <text>` → **OpenRouter** `meta-llama/llama-3.2-3b-instruct` (free).
- `/news <topic>` → `web_search` with `" news"` suffix.
- `/yt <url>` → YouTube metadata + `summarize_sources`.
- Inline link preview: `fmt()` in `main.js` rewrites `http(s)://` URLs in assistant output into clickable cards.

**Live key status (2026-07-16):** `Open_Router_API` = VALID (auth 200, used by `web_search` Sonar, `image_generator`, `translate`). `GEMINI_API_KEY` = PRESENT but 401 on Gemini API (do not rely on it for TTS/image). `ELEVENLABS_API_KEY` = present (neural_voice standby). `MINIMAX_AI_API_KEY` = present.
