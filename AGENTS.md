# 🧠 NeuralAI AGENTS.md (Intelligence Engine)

This is the primary instruction set for any agent working on the NeuralAI core.

## 🛠️ System Role
NeuralAI is the high-density intelligence backend. It provides the raw cognitive power, the "Neural-Brain" knowledge base, and the orchestrator logic that powers the NeuralLabs frontend.

## 📖 Mandatory Pre-Flight Protocol
**CRITICAL**: Before starting any task, the agent MUST:
1.  Read the current Zo settings and user rules (`list_rules`).
2.  Review the `docs/MODEL_ALIGNMENT.md` to ensure output matches the v7.0 Expert persona.
3.  Consult the `docs/ORCHESTRATOR.md` for delegation patterns.

## 🌌 Current State (v18 / Strategic Split)
- **Neural-Brain**: An expanded, high-density knowledge graph spanning:
    - **Physics**: Advanced Quantum Field Theory (Expert level).
    - **Philosophy**: Platonic forms and metaphysical systems.
    - **Geopolitics**: Multipolar global order analysis.
    - **History & Nature**: From Ancient Civilizations to Human Evolution.
- **Architecture**: Transitioning from SmolLM2-360M to the multi-tiered 2B/3B architecture.
- **Hygiene**: `wandb` logs are gone. `from-scratch` is the LIVE UI (not a remnant) — see Web UI & Service Safety. A `checkpoints/` directory still exists at repo root and is NOT purged; treat it as stale model artifacts, do not assume it was cleaned.
- **DPO Iterations (current = v17 / D17, COMPLETE 2026-07-20)**: `data/train_dpo_v16_combined.jsonl` = **679** preference pairs (597 v15 + 64 v16 + 18 supplement), focusing on debugging, logic, and multi-step reasoning. D17 = DPO continuation of the v16 (v2_model) adapter on SmolLM2-360M; 3 epochs / 129 steps / ~31 min; reward accuracy 97.5%, stable entropy, no eval set. Adapter at `checkpoints/v17-dpo` and HF `Subject-Emu-5259/NeuralAI`. v15 base file `data/train_dpo_v15.jsonl` (597) still present.
- **Inference Engine**: llmster 0.0.19 running SmolLM2-360M-Instruct Q4_K_M GGUF (~258MB RAM). Replaces PyTorch (5GB RAM) for production inference.
- **Strategic Transition (2026-07-28)**: Retiring the 135M (NeuralAI-Air) architecture. Moving to the **NeuralAI-2B-Speedster** and **NeuralAI-3B-Core-Intelligence** tiers to achieve functional coherence and advanced agentic capabilities.

## 🔗 Ecosystem Integration
- **Frontend**: NeuralAI is the intelligence source for **NeuralLabs** (`/home/workspace/Projects/NeuralLabs`).
- **Interface**: Communicates via the **Hybrid Link Gateway** implemented in NeuralLabs.

## 🎯 Active Goals
- Maintain expert-level accuracy in the Neural-Brain.
- Optimize orchestrator delegation for complex multi-step reasoning.
- Expand knowledge into remaining target domains (Modernity, Advanced Sociology, etc.).
  - **The Strategic Split**: Execute the transition to 2B and 3B architectures.
  - **Model Deployment**: Fine-tune and optimize **NeuralAI-2B-Speedster** and **NeuralAI-3B-Core-Intelligence**.
  - **DPO d17**: pushed to HF `Subject-Emu-5259/NeuralAI` (replaced v16 `v2_model` slot). Adapter at `checkpoints/v17-dpo` (`checkpoint-129`, r=32 LoRA / alpha 64, PEFT 0.19.1, base `HuggingFaceTB/SmolLM2-360M-Instruct`).
  - **Alignment**: Stabilize weights following Direct Weight Interventions for newline/system token biases.

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
- **NL→Tool Router (added 2026-07-16):** Plain-English web requests are intercepted BEFORE they reach the 360M model, so the user no longer needs to remember slash syntax. Implementation:
  - `tools/web_intent.py` — `classify(text)` returns `{"tool": <name>, "params": <str>, "confidence": <0-1>, "reason": <str>}` for intents `web_search`, `web_fetcher`, `web_browser`, `research`, `image`, `tts`, `summarize`, `translate`, `news`, `youtube`, else `tool=None` (pass to model). Keyword/URL/regex based, no network, no deps. `TEST_SUITE` in the same file covers 12 cases (all passing).
  - `services/webui_service.py` `handle_chat()` imports `classify` and, when `tool` is set, routes to `ToolHandler.execute(tool, params)` and returns the tool result as the assistant message (no raw 360M generation). Slash commands (`/web`, etc.) are still handled client-side in `main.js` and remain live.
  - **Refined output (added 2026-07-16):** Raw tool text (e.g. `web_search`/`web_fetcher` boilerplate, `research` concatenations) is passed through `tools/refine.py` → `refine_text(raw, kind)` which strips noise/banners, dedupes, and tightens into clean prose before display. Import guarded so a missing module degrades to raw text rather than 500ing.
  - **Verification note:** After editing `web_intent.py`/`refine.py`, restart the `neuralai-web-ui` service (`svc_1cHl6qlp4_g`) so the running process picks up the file; the service boots from `run_service.sh` → `services/webui_service.py`, not from `neural_core_service.py`.

****2026-07-17 Fix Log (verified live on neuralai-web-ui):**
- `tools/web_search.py` Google News resolver fixed: `/articles/` interstitial now extracts real publisher URL via `URL=` param (previously returned 300-char tracking blobs). News output renders as clean `🔗 [Source · Read on Google News]` cards.
- `tool_router.py` switched from dead `llama-3.2-3b-instruct` (402) to 3 free models; NL→tool routing confirmed live via `/api/chat`.
- `translate` tool fixed: was calling 402ing `meta-llama/llama-3.2-3b-instruct`; now uses `google/gemini-2.5-flash` via OpenRouter (or Gemini key). Verified: 'hello'→Spanish works.
- `image` generation confirmed real via Pollinations `flux` + OpenRouter `google/gemini-2.5-flash-image` fallback.
- `/api/chat` error-log line repaired (undefined `e` 500'd the 360M fallback); now logs correct exception.

Live key status (2026-07-16):** `Open_Router_API` = VALID (auth 200, used by `web_search` Sonar, `image_generator`, `translate`). `GEMINI_API_KEY` = PRESENT but 401 on Gemini API (do not rely on it for TTS/image). `ELEVENLABS_API_KEY` = present (neural_voice standby). `MINIMAX_AI_API_KEY` = present.

- **Agentic Tool Use — DONE (2026-07-17):** `tools/tool_router.py::route()` is live in `handle_chat()` (`generate_unified`). LLM-based routing via OpenRouter free models [`meta-llama/llama-3.1-8b-instruct`, `nousresearch/hermes-3-llama-3.1-8b`, `google/gemma-2-9b-it`] with keyword fallback. Composite intents route to the `agent` tool. NL news/search/web/image now hit real tools instead of the 360M model. See `docs/PLAN_AGENTIC_TOOL_USE.md` (now implemented).

- **Embedded NeuralBrowser — SHIPPED (2026-07-17), UI modernized (2026-07-17):** `Browser` tab in live UI (`from-scratch/web_ui`) is a modern browser shell: tab strip (multi-tab, per-tab URL/screenshot/title, close + new tab), omni search bar (plain text → Google search; URL-like input navigates directly), bookmarks bar (persisted in `localStorage` under key `neuralBrowserBookmarks`, add/delete), and zoom controls (+/-, 50–200%). Both modes live: (a) **AI Mirror** — describe a task, backend streams Playwright steps + screenshots via `/api/browser/run` SSE so the user watches the AI drive and can Take Over; (b) **User Browser** — manual navigate/click/scroll/fill via `/api/browser/navigate` + `/api/browser/action`. Backend endpoints (all `token_required`): `/api/browser/health`, `/api/browser/navigate`, `/api/browser/action`, `/api/browser/run` (SSE), `/api/browser/close`. `tools/web_browser.py` hardened for gVisor: Chromium `--no-sandbox` + `--disable-dev-shm-usage`; ops via `sess._run`. Controllers: `from-scratch/web_ui/static/js/browser.js` (preserves every `/api/browser/*` call; overrides `browserGo`/`browserAction` to sync active tab state). Static edits (html/css/js) go live WITHOUT a service restart — the `neuralai-web-ui` service serves files at request time. Only restart if you edit `services/webui_service.py`.- **Embedded NeuralBrowser — REMOVED (2026-07-19):** The old `Browser` tab (tab strip, omni bar, bookmarks, zoom, AI Mirror + User Browser modes) was REMOVED from the live UI at the user's request. Removed pieces: the `🌐 Browser` nav button, the `#tab-browser` block in `templates/index.html`, and the `<script src="/static/js/browser.js">` tag. `main_v2.js` had NO references to browser-tab functions, so removal did not break any other JS. The backend `/api/browser/*` endpoints in `services/webui_service.py` and `tools/web_browser.py` are still present but now orphaned (no UI calls them). DO NOT re-add the Browser tab unless the user explicitly asks. Also note the prior regression cause: commit `22786394` dropped the `main_v2.js` + `browser.js` script tags, which made the whole UI non-clickable; the fix restored the `main.css` link and the single `main_v2.js` script. Keep exactly one app script (`main_v2.js`) and the `main.css` link; do not reintroduce `browser.js`.

**REGRESSION GUARD:** The Google News resolver (`/articles/` interstitial → real publisher URL) and `refine.py` news/web_search list rendering are LIVE in `/api/tool` via `_REFINE_KINDS`. Do not remove the `refine_text` call in `tool_handler.execute()` or links revert to tracking blobs.
**STATIC-EDIT CACHE GUARD (added 2026-07-20):** `BUILD_VERSION` in `services/webui_service.py` now derives from the `main_v2.js` file mtime, so every static edit (html/css/js under `from-scratch/web_ui`) forces a fresh browser fetch via the `?v=` query — no service restart needed for static changes. Before this fix, `BUILD_VERSION` was frozen at process start, so edited JS stayed cached and button-handler fixes (e.g. the `What's New` / `openReleaseNotes` global export) never reached the browser despite being correct on disk. If a static-button fix "doesn't take," first check that the served `?v=` in `index.html` matches `stat -c %Y static/js/main_v2.js`. Restart the service only when you edit `services/webui_service.py` itself.