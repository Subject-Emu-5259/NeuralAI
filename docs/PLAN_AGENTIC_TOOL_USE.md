# Plan: Agentic Tool Use (Model Calls Its Own Tools)

**Date:** 2026-07-16
**Status:** Planned — NOT implemented
**Owner:** De'Andrew P Harris
**Related:** `AGENTS.md` §Web Tool Chain, §NL→Tool Router

---

## Problem Statement

The user expects NeuralAI to answer requests like *"search the web for the latest news"*
by actually calling a web tool. Today it does not reliably:

1. **Plain-English routing is brittle.** `tools/web_intent.py::detect_web_intent()` is
   keyword/regex based. Phrasings it did not anticipate (e.g. the example above) return
   `None`, so `handle_chat()` (`webui_service.py:1042`) falls through to the 360M model
   with **no web access**. The model then hallucinates a news list — exactly the failure
   the user reported.
2. **When a tool does fire, output is ugly.** Raw tool text (boilerplate, dedup gaps,
   banner noise) reaches the chat UI. `tools/refine.py::refine_text()` exists but the
   result is still a flat wall of text, not clean structured markdown (headings, bullets,
   source links grouped).

Goal: make the model *decide* tool use via an LLM router, with graceful fallback, and
render tool results as neat structured markdown.

---

## Root Cause

- The pre-model interceptor (`detect_web_intent`) is a static matcher, not a reasoning
  step. Any phrasing outside its keyword set bypasses tools entirely.
- `refine_text()` tightens prose but does not restructure into scannable sections/sources.

---

## Proposed Design

### 1. New LLM-based router — `tools/tool_router.py`
- `async def route(prompt: str) -> dict` returning
  `{"tool": <name|None>, "params": <str>, "confidence": <0-1>, "reason": <str>}`.
- Backed by OpenRouter `meta-llama/llama-3.2-3b-instruct` (free, already used by
  `/translate`). System prompt: *"You are a tool router for NeuralAI. Given a user
  message, decide if it needs one of these tools: web_search, web_fetcher, web_browser,
  research, image, tts, summarize, translate, news, youtube. Return JSON only."*
- **Fallback:** on API error / bad JSON, call `detect_web_intent(prompt)` so the existing
  keyword path still works. Never let a router failure silently drop to the 360M model
  without a tool when one was warranted.
- Keep the 10 tool names identical to `ToolHandler.execute()` handlers dict (no new
  registrations needed).

### 2. Wire into `handle_chat()` (`webui_service.py:~1042`)
- Replace `detect_web_intent(prompt)` call with `await route(prompt)`.
- When `tool` is set and confidence ≥ threshold (e.g. 0.5), route to
  `ToolHandler.execute(tool, params)` and return the refined result — same path as today.
- When `tool` is `None`, pass to the 360M model as before.

### 3. Neater output — extend `tools/refine.py`
- Add a `kind="news"` / `kind="web_search"` branch that structures results into:
  - A short 1–2 sentence lead
  - Grouped bullets by theme (World, Tech, Sports, Health…)
  - A **Sources** block with numbered links (already present in raw output)
- Preserve the import-guard so a missing module degrades to raw text, not a 500.

### 4. Verification (before claiming done)
- Restart `neuralai-web-ui` service (`svc_1cHl6qlp4_g`) after edits.
- `curl` the live endpoint `https://neuralai-web-ui-deandrewharris.zocomputer.io/api/chat`
  with `"search the web for the latest news"` → confirm a real tool (`web_search`/`news`)
  fired and the response is structured markdown, not a hallucinated list.
- Confirm slash commands (`/web`, `/news`, …) still work (regression check).
- Confirm non-tool chitchat ("hello") still reaches the 360M model.

---

## Out of Scope (this pass)
- Multi-step agentic loops (tool → observe → call again). This plan is single-shot routing.
- Adding new tools beyond the existing 10.
- Fixing LM Studio `:1234` auto-start persistence (separate OPEN DECISION in AGENTS.md).

## Files Touched
- `tools/tool_router.py` (new)
- `services/webui_service.py` (`handle_chat`, ~line 1042)
- `tools/refine.py` (`refine_text` structuring branch)
- `AGENTS.md` (cross-reference, done)
