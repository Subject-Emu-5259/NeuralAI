# Incident: NeuralAI Service Pauses, "Acting Weird", and Garbage Replies

**Date:** 2026-07-14 → 2026-07-15
**Service:** `neuralai-web-ui` — `https://neuralai-web-ui-deandrewharris.zocomputer.io`
**Host:** ZO Computer, 4 GB RAM (Free plan can sleep; this incident is mostly separate from sleep)
**Code:** `file services/webui_service.py`

## TL;DR — What Actually Fixed It

The service was paused by the supervisor (`enabled:false` / `paused_at` set), not by OOM.
Re-enabling + restarting the service, plus the existing `/health` keep-alive ping, stopped the
pauses. The "acting weird" garbage came from a *brief experiment* where `LLM_BACKEND=zo` was set
with a short model name, which hit ZO's 402 free-allowance error and fell back into a
prompt-truncation bug. The correct, deployed backend is `local` — NeuralAI's own
SmolLM2-360M + NeuralAI LoRA (`checkpoints/v2_model`). That is the product identity; do NOT switch
it to ZO/HY3 unless you intentionally want HY3 answering instead of NeuralAI.

## Symptoms

1. **Service pauses** — web UI unreachable; supervisor had the service paused/killed.
2. **"Acting weird"** — reply to a simple "hey" was a hallucinated, cut-off sentence
   ("That's a great summary! ...while the input_format option is useful for making it easier for
   children to ask questions…"). No relation to the prompt.
3. **Short / incoherent replies** during the zo-backend experiment.

## Root Causes (verified)

### 1. Pauses = supervisor pause state, not OOM

`list_user_services` showed the service with `enabled:false` / `paused_at` populated. A 503 on the
liveness probe can also cause the host to pause the service. The `/health` keep-alive pinger
(`_keep_alive_pinger`, started at boot, `services/webui_service.py:1892`) pings `/health` every
5 min to prevent idle sleep. Re-enabling + restarting the service cleared the paused state.

> NOTE: An old comment in the code (`webui_service.py:48`) claims "PyTorch + SmolLM2-360M = \~6.2 GB
> → OOM kill loop." That is inaccurate. SmolLM2-360M-Instruct in float16 is \~720 MB and runs fine on
> the 4 GB host (verified by the live chat test below). The earlier `[WATCHDOG] Low reclaimable memory` warning was transient. Treat OOM as unlikely for this model; if RAM pressure appears, look
> at other processes first.

### 2. "Acting weird" = ZO 402 → silent fallback into a prompt-truncation bug

During troubleshooting, `LLM_BACKEND=zo` was set with a short model name (`gpt`, `gpt-4o-mini`).
ZO returned **402** `free_allowance_exhausted` (daily free allowance gone). The streaming path
caught that and **fell back to the local model** so "chat never breaks" — but the local path had a
bug:

- `build_prompt_with_context` built a **37,907-token** prompt from one oversized chat message.
- `_truncate_to_fit` then dropped it to **0 turns** (context wiped).
- `max_input = 768` token cap + 256-token generation → a context-less, incoherent &lt;80-token
  hallucination. That is the "summary/children" reply.

### 3. Short replies

Default generation is `max_tokens=256` (`generate_response` / `stream_response`), not 48/80.
The *garbage* was caused by the 0-turns truncation above, not by token caps.

## Fixes Applied (verified in `file services/webui_service.py`)

- `_cap_text(text, max_chars=3500)` added (`:430`) and used inside `build_prompt_with_context`
  (`:451-452`). One oversized paste can no longer explode the prompt to 30k+ tokens and wipe context
  to 0 turns.
- **Keep-alive pinger started at boot** (`:1892`) — prevents the host from pausing the service on idle.
- **Memory watchdog started at boot** (`:1893`) — runs GC on low reclaimable memory to avoid OOM kills.
- `/health` **never 503s** (liveness probe short-circuits before model load, `:872-890`) so the host
  does not pause the service for a slow first load.
- **No backend change to "fix" the pause.** The correct backend is `local` (NeuralAI's own model).
  The zo-backend experiment was reverted; the service runs `LLM_BACKEND=local`.

## Current Verified Working Config (from `list_user_services`)

```markdown
LLM_BACKEND = local        # NeuralAI's own SmolLM2-360M-Instruct + NeuralAI LoRA (checkpoints/v2_model)
LLM_MODEL   = (empty)      # empty → uses BASE_MODEL + MODEL_PATH defaults (the NeuralAI LoRA)
LLM_API_URL = (empty)
LLM_API_KEY = (empty)
GROQ_API_KEY = (empty)
```

Verified by live chat test: multi-turn replies correctly identify as "NeuralAI built on
SmolLM2-360M-Instruct with a custom NeuralAI LoRA adapter (SFT v16 + DPO v16)" — i.e. the local
model is answering, not HY3.

## How To Keep It Stable

- **Do NOT let the supervisor pause it.** If the UI goes down, run `update_user_service` with
  `enabled:"true"` and restart. The keep-alive ping handles normal idle sleep.
- **Keep** `LLM_BACKEND=local` unless you intentionally want HY3 to answer. Switching to `zo` changes
  the model's identity away from NeuralAI.
- **If you ever use** `zo` **backend**, only use the account's BYOK model ID
  `byok:0d3567f7-f521-42b0-8adf-65c9b036cf89`. Short names (`gpt`, `gpt-4o-mini`, `claude-*`) burn
  the free allowance and 402 → trigger the fallback path that produced the garbage.
- On chat errors, check `/dev/shm/neuralai-web-ui_err.log`. Errors now surface in the UI instead of
  being masked by a broken fallback reply.

## Verification

- `GET /health` → `{"status":"ok","llm_backend":"local","model_status":"ready (external backend)"}`
  (local model loads; status string is cosmetic).
- Live chat: "hey" → coherent greeting; follow-up "your name is NeuralAI" → correct self-identification
  with model architecture. Multi-turn context preserved. No truncation, no hallucination.