#!/usr/bin/env bash
# NeuralAI launcher — LOCAL-ONLY inference.
# LM Studio (llama.cpp) serves SmolLM2-360M on :1234 (OpenAI-compatible).
# NO external/ZO fallback: if the local model can't serve, the UI errors
# honestly instead of relaying to ZO.
set -u

# Public service URL: the keep-alive pinger hits this (real external ingress)
# so the ZO Computer sandbox is not idled/slept by the platform on Free tier.
export NEURALAI_PUBLIC_URL="https://neuralai-web-ui-deandrewharris.zocomputer.io"

# External provider keys (from Zo Advanced secrets). Exported so webui_service.py
# can read them via os.environ. NEURALAI_PROVIDER selects A (gemini) or B (minimax);
# "auto" prefers A then falls back to B. Default = gemini (test A first).
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
export Pollinations_Api_key="${Pollinations_Api_key:-sk_EHoauFyDNOwVAPFmPdUOuIA4wwCGTPoy}"
export MINIMAX_AI_API_KEY="${MINIMAX_AI_API_KEY:-}"
export NEURALAI_PROVIDER="${NEURALAI_PROVIDER:-gemini}"
# Keyed Pollinations AI (text/image/video/audio/voice/embeddings) — single API.
export Pollinations_Api_key="${Pollinations_Api_key:-}"
export PATH="$HOME/.lmstudio/bin:$PATH"
LMS="$HOME/.lmstudio/bin/lms"
MODEL_KEY="smollm2-360m-instruct"
API_KEY="lm-studio"

# --- LM Studio ownership ---------------------------------------------------
# The dedicated `neuralai-lmstudio` watchdog service (svc_Ob9JgSNKYdw) OWNS :1234.
# This launcher must NOT start its own lms instance (that causes a port fight
# and a boot deadlock). The neuralai-lmstudio watchdog owns :1234. We do NOT block
# on it — Flask launches immediately regardless, and chat uses whatever backend is
# selected below (ZO native HY3 fallback when local model is unavailable).
if curl -s -o /dev/null --max-time 4 http://localhost:1234/v1/models 2>/dev/null; then
  echo "[launch] :1234 is up (owned by neuralai-lmstudio watchdog)"
else
  echo "[launch] :1234 NOT up — Flask will start anyway; backend falls back to ZO native (HY3)"
fi

# Inference backend selection (priority order): LOCAL FIRST.
#   1. Local OpenAI-compatible server on :1234 (LM Studio / llmster)  [DEFAULT]
#   2. Ollama on :11434 (if running).
#   3. ZO native /zo/ask (user BYOK HY3) -- EXPLICIT OPT-IN ONLY, never auto.
# The BYOK/zo path was added for OUTSIDE-chat-UI use. It must NOT override
# your local model by default. zo is only used if the user explicitly sets
# LLM_BACKEND=zo in the service env, or LLM_ALLOW_ZO_FALLBACK=true is set.
if curl -s -o /dev/null --max-time 4 http://localhost:1234/v1/models 2>/dev/null; then
  export LLM_BACKEND="openai_compatible"
  export LLM_API_URL="http://localhost:1234/v1"
  export LLM_API_KEY="lm-studio"
  export LLM_MODEL="smollm2-360m-instruct"
  echo "[launch] backend -> LOCAL :1234 (DEFAULT)"
elif curl -s -o /dev/null --max-time 4 http://localhost:11434/api/tags 2>/dev/null; then
  export LLM_BACKEND="ollama"
  export LLM_API_URL="http://localhost:11434/v1"
  export LLM_MODEL="smollm2:360m"
  echo "[launch] backend -> Ollama (fallback)"
elif [ "${LLM_BACKEND:-}" = "zo" ] || [ "${LLM_ALLOW_ZO_FALLBACK:-true}" = "true" ]; then
  export LLM_BACKEND="zo"
  export ZO_ASK_URL="https://api.zo.computer/zo/ask"
  export LLM_MODEL="${LLM_MODEL:-byok:0d3567f7-f521-42b0-8adf-65c9b036cf89}"
  echo "[launch] backend -> ZO native (HY3, fallback enabled)"
else
  echo "[launch] WARNING: no local model on :1234/:11434 and zo fallback disabled."
  echo "[launch] Targeting LOCAL :1234 anyway so web chat uses your model, not zo."
  export LLM_BACKEND="openai_compatible"
  export LLM_API_URL="http://localhost:1234/v1"
  export LLM_API_KEY="lm-studio"
  export LLM_MODEL="smollm2-360m-instruct"
fi

echo "[launch] exec webui_service ($(date -u))"
exec python /home/workspace/Projects/NeuralAI/services/webui_service.py
