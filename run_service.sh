#!/usr/bin/env bash
# NeuralAI launcher — LOCAL-ONLY inference.
# Llama.cpp/LM Studio serves NeuralAI-Air-135M on :1234 (OpenAI-compatible).
# NO external/ZO fallback: if the local model can't serve, the UI errors
# honestly instead of relaying to ZO.
set -u

# Public service URL: the keep-alive pinger hits this (real external ingress)
# so the ZO Computer sandbox is not idled/slept by the platform on Free tier.
export NEURALAI_PUBLIC_URL="https://neuralai-web-ui-deandrewharris.zocomputer.io"

# Use the same API key the local llama.cpp/LM Studio server expects.
API_KEY="lm-studio"

# Stable default served from the neuralai-lmstudio watchdog.
MODEL_KEY="SmolLM2-360M-Instruct"

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

# --- LM Studio ownership ---------------------------------------------------
# The dedicated `neuralai-lmstudio` watchdog service (svc_Ob9JgSNKYdw) OWNS :1234.
# This launcher must NOT start its own lms instance (that causes a port fight
# and a boot deadlock). The neuralai-lmstudio watchdog owns :1234. We do NOT block
# on it — Flask launches immediately regardless, and chat uses whatever backend is
# selected below (ZO native HY3 fallback when local model is unavailable).
if curl -s -H "Authorization: Bearer ${API_KEY}" -o /dev/null --max-time 4 http://localhost:1234/v1/models 2>/dev/null; then
  echo "[launch] :1234 is up (owned by neuralai-lmstudio watchdog)"
else
  echo "[launch] :1234 NOT up — Flask will start, but chat will fail until local model is loaded"
fi

# Inference backend selection: LOCAL ONLY.
#   1. Local OpenAI-compatible server on :1234 (LM Studio / llmster)  [DEFAULT]
#   2. Ollama on :11434 (if running).
# ZO fallback is REMOVED. This is a local-first NeuralAI instance.
LOCAL_MODEL_JSON=$(curl -s -H "Authorization: Bearer ${API_KEY}" --max-time 5 http://localhost:1234/v1/models 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); ms=d.get('data',[]); print(ms[0]['id']) if ms else print('')" 2>/dev/null || true)

if [ -n "$LOCAL_MODEL_JSON" ]; then
  export LLM_BACKEND="openai_compatible"
  export LLM_API_URL="http://localhost:1234/v1"
  export LLM_API_KEY="$API_KEY"
  export LLM_MODEL="$LOCAL_MODEL_JSON"
  export BASE_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
  echo "[launch] backend -> LOCAL :1234 ($LLM_MODEL)"
elif curl -s -o /dev/null --max-time 4 http://localhost:11434/api/tags 2>/dev/null; then
  export LLM_BACKEND="ollama"
  export LLM_API_URL="http://localhost:11434/v1"
  export LLM_MODEL="smollm2:360m"
  echo "[launch] backend -> Ollama (fallback)"
else
  echo "[launch] WARNING: no local model on :1234/:11434. Chat will fail until one is loaded."
  echo "[launch] Targeting LOCAL :1234 anyway so web chat is local-first."
  export LLM_BACKEND="openai_compatible"
  export LLM_API_URL="http://localhost:1234/v1"
  export LLM_API_KEY="$API_KEY"
  export LLM_MODEL="SmolLM2-360M-Instruct"
  export BASE_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct"
fi

echo "[launch] exec webui_service ($(date -u))"
exec python /home/workspace/Projects/NeuralAI/services/webui_service.py
