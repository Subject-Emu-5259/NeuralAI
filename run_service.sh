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

# --- Orphan guard -------------------------------------------------------------
# Only ONE lms instance may ever own :1234. Kill any lms/llama-server leftovers
# BEFORE touching the port, so a stale session can't grab it or leave orphans.
pkill -f "$HOME/.lmstudio/bin/lms" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
"$LMS" server stop 2>/dev/null || true
sleep 2

# Flock so two concurrent launches (deploy + manual) can't both start lms.
LOCK=/tmp/neuralai_lms.lock
(
  flock -n 9 || { echo "[launch] another lms launcher holds the lock, skipping"; exit 0; }
  echo "[launch] checking LM Studio on :1234"
  if ! curl -s -o /dev/null --max-time 4 http://localhost:1234/v1/models 2>/dev/null; then
    echo "[launch] starting LM Studio server..."
    ( "$LMS" server start >/dev/null 2>&1 || true ) &
    for i in $(seq 1 15); do
      curl -s -o /dev/null --max-time 3 http://localhost:1234/v1/models 2>/dev/null && break
      sleep 2
    done
  fi

  # Unload any other model first so only $MODEL_KEY occupies :1234 (kills orphans).
  echo "[launch] ensuring only $MODEL_KEY is loaded on :1234..."
  "$LMS" unload --all >/dev/null 2>&1 || true
  echo "[launch] loading model $MODEL_KEY into LM Studio..."
  "$LMS" load "$MODEL_KEY" --yes >/dev/null 2>&1 || true
  for i in $(seq 1 30); do
    curl -s --max-time 3 http://localhost:1234/v1/models 2>/dev/null | grep -q "$MODEL_KEY" && break
    sleep 2
  done
) 9>"$LOCK"

if curl -s --max-time 4 http://localhost:1234/v1/models 2>/dev/null | grep -q "$MODEL_KEY"; then
  echo "[launch] LOCAL backend READY -> openai_compatible (:1234)"
else
  echo "[launch] ERROR: local model not available on :1234 — UI will error (NO ZO fallback)"
fi

# Release the lock file descriptor scope (subshell above already closed it).

# Always local-only. No fallback to ZO.
export LLM_BACKEND="openai_compatible"
export LLM_API_URL="http://localhost:1234/v1"
export LLM_API_KEY="$API_KEY"
export LLM_MODEL="$MODEL_KEY"

echo "[launch] exec webui_service ($(date -u))"
exec python /home/workspace/Projects/NeuralAI/services/webui_service.py
