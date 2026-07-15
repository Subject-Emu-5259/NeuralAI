#!/usr/bin/env bash
# NeuralAI launcher — LOCAL-ONLY inference.
# LM Studio (llama.cpp) serves SmolLM2-360M on :1234 (OpenAI-compatible).
# NO external/ZO fallback: if the local model can't serve, the UI errors
# honestly instead of relaying to ZO.
set -u

# Public service URL: the keep-alive pinger hits this (real external ingress)
# so the ZO Computer sandbox is not idled/slept by the platform on Free tier.
export NEURALAI_PUBLIC_URL="https://neuralai-web-ui-deandrewharris.zocomputer.io"
export PATH="$HOME/.lmstudio/bin:$PATH"
LMS="$HOME/.lmstudio/bin/lms"
MODEL_KEY="smollm2-360m-instruct"
API_KEY="lm-studio"

echo "[launch] checking LM Studio on :1234"
if ! curl -s -o /dev/null --max-time 4 http://localhost:1234/v1/models 2>/dev/null; then
  echo "[launch] starting LM Studio server..."
  ( "$LMS" server start >/dev/null 2>&1 || true ) &
  for i in $(seq 1 15); do
    curl -s -o /dev/null --max-time 3 http://localhost:1234/v1/models 2>/dev/null && break
    sleep 2
  done
fi

echo "[launch] loading model $MODEL_KEY into LM Studio..."
"$LMS" load "$MODEL_KEY" --yes >/dev/null 2>&1 || true
for i in $(seq 1 30); do
  curl -s --max-time 3 http://localhost:1234/v1/models 2>/dev/null | grep -q "$MODEL_KEY" && break
  sleep 2
done

if curl -s --max-time 4 http://localhost:1234/v1/models 2>/dev/null | grep -q "$MODEL_KEY"; then
  echo "[launch] LOCAL backend READY -> openai_compatible (:1234)"
else
  echo "[launch] ERROR: local model not available on :1234 — UI will error (NO ZO fallback)"
fi

# Always local-only. No fallback to ZO.
export LLM_BACKEND="openai_compatible"
export LLM_API_URL="http://localhost:1234/v1"
export LLM_API_KEY="$API_KEY"
export LLM_MODEL="$MODEL_KEY"

echo "[launch] exec webui_service ($(date -u))"
exec python /home/workspace/Projects/NeuralAI/services/webui_service.py
