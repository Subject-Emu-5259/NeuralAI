#!/usr/bin/env bash
# NeuralAI inference server launcher.
# Registers the custom "neuralai-intel" chat format and serves the active GGUF.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-1234}"
API_KEY="${API_KEY:-lm-studio}"
N_CTX="${N_CTX:-2048}"
N_THREADS="${N_THREADS:-2}"

MODEL_PATH="$(python3 "$ROOT/scripts/model_manager.py" get-path)"
MODEL_ID="$(python3 "$ROOT/scripts/model_manager.py" get-id)"
CHAT_FORMAT="$(python3 "$ROOT/scripts/model_manager.py" get-format)"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: GGUF not found for active model ($MODEL_ID): $MODEL_PATH" >&2
    echo "Set an active model with: python3 scripts/model_manager.py set <mamba-k1|mamba-k2>" >&2
    exit 1
fi

echo "[neuralai-lmstudio] active model=$MODEL_ID path=$MODEL_PATH format=$CHAT_FORMAT"

exec python3 -u "$ROOT/services/lmstudio_server.py" \
  --model "$MODEL_PATH" \
  --model_alias "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --n_gpu_layers 0 \
  --n_ctx "$N_CTX" \
  --n_threads "$N_THREADS" \
  --n_threads_batch "$N_THREADS" \
  --api_key "$API_KEY" \
  --chat_format "$CHAT_FORMAT" \
  --verbose false
