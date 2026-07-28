#!/usr/bin/env bash
# NeuralAI inference server launcher.
# Picks the active GGUF from config/active_model.json via scripts/model_manager.py.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-1234}"
API_KEY="${API_KEY:-lm-studio}"
N_CTX="${N_CTX:-2048}"
N_THREADS="${N_THREADS:-2}"

MODEL_PATH="$(python3 "$ROOT/scripts/model_manager.py" get-path)"
MODEL_ID="$(python3 "$ROOT/scripts/model_manager.py" get-id)"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: GGUF not found for active model ($MODEL_ID): $MODEL_PATH" >&2
    echo "Falling back to base SmolLM2-360M-Instruct" >&2
    MODEL_PATH="/root/.lmstudio/models/bartowski/SmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct-Q4_K_M.gguf"
fi

echo "[neuralai-lmstudio] active model=$MODEL_ID path=$MODEL_PATH"

exec python3 -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --model_alias "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --n_gpu_layers 0 \
  --n_ctx "$N_CTX" \
  --n_threads "$N_THREADS" \
  --n_threads_batch "$N_THREADS" \
  --api_key "$API_KEY" \
  --chat_format chatml \
  --verbose false
