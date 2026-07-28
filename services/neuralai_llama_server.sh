#!/usr/bin/env bash
set -u

# NeuralAI local inference server
# Uses the llama.cpp server bundled with LM Studio to serve the NeuralAI-Air-135M
# standalone GGUF on port 1234. The Air model is a standard LLaMA-architecture
# causal decoder (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings), so it loads
# directly without custom backend support.

export HOME=/root
export PATH="/root/.lmstudio/bin:/usr/local/bin:/usr/bin:/bin"

GGUF_DIR="/home/workspace/Projects/NeuralAI/models/NeuralAI-Air-135M-GGUF"
MODEL_AIR="${GGUF_DIR}/NeuralAI-Air-135M-SFT.F16.gguf"

if [[ ! -f "$MODEL_AIR" ]]; then
    echo "[FATAL] NeuralAI-Air SFT GGUF not found at $MODEL_AIR" >&2
    exit 1
fi

# Pick the llama.cpp backend binary. Use the plain CPU build if the AVX2 binary fails.
BACKEND_DIR="/root/.lmstudio/extensions/backends"
LLAMA_SERVER="${BACKEND_DIR}/llama.cpp-linux-x86_64-avx2-2.24.0/llama-server"
if ! "$LLAMA_SERVER" --version >/dev/null 2>&1; then
    LLAMA_SERVER="${BACKEND_DIR}/llama.cpp-linux-x86_64-2.24.0/llama-server"
fi

exec "$LLAMA_SERVER" \
    --model "$MODEL_AIR" \
    --port 1234 \
    --host 127.0.0.1 \
    --ctx-size 2048 \
    --threads 2 \
    --alias "NeuralAI-Air-135M-SFT" \
    --log-disable
