#!/usr/bin/env bash
# NeuralAI LM Studio / llama.cpp server launcher for NeuralAI-Air-135M-SFT.
# Serves the local FP16 GGUF on :1234 with an OpenAI-compatible API.
set -u
MODEL_GGUF="/home/workspace/Projects/NeuralAI/models/NeuralAI-Air-135M-SFT-v3.gguf"
PORT="1234"
API_KEY="lm-studio"
N_CTX="2048"
N_THREADS="2"

export PATH="$HOME/.lmstudio/bin:$PATH"

# Prefer LM Studio CLI if the GGUF has been imported; otherwise fall back to
# llama-cpp-python server, which is more reliable inside the ZO 4 GB sandbox.

# Option 1: Try LM Studio server + load
# lms server start --port "$PORT" --bind 127.0.0.1 --cors --yes >/dev/null 2>&1 &
# sleep 3
# ~/.lmstudio/bin/lms load --yes --identifier NeuralAI-Air-135M-SFT --gpu off --context-length "$N_CTX" neuralai-air-135m-sft 2>/dev/null

# Option 2 (current): llama-cpp-python server — lightweight and stable.
exec python3 -m llama_cpp.server \
  --model "$MODEL_GGUF" \
  --model_alias "NeuralAI-Air-135M-SFT" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --n_gpu_layers 0 \
  --n_ctx "$N_CTX" \
  --n_threads "$N_THREADS" \
  --n_threads_batch "$N_THREADS" \
  --api_key "$API_KEY" \
  --chat_format chatml \
  --verbose false
