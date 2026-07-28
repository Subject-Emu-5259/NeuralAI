#!/usr/bin/env python3
"""NeuralAI local inference server using llama-cpp-python OpenAI-compatible server."""
import os
import sys
from pathlib import Path

GGUF = Path("/home/workspace/Projects/NeuralAI/models/NeuralAI-Air-135M-GGUF/NeuralAI-Air-135M-SFT.Q4_K_M.gguf")
if not GGUF.exists():
    print(f"[FATAL] GGUF not found: {GGUF}", file=sys.stderr)
    sys.exit(1)

os.environ.setdefault("HOST", "127.0.0.1")
os.environ.setdefault("PORT", "1234")

# Match previous llama-server defaults: n_ctx 2048, 2 threads, no verbose spam.
os.environ.setdefault("MODEL", str(GGUF))
os.environ.setdefault("N_CTX", "2048")
os.environ.setdefault("N_THREADS", "2")

# Ensure we use CPU; this container has no GPU.
os.environ.setdefault("N_GPU_LAYERS", "0")

from llama_cpp.server.__main__ import main

if __name__ == "__main__":
    sys.argv = ["llama-cpp-server"]
    main()
