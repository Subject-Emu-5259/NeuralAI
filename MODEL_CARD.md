# NeuralAI Model Card

## Overview

NeuralAI is a multi-model AI platform with 5 models spanning transformer fine-tunes and owned Mamba SSM base models. All models are developed by De'Andrew Preston Harris and deployed via Hugging Face under `Subject-Emu-5259`.

## Active Models

### Fine-Tuned (Transformer)

| Model | HF Repo | Params | Training | Status |
|-------|---------|--------|----------|--------|
| **SmolLM2-360M DPO v17** | `Subject-Emu-5259/NeuralAI` | 360M | DPO 679 pairs | ✅ Production |
| **NeuralAI-Air-135M SFT v19** | `Subject-Emu-5259/NeuralAI-Air-135M-SFT-v19` | 135M | SFT 320 steps | ✅ Production |

### Owned Base (Mamba SSM)

| Model | HF Repo | Params | Training | Status |
|-------|---------|--------|----------|--------|
| **Mamba K1** | `Subject-Emu-5259/NeuralAI-Mamba-K1` | 130M | SFT 50 steps | ✅ Complete |
| **Mamba K2** | `Subject-Emu-5259/NeuralAI-Mamba-K2` | 793M | Q4_K_M GGUF | ✅ Complete |
| **Mamba K3** | Training pipeline | 790M | SFT 500-1000 steps | 🔄 In Training |

## Interaction Modes

### Production (SmolLM2-360M + Air-135M)
- **Chat**: Web UI at `neuralai-web-ui-deandrewharris.zocomputer.io`
- **API**: OpenAI-compatible via `/v1/chat/completions`
- **Inference**: llmster + llama.cpp, ~258MB RAM

### Mamba K1
- **Local**: `transformers` + `MambaForCausalLM`, 493MB safetensors
- **GGUF**: Convert and quantize for llama.cpp inference
- **Performance**: ~19 tok/s on CPU

### Mamba K2
- **LM Studio**: Import GGUF directly
- **llama.cpp**: Standard Q4_K_M inference
- **Size**: 460MB, fits 8GB RAM

### Mamba K3 (In Training)
- **Training**: LoRA rank 16 on SSM projection layers
- **Data**: 10K+ UltraChat samples
- **Hardware**: Google Colab T4/L4, ~7-12 hours
- **Output**: GGUF Q4_K_M for LM Studio

## Benchmarks

| Model | Perplexity | Speed | Memory |
|-------|-----------|-------|--------|
| Mamba K1 | High (undertrained) | 19 tok/s | 493MB (fp32) |
| Mamba K2 | N/A (base only) | TBD | 460MB (Q4_K_M) |
| Mamba K3 | Target: <15 | Target: 30+ tok/s | ~250MB (Q4_K_M) |
| SmolLM2-360M | ~12 (DPO v17) | ~25 tok/s | 258MB |

## Architecture Notes

- **Mamba SSM**: Linear O(n) complexity vs transformer O(n²) — better for long context
- **LoRA**: All fine-tuning uses LoRA adapters merged into base weights
- **GGUF**: K2 ships as GGUF for zero-setup local inference
- **Ownership**: Mamba K1/K2/K3 are NeuralAI's fully owned base models — no adapter-only limitations
