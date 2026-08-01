# 🧠 NeuralAI — Model Card

**Last Updated: August 1, 2026**

---

## Model Family Overview

NeuralAI now spans two architectures across five registered models:

```
Transformer (Fine-Tuned)              SSM (Owned Base)
├── SmolLM2-360M DPO v17             ├── Mamba K1 · 130M
└── NeuralAI-Air-135M SFT v19        ├── Mamba K2 · 790M GGUF
                                     └── Mamba K3 · 790M SFT (training)
```

---

## Active Models

### 🧬 Mamba K1 — First Owned Base Model

| Property | Value |
|----------|-------|
| **Architecture** | Mamba SSM (State Space Model) |
| **Parameters** | 129 million |
| **Base** | `state-spaces/mamba-130m-hf` |
| **Training** | SFT 50 steps, 1K UltraChat, LoRA r=8 |
| **Loss** | 6.78 |
| **Inference** | ~19 tok/s CPU |
| **Format** | Merged safetensors (493MB) |
| **HF Repo** | [Subject-Emu-5259/NeuralAI-Mamba-K1](https://huggingface.co/Subject-Emu-5259/NeuralAI-Mamba-K1) |
| **Status** | ✅ Complete — Proof of Concept |

### 🧬 Mamba K2 — Scaled Owned Base

| Property | Value |
|----------|-------|
| **Architecture** | Mamba SSM |
| **Parameters** | 790 million |
| **Base** | `state-spaces/mamba-790m-hf` |
| **Hidden size** | 2048 |
| **Layers** | 48 |
| **Format** | Q4_K_M GGUF (460MB) |
| **Inference** | LM Studio / llama.cpp |
| **HF Repo** | [Subject-Emu-5259/NeuralAI-Mamba-K2](https://huggingface.co/Subject-Emu-5259/NeuralAI-Mamba-K2) |
| **Status** | ✅ Ready for local inference |

### 🔬 Mamba K3 — Full SFT Training

| Property | Value |
|----------|-------|
| **Architecture** | Mamba SSM |
| **Parameters** | 790 million |
| **Base** | `state-spaces/mamba-790m-hf` |
| **Training** | SFT 500–1000 steps, 10K+ UltraChat, LoRA r=32 |
| **Target Loss** | < 3.0 |
| **Output** | Merged model → Q4_K_M GGUF |
| **Status** | 🔄 In Training |

### 🧠 SmolLM2-360M DPO v17 — Production Transformer

| Property | Value |
|----------|-------|
| **Architecture** | Transformer decoder |
| **Parameters** | 360 million |
| **Base** | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| **Training** | DPO 679 pairs, 3 epochs, 129 steps |
| **Reward Accuracy** | 97.5% |
| **Inference** | llmster (258MB RAM) |
| **HF Repo** | [Subject-Emu-5259/NeuralAI](https://huggingface.co/Subject-Emu-5259/NeuralAI) |
| **Status** | ✅ Production |

### ✈️ NeuralAI-Air-135M SFT v19 — Lightweight Transformer

| Property | Value |
|----------|-------|
| **Architecture** | Custom decoder-only Transformer |
| **Parameters** | 133.72 million |
| **Base** | `Subject-Emu-5259/NeuralAI-Air-135M` |
| **Training** | SFT 320 steps |
| **Format** | Q4_K_M GGUF (269MB) |
| **HF Repo** | [Subject-Emu-5259/NeuralAI-Air-135M-SFT-v19](https://huggingface.co/Subject-Emu-5259/NeuralAI-Air-135M-SFT-v19) |
| **Status** | ✅ Production |

---

## Architecture Comparison

| Property | Transformer (SmolLM2/Air) | Mamba SSM (K1/K2/K3) |
|----------|--------------------------|----------------------|
| **Complexity** | \(O(n^2)\) attention | \(O(n)\) linear |
| **Long context** | Memory-hungry | Efficient |
| **Inference speed** | Slower at length | Fast at any length |
| **Ecosystem** | Mature (HF, llama.cpp) | Growing |
| **NeuralAI owns weights?** | LoRA adapter only | ✅ Full merged model |
| **Training maturity** | Proven (DPO pipelines) | Early (SFT proof-of-concept) |

---

## Why Mamba SSM?

Mamba models use **Selective State Space Models** instead of attention. This is a fundamentally different approach to sequence modeling:

- **Linear complexity**: Processes tokens in \(O(n)\) time vs Transformer's \(O(n^2)\)
- **No attention heads**: No quadratic memory blowup at long context
- **No positional embeddings**: Naturally handles variable-length sequences
- **Emerging ecosystem**: Fewer tools than Transformers, but rapidly growing

For NeuralAI, owning full Mamba model weights (not just LoRA adapters on someone else's base) represents a strategic milestone toward full model sovereignty.

---

## Intended Use

- Conversational AI assistant for the NeuralAI stack
- Local/private inference via LM Studio or llama.cpp
- Research into SSM-based language models
- Foundation for future DPO alignment and domain-specific fine-tuning

## Limitations

| Model | Key Limitations |
|-------|----------------|
| **Mamba K1** | Undertrained (50 steps). Loops, echoes, tangents. Not suitable for production. |
| **Mamba K2** | Untrained base model. Requires SFT (K3) for useful output. |
| **Mamba K3** | In training. Quality TBD. |
| **SmolLM2-360M DPO v17** | Small by modern standards. May lack depth on specialized topics. |
| **Air-135M SFT v19** | Very small (135M). Small training set (37 samples for initial SFT). Factual outputs need independent verification. |

---

## Developer

Built by **De'Andrew Preston Harris** ([LinkedIn](https://linkedin.com/in/deandrewharris94/) · [GitHub](https://github.com/Subject-Emu-5259)) with Google Gemini AI Studio/Colab collaboration.

From Memphis, Tennessee. Raised in West Memphis, Arkansas. AI Software Engineering at Maestro College.

---

## Framework Versions

- PEFT 0.19.0+
- Transformers 5.x
- TRL 1.9+
- PyTorch 2.x
- llmster 0.0.19 (inference)
- llama.cpp (GGUF inference)

## Citation

```bibtex
@software{neuralai2026,
  author       = {Harris, De'Andrew Preston},
  title        = {NeuralAI: The Generative AI Engine},
  year         = {2026},
  url          = {https://github.com/Subject-Emu-5259/NeuralAI},
  version      = {7.3.0},
  description  = {Multi-model AI engine with owned Mamba SSM base models and DPO-aligned Transformer fine-tunes}
}
```
