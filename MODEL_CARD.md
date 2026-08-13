# 🧠 NeuralAI — Model Card

**Last Updated:** August 13, 2026

NeuralAI currently maintains a focused, two-model fleet. All legacy models (Air-135M, Mamba K2/K3, older DPO checkpoints) have been retired from active development and removed from the repository.

---

## Model Family Overview

| Model | Architecture | Parameters | Role | Status |
|-------|-------------|------------|------|--------|
| **Mamba K1** | Mamba SSM | 130M | NeuralAI's first owned base model | 🔬 R&D / chat training |
| **NeuralAI Powered by SmolLM2‑360M** | Transformer + LoRA | 360M base | Live chat backend, awareness tuned | ⚡ Active inference |

---

## 🧬 Mamba K1

| Property | Value |
|----------|-------|
| **Architecture** | Mamba SSM (`MambaForCausalLM`) |
| **Parameters** | ~130M |
| **Hidden size** | 768 |
| **Layers** | 24 |
| **State size** | 16 |
| **Vocabulary** | 50,280 |
| **Base** | `state-spaces/mamba-130m-hf` |
| **Training** | LoRA SFT with vocabulary-safe chat format; iterative v2/v3 GGUF merges |
| **Formats** | Merged safetensors · Q4_K_M GGUF · F16 GGUF |
| **HF Repo** | [Subject-Emu-5259/NeuralAI-Mamba-K1](https://huggingface.co/Subject-Emu-5259/NeuralAI-Mamba-K1) |

### What Mamba K1 learned
- Vocabulary-safe instruction following using the NeuralAI "intel" chat format
- Assistant-style chat behavior (greetings, structured answers, refusals)
- NeuralAI identity and creator anchoring
- Step-by-step reasoning for coding, math, and writing prompts

### Intended use
- Research into owned SSM-based language models
- Local/private inference via Hugging Face Transformers or llama.cpp
- Future base for domain-specific fine-tuning and DPO alignment

### Limitations
- Small scale (130M); not production-grade yet
- Chat coherence training is still in progress
- May loop, echo, or drift on long contexts

---

## 🧠 NeuralAI Powered by SmolLM2‑360M

| Property | Value |
|----------|-------|
| **Architecture** | Transformer decoder (`SmolLM2ForCausalLM`) |
| **Base model** | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| **Parameters** | 360M (base) |
| **Fine-tune** | LoRA SFT |
| **LoRA rank/alpha** | v1: 8 / 16 | v2: 16 / 32 |
| **Dataset** | v1: 83 prompt/response pairs | v2: 506 prompt/response pairs |
| **Epochs** | 3 | 5 |
| **v1 final loss / steps** | 2.7186 / 30 | — |
| **v2 final loss / steps** | — | **0.1252 / 320** |
| **v2 runtime** | — | CPU-only, completed |
| **Active format** | — | `NeuralAI-Smol-Awareness-v2-Q8_0.gguf` (~369MB) |
| **Status** | Baseline | ⚡ Active inference backend (v2) |
| **HF Repo** | [Subject-Emu-5259/NeuralAI-Powered-By-SmolLM2360](https://huggingface.co/Subject-Emu-5259/NeuralAI-Powered-By-SmolLM2360) |

### Training data categories

| Category | Focus |
|----------|-------|
| **brand** | NeuralAI identity, creator (De'Andrew Preston Harris), mission, local-first AI values |
| **model** | NeuralAI's own Mamba K-family, current SmolLM2 chat backend |
| **site** | Web UI features — Model Manager, terminal, chat history, slash commands, settings |
| **tools** | `/web`, `/img`, `/speak`, `/summarize`, `/translate`, `/news`, `/yt` slash commands |
| **chat** | Multi-turn behavior, greetings, current-context recall |
| **assistant** | Capabilities, limitations, safety refusals, tool usage, AI identity |
| **companion** | Empathetic responses, emotional support, loneliness/sadness, boundaries |
| **refusal** | Harmful requests, consciousness denial, off-topic redirections |

### What it learned
- Correctly identifies itself as NeuralAI and names De'Andrew Preston Harris as creator on direct prompts
- Provides helpful, structured assistant responses
- Generates empathetic companion replies with appropriate human-support redirection
- Understands assistant limitations and refuses harmful or consciousness claims

### What it didn't fully learn (v1)
- Exact site URLs and slash-command syntax on all rephrasings
- Robust denial of consciousness across rephrased prompts
- Precise model-family naming (K1/K2/K3) when asked indirectly

These gaps are targeted by the v2 awareness dataset expansion.

### Intended use
- Live chat backend for the NeuralAI web UI
- Local/private inference via PEFT or GGUF
- Research into small-model awareness and identity tuning

### Limitations
- Small base model (360M); depth on specialized topics is limited
- Awareness retention is fragile across rephrasing without sufficient data
- Outputs should be verified for critical factual or medical decisions

---

## Architecture Comparison

| Property | Transformer (SmolLM2) | Mamba SSM (K1) |
|----------|------------------------|----------------|
| **Complexity** | \(O(n^2)\) attention | \(O(n)\) linear |
| **Long context** | Memory-hungry | Efficient |
| **Inference speed** | Slower at length | Fast at any length |
| **Ecosystem** | Mature (HF, llama.cpp) | Growing |
| **NeuralAI owns weights?** | LoRA adapter only | ✅ Full merged model |
| **Training maturity** | Proven (DPO + SFT pipelines) | Early R&D |

---

## Developer

Built by **De'Andrew Preston Harris** ([LinkedIn](https://linkedin.com/in/deandrewharris94/) · [GitHub](https://github.com/Subject-Emu-5259)).

From Memphis, Tennessee. Raised in West Memphis, Arkansas. AI Software Engineering at Maestro College.

---

## Citation

```bibtex
@software{neuralai2026,
  author       = {Harris, De'Andrew Preston},
  title        = {NeuralAI: The Generative AI Engine},
  year         = {2026},
  url          = {https://github.com/Subject-Emu-5259/NeuralAI},
  description  = {Local-first AI engine with owned Mamba SSM base models and awareness-tuned Transformer fine-tunes}
}
```

(The v2 training run is complete: 5 epochs, 320 steps, final training loss **0.1252**.)

### Training comparison

v2 dominated v1 on every metric except the base-loss floor, with the expanded dataset and higher LoRA rank producing a much better fit.

| Metric | v1 | v2 |
|--------|----|----|
| Dataset pairs | 83 | 506 |
| Epochs | 3 | 5 |
| Trainable params | 1.64M | 2.62M |
| LoRA r/α | 8 / 16 | 16 / 32 |
| Final training loss | 2.7186 | 0.1252 |
| GGUF active | `NeuralAI-Smol-Awareness-Q8_0.gguf` | `NeuralAI-Smol-Awareness-v2-Q8_0.gguf` |
