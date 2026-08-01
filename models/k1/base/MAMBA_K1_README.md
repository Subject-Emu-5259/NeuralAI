# 🐍 NeuralAI Mamba K1

**NeuralAI's first owned base model** — created by De'Andrew Harris & Gemini.

---

## Identity

| Field | Value |
|-------|-------|
| **Name** | Mamba K1 |
| **Version** | 1.0.0 |
| **Architecture** | Mamba-130M (SSM — Structured State Space Model) |
| **Base Model** | `state-spaces/mamba-130m-hf` |
| **Training** | SFT (Supervised Fine-Tuning) via LoRA |
| **Parameters** | 129,135,360 (~129M) |
| **Hidden Size** | 768 |
| **Layers** | 24 |
| **Vocab Size** | 50,280 |
| **Model Size** | ~493 MB (safetensors, FP32) |
| **LoRA Config** | r=8, alpha=16, targets: `in_proj`, `dt_proj`, `x_proj` |
| **Training Data** | 1,000 UltraChat samples |
| **Training Steps** | 50 (1 epoch) |
| **Final Loss** | 6.78 |
| **Framework** | TRL 1.9.2 / Transformers 5.13.1 / PyTorch 2.11.0 / PEFT 0.19.1 |
| **Created** | 2026-07-31 |
| **Creator** | De'Andrew Harris & Google Gemini |

---

## What Makes Mamba K1 Special

Unlike traditional transformer models (like SmolLM2 or LLaMA), Mamba uses **State Space Models (SSMs)** — a fundamentally different architecture that processes text sequentially with constant memory, not quadratic attention. This is NeuralAI's first foray into owning its architecture choices, not just fine-tuning existing transformer models.

K1 is a proof-of-concept: it proves NeuralAI can train, merge, and deploy a custom SSM-based language model from scratch.

---

## How to Load

```python
from transformers import MambaForCausalLM, AutoTokenizer

model = MambaForCausalLM.from_pretrained(
    "/home/workspace/Projects/NeuralAI/models/k1/base",
    torch_dtype="auto",
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/workspace/Projects/NeuralAI/models/k1/base"
)

messages = [{"role": "user", "content": "Hello!"}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Known Limitations

- **129M parameters** — very small model, limited reasoning capability
- **50 training steps** — minimal training, loss of 6.78
- **Mamba architecture** — requires `mamba-ssm` for fast inference (CPU fallback works but is slow)
- **Cannot be converted to GGUF** — llama.cpp doesn't support Mamba SSM architecture
- **Not production-ready** — proof-of-concept base model for NeuralAI's SSM research track

---

## Roadmap

- **K2**: Scale to 370M+ Mamba parameters with deeper training
- **K3**: Mamba-2 architecture with hybrid attention
- **Multi-tier**: Mamba for efficient on-device, transformers for cloud

---

*"First of its kind, first of its name." — NeuralAI Mamba Series*
