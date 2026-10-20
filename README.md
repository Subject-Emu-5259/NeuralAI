# NEA1 — NeuralAI's From-Scratch Language Model

**Your AI. Built from zero. Owned outright.**

NEA1 is a ~100M-parameter causal language model designed, built, and trained entirely from scratch by **De'Andrew Preston Harris**, founder of NeuralAI. No pretrained weights, no fine-tune of someone else's base — a custom Llama-style architecture, a custom byte-level BPE tokenizer, and a training pipeline that runs on free-tier compute.

The point of NEA1 is simple: prove that a capable, owned language model does not require a datacenter or a frontier-lab budget.

## At a glance

| | |
| --- | --- |
| **Parameters** | 100,111,872 |
| **Architecture** | Llama-style decoder (RoPE + GQA + SwiGLU) |
| **Layers** | 12 |
| **Hidden size** | 768 |
| **Attention** | 12 query heads / 4 KV heads (grouped-query) |
| **Context** | 2,048 positions |
| **Tokenizer** | Custom byte-level BPE, 32,000 vocab (v2) |
| **Pretraining data** | 768M tokens (FineWeb) |
| **Training hardware** | Kaggle T4 ×2 |

## Evaluation

| Checkpoint | Result | Notes |
| --- | --- | --- |
| step-100000 (locked SFT base) | **PPL 47.59** | Fluency eval: 50 held-out docs, 512-token context, 25,550 tokens |
| step-91000 | PPL 47.80 | Same eval — step-100000 shows no regression |
| step-80000 (current release) | **Held-out PPL 54.60** | Native f32, 2,000-doc held-out set, 512-token context — full protocol in `MODEL_CARD.md` |

## Status

- ✅ Pretraining complete — step-100000 verified as the base (no third pretrain stage)
- 🔄 Supervised fine-tuning (SFT) in progress — instruction, tool-use, and identity data; gated on a golden-40 evaluation (≥60%) plus a held-out perplexity guardrail
- ⚡ Serving — the v7 step-80000 checkpoint answers live through a FastAPI inference endpoint

## Repository layout

- `model/` — the NEA1 model definition and configuration (single source of truth for the architecture)
- `tokenizer/` — the v2 byte-level BPE tokenizer (vocab + merges)
- `training/` — data generation and training/smoke-test scripts for the pretraining and SFT pipelines
- `eval/` — fluency/perplexity evaluation
- `serving/` — the tool-use protocol used by the inference service
- `MODEL_CARD.md` — full model card: architecture, training, evaluation, intended use, limitations

## Why NEA1 exists

NeuralAI was founded by **De'Andrew Preston Harris** — AI Software Engineering student at Maestro College, builder, and father from Memphis, TN / West Memphis, AR. The mission is local-first, owned AI: train and merge your own weights instead of renting someone else's API forever.

- **Ownership** — own the weights, own the model, own the name
- **Privacy** — inference that can run on your hardware
- **Discipline** — small models, tight alignment, fast iteration

## License

Apache-2.0

---

**NeuralAI — built different.**
