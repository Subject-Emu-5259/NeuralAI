---
license: apache-2.0
language:
  - en
tags:
  - neuralai
  - nae1
  - text-generation
  - causal-lm
  - llama-cpp
  - gguf
  - from-scratch
  - pretraining
  - local-ai
  - tiny-model
pipeline_tag: text-generation
---

<img src="https://huggingface.co/Subject-Emu-5259/NeuralAI-Nae1/resolve/main/neuralai-nae1-banner.png" alt="NeuralAI Nae1 banner" width="100%" />

<h1 align="center">NeuralAI — Nae1</h1>

<p align="center">
  <strong>Trained from scratch. Every weight owned.</strong><br/>
  A 100M-parameter Llama-style decoder pretrained end-to-end on FineWeb. No distillation, no fine-tune of someone else's base.
</p>

---

## Current release: step-80000 base (v2 lineage)

| Property | Value |
| --- | --- |
| **Architecture** | Llama-style decoder: RoPE, GQA, SwiGLU |
| **Parameters** | 100,111,872 (100.1M) |
| **Layers / hidden / heads** | 12 layers · 768 hidden · 12 heads · 4 KV heads (GQA) |
| **Vocabulary** | 32,000 tokens, ByteLevel BPE v2 (`tokenizer_v2/`) |
| **Context** | Trained at 2,048; evaluated at `n_ctx=512` |
| **Training** | 80,000 steps. Resumed 60,000 → 80,000 on Kaggle T4, peak LR 6e-4, cosine to a 10% floor |
| **Held-out perplexity** | **54.60** native f32 (2000 docs, 1,454,064 tokens, `n_ctx=512`) |
| **GGUF f32 / Q4_K_M** | **55.19 / 55.88** on the same 2000 docs (llama.cpp scoring, token counts differ slightly, see below) |
| **Weights in this repo** | `ckpt_v7_step80000_bf16/` (safetensors, serving copy) + `tokenizer_v2/`. GGUF Q4_K_M: `nae1-step80000-Q4_K_M.gguf` |
| **License** | Apache 2.0 |

> **Lineage note.** The v1 run (439-token vocab, 32k steps) is kept below for history. The v2 run rebuilt the tokenizer at 32k and continued training to step 80000. These are different lineages. Do not compare the v1 numbers (for example 54.16) with the v2 numbers. The v2 eval is on the 2000-doc held-out set (`evals/`).

---

## Training

**v2 continuation.** Training resumed at step 60,000 and ran to step 80,000. The chart shows the logged train loss and the in-training held-out PPL. The in-training PPL uses a chunked stream protocol and is **not** the 54.60 number. It is shown only to illustrate the trend.

<p align="center">
  <img src="https://huggingface.co/Subject-Emu-5259/NeuralAI-Nae1/resolve/main/neuralai-nae1-training.png" alt="Nae1 step 60000 to 80000 training loss and in-training held-out PPL" width="92%" />
</p>

**v1 lineage (frozen).** Eight cosine cycles from random initialization, 0 → 32,000 steps. with the tokenizer and objective described in the lineage note above.

---

## Evaluation

All v2 claims use one shared held-out corpus: `heldout_fineweb.jsonl` (2,000 docs from unseen crawl CC-MAIN-2014-10), `n_ctx=512`. Documents are scored independently in `n_ctx` windows.

<p align="center">
  <img src="https://huggingface.co/Subject-Emu-5259/NeuralAI-Nae1/resolve/main/neuralai-nae1-comparison.png" alt="Held-out perplexity lineage: bars on different eval protocols are not directly comparable" width="92%" />
</p>

| Model / format | Held-out PPL | Tokens | Eval report |
| --- | --- | --- | --- |
| step-80000, native f32 | **54.60** | 1,454,064 | `evals/eval_base80000_native_n512.json` |
| step-80000, GGUF f32 | 55.19 (+1.1%) | 1,454,478 | `evals/eval_stablelm_base80000_f32_full.json` |
| step-80000, GGUF Q4_K_M | **55.88** (+2.3%) | 1,454,478 | `evals/eval_base80000_stablelm_Q4_K_M_n512.json` |
| step-2000 SFT (sweep candidate, not released) | 59.43 (worse than base) | 1,454,064 | `evals/eval_sftv3b_lr1e-5_rp10_2000_native_n512.json` |

**Protocol caveats.**
- The native number is torch teacher-forced scoring. The GGUF numbers use llama.cpp `llm.eval` scoring. The two runs do not produce identical token counts (1,454,064 vs 1,454,478), so the +1.1% and +2.3% gaps mix the GGUF approximation with scoring differences.
- The GGUF export drops the LayerNorm bias tensors to fit llama.cpp's RMSNorm. This is a measured approximation, not an exact conversion. Logits do not match the native forward pass exactly.
- The probe "The quick brown fox" (3 tokens) scores 67,746 PPL. That is noise on a 3-token sequence. Held-out PPL is the metric.

**Withdrawn figures.** Earlier revisions of this card cited 52.26, 57.64 and 43.09 for step-80000 and step-60000. No saved eval report backs them, so they have been removed. The 43.09 figure came from the in-training chunked protocol and is not comparable to 54.60.

### Lineage (historical, different tokenizer)

| Checkpoint (v1 lineage, 439-token vocab) | Held-out PPL (Q4_K_M, 992 docs, n512) |
| --- | --- |
| step-32000 | 54.16 |

Earlier v1 steps (8k–28k) were scored on an older eval path and are not comparable.

The v1 cross-check on unseen CC-MAIN-2014-10 (300 docs) gave 57.09 at step-32000 vs 143.64 at step-28000. That confirmed the step-32000 drop was real generalization rather than leakage. It is a v1 number and does not apply to v2.

---

## SFT status

The SFT scaffolding is fixed: response-only loss masking in the smoke cell and a vocab guard that refuses the retired 439-token tokenizer. The step-2000 SFT run scored **59.43**, worse than the 54.60 base on the same protocol. **No SFT checkpoint is released.** The full sweep and selection have not been run.

---

## Quickstart

The GGUF is a llama.cpp model file. The perplexity numbers above were scored with the exact ByteLevel BPE v2 token IDs from `tokenizer_v2/`. For exact token IDs, use the tokenizer files in this repo.

```python
from llama_cpp import Llama

llm = Llama(model_path="nae1-step80000-Q4_K_M.gguf", n_ctx=512, verbose=False)
out = llm("The quick brown fox", max_tokens=32, temperature=0.0)
print(out["choices"][0]["text"])
```

This is a pretrained base model. It has no chat template and has not been instruction-tuned.

---

## Files

- `nae1-step80000-Q4_K_M.gguf`: current GGUF (Q4_K_M, 76.6 MiB)
- `ckpt_v7_step80000_bf16/`: serving safetensors checkpoint
- `tokenizer_v2/`: 32k ByteLevel BPE tokenizer used by step-80000
- `evals/`: the eval JSON reports behind every number on this card
- `nae1-llama-Q4_K_M.gguf`, `nae1-llama-f32.gguf`, `eval_step*_q4_n512.json`, `ckpt_v7_step60000*`, `tokenizer/`: v1 lineage history, kept for reference
