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
  - sft
  - tool-use
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

## Links

- **Chat with NEA1 — live demo:** https://neuralai-nae1.deandrewh26-d6e.workers.dev
- **NeuralAI — the company:** https://neuralai.wasmer.app
- **Model weights (Hugging Face):** https://huggingface.co/Subject-Emu-5259/NeuralAI-Nae1
- **Code + evals (GitHub):** https://github.com/Subject-Emu-5259/NeuralAI
- **Inference service (GitHub):** https://github.com/Subject-Emu-5259/nae1-inference

---

## Current release: NeuralAI the assistant — SFT v13.2 (LoRA step-12000)

| Property | Value |
| --- | --- |
| **What it is** | Instruction-tuned assistant checkpoint with tool use (`search`, `fetch`, `read_file`) and working plain chat (greetings, identity, small talk), built from the NEA1 pretrain line |
| **Base** | Pretrain step-100000. Same 100.1M architecture |
| **Method** | LoRA rank 32 on attention + MLP with the base weights frozen; adapters merged into a standard checkpoint. Same recipe as the v9–v12.2 gate winners — v13.2 changes the data, not the method |
| **SFT data** | 27,027 rows: the full v13.1 set (25,999) as replay plus 1,028 new rows in two targeted fix banks — a date-roll bank (614 rows: month-end tomorrow cells in every frame, same-row today→tomorrow contrast pairs, and decomposition drills that teach the day number rolling to 1 as the month advances) and an honesty-after-distractor bank (414 rows: date questions with no date given, asked after distractor questions, answered with an honest decline — plus bracket-present controls). Every checkable answer is machine-verified |
| **Training** | LoRA on Kaggle T4, adapter LR peak 1e-4 cosine, 12,000 steps (final train loss 2.5527); step-12000 selected by the gate below |
| **Golden-40 tool-use gate** | **34.5/40 (86.2%) — PASS** (bar 60%). Tool calls 38/40, grounded answers 31/40. Report: `evals/eval_sftv13_2lora_step12000_golden40.json` |
| **Fluency guardrail** | PPL **48.01** on the 50-doc guardrail set vs **47.59** for the step-100000 base on the same set. This is a different, smaller set than the 2000-doc set used below — the numbers are not comparable across sets |
| **Plain chat + conversation** | v13.2's headline is the last two date behaviors. The month-boundary "tomorrow" that every build since v12 missed now rolls in full — after Sunday, August 31, 2025 it answers "Tomorrow is Monday, September 1, 2025." — and the no-bracket honesty slip (echoing a distractor instead of declining) is fixed: asked today's date with none given, it says it doesn't know. Date8 is **8/8** with both no-bracket honesty probes passing. Everything else holds: multiturn12 **11/12** with zero parroting and the Australia follow-up to Canberra, memory6 **5/6**, casual7 **7/7**, and the smoke checks pass — the locked greeting verbatim ("Hey, I'm NeuralAI — how can I help you today?"), "who created you?" naming De'Andrew Preston Harris, and "what is 2 + 2?" answering "2 + 2 = 4." (reports: `evals/eval_sftv13_2lora_step12000_multiturn12.json`, `evals/eval_sftv13_2lora_step12000_date8.json`, `evals/eval_sftv13_2lora_step12000_smoke.json`) |
| **Weights in this repo** | `sft_v13_2_lora_step12000/` (safetensors f32, config, tokenizer with chat template) — the same release also sits at the repo root, so the repository loads directly |
| **Serving** | Production since 2026-10-23: served from NeuralAI's own infrastructure; Render `nae1-inference` runs the same bf16 copy (`sft_v13_2_lora_step12000_bf16/`) as the fallback |
| **Chat template** | Ships in `sft_v13_2_lora_step12000/tokenizer_config.json`. Turns are `User: ...` / `NeuralAI: ...`; the generation prompt ends `NeuralAI:` with no trailing space (see the SFT section for why) |

---

## Previous release: SFT v12.2 (LoRA step-12000)

| Property | Value |
| --- | --- |
| **What it is** | Instruction-tuned assistant checkpoint with tool use (`search`, `fetch`, `read_file`) and working plain chat (greetings, identity, small talk), built from the NEA1 pretrain line |
| **Base** | Pretrain step-100000. Same 100.1M architecture |
| **Method** | LoRA rank 32 on attention + MLP with the base weights frozen; adapters merged into a standard checkpoint. Same recipe as the v9–v11 gate winners — v12.2 changes the data, not the method |
| **SFT data** | 22,290 rows: the full v11 set (15,831) as replay plus 6,459 new rows in targeted fix banks — arithmetic drills, memory recall and honesty, two-part questions, plain-chat corrections, a redesigned date bank (1,220 rows: the date given in the conversation stated back in full — immediately, after a greeting, after distractors, and against event-date decoys — plus tomorrow/yesterday rows and no-date honesty rows), and capital-decoy reinforcement with Australia weighted. Every checkable answer is machine-verified |
| **Training** | LoRA on Kaggle T4, adapter LR peak 1e-4 cosine, 12,000 steps (final train loss 2.8408); step-12000 selected by the gate below |
| **Golden-40 tool-use gate** | **35.0/40 (87.5%) — PASS** (bar 60%). Tool calls 38/40, grounded answers 32/40. Report: `evals/eval_sftv12_2lora_step12000_golden40.json` |
| **Fluency guardrail** | PPL **48.1** on the 50-doc guardrail set vs **47.59** for the step-100000 base on the same set. This is a different, smaller set than the 2000-doc set used below — the numbers are not comparable across sets |
| **Plain chat + conversation** | v12.2's headline is dates. Date questions were the previous build's blocker (3/8 on the date8 gate set); the redesigned date bank takes date8 to **7/8** with both no-bracket honesty probes passing — the model states the date given in the conversation back in full, and says so when none was given instead of inventing one. The one remaining miss is relative arithmetic across a month boundary ("tomorrow" after August 31 repeats August 31 instead of September 1), a v13 item. Everything else holds: multiturn12 **11/12** with zero parroting and the Australia follow-up back to Canberra, memory6 **6/6**, casual7 **7/7**, and the smoke checks pass — the locked greeting verbatim ("Hey, I'm NeuralAI — how can I help you today?"), "who created you?" naming De'Andrew Preston Harris, and "what is 2 + 2?" answering "2 + 2 = 4." (reports: `evals/eval_sftv12_2lora_step12000_multiturn12.json`, `evals/eval_sftv12_2lora_step12000_date8.json`, `evals/eval_sftv12_2lora_step12000_smoke.json`) |
| **Weights in this repo** | `sft_v12_2_lora_step12000/` (safetensors f32, config, tokenizer with chat template) |
| **Serving** | Served in production from 2026-10-22 until the v13.2 swap on 2026-10-23 |
| **Chat template** | Ships in `sft_v12_2_lora_step12000/tokenizer_config.json`. Turns are `User: ...` / `NeuralAI: ...`; the generation prompt ends `NeuralAI:` with no trailing space (see the SFT section for why) |

---

## Previous release: SFT v11 (LoRA step-12000)

| Property | Value |
| --- | --- |
| **What it is** | Instruction-tuned assistant checkpoint with tool use (`search`, `fetch`, `read_file`) and working plain chat (greetings, identity, small talk), built from the NEA1 pretrain line |
| **Base** | Pretrain step-100000. Same 100.1M architecture |
| **Method** | LoRA rank 32 on attention + MLP with the base weights frozen; adapters merged into a standard checkpoint. Same recipe as the v9/v10 gate winners — v11 changes the data, not the method |
| **SFT data** | 15,831 rows: the v10 set (11,099 — with its 48 drifted greetings rewritten to the locked line) plus 4,866 new multi-turn rows: topic switches, same-topic follow-ups, recall with distractors, tool+chat mixes, and identity/greetings arriving mid-conversation. Every greeting turn in the set is the locked line verbatim, and every checkable answer is machine-verified — with 25% FineWeb replay mixed into training |
| **Training** | LoRA on Kaggle T4, adapter LR peak 1e-4 cosine, 12,000 steps; step-12000 selected by the gate below |
| **Golden-40 tool-use gate** | **35.5/40 (88.8%) — PASS** (bar 60%). Tool calls 39/40, grounded answers 32/40. Report: `evals/eval_sftv11lora_step12000_golden40.json` |
| **Fluency guardrail** | PPL **48.08** on the 50-doc guardrail set vs **47.59** for the step-100000 base on the same set (`evals/eval_sftv11lora_step12000_ppl50_n512.json`). This is a different, smaller set than the 2000-doc set used below — the numbers are not comparable across sets |
| **Plain chat + conversation** | v11's headline is real conversation. A new gate set (multiturn12) passes **11/12 with zero cross-topic parroting**: independent questions inside one conversation stay on their own topics, follow-ups and recall-with-distractors work, and tool use survives inside longer transcripts (report: `evals/eval_sftv11lora_step12000_multiturn12.json`). The locked greeting is restored verbatim — "hello" returns "Hey, I'm NeuralAI — how can I help you today?" word-for-word — and v10's fixes hold: "who created you?" names De'Andrew Preston Harris and "what is 2 + 2?" answers "2 + 2 = 4." (smoke report: `evals/eval_sftv11lora_step12000_smoke.json`) |
| **Weights in this repo** | `sft_v11_lora_step12000/` (safetensors f32, config, tokenizer with chat template) |
| **Serving** | Served in production on 2026-10-22 until the v12.2 swap the same day |
| **Chat template** | Ships in `sft_v11_lora_step12000/tokenizer_config.json`. Turns are `User: ...` / `NeuralAI: ...`; the generation prompt ends `NeuralAI:` with no trailing space (see the SFT section for why) |

---

## Previous release: SFT v9 (LoRA step-12000)

| Property | Value |
| --- | --- |
| **What it is** | Instruction-tuned assistant checkpoint with tool use (`search`, `fetch`, `read_file`) and working plain chat (greetings, identity, small talk), built from the NEA1 pretrain line |
| **Base** | Pretrain step-100000. Same 100.1M architecture |
| **Method** | LoRA rank 32 on attention + MLP with the base weights frozen; adapters merged into a standard checkpoint. Picked over full fine-tuning at the v9 gate: the full-FT arm scored the same on golden-40 but failed the fluency bar |
| **SFT data** | 9,827 rows: the v8 set (8,177) plus 1,650 new rows aimed at plain chat — greetings/small talk, identity, direct factual Q&A, instruction, reasoning, advice, groundedness — with 25% FineWeb replay mixed into training |
| **Training** | LoRA on Kaggle T4, adapter LR peak 1e-4 cosine, 12,000 steps; step-12000 selected by the gate below |
| **Golden-40 tool-use gate** | **33.5/40 (83.8%) — PASS** (bar 60%). Tool calls 39/40, grounded answers 28/40. Report: `evals/eval_sftv9lora_step12000_golden40.json` |
| **Fluency guardrail** | PPL **47.96** on the 50-doc guardrail set vs **47.59** for the step-100000 base on the same set (`evals/eval_sftv9lora_step12000_ppl50_n512.json`) — the v8 release's fluency cost (59.26) is closed. This is a different, smaller set than the 2000-doc set used below — the numbers are not comparable across sets |
| **Plain chat** | The v8 release's live failure — a bare "hello" sent it into a tool call — is fixed: the locked greeting comes back word-for-word, and small talk, identity, and simple factual answers work (smoke report: `evals/eval_sftv9lora_step12000_smoke.json`). Two residuals, both data-emphasis problems targeted by v10: the creator answer names NeuralAI rather than De'Andrew Preston Harris, and simple arithmetic is unreliable |
| **Weights in this repo** | `sft_v9_lora_step12000/` (safetensors f32, config, tokenizer with chat template) |
| **Serving** | Served in production on 2026-10-21 until the v10 swap the same day |
| **Chat template** | Ships in `sft_v9_lora_step12000/tokenizer_config.json`. Turns are `User: ...` / `NeuralAI: ...`; the generation prompt ends `NeuralAI:` with no trailing space (see the SFT section for why) |

---

## Previous release: SFT v8 (step-16000)

| Property | Value |
| --- | --- |
| **What it is** | Instruction-tuned assistant checkpoint with tool use (`search`, `fetch`, `read_file`), built from the NEA1 pretrain line |
| **Base** | Pretrain step-100000 (the pretrain continued past the step-80000 base release below to step-100000; the SFT line starts there). Same 100.1M architecture |
| **SFT data** | 8,177 unique rows: 6,500 machine-verified tool traces, 1,265 teacher rows, 400 chat, 12 identity |
| **Training** | Full SFT on Kaggle T4, LR peak 5e-6 cosine; step-16000 selected from the 20,000-step run by the gate below |
| **Golden-40 tool-use gate** | **32.5/40 (81.2%) — PASS** (bar 60%). Tool calls 36/40, grounded answers 29/40. Report: `evals/eval_sftv8_step16000_golden40.json` |
| **Fluency guardrail** | PPL **59.26** on a 50-doc guardrail set vs **47.59** for the step-100000 base on the same set (`evals/eval_sftv8_step16000_ppl50_n512.json`). This is a different, smaller set than the 2000-doc set used below — the numbers are not comparable across sets. The fluency cost is this release's known trade-off; closing it is the v9 program |
| **Weights in this repo** | `sft_v8_step16000/` (safetensors f32, config, tokenizer with chat template) |
| **Serving** | Served in production on 2026-10-21 until the v9 swap the same day |
| **Chat template** | Ships in `sft_v8_step16000/tokenizer_config.json`. Turns are `User: ...` / `NeuralAI: ...`; the generation prompt ends `NeuralAI:` with no trailing space (see the SFT section for why) |

---

## Base release: step-80000 (v2 lineage)

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
| **Weights in this repo** | `ckpt_v7_step80000_bf16/` (safetensors, former serving copy — production moved to the SFT line on 2026-10-21: v8, then v9, v10, v11, v12.2, and v13.2) + `tokenizer_v2/`. GGUF Q4_K_M: `nae1-step80000-Q4_K_M.gguf` |
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

## SFT: from base to assistant

The gate is golden-40: 40 held-out tasks whose facts exist only inside each task's tool-result block, so a grounded answer cannot come from pretraining memory. Scoring: 0.5 for a correct tool call, 0.5 for an answer containing the fact. Bar: at least 60% and strictly better than the base model, which scores 3.0/40 with 0/40 tool calls.

| Version | Score | Calls | Grounded | Guardrail PPL (50-doc) | Outcome |
| --- | --- | --- | --- | --- | --- |
| v5 | 3.5/40 | 0/40 | 7/40 | 50.09 | Fail — the data taught an answer template instead of reading tool results |
| v6 | 13.0/40 | 0/40 | 26/40 | 49.93 | Fail — grounded answers learned, calls never emitted |
| v7 | 12.5/40 | 0/40 | 25/40 | 79.19 | Fail — calls never emitted; LR 2e-5 blew the fluency guardrail |
| v8 step-20000 | 32.0/40 | 35/40 | 29/40 | 59.73 | Pass |
| v8 step-16000 | 32.5/40 | 36/40 | 29/40 | 59.26 | Pass — released 2026-10-21, superseded by v9 the same day |
| v9 full-FT step-12000 | 33.5/40 | 38/40 | 29/40 | 52.48 | Fail — fluency bar (≤52) missed by 0.48; plain-chat facts not learned |
| v9 LoRA step-12000 | 33.5/40 | 39/40 | 28/40 | 47.96 | Pass — released 2026-10-21, superseded by v10 the same day |
| v10 LoRA step-12000 | 34.0/40 | 39/40 | 29/40 | 48.01 | Pass — released 2026-10-21, superseded by v11 on 2026-10-22 |
| v11 LoRA step-12000 | 35.5/40 | 39/40 | 32/40 | 48.08 | Pass — released 2026-10-22, superseded by v12.2 the same day (adds multiturn12 11/12) |
| v12.2 LoRA step-12000 | 35.0/40 | 38/40 | 32/40 | 48.1 | Pass — released 2026-10-22, superseded by v13.2 on 2026-10-23 (date8 7/8, memory6 6/6, casual7 7/7) |
| **v13.2 LoRA step-12000 (released)** | **34.5/40** | **38/40** | **31/40** | **48.01** | **Pass — selected (date8 8/8, month-boundary tomorrow fixed, casual7 7/7)** |

**Root cause of the v5–v7 zero calls, proven at token level.** The training kernel's assistant-span mask started after the space in `NeuralAI: `, but the ByteLevel tokenizer fuses that space and the following `»` into a single token (id 4776) — so the first token of every tool call was excluded from the loss in every run, and the models learned the call body conditional on a first token that was never trained. v8 fixes the span, carries a pre-flight assertion (junction token supervised in 7,161 of 7,161 tool traces, confirmed in the training log), and ends the generation prompt at `NeuralAI:` so it tokenizes as a true prefix of the training format. The shipped chat template uses the same junction.

**Tool protocol.** The assistant emits a call line — `» search("query")`, `» fetch("url")`, or `» read_file("path")`. The host executes the call, returns the result block into the transcript, and the assistant answers from the returned text.

**Known limitations.** General prose is still rough at 100M parameters. v13.2 closed the date line: date8 is 8/8, the month-boundary "tomorrow" rolls in full (August 31 → "Tomorrow is Monday, September 1, 2025."), and the no-bracket date question declines honestly even after a distractor — at no fluency cost (48.01 against the base's 47.59 on the guardrail set). What remains: garbled or vacuous phrasing on some definition and follow-up questions, an occasional name-recall slip inside longer memory probes, repetition loops on open-ended prompts at greedy decoding, and live-tool search turns that do not always follow the call-then-answer shape — the search behavior remains a genuine model gap.

---

## Quickstart

The GGUF is a llama.cpp model file. The perplexity numbers above were scored with the exact ByteLevel BPE v2 token IDs from `tokenizer_v2/`. For exact token IDs, use the tokenizer files in this repo.

```python
from llama_cpp import Llama

llm = Llama(model_path="nae1-step80000-Q4_K_M.gguf", n_ctx=512, verbose=False)
out = llm("The quick brown fox", max_tokens=32, temperature=0.0)
print(out["choices"][0]["text"])
```

The GGUF quickstart above loads the pretrained base (step-80000), which has no chat template. For the assistant, use `sft_v13_2_lora_step12000/` with the chat template in its `tokenizer_config.json`.

---

## Files

- `model.safetensors` + `config.json` + `tokenizer.json` + `tokenizer_config.json` (repo root): the current v13.2 assistant release, identical to `sft_v13_2_lora_step12000/`
- `sft_v13_2_lora_step12000/`: current assistant release (SFT v13.2, LoRA merged) — safetensors f32 + tokenizer with chat template
- `sft_v13_2_lora_step12000_bf16/`: bf16 serving copy of the current assistant release (what production runs)
- `sft_v12_2_lora_step12000/`: previous assistant release (SFT v12.2, LoRA merged) — safetensors f32 + tokenizer with chat template
- `sft_v12_2_lora_step12000_bf16/`: bf16 serving copy of the v12.2 release
- `sft_v11_lora_step12000/`: earlier assistant release (SFT v11, LoRA merged) — safetensors f32 + tokenizer with chat template
- `sft_v11_lora_step12000_bf16/`: bf16 serving copy of the v11 release
- `sft_v10_lora_step12000/`: previous assistant release (SFT v10, LoRA merged) — safetensors f32 + tokenizer with chat template
- `sft_v10_lora_step12000_bf16/`: bf16 serving copy of the v10 release
- `sft_v9_lora_step12000/`: previous assistant release (SFT v9, LoRA merged) — safetensors f32 + tokenizer with chat template
- `sft_v9_lora_step12000_bf16/`: bf16 serving copy of the v9 release
- `sft_v8_step16000/`: previous assistant release (SFT v8) — safetensors f32 + tokenizer with chat template
- `sft_v8_step16000_bf16/`: bf16 serving copy of the v8 release
- `nae1-step80000-Q4_K_M.gguf`: base GGUF (Q4_K_M, 76.6 MiB)
- `ckpt_v7_step80000_bf16/`: serving safetensors checkpoint
- `tokenizer_v2/`: 32k ByteLevel BPE tokenizer used by step-80000
- `evals/`: the eval JSON reports behind every number on this card
- `eval_step*_q4_n512.json`, `ckpt_v7_step60000*`, `tokenizer/`: v1 lineage history, kept for reference. (The v1-era GGUF ports `nae1-llama-f32.gguf` / `nae1-llama-Q4_K_M.gguf` were removed on 2026-10-21 — they carried the 439-token v1 tokenizer and made this page misreport the model size as 76.2M.)
