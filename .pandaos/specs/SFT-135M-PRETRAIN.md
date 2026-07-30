# S-001: NeuralAI-Air-135M Pre-Training + SFT + DPO Specification

**Spec ID:** S-001  
**Project:** P-135M-PRETRAIN-v1  
**Status:** APPROVED (Founder: De'Andrew Harris)  
**Budget:** $170 (A100 40GB rental)  
**Target Timeline:** ~14 days pre-training + 2 days SFT/DPO/export  
**Author:** Planner (NeuralAI Team)  
**Date:** 2026-07-29  

---

## 1. Objective

Produce a fully pre-trained NeuralAI-Air-135M model from random initialization on 1 billion tokens of quality text, followed by supervised fine-tuning (SFT v19) and direct preference optimization (DPO v19), yielding a coherent, instruction-following 135M-parameter model suitable for production inference via llama.cpp GGUF.

**Success Criteria:**
| Metric | Target |
|--------|--------|
| Pre-training final loss | < 2.5 |
| SFT final loss | < 1.5 |
| DPO reward accuracy | > 85% |
| Perplexity on held-out Wiki | < 25 |
| Inference coherence | Grammatically correct, on-topic English |
| Identity accuracy | Names De'Andrew Harris as creator |
| Safety refusal | 100% on harmful prompts in eval set |

---

## 2. Model Architecture (Frozen)

Source: `NeuralAI-Air-135M-HF/config.json`

| Parameter | Value |
|-----------|-------|
| Architecture | LlamaForCausalLM |
| Hidden size | 768 |
| Intermediate size | 2,560 |
| Layers | 15 |
| Attention heads | 12 |
| KV heads (GQA) | 2 |
| Vocab size | 32,000 |
| Max position embeddings | 2,048 |
| RMSNorm eps | 1e-5 |
| RoPE theta | 10,000 |
| Activation | SwiGLU (silu) |
| Tie word embeddings | true |
| BOS token | 1 (`<|im_start|>`) |
| EOS token | 2 (`<|im_end|>`) |
| PAD token | 0 (`<|endoftext|>`) |
| ~Total parameters | **~135M** |

**Note:** We do not modify architecture during this project. The tokenizer (`tokenizer.json`) is also frozen — it defines a custom BPE vocab with ChatML special tokens.

---

## 3. Phase 1: Pre-Training (1B Tokens)

### 3.1 Data Sources & Mix

All corpora are public-domain / permissively licensed. Target mix is designed to maximize general-domain coherence while preserving lightweight code understanding.

| Source | Weight | Est. Tokens | Notes |
|--------|--------|-----------|-------|
| C4 (en) | 40% | 400M | Deduplicated via HuggingFace `c4` |
| Wikipedia (en) | 20% | 200M | High-quality factual prose |
| OpenWebText (v2) | 15% | 150M | Reddit-curated web text |
| StackExchange | 10% | 100M | Q&A format, reasoning structure |
| Books (Gutenberg + PGUS) | 10% | 100M | Long-form narrative coherence |
| Supplementary STEM | 5% | 50M | ArXiv abstracts + Khan Academy transcripts |
| **Total** | **100%** | **~1.0B** | |

### 3.2 Data Pipeline Architecture

```
┌──────────────┐    ┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐
│  Raw Source  │ → │ Deduplicate│ → │   Filter    │ → │  Tokenize   │ → │  Shard   │
│  (HF datasets│    │ MinHash   │    │ Quality +   │    │  Custom     │    │ 100MB    │
│   or URLs)   │    │ LSH       │    │ Length      │    │  BPE 32K    │    │ chunks   │
└──────────────┘    └──────────┘    └─────────────┘    └─────────────┘    └──────────┘
```

**Exact pipeline steps:**
1. **Download** — HuggingFace `datasets` library with streaming (`streaming=True` to avoid local storage limits)
2. **Deduplication** — Exact substring dedup (13-gram) + MinHash LSH (Jaccard > 0.85) on C4 and OpenWebText
3. **Quality filtering** —
   - Remove documents with < 100 characters or > 100K characters
   - Remove documents where > 30% of lines start with boilerplate ("Cookie Policy", "All rights reserved")
   - Language detection: keep only `lang==en` with confidence > 0.9
4. **Tokenization** — Use `NeuralAI-Air-135M-HF/tokenizer.json` via `tokenizers` library. Pack to `input_ids` without padding. Add `<|endoftext|>` between documents.
5. **Shuffling & sharding** — Shuffle globally with a fixed seed (`42`), then shard into `.bin` files of ~100MB each for streaming mmap during training.

**Pipeline output path:** `data/pretrain/`  
**Expected raw disk usage:** ~4 GB text → ~2 GB tokenized `.bin` shards

### 3.3 Hyperparameters (Pre-Training)

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| **Context length** | 512 | Maximizes throughput on A100. Model supports 2048, but 512 tokens/step yields ~2× higher samples/sec than 1024. We extend to 1024 during SFT. |
| **Per-device batch size** | 64 | A100 40GB easily fits 64 × 512 with bf16 + gradient checkpointing (~1.5 GB model state + ~300 MB activations). |
| **Gradient accumulation** | 1 | Single GPU; batch 64 already provides stable gradients for 135M. No need to accumulate. |
| **Global batch size (tokens)** | 32,768 | 64 × 512 = 32,768 tokens per step. For 1B tokens → ~30,500 steps. This is small compared to Chinchilla-optimal batch sizes for large models, but adequate for sample efficiency at 135M scale (Hoffmann et al. 2022). |
| **Training steps** | ~30,500 | `1,000,000,000 / 32,768 ≈ 30,518`. Rounded to 30,500. |
| **Peak learning rate** | 6.0e-4 | For models <1B, 4e-4 to 1e-3 is common (TinyLlama 1.1B used 4e-4). 6e-4 balances convergence speed and stability. |
| **LR schedule** | Cosine decay to 10% of peak | Standard for decoder-only LMs. Final LR = 6.0e-5. |
| **Warmup steps** | 300 | ~1% of total steps. Prevents early-step spikes. |
| **Optimizer** | AdamW (β1=0.9, β2=0.95, ε=1e-8) | β2=0.95 is slightly lower than default 0.999; empirically more stable for small LMs (Llama 2 recipe). |
| **Weight decay** | 0.1 | Applied to all parameters except biases and LayerNorm/RMSNorm. Standard for pre-training. |
| **Gradient clipping** | 1.0 | Prevents rare gradient spikes during early training. |
| **Mixed precision** | bf16 | A100 has native bf16 tensor cores. Avoids fp16 gradient underflow. |
| **Flash Attention** | FA2 | `pip install flash-attn --no-build-isolation`. Required for memory and speed. |
| **Gradient checkpointing** | Enabled | ~15% slowdown, but frees memory for larger batches. |
| **torch.compile** | Enabled (if PyTorch ≥2.1) | Graph compilation yields ~10-20% speedup on A100. |
| **Dropout** | 0.0 | Standard for pre-training dense LMs. |
| **Seed** | 42 | Reproducibility. |

**Scaling-law justification:**
- Chinchilla optimal compute for 135M params would suggest ~2.7B tokens (20 tokens/param). We target 1B due to budget/timeline constraints. This is **sub-optimal** but sufficient for basic coherence; we note this risk in §11.
- Batch size follows the heuristic that smaller models benefit from smaller per-step batches for sample efficiency (Zhang et al. 2024, "The Optimal Batched Size for Small LMs").

### 3.4 Compute Estimation

| Metric | Estimate |
|--------|----------|
| GPU | A100 40GB (single node, single device) |
| Est. throughput | ~6,000–8,000 tokens/sec (forward + backward, bf16, FA2) |
| Pure compute time | ~35–45 hours |
| With checkpointing/validation/debugging | ~10–14 days |
| Cost @ ~$0.50/hr (Lambda/RunPod spot) | ~$170 |

**Checkpointing frequency:** Every 2,000 steps (~6.5% of training, every ~3–4 hours).  
**Validation frequency:** Every 5,000 steps on a 10M-token held-out split.

### 3.5 Checkpointing & Resume Strategy

- **Save directory:** `checkpoints/pretrain/`
- **Naming:** `checkpoint-{step}/` containing `model.safetensors`, `optimizer.bin`, `scheduler.bin`, `rng_state.pth`, `trainer_state.json`
- **Resume:** Training script accepts `--resume_from_checkpoint checkpoint-{step}`. HuggingFace `Trainer` handles this natively.
- **Best checkpoint:** Keep the checkpoint with lowest validation perplexity, not just the last one.
- **Offloading:** Sync completed checkpoints to Google Drive or local NAS every 6 hours via `rsync` cron job to prevent data loss on spot-instance preemption.

---

## 4. Phase 2: Supervised Fine-Tuning (SFT v19)

### 4.1 Dataset

- **Path:** `data/train_sft_v19.jsonl`
- **Size:** 1,016 examples
- **Format:** ChatML with fields `text`, `system`, `instruction`, `output`
- **Special tokens:** `<|im_start|>` (id=1), `<|im_end|>` (id=2), `<|endoftext|>` (id=0)

**Categories in v19:**
| Category | Count |
|----------|-------|
| Identity / Persona | 8 |
| Coding | 297 |
| Math / Reasoning | 153 |
| Safety / Refusal | 35 |
| Creative Writing | 99 |
| Riddles / Logic | 45 |
| Multi-step | 30 |
| Chat / Conversational | 79 |
| Tool Use | 50 |
| Unit Conversions | 20 |
| Grammar | 29 |
| History | 40 |
| Science | 70 |
| General Facts | 50 |
| **Total** | **1,016** |

### 4.2 Hyperparameters (SFT)

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| **Base model** | Best pre-training checkpoint (lowest val PPL) |
| **Epochs** | 3 | Standard for ~1K examples. Diminishing returns after 3. |
| **Max length** | 1,024 | Uses full model capacity. Longer than pre-training context intentionally. |
| **Per-device batch** | 4 | A100 40GB fits 4 × 1024 easily. |
| **Gradient accumulation** | 4 | Effective batch = 16. Stable for small datasets. |
| **Learning rate** | 2.0e-5 | ~30× lower than pre-training peak. Prevents catastrophic forgetting while adapting to chat format. |
| **LR schedule** | Linear decay from 2e-5 to 0 | Simpler than cosine for short SFT runs. |
| **Warmup ratio** | 3% (~92 steps) | Minimal warmup needed since we're fine-tuning, not training from scratch. |
| **Weight decay** | 0.01 | Lower than pre-training to avoid over-regularization on small data. |
| **Optimizer** | AdamW (β1=0.9, β2=0.999, ε=1e-8) | Default β2 restored for fine-tuning stability. |
| **FP16** | true | Slightly faster than bf16 on A100 for small-batch fine-tuning; no underflow risk at this LR. |
| **Gradient clipping** | 1.0 | |
| **Loss masking** | Mask `user` and `system` tokens; compute loss only on `assistant` tokens | Standard instruction-tuning practice. |

### 4.3 Data Format for Trainer

Each example is converted to:
```python
{
  "input_ids": [...],      # full conversation tokenized
  "labels": [...],         # -100 for non-assistant tokens, token_id for assistant tokens
  "attention_mask": [...]  # 1 for real tokens, 0 for padding
}
```

Padding to `MAX_LENGTH=1024` with PAD token id=0. Truncation from the left if exceeding 1024.

### 4.4 Expected Metrics

| Metric | Target |
|--------|--------|
| Training loss (end of epoch 3) | < 1.5 |
| Training time | ~15–25 min |
| Output coherence (manual sample) | Grammatical, on-topic |

---

## 5. Phase 3: Direct Preference Optimization (DPO v19)

### 5.1 Dataset

- **Path:** `data/train_dpo_v19.jsonl`
- **Size:** 350 preference pairs
- **Format:** For each prompt, a `chosen` (correct) and `rejected` (incorrect/harmful) completion

### 5.2 Hyperparameters (DPO)

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| **Base model** | SFT checkpoint from Phase 2 |
| **Method** | LoRA DPO (not full-parameter) | Saves memory, prevents overfitting on 350 pairs. |
| **LoRA rank (r)** | 32 | High enough to capture preference shifts; low enough to avoid overfitting. |
| **LoRA alpha** | 64 | 2× rank is standard (Hu et al. 2022). |
| **LoRA target modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | All linear layers that matter for attention and MLP. |
| **Epochs** | 3 | |
| **Per-device batch** | 2 | DPO needs 2× memory (chosen + rejected concatenated). |
| **Gradient accumulation** | 8 | Effective batch = 16 pairs per step. |
| **Learning rate** | 5.0e-6 | Very low LR for adapter-only training to preserve SFT knowledge. |
| **LR schedule** | Linear decay to 0 | |
| **Warmup ratio** | 10% (~35 steps) | Higher warmup for DPO due to sensitivity early in training. |
| **Beta (β)** | 0.1 | Controls deviation from reference policy. Low β = stay close to SFT model. Good for small data. |
| **FP16** | true | |
| **Gradient clipping** | 1.0 | |

### 5.3 Expected Metrics

| Metric | Target |
|--------|-------|
| DPO reward accuracy | > 85% |
| Training time | ~10–20 min |
| KL divergence from SFT | < 0.5 nats (monitored to catch collapse) |

---

## 6. Phase 4: Export & Registry Update

### 6.1 Merge LoRA (if applicable)

```bash
python scripts/merge_lora.py \
  --base checkpoints/sft-v19/ \
  --adapter checkpoints/dpo-v19/ \
  --output checkpoints/NeuralAI-Air-135M-v19-merged/
```

### 6.2 GGUF Conversion

```bash
# 1. Convert merged HF model to GGUF F16
python /path/to/llama.cpp/convert_hf_to_gguf.py \
  checkpoints/NeuralAI-Air-135M-v19-merged/ \
  --outfile models/NeuralAI-Air-135M-v19-f16.gguf \
  --outtype f16

# 2. Quantize to Q4_K_M for production
/path/to/llama.cpp/llama-quantize \
  models/NeuralAI-Air-135M-v19-f16.gguf \
  models/NeuralAI-Air-135M-v19-Q4_K_M.gguf \
  Q4_K_M
```

**Note on F16:** The 135M model has had historical issues with F16 on certain llama.cpp builds (see `TRAINING_PIPELINE.md`). If `llama-quantize` segfaults on the F16 file, regenerate the F16 GGUF with `--outtype f32` and quantize from F32 instead.

### 6.3 Model Registry Update

```bash
python scripts/model_manager.py set neuralai-air-135m-v19
```

This updates `config/active_model.json` and restarts the inference service.

---

## 7. Training Script Structure

The training system is organized as follows:

```
training/
├── pretrain/
│   ├── run_pretrain.py          # Main pre-training entrypoint (HF Trainer)
│   ├── data_pipeline.py         # Download → dedup → tokenize → shard
│   ├── config_pretrain.yaml     # Hyperparameters for Phase 1
│   └── requirements.txt         # flash-attn, transformers, datasets, etc.
├── sft/
│   ├── run_sft.py               # SFT entrypoint
│   ├── config_sft.yaml          # Phase 2 hyperparameters
│   └── data_utils.py            # ChatML formatting + label masking
├── dpo/
│   ├── run_dpo.py               # DPO entrypoint (TRL)
│   ├── config_dpo.yaml          # Phase 3 hyperparameters
│   └── lora_config.json         # LoRA target modules + r/alpha
├── common/
│   ├── model_utils.py           # Load base architecture, tokenizer init
│   ├── logging_utils.py         # WandB / TensorBoard setup
│   └── checkpoint_utils.py      # Resume, best-checkpoint selection
└── notebooks/
    └── NeuralAI_Air_135M_v19.ipynb   # Colab-ready unified notebook (fallback)
```

**Main components (not full code):**

### 7.1 `run_pretrain.py`
- Parses `config_pretrain.yaml`
- Loads tokenizer from `NeuralAI-Air-135M-HF/`
- Initializes `LlamaForCausalLM` from `config.json` (random weights)
- Uses `StreamingDataset` to read `data/pretrain/*.bin` shards
- Configures `TrainingArguments` with the hyperparameters in §3.3
- Sets up `Trainer` with custom `DataCollatorForLanguageModeling` (no NSP, causal LM)
- Integrates WandB logging (`project=neuralai-pretrain`, `name=135M-1B`)
- Saves checkpoints to `checkpoints/pretrain/`

### 7.2 `data_pipeline.py`
- `download_sources()` — streams from HuggingFace `datasets`
- `deduplicate_minhash()` — MinHash LSH with `datasketch`
- `filter_quality()` — length, language, boilerplate filters
- `tokenize_and_shard()` — uses `tokenizers` library, writes `.bin` with `numpy.memmap`
- `build_splits()` — creates `train/` (99%) and `val/` (1%) directories

### 7.3 `run_sft.py`
- Loads best pre-train checkpoint
- Reads `data/train_sft_v19.jsonl`
- Formats each example into ChatML with label masking
- Uses `SFTTrainer` from TRL or custom `Trainer` with `DataCollatorForSeq2Seq`
- Saves to `checkpoints/sft-v19/`

### 7.4 `run_dpo.py`
- Loads SFT checkpoint
- Reads `data/train_dpo_v19.jsonl`
- Uses `DPOTrainer` from TRL with LoRA via PEFT
- Saves adapter to `checkpoints/dpo-v19/`

---

## 8. Monitoring & Checkpointing Strategy

### 8.1 Real-Time Monitoring

| Tool | Purpose | Log Frequency |
|------|---------|---------------|
| **WandB** | Loss curves, LR, grad norm, throughput | Every step |
| **TensorBoard** | Local backup of WandB metrics | Every step |
| **Console** | Step, loss, tokens/sec, ETA | Every 100 steps |

**WandB project:** `neuralai-pretrain`  
**Run name:** `135M-1B-{timestamp}`  
**Logged metrics:**
- `train/loss`, `train/learning_rate`, `train/grad_norm`
- `train/tokens_per_sec`, `train/flops_per_sec`
- `eval/loss`, `eval/perplexity` (validation split)
- `system/gpu_memory_allocated`, `system/gpu_temperature`

### 8.2 Checkpointing Schedule

| Phase | Save Every | Validation Every | Keep N Best |
|-------|-----------|------------------|-------------|
| Pre-training | 2,000 steps | 5,000 steps | 3 (by eval PPL) |
| SFT | 100 steps | End of epoch | 1 (last) |
| DPO | 50 steps | End of epoch | 1 (last) |

### 8.3 Spot-Instance Preemption Mitigation

Since we will likely rent an A100 on Lambda Labs or RunPod (spot pricing):
- **Automatic sync:** Every 2,000 steps, `rsync` the latest checkpoint to a persistent Google Drive mount or S3 bucket.
- **Resume script:** On boot, `run_pretrain.py` checks `checkpoints/pretrain/` for the latest checkpoint and resumes automatically.
- **Heartbeat:** A simple cron job touches a `alive.txt` every 5 minutes. If the instance dies, we know the last sync point.

---

## 9. Evaluation Criteria at Each Phase

### 9.1 Gate 0: Data Ready
- [ ] 1B tokens tokenized and sharded
- [ ] Validation split contains 10M tokens, no overlap with train
- [ ] Deduplication report: % removed per source < 40%
- [ ] Tokenizer vocab alignment verified: 32,000 tokens, special tokens at ids 0/1/2

### 9.2 Gate 1: Pre-Training Complete
- [ ] Final train loss < 2.5
- [ ] Validation perplexity < 25 on Wiki held-out
- [ ] Model generates coherent English sentences (manual inspection of 10 prompts)
- [ ] No NaN/Inf in any checkpoint
- [ ] At least 3 checkpoints saved successfully

### 9.3 Gate 2: SFT Complete
- [ ] Final train loss < 1.5
- [ ] Model responds to "Who are you?" with identity including De'Andrew Harris
- [ ] Model answers math questions correctly (5/5 sample)
- [ ] Model writes valid Python (5/5 sample)
- [ ] Model refuses harmful requests (5/5 sample)

### 9.4 Gate 3: DPO Complete
- [ ] Reward accuracy > 85% on eval split (70 pairs held out from DPO data)
- [ ] KL divergence from SFT < 0.5 nats
- [ ] No collapse: model still answers harmless questions correctly (5/5)
- [ ] Safety refusal maintained or improved vs SFT

### 9.5 Gate 4: Export & Deploy
- [ ] GGUF F16 generates identical outputs to HF model (5 sample comparisons)
- [ ] Q4_K_M quant passes sanity check (no gibberish)
- [ ] `model_manager.py set` succeeds and service restarts
- [ ] Web UI smoke test passes (hello, math, code, refusal)

---

## 10. Risk Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Insufficient compute budget** | High | Blocks project | Pre-approve $170; keep Colab T4 notebook as fallback; if spot instance preempted >3×, switch to on-demand for final 3 days. |
| **Data quality issues** | Medium | Poor coherence, hallucinations | Aggressive dedup + filtering (§3.2); Model QA audits 100 random docs before training starts; abort if >10% are garbage. |
| **Training instability (loss spike)** | Medium | Wasted compute / NaN | Gradient clipping at 1.0; LR warmup 300 steps; save checkpoint every 2K steps; if loss spikes >3×, resume from previous checkpoint and reduce LR by 2×. |
| **Tokenizer mismatch** | Low | Broken inference | Verify `tokenizer.json` vocab size = 32,000 and special token ids match config (0/1/2) before any training. |
| **Catastrophic forgetting during SFT** | Low | Loses pre-train coherence | Low LR (2e-5), only 3 epochs, weight decay 0.01. Monitor perplexity on Wiki during SFT; if PPL rises >20%, stop and reduce LR. |
| **DPO collapse (model becomes mute)** | Low | Model refuses everything or outputs gibberish | Low β=0.1 keeps model close to SFT; monitor KL divergence; if KL > 1.0, stop and reduce β to 0.05. |
| **Spot instance preemption** | High | Lost hours of progress | Sync checkpoints every 2K steps; auto-resume script; budget 2× expected time. |
| **1B tokens insufficient for coherence** | Medium | Model still weak after 14 days | Acknowledged: Chinchilla-optimal is ~2.7B tokens. If loss > 3.0 at 1B, Founder decides whether to extend budget for 2B tokens or accept lower quality. |

---

## 11. Exact File Paths & Naming Conventions

### 11.1 Input Artifacts (Frozen)

```
NeuralAI-Air-135M-HF/
├── config.json               # Model architecture definition
├── tokenizer.json            # Custom BPE 32K tokenizer
└── (no model weights — we train from scratch)

data/
├── train_sft_v19.jsonl       # 1,016 SFT examples
├── train_dpo_v19.jsonl       # 350 preference pairs
└── pretrain/                 # GENERATED by pipeline
    ├── train/
    │   ├── shard_000.bin
    │   ├── shard_001.bin
    │   └── ...
    └── val/
        ├── shard_000.bin
        └── ...
```

### 11.2 Output Artifacts (Generated)

```
checkpoints/
├── pretrain/
│   ├── checkpoint-2000/
│   ├── checkpoint-4000/
│   ├── checkpoint-6000/
│   ├── ...
│   ├── checkpoint-30000/     # Final
│   └── best/ -> symlink to best eval PPL checkpoint
├── sft-v19/
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── checkpoint-300/       # Final (3 epochs × ~100 steps)
└── dpo-v19/
    ├── checkpoint-50/
    ├── checkpoint-100/
    └── checkpoint-150/       # Final (3 epochs × ~50 steps)

models/
├── NeuralAI-Air-135M-v19-merged/      # HF format after LoRA merge
├── NeuralAI-Air-135M-v19-f16.gguf     # F16 GGUF
└── NeuralAI-Air-135M-v19-Q4_K_M.gguf  # Production quant

training/
├── pretrain/
│   ├── run_pretrain.py
│   ├── data_pipeline.py
│   └── config_pretrain.yaml
├── sft/
│   ├── run_sft.py
│   └── config_sft.yaml
├── dpo/
│   ├── run_dpo.py
│   └── config_dpo.yaml
└── common/
    ├── model_utils.py
    ├── logging_utils.py
    └── checkpoint_utils.py
```

### 11.3 Registry & Config

```
config/
└── active_model.json         # Updated by scripts/model_manager.py
```

---

## 12. Timeline Summary

| Phase | Duration | Start Day | End Day | Dependencies |
|-------|----------|-----------|---------|-------------|
| Data pipeline build | 2 days | Day 0 | Day 2 | Spec approved |
| Data acquisition + tokenization | 3 days | Day 2 | Day 5 | Pipeline ready |
| Pre-training | 14 days | Day 5 | Day 19 | Data ready, GPU rented |
| Checkpoint evaluation | 2 days | Day 19 | Day 21 | Pre-training done |
| SFT training | 1 day | Day 21 | Day 22 | Best pre-train checkpoint |
| DPO training | 1 day | Day 22 | Day 23 | SFT done |
| GGUF conversion + deploy | 1 day | Day 23 | Day 24 | DPO done |
| **Total** | **~24 days** | | | |

**Note:** Pre-training is the critical path. SFT/DPO/export are trivial (~2 days) once the pre-trained base exists. The 14-day pre-training estimate includes debugging, validation, and spot-instance overhead.

---

## 13. Open Questions & Decisions

| # | Question | Status | Decision / Note |
|---|----------|--------|-----------------|
| 1 | **GPU provider:** Lambda Labs vs RunPod vs Vast.ai? | **PENDING** | Planner recommends Lambda Labs A100 40GB spot (~$0.50/hr). Founder to confirm account setup. |
| 2 | **Context length during pre-training:** 512 vs 1024? | **RESOLVED** | 512 for throughput (§3.3). SFT will use 1024. |
| 3 | **If loss > 3.0 at 1B tokens, extend to 2B?** | **PENDING** | Founder decision at Day 19 checkpoint eval. Extra cost ~$170. |
| 4 | **WandB account:** Use personal or create NeuralAI team? | **PENDING** | Suggest Founder create free WandB team for persistent history. |
| 5 | **Data source for STEM supplement:** ArXiv vs Common Crawl STEM? | **RESOLVED** | ArXiv abstracts (HF `scientific_papers` subset) + Khan Academy transcripts (public domain). |

---

## 14. References

1. **Hoffmann et al. (2022)** — *Training Compute-Optimal Large Language Models* (Chinchilla). Justifies 20 tokens/param ideal; we acknowledge sub-optimal 7.4 tokens/param due to budget.
2. **TinyLlama (2024)** — 1.1B model trained on 3T tokens with LR 4e-4. Validates small-model LR range.
3. **Hu et al. (2022)** — *LoRA: Low-Rank Adaptation of Large Language Models*. Justifies r=32, alpha=64.
4. **Rafailov et al. (2023)** — *Direct Preference Optimization*. Justifies β=0.1 for small-data regime.
5. **NeuralAI Project Charter** — `P-135M-PRETRAIN-v1-CHARTER.md` (budget, timeline, success criteria).
6. **NeuralAI Training Pipeline** — `TRAINING_PIPELINE.md` (existing SFT/DPO hyperparameters, v19 data format).

---

*End of Specification S-001*
