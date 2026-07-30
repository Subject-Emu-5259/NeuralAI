# CO CEO DECISION MEMO: NeuralAI-Air-135M Pre-Training Integration

**From:** CO Founder CEO & Supervisor  
**To:** De'Andrew Harris (Founder)  
**Re:** Resolution of Planner + AI Engineer specs, ready for Builder  
**Date:** 2026-07-30

---

## Status

Both specialist agents completed their specs:
- **Planner** (`S-001`) — Full training pipeline, hyperparameters, evaluation gates, timeline
- **AI Engineer** (`ARCH-135M-PRETRAIN.md` + `DATA-135M-PRETRAIN.md`) — Architecture, data pipeline, dedup strategy, streaming loader

I reviewed both. Three conflicts required CEO arbitration:

---

## Conflict 1: Context Length During Pre-Training

| Source | Proposal | Rationale |
|--------|----------|-----------|
| Planner | **512** | 2× throughput, more steps = more parameter updates |
| AI Engineer | **1024** | Better coherence, single pipeline |

**CEO Decision: 512 for pre-training, 1024 for SFT**

- Pre-training at 512 yields ~6,000–8,000 tokens/sec on A100 with FA2
- SFT at 1024 uses full model capacity for instruction following
- This is standard practice (TinyLlama 1.1B pre-trained at 2048, SFT at 2048; we just swap the split)
- Model supports 2048 natively; extending during SFT is trivial

**Risk accepted:** Slightly weaker long-range coherence in base model, but SFT teaches task structure anyway.

---

## Conflict 2: Batch Size / Gradient Accumulation / Steps

| Source | Batch | Grad Accum | Context | Tokens/Step | Steps for 1B |
|--------|-------|-----------|---------|-------------|--------------|
| Planner | 64 | 1 | 512 | 32,768 | **30,500** |
| AI Engineer | 64 | 8 | 1024 | 524,288 | **1,908** |

**CEO Decision: Planner's numbers (batch 64, no grad accum, 512 ctx, 30,500 steps)**

- 30,500 steps means 30,500 optimizer updates for 135M params
- 1,908 steps means only 1,908 updates — **far too few for a model this small to learn**
- Small models benefit from many small updates, not few huge ones (Hoffmann et al. 2022)
- Gradient accumulation adds complexity for no benefit on a single A100 with this model size

**Revised compute estimate:**
- Throughput: ~6,000–8,000 tokens/sec at 512 + FA2
- 30,500 steps × 32,768 tokens = 1B tokens
- Wall time: **~42–56 hours pure compute**
- With checkpointing, validation, debugging, restarts: **~14 days**
- Budget: **~$50–$70** (not $170) if we run efficiently

---

## Conflict 3: Checkpointing Frequency

| Source | Save Every | Val Every |
|--------|-----------|-----------|
| Planner | 2,000 steps | 5,000 steps |
| AI Engineer | 500 steps | 500 steps |

**CEO Decision: 1,000 steps save, 2,000 steps validate**

- 500-step saves are too aggressive — each save is ~1.5GB, disk fills fast
- 2,000-step saves risk losing 4–5 hours of work on spot preemption
- 1,000-step saves = ~2–3 hours of work per checkpoint, reasonable
- Validation every 2,000 steps = ~3× per day, enough to catch divergence

---

## Unified Pre-Training Hyperparameters (CEO-Approved)

| Parameter | Value | Owner |
|-----------|-------|-------|
| Context length | 512 | CEO decision |
| Per-device batch | 64 | Planner |
| Gradient accumulation | 1 (disabled) | CEO decision |
| Effective tokens/step | 32,768 | — |
| Total steps (1B tokens) | 30,500 | — |
| Peak LR | 6.0e-4 | Planner |
| LR schedule | Cosine decay to 6.0e-5 | Planner |
| Warmup steps | 300 | Planner |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) | Planner |
| Weight decay | 0.1 | Planner |
| Gradient clipping | 1.0 | Planner |
| Mixed precision | bf16 | AI Engineer |
| Flash Attention | FA2 (mandatory) | AI Engineer |
| Gradient checkpointing | Enabled | AI Engineer |
| torch.compile | Enabled (PyTorch ≥2.1) | AI Engineer |
| Dropout | 0.0 | Planner |
| Seed | 42 | Planner |
| Checkpoint save | Every 1,000 steps | CEO decision |
| Validation | Every 2,000 steps | CEO decision |
| Best checkpoint kept | 3 (by eval PPL) | Planner |

---

## Data Mix (CEO-Approved)

| Source | % | Target Tokens | HF Dataset |
|--------|---|--------------|------------|
| C4 (en) | 30% | 300M | `c4` streaming |
| Books (Gutenberg) | 30% | 300M | `HuggingFaceM4/ProjectGutenberg` |
| OpenWebText | 20% | 200M | `Skylion007/openwebtext` |
| Wikipedia (en) | 10% | 100M | `wikimedia/wikipedia` |
| StackExchange | 10% | 100M | `HuggingFaceTB/stackexchange` |
| **Total** | **100%** | **~1.0B post-dedup** | |

---

## GPU Provider Recommendation (CEO-Approved)

**Lambda Labs A100 40GB** — ~$1.29/hr on-demand, no preemption, reliable.

Alternative: **RunPod** (~$1.19/hr) if Lambda out of stock.

Expected cost at 56 hours pure compute: **~$72**. Well under the $170 budget.

---

## Next Action

Once you approve this memo, I will immediately delegate to **Builder** to implement:
1. `training/pretrain/` — data pipeline + pre-training script
2. `training/sft/` — SFT script (reusing v19 data)
3. `training/dpo/` — DPO script (reusing v19 preferences)
4. `training/common/` — shared utilities

Then **Model QA** audits the data pipeline before any training begins.

**Builder ETA:** ~2 days to implement + test all scripts locally (on CPU, not training).

---

## Decision Required

**De'Andrew — approve the integrated spec so Builder can start?**

Reply:
- **"Approved — Builder go"** → I delegate immediately, scripts built in 2 days
- **"Approved — but use 1024 context"** → Override CEO decision, AI Engineer's numbers
- **"Modify X"** → Tell me what to change
- **"Deny — too expensive / too long"** → We pivot to pre-trained base model

**Awaiting your call, Founder.**
