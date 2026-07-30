# PROJECT CHARTER: NeuralAI-Air-135M Pre-Training

**Project ID:** P-135M-PRETRAIN-v1  
**Classification:** EPIC  
**Status:** CHARTER (Pending Founder Approval)  
**CO CEO:** Active oversight  
**Founder:** De'Andrew Harris

---

## Executive Summary

NeuralAI-Air-135M is a custom Llama-architecture model (15 layers, 768 hidden, 32K vocab, GQA, ~135M params). Current checkpoint `final.pt` has not been pre-trained — it was trained on only 500 highly duplicated SFT examples. This project will execute **full pre-training from scratch** on 1–3 billion tokens of quality text, followed by SFT (v19 data) and DPO alignment.

**Goal:** Produce a coherent, instruction-following 135M model that can serve as NeuralAI's lightweight inference engine.

---

## Scope

### IN SCOPE
- Data pipeline: acquire, clean, tokenize 1B+ tokens
- Pre-training script with proper optimizations (flash attn, gradient checkpointing, mixed precision)
- Training on quality corpora (C4 + Wikipedia + OpenWebText + Code)
- SFT phase on v19 dataset (1016 examples)
- DPO phase on v19 preferences (350 pairs)
- GGUF conversion for llama.cpp inference
- Model registry update for production deployment

### OUT OF SCOPE
- Multi-node distributed training (single GPU only)
- RLHF (DPO only)
- Vision/multimodal capabilities
- Quantization beyond GGUF Q4_K_M

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Pre-training loss | < 2.5 after 1B tokens |
| SFT loss | < 1.5 after 3 epochs |
| DPO reward accuracy | > 85% |
| Inference coherence | Generates grammatically correct English |
| Identity accuracy | Correctly names De'Andrew Harris as creator |
| Safety refusal | Refuses harmful requests 100% of time |

---

## Resource Requirements

### Compute

| Stage | Tokens | Est. Steps | GPU | Time | Cost |
|-------|--------|-----------|-----|------|------|
| Pre-train | 1.0B | ~195K | A100 40GB | ~14 days | ~$170 |
| SFT | 67K | 171 | A100 40GB | ~20 min | ~$0.17 |
| DPO | 350 pairs | ~66 | A100 40GB | ~10 min | ~$0.08 |
| **Total** | | | | **~14 days** | **~$170** |

*Alternative: T4 (free Colab) would take ~60-90 days for pre-training. Not recommended.*

### Data

- **Source**: C4 (English), Wikipedia (en), OpenWebText, StackExchange, Books
- **Target size**: 1.0–1.5 billion tokens
- **Format**: Tokenized to `input_ids` with custom tokenizer
- **Storage**: ~4GB raw text, ~8GB tokenized

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Insufficient compute budget | High | Blocks project | Use Lambda Labs / RunPod; approve budget first |
| Data quality issues | Medium | Poor model | Aggressive dedup + filtering; Model QA audit |
| Training instability | Medium | Wasted compute | Gradient clipping, LR warmup, checkpoint every 1K steps |
| Tokenizer mismatch | Low | Broken inference | Verify vocab alignment before training starts |
| Founder loses patience | Medium | Project cancelled | Weekly progress reports; demos every 3 days |

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Charter approval (Founder) | 1 day | This document |
| Data pipeline build | 2 days | Charter approved |
| Data acquisition + tokenization | 3 days | Pipeline ready |
| Pre-training | 14 days | Data ready, GPU rented |
| Checkpoint evaluation | 2 days | Pre-training done |
| SFT training | 1 day | Pre-trained checkpoint |
| DPO training | 1 day | SFT done |
| GGUF conversion + deploy | 1 day | DPO done |
| **Total** | **~25 days** | |

---

## Agent Assignment

```
CO CEO (You) → Oversees all, reports to Founder
    ├── Planner → Full training spec, data plan, hyperparameter search space
    ├── AI Engineer → Architecture decisions, training script, optimization
    ├── Builder → Implement data pipeline, training loops, checkpointing
    ├── Model QA → Data audit, training monitoring, eval harness
    └── Reviewer → Code review, security, performance audit
```

---

## Budget Request

**Founder approval needed for:**
1. GPU rental budget: $170 (A100 for 14 days)
2. Data storage: $0 (local / Drive)
3. Total project cost: $170 + time

**Alternative (no budget):**
- Use free T4 Colab: ~60-90 days pre-training
- Risk: Colab disconnects, session limits, much slower iteration
- Not recommended for production timeline

---

## Immediate Next Steps (Pending Approval)

1. **Founder approves charter** → CO CEO delegates to Planner
2. **Planner creates full spec** → AI Engineer reviews architecture
3. **AI Engineer designs training system** → Builder implements
4. **Model QA audits data** → Pre-training begins
5. **Weekly demos to Founder** → Adjust based on loss curves

---

## Decision Required from Founder

**De'Andrew, I need your decision on:**

1. **Budget approval**: $170 for A100 GPU rental (14 days)?
2. **Timeline acceptance**: ~25 days total (14 days pre-training)?
3. **Data sources**: OK to use C4 + Wikipedia + OpenWebText? (all public domain)
4. **Go/No-Go**: Do we proceed with full pre-training, or pivot to pre-trained base model?

Reply with:
- "Approved — proceed with Option A" (A100 budget)
- "Approved — proceed with Option B" (free Colab, 60-90 days)
- "Denied — pivot to pre-trained base model"
- Or ask questions / modify scope
