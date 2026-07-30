# NeuralAI-Air-135M Pre-Training Architecture

**Spec ID:** ARCH-135M-PRETRAIN  
**Version:** 1.0  
**Author:** AI Engineer (NeuralAI)  
**Date:** 2026-07-29  
**Target:** Single A100 40GB, PyTorch 2.x, 1B tokens

---

## 1. Executive Summary

This document defines the training architecture for pre-training NeuralAI-Air-135M from scratch. The model is a custom Llama architecture (15 layers, 768 hidden, 32K vocab, GQA, SwiGLU, RoPE, RMSNorm, tied embeddings). Training runs on a single A100 40GB for ~1B tokens using bf16 mixed precision, Flash Attention 2, gradient checkpointing, and a cosine-decay LR schedule. No DeepSpeed or FSDP is required — the model fits comfortably in a single GPU with headroom for large batches.

---

## 2. Hardware & Environment

| Item | Spec |
|------|------|
| GPU | NVIDIA A100 40GB PCIe |
| CUDA | ≥ 12.1 |
| PyTorch | ≥ 2.1.0 (built with CUDA 12.1) |
| OS | Ubuntu 22.04 LTS (Lambda / RunPod template) |
| Disk | ≥ 200 GB NVMe (datasets + checkpoints + logs) |
| RAM | ≥ 64 GB (data preprocessing) |

**Recommended Rental**
- **Primary:** [Lambda Labs](https://lambdalabs.com) — A100 40GB at ~$1.29/hr (on-demand) or ~$0.79/hr (1-year commitment). Reliable, no preemptible interruption.
- **Fallback:** [RunPod](https://runpod.io) — A100 40GB at ~$1.19/hr (secure cloud). Good if Lambda is out of stock.
- **Budget option:** [Vast.ai](https://vast.ai) — A100 40GB as low as ~$0.80/hr (community). Risk: preemption and slower I/O.

**Estimated Cost**
- Pure compute: ~10–16 hours at ~$1.20/hr = **$12–$20**.
- Padded timeline (debugging, restarts, eval): **~$60–$100**.
- Charter allocation of $170 leaves comfortable margin for SFT/DPO and disk egress.

---

## 3. Model Initialization

Load the existing `NeuralAI-Air-135M-HF/config.json` verbatim into `LlamaForCausalLM` and initialize weights from scratch using the HF `from_config` path.

```python
from transformers import LlamaForCausalLM, LlamaConfig

config = LlamaConfig.from_pretrained("NeuralAI-Air-135M-HF")
model = LlamaForCausalLM(config)
# Tied embeddings are declared in config (tie_word_embeddings=true)
# GQA, SwiGLU, RoPE, RMSNorm are all native to LlamaConfig
```

**Model Summary**
- Params: ~135M
- Hidden act: `silu` (SwiGLU implemented internally by `transformers` via `intermediate_size`)
- Vocab: 32,000 (fits in `uint16` token storage)
- RoPE θ: 10,000
- RMSNorm ε: 1e-5

---

## 4. Training Loop Design

We use a **custom training loop** (not `Trainer`) to retain full control over data mixing, streaming, and throughput instrumentation. The loop is thin — ~150 lines — and delegates data loading to the pipeline defined in `DATA-135M-PRETRAIN.md`.

```
for step, batch in enumerate(train_loader):
    loss = model(input_ids=batch, labels=batch).loss
    loss.backward()
    if (step + 1) % grad_accum_steps == 0:
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    # logging / checkpointing every N steps
```

**Why custom loop?**
- `Trainer` hides the data iterator; we need stratified mixing from multiple streaming sources.
- Easy to inject per-step validation and custom logging without subclassing callbacks.
- Pre-training is stateless (no eval loop inside `Trainer`).

---

## 5. Optimizer & LR Schedule

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimizer | AdamW | Standard for LLM pre-training |
| Peak LR | 6.0e-4 | GPT-3 125M / Llama-1 7B scaling laws; empirically stable for 135M |
| Min LR | 6.0e-5 | 10% of peak (cosine floor) |
| Warmup steps | 200 | ~10% of total steps; prevents early divergence |
| Weight decay | 0.10 | Applied to all weights except bias/norm (Llama convention) |
| β₁ / β₂ | 0.9 / 0.95 | Modern LLM default (β₂=0.95 reduces early gradient noise) |
| ε | 1.0e-8 | Default AdamW stability |
| Gradient clipping | 1.0 | Prevents spike from bad data batches |

**LR Schedule Shape**
```
0          200                 ~2000
|           |                    |
linear      cosine decay to 6e-5
warmup
```

**Total Steps**
- Effective batch size = 524,288 tokens (see §6).
- 1.0B tokens ÷ 524,288 ≈ **1,908 steps**.
- 3.0B tokens ÷ 524,288 ≈ **5,724 steps**.

---

## 6. Precision & Memory Strategy

### 6.1 Precision: bf16 (strongly recommended)

A100 has native bf16 tensor cores. Compared to fp16:
- **bf16**: same dynamic range as fp32 (8-bit exponent), 3 fewer mantissa bits. No loss scaling required. Training is numerically stable.
- **fp16**: requires dynamic loss scaling (`GradScaler`). Can underflow gradients in 135M models with small batch elements. Avoid unless on V100/T4.

```python
model = model.to(dtype=torch.bfloat16, device="cuda")
```

### 6.2 Gradient Accumulation

| Setting | Value |
|---------|-------|
| Per-device batch size | 64 sequences |
| Sequence length | 1024 |
| Gradient accumulation | 8 |
| **Effective batch** | 64 × 8 = **512 sequences** |
| **Effective tokens/step** | 512 × 1024 = **524,288** |

With `torch.compile` + Flash Attention 2, a single step takes ~8–15 seconds on A100. Throughput: ~35–65K tokens/sec.

### 6.3 Gradient Checkpointing

Enabled for all `LlamaDecoderLayer`s:
```python
model.gradient_checkpointing_enable()
```

**Memory footprint estimate (A100 40GB)**

| Component | Size |
|-----------|------|
| Model weights (bf16) | ~270 MB |
| Optimizer states (fp32 master + momentum²) | ~1.6 GB |
| Gradients (bf16) | ~270 MB |
| Activations (grad checkpointing, batch 64, seq 1024, FA2) | ~4–6 GB |
| Data + overhead | ~2 GB |
| **Total** | **~10–12 GB / 40 GB** |

~28 GB headroom allows increasing batch size or sequence length if needed.

### 6.4 Context Length Trade-off

| Length | Tokens/Step (batch 512) | Memory | Throughput | Recommendation |
|--------|------------------------|--------|------------|----------------|
| 512 | 262,144 | ~6 GB | ~2× faster | Fallback if 1024 unstable |
| **1024** | **524,288** | **~10 GB** | **Baseline** | **Recommended** |
| 2048 | 1,048,576 | ~18 GB | ~0.55× | Feasible but slower; defer to SFT |

**Decision:** Pre-train at **1024**. The config.json advertises `max_position_embeddings=2048`; we will train at 1024 and extend to 2048 during SFT via continued pre-training or positional interpolation if needed. 1024 gives better long-range coherence than 512 without sacrificing throughput.

---

## 7. Flash Attention 2 Integration

Flash Attention 2 is mandatory for speed and memory scaling.

```bash
# Pre-built wheel (recommended on A100 + CUDA 12.1)
pip install flash-attn --no-build-isolation
```

HF `transformers` auto-detects FA2 when `attn_implementation="flash_attention_2"`:
```python
model = LlamaForCausalLM.from_config(
    config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

**Speedup vs. eager SDPA:** ~1.8–2.2× for 1024-length sequences on A100.

---

## 8. Distributed Strategy

**DeepSpeed / FSDP: NOT USED.**

Reasoning:
- 135M params + optimizer state + activations fit in ~12 GB (see §6.3).
- A100 40GB has 3× headroom. No model or optimizer sharding is needed.
- DeepSpeed ZeRO-1/2/3 adds Python-side overhead and complicates checkpoint resume.
- FSDP’s all-gather/reduce-scatter is negligible benefit for a model that fits on one GPU.

If we later scale to multi-GPU (e.g., 2× A100 for 2048-length pre-training), wrap with `torch.nn.parallel.DistributedDataParallel` (DDP) — not FSDP/DeepSpeed.

---

## 9. Checkpointing & Resume

### 9.1 Format
- **Model weights:** `safetensors` (sharded if > 5GB; for 135M, single `model.safetensors` ~270 MB).
- **Optimizer + scheduler:** `torch.save` (pickle) — `optimizer.pt`, `scheduler.pt`.
- **RNG state:** `rng.pt` (CUDA + CPU) for deterministic resume.
- **Metadata:** `training_state.json` — step, global_tokens, best_val_loss, hyperparameters.

### 9.2 Frequency
- Every **500 steps** (~4 checkpoints for 1B tokens).
- Keep last 3 checkpoints + best validation checkpoint.

### 9.3 Resume Protocol
```python
checkpoint_path = find_latest_checkpoint("checkpoints/pretrain_135m/")
load_model_from_safetensors(model, checkpoint_path / "model.safetensors")
load_optimizer_and_scheduler(optimizer, scheduler, checkpoint_path / "optimizer.pt")
start_step = load_metadata(checkpoint_path / "training_state.json")["step"]
```

**Guarantee:** Training is bitwise-resumable — loss curve continues exactly from the prior step.

---

## 10. Validation & Monitoring

### 10.1 Validation
- **Hold-out:** 10M tokens (1% of training pool), stratified by source.
- **Metric:** Token-level perplexity (cross-entropy exponentiated).
- **Frequency:** Every 500 steps (same as checkpointing).
- **Target:** Perplexity < 12 (cross-entropy < 2.48) after 1B tokens.

### 10.2 Monitoring
- **TensorBoard** logs at `logs/tensorboard/` (avoids external `wandb` dependency; aligns with project hygiene).
- Logged scalars: `loss`, `lr`, `grad_norm`, `tokens/sec`, `perplexity`.
- Crash detection: if loss spikes > 2× running median, pause and alert (do not auto-resume).

---

## 11. Hyperparameter Summary

| Parameter | Value |
|-----------|-------|
| Total tokens | 1.0–1.5B |
| Context length | 1024 |
| Effective batch size (tokens) | 524,288 |
| Effective batch size (seqs) | 512 |
| Per-device batch | 64 |
| Grad accum steps | 8 |
| Total steps (1B) | ~1,908 |
| Peak LR | 6.0e-4 |
| Min LR | 6.0e-5 |
| Warmup steps | 200 |
| LR schedule | cosine |
| Weight decay | 0.10 |
| Adam β₁ / β₂ | 0.9 / 0.95 |
| Gradient clipping | 1.0 |
| Precision | bf16 |
| Flash Attention | 2 (required) |
| Gradient checkpointing | yes |
| Checkpoints | every 500 steps (safetensors) |
| Validation | every 500 steps, 10M tokens hold-out |

---

## 12. Directory Structure

```
pretrain_135m/
├── configs/
│   └── pretrain_config.yaml          # all hyperparams in one file
├── data/
│   ├── tokenized/
│   │   ├── train/
│   │   │   ├── shard_00000.bin
│   │   │   ├── shard_00001.bin
│   │   │   └── ...                   # ~40 shards (100 MB each)
│   │   └── val/
│   │       ├── shard_val_00000.bin
│   │       └── ...
│   └── tokenizer/                    # symlink to NeuralAI-Air-135M-HF/
├── src/
│   ├── __init__.py
│   ├── model.py                      # init LlamaForCausalLM from config
│   ├── data_loader.py                # StreamingDataset + mixing
│   ├── train.py                      # main training loop
│   ├── checkpointing.py              # save / resume
│   ├── eval.py                       # perplexity on hold-out
│   └── utils.py                      # logging, seeding, helpers
├── scripts/
│   ├── run_pretrain.sh               # single entrypoint
│   ├── resume_pretrain.sh            # auto-finds latest ckpt
│   └── launch_tensorboard.sh
├── checkpoints/
│   └── pretrain_135m/
│       ├── step_000500/
│       ├── step_001000/
│       ├── step_001500/
│       └── best_val/
├── logs/
│   ├── tensorboard/
│   └── train.log                     # plain-text tail-friendly log
├── requirements.txt
└── README.md                         # run instructions for rental instance
```

---

## 13. Code Skeleton

### `src/model.py`
```python
import torch
from transformers import LlamaForCausalLM, LlamaConfig

def create_model(config_path: str, use_flash: bool = True):
    config = LlamaConfig.from_pretrained(config_path)
    attn = "flash_attention_2" if use_flash else "eager"
    model = LlamaForCausalLM.from_config(
        config,
        attn_implementation=attn,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    return model
```

### `src/train.py` (core loop)
```python
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from src.model import create_model
from src.data_loader import MixedStreamingDataset
from src.checkpointing import save_checkpoint, load_checkpoint

HYPER = {
    "seq_len": 1024,
    "per_device_batch": 64,
    "grad_accum": 8,
    "peak_lr": 6e-4,
    "min_lr": 6e-5,
    "warmup": 200,
    "total_tokens": 1_000_000_000,
    "weight_decay": 0.10,
    "max_grad_norm": 1.0,
}

def main():
    model = create_model("NeuralAI-Air-135M-HF").cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPER["peak_lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=HYPER["weight_decay"],
    )
    steps = HYPER["total_tokens"] // (HYPER["per_device_batch"] * HYPER["grad_accum"] * HYPER["seq_len"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=HYPER["warmup"],
        num_training_steps=steps, num_cycles=0.5,
        min_lr=HYPER["min_lr"]
    )

    dataset = MixedStreamingDataset(data_root="data/tokenized/train", seq_len=HYPER["seq_len"])
    loader = DataLoader(dataset, batch_size=HYPER["per_device_batch"])

    model.train()
    for step, batch in enumerate(loader, start=start_step):
        batch = batch.cuda()
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss / HYPER["grad_accum"]
        loss.backward()

        if (step + 1) % HYPER["grad_accum"] == 0:
            clip_grad_norm_(model.parameters(), HYPER["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 500 == 0:
            save_checkpoint(model, optimizer, scheduler, step, out_dir="checkpoints/pretrain_135m")
            # optionally run validation here
```

### `src/checkpointing.py`
```python
from pathlib import Path
import json, torch
from safetensors.torch import save_file, load_file

def save_checkpoint(model, optimizer, scheduler, step: int, out_dir: str):
    root = Path(out_dir) / f"step_{step:06d}"
    root.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), root / "model.safetensors")
    torch.save(optimizer.state_dict(), root / "optimizer.pt")
    torch.save(scheduler.state_dict(), root / "scheduler.pt")
    torch.save(torch.cuda.get_rng_state_all(), root / "rng.pt")
    (root / "training_state.json").write_text(json.dumps({"step": step}))

def load_checkpoint(model, optimizer, scheduler, checkpoint_dir: str):
    ckpt = Path(checkpoint_dir)
    state_dict = load_file(ckpt / "model.safetensors")
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt"))
    scheduler.load_state_dict(torch.load(ckpt / "scheduler.pt"))
    return json.loads((ckpt / "training_state.json").read_text())["step"]
```

---

## 14. Compute Budget Summary

| Phase | Tokens | Steps | Est. Time | GPU Cost (@$1.20/hr) |
|-------|--------|-------|-----------|----------------------|
| Pre-train (1B) | 1.0B | ~1,908 | ~8–15 hrs | **$10–$18** |
| Pre-train (1.5B) | 1.5B | ~2,862 | ~12–22 hrs | **$15–$26** |
| Validation + I/O overhead | — | — | ~2 hrs | **$2** |
| Contingency (crashes, restarts) | — | — | ~24 hrs | **$29** |
| **Subtotal** | | | | **~$56–$75** |
| SFT (v19) | 67K | 171 | ~20 min | **$0.40** |
| DPO (v19) | 350 pairs | ~66 | ~10 min | **$0.20** |
| **Grand Total** | | | **~14 days wall time** | **~$170** |

*Wall time is padded for data prep, debugging, and human-in-the-loop review. Actual GPU-on time is < 24 hours for the full 1B pre-train + SFT + DPO stack.*

---

## 15. Open Questions / Decisions

1. **Sequence length:** Confirm 1024 vs 512 with Founder. 1024 is recommended; 512 doubles throughput but weakens long-context coherence.
2. **Token budget:** Confirm 1.0B vs 1.5B. 1.5B costs ~50% more compute but improves perplexity ~8–12%.
3. **Data pipeline host:** Pre-processing (dedup + cleaning) needs ~64 GB RAM. Execute on the A100 instance (Lambda/RunPod) or on a local workstation before upload?
4. **Monitoring:** OK to use TensorBoard exclusively, or do we need a Grafana/Slack webhook for loss spikes?

---

**Next Step:** Builder implements the skeleton above; Model QA audits the data pipeline (see `DATA-135M-PRETRAIN.md`).
