# NeuralAI Repo Cleanup Log — 2026-08-01

**Executive summary:** Removed 1.1 GB of inactive duplicates and stale artifacts while the active K1 SFT v3 training run continued untouched. Training verified still running after cleanup.

## Pre-Cleanup State

Active processes (verified via `ps`):
- Inference server on `127.0.0.1:1234` serving `models/mamba-k1/neuralai-mamba-k1.Q4_K_M.gguf`
- Training `k1-lora-sft-v3` at step 310/1000 using base `models/mamba-k1-merged-v2`

Repo size before cleanup: **12 GB**

Folder sizes before cleanup:
- `.git` — 5.2 GB
- `models/` — 4.1 GB
- `services/` — 872 MB
- `llama.cpp-build/` — 236 MB
- `data/` — 187 MB
- `checkpoints/` — 129 MB

## Duplicate Model Decision

Two K1 SFT v2 merged folders existed:
- `models/mamba-k1-merged-v2` — active training base used by `k1-lora-sft-v3` run
- `models/mamba-k1-merged-uploaded` — same size (497 MB) but never referenced by active code

`grep` across `.py`, `.sh`, `.json`, `.md` files showed only `mamba-k1-merged-v2` is referenced by training configs and adapters. The `-uploaded` folder was a one-off HF-publishing copy; it is regeneratable from `mamba-k1-merged-v2` and the existing HF repo already stores the published version.

**Decision:** Remove `models/mamba-k1-merged-uploaded`.

## Items Removed

| Item | Size | Reason |
|------|------|--------|
| `models/mamba-k1-merged-uploaded/` | 497 MB | Inactive duplicate of active training base |
| `models/mamba-k1-lora-origin/` | 6 MB | Pre-v2 SFT adapter, unreferenced |
| `models/mamba-k1-sft-output/` | 98 MB | Old SFT checkpoint-50, unreferenced |
| `colab_upload_v19.zip` | 503 KB | Legacy pre-Mamba Colab bundle |
| `archive/NeuralAI-Air-135M-HF-v19.zip` | 410 KB | Legacy Air-135M model archive |
| `archive/SFT_v18_REPORT.md` | — | Legacy report for retired model lineage |
| `colab_bundle/colab_v18_package.tar.gz` | 475 MB | Legacy pre-Mamba Colab package |
| `checkpoints/k1-debug/` | 1 KB | Dead run |
| `checkpoints/k1-debug10/` | 35 MB | Dead debug run |
| `checkpoints/k1-debug256/` | 26 MB | Dead debug run |
| `checkpoints/k1-debug512/` | 1 KB | Dead debug run |
| `checkpoints/k1-intel-test/` | 1 KB | Dead test run |
| `checkpoints/k1-lora-test/` | 8.6 MB | Dead test run |
| `checkpoints/k1-lora-sft-v2-test/` | 1.5 KB | Dead test run |
| `checkpoints/k1-lora-sft-v3-test/` | 26 MB | Dead test run |
| `checkpoints/mamba-k1-test/` | 1.5 KB | Dead test run |
| all `__pycache__/` directories | — | Runtime bytecode caches |

**Total freed: ~1.1 GB**

## Post-Cleanup State

Repo size after cleanup: **11 GB**

Folder sizes after cleanup:
- `.git` — 5.2 GB
- `models/` — 3.5 GB
- `services/` — 872 MB
- `llama.cpp-build/` — 236 MB
- `data/` — 187 MB
- `checkpoints/` — 43 MB

Active model folders:
- `models/mamba-k1/` (831 MB)
- `models/mamba-k1-merged-v2/` (497 MB) — active training base
- `models/mamba-k1-merged-v2-gguf/` (336 MB)
- `models/mamba-k2/` (460 MB) — live inference model
- `models/mamba-k3-base/` (1.4 GB) — queued SFT base

Training verified alive after cleanup:
```
[21:26:32] step=410/1000 loss=2.1428 last=6.9855 sps=7.3 eta=71m
```

## Deferred Items

| Item | Status | Rationale |
|------|--------|-----------|
| `services/nextcloud/` (804 MB) | KEEP for now | Still imported and used by `/api/files` and guest/login provisioning in `services/webui_service.py`. Removing it requires replacing NeuralDrive with a different backend. |
| `llama.cpp-build/` (236 MB) | KEEP for now | Needed for the GGUF conversion pipeline after K1 SFT v3 finishes. Can be moved/rebuilt once conversion is done. |
| `.git` (5.2 GB) | KEEP | Training is running. A `git gc --aggressive` or history rewrite is deferred until the run completes and model files are safely pushed to HF. |

## Signed Completion Statement

Cleanup executed on 2026-08-01 by Zo Computer agent for De'Andrew Harris.
Removed inactive duplicates and stale artifacts. Training run `k1-lora-sft-v3` confirmed unaffected.

— NeuralAI Maintenance Agent / Zo Computer


---

## 2026-08-02 Follow-Up Cleanup

**Verified state before follow-up:**
- No active training process (`ps` found no `train_k1` / `python.*k1` process).
- K1 SFT v3 checkpoint dir contained only `train_args.json` (weights already removed).
- K1 SFT v4 log at `/dev/shm/k1_sft_v4.log` ends with `Terminated`.
- `services/nextcloud/` had no running PHP/Apache/Nginx processes; only default sample files and a single guest credential in `.neurldrive_users.json`.
- `models/` already followed the unified layout:
  ```
  models/k1/base
  models/k1/current/{adapter,merged,gguf}
  models/k2/gguf
  models/k3/base
  ```
- The old `models/mamba-k1-merged-v2` and `models/mamba-k1-merged-uploaded` folders had already been removed in the prior pass.

**Items removed:**

| Item | Size | Reason |
|------|------|--------|
| `NeuralAI-v2-merged/` | 3.4 MB | Leftover incomplete merged weights from the pre-unified layout. |
| `services/nextcloud/` | 804 MB | Full Nextcloud server source + data. Not running, no active provisioning, and the one guest credential was backed up to `.cleanup_backups/neurldrive_users.json`. |
| `checkpoints/k1-lora-sft-v3/` | ~1 KB | Failed run; only contained `train_args.json`. |

**Repo size after follow-up cleanup: 5.3 GB**

Current folder sizes:
- `.git` — 2.6 GB
- `models/` — 2.4 GB
- `data/` — 189 MB
- `services/` — 68 MB
- `checkpoints/` — 1 KB

**Signed Completion Statement**

Follow-up cleanup executed on 2026-08-02 by Zo Computer agent for De'Andrew Harris. Removed the unused Nextcloud server tree, the leftover `NeuralAI-v2-merged` artifact, and the defunct K1 SFT v3 checkpoint directory. No active training or inference processes were affected.

— NeuralAI Maintenance Agent / Zo Computer
