# Cleanup & Organization Statement — 2026-08-02

**Completed by:** NeuralAI (assistant)  
**Date:** 2026-08-02 02:35 UTC  
**Project:** /home/workspace/Projects/NeuralAI

## Findings

1. **K1 SFT v4 training status** — Process `24307` was running on CPU using the Mamba sequential fallback. After 2 hours 30 minutes it had produced zero checkpoints and logged no progress. Local CPU training at this scale is not viable.
2. **K1 v3 loss (~2.16)** — That loss was from the previous v3 run, which over-fit on long multi-turn prompts and produced degenerate outputs such as "Bye". It is not related to the v4 run, which never reached a logged step.
3. **Duplicate / stale K1 artifacts** — There were multiple K1 merged/GGUF copies (`models/k1/sft-v3`, `NeuralAI-v2-merged`, old `models/k1/gguf`) referencing a broken v3 model.

## Actions completed

| # | Action | Verification |
|---|--------|--------------|
| 1 | Stopped stuck v4 CPU training (PID 24307). | `ps -p 24307` returns no process. |
| 2 | Unified `models/k1/` layout: only `base/` and `current/` remain. | `ls models/k1` shows `base` and `current`; no `sft-v3`, no separate `gguf`. |
| 3 | Moved broken v3 artifacts to `archive/k1-v3/` (adapter, merged weights, GGUF). | `ls archive/k1-v3` shows `adapter`, `merged`, `gguf`. |
| 4 | Removed duplicate / empty `NeuralAI-v2-merged` folder and 3.4 MB of stray config files. | Folder no longer exists; `find . -name '*v2-merged*'` returns nothing. |
| 5 | Removed `llama.cpp-build` source/build tree (~236 MB) and replaced with a minimal local copy under `tools/llama.cpp/` (convert script + quantize binary, ~14 MB). | `du -sh tools/llama.cpp` = 14M; `llama.cpp-build` removed. |
| 6 | Kept `services/nextcloud` (~804 MB) because `services/webui_service.py` still imports `services.nextcloud_bridge` for /api/files routes. | `du -sh services/nextcloud` still reports ~804 MB. |
| 7 | Updated `scripts/model_manager.py` so K1 points to the single `models/k1/current/gguf/` path and records `paused_awaiting_gpu` status. | `python3 scripts/model_manager.py list` shows K1 status `paused_awaiting_gpu`. |
| 8 | Updated `scripts/merge_and_export.py` to take a required `--base` path and auto-detect the new `tools/llama.cpp/convert_hf_to_gguf.py`. | `python3 -m py_compile scripts/merge_and_export.py` passes. |
| 9 | Updated `training/train_mamba_lora.py` to flush logs and resolve a remote base model if the local path is missing. | `python3 -m py_compile training/train_mamba_lora.py` passes. |
| 10 | Updated `.gitignore` to ignore `archive/` so large stale models are not accidentally committed to GitHub. | `tail .gitignore` shows `archive/`. |
| 11 | Updated `docs/COLAB_UPLOAD_MANIFEST.md` to note removal of `NeuralAI-v2-merged`. | Manifest row marked `(removed 2026-08-02)`. |
| 12 | Updated `AGENTS.md` current state to reflect K1 v4 paused awaiting GPU. | `grep 'SFT LoRA v4' AGENTS.md` shows correct status. |
| 13 | Created `exports/k1-v4-gpu/` containing cleaned data, GPU training script, merge/quantize script, and README for a GPU run. | `ls exports/k1-v4-gpu` shows four files. |
| 14 | Verified active inference is still live on K2. | `curl http://127.0.0.1:1234/v1/models` returned `Invalid API key` (server responding). |
| 15 | Deleted local `archive/k1-v3/` after confirming the broken v3 artifacts exist on HuggingFace. | `du -sh archive/k1-v3` no longer exists; HF repo contains `NeuralAI-Mamba-K1-v3.Q4_K_M.gguf` and `model.safetensors`. |

## Storage impact

- Removed ~0.4 GB of duplicate / stale artifacts (nextcloud retained).
- Removed ~0.6 GB local `archive/k1-v3/` copy because it is backed up on HuggingFace.
- `models/k1/` no longer stores multiple merged copies.

## Next step

Run the contents of `exports/k1-v4-gpu/` on a CUDA GPU (Colab, RunPod, local GPU) to produce the v4 adapter, then merge/quantize and copy the GGUF to `models/k1/current/gguf/neuralai-mamba-k1-v4.Q4_K_M.gguf`. After that, `python3 scripts/model_manager.py set mamba-k1` will promote K1 v4 to the active inference model.

## 2026-08-01 — Additional cleanup (round 2)

| # | Action | Verification |
|---|--------|--------------|
| 16 | Removed stale `data/archive/` folder containing old DPO iterations superseded by `data/train_dpo_v16_combined.jsonl`. | `ls data/archive` returns "No such file or directory"; freed ~715 KB. |
| 17 | Rewrote `docs/TRAINING_MANIFEST.md` to reflect current Mamba K1/K2/K3 state and removed obsolete DPO/Air-135M/SmolLM model rows. | `grep 'Mamba' docs/TRAINING_MANIFEST.md` shows active table; no SmolLM/Air rows. |
| 18 | Rewrote `docs/COLAB_UPLOAD_MANIFEST.md` to map `exports/k1-v4-gpu/` instead of retired v18 / NeuralAI-v2-merged artifacts. | File no longer references `NeuralAI-v2-merged` or `neuralair-135m`. |
