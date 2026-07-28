# NeuralAI Local Cleanup Audit — 2026-07-25

Keep directive from user:
- KEEP: SmolLM2-360M base runtime (via `services/neuralai_llama_server.sh` → LM Studio GGUF)
- KEEP: NeuralAI-Air-135M (own base model work + training scripts)
- KEEP: Web UI (`from-scratch/web_ui/` + `services/webui_service.py` + required tool chain)
- KEEP: only the *new* training for the base model (Air-135M colab + scripts)
- DELETE: everything else

## Kept (after cleanup)
- `AGENTS.md`, `README.md`
- `run_service.sh`
- `from-scratch/web_ui/` — kept live templates/static; removed stale old Python UI code and `space_fix/`
- `services/webui_service.py`, `services/neuralai_llama_server.sh`, `services/reminder_daemon.py`, `services/neural_voice/`, `services/nextcloud_bridge.py`
- `tools/` — kept only the live tool chain imported by `tool_handler.py` / `webui_service.py`
- `data/neuralai.db`, `data/release_notes.json`, `data/conversations.json`
- `NeuralAI-Air-135M/` (own base model)
- `colab/NeuralAI_Air_135M_pretrain.ipynb` (new training plan)

## Deleted in this cleanup
| Path | Size (approx) |
|------|---------------|
| `adapter/` (D17 old LoRA adapter — already on HF) | 38 MB |
| `models/NeuralAI-v17-lora.gguf` (old merged export) | 34 MB |
| `.github/` | 9 KB |
| `docs/` | 88 KB |
| `knowledge_base/` | 16 KB |
| `neural-brain/` | 43 KB |
| `diffusion_toy/` | 4 KB |
| `scripts/` (deploy_v16, llmster_auto_load, orchestrator proto, HF sync, zo_cleanup) | 20 KB |
| `data/train_dpo_v*.jsonl`, `data/train_sft_v16.jsonl`, `data/6ce093be.json`, `data/indexed_files.json` | ~480 KB |
| Stale service files: `services/diffusion_engine.py`, `services/fix_service.py`, `services/neural_cloud_client.py`, `services/run_safe.sh`, `services/start_all.sh`, `services/status.sh`, `services/stop_all.sh`, `services/storage_service.py` | ~100 KB |
| Unused tool files: `tools/browser_proxy.py`, `tools/web_surf_agent.py`, `tools/memory_graph.py`, `tools/agent_runner.py`, `tools/_tool_layer.py`, `tools/neural_layout/`, `services/_tool_layer.py` | ~100 KB |
| Old Web UI files: `from-scratch/web_ui/app.py`, `neuralai_engine.py`, `neuralai_router.py`, `neuralai_terminal.py`, `rag.py`, `terminal.py`, `terminal_ws.py`, `tools_api.py`, `space_fix/`, old `main.js`/`browser.js` | ~700 KB |
| Other root files: `train_d17_dpo.py`, `merge_lora.py`, `neuralair_distillation_loss.py`, `upload_folder.py`, `upload_space.py`, `_jscheck.py`, `neuralai_banner.svg`, `render.yaml`, `wrangler.jsonc`, `.env.browser.example`, `.dockerignore`, `COMPANY.md`, `IMPLEMENTATION_PLAN.md`, `MODEL_ALIGNMENT.md`, `MODEL_CARD.md`, `ORCHESTRATOR.md`, `RESPONSE_GENERATION_IMPROVEMENTS.md` | ~200 KB |

## Remaining large item flagged separately
- `.git/` contains the full repository history and takes ~642 MB. It does not belong to the base model / training / UI runtime. Recommend deleting it after pushing any important state, or keep it if history matters.

## Notes
- Live service `neuralai-web-ui` was not restarted. Imports verified before cleanup.
