# 🚀 NeuralAI Development Roadmap

**Version Target: 5.0**
**Last Updated: May 17, 2026**

---

## ✅ Completed Milestones

### Phase 0: Core System ✓
- [x] SmolLM2-360M base model fine-tuned with QLoRA
- [x] Chat streaming (SSE) working
- [x] Web UI deployed at https://neuralai-deandrewharris.zocomputer.io
- [x] SQLite persistence for conversations
- [x] RAG document indexing
- [x] Unified Service Migration: Consolidated Model + UI + Terminal into `neural_core_service.py` (May 17, 2026)
- [x] Fixed Terminal consistency: Aligned `/write` and `/read` endpoints with frontend JS polling
- [x] Fixed Chat streaming: Re-implemented SSE logic in unified service to support live UI updates

### Phase 1: Tool Ecosystem ✓
- [x] Code Execution Sandbox - `run this code: ...`
- [x] File Manager - `search files for ...`
- [x] Web Fetcher - `fetch https://...`
- [x] Database Connector - `show tables` / `query database`
- [x] Git Assistant - `git status` / `git log`
- [x] Tool detection and routing in chat
- [x] Fixed SSE streaming (was sending escaped newlines)

### Phase 2: DPO Alignment ✓
- [x] DPO training pipeline implemented
- [x] Preference dataset expanded to 31 pairs
- [x] Successful DPO training run completed (May 17, 2026)
- [x] Model aligned for better response quality and instruction following

---

## 🔄 Current State

| Component | Status |
| --- | --- |
| **Model** | SmolLM2-360M-Instruct + LoRA + DPO (365M params) |
| **Training Samples** | 404 (347 original + 57 new) |
| **DPO Pairs** | 31 preference pairs |
| **Inference** | CPU float32 (~2-3 sec first token, 5-10 tokens/sec) |
| **Tools** | 5 tools connected and working |
| **Chat** | ✅ Live streaming responses |
| **Eval Suite** | Created, pending execution |
| **Maintenance** | ✅ Unified service stable (May 17) |

---

## 🎯 Next Steps (Priority Order)

### 1. GPU Deployment (Blocked - No GPU on current Zo)
**Goal:** 10x faster inference

| Metric | CPU (Current) | GPU (Target) |
| --- | --- | --- |
| Time to first token | 2-3 sec | 0.1-0.3 sec |
| Tokens/sec | 5-10 | 50-100 |
| Response (100 tokens) | 10-20 sec | 1-2 sec |

**Options:**
- Request GPU from Zo support
- Use Google Colab for training/inference
- Deploy to cloud GPU (RunPod, Lambda Labs)

### 2. Training Data Expansion
**Status:** In progress (404 → target 1000+)

**Categories to expand:**
- Advanced coding: +100 samples
- API design: +50 samples
- DevOps commands: +50 samples
- Multi-turn reasoning: +50 samples
- Tool chaining: +50 samples

### 3. Evaluation Suite
**Status:** Created, pending execution

**Benchmarks:**
- Code correctness: Generated code runs
- Response helpfulness: Quality scoring
- Safety: Refuses harmful requests
- Latency: Inference speed

---

## 📊 Data Files

```
data/
├── train.jsonl              # 347 original samples
├── train_v3.jsonl           # 404 samples (latest)
├── train_dpo.jsonl          # 13 DPO pairs
├── train_dpo_expanded.jsonl # 31 DPO pairs
└── train_expanded.jsonl     # 363 samples
```

## 📁 Project Structure

```
NeuralAI/
├── checkpoints/final_model/    # LoRA adapter
├── data/                       # Training data
├── eval/benchmarks.py          # Evaluation suite
├── from-scratch/web_ui/        # Flask app + static files
│   ├── app.py                  # Main Flask server
│   ├── neuralai_engine.py      # Model + tools
│   └── neuralai_router.py      # Routing logic
├── tools/                      # Tool implementations
│   ├── code_sandbox.py
│   ├── file_manager.py
│   ├── web_fetcher.py
│   ├── db_connector.py
│   └── git_assistant.py
└── training/                   # Training scripts
    ├── train_dpo.py
    ├── generate_training_v3.py
    └── NeuralAI_TPU_Training.ipynb
```

---

## 🔗 Quick Links

- **Live Chat:** https://neuralai-deandrewharris.zocomputer.io
- **GitHub:** https://github.com/Subject-Emu-5259/NeuralAI
- **Local Dev:** http://localhost:5000

---

## 📝 Commands

```bash
# Start the service
cd /home/workspace/Projects/NeuralAI/from-scratch/web_ui
python3 app.py

# Generate training data
python3 training/generate_training_v3.py

# Run evaluation
python3 eval/benchmarks.py

# DPO training (requires GPU)
python3 training/train_dpo.py
```

---

**Next Session Goals:**
1. Run evaluation benchmarks
2. Expand training data to 1000+ samples
3. Request GPU or prepare Colab notebook for DPO training
