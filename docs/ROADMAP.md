# 🚀 NeuralAI Development Roadmap

**Version Target: 6.0 (The Workstation Pivot)**
**Last Updated: May 19, 2026**

---

## ✅ Completed Milestones (NeuralAI Legacy)

### Phase 0: Core System ✓
- [x] SmolLM2-360M base model fine-tuned with QLoRA
- [x] Chat streaming (SSE) working
- [x] Web UI deployed (NeuralAI → NeuralAI v1.0)
- [x] Unified Service Migration: Consolidated Model + UI + Terminal into `neural_core_service.py`
- [x] Fixed Chat streaming and Terminal consistency

### Phase 1: Tool Ecosystem ✓
- [x] Code Execution Sandbox, File Manager, Web Fetcher, DB Connector, Git Assistant
- [x] Tool detection and routing in chat

### Phase 2: DPO Alignment ✓
- [x] DPO training pipeline implemented
- [x] Preference dataset expanded to 244 pairs (v12.0)
- [x] Memphis Culture & Founder Context integration

---

## 🏗️ Phase 3: NeuralAI Evolution (In Progress)

### 1. Workstation Orchestration
- [x] Establish distinction: NeuralAI (Model) vs. NeuralAI (Hub)
- [x] UI Overhaul: Added "Workstation Dashboard" tab with project/model/shell status
- [x] Robust Multi-Turn Context Support (10-message sliding window)
- [x] Integrated Multi-Modal Speech-to-Speech (Gemini Live + ElevenLabs Fallback)
- [ ] Transition UI from Chat-Only to Multi-Panel Workstation (Expand dashboard features)
- [ ] Implement System-Wide Context Layer
- [ ] Add "Vibe Stack" Workflow Registry

### 2. Neural Knowledge Graph
- [ ] Implement Persistent Memory (Graph-based)
- [ ] Automate Infrastructure Learning
- [ ] Sync with Zo Supermemory

---

## 📊 System Status
- **Main Service:** **READY** (Unified `neural_core_service.py`)
- **Voice Service:** **READY** (ElevenLabs v2 Migrated)
- **Model:** SmolLM2-360M-Instruct + DPO v13.0
- **Context:** System-wide (Expanding)


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

# Generate v5 DPO data
python3 training/generate_dpo_v5.py

# DPO training (currently running in background)
python3 training/train_dpo.py
```

---

**Next Session Goals:**
1. Run evaluation benchmarks
2. Expand training data to 1000+ samples
3. Request GPU or prepare Colab notebook for DPO training
