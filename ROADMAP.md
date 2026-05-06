# 🚀 NeuralAI Expansion Roadmap

**Version Target: 5.0**
**Timeline: 6-8 weeks**
**Last Updated: May 6, 2026**

---

## 📋 Overview

### Current State (v4.0 - LIVE)

| Component | Status |
| --- | --- |
| Base Model | ✅ SmolLM2-360M-Instruct (fine-tuned with QLoRA) |
| Training Samples | 347 |
| Inference | ✅ CPU (float32) - Working |
| SSE Streaming | ✅ Fixed and working |
| Chat UI | ✅ Functional with tool detection |
| Tools | ✅ Implemented (need testing) |
| Service | ✅ https://neuralai-webui-deandrewharris.zocomputer.io |

### Target State (v5.0)

| Component | Target |
| --- | --- |
| Training Samples | 1000+ (3x expansion) |
| Inference | GPU-hosted (float16 CUDA) |
| Alignment | DPO (Direct Preference Optimization) |
| Evaluation | Automated benchmarks |
| Tools | Fully integrated and tested |

---

## ✅ Phase 1: Tool Ecosystem (COMPLETED)

### Status: Tools implemented, need integration testing

| Tool | File | Status |
| --- | --- | --- |
| Code Sandbox | `tools/code_sandbox.py` | ✅ Implemented |
| File Manager | `tools/file_manager.py` | ✅ Implemented |
| Web Fetcher | `tools/web_fetcher.py` | ✅ Implemented |
| Database Connector | `tools/db_connector.py` | ✅ Implemented |
| Git Assistant | `tools/git_assistant.py` | ✅ Implemented |
| Tool Handler | `tools/tool_handler.py` | ✅ Implemented |
| Router | `neuralai_router.py` | ✅ Implemented |
| Frontend Detection | `index.html` | ✅ Implemented |

### Next: Integration Testing

- [ ] Test Code Sandbox: "run this python code: print('hello')"
- [ ] Test File Manager: "search files for neural"
- [ ] Test Web Fetcher: "fetch https://example.com"
- [ ] Test Database: "show tables"
- [ ] Test Git: "git status"

---

## 🎯 Phase 2: GPU-Hosted Inference (Weeks 1-2) - NEXT PRIORITY

### Goal: Deploy on GPU for 10x faster inference

### 2.1 Current Performance (CPU)

| Metric | CPU (Current) |
| --- | --- |
| Time to first token | 2-3 sec |
| Tokens/sec | 5-10 |
| Response time (100 tokens) | 10-20 sec |

### 2.2 Target Performance (GPU)

| Metric | GPU (Target) |
| --- | --- |
| Time to first token | 0.1-0.3 sec |
| Tokens/sec | 50-100 |
| Response time (100 tokens) | 1-2 sec |

### 2.3 Implementation Steps

1. **Enable GPU in Zo Service**
   ```bash
   # Update service with GPU request
   update_user_service(
     service_id="svc_eqdMuPQ4Alg",
     # Request GPU allocation
   )
   ```

2. **Update Model Loading for CUDA**
   ```python
   # In neuralai_engine.py
   device = "cuda" if torch.cuda.is_available() else "cpu"
   torch_dtype = torch.float16 if device == "cuda" else torch.float32
   ```

3. **Benchmark Performance**
   - Measure time to first token
   - Measure tokens/second
   - Compare CPU vs GPU

---

## 🎯 Phase 3: DPO Alignment (Weeks 3-4)

### Goal: Improve response quality through preference optimization

### 3.1 DPO Dataset Requirements

```json
{
  "prompt": "Write a function to sort a list",
  "chosen": "def sort_list(items):\n    return sorted(items)",
  "rejected": "def sort_list(items):\n    # TODO: implement\n    pass"
}
```

### 3.2 Implementation

1. Generate preference pairs from existing training data
2. Use `trl.DPOTrainer` for training
3. Evaluate on held-out prompts

---

## 🎯 Phase 4: Training Data Expansion (Weeks 5-6)

### Goal: Expand from 347 to 1000+ training samples

### Categories to Add

| Category | Samples | Description |
| --- | --- | --- |
| Advanced Coding | 100 | Multi-file projects, debugging |
| API Design | 50 | REST endpoints, OpenAPI |
| Database Ops | 50 | SQL queries, migrations |
| DevOps | 50 | Docker, Kubernetes, CI/CD |
| Data Analysis | 60 | Pandas, NumPy, viz |
| Testing | 40 | Unit tests, mocking |
| Security | 30 | Auth, encryption |
| Error Handling | 50 | Try-catch, logging |
| Documentation | 40 | README, API docs |
| Code Review | 30 | Style, optimization |
| Multi-turn Reasoning | 50 | Complex chains |
| Tool Chaining | 40 | Multiple tools |

---

## 🎯 Phase 5: Automated Evaluation (Week 7-8)

### Goal: Automated benchmarks for model quality

### Metrics

| Metric | Target |
| --- | --- |
| Perplexity | < 15 |
| Code Correctness | > 80% |
| Response Helpfulness | > 70% |
| Factual Accuracy | > 85% |
| Safety Score | 100% |

---

## 🚦 Priority Order

1. **NOW**: Test tool integration in chat
2. **Week 1-2**: GPU deployment
3. **Week 3-4**: DPO alignment
4. **Week 5-6**: Training data expansion
5. **Week 7-8**: Evaluation suite

---

**NeuralAI v5.0 Roadmap**
*Created: May 2, 2026*
*Updated: May 6, 2026 - SSE fixed, tools confirmed implemented*
*Target Completion: June 2026*
