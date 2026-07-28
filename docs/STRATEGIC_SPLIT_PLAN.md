# 🚀 NeuralAI Strategic Split Plan

## 🎯 Overview
Transitioning from the single-model approach (SmolLM2-360M) to a multi-tiered architecture to accommodate increased cognitive complexity and diverse use cases.

## 🏗️ The Two Tiers

### 1. NeuralAI-2B-Speedster
**Target**: Low-latency, high-efficiency conversational tasks.
- **Parameters**: ~2.0B
- **Hidden Size**: 2048
- **Layers**: 26
- **Attention Heads**: 16
- **Intermediate Size**: 8192
- **Primary Use Case**: Real-time chat, mobile deployment, rapid reasoning.

### 2. NeuralAI-3B-Core-Intelligence
**Target**: High-fidelity, deep reasoning, and complex orchestrator tasks.
- **Parameters**: ~3.0B
- **Hidden Size**: 3072
- **Layers**: 32
- **Attention Heads**: 24
- **Intermediate Size**: 12288
- **Primary Use Case**: Complex problem solving, heavy knowledge retrieval, advanced agentic tool use.

## 🛠️ Phase Roadmap

### Phase 1: Architecture & Base Training
- [ ] Finalize transformer topology for 2B and 3B.
- [ ] Scaled weight initialization to prevent activation spikes.
- [ ] Pre-training on high-density synthetic data (Neural-Brain).

### Phase 2: Alignment (SFT & DPO)
- [ ] **SFT Pass**: Large-scale ChatML instruction tuning.
- [ ] **DPO Pass**: Preference optimization to fix repetition and bias (newline, system, etc.).
- [ ] **Targeted Interventions**: Manual weight zeroing/suppression for stubborn token biases.

### Phase 3: Optimization & Deployment
- [ ] Quantization (GGUF/EXL2) for edge and cloud deployment.
- [ ] Integration with the Hybrid Link Gateway.
- [ ] Final validation against expert-level benchmark sets.

---
*Status: ACTIVE | Version: 1.0 | Date: 2026-07-28*
