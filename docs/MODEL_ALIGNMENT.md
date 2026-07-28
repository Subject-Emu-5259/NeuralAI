# 🌌 NeuralAI Model Alignment: v7.3

This document tracks the integration of the Neural-Brain knowledge into the active model weights.

## 🔄 Synchronization Status
- **Knowledge Base**: High-Density Markdown Assets (Neural-Brain)
- **Method**: RAG (Retrieval-Augmented Generation) + Context Injection + SFT alignment
- **Current Layer**: Expert Transition
- **Latest Weight Snapshot**: DPO v17.0 aligned — 679 DPO pairs, final loss `~0.396`, reward accuracy `0.975`
- **Latest SFT Snapshot**: NeuralAI-Air-135M SFT v17 — 37 instruction/response pairs, 3 epochs
- **Latest Run Completion**: `2026-07-26`
- **SmolLM2 Adapter Live**: [`Subject-Emu-5259/NeuralAI`](https://huggingface.co/Subject-Emu-5259/NeuralAI) (auto-pulled by demo + web UI)
- **Air 135M SFT Live**: [`Subject-Emu-5259/NeuralAI-Air-135M-SFT`](https://huggingface.co/Subject-Emu-5259/NeuralAI-Air-135M-SFT) — first supervised fine-tune of the from-scratch 135M base
- **Inference Backend**: llmster 0.0.19 (LM Studio headless, llama.cpp) on port 1234
- **SmolLM2 Model**: SmolLM2-360M-Instruct Q4_K_M GGUF (~258MB RAM)

## 🧪 Expert-Level Modules
- **Physics**: [ACTIVE] Quantum Field Theory and Quantum Mechanics modules populated.
- **Philosophy**: [ACTIVE] Platonic Philosophy module populated.
- **Geopolitics**: [ACTIVE] Modern Geopolitics module populated.
- **Esoteric Mysticism**: [ACTIVE] Mysticism module populated in Culture.
- **History & Nature**: [ACTIVE] Ancient Civilizations, Human Evolution, and Bio Foundations populated.

## 🆕 Recent Alignment Runs

### DPO v17.0 (D17) — 2026-07-20
- **Base**: `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Dataset**: 679 DPO pairs (`data/train_dpo_v16_combined.jsonl`)
- **Epochs**: 3, **Steps**: 129
- **Reward accuracy**: 0.975, **Reward margin**: ~0.9
- **Adapter**: `checkpoints/v17-dpo` and HF `Subject-Emu-5259/NeuralAI`

### NeuralAI-Air-135M SFT v17 — 2026-07-26
- **Base**: `Subject-Emu-5259/NeuralAI-Air-135M` (custom `neuralai-air` architecture, 133.72M params)
- **Dataset**: 37 SFT samples (`data/train_sft_v17.jsonl`)
- **Epochs**: 3, **Steps per epoch**: 3
- **Batch**: 4, **Accum**: 4, **LR**: 5e-5
- **Output**: full fine-tuned model pushed to `Subject-Emu-5259/NeuralAI-Air-135M-SFT`
