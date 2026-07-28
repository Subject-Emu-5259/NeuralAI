---
title: NeuralAI Web Chat
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: NeuralAI — SmolLM2-360M + LoRA/DPO chat UI
license: apache-2.0
tags:
  - llm
  - dpo
  - smolllm
  - neuralai
  - chat
---

# NeuralAI Web Chat 🧠

Hosted web chat UI for **NeuralAI** — a SmolLM2-360M-Instruct model fine-tuned
with LoRA + DPO. The UI is the custom Google-style chat interface from the
NeuralAI repo (`from-scratch/web_ui`), served by the unified
`services/neural_core_service.py` backend.

Features:
- Streaming chat with the fine-tuned model
- Conversation history (SQLite)
- Memory & behavioral rules
- Guest / Maestro access (no account required)
- Document upload & RAG-ready endpoints

The latest LoRA adapter is pulled automatically from the
`Subject-Emu-5259/NeuralAI` Hugging Face repo on startup, so retraining and
pushing the adapter updates this Space on the next restart.

By DeAndrew P. Harris.
