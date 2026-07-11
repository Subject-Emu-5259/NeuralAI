---
title: NeuralAI v2 Chat
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: NeuralAI v2 - SmolLM2-360M LoRA + DPO chat demo
tags:
  - llm
  - dpo
  - smolllm
  - neuralai
  - peft
  - chat
---

# NeuralAI v2 Chat Demo 🧠

Live chat demo for the **NeuralAI v2** LoRA adapter (DPO on SmolLM2-360M-Instruct).

This Space runs the **fine-tuned adapter locally** with PEFT merging — no Inference API needed.

## Model Details

- **Adapter:** `Subject-Emu-5259/NeuralAI` (LoRA, rank 16, α 32)
- **Base:** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Training:** LoRA + DPO, 3 epochs, 363 samples
- **Author:** De'Andrew P. Harris

## Usage

Just type a message and press Send. The model runs on the Space's CPU/GPU.

---

Built with 🤗 Transformers + PEFT + Gradio