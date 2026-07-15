---
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: peft
model_name: NeuralAI
model_type: adapter
license: apache-2.0
language:
- en
tags:
- text-generation
- dpo
- lora
- peft
- smollm2
- reasoning
- code-generation
- debugging
- multi-step-reasoning
- edge-ai
pipeline_tag: text-generation
inference:
  parameters:
    max_new_tokens: 512
    temperature: 0.7
    top_p: 0.95
    repetition_penalty: 1.1
---

# NeuralAI v15.0 — DPO-Aligned LoRA Adapter

NeuralAI is a DPO-aligned LoRA adapter for [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), fine-tuned for expert-level reasoning, code generation, debugging, and multi-step logic tasks.

## Highlights

- **597 DPO preference pairs** covering code correctness, logic, reasoning, debugging, and multi-step tasks
- **Reward margin**: improved from ~0.5 to ~3.5 (model strongly prefers chosen responses)
- **Final training loss**: 0.305
- **Edge-optimized**: Runs on CPU with 4GB RAM — no GPU required
- **Gemini-style alignment**: Helpful, structured, conversational tone with step-by-step explanations

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = PeftModel.from_pretrained(base_model, "Subject-Emu-5259/NeuralAI")

messages = [{"role": "user", "content": "Write a Python function to check API health."}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=256, temperature=0.7, top_p=0.95)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Training details

| Parameter | Value |
|---|---|
| Base model | HuggingFaceTB/SmolLM2-360M-Instruct |
| Method | DPO (Direct Preference Optimization) |
| Dataset | 597 preference pairs (v15 expanded) |
| Epochs | 3 |
| Steps | 450 |
| Final loss | 0.305 |
| Reward margin | ~3.5 |
| LoRA rank | 16 |
| Hardware | Apple Silicon MPS (MacBook Air M4) |
| Duration | ~12 minutes |
| Completed | 2026-07-11 |

## Framework versions

- PEFT: 0.17.1
- TRL: 0.24.0
- Transformers: 4.57.6
- PyTorch: 2.8.0

## Use cases

- **Code generation and debugging**: Multi-step reasoning for code correctness
- **Logic and math**: Complex problem decomposition
- **Edge deployment**: CPU-optimized for local/private AI
- **Agentic workflows**: Tool-use and multi-step task execution

## Citation

```bibtex
@inproceedings{rafailov2023direct,
    title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
    author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
    year         = 2023,
    booktitle    = {NeurIPS 2023},
}
```

## 🔌 Use NeuralAI as an OpenAI-compatible backend (BYO API / ZO Computer BYOK)

NeuralAI's hosted service exposes an OpenAI-compatible chat API so it can power other chat UIs, including **ZO Computer's Bring Your Own Key (BYOK)**.

- **Base URL**: `https://neuralai-web-ui-deandrewharris.zocomputer.io/v1`
- **Model id**: `neuralai`
- **Auth**: Personal API key (generate in NeuralAI Settings → Developer/API Access; keys are hashed + revocable)
- **Endpoints**: `POST /v1/chat/completions` (SSE streaming + non-streaming JSON, CORS-enabled) and `POST /v1/models`. `GET` probes on these paths now return `200` so host validation passes.

Full setup walkthrough: [`docs/BYOK_ZO_INTEGRATION.md`](https://github.com/Subject-Emu-5259/NeuralAI/blob/master/docs/BYOK_ZO_INTEGRATION.md).
Full project documentation: [GitHub README](https://github.com/Subject-Emu-5259/NeuralAI).
