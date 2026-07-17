---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
  - smollm2
  - lora
  - dpo
  - peft
  - neuralai
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
model_name: NeuralAI v2
---

# NeuralAI v2 (merged)

This repository hosts the **merged** weights of the [NeuralAI v2 LoRA + DPO adapter](https://huggingface.co/Subject-Emu-5259/NeuralAI/tree/main).

It is the same model and tokenizer as the LoRA-only repo, but with the adapter weights fused into the base — load it directly without `peft.PeftModel.from_pretrained()`.

## Model

- **Base:** `HuggingFaceTB/SmolLM2-360M-Instruct`
- **Adapter:** LoRA rank 16, alpha 32, dropout 0.05
- **Method:** DPO preference alignment on top of SFT
- **Params:** 361.8M
- **License:** Apache-2.0
- **Framework:** PyTorch + Transformers 5.5.4 + PEFT 0.19.0

## How to load

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Subject-Emu-5259/NeuralAI"  # this repo
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

prompt = "NeuralAI is"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, top_p=0.9, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Training

| Field | Value |
|---|---|
| Method | LoRA + DPO preference alignment |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Train / val | 363 / 41 |
| Duration | ~26 min |
| Completed | 2026-05-17 |
| Framework | PyTorch + Transformers + PEFT 0.19.0 |

By De'Andrew P. Harris. NeuralAI is part of the NeuralAI / NeuralLabs product stack.
