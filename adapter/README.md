---
base_model: HuggingFaceTB/SmolLM2-360M-Instruct
library_name: peft
tags:
- lora
- dpo
- smollm2
- text-generation
- transformers
- trl
license: apache-2.0
language:
- en
pipeline_tag: text-generation
model_name: NeuralAI v17-dpo (D17)
---

# NeuralAI — D17 DPO LoRA Adapter (v17)

NeuralAI is a DPO-aligned generative AI engine by De'Andrew Preston Harris.
This repository publishes the **D17 DPO LoRA adapter** (v17), a continuation of
the v16 adapter, trained on top of `HuggingFaceTB/SmolLM2-360M-Instruct`.

## Training (D17 / v17 — 2026-07-20)

- **Base model:** `HuggingFaceTB/SmolLM2-360M-Instruct` (360M)
- **Method:** DPO (TRL), LoRA r=32 / alpha=64, dropout 0.05
- **Seed adapter:** v16 (`checkpoints/v2_model`)
- **Dataset:** 679 preference pairs (`data/train_dpo_v16_combined.jsonl`)
- **Epochs:** 3 — **Steps:** 129 — **Duration:** ~31 min (Colab GPU)
- **Final train loss:** ~0.39 (0.692 → 0.396)
- **Reward accuracy:** 0.975 (chosen preferred over rejected)
- **Reward margin:** ~0.9 (stable, no collapse)
- **Entropy:** ~2.15–2.33 (stable)

D17 is a DPO *continuation* of v16 — an alignment refinement, not new knowledge
injection. Across the 679-pair distribution it sharpened preferences toward
correct code, safe refusals, identity-accurate responses, sound math/reasoning,
clean code style, concise answers, and proper tool usage, while down-ranking
rejected patterns. Chosen-response rewards rose (~0.54 → 0.69) and rejected
rewards fell (~−0.19 → −0.26) with a healthy ~0.9 margin and stable entropy —
textbook successful DPO with no sign of reward hacking or mode collapse. No
held-out eval set was configured, so generalization is inferred from training
reward signals only.

## Usage (PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
model = PeftModel.from_pretrained(base, "Subject-Emu-5259/NeuralAI")
tokenizer = AutoTokenizer.from_pretrained("Subject-Emu-5259/NeuralAI")
```

## Links

- **GitHub:** https://github.com/Subject-Emu-5259/NeuralAI
- **Live demo:** https://neuralai-web-ui-deandrewharris.zocomputer.io
- **Founder:** De'Andrew Preston Harris (D. Harris / Dre)
