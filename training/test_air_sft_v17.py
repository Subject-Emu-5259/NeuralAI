#!/usr/bin/env python3
"""Quick CPU smoke test for NeuralAI-Air-135M-SFT v17."""
import json
import os
import sys
import importlib.util
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

spec = importlib.util.spec_from_file_location("air", os.path.join(PROJ, "NeuralAI-Air-135M", "NeuralAI-Air-135M.py"))
air = importlib.util.module_from_spec(spec)
spec.loader.exec_module(air)

MODEL_DIR = os.path.join(PROJ, "models", "NeuralAI-Air-135M-SFT")
HF_DIR = os.path.join(PROJ, "models", "NeuralAI-Air-135M-SFT")

tokenizer_dir = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "tokenizer.json")) else HF_DIR
# AutoTokenizer preserves the special_tokens_map and added-token IDs.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
print("vocab:", len(tokenizer))

cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
valid_keys = {f.name for f in air.NeuralAIAir135MConfig.__dataclass_fields__.values()}
config = air.NeuralAIAir135MConfig(**{k: v for k, v in cfg.items() if k in valid_keys})

device = torch.device("cpu")
model = air.NeuralAIAir135MModel(config).to(device)
state = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

eos_id = config.eos_token_id
im_start = tokenizer.decode([1])
im_end = tokenizer.decode([2])


def chat_prompt(messages, add_generation=True):
    parts = []
    for m in messages:
        parts.append(f"{im_start}{m['role']}\n{m['content'].strip()}{im_end}")
    if add_generation:
        parts.append(f"{im_start}assistant\n")
    return "\n".join(parts)


@torch.no_grad()
def generate(prompt, max_new=80, temperature=0.7, top_p=0.9):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(device)
    generated = input_ids
    for _ in range(max_new):
        logits, _ = model(generated[:, -config.max_position_embeddings:])
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_probs[mask] = 0.0
        if sorted_probs.sum() == 0:
            sorted_probs[0, 0] = 1.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_t = torch.gather(sorted_indices, dim=-1, index=torch.multinomial(sorted_probs, num_samples=1))
        generated = torch.cat((generated, next_t), dim=1)
        if next_t.item() == eos_id:
            break
    out = generated[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(out, skip_special_tokens=False)


PROMPTS = [
    [{"role": "system", "content": "You are NeuralAI, a helpful assistant."},
     {"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "Explain quantum field theory in one sentence."}],
    [{"role": "user", "content": "Who are you?"}],
]

for messages in PROMPTS:
    prompt = chat_prompt(messages)
    print("\n--- prompt ---")
    print(prompt[:200])
    print("--- output ---")
    print(generate(prompt))
