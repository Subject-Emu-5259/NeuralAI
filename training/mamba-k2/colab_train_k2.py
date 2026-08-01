#!/usr/bin/env python3
"""
NeuralAI Mamba K2 — Colab Training Pipeline
============================================
Base: state-spaces/mamba-790m-hf (793M params, Mamba SSM)
Data: UltraChat (10K+ samples)
Steps: 500–1000
Output: Fine-tuned Mamba K2 model → GGUF Q4_K_M for LM Studio

⚠️  RUN THIS ON GOOGLE COLAB WITH A T4 GPU (free tier works)
    Runtime → Change runtime type → T4 GPU

⏱️  Estimated time:
    - 500 steps × batch 4 × 790M params ≈ 1.5 hours on T4
    - 1000 steps × batch 4 ≈ 3 hours on T4
    - GGUF conversion + quantize ≈ 2 minutes
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
!pip install -q transformers datasets accelerate peft torch trl \
  mamba-ssm --no-build-isolation 2>&1 | tail -3

import os
import torch
import json
from datetime import datetime

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# CELL 2: Load base model (Mamba-790M)
# ============================================================
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

MODEL_ID = "state-spaces/mamba-790m-hf"
OUTPUT_DIR = "/content/mamba-k2-output"

print(f"Loading base model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

# ============================================================
# CELL 3: Chat template (NeuralAI ChatML format)
# ============================================================
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'].strip() }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'].strip() }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'].strip() }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)

tokenizer.chat_template = CHAT_TEMPLATE
print("Chat template set: NeuralAI ChatML")

# ============================================================
# CELL 4: Load & format UltraChat (10K+ samples)
# ============================================================
SYSTEM_PROMPT = (
    "You are NeuralAI K2, a 793M parameter Mamba architecture assistant "
    "built and owned by NeuralAI. You are direct, concise, and helpful. "
    "Respond naturally without self-promotion or corporate tone."
)

def format_ultrachat(examples):
    """Convert UltraChat to NeuralAI ChatML format."""
    texts = []
    for i in range(len(examples["data"])):
        messages = examples["data"][i]
        if not isinstance(messages, list):
            continue
        formatted = []
        for msg in messages[:6]:  # Max 6 turns
            role = "user" if msg.get("role") == "user" else "assistant"
            content = msg.get("content", "").strip()
            if content:
                formatted.append({"role": role, "content": content})
        if len(formatted) >= 2:
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}] + formatted,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
    return {"text": texts}

print("Loading UltraChat (10K samples)...")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:11000]")
dataset = dataset.map(format_ultrachat, batched=True, remove_columns=dataset.column_names)
dataset = dataset.filter(lambda x: len(x["text"]) > 100 and len(x["text"]) < 4096)
dataset = dataset.train_test_split(test_size=0.05, seed=42)

print(f"Train: {len(dataset['train'])} samples")
print(f"Eval:  {len(dataset['test'])} samples")
print(f"\nSample (first 300 chars):\n{dataset['train'][0]['text'][:300]}...")

# ============================================================
# CELL 5: Tokenize
# ============================================================
MAX_LENGTH = 2048

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
print(f"Tokenized: {len(tokenized['train'])} train, {len(tokenized['test'])} eval")

# ============================================================
# CELL 6: LoRA config
# ============================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["x_proj", "in_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# CELL 7: Training args
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=10,
    save_steps=250,
    eval_steps=250,
    save_total_limit=3,
    max_steps=750,                         # <= CHANGE THIS: 500–1000
    fp16=True,
    gradient_checkpointing=False,
    dataloader_num_workers=2,
    report_to="none",
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
)

# ============================================================
# CELL 8: Train
# ============================================================
print(f"🚀 Starting Mamba K2 training — {training_args.max_steps} steps...")
print(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   Samples: {len(tokenized['train'])}")
print(f"   Output: {OUTPUT_DIR}")

start = datetime.now()
trainer.train()
elapsed = datetime.now() - start
print(f"\n✅ Training complete in {elapsed}")
print(f"   Final loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")

# ============================================================
# CELL 9: Save model + tokenizer
# ============================================================
FINAL_DIR = "/content/mamba-k2-final"

model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

# Save training metadata
metadata = {
    "model": "NeuralAI Mamba K2",
    "base": "state-spaces/mamba-790m-hf",
    "architecture": "Mamba SSM",
    "params": "793M",
    "lora": {"r": 32, "alpha": 64},
    "steps": training_args.max_steps,
    "samples": len(tokenized["train"]),
    "dataset": "HuggingFaceH4/ultrachat_200k (10K+ subset)",
    "chat_template": "NeuralAI ChatML (<|im_start|> / <|im_end|>)",
    "system_prompt": SYSTEM_PROMPT,
    "trained_at": datetime.now().isoformat(),
    "ownership": "NeuralAI — Second Owned Base Model",
    "runtime": "llama.cpp GGUF Q4_K_M / transformers PyTorch",
}
with open(f"{FINAL_DIR}/neuralai_mamba_k2_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved to {FINAL_DIR}")

# ============================================================
# CELL 10: Quick generation test
# ============================================================
test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the capital of France and why is it significant?"},
]

inputs = tokenizer.apply_chat_template(test_messages, tokenize=True, return_tensors="pt", add_generation_prompt=True).cuda()

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
print("Test generation:")
print(response)

# ============================================================
# CELL 11: Download to local & GGUF conversion
# ============================================================
from google.colab import files
import shutil

# Zip the model for download
shutil.make_archive("/content/mamba-k2-final", "zip", FINAL_DIR)
print(f"Zipped: /content/mamba-k2-final.zip ({os.path.getsize('/content/mamba-k2-final.zip') / 1e6:.0f} MB)")

# Download to your computer
files.download("/content/mamba-k2-final.zip")

print("""
📋 NEXT STEPS (run these locally on Zo after download):
1. Extract the zip to Projects/NeuralAI/models/mamba-k2/
2. Convert to GGUF:
   python3 llama.cpp/convert_hf_to_gguf.py models/mamba-k2 --outtype f16 --outfile models/mamba-k2-f16.gguf
3. Quantize to Q4_K_M:
   llama.cpp/llama-quantize models/mamba-k2-f16.gguf Q4_K_M models/NeuralAI-Mamba-K2.Q4_K_M.gguf
4. Register in LM Studio or NeuralAI model_manager
5. Run benchmarks: python3 benchmarks/run_evals.py --model mamba-k2
""")
