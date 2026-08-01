#!/usr/bin/env python3
"""
Mamba K2 Training Script — NeuralAI's Owned Base Model v2
==========================================================
Run on Google Colab with T4/V100 GPU (free tier works).

Architecture: Mamba-790M (SSM, 793M params, 48 layers, d_model=1536)
Training: SFT on 10K+ OpenHermes-2.5 samples, 500-1000 steps
Output: LoRA adapter merged → full model → ONNX export
"""

import os
import json
import torch
import warnings
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    MambaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import time

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION — Adjust these before running
# ============================================================
BASE_MODEL = "state-spaces/mamba-790m-hf"
OUTPUT_DIR = "/content/drive/MyDrive/NeuralAI/mamba-k2"
MAX_LENGTH = 2048
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
SAVE_STEPS = 100
EVAL_STEPS = 100
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# ============================================================
# STEP 1: Load tokenizer & base model
# ============================================================
print("🧠 Loading Mamba-790M...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = MambaForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
print(f"✅ Base model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# ============================================================
# STEP 2: Load 10K+ training samples (OpenHermes-2.5)
# ============================================================
print("\n📚 Loading OpenHermes-2.5 dataset (10K+ samples)...")
dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
dataset = dataset.shuffle(seed=42).select(range(10500))

def format_chatml(examples):
    """Format each conversation into ChatML for training."""
    texts = []
    for conv in examples["conversations"]:
        formatted = ""
        for turn in conv:
            role = turn["from"]
            content = turn["value"]
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "human":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "gpt":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        texts.append(formatted)
    return {"text": texts}

print("   Formatting ChatML...")
dataset = dataset.map(format_chatml, batched=True, remove_columns=dataset.column_names)

def tokenize(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("   Tokenizing...")
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

split = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds = split["test"]

print(f"   Train: {len(train_ds):,} samples | Eval: {len(eval_ds):,} samples")
print(f"   Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"   Steps/epoch: {len(train_ds) // (BATCH_SIZE * GRADIENT_ACCUMULATION)}")
print(f"   Total steps: {(len(train_ds) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * NUM_EPOCHS}")

# ============================================================
# STEP 3: Configure LoRA
# ============================================================
print("\n🔧 Configuring LoRA adapters...")
# Mamba uses SSM layers — target x_proj, dt_proj, in_proj, out_proj
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["x_proj", "dt_proj", "in_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ============================================================
# STEP 4: Training
# ============================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    dataloader_num_workers=2,
    remove_unused_columns=False,
    gradient_checkpointing=False,  # Mamba doesn't support gradient checkpointing
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print(f"\n🚀 Starting training on {torch.cuda.get_device_name(0)}...")
start_time = time.time()
trainer.train()
elapsed = time.time() - start_time
print(f"\n✅ Training complete! Duration: {elapsed/60:.1f} min")

# ============================================================
# STEP 5: Save artifacts
# ============================================================
final_adapter_dir = f"{OUTPUT_DIR}/mamba-k2-adapter"
model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)
print(f"\n💾 LoRA adapter saved to {final_adapter_dir}")

# Save chat template
chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)
with open(f"{final_adapter_dir}/chat_template.jinja", "w") as f:
    f.write(chat_template)
tokenizer.chat_template = chat_template
tokenizer.save_pretrained(final_adapter_dir)

# ============================================================
# STEP 6: Merge adapter into full model
# ============================================================
print("\n🔗 Merging LoRA adapter into base model...")
merged_dir = f"{OUTPUT_DIR}/mamba-k2-merged"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"💾 Merged model saved to {merged_dir}")

# ============================================================
# STEP 7: Quick generation test
# ============================================================
print("\n🧪 Running generation test...")
test_messages = [
    {"role": "system", "content": "You are NeuralAI Mamba K2, a helpful AI assistant."},
    {"role": "user", "content": "Explain what a State Space Model is in simple terms."},
]
test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
test_inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **test_inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n{result}")

print("\n" + "="*60)
print("🎉 Mamba K2 training pipeline complete!")
print(f"   Adapter: {final_adapter_dir}")
print(f"   Merged:  {merged_dir}")
print(f"   Eval loss: {trainer.state.best_metric:.4f}")
print("="*60)
