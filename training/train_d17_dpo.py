"""
NeuralAI D17 — DPO iteration on CPU (no CUDA in Zo sandbox).
Base: SmolLM2-360M-Instruct + v16 adapter (checkpoint-69, best) as the new reference.
Trainer seeds LoRA on top of the merged v16 adapter, runs DPO over train_dpo_v16_combined.jsonl.
Output: checkpoints/v17-dpo (new adapter, checkpoint-69+).
"""
import os, json, time, datetime
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
V16_ADAPTER = "./checkpoints/v2_model"
DATA = "data/train_dpo_v16_combined.jsonl"
OUT = "./checkpoints/v17-dpo"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{datetime.datetime.now()}] device={device}", flush=True)

# ---- load base + v16 adapter as the starting (reference) model ----
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="cpu"
)
print(f"[{datetime.datetime.now()}] base loaded", flush=True)
model = PeftModel.from_pretrained(base_model, V16_ADAPTER)
print(f"[{datetime.datetime.now()}] v16 adapter attached; trainable before new lora:",
      sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

# attach a FRESH LoRA head on top of v16 for D17 DPO
lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ---- load DPO data ----
rows = [json.loads(l) for l in open(DATA) if l.strip()]
print(f"[{datetime.datetime.now()}] DPO pairs loaded: {len(rows)}", flush=True)

def fmt(r):
    return {
        "prompt": r["prompt"],
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    }
ds = Dataset.from_list([fmt(r) for r in rows])

# ---- DPO config (CPU-safe) ----
cfg = DPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=torch.cuda.is_available(),
    fp16=False,
    optim="adamw_torch",
    max_length=1024,
    report_to="none",
)
# force CPU since no CUDA
cfg.device = torch.device("cpu")

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # implicit reference = model before training (v16+base)
    args=cfg,
    tokenizer=tok,
    train_dataset=ds,
)
print(f"[{datetime.datetime.now()}] starting D17 DPO train...", flush=True)
t0 = time.time()
trainer.train()
dt = time.time() - t0
print(f"[{datetime.datetime.now()}] DPO train done in {dt/60:.1f} min", flush=True)

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
with open(os.path.join(OUT, "dpo_run.json"), "w") as f:
    json.dump({
        "name": "NeuralAI D17 DPO",
        "base": BASE,
        "seed_adapter": V16_ADAPTER,
        "dataset": DATA,
        "pairs": len(rows),
        "epochs": 3,
        "lora_r": 32,
        "train_seconds": round(dt, 1),
        "finished_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }, f, indent=2)
print(f"[{datetime.datetime.now()}] saved adapter -> {OUT}", flush=True)
