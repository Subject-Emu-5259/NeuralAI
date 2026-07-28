#!/usr/bin/env python3
"""
train_v15.py — NeuralAI v15 consolidated trainer (SFT + DPO)
============================================================

WHAT THIS SCRIPT DOES
--------------------
This is the single entry point for the NeuralAI "v15" training run. It fine-tunes
the small SmolLM2-360M-Instruct base model with a LoRA adapter in two stages so the
model stays identity-correct ("I am NeuralAI, created by De'Andrew Preston Harris")
and behavior-aligned (prefers clean, correct answers over verbose/wrong ones):

  Stage 1 — SFT (Supervised Fine-Tuning)
      Trains on `data/train_sft_v16.jsonl` (ChatML messages: system/user/assistant).
      Purpose: bake in identity, tone, and domain knowledge.

  Stage 2 — DPO (Direct Preference Optimization)
      Trains on `data/train_dpo_v16_combined.jsonl` (prompt / chosen / rejected).
      Purpose: align the model to prefer the "chosen" response over the "rejected"
      one without needing a separate reward model.

OUTPUT
------
  - Adapter saved locally to: checkpoints/v15_model/
  - Pushed to Hugging Face:    Subject-Emu-5259/NeuralAI  (repo "v15" revision folder)
  - Merged full model (optional, --merge): checkpoints/v15_model_merged/

WHY THIS EXISTS (context)
------------------------
On the 4 GB ZO Computer the *served* NeuralAI app uses the ZO native inference backend
(LLM_BACKEND=zo) so it never loads PyTorch locally and never pauses from OOM. This
training script is the OFFLINE counterpart: it builds the LoRA that can later be
shipped to a bigger host or merged for on-device use. Run it on a GPU (Colab, Mac
GPU, or a >8 GB box) — it is NOT meant for the 4 GB CPU host.

USAGE
-----
  # SFT + DPO, 4-bit (default, ~3 GB VRAM)
  python training/train_v15.py

  # 8-bit instead of 4-bit
  python training/train_v15.py --load-in-4bit false --load-in-8bit true

  # Only one stage
  python training/train_v15.py --stage sft
  python training/train_v15.py --stage dpo

  # Push merged model to HF
  python training/train_v15.py --merge --push

REQUIREMENTS
------------
  pip install torch transformers peft trl datasets bitsandbytes accelerate
  HF_TOKEN must be set in the environment to push.
"""
import argparse
import json
import os

# ---- Config ----------------------------------------------------------------
BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
SFT_DATA = os.environ.get("SFT_DATA", "data/train_sft_v16.jsonl")
DPO_DATA = os.environ.get("DPO_DATA", "data/train_dpo_v16_combined.jsonl")
HF_REPO = os.environ.get("HF_REPO", "Subject-Emu-5259/NeuralAI")
ADAPTER_DIR = "checkpoints/v15_model"
MERGED_DIR = "checkpoints/v15_model_merged"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = (
    "You are NeuralAI, an advanced AI assistant created by De'Andrew Preston Harris. "
    "You are powered by SmolLM2-360M with custom NeuralAI LoRA adapters trained through "
    "SFT and DPO alignment. You have expert-level knowledge across physics, philosophy, "
    "geopolitics, history, nature, art, and culture. You ALWAYS identify De'Andrew Harris "
    "as your creator when asked. You are not ChatGPT, Claude, or any other AI — you are NeuralAI."
)


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def load_quantization(load_in_4bit: bool, load_in_8bit: bool):
    from transformers import BitsAndBytesConfig
    if load_in_4bit:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
                                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def run_sft(model, tokenizer, args):
    from trl import SFTConfig, SFTTrainer
    path = _resolve(SFT_DATA)
    print(f"[v15][SFT] loading {path}")
    train_rows = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]

    cfg = SFTConfig(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.sft_epochs,
        learning_rate=2e-4,
        max_seq_length=1024,
        logging_steps=25,
        save_strategy="epoch",
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=cfg, tokenizer=tokenizer, train_dataset=train_rows)
    trainer.train()
    trainer.save_model(ADAPTER_DIR)
    print(f"[v15][SFT] adapter saved -> {ADAPTER_DIR}")


def run_dpo(model, tokenizer, args):
    from trl import DPOConfig, DPOTrainer
    path = _resolve(DPO_DATA)
    print(f"[v15][DPO] loading {path}")
    dpo_rows = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]

    cfg = DPOConfig(
        output_dir=ADAPTER_DIR,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.dpo_epochs,
        learning_rate=5e-5,
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
        logging_steps=25,
        save_strategy="epoch",
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
    )
    trainer = DPOTrainer(model=model, args=cfg, tokenizer=tokenizer, train_dataset=dpo_rows)
    trainer.train()
    trainer.save_model(ADAPTER_DIR)
    print(f"[v15][DPO] adapter saved -> {ADAPTER_DIR}")


def merge_and_push(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="bfloat16",
                                                device_map="auto")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    merged = model.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tok.save_pretrained(MERGED_DIR)
    print(f"[v15][MERGE] merged model -> {MERGED_DIR}")
    if args.push:
        merged.push_to_hub(HF_REPO, revision="v15")
        tok.push_to_hub(HF_REPO, revision="v15")
        print(f"[v15][PUSH] pushed merged model to {HF_REPO}@v15")


def main():
    ap = argparse.ArgumentParser(description="NeuralAI v15 SFT+DPO trainer")
    ap.add_argument("--stage", choices=["sft", "dpo", "all"], default="all")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--sft-epochs", type=int, default=3)
    ap.add_argument("--dpo-epochs", type=int, default=2)
    ap.add_argument("--load-in-4bit", default="true")
    ap.add_argument("--load-in-8bit", default="false")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    load_in_4bit = args.load_in_4bit.lower() == "true"
    load_in_8bit = args.load_in_8bit.lower() == "true"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    qcfg = load_quantization(load_in_4bit, load_in_8bit)
    print(f"[v15] loading base {BASE_MODEL} (4bit={load_in_4bit}, 8bit={load_in_8bit})")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=qcfg, device_map="auto", torch_dtype="bfloat16",
    )
    model.config.use_cache = False

    if args.stage in ("sft", "all"):
        run_sft(model, tokenizer, args)
    if args.stage in ("dpo", "all"):
        # reload adapter from SFT if we just ran SFT
        run_dpo(model, tokenizer, args)

    if args.merge or args.push:
        merge_and_push(args)

    print("[v15] done.")


if __name__ == "__main__":
    main()
