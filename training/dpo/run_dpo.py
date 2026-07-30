"""DPO entrypoint for NeuralAI-Air-135M v19.

Loads the SFT checkpoint, reads preference pairs from ``data/train_dpo_v19.jsonl``,
applies LoRA via PEFT (r=32, alpha=64, all linear layers), and trains with
``DPOTrainer`` from TRL.

Typical usage::

    python -m training.dpo.run_dpo --config training/dpo/config_dpo.yaml
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Lazy-load heavy dependencies so the module imports cleanly even when
# PyTorch / transformers / PEFT are not installed.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
except ImportError:
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TrainingArguments = None  # type: ignore
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


def load_dpo_dataset(
    path: str, tokenizer, max_length: int = 1024
) -> Dict[str, List[str]]:
    """Load DPO preference pairs into a dict of lists.

    Expected JSONL fields (any combination):
      - ``prompt``, ``chosen``, ``rejected``
      - ``system``, ``instruction``, ``output`` (chosen), ``rejected_output``

    Each example is truncated from the left to *max_length* tokens.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DPO dataset not found: {path}")

    prompts: List[str] = []
    chosen: List[str] = []
    rejected: List[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt_text = obj.get("prompt", "")
            if not prompt_text:
                system = obj.get("system", "")
                instruction = obj.get("instruction", "")
                parts: List[str] = []
                if system:
                    parts.append(f"<|im_start|>system\n{system}<|im_end|>")
                parts.append(f"<|im_start|>user\n{instruction}<|im_end|>")
                prompt_text = "\n".join(parts)

            chosen_text = obj.get("chosen", obj.get("output", ""))
            rejected_text = obj.get("rejected", obj.get("rejected_output", ""))

            if chosen_text and not chosen_text.startswith("<|im_start|>assistant"):
                chosen_text = f"<|im_start|>assistant\n{chosen_text}<|im_end|>"
            if rejected_text and not rejected_text.startswith("<|im_start|>assistant"):
                rejected_text = f"<|im_start|>assistant\n{rejected_text}<|im_end|>"

            full_chosen = f"{prompt_text}\n{chosen_text}"
            full_rejected = f"{prompt_text}\n{rejected_text}"

            def truncate(text: str, length: int) -> str:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > length:
                    ids = ids[-length:]
                return tokenizer.decode(ids, skip_special_tokens=False)

            prompts.append(prompt_text)
            chosen.append(truncate(full_chosen, max_length))
            rejected.append(truncate(full_rejected, max_length))

    return {"prompt": prompts, "chosen": chosen, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuralAI-Air-135M DPO v19"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/dpo/config_dpo.yaml",
        help="Path to DPO config YAML",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from a specific checkpoint",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    if torch is None:
        print(
            "[ERROR] PyTorch is required but not installed. "
            "Install: pip install torch>=2.1.0"
        )
        sys.exit(1)

    import yaml

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    # Load SFT model
    base_path = config["model"]["base_model_path"]
    if not os.path.isdir(base_path):
        print(f"[ERROR] SFT model not found at {base_path}")
        sys.exit(1)

    logger.info(f"Loading SFT model from {base_path}")
    model = AutoModelForCausalLM.from_pretrained(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # Load LoRA config
    lora_path = os.path.join(os.path.dirname(args.config), "lora_config.json")
    if os.path.exists(lora_path):
        with open(lora_path) as fh:
            lora_cfg = json.load(fh)
    else:
        lora_cfg = config.get("lora", {})

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 32),
        lora_alpha=lora_cfg.get("lora_alpha", 64),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Reference model (frozen copy of SFT base)
    ref_model = AutoModelForCausalLM.from_pretrained(base_path)

    # Load dataset
    data_path = config["data"]["train_file"]
    try:
        dpo_data_dict = load_dpo_dataset(
            data_path, tokenizer, max_length=config["data"]["max_length"]
        )
        logger.info(f"Loaded {len(dpo_data_dict['prompt'])} DPO pairs")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if len(dpo_data_dict["prompt"]) == 0:
        logger.error("No DPO pairs loaded. Check dataset format.")
        sys.exit(1)

    # Convert to HF Dataset (required by DPOTrainer)
    try:
        from datasets import Dataset
    except ImportError as exc:
        print(
            "[ERROR] datasets is required for DPO. "
            "Install: pip install datasets>=2.14.0"
        )
        sys.exit(1)

    dpo_dataset = Dataset.from_dict(dpo_data_dict)

    # Training arguments
    train_cfg = config["training"]
    out_dir = config["checkpointing"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_schedule"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg.get("fp16", False),
        logging_steps=config["logging"].get("log_every_n_steps", 10),
        save_steps=config["checkpointing"]["save_every_n_steps"],
        evaluation_strategy="no",
        seed=train_cfg["seed"],
        report_to=["wandb"]
        if os.environ.get("WANDB_DISABLED", "false").lower() != "true"
        else [],
        run_name=config["logging"].get("wandb_run_name"),
        remove_unused_columns=False,
    )

    # DPOTrainer
    try:
        from trl import DPOTrainer
    except ImportError as exc:
        print(
            "[ERROR] trl is required for DPO. "
            "Install: pip install trl>=0.8.0"
        )
        sys.exit(1)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dpo_dataset,
        tokenizer=tokenizer,
        beta=train_cfg.get("beta", 0.1),
    )

    resume = args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=resume)
    if train_result is not None:
        logger.info(f"DPO complete. Final loss: {train_result.training_loss:.4f}")

    final_dir = os.path.join(out_dir, "final_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"DPO adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
