"""SFT entrypoint for NeuralAI-Air-135M v19.

Loads the best pre-training checkpoint (lowest validation perplexity),
reads ChatML-formatted SFT data, masks user/system tokens so loss is
computed only on assistant replies, and fine-tunes with ``SFTTrainer``
from TRL (falling back to standard ``Trainer`` if TRL is unavailable).

Typical usage::

    python -m training.sft.run_sft --config training/sft/config_sft.yaml
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
# PyTorch / transformers are not installed.
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
        Trainer,
    )
    from transformers.trainer_utils import get_last_checkpoint
except ImportError:
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TrainingArguments = None  # type: ignore
    DataCollatorForSeq2Seq = None  # type: ignore
    Trainer = None  # type: ignore
    get_last_checkpoint = None  # type: ignore


def load_sft_dataset(
    path: str, tokenizer, max_length: int = 1024
) -> List[Dict[str, Any]]:
    """Load SFT examples and format into ``{input_ids, labels, attention_mask}``.

    Each JSONL line may contain:
      - ``text`` (pre-formatted ChatML)
      - ``system``, ``instruction``, ``output`` (structured)

    User and system tokens are masked with ``-100``. Only assistant tokens
    contribute to the loss. Sequences are padded to *max_length* and
    truncated from the **left** if they exceed it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"SFT dataset not found: {path}")

    examples: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            system = obj.get("system", "")
            instruction = obj.get("instruction", "")
            output = obj.get("output", "")
            text = obj.get("text", "")

            input_ids: List[int] = []
            labels: List[int] = []

            if system:
                sys_text = f"<|im_start|>system\n{system}<|im_end|>"
                sys_ids = tokenizer.encode(sys_text, add_special_tokens=False)
                input_ids.extend(sys_ids)
                labels.extend([-100] * len(sys_ids))

            if instruction:
                user_text = f"<|im_start|>user\n{instruction}<|im_end|>"
                user_ids = tokenizer.encode(user_text, add_special_tokens=False)
                input_ids.extend(user_ids)
                labels.extend([-100] * len(user_ids))

            if output:
                assistant_text = f"<|im_start|>assistant\n{output}<|im_end|>"
                assistant_ids = tokenizer.encode(
                    assistant_text, add_special_tokens=False
                )
                input_ids.extend(assistant_ids)
                labels.extend(assistant_ids)

            if text and not (system or instruction or output):
                all_ids = tokenizer.encode(text, add_special_tokens=False)
                input_ids = all_ids
                mask_point = int(len(all_ids) * 0.6)
                labels = [-100] * mask_point + all_ids[mask_point:]

            if not input_ids:
                continue

            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                pad_id = tokenizer.pad_token_id or 0
                input_ids = [pad_id] * pad_len + input_ids
                labels = [-100] * pad_len + labels

            attention_mask = [
                1 if tid != (tokenizer.pad_token_id or 0) else 0
                for tid in input_ids
            ]

            examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
            )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuralAI-Air-135M SFT v19"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/sft/config_sft.yaml",
        help="Path to SFT config YAML",
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

    # Load best pretrain checkpoint
    base_path = config["model"]["base_model_path"]
    if not os.path.isdir(base_path):
        from training.common.checkpoint_utils import (
            find_best_checkpoint,
            find_latest_checkpoint,
        )

        best = find_best_checkpoint("checkpoints/pretrain", metric="eval_loss")
        if best:
            base_path = best
        else:
            latest = find_latest_checkpoint("checkpoints/pretrain")
            if latest:
                base_path = latest
            else:
                print(
                    f"[ERROR] Base model not found at {base_path} "
                    f"and no pretrain checkpoints exist."
                )
                sys.exit(1)

    logger.info(f"Loading base model from {base_path}")
    model = AutoModelForCausalLM.from_pretrained(base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # Load dataset
    train_path = config["data"]["train_file"]
    try:
        examples = load_sft_dataset(
            train_path, tokenizer, max_length=config["data"]["max_length"]
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    if len(examples) == 0:
        logger.error("No SFT examples loaded. Check dataset format.")
        sys.exit(1)

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, data: List[Dict[str, Any]]):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            return self.data[idx]

    train_dataset = SFTDataset(examples)
    logger.info(f"Loaded {len(train_dataset)} SFT examples")

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
        weight_decay=train_cfg["weight_decay"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        adam_epsilon=train_cfg["adam_eps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg.get("fp16", False),
        logging_steps=config["logging"].get("log_every_n_steps", 10),
        save_steps=config["checkpointing"]["save_every_n_steps"],
        evaluation_strategy=config["checkpointing"]["eval_strategy"],
        seed=train_cfg["seed"],
        report_to=["wandb"]
        if os.environ.get("WANDB_DISABLED", "false").lower() != "true"
        else [],
        run_name=config["logging"].get("wandb_run_name"),
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        padding="longest",
        max_length=config["data"]["max_length"],
    )

    # Try TRL SFTTrainer first, fall back to standard Trainer
    try:
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            dataset_text_field=None,
            data_collator=collator,
        )
        logger.info("Using TRL SFTTrainer")
    except ImportError:
        logger.warning("trl not installed; falling back to standard Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )

    resume = args.resume_from_checkpoint
    if resume is None and get_last_checkpoint is not None:
        resume = get_last_checkpoint(out_dir)

    train_result = trainer.train(resume_from_checkpoint=resume)
    if train_result is not None:
        logger.info(f"SFT complete. Final loss: {train_result.training_loss:.4f}")
    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"SFT model saved to {final_dir}")


if __name__ == "__main__":
    main()
