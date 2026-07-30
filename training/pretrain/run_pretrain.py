"""Main pre-training entrypoint for NeuralAI-Air-135M.

Initializes a ``LlamaForCausalLM`` from ``config.json`` with **RANDOM**
weights, loads a ``StreamingDataset`` from ``uint16`` ``.bin`` shards, and
trains with the HuggingFace ``Trainer`` using CEO-approved hyperparameters.

Typical usage::

    python -m training.pretrain.run_pretrain --config training/pretrain/config_pretrain.yaml
    python -m training.pretrain.run_pretrain --config training/pretrain/config_pretrain.yaml --resume_from_checkpoint checkpoints/pretrain/checkpoint-1000
"""

import os
import sys
import argparse
import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Lazy-load heavy dependencies so the module imports cleanly even when
# PyTorch / transformers are not installed (e.g. local structure verification).
try:
    import numpy as np
    import torch
    from transformers import (
        LlamaForCausalLM,
        LlamaConfig,
        PreTrainedTokenizerFast,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from transformers.trainer_utils import get_last_checkpoint
except ImportError:
    np = None  # type: ignore
    torch = None  # type: ignore
    LlamaForCausalLM = None  # type: ignore
    LlamaConfig = None  # type: ignore
    PreTrainedTokenizerFast = None  # type: ignore
    TrainingArguments = None  # type: ignore
    Trainer = None  # type: ignore
    DataCollatorForLanguageModeling = None  # type: ignore
    get_last_checkpoint = None  # type: ignore


class StreamingDataset:
    """Memory-mapped dataset reading flat ``uint16`` ``.bin`` shards.

    Each shard is a flat array of token ids. The dataset reshapes it into
    contiguous ``(seq_len,)`` blocks without cross-document masking (the
    causal LM learns ``<|im_end|>`` as a hard boundary).
    """

    def __init__(self, data_dir: str, seq_len: int = 512, seed: int = 42):
        if torch is None or np is None:
            raise RuntimeError("PyTorch and NumPy are required to use StreamingDataset.")
        self.seq_len = seq_len
        self.data_dir = Path(data_dir)
        self.shards: list = []
        self.shard_sizes: list = []
        self.total_tokens = 0

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}. "
                "Run data_pipeline.py first to generate tokenized shards."
            )

        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            import json

            with open(manifest_path) as fh:
                manifest = json.load(fh)
            for entry in manifest.get("shards", []):
                path = Path(entry["file"])
                if not path.exists():
                    path = self.data_dir / path.name
                if not path.exists():
                    raise FileNotFoundError(f"Shard file not found: {path}")
                mem = np.memmap(str(path), dtype=np.uint16, mode="r")
                self.shards.append(mem)
                size = len(mem)
                self.shard_sizes.append(size)
                self.total_tokens += size
        else:
            for path in sorted(self.data_dir.glob("*.bin")):
                mem = np.memmap(str(path), dtype=np.uint16, mode="r")
                self.shards.append(mem)
                size = len(mem)
                self.shard_sizes.append(size)
                self.total_tokens += size

        if not self.shards:
            raise ValueError(
                f"No .bin shards found in {data_dir}. "
                "Run data_pipeline.py first."
            )

        self.num_sequences = self.total_tokens // self.seq_len
        self.cumsum = [0]
        for size in self.shard_sizes:
            self.cumsum.append(self.cumsum[-1] + size)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Index {idx} out of range for {self.num_sequences} sequences"
            )
        start_token = idx * self.seq_len
        shard_idx = 0
        while (
            shard_idx + 1 < len(self.cumsum)
            and self.cumsum[shard_idx + 1] <= start_token
        ):
            shard_idx += 1
        offset = start_token - self.cumsum[shard_idx]
        shard = self.shards[shard_idx]
        end = offset + self.seq_len
        seq = shard[offset:end].astype(np.int64)
        if len(seq) < self.seq_len:
            pad = np.zeros(self.seq_len - len(seq), dtype=np.int64)
            seq = np.concatenate([seq, pad])
        attention_mask = (seq != 0).astype(np.int64)
        return {
            "input_ids": seq,
            "labels": seq.copy(),
            "attention_mask": attention_mask,
        }


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def compute_flops_per_step(
    batch_size: int, seq_len: int, num_params: int
) -> int:
    """Rough FLOP estimate for a decoder-only transformer training step.

    Uses the heuristic ``6 * P * D`` tokens (forward + backward) where
    ``D = batch_size * seq_len``.
    """
    tokens = batch_size * seq_len
    return 6 * num_params * tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuralAI-Air-135M Pre-Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/pretrain/config_pretrain.yaml",
        help="Path to pretrain config YAML",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    if torch is None or np is None:
        print(
            "[ERROR] PyTorch and NumPy are required but not installed. "
            "Install: pip install torch>=2.1.0 numpy>=1.24.0"
        )
        sys.exit(1)

    config = load_config(args.config)

    # Device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(
        torch,
        config["model"].get("torch_dtype", "bfloat16"),
        torch.float32,
    )
    if device == "cpu" and torch_dtype == torch.bfloat16:
        torch_dtype = torch.float32
        logger.warning("CPU detected: falling back to float32")

    # Model init (random weights — from_config, NOT from_pretrained)
    model_path = config["model"]["config_path"]
    if not os.path.isdir(model_path):
        print(f"[ERROR] Model config directory not found: {model_path}")
        sys.exit(1)

    hf_config = LlamaConfig.from_pretrained(model_path)
    use_flash = (
        config["model"].get("use_flash_attention", True) and device == "cuda"
    )
    attn_impl = "flash_attention_2" if use_flash else "eager"
    model = LlamaForCausalLM.from_config(
        hf_config,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
    )
    if config["model"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    if (
        config["model"].get("torch_compile", False)
        and hasattr(torch, "compile")
        and device == "cuda"
    ):
        logger.info("Enabling torch.compile...")
        model = torch.compile(model)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f"Model initialized: {total_params:,} total params, "
        f"{trainable_params:,} trainable"
    )
    logger.info(
        f"Using device={device}, dtype={torch_dtype}, attn={attn_impl}"
    )

    # Tokenizer
    tokenizer_path = config["tokenizer"]["tokenizer_path"]
    tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        print(f"[ERROR] tokenizer.json not found at {tokenizer_path}")
        sys.exit(1)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    # Datasets
    data_cfg = config["data"]
    train_dir = os.path.join(data_cfg["data_dir"], "tokenized", "train")
    val_dir = os.path.join(data_cfg["data_dir"], "tokenized", "val")

    train_dataset: Optional[StreamingDataset] = None
    val_dataset: Optional[StreamingDataset] = None
    try:
        train_dataset = StreamingDataset(
            train_dir,
            seq_len=config["training"]["context_length"],
            seed=config["training"]["seed"],
        )
        logger.info(f"Train dataset: {len(train_dataset):,} sequences")
    except Exception as exc:
        logger.error(f"Failed to load train dataset: {exc}")
    if os.path.isdir(val_dir):
        try:
            val_dataset = StreamingDataset(
                val_dir,
                seq_len=config["training"]["context_length"],
                seed=config["training"]["seed"],
            )
            logger.info(f"Val dataset: {len(val_dataset):,} sequences")
        except Exception as exc:
            logger.warning(f"Failed to load val dataset: {exc}")

    if train_dataset is None:
        logger.error(
            "No train dataset available. "
            "Ensure data/pretrain/tokenized/train/ contains .bin shards."
        )
        sys.exit(1)

    # Training arguments
    train_cfg = config["training"]
    out_dir = config["checkpointing"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=False,
        max_steps=train_cfg["max_steps"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["peak_lr"],
        weight_decay=train_cfg["weight_decay"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        adam_epsilon=train_cfg["adam_eps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_schedule"],
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        logging_steps=config["logging"].get("log_every_n_steps", 100),
        save_steps=config["checkpointing"]["save_every_n_steps"],
        eval_steps=config["checkpointing"]["eval_every_n_steps"],
        evaluation_strategy="steps" if val_dataset is not None else "no",
        save_total_limit=config["checkpointing"].get("keep_n_best", 3),
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=train_cfg["seed"],
        report_to=["wandb"]
        if os.environ.get("WANDB_DISABLED", "false").lower() != "true"
        else [],
        run_name=config["logging"].get("wandb_run_name", "135M-1B"),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Data collator (causal LM — no masked language modeling)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # WandB
    try:
        import wandb

        wandb.init(
            project=config["logging"]["wandb_project"],
            name=config["logging"].get("wandb_run_name"),
            config=config,
        )
    except Exception as exc:
        logger.warning(f"WandB init failed: {exc}")

    # Resume logic
    resume_path = args.resume_from_checkpoint
    if resume_path is None and get_last_checkpoint is not None:
        resume_path = get_last_checkpoint(out_dir)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Estimates
    tokens_per_step = (
        train_cfg["per_device_batch_size"]
        * train_cfg["gradient_accumulation_steps"]
        * train_cfg["context_length"]
    )
    total_tokens_target = train_cfg.get(
        "total_tokens", tokens_per_step * train_cfg["max_steps"]
    )
    est_flops = compute_flops_per_step(
        train_cfg["per_device_batch_size"],
        train_cfg["context_length"],
        total_params,
    )
    est_hours = total_tokens_target / (tokens_per_step * max(1, 6000)) / 3600

    logger.info(f"Tokens/step:          {tokens_per_step:,}")
    logger.info(f"Target tokens:        {total_tokens_target:,}")
    logger.info(f"Est FLOPs/step:       {est_flops:,.0e}")
    logger.info(f"Est wall time (hrs):  ~{est_hours:.1f} (at 6k tok/s)")
    logger.info(f"Resume checkpoint:    {resume_path}")

    # Train
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    elapsed = time.time() - start_time

    steps_done = trainer.state.global_step
    tokens_processed = steps_done * tokens_per_step
    tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0

    logger.info(f"Training complete: {steps_done} steps in {elapsed/3600:.2f}h")
    logger.info(f"Tokens/sec: {tokens_per_sec:,.1f}")
    if train_result is not None:
        logger.info(f"Final loss: {train_result.training_loss:.4f}")

    # Save final
    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
