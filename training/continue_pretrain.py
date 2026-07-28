"""
Continue pre-training NeuralAI-Air-135M from a checkpoint.

Loads a prior PyTorch state_dict/config (e.g. from train_hf.py) and trains on
fast memmapped token .bin files produced by pretrain_data_pipeline.py.

Usage:
    python training/continue_pretrain.py \
        --checkpoint NeuralAI-Air-135M/checkpoints/final.pt \
        --token_dir data/pretrain_tokens \
        --out_dir checkpoints/neuralair-135m-continued \
        --steps 20000 --batch 8 --seq 512 --lr 3e-4
"""
import os
import math
import time
import argparse
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import importlib.util
_MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "NeuralAI-Air-135M/NeuralAI-Air-135M.py")
_spec = importlib.util.spec_from_file_location("neuralai_air_135m_internal", _MODEL_FILE)
_neuralai_air_135m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_neuralai_air_135m)
NeuralAIAir135MConfig = _neuralai_air_135m.NeuralAIAir135MConfig
NeuralAIAir135MModel = _neuralai_air_135m.NeuralAIAir135MModel
get_device = _neuralai_air_135m.get_device


@dataclass
class TrainConfig:
    token_dir: str = "data/pretrain_tokens"
    batch_size: int = 8
    seq_len: int = 512
    max_steps: int = 20_000
    warmup_steps: int = 1_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 100
    save_every: int = 2_000
    eval_every: int = 1_000
    output_dir: str = "checkpoints/neuralair-135m-continued"
    checkpoint: str = ""
    resume_optimizer: bool = True


def get_cosine_lr(step, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class MemmapTokenDataset(Dataset):
    """Reads uint16 token files with optional shuffled sequence starts."""

    def __init__(self, token_dir: str, seq_len: int):
        self.seq_len = seq_len
        self.files = [
            os.path.join(token_dir, f)
            for f in os.listdir(token_dir)
            if f.endswith(".bin")
        ]
        if not self.files:
            raise FileNotFoundError(f"No .bin token files in {token_dir}")

        self.mmaps = []
        self.cumlen = []
        total = 0
        for path in self.files:
            tokens = np.memmap(path, dtype=np.uint16, mode="r")
            usable = max(0, len(tokens) - seq_len - 1)
            self.mmaps.append(tokens)
            total += usable
            self.cumlen.append(total)
        self.total = total
        print(f"Loaded {len(self.files)} token files, {self.total:,} usable sequences.")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Find which file
        file_idx = 0
        while idx >= self.cumlen[file_idx]:
            file_idx += 1
        start = idx if file_idx == 0 else idx - self.cumlen[file_idx - 1]
        tokens = self.mmaps[file_idx]
        x = tokens[start : start + self.seq_len + 1]
        x = torch.from_numpy(x.astype(np.int64))
        return {"input_ids": x[:-1], "labels": x[1:]}


def load_checkpoint(path: str, model, optimizer, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        config_dict = ckpt.get("config")
        start_step = ckpt.get("step", 0)
        opt_state = ckpt.get("optimizer")
    else:
        state = ckpt
        config_dict = None
        start_step = 0
        opt_state = None

    if config_dict:
        config = NeuralAIAir135MConfig(**config_dict)
        model.load_state_dict(state, strict=False)
    else:
        config = NeuralAIAir135MConfig()
        model.load_state_dict(state, strict=False)

    if optimizer is not None and opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
            print("Resumed optimizer state.")
        except Exception as e:
            print(f"Could not resume optimizer state: {e}")

    print(f"Loaded checkpoint from step {start_step}.")
    return start_step, config


def evaluate(model, loader, device, batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _, loss = model(batch["input_ids"], batch["labels"])
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--token_dir", default="data/pretrain_tokens")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--save_every", type=int, default=2_000)
    parser.add_argument("--eval_every", type=int, default=1_000)
    parser.add_argument("--out_dir", default="checkpoints/neuralair-135m-continued")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reset_optimizer", action="store_true", help="Start optimizer from scratch.")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    cfg = TrainConfig(
        checkpoint=args.checkpoint,
        token_dir=args.token_dir,
        batch_size=args.batch,
        seq_len=args.seq,
        max_steps=args.steps,
        warmup_steps=args.warmup,
        lr=args.lr,
        min_lr=args.min_lr,
        save_every=args.save_every,
        eval_every=args.eval_every,
        output_dir=args.out_dir,
        resume_optimizer=not args.reset_optimizer,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    model_config = NeuralAIAir135MConfig()
    model = NeuralAIAir135MModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)

    start_step, loaded_config = load_checkpoint(cfg.checkpoint, model, optimizer if cfg.resume_optimizer else None, device)
    if loaded_config:
        model_config = loaded_config

    print(f"Model: {model.count_parameters() / 1e6:.2f}M params")
    print(f"Device: {device}")

    dataset = MemmapTokenDataset(cfg.token_dir, cfg.seq_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    iter_loader = iter(loader)

    # Simple repeating iterator; DataLoader shuffle handles randomization across epochs.
    def next_batch():
        nonlocal iter_loader
        try:
            return next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            return next(iter_loader)

    model.train()
    step = start_step
    start_time = time.time()

    while step < cfg.max_steps:
        batch = next_batch()
        batch = {k: v.to(device) for k, v in batch.items()}

        for g in optimizer.param_groups:
            g["lr"] = get_cosine_lr(step, cfg)

        optimizer.zero_grad()
        logits, loss = model(batch["input_ids"], batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0:
            elapsed = time.time() - start_time
            print(f"Step {step} | loss {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if step > 0 and step % cfg.save_every == 0:
            save_path = os.path.join(cfg.output_dir, f"checkpoint-{step}.pt")
            torch.save({
                "step": step,
                "state_dict": model.state_dict(),
                "config": model_config.to_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            print(f"Saved checkpoint -> {save_path}")

        if step > 0 and step % cfg.eval_every == 0:
            eval_loss = evaluate(model, loader, device, batches=20)
            print(f"Eval loss: {eval_loss:.4f}")

        step += 1

    final_path = os.path.join(cfg.output_dir, "final.pt")
    torch.save({
        "step": step,
        "state_dict": model.state_dict(),
        "config": model_config.to_dict(),
        "optimizer": optimizer.state_dict(),
    }, final_path)
    print(f"Continued pretraining complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
