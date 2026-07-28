"""
NeuralAI-Air-135M — Supervised Fine-Tuning (SFT v17)

Self-contained SFT script for the from-scratch 135M base model.
- Inlines the model architecture (GQA + RoPE + RMSNorm + SwiGLU, weight tying).
- Loads base weights from Hugging Face: Subject-Emu-5259/NeuralAI-Air-135M
- ChatML instruction formatting with assistant-only loss masking.
- Mixed-precision full fine-tune (fp16 autocast, fp32 master weights, GradScaler).
- Saves checkpoint and pushes to Hugging Face.

Usage (Colab or local GPU):
    python train_sft_v17.py \
        --data /content/NeuralAI/data/train_sft_v17.jsonl \
        --out_dir /content/sft_model_v17 \
        --hf_repo Subject-Emu-5259/NeuralAI-Air-135M-SFT \
        --push
"""
import os
import json
import math
import argparse
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────
# Model architecture (inlined — no external imports needed)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NeuralAIAir135MConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 15
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    intermediate_size: int = 2560
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    def to_dict(self):
        return asdict(self)


class NeuralAIRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_pos=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos_cached[:seq_len, :], self.sin_cached[:seq_len, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NeuralAIRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class NeuralAIGQAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_group_size = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = NeuralAIRotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, x):
        B, S, C = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, S, C)
        return self.o_proj(attn)


class NeuralAISwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class NeuralAIDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = NeuralAIGQAAttention(config)
        self.mlp = NeuralAISwiGLU(config)
        self.input_layernorm = NeuralAIRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NeuralAIRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x):
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class NeuralAIAir135MModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NeuralAIDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = NeuralAIRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, eos_token_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(input_ids[:, -self.config.max_position_embeddings:])
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return input_ids


# ──────────────────────────────────────────────────────────────────────
# Model + tokenizer loading from Hugging Face
# ──────────────────────────────────────────────────────────────────────

def load_config_from_hf(snapshot_dir):
    with open(os.path.join(snapshot_dir, "config.json")) as f:
        cfg = json.load(f)
    return NeuralAIAir135MConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rms_norm_eps=cfg["rms_norm_eps"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        bos_token_id=cfg.get("bos_token_id", 1),
        eos_token_id=cfg.get("eos_token_id", 2),
        pad_token_id=cfg.get("pad_token_id", 0),
    )


def _load_state_dict(snapshot_dir):
    """Load weights from safetensors or pytorch pickle; handle sharded checkpoints."""
    import glob

    safetensor_files = sorted(glob.glob(os.path.join(snapshot_dir, "*.safetensors")))
    if safetensor_files:
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise RuntimeError("Base checkpoint is in safetensors format; install with: pip install safetensors")
        state = {}
        for path in safetensor_files:
            state.update(load_file(path, device="cpu"))
        return state

    bin_files = sorted(glob.glob(os.path.join(snapshot_dir, "pytorch_model*.bin")))
    if bin_files:
        state = {}
        for path in bin_files:
            part = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(part, dict):
                state.update(part)
            else:
                state.update(part.state_dict() if hasattr(part, "state_dict") else {})
        return state

    raise FileNotFoundError(f"No model weights found in {snapshot_dir} (looked for *.safetensors or pytorch_model*.bin)")


def load_base_model(hf_repo, device):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    print(f"⬇  Downloading base model from {hf_repo} ...")
    snapshot_dir = snapshot_download(repo_id=hf_repo)
    print(f"   snapshot: {snapshot_dir}")

    config = load_config_from_hf(snapshot_dir)
    model = NeuralAIAir135MModel(config)

    state = _load_state_dict(snapshot_dir)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"   ⚠ missing keys: {missing[:6]}{'...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"   ⚠ unexpected keys: {unexpected[:6]}{'...' if len(unexpected) > 6 else ''}")
    # ensure weight tying after load
    if config.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight

    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = config.pad_token_id

    print(f"   ✓ model loaded — {model.count_parameters() / 1e6:.2f}M params")
    return model, config, tokenizer


# ──────────────────────────────────────────────────────────────────────
# ChatML formatting + instruction masking
# ──────────────────────────────────────────────────────────────────────

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_chatml(messages):
    """Convert a list of {role, content} messages into a full ChatML string."""
    parts = []
    for msg in messages:
        parts.append(f"{IM_START}{msg['role']}\n{msg['content'].strip()}{IM_END}")
    return "".join(parts)



class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # accept either {"messages": [...]} or {"prompt": "...", "completion": "..."}
                if "messages" in obj:
                    self.samples.append(obj["messages"])
                elif "prompt" in obj and "completion" in obj:
                    self.samples.append([
                        {"role": "user", "content": obj["prompt"]},
                        {"role": "assistant", "content": obj["completion"]},
                    ])

    def __len__(self):
        return len(self.samples)

    def _tokenize_chunk(self, text):
        return self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]

    def __getitem__(self, idx):
        messages = self.samples[idx]
        input_ids = []
        targets = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()

            header_ids = self._tokenize_chunk(f"{IM_START}{role}\n")
            content_ids = self._tokenize_chunk(content)
            footer_ids = self._tokenize_chunk(IM_END)

            input_ids.extend(header_ids + content_ids + footer_ids)

            if role == "assistant":
                targets.extend([-100] * len(header_ids))
                targets.extend(content_ids)
                targets.extend([-100] * len(footer_ids))
            else:
                targets.extend([-100] * (len(header_ids) + len(content_ids) + len(footer_ids)))

        input_ids = input_ids[: self.max_length]
        targets = targets[: self.max_length]

        # pad/trim to same length (model forward requires matching shapes)
        if len(input_ids) != len(targets):
            length = min(len(input_ids), len(targets))
            input_ids = input_ids[:length]
            targets = targets[:length]

        if len(input_ids) < 2:
            input_ids = [self.tokenizer.pad_token_id, self.tokenizer.pad_token_id]
            targets = [-100, -100]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)
    padded_ids = []
    padded_targets = []
    for b in batch:
        ids = b["input_ids"]
        tgt = b["targets"]
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat((ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)))
            tgt = torch.cat((tgt, torch.full((pad_len,), -100, dtype=torch.long)))
        padded_ids.append(ids)
        padded_targets.append(tgt)
    return torch.stack(padded_ids), torch.stack(padded_targets)


# ──────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────

def get_linear_warmup_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _count_tokens_seen(input_ids, pad_token_id):
    return (input_ids != pad_token_id).sum().item()


# ──────────────────────────────────────────────────────────────────────
# Save + push
# ──────────────────────────────────────────────────────────────────────

def save_model_for_hf(model, tokenizer, config, out_dir, save_dtype=torch.float32):
    from transformers import PreTrainedTokenizerFast
    import shutil

    os.makedirs(out_dir, exist_ok=True)

    # save weights in requested dtype
    final_state = {}
    for name, param in model.named_parameters():
        final_state[name] = param.detach().to(save_dtype).cpu()
    torch.save(final_state, os.path.join(out_dir, "pytorch_model.bin"))

    # copy tokenizer artifacts from source snapshot
    src_dir = getattr(tokenizer, "_processor", None)
    if src_dir is None and hasattr(tokenizer, "vocab_file"):
        src_dir = os.path.dirname(tokenizer.vocab_file) if tokenizer.vocab_file else None
    if src_dir and os.path.isdir(src_dir):
        for fname in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"]:
            src = os.path.join(src_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, fname))

    # config
    cfg = config.to_dict()
    cfg["architectures"] = ["NeuralAIAir135MModel"]
    cfg["model_type"] = "neuralai_air_135m"
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # generation config
    gen_cfg = {
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump(gen_cfg, f, indent=2)

    # README if missing
    readme = os.path.join(out_dir, "README.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write("# NeuralAI-Air-135M-SFT\n\nSupervised fine-tune of NeuralAI-Air-135M.\n")

    print(f"   ✓ saved to {out_dir}")


def push_to_hf(out_dir, repo_id, private=False, token=None):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"⬆  Pushing to Hugging Face: {repo_id} ...")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id)
    print(f"   ✓ pushed to https://huggingface.co/{repo_id}")


# ──────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────

def train_sft(args):
    # device + dtype setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        amp_dtype = torch.float16
        amp_device = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        # MPS autocast does not support fp16; run full fp32
        amp_dtype = None
        amp_device = None
        print("⚠ MPS detected: mixed precision disabled (full fp32).")
    else:
        device = torch.device("cpu")
        amp_dtype = None
        amp_device = None

    print(f"🚀 NeuralAI-Air-135M SFT v17 — device: {device}")
    print(f"   data: {args.data}")
    print(f"   epochs: {args.epochs} | batch: {args.batch_size} | accum: {args.grad_accum} | lr: {args.lr}")

    # base model
    model, config, tokenizer = load_base_model(args.base_repo, device)

    # dataset
    dataset = InstructionDataset(args.data, tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        drop_last=False,
        num_workers=0,
    )
    if len(dataset) == 0:
        raise ValueError(f"No valid training samples found in {args.data}")
    print(f"   samples: {len(dataset)} | steps per epoch: {math.ceil(len(loader) / args.grad_accum)}")

    # optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    total_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
    scheduler = get_linear_warmup_cosine_schedule(optimizer, args.warmup_steps, total_steps)
    use_amp = amp_device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    tokens_seen = 0

    print("\n🏋 Training ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        for batch_idx, (input_ids, targets) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()
            with autocast_ctx:
                logits, loss = model(input_ids, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"   ⚠ step {global_step}: non-finite loss, skipping")
                continue

            loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            running_loss += loss.item() * args.grad_accum
            micro_step += 1
            tokens_seen += _count_tokens_seen(input_ids, tokenizer.pad_token_id)

            if micro_step == args.grad_accum or batch_idx == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                micro_step = 0

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / args.log_every
                    lr = scheduler.get_last_lr()[0]
                    print(f"   epoch {epoch} step {global_step} | loss {avg_loss:.4f} | lr {lr:.2e} | tokens {tokens_seen:,}")
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    ckpt_dir = os.path.join(args.out_dir, f"checkpoint-{global_step}")
                    save_model_for_hf(model, tokenizer, config, ckpt_dir)

        # end-of-epoch checkpoint
        ckpt_dir = os.path.join(args.out_dir, f"checkpoint-epoch-{epoch}")
        save_model_for_hf(model, tokenizer, config, ckpt_dir)

    # final model
    print("\n💾 Saving final model ...")
    save_model_for_hf(model, tokenizer, config, args.out_dir)

    if args.push:
        push_to_hf(args.out_dir, args.hf_repo, private=args.private, token=args.hf_token)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NeuralAI-Air-135M SFT v17")
    parser.add_argument("--data", default="/content/NeuralAI/data/train_sft_v17.jsonl", help="Path to SFT JSONL")
    parser.add_argument("--base_repo", default="Subject-Emu-5259/NeuralAI-Air-135M", help="HF base model repo")
    parser.add_argument("--hf_repo", default="Subject-Emu-5259/NeuralAI-Air-135M-SFT", help="HF target repo")
    parser.add_argument("--out_dir", default="/content/sft_model_v17", help="Local output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--push", action="store_true", help="Push final model to HF")
    parser.add_argument("--private", action="store_true", help="Make HF repo private")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token; falls back to HF_TOKEN env var or cached login")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on CUDA (PyTorch 2.0+)")
    args = parser.parse_args()

    if not args.hf_token:
        args.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # optional torch.compile for CUDA
    if args.compile and torch.cuda.is_available() and hasattr(torch, "compile"):
        print("   ⚡ torch.compile enabled")

    train_sft(args)
    print("\n✅ SFT v17 complete.")


if __name__ == "__main__":
    main()
