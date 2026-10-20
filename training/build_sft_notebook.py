#!/usr/bin/env python3
"""Emit the v4 SFT notebook and kernel metadata after prompt masking + data repair.

SFT v4 sweep — response-only objective + repaired fixtures, rebased onto the v7 step-80000 checkpoint and the 32k v2 tokenizer:
  1. rebase: v7 step-80000 (32k vocab; replaces the 439-token step-32000 base)
  2. LR down: peaks {3e-6, 1e-6, 3e-7} (v2 ran 1e-4 and catastrophically forgot the LM)
  3. warmup(100) + cosine anneal to a 10% floor so every final ckpt is annealed
  4. FineWeb replay: 10-25% of micro-batches drawn from nae1-fineweb-replay-v2
     (pre-tokenized CC-MAIN-2013-20 stream, BOS+ids+EOS — same format as pretrain,
      REBUILT with the v2 tokenizer; the v1 stream is 439-id garbage for this model)
  5. repaired data: nae1-sft-data-v4 (query-specific facts and fixture-matched
     tool results; raw text, so tokenizer-agnostic; synthetic, not human-reviewed)

One kernel runs all three configs sequentially, each FROM the fresh step-80000
base, saving ckpts/{config}/step_{1000,2000,3000} for post-hoc golden+PPL
selection locally (on-kaggle golden eval is impractical on the T4 kernel).

The sweep cell is a template with __TOKEN__ placeholders substituted at build
time (NOT an f-string — the cell is full of literal braces, so token
substitution avoids the brace-escaping class of bug that bit pretrain v3).
Build-time guards (compile + AST undefined-name check) make template bugs fail
locally; scripts/smoke_sft_cell.py executes the generated cell verbatim
against stub datasets so runtime bugs fail locally too.

Usage:
  python3 scripts/build_sft_notebook.py            # full-size sweep (Kaggle)
  python3 scripts/build_sft_notebook.py --steps-per 4 --save-every 2 --accum 1 --seq 64
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "kernel_push_sft_v4"

DEFAULT_CONFIGS = [
    {"name": "lr3e-6_rp10", "lr": 3e-6, "replay": 0.10},
    {"name": "lr1e-6_rp25", "lr": 1e-6, "replay": 0.25},
    {"name": "lr3e-7_rp25", "lr": 3e-7, "replay": 0.25},
]
# Dataset wiring for the v7/v2 lineage. All four slugs are substitutable tokens
# (and --owner prefixes them) so the sweep can be re-pointed at another account
# without editing the CELL template. sft.jsonl is RAW TEXT, tokenized in-kernel,
# so nae1-sft-data-v4 stays valid under the 32k tokenizer; the replay npy is
# PRE-tokenized and therefore must be rebuilt whenever the tokenizer changes.
DEFAULT_OWNER = "takenover"
DEFAULT_BASE_SLUG = "nae1-pretrain-v2-step80000"   # v7 step-80000, 32k vocab
DEFAULT_TOKENIZER_SLUG = "nae1-tokenizer-v2"        # 32k (v1 was 439)
DEFAULT_REPLAY_SLUG = "nae1-fineweb-replay-v2"      # rebuilt for the v2 tokenizer
DEFAULT_SFT_DATA_SLUG = "nae1-sft-data-v4"          # repaired, raw text — tokenizer-agnostic
# The Nae1 embedding table size; a 439-token tokenizer passes every id<32000
# check below silently, so we assert the tokenizer's real vocab directly.
MODEL_VOCAB = 32000
TEMPLATE_TOKENS = ("__STEPS_PER__", "__WARMUP__", "__ACCUM__", "__SAVE_EVERY__",
                   "__SEQ__", "__CONFIGS__", "__BASE_SLUG__", "__TOKENIZER_SLUG__",
                   "__REPLAY_SLUG__", "__SFT_DATA_SLUG__", "__MODEL_VOCAB__")

CELL = r'''
import json, math, os, random, re, sys, time
from pathlib import Path
import torch
from tokenizers import ByteLevelBPETokenizer

OUT = Path("/kaggle/working")
# --- robust dataset discovery (mount layout has varied across runs) ---
_tree = []
if Path("/kaggle/input").exists():
    for p in sorted(Path("/kaggle/input").rglob("*")):
        if len(_tree) < 80: _tree.append(str(p))
print("/kaggle/input tree:", _tree, flush=True)

def _find(fname, slug):
    for c in [Path("/kaggle/input") / slug,
              Path("/kaggle/input/takenover") / slug,
              Path("/kaggle/input/datasets/takenover") / slug,
              Path("/kaggle/input/datasets") / slug]:
        if (c / fname).is_file(): return c / fname
    for h in Path("/kaggle/input").rglob(fname):
        if slug in h.parts and h.is_file(): return h
    return None

CKPT_F = _find("model.safetensors", "__BASE_SLUG__")
if CKPT_F is None:
    raise FileNotFoundError("__BASE_SLUG__ not attached — refusing to re-init")
CKPT_DIR = CKPT_F.parent
DATA_F = _find("sft.jsonl", "__SFT_DATA_SLUG__")
if DATA_F is None:
    raise FileNotFoundError("__SFT_DATA_SLUG__ not attached")
REPLAY_F = _find("fineweb_replay_tokens.npy", "__REPLAY_SLUG__")
if REPLAY_F is None:
    raise FileNotFoundError("__REPLAY_SLUG__ not attached")
print(f"CKPT_DIR={CKPT_DIR}\nDATA_F={DATA_F}\nREPLAY_F={REPLAY_F}", flush=True)

VOCAB_F = _find("nae1_tokenizer-vocab.json", "__TOKENIZER_SLUG__")
MERGES_F = _find("nae1_tokenizer-merges.txt", "__TOKENIZER_SLUG__")
if VOCAB_F is None or MERGES_F is None:
    raise FileNotFoundError("__TOKENIZER_SLUG__ not attached")
tok = ByteLevelBPETokenizer(str(VOCAB_F), str(MERGES_F))
# Fail fast if the wrong tokenizer is attached: the 439-token v1 tokenizer emits
# ids that are all < 32000, so it sails past the max-id asserts below and trains
# the sweep on tokens the model's embeddings never learned. Check real vocab.
_tok_vocab = len(tok.get_vocab())
if _tok_vocab != __MODEL_VOCAB__:
    raise SystemExit(
        f"tokenizer vocab {_tok_vocab} != model vocab __MODEL_VOCAB__ "
        f"(attached __TOKENIZER_SLUG__?) — refusing to train on mismatched ids")
print(f"tokenizer vocab {_tok_vocab} == model vocab __MODEL_VOCAB__", flush=True)
sys.path.insert(0, "/kaggle/input/__TOKENIZER_SLUG__")  # harmless if absent
sys.path.insert(0, "/kaggle/working")
from nae1_model import Nae1ForCausalLM

SEQ = __SEQ__
OBJECTIVE = "assistant-response-only; tool-results-masked"

# ---- tokenize traces and train only on assistant-generated target spans ----
# User prompt/header tokens have label -100 (CrossEntropyLoss ignore_index).
# Tool-result blocks are environment feedback, not model output: keep them in
# the input context but mask their labels. The » call line and final answer remain
# supervised, teaching the model to request a tool and then answer after feedback.
import numpy as np
_ASSISTANT_RE = re.compile(r"(?m)^NeuralAI: ")
_USER_RE = re.compile(r"(?m)^User: ")
_TOOL_BLOCK_RE = re.compile(r'<<tool name="[^\"]+" query="[^\"]*">>.*?<</tool>>', re.S)

def _encode_sft_trace(text):
    encoded = tok.encode(text)
    ids = encoded.ids
    offsets = encoded.offsets
    assistant_headers = list(_ASSISTANT_RE.finditer(text))
    if not assistant_headers:
        raise ValueError("SFT trace has no NeuralAI response header")
    # In the shared chat template, every assistant turn is delimited by its
    # header and the next User turn (or end of trace). This includes the final
    # answer after tool results, while the tool spans are independently masked.
    assistant_spans = []
    for header in assistant_headers:
        next_user = _USER_RE.search(text, header.end())
        end = next_user.start() if next_user else len(text)
        if end > header.end():
            assistant_spans.append((header.end(), end))
    tool_spans = [(m.start(), m.end()) for m in _TOOL_BLOCK_RE.finditer(text)]
    labels = []
    for token_id, (start, end) in zip(ids, offsets):
        assistant_output = any(a <= start and end <= b for a, b in assistant_spans)
        environment_feedback = any(a <= start < b for a, b in tool_spans)
        labels.append(token_id if assistant_output and not environment_feedback else -100)
    # Include the assistant EOS in the actual input stream and its shifted
    # label mask; it is the target immediately after the final response.
    eos_start = len(text)
    ids.append(3)
    labels.append(3 if assistant_spans and assistant_spans[-1][1] == eos_start else -100)
    if not any(label != -100 for label in labels):
        raise ValueError("SFT trace has no supervised assistant output tokens")
    return ids, labels

chunks, mask_chunks, total, supervised = [], [], 0, 0
with DATA_F.open() as fh:
    for line_no, line in enumerate(fh, 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"{DATA_F}:{line_no}: expected a {{'text': str}} record")
        ids, labels = _encode_sft_trace(record["text"])
        chunks.append(np.asarray(ids, dtype=np.uint16))
        mask_chunks.append(np.asarray([label != -100 for label in labels], dtype=np.bool_))
        total += len(ids); supervised += sum(label != -100 for label in labels)
stream = torch.from_numpy(np.concatenate(chunks).astype(np.int64, copy=False))
sft_target_mask = torch.from_numpy(np.concatenate(mask_chunks))
del chunks, mask_chunks
print(f"sft stream: {total:,} tokens, {len(stream):,} ids; "
      f"assistant/EOS targets={supervised:,} ({supervised / total:.1%})", flush=True)

# Exclude windows with no generated targets; torch's mean CE is undefined on an
# all--100 batch. The stream remains packed from multiple traces as in prior SFT runs.
WINDOW_STARTS = [s for s in range(0, len(stream) - SEQ, SEQ)
                 if bool(sft_target_mask[s + 1:s + SEQ + 1].any())]
# Append a final overlapping window when the trailing stream remainder contains
# targets, rather than silently discarding the final few responses.
_final_start = max(0, len(stream) - SEQ - 1)
if (_final_start not in WINDOW_STARTS
        and bool(sft_target_mask[_final_start + 1:_final_start + SEQ + 1].any())):
    WINDOW_STARTS.append(_final_start)
if not WINDOW_STARTS:
    raise ValueError("SFT dataset has no windows with assistant-generated targets")
print(f"SFT target windows: {len(WINDOW_STARTS):,}", flush=True)

# ---- FineWeb replay stream (uint16 npy -> int64; ~30M tokens, ~240MB) ----
replay = torch.from_numpy(np.load(str(REPLAY_F)).astype(np.int64))
print(f"replay stream: {len(replay):,} ids (max id {int(replay.max())})", flush=True)
assert int(replay.max()) < 32000, "replay ids exceed vocab — wrong file?"
assert int(stream.max()) < 32000, "sft ids exceed vocab — wrong tokenizer?"

class W(torch.utils.data.Dataset):
    def __len__(self): return len(WINDOW_STARTS)
    def __getitem__(self, i):
        s = WINDOW_STARTS[i]
        return stream[s:s + SEQ + 1], sft_target_mask[s:s + SEQ + 1]
loader = torch.utils.data.DataLoader(W(), batch_size=1, shuffle=True, drop_last=True)
print(f"sft windows: {len(loader)}", flush=True)

# ---- sweep configs: LR peak x FineWeb replay ratio (the two stall levers) ----
CONFIGS = __CONFIGS__
STEPS_PER, WARMUP, ACCUM, SAVE_EVERY = __STEPS_PER__, __WARMUP__, __ACCUM__, __SAVE_EVERY__

def lr_at(peak, s):
    if s < WARMUP: return peak * (s + 1) / WARMUP
    p = (s - WARMUP) / max(1, STEPS_PER - WARMUP)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))  # cosine -> 10% floor

def next_sft(it_holder):
    try: return next(it_holder[0])
    except StopIteration:
        it_holder[0] = iter(loader); return next(it_holder[0])

saved = []
for ci, cfg in enumerate(CONFIGS):
    print(f"\n=== config {cfg['name']}: lr={cfg['lr']:.0e} replay={cfg['replay']:.2f} ===", flush=True)
    torch.manual_seed(1234 + ci)
    rng = random.Random(1234 + ci)          # replay interleave draws (deterministic per config)
    if ci > 0:
        del model, opt, scaler
        torch.cuda.empty_cache()
    model = Nae1ForCausalLM.from_pretrained(str(CKPT_DIR)).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()
    it_holder = [iter(loader)]
    model.train()
    t0 = time.time(); replay_used = 0; micros = 0
    for step in range(STEPS_PER):
        lr = lr_at(cfg["lr"], step)
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(); loss_acc = 0.0
        for _ in range(ACCUM):
            micros += 1
            if rng.random() < cfg["replay"]:
                s = rng.randrange(0, len(replay) - SEQ - 1)
                batch = replay[s:s + SEQ + 1].unsqueeze(0)
                target_mask = None  # replay is ordinary next-token language modeling
                replay_used += 1
            else:
                batch, target_mask = next_sft(it_holder)
            # Nae1ForCausalLM.forward shifts labels internally; mask target labels
            # at prompt and tool-feedback positions without pre-shifting them.
            batch = batch.to("cuda", dtype=torch.long)
            labels = batch.clone()
            if target_mask is not None:
                # Forward shifts labels internally, so mask exactly the target
                # position for each next-token prediction.
                labels[~target_mask.to("cuda")] = -100
            # Pass all SEQ+1 tokens: Nae1ForCausalLM shifts logits/labels inside
            # forward, so pre-truncating here would drop the final target and can
            # turn a window supervised only at its last token into an all-ignored
            # (NaN) loss batch.
            with torch.cuda.amp.autocast():
                # loss is ALREADY /ACCUM: summing the ACCUM microbatches gives the
                # mean raw loss. The old "* ACCUM" logged 4x the true loss.
                loss = model(input_ids=batch, labels=labels)["loss"] / ACCUM
            scaler.scale(loss).backward(); loss_acc += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if (step + 1) % 100 == 0:
            print(f"{cfg['name']} step {step + 1}/{STEPS_PER} loss={loss_acc:.4f} lr={lr:.2e} "
                  f"{(step + 1) / (time.time() - t0):.1f} it/s "
                  f"replay={replay_used / max(1, micros):.3f}", flush=True)
        if (step + 1) % SAVE_EVERY == 0:
            d = OUT / "ckpts" / cfg["name"] / f"step_{step + 1}"
            model.save_pretrained(str(d), config=model.config)
            (d / "train_state.json").write_text(
                json.dumps({"step": step + 1, "config": cfg["name"],
                            "objective": OBJECTIVE,
                            "lr": cfg["lr"], "replay": cfg["replay"]}))
            saved.append(str(d))
            print(f"saved {d}", flush=True)
    print(f"config {cfg['name']} done: replay fraction used "
          f"{replay_used / max(1, micros):.3f} (target {cfg['replay']:.2f})", flush=True)

print("\nSFT v4 complete")
print(json.dumps({"saved": saved}, indent=1), flush=True)
'''

model_src = (REPO / "scripts" / "nae1_model.py").read_text()
config_src = (REPO / "scripts" / "nae1_config.py").read_text()
CELL1 = ("from pathlib import Path as _P\n"
         "_P(\"/kaggle/working/nae1_model.py\").write_text(" + repr(model_src) + ")\n"
         "_P(\"/kaggle/working/nae1_config.py\").write_text(" + repr(config_src) + ")\n")


def _check_undefined_names(src: str) -> None:
    """Every loaded Name in the cell must be defined in-cell (f-string/template
    leaks like pretrain v3's bare `seq_len` die here, locally)."""
    tree = ast.parse(src, "sft-sweep")
    defined = set(dir(builtins)) | {"__name__", "__file__"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            defined.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Import):
            defined.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            defined.update(a.asname or a.name for a in n.names)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            defined.update(n.names)
    undefined = sorted({n.id for n in ast.walk(tree)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                       - defined)
    if undefined:
        raise SystemExit(f"undefined name(s) in generated sft cell: {undefined}")


def build(steps_per: int = 3000, warmup: int = 100, accum: int = 4,
          save_every: int = 1000, seq: int = 2048,
          configs=None, base_slug: str = DEFAULT_BASE_SLUG,
          tokenizer_slug: str = DEFAULT_TOKENIZER_SLUG,
          replay_slug: str = DEFAULT_REPLAY_SLUG,
          sft_data_slug: str = DEFAULT_SFT_DATA_SLUG,
          model_vocab: int = MODEL_VOCAB) -> dict:
    configs = configs if configs is not None else DEFAULT_CONFIGS
    cell = (CELL
            .replace("__STEPS_PER__", str(steps_per))
            .replace("__WARMUP__", str(warmup))
            .replace("__ACCUM__", str(accum))
            .replace("__SAVE_EVERY__", str(save_every))
            .replace("__SEQ__", str(seq))
            .replace("__CONFIGS__", json.dumps(configs))
            .replace("__BASE_SLUG__", base_slug)
            .replace("__TOKENIZER_SLUG__", tokenizer_slug)
            .replace("__REPLAY_SLUG__", replay_slug)
            .replace("__SFT_DATA_SLUG__", sft_data_slug)
            .replace("__MODEL_VOCAB__", str(model_vocab)))
    leftover = [t for t in TEMPLATE_TOKENS if t in cell]
    if leftover:
        raise SystemExit(f"template token(s) not substituted: {leftover}")
    compile(cell, "sft-sweep", "exec")
    _check_undefined_names(cell)

    # nbformat 4.5+ requires a unique id per cell (^[a-zA-Z0-9-_]+$); without them
    # Kaggle logs MissingIDFieldWarning on every push and validators will hard-fail
    # in future nbformat releases. Ids are deterministic so rebuilds are diff-clean.
    return {"cells": [
        {"cell_type": "markdown", "id": "sft-v4-intro", "metadata": {},
         "source": ["# Nae1 SFT v4 — response-only objective + repaired fixtures (rebased on v7 step-80000)\n",
                    "\n",
                    f"Base `{base_slug}` (v7, step 80000, 32k vocab), tokenizer "
                    f"`{tokenizer_slug}` (32k),\n",
                    f"replay `{replay_slug}` (rebuilt for the v2 tokenizer), data "
                    f"`{sft_data_slug}` (raw text).\n",
                    "\n",
                    "Objective: loss only on assistant-generated output; prompt tokens and injected tool results are masked.\n",
                    "AdamW weight decay 0.1 + 10–25% FineWeb replay regularize against forgetting.\n",
                    "Configs (each trained from the fresh step-80000 base):\n",
                    "1. `lr3e-6_rp10` — LR peak 3e-6, 10% FineWeb replay\n",
                    "2. `lr1e-6_rp25` — LR peak 1e-6, 25% FineWeb replay\n",
                    "3. `lr3e-7_rp25` — LR peak 3e-7, 25% FineWeb replay\n",
                    "\n",
                    f"Warmup {warmup} + cosine to a 10% LR floor; {steps_per} steps each; "
                    f"checkpoints at\n",
                    "1000/2000/3000 per config for post-hoc golden + held-out PPL selection."]},
        {"cell_type": "code", "id": "write-model-src", "metadata": {},
         "execution_count": None, "outputs": [], "source": [CELL1]},
        {"cell_type": "code", "id": "sft-sweep", "metadata": {},
         "execution_count": None, "outputs": [], "source": [cell]},
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11"}},
     "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps-per", type=int, default=3000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--owner", default=DEFAULT_OWNER,
                    help="Kaggle owner for the dataset + kernel ids (e.g. takenover, "
                         "subjectemu5259) — use the account that actually has quota")
    ap.add_argument("--base-slug", default=DEFAULT_BASE_SLUG)
    ap.add_argument("--tokenizer-slug", default=DEFAULT_TOKENIZER_SLUG)
    ap.add_argument("--replay-slug", default=DEFAULT_REPLAY_SLUG)
    ap.add_argument("--sft-data-slug", default=DEFAULT_SFT_DATA_SLUG)
    ap.add_argument("--kernel-slug", default="nae1-sft-v4-sweep")
    a = ap.parse_args()

    nb = build(a.steps_per, a.warmup, a.accum, a.save_every, a.seq,
               base_slug=a.base_slug, tokenizer_slug=a.tokenizer_slug,
               replay_slug=a.replay_slug, sft_data_slug=a.sft_data_slug)
    a.out.mkdir(exist_ok=True)
    (a.out / "nae1-sft-v4.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    # Attached datasets must be visible to the account that pushes/runs the kernel.
    dataset_sources = [f"{a.owner}/{a.tokenizer_slug}",
                        f"{a.owner}/{a.base_slug}",
                        f"{a.owner}/{a.sft_data_slug}",
                        f"{a.owner}/{a.replay_slug}"]
    (a.out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{a.owner}/{a.kernel_slug}", "title": "Nae1 SFT V4 Sweep",
        "code_file": "nae1-sft-v4.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": dataset_sources,
        "competition_sources": [], "kernel_sources": [], "model_sources": []},
        indent=2) + "\n")
    print("wrote", a.out)
    print("dataset_sources:", json.dumps(dataset_sources))
    print("kernel id:", f"{a.owner}/{a.kernel_slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
