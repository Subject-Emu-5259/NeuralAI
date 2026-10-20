#!/usr/bin/env python3
"""Local dry-run smoke test for the SFT sweep notebook (kernel_push_sft_v3).

Executes the *generated* notebook's code cells VERBATIM (byte-for-byte what
Kaggle runs) inside a sandbox with tiny stub datasets, so template/runtime bugs
fail here in seconds instead of on a paid T4 session. Mirrors the pretrain
smoke (scripts/smoke_pretrain_cell.py) and reuses its shims.

What is real, what is shimmed:
  REAL   - both generated cells: the model/config source writer, dataset
           discovery (_find), tokenizer load + the tokenizer-vocab guard, SFT
           jsonl tokenization (real 32k v2 BPE from nae1_tokenizer_v2/), replay
           npy load, window/loader, the 3-config sweep loop (fresh
           from_pretrained per config, replay interleave, LR schedule,
           checkpoint saves)
  REAL   - the base checkpoint is a real Nae1ForCausalLM saved by the repo's
           own classes (tiny: n_embd=64, n_layer=2) so from_pretrained is
           exercised end-to-end without a 400MB download
  SHIMMED- /kaggle/* -> sandbox dir (pathlib.Path subclass from the pretrain
           smoke), device "cuda" -> "cpu", cuda amp -> disabled CPU scaler /
           nullcontext

Usage:
  python3 scripts/smoke_sft_cell.py
  python3 scripts/smoke_sft_cell.py --keep    # keep the sandbox dir
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))  # for the shared shim module

import smoke_pretrain_cell as pretrain_smoke  # noqa: E402  (shim reuse)

# Slugs the generated cell asks for — single-sourced from the builder so a
# rewiring there can't drift out of sync with what this harness seeds.
from build_sft_notebook import (DEFAULT_BASE_SLUG, DEFAULT_TOKENIZER_SLUG,  # noqa: E402
                                DEFAULT_REPLAY_SLUG, DEFAULT_SFT_DATA_SLUG)

CONFIGS = [{"name": "smokeA_rp10", "lr": 3e-6, "replay": 0.10},
           {"name": "smokeB_rp25", "lr": 1e-6, "replay": 0.25}]

SFT_LINES = [
    json.dumps({"text": (
        f"User message {i}: please explain how the weather forecast works today."
        "\nNeuralAI: A forecast estimates future atmospheric conditions using observations, "
        "physical models, and uncertainty.\n")})
    for i in range(80)
]


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_sft_notebook", REPO / "scripts" / "build_sft_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tiny_ckpt(dest: Path) -> None:
    """A real Nae1 checkpoint (tiny config) via the repo's own classes."""
    import torch
    from nae1_config import Nae1Config
    from nae1_model import Nae1ForCausalLM
    torch.manual_seed(0)
    cfg = Nae1Config(vocab_size=32000, n_positions=2048, n_embd=64, n_layer=2,
                     n_head=4, n_kv_heads=2, bos_token_id=2, eos_token_id=3,
                     pad_token_id=0, tie_word_embeddings=True)
    model = Nae1ForCausalLM(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dest), config=cfg)


def seed_inputs(sandbox: Path, base_slug: str = DEFAULT_BASE_SLUG,
                tokenizer_slug: str = DEFAULT_TOKENIZER_SLUG,
                replay_slug: str = DEFAULT_REPLAY_SLUG,
                sft_data_slug: str = DEFAULT_SFT_DATA_SLUG) -> None:
    """Seed the four datasets the cell looks for, under the slugs it asks for.

    Pass the *old* tokenizer_slug/base_slug to reproduce a mismatched wiring.
    """
    import numpy as np
    inp = sandbox / "input" / "takenover"          # second _find() candidate
    make_tiny_ckpt(inp / base_slug)                 # tiny 32k-vocab ckpt
    d = inp / sft_data_slug; d.mkdir(parents=True)
    (d / "sft.jsonl").write_text("\n".join(SFT_LINES) + "\n")
    d = inp / replay_slug; d.mkdir(parents=True)
    # Real replay ids are < 32000 (the v2 vocab). The OLD corpus was built with
    # the 439-token tokenizer, which also satisfies id<32000 — that is exactly
    # why the vocab guard exists.
    ids = np.random.default_rng(0).integers(5, 32000, size=50_000, dtype=np.uint16)
    np.save(str(d / "fineweb_replay_tokens.npy"), ids)
    tok_src = REPO / ("nae1_tokenizer_v2" if tokenizer_slug == DEFAULT_TOKENIZER_SLUG
                      else "nae1_tokenizer")
    d = inp / tokenizer_slug; d.mkdir(parents=True)
    for f in ("nae1_tokenizer-vocab.json", "nae1_tokenizer-merges.txt"):
        shutil.copy(tok_src / f, d / f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    builder = load_builder()
    sandbox = Path(tempfile.mkdtemp(prefix="nae1_sft_smoke_"))
    failures = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            failures.append(msg)

    try:
        # Tiny build: 4 steps, save every 2 -> two saves per config.
        nb = builder.build(steps_per=4, warmup=2, accum=1, save_every=2, seq=64,
                           configs=CONFIGS)
        assert [c["id"] for c in nb["cells"]] == ["sft-v4-intro", "write-model-src", "sft-sweep"]

        # Build the real tiny ckpt + stub datasets BEFORE installing the shims
        # (uses the repo classes directly; the cell's own import must load the
        # cell-written copy from the sandbox, so drop our cached modules).
        seed_inputs(sandbox)
        for name in ("nae1_model", "nae1_config"):
            sys.modules.pop(name, None)
        while str(REPO / "scripts") in sys.path:
            sys.path.remove(str(REPO / "scripts"))

        pretrain_smoke.HOLDER["root"] = sandbox
        pretrain_smoke.install_shims(stub_dir=sandbox / "_stubs")  # hf stub unused here

        os.chdir(sandbox)
        sys.path.insert(0, str(sandbox))  # /kaggle/working equivalent for nae1_model import

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                for cell in nb["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    src = cell["source"]
                    src = "".join(src) if isinstance(src, list) else src
                    exec(compile(src, cell["id"], "exec"), {"__name__": "__main__"})
        except Exception:
            print("---- cells stdout until failure ----")
            print(buf.getvalue())
            raise
        out = buf.getvalue()

        check("sft stream:" in out, "cells: SFT jsonl tokenized into a stream")
        check("SFT target windows:" in out, "response-loss windows selected")
        check("sft windows:" in out, "cells: window loader built")
        for cfg in CONFIGS:
            check(f"=== config {cfg['name']}" in out, f"cells: config {cfg['name']} started")
            check(f"config {cfg['name']} done: replay fraction used" in out,
                  f"cells: config {cfg['name']} finished with replay accounting")
        for cfg in CONFIGS:
            for step in (2, 4):
                d = sandbox / "ckpts" / cfg["name"] / f"step_{step}"
                check((d / "model.safetensors").is_file()
                      and (d / "train_state.json").is_file(),
                      f"ckpt {cfg['name']}/step_{step} written")
                state = json.loads((d / "train_state.json").read_text())
                check(state.get("step") == step and state.get("config") == cfg["name"]
                      and state.get("objective") == "assistant-response-only; tool-results-masked"
                      and state.get("lr") == cfg["lr"] and state.get("replay") == cfg["replay"],
                      f"ckpt {cfg['name']}/step_{step} train_state {state}")
                check(state["objective"] == "assistant-response-only; tool-results-masked",
                      f"ckpt {cfg['name']}/step_{step} records the masked objective")
        sizes = [p.stat().st_size for p in (sandbox / "ckpts").glob("*/*/model.safetensors")]
        check(len(sizes) == 4 and all(s < 60_000_000 for s in sizes),
              f"tiny base ckpt used (4 files < 60MB, got {sizes})")
        check("SFT v4 complete" in out, "cells: sweep completed")
        tail = out[out.rindex("SFT v4 complete"):]
        saved_list = json.loads(tail[tail.index("{"):])["saved"]
        check(len(saved_list) == 4, f"final saved list has 4 ckpts (2 configs x 2 saves, got {len(saved_list)})")
    finally:
        os.chdir(REPO)
        if a.keep:
            print(f"kept sandbox: {sandbox}")
        else:
            shutil.rmtree(sandbox, ignore_errors=True)

    if failures:
        print(f"\nSMOKE FAILED: {len(failures)} check(s)")
        return 1
    print("\nSMOKE PASSED: SFT sweep cells executed locally (stub datasets, CPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
