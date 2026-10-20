#!/usr/bin/env python3
"""Negative test for the SFT cell's tokenizer-vocab guard.

Rewired for the v7/v2 lineage, the generated sweep cell refuses to start when
the attached tokenizer's real vocab does not match the model's 32000
embeddings. Point it at the OLD 439-token tokenizer and confirm the cell
aborts BEFORE any training step.

Why this matters: every id-range assert in the cell is `< 32000`, and a
439-token tokenizer only ever emits ids 0..438 — so it passes all of them.
The old nae1-fineweb-replay corpus shipped in exactly that state. The id
asserts structurally cannot catch this; only a real vocab comparison can.

Executes the generated cells VERBATIM against stub datasets (same sandbox and
shims as scripts/smoke_sft_cell.py), so this is byte-for-byte what Kaggle runs.

Usage:
  python3 scripts/neg_sft_vocab_guard.py                      # old 439 tokenizer
  python3 scripts/neg_sft_vocab_guard.py --tokenizer-slug nae1-tokenizer-v2
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
sys.path.insert(0, str(REPO / "scripts"))

OLD_TOK = "nae1-tokenizer-v1"   # 439 tokens — must be refused


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "smoke_sft_cell", REPO / "scripts" / "smoke_sft_cell.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-slug", default=OLD_TOK,
                    help="slug to attach (default: the old 439-token one, which "
                         "must be refused; pass nae1-tokenizer-v2 for the positive case)")
    a = ap.parse_args()
    # The old tokenizer must be refused; a correctly-wired one must not be.
    expect_guard = a.tokenizer_slug == OLD_TOK

    harness = load_harness()
    builder = harness.load_builder()

    nb = builder.build(steps_per=4, warmup=2, accum=1, save_every=2, seq=64,
                       configs=harness.CONFIGS, tokenizer_slug=a.tokenizer_slug)

    sandbox = Path(tempfile.mkdtemp(prefix="nae1_neg_"))
    harness.seed_inputs(sandbox, tokenizer_slug=a.tokenizer_slug)
    tok_dir = sandbox / "input" / "takenover" / a.tokenizer_slug
    vocab = json.loads((tok_dir / "nae1_tokenizer-vocab.json").read_text())
    print(f"attached tokenizer {a.tokenizer_slug}: vocab = {len(vocab)} "
          f"(model needs {builder.MODEL_VOCAB})", flush=True)

    # Drop cached modules / sys.path so the cell's own imports load from sandbox.
    for name in ("nae1_model", "nae1_config"):
        sys.modules.pop(name, None)
    while str(REPO / "scripts") in sys.path:
        sys.path.remove(str(REPO / "scripts"))

    harness.pretrain_smoke.HOLDER["root"] = sandbox
    harness.pretrain_smoke.install_shims(stub_dir=sandbox / "_stubs")
    os.chdir(sandbox)
    sys.path.insert(0, str(sandbox))

    buf = io.StringIO()
    caught: BaseException | None = None
    try:
        with contextlib.redirect_stdout(buf):
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = cell["source"]
                src = "".join(src) if isinstance(src, list) else src
                exec(compile(src, cell["id"], "exec"), {"__name__": "__main__"})
    except BaseException as e:          # SystemExit is the expected outcome
        caught = e
    finally:
        os.chdir(REPO)
        shutil.rmtree(sandbox, ignore_errors=True)

    out = buf.getvalue()
    print("---- cell stdout tail ----")
    print("\n".join(out.splitlines()[-5:]))
    print("---- result ----")

    if expect_guard:
        if caught is None:
            print("FAIL: cell ran to completion on a mismatched tokenizer "
                  "(vocab guard did NOT fire)")
            return 1
        msg = str(caught)
        print(f"raised {type(caught).__name__}: {msg}")
        if "vocab" in msg and str(builder.MODEL_VOCAB) in msg:
            print(f"PASS: vocab guard refused {a.tokenizer_slug} before training")
            return 0
        print("FAIL: raised, but not the vocab guard")
        return 2

    if caught is not None:
        print(f"FAIL: correct tokenizer was refused: {type(caught).__name__}: {caught}")
        return 1
    if "sft stream:" not in out or "SFT v4 complete" not in out:
        print("FAIL: cells did not run to completion")
        return 1
    print("PASS: correctly-wired tokenizer accepted, sweep completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
