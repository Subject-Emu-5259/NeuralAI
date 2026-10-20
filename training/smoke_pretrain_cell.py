#!/usr/bin/env python3
"""Local dry-run smoke test for the pretrain-v2 notebook cell.

Executes the *generated* cell source VERBATIM (byte-for-byte the text Kaggle runs)
inside sandboxes, so template/runtime bugs fail here in ~2 minutes instead of
~13 minutes into a paid Kaggle run. The v3 kernel died on a bare `seq_len`
NameError after streaming the whole corpus; the resume path once crashed on the
tied lm_head.weight; this smoke test caught both.

What is real, what is shimmed:
  REAL   - the entire cell source (imports, tokenizer extraction, corpus loop,
           windowing + held-out split, DataLoader, Nae1Config/Nae1ForCausalLM,
           train loop, checkpoint save, held-out PPL + greedy sample, resume)
  REAL   - tokenizer (embedded v2 32k BPE, decoded by the cell itself),
           pyarrow parquet reads (tiny stub shards generated here),
           torch training on CPU (100M model, tiny step count)
  SHIMMED- /kaggle/* -> sandbox dir (pathlib.Path subclass),
           huggingface_hub.hf_hub_download -> serves local stub parquet copies,
           device "cuda" -> "cpu" (Tensor.to / Module.to / torch.tensor),
           torch.cuda.amp.GradScaler -> disabled CPU scaler,
           torch.cuda.amp.autocast -> nullcontext

Phases:
  phase 1 (sandbox A): fresh train, steps=6 save_every=3 -> ckpt_step_3 then
    ckpt_step_6 (latest-only deletion), held-out ppl trend [3, 6]
  phase 2 (sandbox B, FRESH working dir): auto-resume via the attached dataset
    mount (/kaggle/input/nae1-pretrain-v2-resume/...). B's input dir carries TWO
    ckpts (step 8 and step 10, step-10 copied from A's step-6 weights with a
    rewritten train_state) to prove numeric (not lexical) selection: must resume
    from ckpt_step_10 and train to 12.
  phase 3 (sandbox A again): in-WORK resume fallback -> resumes ckpt_step_6,
    trains to 8, ppl trend [3, 6, 8]

Usage:
  python3 scripts/smoke_pretrain_cell.py
  python3 scripts/smoke_pretrain_cell.py --keep      # keep the sandbox dirs
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

STUB_DOCS = [
    f"Stub training document number {i}. The quick brown fox jumps over the lazy dog "
    f"while the alphabet soup bubbles. Synthetic corpus for template smoke testing "
    f"only; its job is to give the data pipeline real tokens to chew on locally."
    for i in range(40)
]

# mutable sandbox target shared with the Path/glob shims (run_phase switches it)
HOLDER: dict = {"root": None}


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_pretrain_notebook", REPO / "scripts" / "build_pretrain_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cell_source(nb: dict) -> str:
    cell = nb["cells"][1]
    assert cell["cell_type"] == "code" and cell["id"] == "pretrain-v2-train", \
        f"unexpected cell layout: {cell.get('id')}"
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def make_stub_shards(stub_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    stub_dir.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({"text": pa.array(STUB_DOCS)})
    # several row groups so the cell's row-group loop is exercised for real
    pq.write_table(tbl, stub_dir / "shard.parquet", row_group_size=10)


def install_shims(stub_dir: Path):
    """Redirect /kaggle paths, stub HF download, map CUDA device/amp to CPU."""
    import torch

    # --- pathlib: /kaggle/... -> current sandbox root (cell does
    # `from pathlib import Path` after this patch, so it picks up the subclass) ---
    _RealPath = type(Path())  # PosixPath on macOS/Linux

    def _rewrite_kaggle(args):
        if args and isinstance(args[0], str) and args[0].startswith("/kaggle"):
            p = args[0]
            rest = (p[len("/kaggle/working"):]
                    if p.startswith("/kaggle/working") else p[len("/kaggle"):])
            return (str(HOLDER["root"]) + rest,) + args[1:]
        return args

    class SandboxPath(_RealPath):
        # NOTE: never call super().__new__(cls, ...) here. Path.__new__ checks
        # `if cls is Path`, but `Path` there is the module-global name, which
        # install_shims rebinds to this subclass just below -- so the check
        # misfires and returns a bare PosixPath whose __init__ never runs
        # (hence the "'PosixPath' object has no attribute '_raw_paths'"
        # crash). object.__new__ is sufficient: on 3.14 the real parsing
        # lives in PurePath.__init__.
        # The /kaggle -> sandbox rewrite MUST live in __init__, not __new__:
        # type.__call__ always forwards the *original* call args to __init__,
        # so a rewrite done only in __new__ would be silently discarded.
        def __new__(cls, *args):
            return object.__new__(cls)

        def __init__(self, *args):
            super().__init__(*_rewrite_kaggle(args))

    import pathlib
    _orig_base = pathlib.Path          # the ORIGINAL base class (name: 'Path')
    pathlib.Path = SandboxPath
    # Pre-patch importers (transformers/huggingface_hub lazy modules) hold the
    # original base Path; its __new__ rebinds via the module global we just
    # replaced, so direct base-Path construction (e.g. hub's extract_commit_hash
    # -> Path(...).as_posix()) would die on a missing _flavour. Give the base the
    # posix flavour so those callers keep working unrewritten.
    # (Python 3.14 removed pathlib's `_flavour` entirely — construction no longer
    # keys off it, so the workaround is only needed where the attribute exists.)
    if hasattr(_RealPath, "_flavour"):
        _orig_base._flavour = _RealPath._flavour

    # --- huggingface_hub: patch the REAL module's hf_hub_download (transformers
    # imports huggingface_hub internally, so replacing the module breaks it).
    # Import transformers first so its internal from-imports bind the real funcs.
    # Each download serves a fresh copy of the stub parquet (the cell unlinks
    # each shard after tokenizing). ---
    import transformers  # noqa: F401  (eager: bind its hub imports before the patch)
    import huggingface_hub

    def hf_hub_download(repo_id, rel, repo_type=None, local_dir=None, **kw):
        dest = HOLDER["root"] / (local_dir or "fw") / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(stub_dir / "shard.parquet", dest)
        return str(dest)

    huggingface_hub.hf_hub_download = hf_hub_download

    # --- CUDA -> CPU ---
    def _cpu_dev(x):
        if isinstance(x, str) and x.startswith("cuda"):
            return "cpu"
        if isinstance(x, torch.device) and x.type == "cuda":
            return torch.device("cpu")
        return x

    _t_to, _m_to, _tensor = torch.Tensor.to, torch.nn.Module.to, torch.tensor
    torch.Tensor.to = lambda self, *a, **k: _t_to(
        self, *[_cpu_dev(x) for x in a], **{k2: _cpu_dev(v) for k2, v in k.items()})
    torch.nn.Module.to = lambda self, *a, **k: _m_to(
        self, *[_cpu_dev(x) for x in a], **{k2: _cpu_dev(v) for k2, v in k.items()})
    torch.tensor = lambda *a, **k: _tensor(
        *a, **{k2: _cpu_dev(v) if k2 == "device" else v for k2, v in k.items()})
    torch.cuda.amp.GradScaler = lambda *a, **k: torch.amp.GradScaler("cpu", enabled=False)
    torch.cuda.amp.autocast = lambda *a, **k: contextlib.nullcontext()


def run_phase(builder, sandbox: Path, **build_kwargs) -> str:
    nb = builder.build(**build_kwargs)
    src = cell_source(nb)
    HOLDER["root"] = sandbox
    os.chdir(sandbox)  # cell writes tokenizer/config/model sources relative to cwd
    sys.path.insert(0, str(sandbox))  # /kaggle/working equivalent on sys.path
    buf = io.StringIO()
    ns: dict = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "pretrain-v2-train", "exec"), ns, ns)
    except Exception:
        print("---- cell stdout until failure ----")
        print(buf.getvalue())
        raise
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the sandbox dirs")
    a = ap.parse_args()

    builder = load_builder()
    sandbox_a = Path(tempfile.mkdtemp(prefix="nae1_smokeA_"))
    sandbox_b = Path(tempfile.mkdtemp(prefix="nae1_smokeB_"))
    stub_dir = sandbox_a.parent / (sandbox_a.name + "_stubs")
    make_stub_shards(stub_dir)
    install_shims(stub_dir)

    failures: list = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            failures.append(msg)

    def ppl_log(sandbox: Path):
        p = sandbox / "pretrain_v2" / "heldout_ppl.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.is_file() else None

    work_a = sandbox_a / "pretrain_v2"
    work_b = sandbox_b / "pretrain_v2"
    common = dict(seq_len=64, accum=2, peak_lr=6e-4, warmup=2, floor_frac=0.1,
                  max_tokens=5000)
    try:
        # ---- Phase 1 (sandbox A): fresh train ----
        print(f"sandbox A: {sandbox_a}")
        out1 = run_phase(builder, sandbox_a, steps=6, save_every=3, **common)
        check("stream ready:" in out1, "phase1: corpus stream built")
        check("windows:" in out1 and "held-out" in out1,
              "phase1: windows computed with held-out split")
        check("saved ckpt_step_3" in out1, "phase1: ckpt_step_3 saved")
        check("saved ckpt_step_6" in out1, "phase1: ckpt_step_6 saved")
        check("PRETRAIN V2 COMPLETE" in out1, "phase1: training loop completed")
        check((work_a / "ckpt_step_6" / "model.safetensors").is_file(),
              "phase1: ckpt_step_6 model.safetensors exists")
        check(out1.count("held-out ppl:") == 2,
              "phase1: held-out ppl logged at both checkpoints (steps 3 and 6)")
        p_log = ppl_log(sandbox_a)
        check([e["step"] for e in p_log] == [3, 6]
              and all(e["ppl"] > 1 and e["ppl"] == e["ppl"] for e in p_log),
              f"phase1: heldout_ppl.jsonl trend [3,6] finite (got {p_log})")
        ts6 = json.loads((work_a / "ckpt_step_6" / "train_state.json").read_text())
        check(ts6.get("step") == 6 and ts6.get("heldout_ppl", 0) > 1,
              f"phase1: ckpt train_state carries heldout_ppl (got {ts6})")
        check((work_a / "ckpt_step_3").exists() is False,
              "phase1: old ckpt_step_3 deleted (latest-only)")
        check(len(list(work_a.glob("ckpt_step_*/"))) == 1, "phase1: exactly one ckpt kept")

        # ---- Phase 2 (sandbox B, fresh working dir): resume via attached dataset ----
        # Simulates the auto-resume pipeline: dataset mounted at
        # /kaggle/input/nae1-pretrain-v2-resume/pretrain_v2 with TWO ckpts whose
        # lexical order ("ckpt_step_10" < "ckpt_step_8") disagrees with numeric
        # order -> the cell must pick step 10.
        print(f"sandbox B: {sandbox_b}")
        ds_root = sandbox_b / "input" / "nae1-pretrain-v2-resume" / "pretrain_v2"
        for step in (8, 10):
            d = ds_root / f"ckpt_step_{step}"
            d.mkdir(parents=True)
            shutil.copy(work_a / "ckpt_step_6" / "model.safetensors", d / "model.safetensors")
            (d / "train_state.json").write_text(json.dumps({"step": step}))
        # resume_ref is REQUIRED here: the cell only looks under /kaggle/input for
        # a resume payload when a resume dataset is actually attached. Without it
        # RESUME_SLUG is "", discovery is skipped by design, and the cell correctly
        # falls through to start=0 — which silently turned these two checks red and
        # looked like a builder bug when it was a stale test.
        out2 = run_phase(builder, sandbox_b, steps=12, save_every=4,
                         resume_ref="takenover/nae1-pretrain-v2-resume", **common)
        check("resumed from ckpt_step_10 at step 10" in out2,
              "phase2: dataset resume picked step 10 (numeric, not lexical)")
        check("saved ckpt_step_12" in out2,
              "phase2: trained past resume and saved ckpt_step_12")
        check("PRETRAIN V2 COMPLETE" in out2, "phase2: completed")
        check(sorted(p.name for p in work_b.glob("ckpt_step_*/")) == ["ckpt_step_12"],
              "phase2: latest-only kept (dataset ckpt copied in, consumed, then superseded)")
        state_b = json.loads((work_b / "ckpt_step_12" / "train_state.json").read_text())
        check(state_b.get("step") == 12 and state_b.get("heldout_ppl", 0) > 1,
              f"phase2: train_state step==12 + heldout_ppl (got {state_b})")
        check([e["step"] for e in ppl_log(sandbox_b)] == [12],
              f"phase2: fresh-WORK ppl trend starts at 12 (got {ppl_log(sandbox_b)})")

        # ---- Phase 3 (sandbox A again): in-WORK resume fallback ----
        out3 = run_phase(builder, sandbox_a, steps=8, save_every=4, **common)
        check("resumed from ckpt_step_6 at step 6" in out3,
              "phase3: in-WORK resume from ckpt_step_6")
        check("saved ckpt_step_8" in out3,
              "phase3: trained past resume and saved ckpt_step_8")
        check("PRETRAIN V2 COMPLETE" in out3, "phase3: completed")
        state_a = json.loads((work_a / "ckpt_step_8" / "train_state.json").read_text())
        check(state_a.get("step") == 8 and state_a.get("heldout_ppl", 0) > 1,
              f"phase3: train_state step==8 + heldout_ppl (got {state_a})")
        check([e["step"] for e in ppl_log(sandbox_a)] == [3, 6, 8],
              f"phase3: ppl trend continued across sessions (got {ppl_log(sandbox_a)})")
    finally:
        os.chdir(REPO)
        if a.keep:
            print(f"kept sandboxes: {sandbox_a} {sandbox_b}")
        else:
            shutil.rmtree(sandbox_a, ignore_errors=True)
            shutil.rmtree(sandbox_b, ignore_errors=True)
            shutil.rmtree(stub_dir, ignore_errors=True)

    if failures:
        print(f"\nSMOKE FAILED: {len(failures)} check(s)")
        return 1
    print("\nSMOKE PASSED: train + dataset-resume + in-WORK resume all executed locally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
