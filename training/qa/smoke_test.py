#!/usr/bin/env python3
"""
training/qa/smoke_test.py

Quick sanity checks for any NeuralAI checkpoint.  Verifies:
  1. Model loads from checkpoint without errors
  2. Tokenizer vocab size == 32,000
  3. Special token IDs: BOS=1, EOS=2, PAD=0
  4. One forward pass + one backward pass on dummy data
  5. Checkpoint files are complete (expected files present)

Works on CPU and GPU.  Prints PASS/FAIL per test.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

# torch / transformers are imported lazily inside test functions so the module
# can be imported in environments where they are not yet installed.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 32_000
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

EXPECTED_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]

# Also acceptable: sharded safetensors
EXPECTED_FILES_ALT = [
    "config.json",
    "model-00001-of-00001.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
def test_load_model(checkpoint_path: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=False,
        )
        return {"name": "model_load", "pass": True, "detail": f"Loaded {type(model).__name__}"}
    except Exception as e:
        return {"name": "model_load", "pass": False, "detail": traceback.format_exc()}


def test_tokenizer_vocab(checkpoint_path: str) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        size = len(tokenizer)
        ok = size == VOCAB_SIZE
        return {
            "name": "tokenizer_vocab_size",
            "pass": ok,
            "detail": f"vocab_size={size} (expected {VOCAB_SIZE})",
        }
    except Exception as e:
        return {"name": "tokenizer_vocab_size", "pass": False, "detail": traceback.format_exc()}


def test_special_tokens(checkpoint_path: str) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        checks = {
            "bos_token_id": (tokenizer.bos_token_id, BOS_ID),
            "eos_token_id": (tokenizer.eos_token_id, EOS_ID),
            "pad_token_id": (tokenizer.pad_token_id, PAD_ID),
        }
        ok = all(actual == expected for actual, expected in checks.values())
        detail = ", ".join(f"{k}={actual} (expected {expected})" for k, (actual, expected) in checks.items())
        return {"name": "special_token_ids", "pass": ok, "detail": detail}
    except Exception as e:
        return {"name": "special_token_ids", "pass": False, "detail": traceback.format_exc()}


def test_forward_backward(checkpoint_path: str) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            trust_remote_code=False,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = PAD_ID

        # Dummy input: batch_size=2, seq_len=8
        input_ids = torch.full((2, 8), tokenizer.pad_token_id, dtype=torch.long, device=device)
        # Put some real tokens in there to avoid all-padding edge cases
        input_ids[0] = torch.tensor([BOS_ID, 100, 200, 300, 400, 500, EOS_ID, PAD_ID], device=device)
        input_ids[1] = torch.tensor([BOS_ID, 600, 700, 800, 900, 1000, EOS_ID, PAD_ID], device=device)

        model.train()
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            return {"name": "forward_backward", "pass": False, "detail": f"Loss is invalid: {loss}"}

        loss.backward()
        # Verify at least one parameter received a gradient
        has_grad = any(p.grad is not None for p in model.parameters())
        if not has_grad:
            return {"name": "forward_backward", "pass": False, "detail": "No gradients found after backward()"}

        return {
            "name": "forward_backward",
            "pass": True,
            "detail": f"Loss={loss.item():.4f}, device={device}, gradients_ok={has_grad}",
        }
    except Exception as e:
        return {"name": "forward_backward", "pass": False, "detail": traceback.format_exc()}


def test_checkpoint_completeness(checkpoint_path: str) -> dict[str, Any]:
    path = Path(checkpoint_path)
    missing = [f for f in EXPECTED_FILES if not (path / f).exists()]
    missing_alt = [f for f in EXPECTED_FILES_ALT if not (path / f).exists()]
    # Pass if either full set exists
    ok = len(missing) == 0 or len(missing_alt) == 0
    if ok:
        detail = "All expected files present."
        if len(missing) == 0:
            detail = f"Standard files present: {EXPECTED_FILES}"
        else:
            detail = f"Sharded files present: {EXPECTED_FILES_ALT}"
    else:
        detail = f"Missing standard: {missing}; Missing alt: {missing_alt}"
    return {"name": "checkpoint_completeness", "pass": ok, "detail": detail}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_smoke_tests(checkpoint_path: str) -> dict[str, Any]:
    import torch

    tests = [
        test_checkpoint_completeness(checkpoint_path),
        test_load_model(checkpoint_path),
        test_tokenizer_vocab(checkpoint_path),
        test_special_tokens(checkpoint_path),
        test_forward_backward(checkpoint_path),
    ]

    all_pass = all(t["pass"] for t in tests)
    any_fail = any(t["pass"] is False for t in tests)

    report = {
        "smoke_version": "1.0",
        "project": "NeuralAI-Air-135M-v19",
        "checkpoint": checkpoint_path,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tests": tests,
        "summary": {
            "overall": "PASS" if all_pass else ("FAIL" if any_fail else "PARTIAL"),
            "passed": sum(1 for t in tests if t["pass"]),
            "failed": sum(1 for t in tests if t["pass"] is False),
            "total": len(tests),
        },
    }
    return report


def print_smoke_report(report: dict) -> None:
    print("=" * 60)
    print("NeuralAI Checkpoint Smoke Test")
    print("=" * 60)
    print(f"Checkpoint: {report.get('checkpoint', 'N/A')}")
    print(f"Device    : {report.get('device', 'N/A')}")
    print()
    for t in report.get("tests", []):
        status = "PASS" if t["pass"] else "FAIL"
        print(f"  [{status:4}] {t['name']}: {t['detail']}")
    print()
    summary = report.get("summary", {})
    print(f"Overall   : {summary.get('overall', 'UNKNOWN')}  ({summary.get('passed', 0)}/{summary.get('total', 0)} passed)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Smoke-test a NeuralAI checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to HF checkpoint directory")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write smoke_report.json")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    report = run_smoke_tests(args.checkpoint)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

    if not args.quiet:
        print_smoke_report(report)

    if report.get("summary", {}).get("overall") == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
