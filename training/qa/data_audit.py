#!/usr/bin/env python3
"""
training/qa/data_audit.py

Audit pre-training tokenized shards before any GPU time is consumed.
Computes token distribution, n-gram collision rates, special-token ratios,
document-length statistics, outlier detection, and prints PASS/FAIL for
Gate 0 criteria per S-001 §9.1.

Output: JSON report at data/pretrain/audit_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants from spec
# ---------------------------------------------------------------------------
VOCAB_SIZE = 32_000
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0
SPECIAL_IDS = {BOS_ID, EOS_ID, PAD_ID}

DATA_DIR = Path("data/pretrain")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
REPORT_PATH = DATA_DIR / "audit_report.json"

SHARD_TOKENS = 50_000_000  # per spec (~100 MB uint16)


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------
def find_shards(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("shard_*.bin"))


def load_shard(path: Path) -> np.ndarray:
    """Memory-map a uint16 shard."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def shard_token_count(path: Path) -> int:
    size_bytes = path.stat().st_size
    return size_bytes // 2  # uint16 = 2 bytes


# ---------------------------------------------------------------------------
# Document segmentation
# ---------------------------------------------------------------------------
def split_docs(flat_tokens: np.ndarray, eos_id: int = EOS_ID) -> list[np.ndarray]:
    """Split a flat token array into documents at every EOS token."""
    boundaries = np.where(flat_tokens == eos_id)[0]
    docs = []
    start = 0
    for b in boundaries:
        docs.append(flat_tokens[start : b + 1])
        start = b + 1
    if start < len(flat_tokens):
        docs.append(flat_tokens[start:])
    return docs


# ---------------------------------------------------------------------------
# Audit computations
# ---------------------------------------------------------------------------
def compute_token_distribution(flat_tokens: np.ndarray, top_k: int = 20) -> dict[str, Any]:
    counter = Counter(int(t) for t in flat_tokens)
    total = len(flat_tokens)
    most_common = counter.most_common(top_k)
    least_common = counter.most_common()[:-top_k - 1:-1]  # bottom k
    freqs = np.fromiter(counter.values(), dtype=np.int64)
    entropy = -np.sum((freqs / total) * np.log2(freqs / total + 1e-12))
    return {
        "vocab_coverage": len(counter) / VOCAB_SIZE,
        "total_tokens": total,
        "entropy_bits": round(float(entropy), 4),
        "most_frequent": [
            {"token_id": int(tid), "count": int(cnt), "pct": round(100.0 * cnt / total, 4)}
            for tid, cnt in most_common
        ],
        "least_frequent": [
            {"token_id": int(tid), "count": int(cnt), "pct": round(100.0 * cnt / total, 6)}
            for tid, cnt in least_common
        ],
    }


def compute_ngram_collisions(docs: list[np.ndarray], n: int = 4, sample_cap: int = 100_000) -> dict[str, Any]:
    """Estimate unique n-grams / total n-grams across a sample of docs."""
    all_ngrams = []
    for doc in docs:
        if len(doc) >= n:
            for i in range(len(doc) - n + 1):
                all_ngrams.append(tuple(int(x) for x in doc[i : i + n]))
                if len(all_ngrams) >= sample_cap:
                    break
        if len(all_ngrams) >= sample_cap:
            break
    total = len(all_ngrams)
    if total == 0:
        return {"total_ngrams": 0, "unique_ngrams": 0, "diversity_ratio": 0.0}
    unique = len(set(all_ngrams))
    return {
        "total_ngrams": total,
        "unique_ngrams": unique,
        "diversity_ratio": round(unique / total, 4),
    }


def check_special_token_ratios(flat_tokens: np.ndarray) -> dict[str, Any]:
    total = len(flat_tokens)
    if total == 0:
        return {"bos_pct": 0.0, "eos_pct": 0.0, "pad_pct": 0.0, "special_pct": 0.0}
    bos = int(np.sum(flat_tokens == BOS_ID))
    eos = int(np.sum(flat_tokens == EOS_ID))
    pad = int(np.sum(flat_tokens == PAD_ID))
    return {
        "bos_count": bos,
        "eos_count": eos,
        "pad_count": pad,
        "bos_pct": round(100.0 * bos / total, 4),
        "eos_pct": round(100.0 * eos / total, 4),
        "pad_pct": round(100.0 * pad / total, 4),
        "special_pct": round(100.0 * (bos + eos + pad) / total, 4),
    }


def compute_length_stats(docs: list[np.ndarray]) -> dict[str, Any]:
    lengths = [len(d) for d in docs]
    if not lengths:
        return {}
    arr = np.array(lengths, dtype=np.int64)
    return {
        "num_documents": int(len(arr)),
        "mean": round(float(np.mean(arr)), 2),
        "median": int(np.median(arr)),
        "std": round(float(np.std(arr)), 2),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "histogram": {
            "<10": int(np.sum(arr < 10)),
            "10-100": int(np.sum((arr >= 10) & (arr < 100))),
            "100-500": int(np.sum((arr >= 100) & (arr < 500))),
            "500-1K": int(np.sum((arr >= 500) & (arr < 1000))),
            "1K-5K": int(np.sum((arr >= 1000) & (arr < 5000))),
            "5K-10K": int(np.sum((arr >= 5000) & (arr < 10000))),
            ">10K": int(np.sum(arr > 10000)),
        },
    }


def detect_outliers(docs: list[np.ndarray]) -> dict[str, Any]:
    short = [i for i, d in enumerate(docs) if len(d) < 10]
    long = [i for i, d in enumerate(docs) if len(d) > 10_000]
    return {
        "docs_lt_10_tokens": {
            "count": len(short),
            "indices": short[:50],  # cap to keep JSON small
        },
        "docs_gt_10K_tokens": {
            "count": len(long),
            "indices": long[:50],
        },
    }


# ---------------------------------------------------------------------------
# Gate 0 criteria (S-001 §9.1)
# ---------------------------------------------------------------------------
def evaluate_gate0(report: dict) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # Criterion 1: 1B tokens tokenized and sharded
    train_tokens = report.get("train", {}).get("total_tokens", 0)
    results["gate0_1_train_tokens_ge_1B"] = {
        "pass": train_tokens >= 1_000_000_000,
        "value": train_tokens,
        "target": ">= 1_000_000_000",
    }

    # Criterion 2: Validation split contains 10M tokens, no overlap with train
    val_tokens = report.get("val", {}).get("total_tokens", 0)
    results["gate0_2_val_tokens_ge_10M"] = {
        "pass": val_tokens >= 10_000_000,
        "value": val_tokens,
        "target": ">= 10_000_000",
    }
    # We can't programmatically prove "no overlap" without hashing every doc,
    # but we can check that train/val shard filenames are disjoint.
    train_shards = set(report.get("train", {}).get("shards", []))
    val_shards = set(report.get("val", {}).get("shards", []))
    overlap = train_shards & val_shards
    results["gate0_2b_no_shard_overlap"] = {
        "pass": len(overlap) == 0,
        "overlap_count": len(overlap),
        "target": "0 overlapping shards",
    }

    # Criterion 3: Deduplication report — % removed per source < 40%
    # (This script audits the *output* shards, not the pipeline itself.
    #  We mark it INFO and note that the pipeline must provide the dedup log.)
    results["gate0_3_dedup_per_source_lt_40pct"] = {
        "pass": None,  # cannot determine from shards alone
        "note": "Requires dedup_manifest.json from data_pipeline. Mark INFO pending manual review.",
        "target": "< 40% per source",
    }

    # Criterion 4: Tokenizer vocab alignment verified: 32K tokens, special at 0/1/2
    vocab_coverage = report.get("train", {}).get("token_distribution", {}).get("vocab_coverage", 0.0)
    results["gate0_4a_vocab_coverage"] = {
        "pass": vocab_coverage >= 0.95,
        "value": vocab_coverage,
        "target": ">= 0.95",
    }
    special = report.get("train", {}).get("special_token_ratios", {})
    eos_count = special.get("eos_count", 0)
    results["gate0_4b_eos_present"] = {
        "pass": eos_count > 0,
        "value": eos_count,
        "target": "> 0",
    }
    # Additional hard check: special IDs are within vocab
    results["gate0_4c_special_ids_in_vocab"] = {
        "pass": all(0 <= sid < VOCAB_SIZE for sid in SPECIAL_IDS),
        "special_ids": list(SPECIAL_IDS),
        "target": f"all in [0, {VOCAB_SIZE})",
    }

    # Data-quality extensions from DATA-135M-PRETRAIN.md §12
    ngram = report.get("train", {}).get("ngram_diversity", {})
    results["gate0_ext_ngram_diversity_gt_0_75"] = {
        "pass": ngram.get("diversity_ratio", 0.0) > 0.75,
        "value": ngram.get("diversity_ratio", 0.0),
        "target": "> 0.75",
    }

    outliers = report.get("train", {}).get("outliers", {})
    results["gate0_ext_outliers_lt_10pct"] = {
        "pass": outliers.get("docs_gt_10K_tokens", {}).get("count", 0) / max(report.get("train", {}).get("length_stats", {}).get("num_documents", 1), 1) < 0.10,
        "value": outliers.get("docs_gt_10K_tokens", {}).get("count", 0),
        "target": "< 10% of documents",
    }

    return results


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------
def audit_split(directory: Path, split_name: str, max_shards: int | None = None) -> dict[str, Any]:
    shards = find_shards(directory)
    if not shards:
        return {
            "split": split_name,
            "directory": str(directory),
            "shards": [],
            "total_tokens": 0,
            "error": f"No shards found in {directory}",
        }

    if max_shards:
        shards = shards[:max_shards]

    total_tokens = sum(shard_token_count(s) for s in shards)
    # Load a sample for detailed stats (first N shards or all if small)
    sample_shards = shards if len(shards) <= 5 else shards[:5]
    flat_sample = np.concatenate([load_shard(s) for s in sample_shards])

    docs = split_docs(flat_sample)

    return {
        "split": split_name,
        "directory": str(directory),
        "shards": [s.name for s in shards],
        "total_tokens": int(total_tokens),
        "sampled_tokens": int(len(flat_sample)),
        "token_distribution": compute_token_distribution(flat_sample),
        "ngram_diversity": compute_ngram_collisions(docs, n=4),
        "special_token_ratios": check_special_token_ratios(flat_sample),
        "length_stats": compute_length_stats(docs),
        "outliers": detect_outliers(docs),
    }


def run_audit(output_path: Path | None = None, max_shards: int | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "audit_version": "1.0",
        "project": "NeuralAI-Air-135M-v19",
        "spec_reference": "S-001 §9.1, DATA-135M-PRETRAIN §12",
    }

    report["train"] = audit_split(TRAIN_DIR, "train", max_shards=max_shards)
    report["val"] = audit_split(VAL_DIR, "val", max_shards=max_shards)
    report["gate0"] = evaluate_gate0(report)

    # Summary
    all_pass = all(
        v.get("pass") is True
        for v in report["gate0"].values()
        if v.get("pass") is not None
    )
    report["summary"] = {
        "overall": "PASS" if all_pass else "FAIL",
        "train_tokens": report["train"].get("total_tokens", 0),
        "val_tokens": report["val"].get("total_tokens", 0),
        "criteria_evaluated": len(report["gate0"]),
        "criteria_passed": sum(1 for v in report["gate0"].values() if v.get("pass") is True),
        "criteria_failed": sum(1 for v in report["gate0"].values() if v.get("pass") is False),
        "criteria_pending": sum(1 for v in report["gate0"].values() if v.get("pass") is None),
    }

    out = output_path or REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict) -> None:
    print("=" * 60)
    print("NeuralAI-Air-135M  Data Audit Report")
    print("=" * 60)
    summary = report.get("summary", {})
    print(f"Overall Gate 0: {summary.get('overall', 'UNKNOWN')}")
    print(f"Train tokens:   {summary.get('train_tokens', 0):,}")
    print(f"Val tokens:     {summary.get('val_tokens', 0):,}")
    print()
    for key, val in report.get("gate0", {}).items():
        status = "PASS" if val.get("pass") is True else ("FAIL" if val.get("pass") is False else "PENDING")
        print(f"  [{status:7}] {key}: {val}")
    print()
    print(f"Full JSON report written to: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global DATA_DIR, TRAIN_DIR, VAL_DIR, REPORT_PATH
    parser = argparse.ArgumentParser(description="Audit pre-training tokenized shards.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to data/pretrain/ directory")
    parser.add_argument("--output", type=str, default=str(REPORT_PATH), help="Path to write audit_report.json")
    parser.add_argument("--max-shards", type=int, default=None, help="Limit number of shards to read (for quick testing)")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    REPORT_PATH = Path(args.output)

    report = run_audit(output_path=REPORT_PATH, max_shards=args.max_shards)
    if not args.quiet:
        print_report(report)

    # Exit non-zero on FAIL so CI can catch it
    if report.get("summary", {}).get("overall") == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
