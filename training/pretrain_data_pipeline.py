"""
Build a clean, pre-training corpus for NeuralAI-Air-135M.

Steps:
  1. Download raw sources (FineWeb-Edu, Wikipedia, C4, OpenWebMath, Starcoderdata).
  2. Filter, dedupe, and normalize text.
  3. Tokenize with the project tokenizer.
  4. Save uint16 memmap .bin files for fast loading by continue_pretrain.py.
  5. Write SFT instruction/response JSONL alongside.

Usage:
  python training/pretrain_data_pipeline.py \
      --tokenizer_dir NeuralAI-Air-135M/tokenizer \
      --out_dir data/pretrain_tokens \
      --sft_out data/sft_pairs.jsonl \
      --max_tokens_per_source 500_000_000 \
      --samples 4

Requirements:
  pip install datasets transformers sentencepiece numpy tqdm
"""
import os
import re
import json
import argparse
from typing import Iterable, Dict, Any

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


# Default high-quality open pre-training sources.
SOURCES = [
    {"name": "fineweb-edu", "path": "HuggingFaceFW/fineweb-edu", "split": "train", "column": "text", "weight": 0.35},
    {"name": "wikipedia", "path": "wikimedia/wikipedia", "split": "20231101.en", "column": "text", "weight": 0.20},
    {"name": "c4", "path": "allenai/c4", "split": "en", "column": "text", "weight": 0.20},
    {"name": "openwebmath", "path": "open-web-math/open-web-math", "split": "train", "column": "text", "weight": 0.10},
    {"name": "starcoderdata", "path": "bigcode/starcoderdata", "split": "train", "column": "content", "weight": 0.15},
]


# Seed SFT prompts to augment with during the SFT stage.
SFT_SEED_PROMPTS = [
    ("What is your name?", "My name is NeuralAI, a local AI assistant built by DeAndrew Harris."),
    ("Who created you?", "I was created by DeAndrew P. Harris as part of the NeuralAI project."),
    ("Summarize the pretraining pipeline.", "Collect raw text, clean and deduplicate it, tokenize it, then train the model with causal language modeling to predict the next token."),
    ("Write a Python hello world.", "print('Hello, world!')"),
    ("What is the capital of Tennessee?", "The capital of Tennessee is Nashville."),
]


def normalize_text(text: str) -> str:
    """Remove excessive whitespace, garbled unicode, and boilerplate markers."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    # Compress whitespace.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    # Drop lines that look like nav/footer tokens.
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(Cookie Policy|Privacy Policy|Terms of Use|Login|Sign up|Copyright ©|\|)$", stripped, re.I):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def ngram_dedup(documents: Iterable[str], n: int = 13, max_dup_per_doc: int = 5) -> Iterable[str]:
    """Streaming deduplication using character n-gram fingerprints."""
    seen = set()
    for doc in documents:
        if not doc:
            continue
        dups = 0
        for i in range(len(doc) - n + 1):
            gram = doc[i : i + n]
            if gram in seen:
                dups += 1
            else:
                seen.add(gram)
        if dups <= max_dup_per_doc:
            yield doc


def chunk_documents(documents: Iterable[str], chunk_size: int = 2048, sep: str = "\n\n") -> Iterable[str]:
    """Pack documents into fixed-length chunks to preserve context windows."""
    buf = []
    length = 0
    for doc in documents:
        doc = doc.strip()
        if not doc:
            continue
        add_len = len(doc) + len(sep)
        if length + add_len > chunk_size and buf:
            yield sep.join(buf).strip()
            buf = [doc]
            length = len(doc)
        else:
            buf.append(doc)
            length += add_len
    if buf:
        yield sep.join(buf).strip()


def stream_source(source: Dict[str, Any], max_tokens: int, tokenizer) -> Iterable[str]:
    """Yield cleaned text chunks from one dataset source, bounded by token budget."""
    name = source["name"]
    print(f"Loading source: {name} ...")
    try:
        ds = load_dataset(source["path"], source["split"], streaming=True, split="train")
    except Exception as e:
        print(f"  Could not load {name}: {e}")
        return

    total_chars = 0
    target_chars = max_tokens * 6  # rough char-per-token estimate
    for i, sample in enumerate(ds):
        text = sample.get(source["column"], "") if source["column"] else sample.get("text", "")
        text = normalize_text(text)
        if len(text) < 200:
            continue
        yield text
        total_chars += len(text)
        if total_chars >= target_chars:
            print(f"  Reached {name} token budget (~{max_tokens} tokens).")
            break
        if i > 0 and i % 10_000 == 0:
            print(f"  Processed {i} {name} samples...")


def tokenize_chunks_to_bins(chunks: Iterable[str], tokenizer, out_dir: str, max_tokens_per_shard: int = 100_000_000):
    """Tokenize text chunks and write uint16 .bin shards."""
    os.makedirs(out_dir, exist_ok=True)

    bos = tokenizer.bos_token_id or tokenizer.cls_token_id or 1
    eos = tokenizer.eos_token_id or 2
    pad = tokenizer.pad_token_id or 0

    current = []
    shard_idx = 0
    total_tokens = 0

    def _flush():
        nonlocal current, shard_idx, total_tokens
        if not current:
            return
        arr = np.array(current, dtype=np.uint16)
        path = os.path.join(out_dir, f"shard_{shard_idx:05d}.bin")
        arr.tofile(path)
        print(f"  Wrote {len(arr):,} tokens -> {path}")
        shard_idx += 1
        total_tokens += len(arr)
        current = []

    for chunk in chunks:
        ids = tokenizer.encode(chunk, add_special_tokens=False)
        current.append(bos)
        current.extend(ids)
        current.append(eos)
        if len(current) >= max_tokens_per_shard:
            _flush()

    _flush()
    print(f"Total pretraining tokens written: {total_tokens:,}")
    return total_tokens


def build_sft_pairs(out_path: str, tokenizer_dir: str, count: int = 1024):
    """Seed plus generate a small SFT dataset from public instruction sources."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pairs = []

    # Seed questions.
    for prompt, response in SFT_SEED_PROMPTS:
        pairs.append({"prompt": prompt, "response": response, "source": "seed"})

    # Add a few from public conversational datasets if available.
    remaining = count - len(pairs)
    if remaining > 0:
        try:
            ds = load_dataset("Open-Orca/OpenOrca", streaming=True, split="train")
            for i, sample in enumerate(ds):
                if i >= remaining:
                    break
                q = sample.get("question", "")
                a = sample.get("response", "")
                if q and a and len(a) > 40:
                    pairs.append({"prompt": q.strip(), "response": a.strip(), "source": "openorca"})
        except Exception as e:
            print(f"Could not load OpenOrca for SFT pairs: {e}")

    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Wrote {len(pairs)} SFT pairs -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", default="NeuralAI-Air-135M/tokenizer", help="Path to HF tokenizer.")
    parser.add_argument("--out_dir", default="data/pretrain_tokens", help="Where to write .bin shards.")
    parser.add_argument("--sft_out", default="data/sft_pairs.jsonl", help="Where to write SFT pairs.")
    parser.add_argument("--max_tokens_per_source", type=int, default=500_000_000, help="Soft token cap per source.")
    parser.add_argument("--samples", type=int, default=4, help="Number of sft prompt/response sets to seed.")
    parser.add_argument("--skip_sources", default="", help="Comma-separated source names to skip.")
    args = parser.parse_args()

    if not os.path.isdir(args.tokenizer_dir):
        print(f"Tokenizer dir not found: {args.tokenizer_dir}. Using SmolLM2-360M-Instruct as fallback.")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    skip = set(args.skip_sources.split(",")) if args.skip_sources else set()
    sources = [s for s in SOURCES if s["name"] not in skip]

    # Normalize token budget per source by weight.
    total_weight = sum(s["weight"] for s in sources)
    per_source = int(args.max_tokens_per_source)

    all_chunks = []
    for source in sources:
        budget = int(per_source * (source["weight"] / total_weight))
        text_stream = stream_source(source, budget, tokenizer)
        cleaned = [normalize_text(t) for t in text_stream]
        deduped = list(ngram_dedup(cleaned, n=13, max_dup_per_doc=5))
        chunked = list(chunk_documents(deduped, chunk_size=2048))
        print(f"Source {source['name']}: {len(cleaned)} docs -> {len(deduped)} after dedup -> {len(chunked)} chunks")
        all_chunks.extend(chunked)

    print(f"Total chunks to tokenize: {len(all_chunks)}")
    total_tokens = tokenize_chunks_to_bins(all_chunks, tokenizer, args.out_dir, max_tokens_per_shard=100_000_000)

    # Build SFT pairs.
    build_sft_pairs(args.sft_out, args.tokenizer_dir, count=args.samples)

    print("\nPipeline complete. You can now run:")
    print(f"  python training/continue_pretrain.py --token_dir {args.out_dir} --checkpoint <final.pt>")


if __name__ == "__main__":
    main()
