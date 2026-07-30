"""Pre-training data pipeline for NeuralAI-Air-135M.

Downloads public-domain corpora via HuggingFace ``datasets``, applies
deduplication (exact SHA-256 + optional MinHash LSH), quality filtering,
tokenization with the existing 32K BPE tokenizer, and shards the result
into memory-mappable ``uint16`` ``.bin`` files.

Typical usage::

    python -m training.pretrain.data_pipeline --config training/pretrain/config_pretrain.yaml
"""

import os
import sys
import json
import hashlib
import re
import unicodedata
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Apply Unicode NFC normalization and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_dedup(documents: List[str]) -> Tuple[List[str], int]:
    """Remove exact duplicates using SHA-256 of normalized lowercase text.

    Returns:
        ``(unique_documents, duplicates_removed)``
    """
    seen: set = set()
    out: List[str] = []
    dupes = 0
    for doc in documents:
        h = hashlib.sha256(normalize_text(doc).lower().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(doc)
        else:
            dupes += 1
    return out, dupes


def near_dedup(
    documents: List[str],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> Tuple[List[str], int]:
    """Remove near-duplicates using MinHash LSH. Keeps the longest doc per cluster.

    Args:
        documents: List of raw text documents.
        threshold: Jaccard similarity threshold for duplication.
        num_perm: Number of MinHash permutations.

    Returns:
        ``(filtered_documents, duplicates_removed)``
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise ImportError(
            "datasketch is required for near-deduplication. "
            "Install: pip install datasketch>=1.5.0"
        ) from exc

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: Dict[str, Any] = {}

    for i, doc in enumerate(documents):
        m = MinHash(num_perm=num_perm)
        words = doc.split()
        for j in range(len(words) - 4):
            shingle = " ".join(words[j : j + 5])
            m.update(shingle.encode("utf-8"))
        sid = str(i)
        minhashes[sid] = m
        lsh.insert(sid, m)

    keep: set = set()
    removed = 0
    for i, doc in enumerate(documents):
        sid = str(i)
        dupes = lsh.query(minhashes[sid])
        if not dupes:
            keep.add(i)
            continue
        dupes_int = [int(d) for d in dupes]
        longest = max(dupes_int, key=lambda idx: len(documents[idx]))
        if longest == i:
            keep.add(i)
        else:
            removed += 1

    return [documents[i] for i in sorted(keep)], removed


def clean_document(doc: str) -> Optional[str]:
    """Apply quality filters. Return ``None`` if the document should be discarded.

    Filters:
      - Length < 100 or > 100_000 characters
      - >30%% lines start with boilerplate phrases
      - >50%% non-alphanumeric characters
      - >30%% identical lines (repetition)
      - Punctuation runs (e.g. ``!!!!!!``)
      - Strips HTML remnants
    """
    doc = normalize_text(doc)
    if len(doc) < 100 or len(doc) > 100_000:
        return None

    lines = doc.split("\n")
    if not lines:
        return None

    boilerplate_prefixes = (
        "Cookie Policy",
        "All rights reserved",
        "Privacy Policy",
        "Terms of Use",
        "Copyright",
    )
    boilerplate_count = sum(
        1
        for l in lines
        if any(l.strip().startswith(bp) for bp in boilerplate_prefixes)
    )
    if boilerplate_count / len(lines) > 0.30:
        return None

    if len([c for c in doc if c.isalnum()]) / max(len(doc), 1) < 0.5:
        return None

    if len(set(lines)) / max(len(lines), 1) < 0.7:
        return None

    if re.search(r"[\.\!\?\*]{4,}", doc):
        return None

    # Strip HTML remnants
    doc = re.sub(r"<[^>]+>", "", doc)
    doc = re.sub(r"&\w+;", "", doc)
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    return doc.strip() or None


def download_source(
    source_name: str,
    spec: Dict[str, Any],
    cache_dir: str = "data/raw",
) -> List[str]:
    """Download a dataset source via HuggingFace ``datasets``.

    Streams if ``spec["streaming"]`` is ``True``, otherwise full download.
    Caps downloads at ``target_chars`` to avoid pulling infinite corpora.

    Args:
        source_name: Human-readable source identifier (e.g. ``c4``).
        spec: Dictionary with keys ``dataset``, ``config``, ``streaming``,
            ``target_chars``.
        cache_dir: Local cache directory for HF datasets.

    Returns:
        List of raw text documents.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets library is required. Install: pip install datasets>=2.14.0"
        ) from exc

    dataset_name = spec["dataset"]
    config_name = spec.get("config")
    streaming = spec.get("streaming", False)
    target_chars = spec.get("target_chars", 100_000_000)

    logger.info(
        f"Downloading {source_name} from {dataset_name} (streaming={streaming}) ..."
    )
    kwargs: Dict[str, Any] = {"cache_dir": cache_dir, "streaming": streaming}
    if config_name:
        kwargs["name"] = config_name

    try:
        ds = load_dataset(dataset_name, **kwargs)
    except Exception as exc:
        logger.error(f"Failed to load {source_name}: {exc}")
        return []

    # Determine text column
    text_col = "text"
    if isinstance(ds, dict):
        split_name = list(ds.keys())[0]
        ds = ds[split_name]

    features = getattr(ds, "features", None)
    if features is not None and text_col not in features:
        for col in ("content", "body", "sentence"):
            if col in features:
                text_col = col
                break

    documents: List[str] = []
    total_chars = 0
    for i, row in enumerate(ds):
        if i % 10_000 == 0 and i > 0:
            logger.info(f"  {source_name}: {i} docs, {total_chars:,} chars")
        text = row.get(text_col, "")
        if not isinstance(text, str):
            continue
        documents.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break

    logger.info(
        f"Downloaded {source_name}: {len(documents)} docs, {total_chars:,} chars"
    )
    return documents


def tokenize_and_shard(
    documents: List[str],
    out_dir: Path,
    tokenizer,
    seq_len: int = 512,
    shard_tokens: int = 50_000_000,
) -> Dict[str, Any]:
    """Tokenize documents, pack with EOS, and shard into ``uint16`` ``.bin`` files.

    Args:
        documents: Clean text documents.
        out_dir: Directory to write ``shard_*.bin`` files.
        tokenizer: HuggingFace tokenizer with ``eos_token_id``.
        seq_len: Sequence length (stored in manifest only; shards are flat).
        shard_tokens: Number of tokens per shard (~100 MB for uint16).

    Returns:
        Manifest dictionary with ``total_tokens``, ``num_shards``, ``shards``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_tokens: List[int] = []
    shard_idx = 0
    total_tokens = 0
    eos_id = tokenizer.eos_token_id or 2

    for doc in tqdm(documents, desc="Tokenizing"):
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        tokens.append(eos_id)
        flat_tokens.extend(tokens)
        total_tokens += len(tokens) + 1

        while len(flat_tokens) >= shard_tokens:
            shard = flat_tokens[:shard_tokens]
            arr = np.array(shard, dtype=np.uint16)
            arr.tofile(out_dir / f"shard_{shard_idx:05d}.bin")
            flat_tokens = flat_tokens[shard_tokens:]
            shard_idx += 1

    if flat_tokens:
        arr = np.array(flat_tokens, dtype=np.uint16)
        arr.tofile(out_dir / f"shard_{shard_idx:05d}.bin")
        shard_idx += 1

    shards = []
    for i in range(shard_idx):
        path = out_dir / f"shard_{i:05d}.bin"
        size = path.stat().st_size // 2
        shards.append({"file": str(path), "num_tokens": int(size)})

    return {
        "total_tokens": total_tokens,
        "num_shards": shard_idx,
        "shards": shards,
    }


def build_splits(
    processed_dir: Path,
    tokenizer,
    seq_len: int = 512,
    seed: int = 42,
    val_ratio: float = 0.01,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build train/val splits from cleaned documents.

    Reads ``*_clean.jsonl`` files in *processed_dir*, shuffles globally,
    reserves ``val_ratio`` for validation, tokenizes, and writes shards.

    Returns:
        ``(train_manifest, val_manifest)`` dictionaries.
    """
    all_docs: List[str] = []
    source_stats: Dict[str, Any] = {}

    for jsonl_path in sorted(processed_dir.glob("*_clean.jsonl")):
        source_name = jsonl_path.name.replace("_clean.jsonl", "")
        docs: List[str] = []
        with open(jsonl_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    text = line
                if text:
                    docs.append(text)
        source_stats[source_name] = {"raw_docs": len(docs)}
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError(f"No cleaned documents found in {processed_dir}")

    import random

    rng = random.Random(seed)
    rng.shuffle(all_docs)

    val_count = max(1, int(len(all_docs) * val_ratio))
    val_docs = all_docs[:val_count]
    train_docs = all_docs[val_count:]

    logger.info(f"Split: {len(train_docs)} train docs, {len(val_docs)} val docs")

    train_dir = processed_dir.parent / "tokenized" / "train"
    val_dir = processed_dir.parent / "tokenized" / "val"

    train_manifest = tokenize_and_shard(
        train_docs, train_dir, tokenizer, seq_len=seq_len
    )
    val_manifest = tokenize_and_shard(
        val_docs, val_dir, tokenizer, seq_len=seq_len
    )

    for manifest in (train_manifest, val_manifest):
        manifest["sequence_length"] = seq_len
        manifest["sources"] = source_stats

    train_manifest["split"] = "train"
    val_manifest["split"] = "val"

    (train_dir / "manifest.json").write_text(
        json.dumps(train_manifest, indent=2)
    )
    (val_dir / "manifest.json").write_text(
        json.dumps(val_manifest, indent=2)
    )

    return train_manifest, val_manifest


def run_pipeline(config: Dict[str, Any]) -> None:
    """Execute the full data pipeline end-to-end."""
    data_dir = Path(config.get("data_dir", "data/pretrain"))
    sources = config.get("sources", {})
    tokenizer_path = config.get("tokenizer_path", "NeuralAI-Air-135M-HF")
    seq_len = config.get("context_length", 512)
    seed = config.get("seed", 42)

    # Load tokenizer
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install: pip install transformers>=4.40.0"
        ) from exc

    tokenizer_json = os.path.join(tokenizer_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        raise FileNotFoundError(
            f"tokenizer.json not found at {tokenizer_path}"
        )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    total_raw = 0
    total_clean = 0
    total_exact_dupes = 0

    for source_name, spec in sources.items():
        docs = download_source(source_name, spec, cache_dir=str(raw_dir))
        if not docs:
            logger.warning(f"No documents downloaded for {source_name}")
            continue

        docs, exact_dupes = exact_dedup(docs)
        total_exact_dupes += exact_dupes
        logger.info(
            f"{source_name} exact dedup: removed {exact_dupes} docs"
        )

        cleaned = [clean_document(d) for d in docs]
        cleaned = [d for d in cleaned if d is not None]
        total_raw += len(docs)
        total_clean += len(cleaned)
        logger.info(
            f"{source_name} cleaning: kept {len(cleaned)} / {len(docs)} docs"
        )

        clean_path = processed_dir / f"{source_name}_clean.jsonl"
        with open(clean_path, "w") as fh:
            for d in cleaned:
                fh.write(json.dumps({"text": d}) + "\n")

    logger.info(
        f"Pipeline stage: exact dedup removed {total_exact_dupes} docs "
        f"({total_exact_dupes / max(total_raw, 1):.2%})"
    )
    logger.info(
        f"Pipeline stage: cleaning kept {total_clean} / {total_raw} docs "
        f"({total_clean / max(total_raw, 1):.2%})"
    )

    train_manifest, val_manifest = build_splits(
        processed_dir, tokenizer, seq_len=seq_len, seed=seed
    )

    logger.info("=== Pipeline Complete ===")
    logger.info(
        f"Train tokens: {train_manifest['total_tokens']:,} "
        f"in {train_manifest['num_shards']} shards"
    )
    logger.info(
        f"Val tokens:   {val_manifest['total_tokens']:,} "
        f"in {val_manifest['num_shards']} shards"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuralAI-Air-135M Pre-Training Data Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/pretrain/config_pretrain.yaml",
        help="Path to pretrain config YAML",
    )
    parser.add_argument(
        "--global-dedup",
        action="store_true",
        help="Enable global MinHash near-deduplication (memory intensive)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required. Install: pip install pyyaml"
        ) from exc

    with open(args.config) as fh:
        full_config = yaml.safe_load(fh)

    run_pipeline(full_config.get("data", {}))


if __name__ == "__main__":
    main()
