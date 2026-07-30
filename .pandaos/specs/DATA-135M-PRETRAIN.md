# NeuralAI-Air-135M Pre-Training Data Pipeline

**Spec ID:** DATA-135M-PRETRAIN  
**Version:** 1.0  
**Author:** AI Engineer (NeuralAI)  
**Date:** 2026-07-29  
**Target:** 1.0–1.5B tokens, custom 32K BPE tokenizer

---

## 1. Executive Summary

This document defines the data pipeline for pre-training NeuralAI-Air-135M from scratch. We acquire five public-domain corpora, apply exact + near-deduplication, clean and normalize text, tokenize with the existing `NeuralAI-Air-135M-HF/tokenizer.json`, and shard the result into memory-mappable `.bin` files for streaming training. The pipeline is reproducible: every stage is scripted, seeded, and logged.

---

## 2. Data Sources & Mixing Ratios

All sources are freely downloadable via HuggingFace `datasets` or direct HTTP dump. No licensing fees.

| Source | HF Dataset / Origin | Target Raw Tokens | Mix % | Notes |
|--------|--------------------|-------------------|-------|-------|
| C4 (English) | `c4`, `en` subset (streaming) | ~350M | 30% | Web crawl, noisy but vast. |
| OpenWebText | `Skylion007/openwebtext` | ~200M | 20% | Reddit-outbound links, higher quality than raw C4. |
| Books (Gutenberg) | `HuggingFaceM4/ProjectGutenberg` | ~300M | 30% | Public domain prose; long coherent narratives. |
| Wikipedia (en) | `wikimedia/wikipedia`, `20231101.en` | ~100M | 10% | Factual, well-structured; caps knowledge density. |
| StackExchange | `HuggingFaceTB/stackexchange` | ~100M | 10% | Technical Q&A; improves reasoning and formatting. |
| **Total Target** | | **~1.05B** | **100%** | Post-dedup yield expected: **~1.0B tokens**. |

**Raw→Clean Yield Assumption:** 5–15% volume loss from deduplication and filtering. We oversample each source by ~10% during download to hit the post-dedup target.

---

## 3. Download Strategy

### 3.1 Streaming vs Full Download
- **C4:** Use `datasets` streaming (`stream=True`). The full en split is ~750 GB raw text; we only materialize the first ~350M tokens worth of documents into local disk.
- **OpenWebText, StackExchange, Wikipedia:** Full download (each < 50 GB raw). Use `datasets.load_dataset(..., cache_dir="data/raw/")`.
- **Project Gutenberg:** Full download (~30 GB). Cache locally.

### 3.2 Resilience
- All downloads are wrapped in `retry` with exponential backoff (HF hub flakiness).
- Checksum verification where available (Wikipedia dumps provide MD5).
- Resume partial downloads via `datasets` built-in caching.

---

## 4. Deduplication Pipeline

Deduplication runs in two passes: **exact** then **near-duplicate**. Both are deterministic.

### 4.1 Exact Deduplication
- **Normalization before hashing:**
  1. Unicode NFC normalization.
  2. Collapse all whitespace (`\s+` → ` `).
  3. Strip leading/trailing whitespace.
  4. Lowercase (optional; we keep case for quality but lowercase for dedup hashing to catch `Hello` vs `hello` duplicates).
- **Hash function:** SHA-256 of normalized text.
- **Storage:** In-memory `set` of hashes (1B tokens ≈ ~2M documents; 64 bytes/hash × 2M = ~128 MB — trivially fits in RAM).
- **Scope:** Applied per-source first, then globally across all sources.

### 4.2 Near-Deduplication (MinHash LSH)
- **Library:** `datasketch` (MinHash + MinHashLSH).
- **Parameters:**
  - Shingle size: **5-grams** (words).
  - Permutations: **128**.
  - Bands: **16** (rows per band = 8).
  - Jaccard threshold: **0.85**.
- **Process:**
  1. For each document, compute MinHash signature from word 5-grams.
  2. Insert into LSH index.
  3. Query index for duplicates; keep the **longest** document in each duplicate cluster.
- **Performance:** ~500 docs/sec on a single CPU core. For 2M documents, ~1 hour.
- **Scope:** Global (cross-source). Prevents C4 and OpenWebText from sharing the same Reddit thread content.

### 4.3 Code / Boilerplate Deduplication (Bonus)
- Documents with > 40% lines matching `^\s*(#|//|import|from|function|def|class)` are flagged as mass code boilerplate (e.g., repeated license headers). Keep only the first occurrence in a sliding window of 10K docs.

---

## 5. Cleaning & Normalization

Applied after deduplication, before tokenization.

| Filter | Rule | Rationale |
|--------|------|-----------|
| Min line length | Discard documents where median line < 20 chars | Removes tables, indexes, menus |
| Max non-alpha ratio | Discard if > 50% chars are non-alphanumeric | Removes binary garbage, hex dumps |
| URL density | Discard if > 10% tokens look like URLs | Removes link farms |
| Repetition | Discard if > 30% of lines are identical | Removes headers/footers/duplicated boilerplate |
| Punctuation run | Discard if contains `……` or `!!!!!!` (>3 repeats) | Removes ASCII art / spam |
| HTML remnant | Strip `<tag>` and `&entity;` | C4 / Wikipedia carry HTML |
| Whitespace | Collapse `\n{3,}` to `\n\n`; trim | Normalizes paragraph breaks |
| Encoding | Force UTF-8; drop on decode error | Sanitizes edge-case dumps |

**Quality Gate:** At least 80% of documents must survive cleaning per source. If a source drops below 70%, pause pipeline and audit filters (risk of over-cleaning).

---

## 6. Tokenization & Packing

### 6.1 Tokenizer
- Use the existing custom BPE tokenizer at `NeuralAI-Air-135M-HF/tokenizer.json`.
- Verify vocab size = 32,000 and `eos_token_id = 2`, `bos_token_id = 1`, `pad_token_id = 0`.
- **Pre-tokenization check:** Encode a 1K-document sample and confirm ~99.5% of tokens are < vocab_size (catches misaligned tokenizer merges).

### 6.2 Packing Strategy
Pre-training uses **concatenation + chunking** (no padding). Documents are separated by `</s>` (EOS token) and concatenated into a flat token stream. The stream is sliced into fixed-length 1024-token blocks.

```
Doc A tokens … </s> Doc B tokens … </s> Doc C tokens … →
[1024] [1024] [1024] …
```

- **Cross-document masking:** Causal LM mask does not span across `</s>` boundaries? In practice, for small models, the performance difference is negligible and packing is standard (e.g., Dolma, SlimPajama). We pack naïvely; the model learns EOS as a hard boundary.
- **Efficiency:** ~2–5% tokens are `</s>` separators; acceptable overhead.

### 6.3 Token Storage Format
- **Format:** Flat binary files of `uint16` (Little-Endian). Vocab 32K < 65536, so `uint16` is sufficient and halves storage vs `int32`.
- **Shard size:** 100 MB per `.bin` file ≈ 50M tokens.
- **Total files:** ~20–25 shards for 1B tokens.
- **Loading:** `np.memmap` for zero-copy, random-seekable access during training.

```python
# Example shard structure
tokens = np.memmap("shard_00000.bin", dtype=np.uint16, mode="r")
# shape: (50_000_000,) — flat
# training loader reshapes to (-1, 1024)
```

---

## 7. Sharding & Storage Format

### 7.1 Train / Validation Split
- **Train:** 99% of tokens (~1.0B).
- **Validation:** 1% of tokens (~10M), stratified by source.
- **Stratification:** For each source, reserve the last 1% of documents (after shuffling with a fixed seed) for validation. Ensures val distribution matches train.

### 7.2 Manifest Files
Each split has a JSON manifest:
```json
{
  "split": "train",
  "total_tokens": 1004531200,
  "sequence_length": 1024,
  "num_sequences": 981,
  "shards": [
    {"file": "shard_00000.bin", "num_tokens": 52428800},
    {"file": "shard_00001.bin", "num_tokens": 52428800}
  ],
  "sources": {
    "c4": 0.30,
    "openwebtext": 0.20,
    "gutenberg": 0.30,
    "wikipedia": 0.10,
    "stackexchange": 0.10
  }
}
```

---

## 8. Streaming & Loading

### 8.1 Why Not WebDataset / HF Datasets Streaming?
- **HF Datasets streaming** is excellent for C4 download, but for the actual training loop we need deterministic, seekable, low-overhead access.
- **WebDataset** is powerful but adds a tar-parsing overhead that is unnecessary for a 1B-token corpus.
- **Decision:** Pre-process everything into `.bin` shards, then load with a lightweight PyTorch `IterableDataset` that interleaves shards according to mixing proportions.

### 8.2 MixedStreamingDataset (PyTorch)
```python
class MixedStreamingDataset(IterableDataset):
    def __init__(self, manifest_path: str, seq_len: int):
        self.manifest = json.load(open(manifest_path))
        self.seq_len = seq_len
        # build per-source shard lists with sampling weights

    def __iter__(self):
        workers = get_worker_info()
        # interleave shards by source proportion
        # yield (seq_len,) int64 tensors
```

**Stratified sampling algorithm:**
1. For each source, open all its shards via `np.memmap`.
2. Maintain a per-source iterator that yields 1024-token chunks.
3. At each training step, sample the next chunk from a source according to the mixing proportion (e.g., C4: 30%, Books: 30%, …).
4. Optional: apply a small temperature to the sampling distribution to smooth source switching (avoids batches of all-Wikipedia).

---

## 9. Validation Split Construction

1. **After cleaning + dedup**, shuffle each source’s document list with `seed=42`.
2. Take the **last 1%** of documents from each source as validation.
3. Tokenize and pack val documents **without cross-document contamination** — each val document is padded/truncated individually to 1024, not concatenated with neighbors. This gives a clean perplexity that does not leak context from unrelated docs.
4. Store val as a single `.bin` shard (~20 MB).

---

## 10. Directory Structure

```
data/
├── raw/                                # downloaded HF datasets (ephemeral, cache)
│   ├── c4/
│   ├── openwebtext/
│   ├── gutenberg/
│   ├── wikipedia/
│   └── stackexchange/
├── processed/                          # deduped + cleaned plain text
│   ├── c4_clean.jsonl
│   ├── openwebtext_clean.jsonl
│   ├── gutenberg_clean.jsonl
│   ├── wikipedia_clean.jsonl
│   ├── stackexchange_clean.jsonl
│   └── dedup_manifest.json           # hashes + decisions
├── tokenized/                          # uint16 .bin shards
│   ├── train/
│   │   ├── manifest.json
│   │   ├── shard_00000.bin
│   │   ├── shard_00001.bin
│   │   └── ...
│   └── val/
│       ├── manifest.json
│       └── shard_val_00000.bin
├── tokenizer/                          # symlink to NeuralAI-Air-135M-HF/
└── logs/
    ├── download.log
    ├── dedup.log
    ├── clean.log
    └── tokenize.log
```

**Storage footprint**
- Raw downloads: ~60–80 GB (mostly C4).
- Cleaned text: ~20 GB.
- Tokenized (uint16): ~2 GB.
- **Total pipeline disk:** < 100 GB.

---

## 11. Code Skeleton

### `src/data_pipeline.py`
```python
import hashlib, json, re, unicodedata
from pathlib import Path
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset
import numpy as np
from transformers import PreTrainedTokenizerFast

TOKENIZER_PATH = "NeuralAI-Air-135M-HF/tokenizer.json"
SEQ_LEN = 1024
SHARD_TOKENS = 50_000_000  # 100 MB uint16

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def exact_dedup(documents: list[str]) -> list[str]:
    seen = set()
    out = []
    for doc in documents:
        h = hashlib.sha256(normalize_text(doc).lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(doc)
    return out

def near_dedup(documents: list[str], threshold=0.85, num_perm=128) -> list[str]:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    for i, doc in enumerate(documents):
        m = MinHash(num_perm=num_perm)
        for shingle in set(doc.split()):
            m.update(shingle.encode("utf8"))
        minhashes[i] = m
        lsh.insert(str(i), m)

    keep = set()
    for i, doc in enumerate(documents):
        dupes = lsh.query(minhashes[i])
        dupes = [int(d) for d in dupes]
        # keep the longest document in the cluster
        longest = max(dupes, key=lambda idx: len(documents[idx]))
        keep.add(longest)
    return [documents[i] for i in sorted(keep)]

def clean_document(doc: str) -> str | None:
    doc = normalize_text(doc)
    lines = doc.split("\n")
    if sum(1 for l in lines if len(l) >= 20) / max(len(lines), 1) < 0.5:
        return None
    if len([c for c in doc if c.isalnum()]) / max(len(doc), 1) < 0.5:
        return None
    if doc.count("http") / max(len(doc.split()), 1) > 0.10:
        return None
    # additional filters as needed
    return doc

def tokenize_and_shard(documents: list[str], out_dir: Path, tokenizer, seq_len=1024):
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_tokens = []
    shard_idx = 0
    for doc in documents:
        tokens = tokenizer.encode(doc, add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        flat_tokens.extend(tokens)
        while len(flat_tokens) >= shard_idx * SHARD_TOKENS + SHARD_TOKENS:
            # write shard when full
            shard = flat_tokens[shard_idx * SHARD_TOKENS : (shard_idx + 1) * SHARD_TOKENS]
            arr = np.array(shard, dtype=np.uint16)
            arr.tofile(out_dir / f"shard_{shard_idx:05d}.bin")
            shard_idx += 1
    # tail shard
    if len(flat_tokens) > shard_idx * SHARD_TOKENS:
        shard = flat_tokens[shard_idx * SHARD_TOKENS :]
        arr = np.array(shard, dtype=np.uint16)
        arr.tofile(out_dir / f"shard_{shard_idx:05d}.bin")
```

### `src/data_loader.py` (training-time)
```python
import json, numpy as np
from pathlib import Path
from torch.utils.data import IterableDataset, get_worker_info
import torch

class MixedStreamingDataset(IterableDataset):
    def __init__(self, manifest_path: str, seq_len: int = 1024):
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.seq_len = seq_len
        self.shards = self._load_shards()

    def _load_shards(self):
        shards = {}
        for entry in self.manifest["shards"]:
            path = Path(entry["file"])
            mem = np.memmap(path, dtype=np.uint16, mode="r")
            shards[path.name] = mem
        return shards

    def __iter__(self):
        worker = get_worker_info()
        # deterministic interleaving by source proportion
        # simplified: yield random-access 1024-blocks from flat memmap
        total_seqs = sum(len(m) for m in self.shards.values()) // self.seq_len
        indices = torch.randperm(total_seqs)  # shuffle global sequence index
        for idx in indices:
            # map idx to (shard, offset)
            # ...
            yield torch.from_numpy(seq.astype(np.int64))
```

*Note: the loader above is a skeleton. The production version maps global indices to `(shard_name, byte_offset)` deterministically across workers using `torch.utils.data.get_worker_info()` for multi-worker DataLoader compatibility.*

---

## 12. Quality Metrics & Acceptance Criteria

Before the training loop starts, the data pipeline must pass these gates:

| Gate | Metric | Target | Action on Fail |
|------|--------|--------|----------------|
| A | Total train tokens | ≥ 1.0B | Re-download or relax filters |
| B | Val tokens | ≥ 10M | Re-split with smaller ratio |
| C | Source mix deviation | ±2% of target | Adjust sampling weights |
| D | Exact dup rate | < 1% of raw | Review dedup normalization |
| E | Near-dup clusters | < 5% of docs | Tune MinHash threshold |
| F | Avg tokens / document | 200–800 | Audit cleaning (over-trimming drops this) |
| G | Unk token rate | < 0.05% | Verify tokenizer coverage on sample |
| H | UTF-8 decode success | 100% | Re-encode offending files |

**Model QA will run:**
1. Per-source token distribution histogram.
2. n-gram diversity (unique 4-grams / total 4-grams) — should be > 0.75.
3. Perplexity of a small reference model (e.g., GPT-2 124M) on val set — sanity check that data is not garbage.

---

## 13. Execution Plan

| Step | Task | Est. Time | Tooling |
|------|------|-----------|---------|
| 1 | Download all sources to `data/raw/` | 2–4 hrs | `datasets.load_dataset(..., streaming=True)` for C4; direct for others |
| 2 | Exact dedup per source | 30 min | `data_pipeline.exact_dedup()` |
| 3 | Near dedup globally | 1–2 hrs | `datasketch` |
| 4 | Clean + filter | 1 hr | `data_pipeline.clean_document()` |
| 5 | Train/val split + tokenize | 2 hrs | `tokenizers` + `numpy` |
| 6 | Shard into `.bin` + write manifests | 30 min | `numpy.tofile()` |
| 7 | QA audit (gates A–H) | 1 hr | `scripts/audit_data.py` |
| **Total** | | **~8–12 hrs** | Single node, CPU-heavy (64 GB RAM recommended) |

*Can run on the A100 instance itself (Lambda/RunPod) or on a local workstation with `rsync` / `scp` of the final `data/tokenized/` directory (~2 GB).*

---

## 14. Python Package List

Add these to `requirements.txt` alongside the training packages:

```
datasets>=2.14.0
transformers>=4.40.0
tokenizers>=0.19.0
datasketch>=1.5.0
zstandard                # C4 decompression
numpy>=1.24.0
tqdm
```

*No GPU required for the pipeline — it is CPU- and memory-bound.*

---

## 15. Open Questions / Decisions

1. **Local vs Cloud preprocessing:** Should the 8–12 hour pipeline run on the A100 rental (burning GPU rent time) or on a local workstation before uploading `data/tokenized/`?
2. **StackExchange scope:** All StackExchange sites, or limit to `stackoverflow.com`, `superuser.com`, and `serverfault.com` for higher technical quality?
3. **Book genre balance:** Project Gutenberg skews 19th-century fiction. Do we want to downsample heavily-represented authors (e.g., Dickens) to improve diversity?
4. **Validation contamination:** Should val documents be explicitly removed from the training split *before* packing, or is the 1% tail split sufficient?

---

**Next Step:** Builder implements `src/data_pipeline.py` and `src/data_loader.py`; Model QA audits the output against gates A–H before any GPU time is consumed.
