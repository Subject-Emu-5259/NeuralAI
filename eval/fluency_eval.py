#!/usr/bin/env python3
"""NEA1 fluency eval: PPL + distinct-2 on 50 docs, n_ctx=512.

Same protocol as the original v3-step90000 eval. Run against any stage-2
checkpoint dir (config.json + model.safetensors). Read-only: no weight changes.
"""
import json
import math
import sys
import torch

sys.path.insert(0, "/home/hatch/workspace/nae1work/scripts")
from nae1_model import Nae1ForCausalLM  # noqa: E402
from tokenizers import Tokenizer, pre_tokenizers  # noqa: E402
from tokenizers.models import BPE  # noqa: E402

N_CTX = 512


def load_tokenizer():
    tok_dir = "/home/hatch/workspace/nae1work/nae1_tokenizer_v2"
    tok = Tokenizer(BPE.from_file(
        vocab=f"{tok_dir}/nae1_tokenizer-vocab.json",
        merges=f"{tok_dir}/nae1_tokenizer-merges.txt",
    ))
    # GPT-2-style byte-level pre-tokenization (matches the v2 BPE training)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return tok


@torch.no_grad()
def eval_checkpoint(ckpt_dir: str, docs: list[str], tok) -> dict:
    model = Nae1ForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.float32)
    model.eval()
    total_nll, total_tok = 0.0, 0
    bigrams: set[tuple[int, int]] = set()
    total_bigrams = 0
    for text in docs:
        ids = tok.encode(text).ids[:N_CTX]
        if len(ids) < 64:
            continue
        x = torch.tensor([ids], dtype=torch.long)
        out = model(input_ids=x, labels=x)
        nll = out["loss"].item() * (len(ids) - 1)  # mean loss -> sum
        total_nll += nll
        total_tok += len(ids) - 1
        for a, b in zip(ids, ids[1:]):
            bigrams.add((a, b))
            total_bigrams += 1
    ppl = math.exp(total_nll / total_tok)
    return {
        "ppl": round(ppl, 2),
        "n_tokens": total_tok,
        "distinct_2": round(len(bigrams) / max(total_bigrams, 1), 4),
    }


def main():
    ckpt = sys.argv[1]
    docs_path = sys.argv[2]
    docs = json.load(open(docs_path))["docs"]
    tok = load_tokenizer()
    # sanity: tokenizer vocab size
    print(f"tokenizer vocab: {tok.get_vocab_size()}", flush=True)
    res = eval_checkpoint(ckpt, docs, tok)
    res["checkpoint"] = ckpt
    res["n_docs"] = len(docs)
    res["n_ctx"] = N_CTX
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
