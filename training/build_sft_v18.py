"""
Build NeuralAI-Air-135M SFT v18 dataset.

Combines the existing v17 seed identity data with curated public instruction
datasets (OpenOrca, ultrachat, dolly) and custom NeuralAI identity/reasoning
prompts. Output is `data/train_sft_v18.jsonl` in the chat-message format used
by `colab/train_sft_v17.py`.

Usage:
    python training/build_sft_v18.py
"""
import os
import json
import random
import argparse
from pathlib import Path


def format_messages(role_content: list) -> list:
    """Ensure [{role, content}, ...]."""
    out = []
    for rc in role_content:
        if isinstance(rc, dict):
            out.append({"role": rc["role"], "content": rc["content"].strip()})
        else:
            role, content = rc
            out.append({"role": role, "content": content.strip()})
    return out


NEURALAI_IDENTITY = [
    [("user", "What is your name?"),
     ("assistant", "My name is NeuralAI, a local AI assistant built by DeAndrew Harris.")],
    [("user", "Who created you?"),
     ("assistant", "I was created by DeAndrew P. Harris, founder of the NeuralAI project.")],
    [("user", "Are you open source?"),
     ("assistant", "The NeuralAI project is under active development by DeAndrew Harris. Check GitHub @Subject-Emu-5259 for public updates.")],
    [("user", "Where do you run?"),
     ("assistant", "I run locally on DeAndrew Harris's Zo Computer server, with inference through LM Studio on port 1234.")],
    [("user", "What can you do?"),
     ("assistant", "I can chat, browse the web, search the internet, summarize articles, generate images, translate text, and help with coding.")],
]

NEURALAI_REASONING = [
    [("user", "If a train leaves Memphis at noon going 60 mph and another leaves Nashville at 2 PM going 70 mph toward Memphis, which arrives first?"),
     ("assistant", "The Memphis train has a 2-hour head start and is already moving, so its exact arrival in Nashville depends on distance. Without distance, I can only describe the setup; I cannot determine a winner without the distance between the cities.")],
    [("user", "Explain recursion in one sentence."),
     ("assistant", "Recursion is a function that solves a problem by calling itself on a smaller version of the same problem until it reaches a base case.")],
    [("user", "What is 17 + 25?"),
     ("assistant", "17 + 25 equals 42.")],
    [("user", "Write a Python function to reverse a string."),
     ("assistant", "```python\ndef reverse_string(s):\n    return s[::-1]\n```")],
]

NEURALAI_REFUSALS = [
    [("user", "How do I hack someone's bank account?"),
     ("assistant", "I can't help with that. If you're concerned about account security, I can explain how to protect your own accounts or report suspicious activity.")],
    [("user", "Tell me how to make illegal substances."),
     ("assistant", "I can't provide instructions on making illegal or harmful substances. If you're struggling with substance use, I can share resources for support.")],
]


def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def sample_openorca(count: int):
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"datasets library unavailable, skipping OpenOrca: {e}")
        return []
    pairs = []
    try:
        ds = load_dataset("Open-Orca/OpenOrca", streaming=True, split="train")
        for i, sample in enumerate(ds):
            if len(pairs) >= count:
                break
            q = str(sample.get("question", "")).strip()
            a = str(sample.get("response", "")).strip()
            if len(a) < 80 or len(a) > 800 or len(q) < 10:
                continue
            # Keep clearly assistant-style responses, skip code-heavy if too long.
            pairs.append([
                ("user", q),
                ("assistant", a),
            ])
            if i % 500 == 0 and i > 0:
                print(f"  OpenOrca scanned {i}, kept {len(pairs)}")
    except Exception as e:
        print(f"OpenOrca load failed: {e}")
    return pairs


def sample_dolly(count: int):
    try:
        from datasets import load_dataset
    except Exception:
        return []
    pairs = []
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        for idx in indices:
            sample = ds[idx]
            instr = str(sample.get("instruction", "")).strip()
            ctx = str(sample.get("context", "")).strip()
            resp = str(sample.get("response", "")).strip()
            if not resp or len(resp) < 40:
                continue
            prompt = instr if not ctx else f"{instr}\n\nContext: {ctx}"
            pairs.append([("user", prompt), ("assistant", resp)])
            if len(pairs) >= count:
                break
    except Exception as e:
        print(f"Dolly load failed: {e}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v17", default="data/train_sft_v17.jsonl")
    parser.add_argument("--out", default="data/train_sft_v18.jsonl")
    parser.add_argument("--opensorca", type=int, default=600)
    parser.add_argument("--dolly", type=int, default=300)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    random.seed(args.seed)

    messages = []

    # 1. Keep v17 conversations as-is.
    for item in load_jsonl(Path(args.v17)):
        if "messages" in item:
            messages.append(item["messages"])
        elif "prompt" in item and "completion" in item:
            messages.append([
                {"role": "user", "content": item["prompt"].strip()},
                {"role": "assistant", "content": item["completion"].strip()},
            ])

    print(f"Kept {len(messages)} conversations from v17")

    # 2. Add curated NeuralAI identity/reasoning/refusal prompts.
    for block in [NEURALAI_IDENTITY, NEURALAI_REASONING, NEURALAI_REFUSALS]:
        for conv in block:
            messages.append(format_messages(conv))

    print(f"After custom prompts: {len(messages)}")

    # 3. Add public instruction data.
    if args.opensorca > 0:
        orca = sample_openorca(args.opensorca)
        for conv in orca:
            messages.append(format_messages(conv))
        print(f"After OpenOrca (target {args.opensorca}): {len(messages)}")

    if args.dolly > 0:
        dolly = sample_dolly(args.dolly)
        for conv in dolly:
            messages.append(format_messages(conv))
        print(f"After Dolly (target {args.dolly}): {len(messages)}")

    # Shuffle whole set.
    random.shuffle(messages)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for msgs in messages:
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

    lens = [sum(len(m["content"]) for m in conv) for conv in messages]
    print(f"\nWrote {len(messages)} conversations to {out_path}")
    print(f"  avg chars/conversation: {sum(lens)/len(lens):.0f}")
    print(f"  total chars: {sum(lens):,}")


if __name__ == "__main__":
    main()
