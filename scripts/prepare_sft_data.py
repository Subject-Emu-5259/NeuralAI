#!/usr/bin/env python3
"""Convert raw UltraChat 'User: ... Assistant: ...' JSONL into a formatted
SFT dataset with assistant-only loss labels.

Supported output formats:
  - intel :  text-only "### System/User/Assistant:" format. Safe for
             GPTNeoXTokenizer-based Mamba models because it uses no
             out-of-vocabulary special tokens (eos token is injected by the
             caller via --eos).
  - llama2:  [INST] ... [/INST] format.
  - chatml:  NeuralAI ChatML with <|im_start|>/। token markers.

Output fields:
  - text:     full-formatted conversation string
  - prompt:   user portion (for eval / assistant-only loss)
  - response: assistant portion
"""
import argparse
import json
import os
import re
import sys

DEFAULT_EOS = "<|endoftext|>"


def raw_to_turns(text: str):
    """Parse raw 'User: ... Assistant: ...' text into alternating turns."""
    pattern = re.compile(r"(?:\n|^)(User|Assistant):\s*")
    parts = pattern.split(text)
    turns = []
    current_role = None
    current_text = ""
    for part in parts:
        if part is None:
            continue
        if part in ("User", "Assistant"):
            if current_role is not None and current_text:
                turns.append((current_role, current_text.strip()))
            current_role = part.lower()
            current_text = ""
        else:
            current_text += part
    if current_role is not None and current_text:
        turns.append((current_role, current_text.strip()))
    return turns


def format_llama2(system: str, turns: list[tuple[str, str]], eos: str) -> tuple[str, str]:
    """Format turns into Llama2-style prompt-completion string."""
    out = ""
    system_block = f" <<SYS>>\n{system}\n<</SYS>>" if system else ""
    for i, (role, content) in enumerate(turns):
        if role == "user":
            if i == 0:
                out += f"[INST]{system_block}\n\n{content} [/INST] "
            else:
                out += f"[INST] {content} [/INST] "
        elif role == "assistant":
            out += f"{content}{eos} "
    return out.strip(), out.strip()


def format_chatml(system: str, turns: list[tuple[str, str]], eos: str) -> tuple[str, str]:
    """Format turns into NeuralAI ChatML string."""
    out = ""
    if system:
        out += f"<|im_start|>system\n{system}\n<|im_end|>\n"
    for role, content in turns:
        out += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    return out.strip(), out.strip()


def format_intel(system: str, turns: list[tuple[str, str]], eos: str) -> tuple[str, str]:
    """Text-only format safe for GPTNeoXTokenizer-based Mamba models.

    Training example ends with an explicit eos token so the model learns to
    stop generating after its response. Inference prompt matches llama.cpp's
    "intel" chat format.
    """
    out_lines = []
    if system:
        out_lines.append(f"### System:\n{system}")
    for role, content in turns:
        label = "### User:" if role == "user" else "### Assistant:"
        out_lines.append(f"{label}\n{content}")
    full = "\n".join(out_lines) + eos

    # PromptEverything except the last assistant response.
    if len(turns) >= 2 and turns[-1][0] == "assistant":
        prompt_lines = []
        if system:
            prompt_lines.append(f"### System:\n{system}")
        for role, content in turns[:-1]:
            label = "### User:" if role == "user" else "### Assistant:"
            prompt_lines.append(f"{label}\n{content}")
        prompt_lines.append("### Assistant:")
        prompt = "\n".join(prompt_lines)
        response = turns[-1][1]
    else:
        prompt = "\n".join(out_lines)
        response = ""
    return full, prompt, response


def make_example(text: str, system: str = "", fmt: str = "intel", eos: str = DEFAULT_EOS) -> dict:
    turns = raw_to_turns(text)
    if not turns:
        return {}
    if fmt == "llama2":
        full, prompt = format_llama2(system, turns, eos)
        response = turns[-1][1] if turns[-1][0] == "assistant" else ""
    elif fmt == "chatml":
        full, prompt = format_chatml(system, turns, eos)
        response = turns[-1][1] if turns[-1][0] == "assistant" else ""
    else:
        full, prompt, response = format_intel(system, turns, eos)
    return {
        "text": full,
        "prompt": prompt,
        "response": response,
    }


def process_file(input_path: str, output_path: str, system: str = "", fmt: str = "intel", eos: str = DEFAULT_EOS, max_lines: int = 0):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", obj.get("conversation", ""))
            if not text:
                continue
            example = make_example(text, system=system, fmt=fmt, eos=eos)
            if not example:
                continue
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
            if max_lines and count >= max_lines:
                break
    print(f"Wrote {count} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Format raw conversations for SFT")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL with raw conversations")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL")
    parser.add_argument("--format", choices=["intel", "chatml", "llama2"], default="intel")
    parser.add_argument("--system", default="You are NeuralAI, a helpful assistant.")
    parser.add_argument("--eos", default=DEFAULT_EOS, help="EOS token appended after each response")
    parser.add_argument("--max-lines", type=int, default=0, help="Limit output lines (0 = all)")
    args = parser.parse_args()
    process_file(args.input, args.output, system=args.system, fmt=args.format, eos=args.eos, max_lines=args.max_lines)


if __name__ == "__main__":
    main()
