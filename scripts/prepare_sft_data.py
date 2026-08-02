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

# Small-model friendly caps.  Empirically ~1/3 chars per token for English text.
DEFAULT_MAX_USER_CHARS = 512
DEFAULT_MAX_RESPONSE_CHARS = 768


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


def _last_pair(turns: list[tuple[str, str]]):
    """Return the last user turn and its assistant reply, if any.

    Small base models (130M) cannot track long multi-turn conversations, and
    long prior turns blow the context window, so we keep only the final pair.
    """
    if not turns:
        return []
    if turns[-1][0] == "assistant" and len(turns) >= 2:
        return [turns[-2], turns[-1]]
    return [turns[-1]]


def _crop(text: str, max_chars: int) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    # Prefer breaking at a word boundary.
    cut = text.rfind(" ", max(0, max_chars - 50), max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut].strip()


def format_llama2(system: str, turns: list[tuple[str, str]], eos: str) -> tuple[str, str, str]:
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
    response = turns[-1][1] if turns and turns[-1][0] == "assistant" else ""
    return out.strip(), out.strip(), response


def format_chatml(system: str, turns: list[tuple[str, str]], eos: str) -> tuple[str, str, str]:
    """Format turns into NeuralAI ChatML string."""
    out = ""
    if system:
        out += f"<|im_start|>system\n{system}\n<|im_end|>\n"
    for role, content in turns:
        out += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
    response = turns[-1][1] if turns and turns[-1][0] == "assistant" else ""
    return out.strip(), out.strip(), response


def format_intel(
    system: str,
    turns: list[tuple[str, str]],
    eos: str,
    max_user_chars: int = DEFAULT_MAX_USER_CHARS,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
) -> tuple[str, str, str]:
    """Text-only format safe for GPTNeoXTokenizer-based Mamba models.

    Training example ends with an explicit eos token so the model learns to
    stop generating after its response. Inference prompt matches llama.cpp's
    "intel" chat format.  Long inputs and responses are cropped to fit the
    small context window used for K1 training.
    """
    turns = _last_pair(turns)
    if not turns:
        return "", "", ""

    user_turns = [t for t in turns if t[0] == "user"]
    assistant_turns = [t for t in turns if t[0] == "assistant"]
    user_content = _crop(user_turns[-1][1], max_user_chars) if user_turns else ""
    response = _crop(assistant_turns[-1][1], max_response_chars) if assistant_turns else ""

    out_lines = []
    if system:
        out_lines.append(f"### System:\n{system}")
    if user_content:
        out_lines.append(f"### User:\n{user_content}")
    if response:
        out_lines.append(f"### Assistant:\n{response}")

    full = "\n".join(out_lines) + eos if response else "\n".join(out_lines)

    prompt_lines = []
    if system:
        prompt_lines.append(f"### System:\n{system}")
    if user_content:
        prompt_lines.append(f"### User:\n{user_content}")
    prompt_lines.append("### Assistant:")
    prompt = "\n".join(prompt_lines)
    return full, prompt, response


def make_example(
    text: str,
    system: str = "",
    fmt: str = "intel",
    eos: str = DEFAULT_EOS,
    single_turn: bool = True,
    max_user_chars: int = DEFAULT_MAX_USER_CHARS,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
) -> dict:
    turns = raw_to_turns(text)
    if not turns:
        return {}
    if fmt == "llama2":
        full, prompt, response = format_llama2(system, turns, eos)
    elif fmt == "chatml":
        full, prompt, response = format_chatml(system, turns, eos)
    else:
        full, prompt, response = format_intel(
            system, turns, eos, max_user_chars, max_response_chars
        )
    if not response:
        return {}
    return {
        "text": full,
        "prompt": prompt,
        "response": response,
    }


def process_file(
    input_path: str,
    output_path: str,
    system: str = "",
    fmt: str = "intel",
    eos: str = DEFAULT_EOS,
    max_lines: int = 0,
    single_turn: bool = True,
    max_user_chars: int = DEFAULT_MAX_USER_CHARS,
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS,
):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    count = 0
    seen = set()
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
            example = make_example(
                text,
                system=system,
                fmt=fmt,
                eos=eos,
                single_turn=single_turn,
                max_user_chars=max_user_chars,
                max_response_chars=max_response_chars,
            )
            if not example:
                continue
            # Deduplicate on prompt to avoid over-representation.
            key = example["prompt"]
            if key in seen:
                continue
            seen.add(key)
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
    parser.add_argument("--single-turn", action="store_true", default=True, help="Keep only the last user/assistant pair")
    parser.add_argument("--multi-turn", dest="single_turn", action="store_false", help="Keep full multi-turn history")
    parser.add_argument("--max-user-chars", type=int, default=DEFAULT_MAX_USER_CHARS, help="Crop user turn to this many chars")
    parser.add_argument("--max-response-chars", type=int, default=DEFAULT_MAX_RESPONSE_CHARS, help="Crop assistant response to this many chars")
    args = parser.parse_args()
    process_file(
        args.input,
        args.output,
        system=args.system,
        fmt=args.format,
        eos=args.eos,
        max_lines=args.max_lines,
        single_turn=args.single_turn,
        max_user_chars=args.max_user_chars,
        max_response_chars=args.max_response_chars,
    )


if __name__ == "__main__":
    main()
