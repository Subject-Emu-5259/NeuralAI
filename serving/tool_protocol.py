# website/tool_protocol.py
"""Plain-text tool protocol: pure functions, no I/O. Shared by server, eval, generator."""
from __future__ import annotations
import re

MAX_TOOL_TOKENS = 250
MAX_ROUNDS = 3
N_CTX = 2048
CTX_RESERVE = 256
ELLIPSIS = " […]"
TOOLS = ("search", "fetch", "read_file")

_CALL_RE = re.compile(
    r'^\s*»\s*(?P<name>[a-z_]+)\(\s*(?P<arg>"(?:[^"\\]|\\.)*")\s*\)\s*$', re.M)
_BLOCK_RE = re.compile(r'<<tool name="(?P<name>[^"]+)" query="(?P<q>[^"]*)">>.*?<</tool>>', re.S)


def count_tokens(tok, text: str) -> int:
    return len(tok.encode(text).ids)


def truncate_to_budget(tok, text: str, budget: int = MAX_TOOL_TOKENS) -> str:
    """Hard ceiling via the real BPE; ellipsis always inside the budget."""
    ids = tok.encode(text).ids
    if len(ids) <= budget:
        return text
    keep = budget - count_tokens(tok, ELLIPSIS)
    return tok.decode(ids[:keep]) + ELLIPSIS


def tool_block(name: str, arg: str, body: str) -> str:
    return f'<<tool name="{name}" query="{arg}">>{body}<</tool>>'


def error_block(name: str, arg: str, code: str, msg: str) -> str:
    return f'<<tool name="{name}" query="{arg}">><error>{code}: {msg}</error><</tool>>'


def find_complete_call(text: str, *, eof: bool) -> tuple[str, str] | None:
    """Return (name, arg) only for a COMPLETE single-line call.
    Without eof, the line must be newline-terminated (mid-stream safe)."""
    if not eof:
        nl = text.rfind("\n")
        if nl < 0:
            return None
        window = text[: nl + 1]
    else:
        window = text
    m = _CALL_RE.search(window)
    if not m:
        return None
    name = m.group("name")
    if name not in TOOLS:
        return None
    return name, m.group("arg")[1:-1]


def build_prompt(history: list[dict], user_msg: str) -> str:
    """THE chat template. Must stay byte-identical to SFT training traces."""
    parts = []
    for msg in history:
        who = "User" if msg["role"] == "user" else "NeuralAI"
        parts.append(f"{who}: {msg['text']}\n")
    parts.append(f"User: {user_msg}\nNeuralAI: ")
    return "".join(parts)


def _collapse_one(text: str) -> str:
    def rep(m: re.Match) -> str:
        return f'<<tool {m.group("name")}("{m.group("q")}") [result summarized]>>'
    return _BLOCK_RE.sub(rep, text)


def gc_transcript(messages: list[dict], tok, budget: int) -> list[dict]:
    """Oldest-first block collapse until build_prompt fits budget. Pure/deterministic.
    Never touches user turns, call lines, or answers (only in-place block collapse)."""
    out = [dict(m) for m in messages]
    i = 0
    while i < len(out) and count_tokens(tok, _render(out)) > budget:
        m = out[i]
        if m["role"] == "assistant" and "<<tool " in m["text"]:
            m["text"] = _collapse_one(m["text"])
        i += 1
    # second pass: keep collapsing until it fits (oldest first, repeatedly)
    while count_tokens(tok, _render(out)) > budget:
        progressed = False
        for m in out:
            if m["role"] == "assistant" and "<</tool>>" in m["text"]:
                before = m["text"]
                m["text"] = _collapse_one(m["text"])
                if m["text"] != before:
                    progressed = True
                    break
        if not progressed:
            break
    return out


def _render(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        who = "User" if msg["role"] == "user" else "NeuralAI"
        parts.append(f"{who}: {msg['text']}\n")
    parts.append("User: <pending>\nNeuralAI: ")
    return "".join(parts)


def visible_text(model_text: str) -> str:
    """What the client may see: blocks stripped, call lines rendered as 🔧."""
    out = _BLOCK_RE.sub("", model_text)
    out = out.replace("»", "🔧")
    return out
