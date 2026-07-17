"""Summarize fetched web sources into a single brief for the /research chain.

Keeps the logic local and dependency-free so it runs inside the live
neuralai-web-ui service without extra packages. If a real LLM backend is
available (llmster/LM Studio on :1234) it can be swapped in later; for now
we extract the most query-relevant sentences across all sources.
"""
from typing import Dict, Any, List
import re


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _split_sentences(text: str) -> List[str]:
    # Keep common abbreviations from splitting; lightweight but sufficient.
    raw = re.split(r"(?<=[.!?])\s+", text or "")
    out = []
    for s in raw:
        s = s.strip()
        if len(s) > 25:  # drop fragments/headers
            out.append(s)
    return out


def _score(sentence: str, query: str) -> float:
    q_terms = [t for t in re.findall(r"[a-zA-Z]{3,}", query.lower())]
    if not q_terms:
        return 1.0
    s_low = sentence.lower()
    return sum(2 if t in s_low else 0 for t in q_terms) + (0.1 if any(t in s_low for t in q_terms) else 0)


def summarize_sources(sources: List[Dict[str, Any]], query: str = "", max_sentences: int = 4) -> str:
    """Build a brief from a list of {title, url, text} dicts."""
    if not sources:
        return "⚠️ No sources to summarize."

    query = query or ""
    scored: List[tuple] = []
    for src in sources:
        text = _clean(src.get("text", ""))
        if not text:
            continue
        for sent in _split_sentences(text)[:40]:  # cap per-source scan
            scored.append((_score(sent, query), sent, src.get("url", "")))

    if not scored:
        return "⚠️ Sources had no summarizable sentences."

    scored.sort(key=lambda x: x[0], reverse=True)
    # de-dup while preserving order of score
    seen = set()
    picks: List[tuple] = []
    for score, sent, url in scored:
        key = sent[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        picks.append((sent, url))
        if len(picks) >= max_sentences:
            break

    lines = [f"📚 Research brief: {query}\n"]
    for i, (sent, url) in enumerate(picks, 1):
        lines.append(f"{i}. {sent}")
    lines.append("")
    lines.append("Sources:")
    for src in sources:
        lines.append(f"• {src.get('title', src.get('url', ''))} — {src.get('url', '')}")
    return "\n".join(lines)
