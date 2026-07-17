"""Detect a natural-language web/tool intent from a plain-English prompt.

The local 360M model can't self-route to tools, so we intercept obvious
web/search/research/fetch requests in the API layer and dispatch them to the
real tool chain (the same one the /web, /fetch, /research slash commands use).

Returns (tool_name, params) or None if no intent is detected.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# (compiled_regex, tool_name, param_builder)
_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # Fetch / read a specific URL  (keep near top so a bare URL becomes fetch)
    (re.compile(r"\b(fetch|open|read|load|visit|get)\b.*(https?://\S+)", re.I), "web_fetcher", "url"),
    # Browse / explore / navigate a SITE  (sits ABOVE the bare-URL -> web_fetcher
    # catch-all so "browse https://x" routes to the real browser tool, not a single fetch)
    (re.compile(r"\b(browse|explore|navigate)\b.*?(https?://\S+)", re.I), "web_browser", "url"),
    (re.compile(r"(https?://\S+)"), "web_fetcher", "url"),
    (re.compile(r"\b(research|deep dive|investigate|look into)\b[:\s]*(.+)", re.I), "research", "query"),
    # News  — capture the full remaining phrase (e.g. "latest news on AI" -> "latest news on AI")
    (re.compile(r"\b(news|headlines|latest (?:news|headline)s?)\b[:\s]*(.*)", re.I), "news", "query"),
    # Translate
    (re.compile(r"\b(translate|translation)\b[:\s]*(.+)", re.I), "translate", "query"),
    # Generic web search  — also catches "latest X", "current X", "recent X", "today X"
    (re.compile(
        r"\b(search|google|look up|find (?:out )?about|what'?s (?:the )?(?:latest|newest)|"
        r"latest|current|recent|today'?s?)\b[:\s]*(.*)", re.I), "web_search", "query"),
]


def _clean(q: str) -> str:
    # strip leading verbs / filler the tool doesn't need, including the literal
    # "the web for" / "the web" phrasing that leaks from "search the web for X"
    # NOTE: do NOT strip 'news'/'latest'/'newest'/'current'/'recent'/'today' here —
    # those are real topics, not just command verbs, and stripping them mangles
    # queries like "latest news on AI" -> "on AI". The trigger-guard above already
    # ensures we only reach this code on a genuine web intent.
    q = re.sub(r"^(search|google|look up|browse|find out about|research|investigate|"
               r"look into|summarize|summarise|translate|headlines|fetch|open|read|load|"
               r"visit|get|what'?s|what is|tell me about)\b[:\s]*",
               "", q, flags=re.I)
    q = re.sub(r"^the web for\s*", "", q, flags=re.I)
    q = re.sub(r"^the web\s*", "", q, flags=re.I)
    # strip a leftover leading determiner that would poison the query
    # (e.g. "the latest news" -> "latest news", not "the latest news")
    q = re.sub(r"^(the|an?|some|any)\s+", "", q, flags=re.I)
    return q.strip().strip('"').strip("'")


def detect_web_intent(prompt: str) -> Optional[Tuple[str, dict]]:
    if not prompt or prompt.strip().startswith("/"):
        return None
    p = prompt.strip()

    # quick guard: only treat as web intent if there's a clear trigger word/url
    if not (re.search(r"https?://|search|google|look up|browse|research|news|headline|"
                      r"summariz|translate|fetch|latest|recent|current|today|web|internet|online|"
                      r"find out about|what'?s",
                      p, re.I)):
        return None

    for rx, tool, mode in _RULES:
        m = rx.search(p)
        if not m:
            continue
        if mode == "url":
            um = re.search(r"https?://\S+", p)
            if um:
                return (tool, {"url": um.group(0).rstrip(").,!?")})
        else:
            if tool == "news":
                # News: keep the whole phrase (e.g. "latest news on AI") so the
                # topic isn't truncated by the regex capture group.
                q = _clean(p)
            else:
                raw = m.group(m.lastindex) if m.lastindex else p
                q = _clean(raw)
                if not q:
                    q = _clean(p)
            if q:
                return (tool, {"query": q, "top_k": 5})
    return None
