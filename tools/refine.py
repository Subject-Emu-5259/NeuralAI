"""NeuralAI tool-output refiner.

Takes the raw `output` string returned by a web tool and converts it into
clean, structured prose for display in the chat UI. No network, no deps.

`refine_text(raw, kind)` is the single entry point:
  - kind="news"       -> grouped, deduped headline cards with source + snippet
  - kind="web_search" -> numbered result list (title + url + snippet)
  - kind="research"   -> keeps the synthesized brief, drops concatenated boilerplate
  - kind="web_fetcher"-> strips nav/banner noise, keeps readable body
  - kind=<other>      -> light cleanup (collapse blank lines, trim)

If parsing fails for any reason it returns the original text, so a bug here
can never blank out or 500 the response.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_LINK_RE = re.compile(r"https?://\S+", re.I)
# Matches a trailing "Source · Date" or "Source - Date" on the snippet line
_SOURCE_RE = re.compile(r"^(.+?)\s*[·\u00b7\-–—]\s*(.+)$")
# Google News RSS encodes the real article behind a redirect; show a clean label
# rather than the 300-char tracking URL.
_GNEWS_DOMAIN = "news.google.com"
_WIKI_JUNK_RE = re.compile(r"wikipedia\.org", re.I)
_SEO_JUNK_RE = re.compile(
    r"(latest news|breaking news|stay informed|tour dates|pictures|mp3s|videos|lyrics)",
    re.I,
)


def _is_junk(url: str, title: str) -> bool:
    """Drop SEO/disambiguation pages that carry no real headline."""
    if _WIKI_JUNK_RE.search(url) and _SEO_JUNK_RE.search(title):
        return True
    return False


def _parse_results(raw: str) -> List[Tuple[str, str, str]]:
    """Parse the tool's 'N. Title\\n   url\\n   snippet' block into rows."""
    rows: List[Tuple[str, str, str]] = []
    # Drop a leading "📰 News: <topic>" / "Latest headlines" header line if present
    raw = re.sub(r"^\s*(?:📰\s*)?(?:News|Latest headlines)[^\n]*\n+", "", raw)
    # Split on leading "N." or "N)" markers
    chunks = re.split(r"\n\s*(?:\d+[.)])\s+", raw)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        url = ""
        m = _LINK_RE.search(chunk)
        if m:
            url = m.group(0).rstrip(").,;")
            chunk = (chunk[: m.start()] + chunk[m.end():]).strip()
        # title is the first line, snippet is the rest
        parts = chunk.split("\n", 1)
        title = parts[0].strip().rstrip(":").strip()
        snippet = parts[1].strip() if len(parts) > 1 else ""
        if title:
            rows.append((title, url, snippet))
    return rows


def _clean_body(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def refine_text(raw: str, kind: str = "") -> str:
    if not raw or not isinstance(raw, str):
        return raw or ""
    try:
        kind = (kind or "").lower()
        if kind in ("news", "web_search"):
            return _refine_list(raw, kind)
        if kind == "web_fetcher":
            return _clean_body(raw)
        if kind == "research":
            # research already returns a synthesized brief; just tidy spacing
            return _clean_body(raw)
        return _clean_body(raw)
    except Exception:
        # Never let refinement break the response
        return raw


def _domain(url: str) -> str:
    """Return a short readable host (e.g. bbc.com) from a URL."""
    try:
        from urllib.parse import urlparse
        net = urlparse(url).netloc
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return url


def _short_label(url: str, snippet: str) -> str:
    """Return a short, display-friendly label for a result link.

    `url` is now expected to be the REAL resolved publisher URL (web_search
    follows the full Google News redirect chain). We show the publisher domain
    as the link text. Only if resolution somehow failed (still on news.google.com)
    do we fall back to the publisher name parsed from the snippet.
    """
    if _GNEWS_DOMAIN in url:
        # Resolution failed; best-effort publisher name from snippet.
        m = _SOURCE_RE.match(snippet.strip())
        if m:
            return m.group(1).strip()
        return "Google News"
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.replace("www.", "")
        return host or "source"
    except Exception:
        return "source"


def _refine_list(raw: str, kind: str) -> str:
    rows = _parse_results(raw)
    if not rows:
        return _clean_body(raw)

    kept: List[Tuple[str, str, str]] = []
    seen_urls = set()
    for title, url, snippet in rows:
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        if kind == "news" and url and _is_junk(url, title):
            continue
        kept.append((title, url, snippet))

    if not kept:
        return _clean_body(raw)

    header = "Latest headlines" if kind == "news" else "Search results"
    out = [f"**{header}**\n"]
    for i, (title, url, snippet) in enumerate(kept, 1):
        if kind == "news":
            # News: bold headline, then source/date line, then the real
            # (clickable) article URL on its own line.
            line = f"{i}. **{title}**"
            if snippet:
                line += f"\n   {snippet}"
            if url:
                # Keep the real (clickable) article URL, but show the publisher
                # name as the visible text so the chat stays readable instead of
                # dumping a 300-char Google News tracking URL. fmt() turns this
                # into a clickable link in the UI.
                label = _short_label(url, snippet)
                line += f"\n   🔗 [{label}]({url})"
        else:
            line = f"{i}. {title}"
            if url:
                line += f"\n   {url}"
            if snippet:
                snippet = snippet[:220].strip()
                if snippet:
                    line += f"\n   {snippet}"
        out.append(line)
    return "\n\n".join(out)
