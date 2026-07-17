"""NeuralAI web search toolkit.

Primary backend: DuckDuckGo Lite (works from sandboxed environments where the
HTML endpoint and Google SERP scraping are blocked). Fallbacks: Bing generic
anchor parse, then Wikipedia opensearch for factual lookups.

Gemini grounding and OpenRouter (Perplexity) are left as OPTIONAL backends but
are disabled by default because the live service keys are currently invalid
(Gemini 401, OpenRouter 401 on completions). Re-enable by setting the relevant
env vars once valid keys are supplied.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def _ddg_lite(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
    html = _get(url)
    # DDG lite emits result rows: <a class="result-link" href="...">title</a>
    rows = re.findall(
        r'<a[^>]*class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.S,
    )
    results: List[Dict[str, str]] = []
    for href, title in rows[:top_k]:
        href = urllib.parse.unquote(href)
        # DDG wraps external links through redirect urls; extract the real one
        m = re.search(r"uddg=([^&]+)", href)
        real = urllib.parse.unquote(m.group(1)) if m else href
        results.append(
            {
                "title": re.sub(r"<[^>]+>", "", title).strip(),
                "url": real,
                "snippet": "",
            }
        )
    return results


def _bing(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(query)
    html = _get(url)
    # Bing serves a consent/JS shell to bots, but result anchors still appear.
    anchors = re.findall(
        r'<h2><a href="(https?://[^"]+)"[^>]*>(.*?)</a></h2>', html, re.S
    )
    results: List[Dict[str, str]] = []
    for href, title in anchors[:top_k]:
        results.append(
            {
                "title": re.sub(r"<[^>]+>", "", title).strip(),
                "url": href,
                "snippet": "",
            }
        )
    return results


def _wikipedia(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&list=search&"
        "format=json&srlimit=" + str(top_k) + "&srsearch=" + urllib.parse.quote(query)
    )
    data = json.loads(_get(url))
    results: List[Dict[str, str]] = []
    for item in data.get("query", {}).get("search", [])[:top_k]:
        title = item["title"]
        results.append(
            {
                "title": title,
                "url": "https://en.wikipedia.org/wiki/"
                + urllib.parse.quote(title.replace(" ", "_")),
                "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
            }
        )
    return results


def search(query: str, top_k: int = 5) -> str:
    """Return a formatted search result string."""
    results = _ddg_lite(query, top_k) or _bing(query, top_k) or _wikipedia(query, top_k)
    if not results:
        return f"❌ No results found for '{query}'."
    formatted = [f"🔎 Search: {query}\n"]
    for i, r in enumerate(results, 1):
        snippet = f" — {r['snippet']}" if r.get("snippet") else ""
        formatted.append(f"{i}. {r['title']}\n   {r['url']}{snippet}\n")
    return "\n".join(formatted)


def first_result_url(formatted: str) -> str:
    """Extract the first URL from a formatted search string."""
    m = re.search(r"https?://[^\s\n]+", formatted)
    return m.group(0) if m else ""


def parse_browse_task(task: str) -> Dict[str, Any]:
    """Parse a free-form browse task into structured steps (best effort)."""
    return {"task": task, "steps": [s.strip() for s in task.split(" then ") if s.strip()]}


class WebSearch:
    """Convenience wrapper returning structured result lists."""

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        return _ddg_lite(query, top_k) or _bing(query, top_k) or _wikipedia(query, top_k)
