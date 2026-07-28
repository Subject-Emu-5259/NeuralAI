"""NeuralAI web search toolkit.

Primary backend: DuckDuckGo Lite (works from sandboxed environments where the
HTML endpoint and Google SERP scraping are blocked). Fallbacks: Bing generic
anchor parse, then Wikipedia opensearch for factual lookups.
News queries use Google News RSS, which is timestamped and bot-friendly.

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
from xml.etree import ElementTree

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def _resolve_real_url(google_news_url: str, timeout: int = 10) -> str:
    """Resolve a Google News RSS link to the real publisher article URL.

    Google News RSS links are tracking blobs of two shapes:
      1. news.google.com/rss/articles/CBMi...?oc=5   (RSS entry link)
      2. news.google.com/articles/CBMi...        (after a 301 from #1)
    Both ultimately point at an HTML interstitial page whose body contains
    the real destination in a `URL=` query param (the `?` is %3F-encoded).
    We fetch the interstitial directly (no redirect-following) and extract
    that param, then un-encode it. Falls back to the cleaned original.
    Also unwraps DuckDuckGo `uddg=` redirects.
    """
    import urllib.parse as _up
    if not google_news_url:
        return "https://news.google.com"
    try:
        # Unwrap DuckDuckGo uddg= redirects first.
        ddg = _up.unquote(google_news_url)
        m = re.search(r"uddg=([^&]+)", ddg)
        if m:
            return _up.unquote(m.group(1)).strip()

        # Normalize: if it is the /rss/ form, go straight to the /articles/ form.
        target = google_news_url
        if "news.google.com/rss/articles/" in target:
            target = target.replace("news.google.com/rss/articles/",
                                    "news.google.com/articles/", 1)

        req = urllib.request.Request(target, headers={"User-Agent": UA},
                                    method="GET")
        ctx = urllib.request.urlopen(req, timeout=timeout)
        if getattr(ctx, "code", 200) == 200:
            html = ctx.read(200_000).decode("utf-8", "ignore")
            ctx.close()
            real = _extract_gnews_interstitial(target, html)
            if real and "news.google.com" not in real:
                return real.strip()
        else:
            ctx.close()
        return target.split("?")[0].strip()
    except Exception:
        pass
    return google_news_url or "https://news.google.com"

def _extract_gnews_interstitial(url: str, html: str = "") -> str:
    """Pull the real URL= param out of a Google News interstitial page."""
    try:
        if not html:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read(200_000).decode("utf-8", "ignore")
        m = re.search(r"[?&]URL=([^&'\"\\)]+)", html)
        if not m:
            return ""
        target = m.group(1).replace("%3F", "?")  # un-encode trailing ?
        return urllib.parse.unquote(target).strip()
    except Exception:
        return ""


def _parse_source_and_date(title: str, pub_date: str):
    """Split 'Headline - Source' into (headline, source) and format a clean date."""
    source = ""
    head = title
    if " - " in title:
        head, source = title.rsplit(" - ", 1)
        head = head.strip()
        source = source.strip()
    clean_date = pub_date
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub_date)
        if dt:
            clean_date = dt.strftime("%b %d, %Y")
    except Exception:
        pass
    return head, source, clean_date


def _google_news(query: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Google News RSS — timestamped, no API key, bot-friendly."""
    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        xml = _get(url)
        root = ElementTree.fromstring(xml)
    except Exception:
        return []
    results: List[Dict[str, str]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title or not link:
            continue
        head, source, clean_date = _parse_source_and_date(title, pub)
        real_url = _resolve_real_url(link)
        # snippet carries source + date so the UI has real info, not just a title
        snippet = "".join(
            [f"{source} " if source else "", f"· {clean_date}" if clean_date else ""]
        ).strip()
        results.append({
            "title": head,
            "url": real_url,
            "snippet": snippet,
            "source": source,
            "date": clean_date,
        })
        if len(results) >= top_k:
            break
    return results


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


def search(query: str, top_k: int = 5) -> list:
    """Return a list of result dicts: {title, url, snippet, source, date}.

    News-style queries prefer Google News RSS for fresh, timestamped results.
    Handlers (tool_handler.py) own final display formatting.
    """
    q_lower = query.lower()
    is_news = "news" in q_lower or "headline" in q_lower or "latest" in q_lower
    if is_news:
        results = _google_news(query, top_k) or _ddg_lite(query, top_k) or _wikipedia(query, top_k)
    else:
        results = _ddg_lite(query, top_k) or _bing(query, top_k) or _wikipedia(query, top_k)
    if not results:
        return []
    return results


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
        q_lower = query.lower()
        is_news = "news" in q_lower or "headline" in q_lower or "latest" in q_lower
        if is_news:
            return _google_news(query, top_k) or _ddg_lite(query, top_k) or _wikipedia(query, top_k)
        return _ddg_lite(query, top_k) or _bing(query, top_k) or _wikipedia(query, top_k)
