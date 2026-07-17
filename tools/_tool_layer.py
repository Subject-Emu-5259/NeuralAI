"""NeuralAI extended tool layer.

Exposes:
  - _load_tool(mod, attr): dynamic import of a tools/<mod>.py member
  - _looks_like_search(text): heuristic for auto web-search
  - process_tool_tags(text): execute <tool>search: ...</tool> / <tool>fetch: url</tool>
    / <tool>browse: task</tool> tags and return appended results (markdown).

Search primary = Gemini grounding (Google Search); fallback = DuckDuckGo scrape.
Fetch = WebFetcher (readability-style text extract).
Browse = agentic Playwright session (click/fill/navigate) — requires Playwright.
"""

import re
import importlib.util as _ilu
import os as _os

_TOOLS_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _load_tool(mod: str, attr: str):
    """Import tools/<mod>.py and return its <attr> member."""
    spec = _ilu.spec_from_file_location(mod, _os.path.join(_TOOLS_DIR, f"{mod}.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, attr)


def _looks_like_search(text: str) -> bool:
    """Heuristic: does the user message look like a web search request?"""
    t = (text or "").lower().strip()
    if not t:
        return False
    triggers = (
        "search", "google", "look up", "find", "who is", "who won", "what is",
        "latest", "news", "current", "today", "weather", "price of", "how to",
        "wiki", "best", "top 10", "near me", "vs ", "compare",
    )
    if any(k in t for k in triggers):
        return True
    if t.endswith("?") and any(w in t for w in ("what", "who", "when", "where", "why", "which", "how")):
        return True
    return False


def process_tool_tags(text: str) -> str:
    """Execute <tool>search: q</tool>, <tool>fetch: url</tool>, <tool>browse: task</tool>."""
    try:
        parts = []
        for kind, args in re.findall(r"<tool>(search|fetch|browse):\s*(.*?)</tool>", text, re.DOTALL):
            args = args.strip()
            if kind == "search":
                web_search = _load_tool("web_search", "search")
                parts.append("[SEARCH]\n" + web_search(args, top_k=5))
            elif kind == "fetch":
                fetcher = _load_tool("web_fetcher", "WebFetcher")
                parts.append("[FETCH " + args + "]\n" + fetcher().fetch(args).get("text", "")[:4000])
            elif kind == "browse":
                get_session = _load_tool("web_browser", "get_session")
                parsed = _load_tool("web_search", "parse_browse_task")(args)
                url = parsed.get("url")
                steps = parsed.get("steps", [])
                if not url and steps:
                    s = _load_tool("web_search", "search")(parsed.get("query") or args, top_k=1)
                    url = _load_tool("web_search", "first_result_url")(s)
                if url:
                    parts.append("[BROWSE]\n" + get_session("default").run(url, steps))
        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        return f"[TOOL_ERROR] {e}"
