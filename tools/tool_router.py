"""
Agentic NL->tool router for NeuralAI.

Replaces the brittle keyword/regex router (web_intent.detect_web_intent) with a
small LLM that decides tool use from plain English. This is what lets the model
"use the tools on its own" instead of requiring the user to remember slash syntax.

Design:
- Primary path: OpenRouter meta-llama/llama-3.2-3b-instruct returns strict JSON
  {tool, params}. Model is cheap + fast; this is a routing decision, not generation.
- Fallback: if the LLM call fails for ANY reason (missing key, timeout, bad JSON,
  402 billing), we fall back to detect_web_intent so behavior degrades to the old
  keyword router rather than passing the prompt to the 360M model and hallucinating.

Composite-intent fast path: a plain-English prompt that clearly needs 2+ tools
(e.g. "research X then generate an image", "find news and summarize it") is routed
to the autonomous agent instead of a single tool. This is the core "agentic" win.

The return shape deliberately matches detect_web_intent: (tool_name, params_dict)
or None. params_dict always carries the natural-language query under "query" so the
tool handler / refine step has something to render.
"""

import os
import json
import logging
from typing import Optional, Tuple, Dict, Any

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from tools.web_intent import detect_web_intent, _parse_translate

logger = logging.getLogger("neuralai.tool_router")

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "nousresearch/hermes-3-llama-3.1-8b",
    "google/gemma-2-9b-it",
]
_TIMEOUT = 8  # seconds; routing must be fast, never block the chat

# Tools the router may pick. Keep in sync with ToolHandler handlers + main.js.
_TOOLS = {
    "web_search": "Search the web for a query and return top results.",
    "web_fetcher": "Fetch the text/markdown content of a single URL. params needs 'url'.",
    "web_browser": "Drive a browser session against a URL (multi-step). params needs 'url'.",
    "research": "Deep research: search -> fetch -> summarize a topic. params needs 'query'.",
    "image": "Generate an image from a prompt. params needs 'prompt'.",
    "speak": "Text-to-speech. params needs 'text'.",
    "summarize": "Summarize a URL or pasted text. params needs 'url' or 'text'.",
    "translate": "Translate text to a target language. params needs 'lang' and 'text'.",
    "news": "Search latest news for a topic. params needs 'query'.",
    "youtube": "Get YouTube video metadata/summary for a URL. params needs 'url'.",
    "agent": "Autonomous multi-step task execution (plan -> act across tools -> reflect). Use for composite requests like 'research X and generate an image', 'compare these 3 sites', or 'build a summary from several URLs'. params needs 'task'.",
}

_SYSTEM = (
    "You are a tool router for an AI assistant. Given the user's message, decide if it "
    "should trigger one of the available tools. If yes, reply with ONLY a JSON object of "
    "the form {\"tool\": <name>, \"params\": <object>}. If the message is a normal chat "
    "question that does NOT need a tool, reply with exactly {\"tool\": null}.\n\n"
    "Available tools:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in _TOOLS.items())
    + "\n\nRules:\n"
    "- For search/news/research/summarize-via-query, put the user's topic in params.query.\n"
    "- For web_fetcher/web_browser/youtube, put the URL in params.url.\n"
    "- For image, put the description in params.prompt.\n"
                "- For speak, put the text in params.text.\n"
            "- For translate, put the text in params.text and the TARGET language in params.lang (use a 2-letter ISO code like 'es', 'fr', 'de', or the language name). Do NOT put the whole sentence in one field.\n"
    "- For summarize with pasted text, put it in params.text.\n"
    "- For agent, put the full task in params.task.\n"
    "- Never invent a URL. If a URL is required but none is present, pick web_search instead.\n"
    "- General knowledge questions, math, coding help, or conversation = {\"tool\": null}."
)


def _call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """Call OpenRouter and return parsed JSON, or None on any failure."""
    if requests is None:
        return None
    api_key = os.environ.get("Open_Router_API") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        for model in _MODELS:
            resp = requests.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 200,
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            content = (
                data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            )
            # Strip code fences if the model added them.
            if content.startswith("```"):
                content = content.strip("`")
                if content.lstrip().lower().startswith("json"):
                    content = content.lstrip()[4:]
            return json.loads(content)
        return None
    except Exception as e:  # noqa: BLE001 - fallback is the point
        logger.warning("tool_router LLM failed: %s", e)
        return None


def _composite_detected(prompt: str) -> bool:
    """
    Cheap offline check: does this plain-English prompt clearly need 2+ tools?
    Only fires on explicit multi-action connectors OR 2+ distinct tool keywords,
    so a single-intent step like 'research X' does NOT recurse into the agent.
    """
    p_low = prompt.lower()
    has_connector = any(
        c in p_low
        for c in (
            " then ", " and then ", " and also ", " after that ",
            " followed by ", " plus ", " + ", " and generate ",
            " and make ", " and create ", " and fetch ", " and summarize ",
            " and translate ", " and draw ", " and research ",
        )
    )
    # Strong multi-action phrases that don't need a literal connector.
    strong_phrases = (
        "research and", "search and", "find and", "summarize and",
        "compare", "build a summary", "make an image of",
    )
    if has_connector or any(ph in p_low for ph in strong_phrases):
        keywords = (
            "search", "research", "find out", "look up", "news", "latest",
            "fetch", "summarize", "translate", "image", "picture", "draw",
            "generate an image", "youtube", "video", "speak", "tts",
        )
        distinct = sum(1 for kw in keywords if kw in p_low)
        # Require at least 2 distinct tool signals OR a strong multi-action phrase.
        if distinct >= 2 or any(ph in p_low for ph in strong_phrases):
            return True
    return False


def route(prompt: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Decide tool use for a plain-English prompt.

    Returns (tool_name, params) like detect_web_intent, or None to pass to the model.
    Always falls back to the keyword router on any LLM failure.
    """
    if not prompt or not prompt.strip():
        return None

    # Translate is a structured 2-slot intent (text + target lang). The small
    # routing LLM reliably detects the tool but routinely drops/mis-shapes the
    # target language, so we route it through the deterministic keyword parser
    # (which correctly splits 'translate <text> to <lang>') and skip the LLM.
    if re.search(r"\btranslate\b", prompt, re.I):
        tr = _parse_translate(prompt)
        if tr:
            return tr

    # Composite-intent fast path: route to the autonomous agent (offline, no LLM call).
    if _composite_detected(prompt):
        return ("agent", {"task": prompt})

    decision = _call_llm(prompt)
    if decision is not None:
        tool = decision.get("tool")
        if not tool:
            # Model says no tool needed. Still let keyword router have a look in
            # case the user typed a raw URL or explicit slash-style intent.
            return detect_web_intent(prompt)
        if tool == "agent":
            params = decision.get("params", {}) or {}
            if "task" not in params:
                params["task"] = prompt
            return (tool, params)
        if tool not in _TOOLS:
            return detect_web_intent(prompt)
        # Translate needs precise lang+text extraction that the LLM often gets
        # wrong (e.g. "translate hello to spanish" -> text="to spanish").
        # Re-parse deterministically so the right language is always chosen.
        if tool == "translate":
            from tools.web_intent import _parse_translate
            tr = _parse_translate(prompt)
            if tr:
                return tr
        params = decision.get("params", {}) or {}
        # Normalize: ensure there is always a query/text to render from.
        if "query" not in params and "text" not in params and "url" not in params and "prompt" not in params:
            params["query"] = prompt
        # Recency hint: news/web_search queries get fresher results.
        if tool in ("news", "web_search") and "query" in params:
            q = params["query"].strip()
            if "today" not in q.lower() and "latest" not in q.lower():
                params["query"] = f"{q} latest today"
        return (tool, params)

    # LLM path failed -> keyword fallback (never hallucinate).
    return detect_web_intent(prompt)
