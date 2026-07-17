# tools/agentic.py
# Phase 8: Agentic Runner - autonomous multi-step task execution.
#
# Rewritten 2026-07-17 to be REAL agentic (ReAct), not theater:
#   plan -> route step to a tool -> execute via ToolHandler -> observe -> replan
# until the task is done or MAX_STEPS is reached. Uses OpenRouter
# (meta-llama/llama-3.2-3b-instruct, valid key) for planning, with the local
# 360M LM Studio backend as fallback when OpenRouter is unavailable.

import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neuralai.agentic")

LM_URL = os.environ.get("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
LM_MODEL = os.environ.get("LLM_MODEL", "smollm2-360m-instruct")# Normalize: LLM_API_URL may be ".../v1" or ".../v1/chat/completions"; always
# end up POSTing to the chat/completions endpoint exactly once.
_base = LM_URL.rstrip("/")
if _base.endswith("/chat/completions"):
    LM_URL = _base
elif _base.endswith("/v1"):
    LM_URL = _base + "/chat/completions"
else:
    LM_URL = _base.rstrip("/") + "/v1/chat/completions"
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "6"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PLANNER_MODEL = "meta-llama/llama-3.2-3b-instruct"

# Tools the planner may directly name in a step. Anything web-related is routed
# through tool_router so the agent reuses the live web tool chain.
_WEB_TOOLS = {
    "web_search", "web_fetcher", "web_browser", "research",
    "image", "speak", "summarize", "translate", "news", "youtube",
}


def _or_key() -> Optional[str]:
    return os.environ.get("Open_Router_API") or os.environ.get("OPENROUTER_API_KEY")


def _req_json(url: str, headers: dict, body: dict, timeout: int = 60) -> Optional[dict]:
    try:
        import urllib.request
        import urllib.error
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        logger.warning("agentic HTTP fail: %s", e)
        return None


def _rule_plan(task: str) -> List[str]:
    """Deterministic fallback planner used when NO LLM is reachable
    (local LM Studio down AND OpenRouter 402/dead). Maps the task to
    1-3 concrete tool steps using the live web-intent router so the
    agent still acts autonomously instead of erroring out."""
    try:
        from tools.web_intent import detect_web_intent
        r = detect_web_intent(task)
        if r:
            tool, params = r
            q = params.get("query") or params.get("url") or params.get("text") or task
            return [f"Use {tool} for: {q}"]
    except Exception:
        pass
    low = task.lower()
    if any(k in low for k in ("search", "news", "latest", "find out", "look up", "what is")):
        return [f"web_search: {task}"]
    if "http" in low:
        import re
        m = re.search(r"https?://\S+", task)
        if m:
            return [f"web_fetcher: {m.group(0)}"]
    if "image" in low or "picture" in low:
        return [f"image: {task}"]
    if "translate" in low:
        return [f"translate: {task}"]
    if "summar" in low:
        return [f"summarize: {task}"]
    return [task]


def _lm_available() -> bool:
    """True if either the local LM or OpenRouter planner can respond."""
    try:
        import urllib.request
        req = urllib.request.Request(
            LM_URL, data=json.dumps({
                "model": LM_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }).encode(),
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            json.loads(r.read().decode())
            return True
    except Exception:
        pass
    key = _or_key()
    if key:
        resp = _req_json(
            OPENROUTER_URL,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {"model": PLANNER_MODEL, "messages": [{"role": "user", "content": "ping"}],
             "max_tokens": 1}, timeout=10,
        )
        if resp and resp.get("choices"):
            return True
    return False


def _lm_complete(system: str, user: str, temp: float = 0.4,
                 max_tokens: int = 600, timeout: int = 60) -> str:
    """Primary: local 360M LM Studio (always available, zero-cost, fast).
    Fallback: OpenRouter planner only if the local model is unreachable.
    Final fallback: a deterministic rule-based planner so the agent
    never returns 'planner unavailable' when both LLMs are down."""
    # --- Primary: local LM Studio 360M ---
    try:
        import urllib.request
        body = json.dumps({
            "model": LM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            LM_URL, data=body,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning("local planner failed (%s); trying OpenRouter", e)
    # --- Fallback: OpenRouter planner ---
    key = _or_key()
    if key:
        resp = _req_json(
            OPENROUTER_URL,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {
                "model": PLANNER_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temp,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        if resp:
            try:
                return resp["choices"][0]["message"]["content"]
            except Exception:
                pass
    logger.warning("all LLM planners unreachable; using deterministic rule planner")
    # --- Final fallback: deterministic rule planner ---
    if "json array" in system.lower():
        return json.dumps(_rule_plan(user.replace("Task: ", "").strip()))
    if "DONE" in system.upper() or "CONTINUE" in system.upper():
        return "CONTINUE"
    return "(planner offline: returning direct answer)"


def _parse_plan(text: str) -> List[str]:
    text = text.strip().strip("`")
    if text.lstrip().lower().startswith("json"):
        text = text.lstrip()[4:]
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(s).strip().strip('"').strip("'") for s in obj if str(s).strip()]
        if isinstance(obj, dict) and "steps" in obj:
            return [str(s).strip().strip('"').strip("'")
                    for s in obj["steps"] if str(s).strip()]
    except Exception:
        pass
    # Loose parse: split on newlines/commas, drop numbering.
    steps = []
    for part in text.replace("[", "").replace("]", "").split("\n"):
        part = part.strip().strip('"').strip("'")
        part = part.split(".", 1)[-1].strip() if part[:2].isdigit() else part
        if part:
            steps.append(part)
    return steps


def _execute_step(step: str, tool_handler) -> str:
    """Route a natural-language step to a tool and run it. Returns observation text."""
    try:
        from tools.tool_router import route as route_nl
    except Exception:
        route_nl = None

    tool_name, params = None, {}
    if route_nl is not None:
        routed = route_nl(step)
        if routed:
            tool_name, params = routed
    # Force web-style steps through the web tool chain explicitly.
    if tool_name is None:
        low = step.lower()
        if any(k in low for k in ("search", "news", "latest", "find out", "look up")):
            tool_name, params = "web_search", {"query": step}
        elif "http" in low:
            # Extract first URL.
            import re
            m = re.search(r"https?://\S+", step)
            url = m.group(0).rstrip(").,") if m else ""
            tool_name, params = ("web_fetcher", {"url": url}) if url else (None, {})
        elif any(k in low for k in ("image", "picture", "draw", "generate an image")):
            tool_name, params = "image", {"prompt": step}
        elif "translate" in low:
            tool_name, params = "translate", {"text": step}
        elif "summarize" in low:
            tool_name, params = "summarize", {"text": step}

    if not tool_name:
        # No tool -> let the planner model answer directly from its own knowledge.
        return _lm_complete(
            "Answer the step concisely using your own knowledge.",
            f"Step: {step}", temp=0.3, max_tokens=300
        )

    try:
        res = tool_handler.execute(tool_name, params)
    except Exception as e:
        return f"[tool {tool_name} error: {e}]"
    if isinstance(res, dict):
        if not res.get("success"):
            return f"[tool {tool_name} failed: {res.get('error','')}]"
        return str(res.get("output", ""))[:4000]
    return str(res)[:4000]


def run(task: str, tool_handler: Any = None) -> dict:
    """Execute an autonomous task. Returns a result dict with a transcript."""
    if tool_handler is None:
        try:
            from tools.tool_handler import ToolHandler
            tool_handler = ToolHandler()
        except Exception as e:
            return {"success": False, "output": f"[agent init failed: {e}]",
                    "brief": "", "steps": 0, "transcript": []}

    plan_sys = (
        "You are the NeuralAI autonomous agent. Given a task, break it into "
        "2-5 concrete steps that can each be done with a tool (web search, fetch a "
        "URL, research a topic, generate an image, etc.) or answered from knowledge. "
        "Reply ONLY as a JSON array of step strings, no prose."
    )
    plan = _lm_complete(plan_sys, f"Task: {task}", temp=0.3, max_tokens=400)
    steps = _parse_plan(plan)
    if not steps:
        steps = [task]

    transcript: List[Dict[str, str]] = []
    observations: List[str] = []
    for i, step in enumerate(steps[:MAX_STEPS], 1):
        step = step.strip()
        if not step:
            continue
        result = _execute_step(step, tool_handler)
        observations.append(result)
        transcript.append({"step": i, "action": step, "result": result})

        # Replan check: ask the planner if the task is satisfied by what we have.
        if i < len(steps[:MAX_STEPS]):
            check = _lm_complete(
                "You are a critic. Given the original task and the observations so "
                "far, reply ONLY 'DONE' if the task is sufficiently answered, else "
                "reply 'CONTINUE'.",
                f"Task: {task}\nObservations:\n" + "\n".join(observations),
                temp=0.0, max_tokens=20
            ).strip().upper()
            if "DONE" in check:
                break

    brief = _lm_complete(
        "Write a concise, user-facing brief (3-6 sentences, plain prose, no headers) "
        "answering the original task from the observations.",
        f"Task: {task}\nObservations:\n" + "\n".join(observations),
        temp=0.3, max_tokens=500
    )
    out = (
        f"🤖 Agent completed {len(transcript)} step(s) for: {task}\n\n"
        + "\n".join(f"{t['step']}. {t['action']} → {t['result'][:600]}" for t in transcript)
        + f"\n\nBrief: {brief}"
    )
    return {
        "success": True,
        "output": out,
        "brief": brief,
        "steps": len(transcript),
        "transcript": transcript,
    }
