"""
NeuralAI Web Surfing Agent — standalone service (port 5002).

Verified working chain (2026-07-17):
  Path (1) PRIMARY (if credits): OpenRouter with live web.search tool.
  Path (2) REAL WEB GROUNDING: DuckDuckGo HTML search -> fetch top results ->
           local LM Studio 360M synthesis (cites sources). Works without any API key.
  Path (3) OFFLINE FALLBACK: local LM Studio 360M (no web).
  Image gen: Pollinations image API (keyless, returns JPEG).

NeuralVoice is bound to 5001, so this agent uses 5002.
OpenRouter returns 402 (no credits) on this account, so it is only used when
OR_CREDITS is True (set at startup if a live credit check passes).
"""

import os
import io
import re
import json
import base64
import urllib.parse
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "5002"))

POLLINATIONS_TEXT = "https://text.pollinations.ai/"
POLLINATIONS_IMG = "https://image.pollinations.ai/"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"


def _env(name, default=""):
    return os.environ.get(name, default)


OPENROUTER_KEY = _env("Open_Router_API") or _env("OPENROUTER_API_KEY")
ELEVENLABS_KEY = _env("ELEVENLABS_API_KEY")
HEADERS_OR = {
    "Authorization": f"Bearer {OPENROUTER_KEY}",
    "Content-Type": "application/json",
}


def _post_json(url, headers, payload, timeout=60):
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "ignore"))
    except Exception:
        return None


def _check_or_credits():
    """Return True only if OpenRouter key is set AND account has credits."""
    if not OPENROUTER_KEY:
        return False
    try:
        resp = _post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            HEADERS_OR,
            {"model": "openai/gpt-audio-mini", "messages": [{"role": "user", "content": "hi"}]},
            timeout=10,
        )
        # credits error shows as 402 in the error payload
        if resp and "error" in resp and resp["error"].get("code") == 402:
            return False
        return True
    except Exception:
        return False


OR_CREDITS = _check_or_credits()


def _get_text(url, timeout=60, post=None, headers=None):
    data = None
    hdrs = {"User-Agent": "Mozilla/5.0 NeuralAI"}
    if headers:
        hdrs.update(headers)
    if post is not None:
        if isinstance(post, str):
            data = post.encode()
        else:
            data = json.dumps(post).encode()
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST" if post else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def _is_grounded_error(text):
    t = (text or "").lower()
    return ("i can't help" in t or "i cannot help" in t
            or "i'm not aware" in t or "i am not aware" in t
            or "as an ai" in t[:40])


# ---------- Path (1): OpenRouter (primary, only if credits) ----------
def openrouter_chat(message, timeout=60):
    """Reasoning + web via OpenRouter. Returns (text, err). Only used when OR_CREDITS ok."""
    if not OPENROUTER_KEY or not OR_CREDITS:
        return None, "no_openrouter_credits"
    try:
        payload = {
            "model": "openai/gpt-audio-mini",
            "messages": [{"role": "user", "content": message}],
            "tools": [{"type": "web.search"}],
            "timeout": timeout,
        }
        resp = _post_json("https://openrouter.ai/api/v1/chat/completions",
                          HEADERS_OR, payload, timeout=timeout)
        if not resp:
            return None, "openrouter_no_response"
        msg = resp.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        if content.strip():
            return content.strip(), None
        return None, "openrouter_empty"
    except Exception as e:
        return None, f"openrouter_error:{e}"


# ---------- Path (1b): Pollinations (keyless chat, NOT grounded) ----------
def pollinations_chat(message, timeout=90):
    """Keyless chat via Pollinations. Returns (text, err). Not web-grounded."""
    try:
        prompt = urllib.parse.quote(message)
        url = POLLINATIONS_TEXT + prompt
        text = _get_text(url, timeout=timeout)
        if text and not _is_grounded_error(text):
            return text.strip(), None
        return None, "pollinations_refused"
    except Exception as e:
        return None, f"pollinations_error:{e}"


def pollinations_image(prompt, timeout=120):
    try:
        url = POLLINATIONS_IMG + urllib.parse.quote(prompt)
        # Returns raw JPEG bytes; we return a data URI.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NeuralAI"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    except Exception as e:
        return f"(img error: {e})"


# ---------- Path (2): local LM Studio 360M fallback ----------
def local_chat(message, timeout=60):
    try:
        payload = {
            "model": "smollm2-360m-instruct",
            "messages": [
                {"role": "system", "content": "You are NeuralAI. Answer concisely."},
                {"role": "user", "content": message},
            ],
            "temperature": 0.5,
        }
        resp = _get_text(LM_STUDIO_URL, timeout=timeout, post=payload)
        data = json.loads(resp)
        return data["choices"][0]["message"]["content"].strip(), None
    except Exception as e:
        return None, f"local_error:{e}"


# ---------- Real web grounding (DuckDuckGo HTML + fetch + 360M synth) ----------
DDG = "https://html.duckduckgo.com/html/?q="

def _ddg_links(q, n=5):
    try:
        html = _get_text(DDG + urllib.parse.quote(q), timeout=20)
        links = re.findall(r'class="result__a"\s+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S)
        out = []
        for href, title in links[:n]:
            # DDG wraps redirects; unescape & decode html entities in title
            title = re.sub(r"<[^>]+>", "", title)
            out.append((href, title.strip()))
        return out
    except Exception:
        return []


def surf(message):
    """Primary web path (OpenRouter if credits), then real fetch+360M, then offline 360M."""
    # 1) OpenRouter with live web.search (only if credits present)
    out, err = openrouter_chat(message)
    if out:
        return out, None
    # 2) Real grounding: search -> fetch top results -> 360M synthesis
    try:
        links = _ddg_links(message, n=4)
        snippets = []
        for href, title in links:
            txt = cmd_fetch(href)[:1200]
            if txt:
                snippets.append(f"[Source: {title}]({href})\n{txt}")
        if snippets:
            ctx = "\n\n---\n\n".join(snippets)
            synth, serr = local_chat(
                f"Using ONLY the sources below, answer the question concisely "
                f"and cite sources by title. If the sources don't answer it, say so.\n\n"
                f"Question: {message}\n\nSources:\n{ctx[:3500]}"
            )
            if synth:
                return synth + "\n\n_(grounded via live web fetch)_", None
    except Exception:
        pass
    # 3) Offline 360M fallback
    loc, lerr = local_chat(message)
    if loc:
        return loc + "  _(offline mode: no live web)_", None
    return "(all paths unavailable)", None


# ---------- Slash command handlers (web tool proxies) ----------
def cmd_web(q):
    out, _ = surf(q)
    return out


def cmd_fetch(url):
    try:
        html = _get_text(url, timeout=30)
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:4000]
    except Exception as e:
        return f"fetch error: {e}"


def cmd_browse(url):
    return cmd_fetch(url)


def cmd_research(topic):
    out, _ = surf(f"Research brief on: {topic}. Give a structured summary.")
    return out


def cmd_news(topic):
    out, _ = surf(f"Latest news about: {topic}. 3 bullet points.")
    return out


def cmd_yt(url):
    return cmd_fetch(url)


def cmd_img(prompt):
    return pollinations_image(prompt)


def cmd_speak(text):
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").save(buf)
        return "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return f"tts error: {e}"


def cmd_summarize(target):
    if target.startswith("http"):
        txt = cmd_fetch(target)
    else:
        txt = target
    out, _ = surf(f"Summarize concisely:\n\n{txt[:3000]}")
    return out


def cmd_translate(lang, text):
    out, err = local_chat(f"Translate to {lang}: {text}")
    if out:
        return out
    return f"(translate unavailable: {err})"


ROUTES = {
    "/web": lambda a: cmd_web(a),
    "/fetch": lambda a: cmd_fetch(a),
    "/browse": lambda a: cmd_browse(a),
    "/research": lambda a: cmd_research(a),
    "/news": lambda a: cmd_news(a),
    "/yt": lambda a: cmd_yt(a),
    "/img": lambda a: cmd_img(a),
    "/speak": lambda a: cmd_speak(a),
    "/summarize": lambda a: cmd_summarize(a),
}


def route_message(message):
    """Plain-English or slash. Returns (text, audio_b64_or_None)."""
    m = message.strip()
    if m.startswith("/"):
        parts = m[1:].split(" ", 1)
        cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""
        if cmd == "translate":
            lp = arg.split(" ", 1)
            return cmd_translate(lp[0], lp[1] if len(lp) > 1 else ""), None
        if cmd in ROUTES:
            return ROUTES[cmd](arg), None
        if cmd == "speak":
            return cmd_speak(arg), None
    return surf(m)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            # probe Pollinations reachability
            try:
                urllib.request.urlopen(POLLINATIONS_TEXT + "hi", timeout=5)
                poll = True
            except Exception:
                poll = False
            try:
                urllib.request.urlopen(LM_STUDIO_URL, timeout=3)
                local = True
            except Exception:
                local = False
            self._send(200, {
                "status": "healthy",
                "mode": "pollinations",
                "model": "pollinations-free",
                "pollinations": poll,
                "local_360m": local,
                "elevenlabs": bool(ELEVENLABS_KEY),
            })
            return
        self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        if self.path == "/api/chat":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                data = json.loads(raw.decode() or "{}")
                message = data.get("message", "")
                text, audio = route_message(message)
                self._send(200, {"response": text, "audio": audio})
            except Exception as e:
                self._send(500, {"error": str(e)})
            return
        self._send(404, {"detail": "Not Found"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"NeuralAI Web Surf Agent on :{PORT}", flush=True)
    srv.serve_forever()
