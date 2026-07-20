"""Translation for NeuralAI.

Backend chain (first that works):
  1. Gemini (GEMINI_API_KEY) when present and valid.
  2. OpenRouter (Open_Router_API) multilingual chat model.
  3. Raises a clear error if no usable backend is configured.

Exposes translate_text(text, target) -> translated string.
"""
import os
import urllib.request
import json


def _translate_openrouter(text: str, target: str) -> str:
    api_key = os.environ.get("Open_Router_API") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Open_Router_API not set")
    prompt = (
        f"Translate the following text to {target}. "
        "Respond with ONLY the translation, no quotes, no commentary:\n\n" + text
    )
    body = json.dumps({
        "model": "google/gemini-2.5-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512,
    })
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body.encode(),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://neuralai-web-ui-deandrewharris.zocomputer.io",
            "X-Title": "NeuralAI",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def _translate_gemini(text: str, target: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    from google.genai import types
    from google import genai
    client = genai.Client(api_key=api_key)
    prompt = (
        f"Translate the following text to {target}. "
        "Respond with ONLY the translation, no quotes, no commentary:\n\n" + text
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return (resp.text or "").strip()


def _translate_pollinations(text: str, target: str) -> str:
    """Keyless fallback via Pollinations text API (no credits needed)."""
    import urllib.parse
    prompt = f"Translate the following text to {target}. Respond with ONLY the translation, no quotes, no commentary:\n\n{text}"
    url = "https://text.pollinations.ai/" + urllib.parse.quote(prompt, safe="")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 NeuralAI"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = resp.read().decode("utf-8", "ignore").strip()
    if not out:
        raise RuntimeError("Pollinations returned empty")
    return out


def translate_text(text: str, target: str = "es") -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    # Gemini (401 dead) -> OpenRouter (needs credits) -> Pollinations (keyless fallback)
    for fn in (_translate_gemini, _translate_openrouter, _translate_pollinations):
        try:
            return fn(text, target)
        except Exception:
            continue
    raise RuntimeError(
        "Translation failed: Gemini, OpenRouter, and Pollinations all failed."
    )
