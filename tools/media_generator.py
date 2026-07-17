"""Unified Pollinations media generation for NeuralAI.

Single keyed client covering text, image, video, audio (TTS), realtime voice,
and embeddings — all OpenAI-compatible against https://gen.pollinations.ai/v1.

Key facts (verified 2026-07-16):
- Image: https://image.pollinations.ai/v1/generations (model "flux") is the FREE
  tier and works with the key. Raw JPEG bytes are returned (not b64 JSON).
- Text chat: https://gen.pollinations.ai/v1/chat/completions (model "openai").
- Audio/TTS, embeddings, video, realtime voice require pollen balance on the key.
  When balance is 0 they return 402/404. We degrade gracefully:
    * TTS falls back to tools.tts (gTTS).
    * Others return a clear "balance required" message instead of fake output.
"""

import base64
import json
import logging
import os
import uuid
from pathlib import Path

import requests

logger = logging.getLogger("neuralai.media")

POLLINATIONS_KEY = os.environ.get("Pollinations_Api_key") or os.environ.get("POLLINATIONS_API_KEY") or ""
BASE = "https://gen.pollinations.ai/v1"
IMG_BASE = "https://image.pollinations.ai/v1/generations"

MEDIA_DIR = Path("/home/workspace/Projects/NeuralAI/from-scratch/web_ui/static/media")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if POLLINATIONS_KEY:
        h["Authorization"] = f"Bearer {POLLINATIONS_KEY}"
    return h


def _save_bytes(data: bytes, ext: str) -> Path:
    fname = f"{uuid.uuid4().hex}.{ext}"
    out = MEDIA_DIR / fname
    out.write_bytes(data)
    return out


def _balance_error(kind: str) -> dict:
    return {
        "success": False,
        "error": (
            f"{kind} requires pollen balance on the Pollinations key "
            f"(this sk_ key currently has 0 balance). Top up at enter.pollinations.ai "
            f"to enable it. Text + image work without balance."
        ),
    }


def generate_text(prompt: str, system: str = "", max_tokens: int = 800) -> dict:
    """Chat completion via Pollinations (works with key)."""
    if not prompt:
        return {"success": False, "error": "Usage: /text <prompt>"}
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = requests.post(
            f"{BASE}/chat/completions",
            headers=_headers(),
            json={"model": "openai", "messages": messages, "max_tokens": max_tokens},
            timeout=60,
        )
        if resp.status_code != 200:
            return {"success": False, "error": f"Pollinations text error {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"success": True, "provider": "pollinations", "text": text}
    except Exception as e:
        logger.exception("text_gen failed")
        return {"success": False, "error": f"Text generation failed: {e}"}


def generate_image(prompt: str, width: int = 768, height: int = 768) -> dict:
    """Image generation via the free Pollinations image endpoint (raw JPEG)."""
    if not prompt:
        return {"success": False, "error": "Usage: /img <prompt>"}
    try:
        resp = requests.post(
            IMG_BASE,
            headers=_headers(),
            json={"model": "flux", "prompt": prompt, "width": width, "height": height, "n": 1},
            timeout=90,
        )
        if resp.status_code != 200 or not resp.content.startswith(b"\xff\xd8"):
            return {"success": False, "error": f"Pollinations image error {resp.status_code}: {resp.text[:200]}"}
        out = _save_bytes(resp.content, "jpg")
        return {
            "success": True,
            "provider": "pollinations",
            "image_path": str(out),
            "image_url": f"/static/media/{out.name}",
        }
    except Exception as e:
        logger.exception("image_gen failed")
        return {"success": False, "error": f"Image generation failed: {e}"}


def generate_video(prompt: str, duration: int = 4) -> dict:
    """Video generation. Requires pollen balance on the key."""
    if not prompt:
        return {"success": False, "error": "Usage: /video <prompt>"}
    try:
        resp = requests.post(
            f"{BASE}/video/generations",
            headers=_headers(),
            json={"model": "pollinations/video", "prompt": prompt, "duration": duration},
            timeout=120,
        )
        if resp.status_code in (402, 404, 426):
            return _balance_error("Video generation")
        if resp.status_code != 200:
            return {"success": False, "error": f"Pollinations video error {resp.status_code}: {resp.text[:200]}"}
        out = _save_bytes(resp.content, "mp4")
        return {"success": True, "provider": "pollinations", "video_url": f"/static/media/{out.name}"}
    except Exception as e:
        logger.exception("video_gen failed")
        return {"success": False, "error": f"Video generation failed: {e}"}


def generate_audio(text: str, voice: str = "alloy") -> dict:
    """TTS via Pollinations; falls back to tools.tts (gTTS) on any failure/balance issue."""
    if not text:
        return {"success": False, "error": "Usage: /audio <text>"}
    try:
        resp = requests.post(
            f"{BASE}/audio/speech",
            headers=_headers(),
            json={"model": "tts-1", "input": text, "voice": voice},
            timeout=60,
        )
        if resp.status_code in (402, 400):
            # Paid tier / wrong model — fall back to gTTS (already live in /speak).
            from tools.tts import text_to_speech
            url = text_to_speech(text)
            return {"success": True, "provider": "gtts-fallback", "audio_url": url}
        if resp.status_code != 200:
            return {"success": False, "error": f"Pollinations audio error {resp.status_code}: {resp.text[:200]}"}
        out = _save_bytes(resp.content, "mp3")
        return {"success": True, "provider": "pollinations", "audio_url": f"/static/media/{out.name}"}
    except Exception:
        try:
            from tools.tts import text_to_speech
            url = text_to_speech(text)
            return {"success": True, "provider": "gtts-fallback", "audio_url": url}
        except Exception as e:
            logger.exception("audio_gen failed")
            return {"success": False, "error": f"Audio generation failed: {e}"}


def generate_embeddings(text: str, model: str = "openai-3-small") -> dict:
    """Text embeddings. Requires pollen balance on the key."""
    if not text:
        return {"success": False, "error": "Usage: /embed <text>"}
    try:
        resp = requests.post(
            f"{BASE}/embeddings",
            headers=_headers(),
            json={"model": model, "input": text},
            timeout=60,
        )
        if resp.status_code == 402:
            return _balance_error("Embeddings")
        if resp.status_code != 200:
            return {"success": False, "error": f"Pollinations embeddings error {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        vec = data["data"][0]["embedding"]
        return {"success": True, "provider": "pollinations", "model": model, "dimensions": len(vec), "embedding": vec}
    except Exception as e:
        logger.exception("embeddings failed")
        return {"success": False, "error": f"Embeddings failed: {e}"}


def realtime_voice_url() -> str:
    """WebSocket URL for realtime voice. Requires pollen balance on the key."""
    return "wss://gen.pollinations.ai/v1/realtime"


# Singleton used by tool_handler
media = {
    "text": generate_text,
    "image": generate_image,
    "video": generate_video,
    "audio": generate_audio,
    "embeddings": generate_embeddings,
    "realtime_url": realtime_voice_url,
}
