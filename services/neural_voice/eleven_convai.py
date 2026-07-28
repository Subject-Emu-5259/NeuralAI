#!/usr/bin/env python3
"""
ElevenLabs Conversational AI (true Speech-to-Speech) bridge for NeuralVoice.

Option B (per user plan): the browser connects DIRECTLY to ElevenLabs's
wss://api.elevenlabs.io/v1/convai/conversation endpoint. ElevenLabs handles
ASR + LLM + TTS in one low-latency audio loop and returns spoken audio. No
separate STT/360M step in NeuralAI's loop.

This module:
  - ensures a Conversational AI agent exists (creates one if missing),
  - returns the connection URL (public agent -> plain wss URL; private agent
    -> signed URL from the server using ELEVENLABS_API_KEY),
  - exposes convai_available() so the main service can advertise/prefer it.
"""
import os
import logging
import httpx

logger = logging.getLogger("NeuralVoice.ConvAI")

ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
_CONVAI_BASE = "https://api.elevenlabs.io/v1/convai"
_AGENT_ID_ENV = "ELEVENLABS_AGENT_ID"  # optional: pin a specific agent

# System prompt: keep the NeuralAI v7.2 Expert persona voice.
SYSTEM_PROMPT = (
    "You are NeuralAI, a high-density intelligence engine with an expert persona. "
    "You are concise, accurate, and conversational. You answer spoken questions naturally, "
    "as if talking to a thoughtful user. Keep replies short enough for natural speech "
    "(usually 1-3 sentences) unless the user asks for depth."
)

_FIRST_MESSAGE = "Hello, I'm NeuralAI. You can speak to me now."


def _headers():
    return {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}


def convai_available() -> bool:
    return bool(ELEVEN_API_KEY) and ELEVEN_API_KEY not in ("", "placeholder")


def get_agent_id() -> str | None:
    """Return the configured/pinned agent id, or None."""
    return os.environ.get(_AGENT_ID_ENV) or None


def _list_agents() -> list:
    if not convai_available():
        return []
    try:
        r = httpx.get(f"{_CONVAI_BASE}/agents", headers=_headers(), timeout=20.0)
        if r.status_code == 200:
            return r.json().get("agents", [])
    except Exception as e:
        logger.warning(f"list agents failed: {e}")
    return []


def _create_agent() -> str | None:
    """Create a production Conversational AI agent if none exists."""
    if not convai_available():
        return None
    payload = {
        "name": "NeuralAI Voice (S2S)",
        "conversation_config": {
            "agent": {
                "prompt": {
                    "prompt": SYSTEM_PROMPT,
                    "first_message": _FIRST_MESSAGE,
                },
                "language": "en",
            },
            "tts": {
                "voice_id": "nPczCjzI2devNBz1zQrb",  # Andrew (matches NeuralVoice default)
                "model_id": "eleven_turbo_v2_5",
            },
            "asr": {"user_input_audio_format": "pcm_16000", "model": "nova-3"},
        },
        "platform_settings": {
            "auth": {"type": "public" }  # public agent -> no signed URL needed
        },
    }
    try:
        r = httpx.post(f"{_CONVAI_BASE}/agents/create", headers=_headers(), json=payload, timeout=30.0)
        if r.status_code in (200, 201):
            aid = r.json().get("agent_id")
            logger.info(f"Created ElevenLabs ConvAI agent: {aid}")
            return aid
        logger.error(f"agent create failed {r.status_code}: {r.text[:160]}")
    except Exception as e:
        logger.error(f"agent create exception: {e}")
    return None


def ensure_agent() -> str | None:
    """Return a usable agent id, creating one on first run if necessary."""
    existing = get_agent_id()
    if existing:
        return existing
    agents = _list_agents()
    if agents:
        aid = agents[0].get("agent_id")
        os.environ[_AGENT_ID_ENV] = aid
        return aid
    aid = _create_agent()
    if aid:
        os.environ[_AGENT_ID_ENV] = aid
    return aid


def connection_url() -> str | None:
    """Public agent -> plain wss URL. Private agent -> signed URL."""
    aid = ensure_agent()
    if not aid:
        return None
    return f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={aid}"


def signed_url() -> str | None:
    """Always fetch a server-signed URL (works for public + private agents)."""
    aid = ensure_agent()
    if not aid:
        return None
    try:
        r = httpx.get(
            f"{_CONVAI_BASE}/conversation/get-signed-url",
            headers=_headers(),
            params={"agent_id": aid},
            timeout=20.0,
        )
        if r.status_code == 200:
            return r.json().get("signed_url")
    except Exception as e:
        logger.warning(f"signed url failed: {e}")
    return None
