"""Text-to-speech for NeuralAI using Gemini TTS (falls back to gTTS if no key).

Exposes text_to_speech(text) -> public audio URL path served by the web UI.
Audio is written to the workspace and returned as a zo.space-style asset URL
so the chat can play it back in the browser.
"""
import os
import uuid
from pathlib import Path

AUDIO_DIR = Path("/home/workspace/Projects/NeuralAI/from-scratch/web_ui/static/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _public_url(filename: str) -> str:
    return f"/static/audio/{filename}"


def text_to_speech(text: str, voice: str = "Zephyr") -> str:
    """Convert text to speech and return a URL to the audio file."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    fname = f"{uuid.uuid4().hex}.mp3"
    out_path = AUDIO_DIR / fname

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            from google.genai import types
            from google import genai
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                ),
            )
            data = resp.candidates[0].content.parts[0].inline_data.data
            out_path.write_bytes(data)
            return _public_url(fname)
        except Exception:
            pass  # fall through to gTTS

    # Fallback: gTTS (no API key required)
    try:
        from gtts import gTTS
        gTTS(text=text, lang="en").save(str(out_path))
        return _public_url(fname)
    except Exception as e:
        raise RuntimeError(f"TTS unavailable (no Gemini key, gTTS failed: {e})")
