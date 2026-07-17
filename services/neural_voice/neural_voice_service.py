#!/usr/bin/env python3
import os
import json
import base64
import asyncio
import logging
import httpx
import traceback
import subprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import sys as _sys
_TOOLS_PATH = "/home/workspace/Projects/NeuralAI/tools"
if _TOOLS_PATH not in _sys.path:
    _sys.path.insert(0, _TOOLS_PATH)
from _tool_layer import process_tool_tags
try:
    from web_intent import detect_web_intent as _detect_web_intent  # plain-English -> tool router
    _HAVE_WEB_INTENT = True
except Exception:
    _HAVE_WEB_INTENT = False

import re as _re

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OR_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"
_OR_STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
_OR_MODEL_CHAT = "openai/gpt-audio-mini"            # audio-in + audio-out S2S LLM
_OR_MODEL_STT = "openai/whisper-large-v3"           # speech -> text
_OR_MODEL_TTS = "openai/gpt-4o-mini-tts"            # text -> raw PCM audio (OpenAI-compatible)
_OR_API_KEY = os.environ.get("Open_Router_API") or os.environ.get("OPENROUTER_API_KEY")


def _or_headers():
    return {
        "Authorization": f"Bearer {_OR_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neuralai-web-ui-deandrewharris.zocomputer.io",
        "X-Title": "NeuralAI Voice",
    }


def _execute_neural_tool(name: str, args) -> str:
    try:
        if name in ("web_search", "search"):
            q = (args.get("query") or args.get("q") or "").strip()
            if not q:
                return "[TOOL] empty query"
            return process_tool_tags(f"<tool>search: {q}</tool>")
        if name in ("fetch_url", "fetch", "web_fetch"):
            url = (args.get("url") or "").strip()
            if not url:
                return "[TOOL] empty url"
            return process_tool_tags(f"<tool>fetch: {url}</tool>")
        if name in ("web_browse", "browse"):
            task = (args.get("task") or "").strip()
            if not task:
                return "[TOOL] empty task"
            return process_tool_tags(f"<tool>browse: {task}</tool>")
        if name in ("execute_code", "code"):
            code = (args.get("code") or "").strip()
            if not code:
                return "[TOOL] empty code"
            try:
                result = subprocess.run(
                    ["python3", "-c", code],
                    capture_output=True,
                    timeout=10,
                    text=True,
                    check=False,
                )
                output = result.stdout or result.stderr or ""
                return output
            except Exception as e:
                return f"[TOOL] code execution error: {e}"
        return f"[TOOL] unknown tool: {name}"
    except Exception as e:
        return f"[TOOL_ERROR] {e}"


NEURAL_SYSTEM_INSTRUCTION = "Use web_search aggressively for current/recent/real-world info, fetch_url for specific pages, web_browse for navigation tasks, and execute_code for computation."

NEURAL_TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="web_search",
        description="Search the web for current info, news, facts, or anything time-sensitive.",
        parameters={
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Search query"}},
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="fetch_url",
        description="Fetch and read the text content of a web page URL.",
        parameters={
            "type": "OBJECT",
            "properties": {"url": {"type": "STRING", "description": "Full https URL to read"}},
            "required": ["url"],
        },
    ),
    types.FunctionDeclaration(
        name="web_browse",
        description="Navigate the web to find information.",
        parameters={
            "type": "OBJECT",
            "properties": {"task": {"type": "STRING", "description": "The task to perform"}},
            "required": ["task"],
        },
    ),
    types.FunctionDeclaration(
        name="execute_code",
        description="Execute code to perform computation.",
        parameters={
            "type": "OBJECT",
            "properties": {"code": {"type": "STRING", "description": "The code to execute"}},
            "required": ["code"],
        },
    ),
]


# Configuration
PORT = int(os.environ.get("VOICE_PORT", 5001))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
MODEL_ID = "gemini-2.5-flash-native-audio-latest"
OR_API_KEY = _OR_API_KEY  # OpenRouter (valid key)

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralVoice")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    # Priority: OpenRouter S2S (valid key) > Gemini Live > ElevenLabs TTS
    if OR_API_KEY:
        mode = "openrouter"
    elif GEMINI_API_KEY:
        mode = "gemini"
    elif ELEVENLABS_API_KEY:
        mode = "elevenlabs"
    else:
        mode = "none"
    return {"message": "NeuralVoice Live Service", "status": "online", "mode": mode,
            "openrouter": bool(OR_API_KEY), "gemini": bool(GEMINI_API_KEY),
            "elevenlabs": bool(ELEVENLABS_API_KEY)}

@app.get("/health")
async def health():
    if OR_API_KEY:
        mode = "openrouter"
    elif GEMINI_API_KEY:
        mode = "gemini"
    elif ELEVENLABS_API_KEY:
        mode = "elevenlabs"
    else:
        mode = "none"
    return {"status": "healthy", "mode": mode, "model": (MODEL_ID if mode == "gemini" else _OR_MODEL_CHAT),
            "openrouter": bool(OR_API_KEY), "gemini": bool(GEMINI_API_KEY), "elevenlabs": bool(ELEVENLABS_API_KEY)}

async def openrouter_voice_session(websocket: WebSocket):
    """OpenRouter streaming speech-to-speech: STT -> NeuralAI web tools -> LLM audio reply."""
    logger.info("Starting OpenRouter S2S voice session")
    async with httpx.AsyncClient(timeout=90.0) as client:
        await websocket.send_json({"type": "status", "message": "Connected to NeuralAI voice (OpenRouter)."})

        # optional proactive greeting so the mic isn't dead on connect
        try:
            await _or_speak(client, "Hello, you can speak to me now.", websocket)
        except Exception as ge:
            logger.warning(f"proactive greeting skipped: {ge}")

        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("Client disconnected from OpenRouter loop")
                break
            except Exception as re:
                logger.error(f"receive error: {re}")
                break

            if data.get("type") == "config":
                continue
            if data.get("type") not in ("audio", "text"):
                continue

            try:
                if data.get("type") == "audio":
                    audio_b64 = data["data"]
                    text = await _or_transcribe(client, audio_b64)
                else:
                    text = data.get("data", "")

                if not text:
                    continue

                logger.info(f"User said: {text[:80]}")
                await websocket.send_json({"type": "user_transcript", "data": text})

                # Decide if this is a web/tool request
                tool_name, tool_params = _route_voice_intent(text)
                context = ""
                if tool_name:
                    logger.info(f"Voice tool: {tool_name}({tool_params})")
                    raw = _execute_neural_tool(tool_name, {"query": tool_params, "q": tool_params, "url": tool_params, "task": tool_params, "text": tool_params, "code": tool_params})
                    context = f"\n\nWeb result for '{text}':\n{raw[:4000]}"
                    await websocket.send_json({"type": "tool_used", "tool": tool_name})

                prompt = (
                    "You are NeuralAI, a helpful voice assistant. "
                    "Answer concisely and conversationally, suitable for text-to-speech."
                    f"{context}\n\nUser: {text}\nAssistant:"
                )
                await _or_speak(client, prompt, websocket, system=True)
            except Exception as e:
                logger.error(f"OpenRouter voice turn failed: {e}")
                await websocket.send_json({"type": "error", "message": f"Voice turn failed: {str(e)}"})


async def _or_transcribe(client: httpx.AsyncClient, audio_b64: str) -> str:
    """Speech-to-text via OpenRouter audio/transcriptions (whisper-large-v3)."""
    try:
        resp = await client.post(
            _OR_STT_URL,
            headers=_or_headers(),
            json={
                "model": _OR_MODEL_STT,
                "input_audio": {"data": audio_b64, "format": "wav"},
            },
        )
        if resp.status_code == 200:
            return resp.json().get("text", "").strip()
        logger.error(f"STT status {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.error(f"transcribe failed: {e}")
    return ""


async def _or_speak(client: httpx.AsyncClient, text_or_prompt: str, websocket: WebSocket, system: bool = False):
    """Generate speech from text (or a prompt if system=True) and stream PCM to client."""
    if system:
        messages = [
            {"role": "system", "content": "You are NeuralAI voice mode. Reply with only the spoken response."},
            {"role": "user", "content": text_or_prompt},
        ]
    else:
        messages = [{"role": "user", "content": text_or_prompt}]

    try:
        # Primary: chat model with audio output (gpt-audio-mini supports modalities:["audio"])
        resp = await client.post(
            _OPENROUTER_URL,
            headers=_or_headers(),
            json={
                "model": _OR_MODEL_CHAT,
                "messages": messages,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "pcm16"},
                "stream": False,
            },
        )
        if resp.status_code == 200:
            content = resp.json().get("choices", [{}])[0].get("message", {})
            audio_b64 = (content.get("audio", {}) or {}).get("data")
            if audio_b64:
                await websocket.send_json({"type": "audio", "data": audio_b64, "sampleRate": 24000})
                await websocket.send_json({"type": "turn_complete"})
                return
            # model returned text only -> synthesize with TTS
            txt = content.get("content", "")
            if txt:
                await _or_tts_fallback(client, txt, websocket)
                return
        logger.error(f"OR speak status {resp.status_code}: {resp.text[:200]}")
        # fallback to dedicated TTS endpoint
        await _or_tts_fallback(client, text_or_prompt, websocket)
    except Exception as e:
        logger.error(f"speech gen failed: {e}")
        await websocket.send_json({"type": "error", "message": f"Speech failed: {str(e)}"})


def _route_voice_intent(text: str):
    """Map spoken text to a NeuralAI web tool, or None for plain chat."""
    t = text.lower()
    if _HAVE_WEB_INTENT:
        try:
            res = _detect_web_intent(text)
            if res and res[0]:
                tool, params = res
                # params may be a dict or a string; normalize to a query string
                if isinstance(params, dict):
                    query = params.get("query") or params.get("url") or params.get("text") or params.get("task") or text
                else:
                    query = str(params)
                return tool, query
        except Exception:
            pass
    # heuristic fallback
    if any(k in t for k in ("search", "look up", "google", "news", "latest", "what is the")):
        return "web_search", text
    if "http" in t:
        m = _re.search(r"https?://\S+", text)
        if m:
            return "fetch_url", m.group(0)
    return None, text

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via WebSocket")
    
    # Check for keys on connection (OpenRouter is primary; Gemini/ElevenLabs are fallbacks)
    g_key = os.environ.get("GEMINI_API_KEY")
    e_key = os.environ.get("ELEVENLABS_API_KEY")
    o_key = os.environ.get("Open_Router_API") or os.environ.get("OPENROUTER_API_KEY")
    
    if not g_key and not e_key and not o_key:
        logger.error("Connection attempt failed: No API keys found in environment")
        await websocket.send_json({"type": "error", "message": "NeuralVoice Server Error: AI Engine Credentials Missing."})
        await websocket.close()
        return

    # Updated Voice Map with High-Quality v2 compatible IDs
    voice_map = {
        "Alexa": {"gemini": "Aoede", "eleven": "Xb7hH8MSUJpSbSDYk0k2"},
        "Sara": {"gemini": "Kore", "eleven": "EXAVITQu4vr4xnSDxMaL"},
        "Andrew": {"gemini": "Fenrir", "eleven": "nPczCjzI2devNBz1zQrb"},
        "Claude": {"gemini": "Puck", "eleven": "JBFqnCBsd6RMkjVDRZzb"}
    }

    current_voice = "Aoede"
    current_eleven_voice = "nPczCjzI2devNBz1zQrb"

    try:
        # Wait for initial config
        try:
            initial_data = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
            if initial_data.get("type") == "config":
                voice_pref = initial_data.get("voice", "Andrew")
                mapping = voice_map.get(voice_pref, voice_map["Andrew"])
                current_voice = mapping["gemini"]
                current_eleven_voice = mapping["eleven"]
                logger.info(f"Using preferred voice: {voice_pref} -> Gemini:{current_voice}, Eleven:{current_eleven_voice}")
        except asyncio.TimeoutError:
            logger.info("No initial config received, using defaults")
        except Exception as e:
            logger.error(f"Error receiving initial config: {e}")

        # OPENROUTER S2S MODE (primary)
        if o_key:
            logger.info("Starting OpenRouter S2S Mode (primary voice path)")
            try:
                await openrouter_voice_session(websocket)
            except Exception as oe:
                logger.error(f"OpenRouter S2S session failed: {oe}")
                await websocket.send_json({"type": "error", "message": f"OpenRouter voice failed: {str(oe)}"})
            return

        # ELEVENLABS MODE
        if not GEMINI_API_KEY and ELEVENLABS_API_KEY:
            logger.info(f"Starting ElevenLabs TTS Mode with voice: {current_eleven_voice}")
            async with httpx.AsyncClient(timeout=60.0) as client:
                while True:
                    try:
                        data = await websocket.receive_json()
                    except WebSocketDisconnect:
                        logger.info("Client disconnected from ElevenLabs loop")
                        break
                    
                    if data.get("type") == "text":
                        text = data["data"]
                        if not text: continue
                        
                        logger.info(f"TTS Request (ElevenLabs): {text[:50]}...")
                        try:
                            # Use PCM 22.05kHz (standard for non-pro tiers)
                            # Force Turbo v2.5 for lower latency and better stability
                            async with client.stream(
                                "POST",
                                f"https://api.elevenlabs.io/v1/text-to-speech/{current_eleven_voice}?output_format=pcm_22050",
                                headers={"xi-api-key": ELEVENLABS_API_KEY},
                                json={
                                    "text": text,
                                    "model_id": "eleven_turbo_v2_5",
                                    "voice_settings": {
                                        "stability": 0.4,
                                        "similarity_boost": 0.8,
                                        "style": 0.5,
                                        "use_speaker_boost": True
                                    }
                                }
                            ) as response:
                                if response.status_code == 200:
                                    chunk_count = 0
                                    async for chunk in response.aiter_bytes(chunk_size=16000):
                                        if chunk:
                                            audio_base64 = base64.b64encode(chunk).decode('utf-8')
                                            await websocket.send_json({"type": "audio", "data": audio_base64, "sampleRate": 22050})
                                            chunk_count += 1
                                    await websocket.send_json({"type": "turn_complete"})
                                    logger.info(f"Sent {chunk_count} audio chunks (22.05kHz) successfully")
                                else:
                                    err_text = await response.aread()
                                    err_msg = f"ElevenLabs API Error {response.status_code}: {err_text.decode()}"
                                    logger.error(err_msg)
                                    await websocket.send_json({"type": "error", "message": err_msg})
                        except Exception as e:
                            logger.error(f"ElevenLabs streaming failed: {e}")
                            await websocket.send_json({"type": "error", "message": f"TTS generation failed: {str(e)}"})
                    
                    elif data.get("type") == "config":
                        voice_pref = data.get("voice")
                        if voice_pref in voice_map:
                            current_eleven_voice = voice_map[voice_pref]["eleven"]
                            logger.info(f"Voice changed to: {current_eleven_voice}")

        # PRIMARY: OpenRouter streaming S2S (valid key)
        if o_key:
            await openrouter_voice_session(websocket)
            return

        # FALLBACK 1: Gemini Live
        elif GEMINI_API_KEY:
            logger.info(f"Starting Gemini Live Mode with model: {MODEL_ID}")
            try:
                client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})

                # Proactivity: let the model speak first so the mic isn't dead on connect.
                try:
                    await session.send_client_content(
                        turns=types.Content(parts=[types.Part(text="Hello, you can speak to me now.")]),
                        turn_complete=True,
                    )
                except Exception as pe:
                    logger.warning(f"proactive greeting skipped: {pe}")

                config = types.LiveConnectConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=current_voice
                            )
                        )
                    ),
                    system_instruction=NEURAL_SYSTEM_INSTRUCTION,
                    tools=[types.Tool(function_declarations=NEURAL_TOOL_DECLARATIONS)],
                )

                async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                    logger.info(f"Connected to Gemini Live API with voice: {current_voice}")

                    # Push-to-talk state: after the model finishes a turn, it goes idle
                    # and only listens again when the client sends a fresh "audio" chunk
                    # (the client runs its own VAD and only streams while the user speaks).
                    # If the client's VAD fails to emit ptt_stop (background noise, tab blur,
                    # ScriptProcessor dropped), the session can hang waiting for more user
                    # audio. To guarantee the listen->speak->listen cadence, the server arms
                    # a listen-timeout after each model turn: if no user input arrives within
                    # LISTEN_TIMEOUT_S, we force a turn_complete to Gemini and re-arm listening.
                    awaiting_speech = {"value": True}
                    _listen_timer = {"task": None}
                    LISTEN_TIMEOUT_S = 4.0  # back to listening if no new speech within this window

                    def _cancel_listen_timer():
                        t = _listen_timer.get("task")
                        if t and not t.done():
                            t.cancel()
                        _listen_timer["task"] = None

                    async def _listen_timeout():
                        try:
                            await asyncio.sleep(LISTEN_TIMEOUT_S)
                            logger.info("Listen-timeout reached; forcing turn_complete to return to listening")
                            try:
                                await session.send(types.LiveClientContent(turn_complete=True))
                            except Exception as te:
                                logger.warning(f"listen-timeout turn_complete failed: {te}")
                            awaiting_speech["value"] = True
                            await websocket.send_json({"type": "turn_complete"})
                        except asyncio.CancelledError:
                            pass

                    def _arm_listen_timer():
                        _cancel_listen_timer()
                        _listen_timer["task"] = asyncio.create_task(_listen_timeout())


                    async def receive_from_gemini():
                        try:
                            while True:
                                try:
                                    message = await session.receive()
                                except StopAsyncIteration:
                                    break
                                if message.server_content:
                                    model_turn = message.server_content.model_turn
                                    if model_turn:
                                        for part in model_turn.parts:
                                            if getattr(part, "inline_data", None):
                                                audio_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                                await websocket.send_json({"type": "audio", "data": audio_base64, "sampleRate": 16000})

                                    if message.server_content.turn_complete:
                                        # Model finished speaking; go idle until the user speaks again.
                                        # Arm the listen-timeout so a stuck client can't hang the session.
                                        awaiting_speech["value"] = True
                                        await websocket.send_json({"type": "turn_complete"})
                                        _arm_listen_timer()

                                if getattr(message, "tool_call", None):
                                    for fc in (message.tool_call.function_calls or []):
                                        fn_name = getattr(fc, "name", "")
                                        fn_args = getattr(fc, "args", {}) or {}
                                        call_id = getattr(fc, "id", None)
                                        logger.info(f"Tool Call: {fn_name}({fn_args})")
                                        result = _execute_neural_tool(fn_name, fn_args)
                                        fr = {"name": fn_name, "response": {"result": result[:6000]}}
                                        if call_id:
                                            fr["id"] = call_id
                                        await session.send_tool_response(
                                            function_responses=[fr]
                                        )
                                        logger.info(f"Tool Response sent for {fn_name}")
                        except Exception as e:
                            logger.error(f"Error receiving from Gemini: {e}")
                            await websocket.send_json({"type": "error", "message": f"Gemini stream error: {str(e)}"})

                    async def receive_from_client():
                        try:
                            while True:
                                data = await websocket.receive_json()
                                if data.get("type") == "audio":
                                    # Client only sends audio while the user is actually speaking
                                    # (VAD gate on the client side). Forward it to Gemini and mark
                                    # that we are now in an active conversation turn.
                                    _cancel_listen_timer()
                                    awaiting_speech["value"] = False
                                    audio_bytes = base64.b64decode(data["data"])
                                    await session.send(
                                        input=types.LiveClientContent(
                                            turns=[types.Content(parts=[types.Part(inline_data=types.Blob(data=audio_bytes, mime_type="audio/pcm"))])]
                                        )
                                    )
                                elif data.get("type") == "text":
                                    _cancel_listen_timer()
                                    awaiting_speech["value"] = False
                                    await session.send(
                                        input=types.LiveClientContent(
                                            turns=[types.Content(parts=[types.Part(text=data["data"])])]
                                        )
                                    )
                                elif data.get("type") == "ptt_start":
                                    # Explicit push-to-talk start (client VAD begin).
                                    _cancel_listen_timer()
                                    awaiting_speech["value"] = False
                                    logger.info("Push-to-talk start")
                                elif data.get("type") == "ptt_stop":
                                    # Client VAD end: signal end of user turn.
                                    try:
                                        await session.send(types.LiveClientContent(turn_complete=True))
                                    except Exception as se:
                                        logger.warning(f"ptt_stop send failed: {se}")
                                    logger.info("Push-to-talk stop")
                        except WebSocketDisconnect:
                            logger.info("Client disconnected")
                        except Exception as e:
                            logger.error(f"Error receiving from client: {e}")

                    await asyncio.gather(receive_from_gemini(), receive_from_client())
            except Exception as e:
                logger.error(f"Gemini Live Connection Failed: {e}")
                logger.error(traceback.format_exc())
                await websocket.send_json({"type": "error", "message": f"AI Engine Connection Failed: {str(e)}"})

        else:
            logger.error("No valid API keys (GEMINI or ELEVENLABS) found")
            await websocket.send_json({"type": "error", "message": "No valid API keys found"})

    except Exception as e:
        logger.error(f"Fatal Voice Bridge Error: {e}")
    finally:
        try:
            await websocket.close()
        except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
