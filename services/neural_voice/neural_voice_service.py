#!/usr/bin/env python3
import os
import json
import base64
import asyncio
import logging
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# Configuration
PORT = int(os.environ.get("VOICE_PORT", 5001))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
MODEL_ID = "gemini-2.0-flash" 

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
    return {"message": "NeuralVoice Live Service", "status": "online", "key_set": bool(GEMINI_API_KEY)}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_ID, "key_set": bool(GEMINI_API_KEY)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected via WebSocket")
    
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
                            async with client.stream(
                                "POST",
                                f"https://api.elevenlabs.io/v1/text-to-speech/{current_eleven_voice}?output_format=pcm_22050",
                                headers={"xi-api-key": ELEVENLABS_API_KEY},
                                json={
                                    "text": text,
                                    "model_id": "eleven_multilingual_v2",
                                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
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

        # GEMINI LIVE MODE
        elif GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
            
            config = {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": current_voice
                        }
                    }
                }
            }
            
            async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                logger.info(f"Connected to Gemini Live API with voice: {current_voice}")
                
                async def receive_from_gemini():
                    try:
                        async for message in session:
                            if message.server_content:
                                model_turn = message.server_content.model_turn
                                if model_turn:
                                    for part in model_turn.parts:
                                        if part.inline_data:
                                            audio_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                            await websocket.send_json({"type": "audio", "data": audio_base64, "sampleRate": 16000})
                                
                                if message.server_content.turn_complete:
                                    await websocket.send_json({"type": "turn_complete"})
                                    
                            if message.tool_call:
                                logger.info(f"Tool Call: {message.tool_call}")
                    except Exception as e:
                        logger.error(f"Error receiving from Gemini: {e}")
                        await websocket.send_json({"type": "error", "message": str(e)})

                async def receive_from_client():
                    try:
                        while True:
                            data = await websocket.receive_json()
                            if data.get("type") == "audio":
                                audio_bytes = base64.b64decode(data["data"])
                                await session.send(
                                    input=types.LiveClientContent(
                                        parts=[types.Part(inline_data=types.Blob(data=audio_bytes, mime_type="audio/pcm"))]
                                    )
                                )
                            elif data.get("type") == "text":
                                await session.send(
                                    input=types.LiveClientContent(
                                        parts=[types.Part(text=data["data"])]
                                    )
                                )
                            elif data.get("type") == "config":
                                # Handle mid-session voice change if needed
                                pass
                    except WebSocketDisconnect:
                        logger.info("Client disconnected")
                    except Exception as e:
                        logger.error(f"Error receiving from client: {e}")

                await asyncio.gather(receive_from_gemini(), receive_from_client())

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
