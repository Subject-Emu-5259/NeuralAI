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
MODEL_ID = "gemini-2.0-flash" # Use stable GA model

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
    
    # Receive initial config if provided
    current_voice = "Aoede" # Default for Gemini
    current_eleven_voice = "21m00Tcm4TlvDq8ikWAM" # Default (Rachel) for ElevenLabs
    
    voice_map = {
        "Alexa": {"gemini": "Aoede", "eleven": "21m00Tcm4TlvDq8ikWAM"},
        "Sara": {"gemini": "Kore", "eleven": "AZnzlk1XhkUvSthjVnxt"},
        "Andrew": {"gemini": "Fenrir", "eleven": "VR6A4W6AnD6yPstS7j4S"},
        "Claude": {"gemini": "Puck", "eleven": "pNInz6obpg8P277XqZzO"}
    }

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

    # Use ElevenLabs if Gemini key is missing but ElevenLabs is present
    if not GEMINI_API_KEY and ELEVENLABS_API_KEY:
        logger.info(f"Using ElevenLabs TTS Mode with voice: {current_eleven_voice}")
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "text":
                    text = data["data"]
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"https://api.elevenlabs.io/v1/text-to-speech/{current_eleven_voice}?output_format=pcm_16000",
                            headers={"xi-api-key": ELEVENLABS_API_KEY},
                            json={"text": text, "model_id": "eleven_monolingual_v1"}
                        )
                        if resp.status_code == 200:
                            audio_base64 = base64.b64encode(resp.content).decode('utf-8')
                            await websocket.send_json({"type": "audio", "data": audio_base64})
                        else:
                            await websocket.send_json({"type": "error", "message": f"ElevenLabs Error: {resp.text}"})
                elif data.get("type") == "config":
                    voice_pref = data.get("voice")
                    if voice_pref in voice_map:
                        current_eleven_voice = voice_map[voice_pref]["eleven"]
                        logger.info(f"ElevenLabs Voice changed to: {current_eleven_voice}")
        except Exception as e:
            logger.error(f"ElevenLabs Bridge Error: {e}")
        return

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set")
        await websocket.send_json({"type": "error", "message": "GEMINI_API_KEY not set. Please add it to the 'neural-voice' service environment variables in Settings > Sites."})
        await websocket.close()
        return

    try:
        client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})
        
        # Initial config with voice mapping
        voice_map = {
            "Alexa": "Aoede",
            "Sara": "Kore",
            "Andrew": "Fenrir",
            "Claude": "Puck"
        }
        
        async def run_session(voice_name):
            config = {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": voice_name
                        }
                    }
                }
            }
            
            async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                logger.info(f"Connected to Gemini Live API with voice: {voice_name}")
                
                async def receive_from_gemini():
                    try:
                        async for message in session:
                            if message.server_content:
                                model_turn = message.server_content.model_turn
                                if model_turn:
                                    for part in model_turn.parts:
                                        if part.inline_data:
                                            audio_base64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                                            await websocket.send_json({"type": "audio", "data": audio_base64})
                                
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
                                # Handle mid-session voice change if needed (reconnect required for Gemini Live usually)
                                new_voice = data.get("voice", "Aoede")
                                logger.info(f"Voice change requested: {new_voice}")
                                # For now, we'll just log it. Real-time change might require re-init
                    except WebSocketDisconnect:
                        logger.info("Client disconnected")
                    except Exception as e:
                        logger.error(f"Error receiving from client: {e}")

                await asyncio.gather(receive_from_gemini(), receive_from_client())

        # Wait for first message to potentially get config
        try:
            initial_data = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
            if initial_data.get("type") == "config":
                voice_pref = initial_data.get("voice", "Andrew")
                current_voice = voice_map.get(voice_pref, "Aoede")
                logger.info(f"Using preferred voice: {current_voice} for {voice_pref}")
            else:
                # If not config, process it normally in the loop (but we need to start session first)
                # For simplicity, we'll start with default and handle messages inside
                pass
        except asyncio.TimeoutError:
            logger.info("No initial config received, using default voice")

        await run_session(current_voice)

    except Exception as e:
        logger.error(f"Gemini Live Session Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": f"Gemini Session Error: {str(e)}"})
        except: pass
    finally:
        logger.info("Cleaning up session")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
