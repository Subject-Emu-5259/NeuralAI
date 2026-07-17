#!/usr/bin/env bash
# Launch NeuralVoice on port 5001 (OpenRouter S2S primary, Gemini/ElevenLabs fallback).
cd "$(dirname "$0")"
export PORT="${PORT:-5001}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-$NEURAL_VOICE_GEMINI_KEY}"
export ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-$NEURAL_VOICE_ELEVENLABS_KEY}"
export Open_Router_API="${Open_Router_API:-$OPENROUTER_API_KEY}"
exec python3 neural_voice_service.py
