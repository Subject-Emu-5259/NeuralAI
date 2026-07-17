#!/usr/bin/env bash
# Auto-launched by neural_core_service._ensure_voice_service()
# Runs NeuralVoice on port 5001 with ElevenLabs S2S.
cd "$(dirname "$0")"
export PORT="${PORT:-5001}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-$NEURAL_VOICE_GEMINI_KEY}"
export ELEVENLABS_API_KEY="${ELEVENLABS_API_KEY:-$NEURAL_VOICE_ELEVENLABS_KEY}"
exec python3 neural_voice_service.py
