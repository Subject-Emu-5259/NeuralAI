#!/bin/bash
export ELEVENLABS_API_KEY=$(python3 -c "import os; print(os.environ.get('ELEVENLABS_API_KEY', ''))")
export GEMINI_API_KEY=$(python3 -c "import os; print(os.environ.get('GEMINI_API_KEY', ''))")
export VOICE_PORT=5001
nohup python3 neural_voice_service.py > /dev/shm/neural_voice.log 2>&1 &
echo "NeuralVoice started on port 5001"
