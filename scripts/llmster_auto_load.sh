#!/bin/bash
# llmster_auto_load.sh — Ensures llmster model is loaded
# Run this on ZO Computer startup or via cron
# Usage: Add to crontab: @reboot /home/workspace/Projects/NeuralAI/scripts/llmster_auto_load.sh

LMS_BIN="$HOME/.lmstudio/bin/lms"
MODEL_ID="smollm2-360m-instruct"
PORT=1234
MAX_RETRIES=5
RETRY_DELAY=2

echo "[llmster] Starting auto-load check..."

# Wait for llmster server to be ready
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "[llmster] Server is responding on port $PORT"
        break
    fi
    
    echo "[llmster] Server not ready, starting daemon (attempt $i/$MAX_RETRIES)..."
    "$LMS_BIN" server start --port $PORT 2>/dev/null || true
    sleep $RETRY_DELAY
done

# Check if model is already loaded
if curl -s "http://localhost:$PORT/v1/models" | grep -q "smollm2"; then
    echo "[llmster] ✓ Model already loaded"
    exit 0
fi

# Load the model
echo "[llmster] Loading $MODEL_ID..."
"$LMS_BIN" load "$MODEL_ID" -y 2>&1

sleep 2

# Verify
if curl -s "http://localhost:$PORT/v1/models" | grep -q "smollm2"; then
    echo "[llmster] ✓ Model loaded successfully"
    exit 0
else
    echo "[llmster] ✗ Model load failed"
    exit 1
fi
