#!/bin/bash
# NeuralAI Unified Service Startup Script (v7.2)
# Starts llmster backend, voice service, and core service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/dev/shm"
LMS_BIN="$HOME/.lmstudio/bin/lms"

echo "========================================="
echo "NeuralAI Unified Startup (v7.2)"
echo "========================================="

# Function to wait for service
wait_for_service() {
    local name=$1
    local port=$2
    local max_wait=${3:-180}
    
    echo "[Startup] Waiting for $name on port $port..."
    
    for i in $(seq 1 $max_wait); do
        if curl -s "http://localhost:$port/api/status" > /dev/null 2>&1; then
            echo "[Startup] ✓ $name is ready!"
            return 0
        fi
        sleep 1
    done
    
    echo "[Startup] ✗ $name failed to start within ${max_wait}s"
    return 1
}

# Function to wait for voice health endpoint
wait_for_voice() {
    local name=$1
    local port=$2
    local max_wait=${3:-30}
    
    echo "[Startup] Waiting for $name on port $port..."
    
    for i in $(seq 1 $max_wait); do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "[Startup] ✓ $name is ready!"
            return 0
        fi
        sleep 1
    done
    
    echo "[Startup] ⚠ $name not available (voice features will be disabled)"
    return 1
}

# Kill existing
echo "[Startup] Stopping any existing services..."
pkill -f "neural_core_service.py" 2>/dev/null || true
pkill -f "neural_voice_service.py" 2>/dev/null || true
# Hard kill any llmster/lms/llama-server leftovers so the guarded auto-load
# starts from a clean slate. Without this, a stale lms session can hold port 1234
# or leave orphaned llama-server workers that the guard would otherwise attach to.
pkill -f "$HOME/.lmstudio/bin/lms" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
# Best-effort: ask lms to stop its server if the CLI is reachable.
if [ -x "$LMS_BIN" ]; then
    "$LMS_BIN" server stop 2>/dev/null || true
fi
sleep 2

# ============================
# Start llmster Backend (port 1234)
# ============================
echo ""
echo "[Startup] Ensuring single-instance llmster backend (guarded)..."
# All llmster startup funnels through the guarded loader so we never stack workers.
# It starts the server once (if needed), loads the model once, and exits if already healthy.
if [ -x "$SCRIPT_DIR/../scripts/llmster_auto_load.sh" ]; then
    "$SCRIPT_DIR/../scripts/llmster_auto_load.sh"
elif [ -x "/home/workspace/Projects/NeuralAI/scripts/llmster_auto_load.sh" ]; then
    /home/workspace/Projects/NeuralAI/scripts/llmster_auto_load.sh
else
    echo "[Startup] ✗ llmster_auto_load.sh not found — cannot start backend"
fi

# Verify llmster is serving
if curl -s "http://localhost:1234/v1/models" > /dev/null 2>&1; then
    echo "[Startup] ✓ llmster backend ready on port 1234"
else
    echo "[Startup] ⚠ llmster not responding, falling back to local PyTorch"
    export LLM_BACKEND=local
fi

# Start Voice Service (port 5001) — lightweight, starts fast
echo ""
echo "[Startup] Starting NeuralVoice Service (port 5001)..."
cd "$SCRIPT_DIR/neural_voice"
nohup python3 neural_voice_service.py > "$LOG_DIR/neural_voice.log" 2> "$LOG_DIR/neural_voice_err.log" &
VOICE_PID=$!
echo "[Startup] Voice PID: $VOICE_PID"
wait_for_voice "NeuralVoice" 5001 15 || true

# Start Unified Service (port 5000)
echo ""
echo "[Startup] Starting Neural Core Service (port 5000)..."
echo "[Startup] Backend: ${LLM_BACKEND:-lmstudio}"

cd "$SCRIPT_DIR"
LLM_BACKEND="${LLM_BACKEND:-lmstudio}" \
LLM_API_URL="http://localhost:1234/v1" \
LLM_MODEL="smollm2-360m-instruct" \
nohup python3 neural_core_service.py > "$LOG_DIR/neuralai.log" 2> "$LOG_DIR/neuralai_err.log" &
CORE_PID=$!
echo "[Startup] Core PID: $CORE_PID"

if ! wait_for_service "NeuralAI" 5000 180; then
    echo "[Startup] Core service failed. Check logs: $LOG_DIR/neuralai_err.log"
    tail -20 "$LOG_DIR/neuralai_err.log"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ NeuralAI Unified Service running! (v7.2)"
echo "  llmster: port 1234 (inference)"
echo "  Core: port 5000 (PID: $CORE_PID)"
echo "  Voice: port 5001 (PID: $VOICE_PID)"
echo "========================================="
echo ""
echo "URL: https://neuralai-deandrewharris.zocomputer.io"
echo "Logs: $LOG_DIR/neuralai.log"
echo ""
