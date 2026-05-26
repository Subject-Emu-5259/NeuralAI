#!/bin/bash
# NeuralAI Unified Service Startup Script
# Starts the single unified service (model + tools + UI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/dev/shm"

echo "========================================="
echo "NeuralAI Unified Startup (v5.1)"
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

# Kill existing
echo "[Startup] Stopping any existing core service..."
pkill -f "neural_core_service.py" 2>/dev/null || true
sleep 2

# Start Unified Service
echo ""
echo "[Startup] Starting Neural Core Service (port 5000)..."
echo "[Startup] This may take 30-60 seconds to load the model..."

cd "$SCRIPT_DIR"
nohup python3.12 neural_core_service.py > "$LOG_DIR/neuralai.log" 2> "$LOG_DIR/neuralai_err.log" &
CORE_PID=$!
echo "[Startup] Core PID: $CORE_PID"

if ! wait_for_service "NeuralAI" 5000 180; then
    echo "[Startup] Core service failed. Check logs: $LOG_DIR/neuralai_err.log"
    tail -20 "$LOG_DIR/neuralai_err.log"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ NeuralAI Unified Service running!"
echo "========================================="
echo ""
echo "URL: https://neuralai-deandrewharris.zocomputer.io"
echo "Logs: $LOG_DIR/neuralai.log"
echo ""
