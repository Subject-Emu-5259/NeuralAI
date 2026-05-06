#!/bin/bash
# NeuralAI Microservices Startup Script
# Starts services in dependency order (like systemd)
# 
# Order:
# 1. Model service (loads model, slowest)
# 2. Tools service (sandbox)
# 3. WebUI service (depends on model + tools)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/tmp/neuralai_logs"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "NeuralAI Microservices Startup"
echo "========================================="

# Function to wait for a service
wait_for_service() {
    local name=$1
    local port=$2
    local max_wait=${3:-120}
    
    echo "[Startup] Waiting for $name on port $port..."
    
    for i in $(seq 1 $max_wait); do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "[Startup] ✓ $name is ready!"
            return 0
        fi
        sleep 1
    done
    
    echo "[Startup] ✗ $name failed to start within ${max_wait}s"
    return 1
}

# Kill any existing services
echo "[Startup] Stopping any existing services..."
pkill -f "model_service.py" 2>/dev/null || true
pkill -f "tools_service.py" 2>/dev/null || true
pkill -f "webui_service.py" 2>/dev/null || true
sleep 2

# =================================
# 1. START MODEL SERVICE
# =================================
echo ""
echo "[Startup] Starting Model Service (port 7001)..."
echo "[Startup] This may take 30-60 seconds to load the model..."

cd "$PROJECT_ROOT/services"
nohup python3 model_service.py > "$LOG_DIR/model.log" 2>&1 &
MODEL_PID=$!
echo "[Startup] Model PID: $MODEL_PID"

# Wait for model (longer timeout since it needs to load)
if ! wait_for_service "Model" 7001 180; then
    echo "[Startup] Model service failed. Check logs: $LOG_DIR/model.log"
    tail -20 "$LOG_DIR/model.log"
    exit 1
fi

# =================================
# 2. START TOOLS SERVICE
# =================================
echo ""
echo "[Startup] Starting Tools Service (port 7002)..."

nohup python3 tools_service.py > "$LOG_DIR/tools.log" 2>&1 &
TOOLS_PID=$!
echo "[Startup] Tools PID: $TOOLS_PID"

if ! wait_for_service "Tools" 7002 30; then
    echo "[Startup] Tools service failed. Check logs: $LOG_DIR/tools.log"
    tail -20 "$LOG_DIR/tools.log"
    exit 1
fi

# =================================
# 3. START WEBUI SERVICE
# =================================
echo ""
echo "[Startup] Starting WebUI Service (port 5000)..."

nohup python3 webui_service.py > "$LOG_DIR/webui.log" 2>&1 &
WEBUI_PID=$!
echo "[Startup] WebUI PID: $WEBUI_PID"

if ! wait_for_service "WebUI" 5000 30; then
    echo "[Startup] WebUI service failed. Check logs: $LOG_DIR/webui.log"
    tail -20 "$LOG_DIR/webui.log"
    exit 1
fi

# =================================
# DONE
# =================================
echo ""
echo "========================================="
echo "✓ All NeuralAI services running!"
echo "========================================="
echo ""
echo "Services:"
echo "  • Model:  http://localhost:7001 (PID: $MODEL_PID)"
echo "  • Tools:  http://localhost:7002 (PID: $TOOLS_PID)"
echo "  • WebUI:  http://localhost:5000 (PID: $WEBUI_PID)"
echo ""
echo "Logs:"
echo "  • Model:  $LOG_DIR/model.log"
echo "  • Tools:  $LOG_DIR/tools.log"
echo "  • WebUI:  $LOG_DIR/webui.log"
echo ""
echo "To stop all services:"
echo "  pkill -f 'model_service.py|tools_service.py|webui_service.py'"
echo ""

# Save PIDs for later
echo "$MODEL_PID" > /tmp/neuralai_model.pid
echo "$TOOLS_PID" > /tmp/neuralai_tools.pid
echo "$WEBUI_PID" > /tmp/neuralai_webui.pid
