#!/bin/bash
# run_safe.sh — Crash-safe wrapper for NeuralAI webui_service
# Auto-restarts the service if it exits with a non-zero code.
# Usage: bash run_safe.sh
# Add this as the ZO Hosting entrypoint instead of direct python invocation.

SERVICE_DIR="/home/workspace/Projects/NeuralAI/services"
SERVICE_PY="$SERVICE_DIR/webui_service.py"
LOG_FILE="/dev/shm/neuralai-web-ui.log"
ERR_FILE="/dev/shm/neuralai-web-ui_err.log"
MAX_RESTARTS=50
RESTART_DELAY=3

restart_count=0

echo "[run_safe] NeuralAI crash-safe launcher started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

while [ $restart_count -lt $MAX_RESTARTS ]; do
    restart_count=$((restart_count + 1))
    echo "[run_safe] === Launch #$restart_count === $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
    
    python3 "$SERVICE_PY" >> "$LOG_FILE" 2>> "$ERR_FILE"
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "[run_safe] Service exited cleanly (code 0). Stopping." | tee -a "$LOG_FILE"
        break
    fi
    
    echo "[run_safe] Service crashed with exit code $exit_code. Restarting in ${RESTART_DELAY}s..." | tee -a "$ERR_FILE"
    sleep $RESTART_DELAY
    
    # Run garbage collection between restarts to free memory
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
done

if [ $restart_count -ge $MAX_RESTARTS ]; then
    echo "[run_safe] CRITICAL: Hit max restarts ($MAX_RESTARTS). Giving up." | tee -a "$ERR_FILE"
fi

echo "[run_safe] Launcher exiting at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
