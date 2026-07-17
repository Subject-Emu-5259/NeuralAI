#!/bin/bash
# llmster_auto_load.sh — Idempotent, single-instance loader for the NeuralAI inference model.
# Guarantees ONE llmster server on port 1234 and ONE loaded model worker.
# Safe to call from @reboot, start_all.sh, cron, or manually — repeated/parallel
# invocations will no-op instead of spawning duplicate llama-server workers.
#
# DUAL GUARD:
#   1. flock on /tmp/llmster_autoload.lock — blocks concurrent runs (re-entrancy).
#   2. Port probe on 1234 — if a server is already bound, we attach to it and never start a second one.

LMS_BIN="$HOME/.lmstudio/bin/lms"
MODEL_ID="smollm2-360m-instruct"
PORT=1234
LOCK_FILE="/tmp/llmster_autoload.lock"
MAX_RETRIES=10
RETRY_DELAY=2

# --- Guard 2 (pre-flight): is a server already listening? If so, just ensure model loaded. ---
server_up() { curl -s -m 3 "http://localhost:$PORT/v1/models" > /dev/null 2>&1; }

# --- Guard 1: single-instance lock (re-entrancy protection) ---
exec 9>"$LOCK_FILE" || { echo "[llmster] Cannot open lock file — aborting."; exit 1; }
if ! flock -n 9; then
    echo "[llmster] Another auto-load instance is already running — aborting to avoid duplicates."
    exit 0
fi

echo "[llmster] Acquired lock (PID $$). Starting idempotent load sequence..."

# If a server is already up, skip starting a new one entirely.
if server_up; then
    echo "[llmster] Server already listening on $PORT — attaching (no new server started)."
else
    echo "[llmster] No server on $PORT — starting exactly one llmster server."
    "$LMS_BIN" server start --port "$PORT" 2>/dev/null || true
    for i in $(seq 1 $MAX_RETRIES); do
        if server_up; then
            echo "[llmster] Server responding on $PORT."
            break
        fi
        echo "[llmster] Waiting for server (attempt $i/$MAX_RETRIES)..."
        sleep "$RETRY_DELAY"
    done
fi

# Confirm a server is actually reachable before loading.
if ! server_up; then
    echo "[llmster] ERROR: server did not come up on $PORT — aborting (lock released on exit)."
    exit 1
fi

# If the model is already loaded, we are done — never call load() again.
if curl -s "http://localhost:$PORT/v1/models" | grep -q "$MODEL_ID"; then
    echo "[llmster] Model '$MODEL_ID' already loaded — single instance confirmed. Done."
    exit 0
fi

# Load exactly once, with conservative resource caps:
#   --parallel 1 : one concurrent request (no worker fan-out / CPU contention)
#   -c 8192      : context length (kept from prior config)
echo "[llmster] Loading $MODEL_ID (parallel=1, context=8192)..."
"$LMS_BIN" load "$MODEL_ID" -y --parallel 1 -c 8192 2>&1

sleep 3

if curl -s "http://localhost:$PORT/v1/models" | grep -q "$MODEL_ID"; then
    echo "[llmster] Model loaded successfully. Single instance confirmed."
    exit 0
else
    echo "[llmster] Model load failed — leaving server up for manual inspection."
    exit 1
fi
# --- CPU THREADS / MLOCK TUNING (read-only note) ---
# llmster derives --threads and --mlock from the model's SAVED SERVER PRESET in LM Studio,
# which the `lms` CLI cannot override at load time. To apply the optimal args:
#   - threads = number of CPU cores (this box: $(nproc))
#   - drop --mlock (avoid pinning all RAM; lets the kernel reclaim under pressure)
# open LM Studio once, go to Server > CPU threads = $(nproc), uncheck "Use mlock", save the preset.
# The guard above guarantees only ONE server/worker ever runs, so even the default preset
# can never multiply into 5 orphans again. No further action needed at the script level.
