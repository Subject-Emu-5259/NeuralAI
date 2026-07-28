#!/bin/bash
# deploy_v16.sh - Deploy DPO v16 adapter to NeuralAI production on ZO Computer
# Run this on the ZO Computer (neuralai-deandrewharris.zocomputer.io)
set -e

echo "=== NeuralAI v16 Deployment ==="
echo ""

PROJ_DIR="/home/workspace/Projects/NeuralAI"
cd "$PROJ_DIR"

# Step 1: Pull latest code from GitHub
echo "[1/5] Pulling latest code from GitHub..."
git pull origin master
echo "✅ Code updated"

# Step 2: Verify adapter files exist
echo ""
echo "[2/5] Verifying v16 adapter files..."
if [ -f "checkpoints/v2_model/adapter_model.safetensors" ] && [ -f "checkpoints/v2_model/adapter_config.json" ]; then
    echo "✅ Adapter files present"
    ls -la checkpoints/v2_model/adapter_model.safetensors
    ls -la checkpoints/v2_model/adapter_config.json
else
    echo "❌ Adapter files missing! Checking for .gitignore issue..."
    git checkout -f checkpoints/v2_model/adapter_model.safetensors checkpoints/v2_model/adapter_config.json
    echo "✅ Force-restored adapter files"
fi

# Step 3: Reload the model in llmster
echo ""
echo "[3/5] Reloading adapter in llmster..."
LMS_BIN="/root/.lmstudio/bin/lms"
if [ -x "$LMS_BIN" ]; then
    # Unload current model if loaded
    "$LMS_BIN" unload 2>/dev/null || true
    sleep 2
    
    # Reload the model
    "$LMS_BIN" load smollm2-360m-instruct -y 2>&1
    echo "✅ Model reloaded"
else
    echo "⚠️  llmster not found at $LMS_BIN, skipping model reload"
fi

# Step 4: Restart NeuralAI core service
echo ""
echo "[4/5] Restarting NeuralAI core service..."
# Kill existing neural_core_service processes
pkill -f "neural_core_service" 2>/dev/null || true
sleep 2

# Start fresh
cd "$PROJ_DIR"
source .venv/bin/activate 2>/dev/null || true
nohup python services/neural_core_service.py > /tmp/neural_core_service.log 2>&1 &
echo "✅ Core service restarted (PID: $!)"

# Step 5: Verify
echo ""
echo "[5/5] Verifying deployment..."
sleep 3
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ NeuralAI service is healthy"
elif curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ NeuralAI service is healthy (port 5000)"
else
    echo "⚠️  Health check failed - check logs: tail -f /tmp/neural_core_service.log"
fi

# Test identity
echo ""
echo "=== Testing identity awareness ==="
curl -s -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "my name De'\''Andrew Harris do u know who I am"}' | python3 -m json.tool 2>/dev/null || echo "Could not reach API directly"

echo ""
echo "=== Deployment complete ==="
echo "Live at: https://neuralai-deandrewharris.zocomputer.io"
