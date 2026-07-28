#!/bin/bash
# NeuralAI Stop Script
# Stops all services gracefully

echo "Stopping NeuralAI services..."

pkill -f "model_service.py" 2>/dev/null && echo "✓ Stopped model service" || echo "  Model service not running"
pkill -f "tools_service.py" 2>/dev/null && echo "✓ Stopped tools service" || echo "  Tools service not running"
pkill -f "webui_service.py" 2>/dev/null && echo "✓ Stopped webui service" || echo "  WebUI service not running"

rm -f /tmp/neuralai_*.pid 2>/dev/null

echo ""
echo "All NeuralAI services stopped."
