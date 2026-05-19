#!/bin/bash
# NeuralAI Status Checker (Unified Service)
# Shows status of the unified NeuralAI service

echo "NeuralAI Service Status (v5.1)"
echo "=============================="
echo ""

# Check unified service
echo "Unified Core Service (port 5000):"
if curl -s http://localhost:5000/api/status 2>/dev/null | python3 -m json.tool 2>/dev/null; then
    echo "  ✓ Running (Ready)"
else
    echo "  ✗ Offline"
fi
echo ""

# Show PIDs
echo "Process IDs:"
pid=$(pgrep -f "neural_core_service.py")
if [ -n "$pid" ]; then
    echo "  • Core: $pid (running)"
else
    echo "  • Core: Not found"
fi
echo ""
