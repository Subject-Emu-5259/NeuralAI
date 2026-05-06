#!/bin/bash
# NeuralAI Status Checker
# Shows status of all services

echo "NeuralAI Service Status"
echo "======================="
echo ""

# Check model service
echo "Model Service (port 7001):"
if curl -s http://localhost:7001/health 2>/dev/null | python3 -m json.tool 2>/dev/null; then
    echo "  ✓ Running"
else
    echo "  ✗ Offline"
fi
echo ""

# Check tools service
echo "Tools Service (port 7002):"
if curl -s http://localhost:7002/health 2>/dev/null | python3 -m json.tool 2>/dev/null; then
    echo "  ✓ Running"
else
    echo "  ✗ Offline"
fi
echo ""

# Check webui service
echo "WebUI Service (port 5000):"
if curl -s http://localhost:5000/api/health 2>/dev/null | python3 -m json.tool 2>/dev/null; then
    echo "  ✓ Running"
else
    echo "  ✗ Offline"
fi
echo ""

# Show PIDs if available
echo "Process IDs:"
for svc in model tools webui; do
    pid_file="/tmp/neuralai_${svc}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "  • $svc: $pid (running)"
        else
            echo "  • $svc: $pid (dead)"
        fi
    fi
done
