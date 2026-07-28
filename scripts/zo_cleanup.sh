#!/bin/bash
# NeuralAI ZO Computer RAM Cleanup Script (v7.2)
# Run this on ZO Computer terminal to clean up lingering processes

echo "========================================="
echo "NeuralAI ZO Computer Cleanup"
echo "========================================="

# Show current memory usage
echo ""
echo "[1/5] Current memory usage:"
free -h 2>/dev/null || cat /proc/meminfo | head -5
echo ""

# Kill lingering find processes (known issue from previous sessions)
echo "[2/5] Killing lingering 'find' processes..."
pkill -f "^find " 2>/dev/null && echo "  ✓ find processes killed" || echo "  (none found)"

# Kill any orphaned Python processes (not our services)
echo "[3/5] Checking orphaned Python processes..."
ORPHAN_PIDS=$(ps aux | grep python | grep -v neural_core_service | grep -v neural_voice | grep -v webui_service | grep -v "grep python" | awk '{print $2}')
if [ -n "$ORPHAN_PIDS" ]; then
    echo "$ORPHAN_PIDS" | xargs kill 2>/dev/null
    echo "  ✓ Orphaned Python processes killed"
else
    echo "  (none found)"
fi

# Check current NeuralAI services
echo "[4/5] Checking NeuralAI services..."
echo "  Core service (port 5000):"
pgrep -f "neural_core_service" > /dev/null && echo "    ✓ Running" || echo "    ✗ Not running"
echo "  Voice service (port 5001):"
pgrep -f "neural_voice" > /dev/null && echo "    ✓ Running" || echo "    ✗ Not running"
echo "  llmster (port 1234):"
pgrep -f "lmstudio" > /dev/null && echo "    ✓ Running" || echo "    ✗ Not running"

# Show final memory usage
echo ""
echo "[5/5] Final memory usage:"
free -h 2>/dev/null || cat /proc/meminfo | head -5
echo ""

echo "========================================="
echo "✓ Cleanup complete"
echo "========================================="
echo ""
echo "To restart services, run:"
echo "  cd ~/NeuralAI/services && bash start_all.sh"
