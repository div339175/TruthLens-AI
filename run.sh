#!/usr/bin/env bash
# TruthLens AI — Unified Startup Script
# Runs both FastAPI Backend (port 8000) and React Frontend (port 5173)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "          Starting TruthLens AI System                    "
echo "=========================================================="

# Trap exits to kill background jobs cleanly
cleanup() {
    echo ""
    echo "Stopping TruthLens AI servers..."
    kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. Start Backend
echo "-> Starting FastAPI Backend on http://localhost:8000 ..."
"$PROJECT_ROOT/.venv/bin/uvicorn" backend.app.main:app --host 0.0.0.0 --port 8000 --reload &

# 2. Give backend 2 seconds to initialize
sleep 2

# 3. Start Frontend
echo "-> Starting React Frontend on http://localhost:5173 ..."
cd "$PROJECT_ROOT/frontend"
npm run dev &

echo ""
echo "=========================================================="
echo " TruthLens AI is LIVE:"
echo "   • Web Application:  http://localhost:5173"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • Backend Health:   http://localhost:8000/health"
echo "=========================================================="
echo " Press Ctrl+C anytime to stop both servers."
echo ""

# Keep running
wait
