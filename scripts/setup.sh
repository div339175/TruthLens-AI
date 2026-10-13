#!/usr/bin/env bash
set -e

echo "============================================="
echo " TruthLens AI - Setup Environment"
echo "============================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[1/4] Checking Python 3.11 virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3.11 -m venv .venv || python3 -m venv .venv
fi

echo "[2/4] Installing backend & ML dependencies..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r backend/requirements.txt

echo "[3/4] Initializing sample development dataset..."
.venv/bin/python scripts/download_dataset.py --sample
.venv/bin/python scripts/prepare_data.py

echo "[4/4] Installing frontend npm packages..."
cd "$PROJECT_DIR/frontend"
npm install

echo "============================================="
echo " Setup complete!"
echo " Run backend:  .venv/bin/uvicorn backend.app.main:app --reload --port 8000"
echo " Run frontend: cd frontend && npm run dev"
echo "============================================="
