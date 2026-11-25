#!/bin/bash
set -e

# Render Deployment Startup Script for NIDS
# Starts FastAPI directly on Render's external $PORT

echo "=== Initializing Network Intrusion Detection System ==="

# Set PYTHONPATH to project root so all submodules resolve cleanly
export PYTHONPATH=".:${PYTHONPATH}"

# Initialize database schema and verify trained model
python -c "import database; database.init_db()"

# Safety check: train baseline model if missing
if [ ! -f "models/svm_model.pkl" ]; then
    echo "[Notice] Model file not found. Generating baseline model..."
    python dataset_utility.py
    python train_model.py
fi

# Determine external port assigned by Render (defaults to 10000 if not set)
PORT="${PORT:-10000}"
echo "Starting FastAPI Production Application on 0.0.0.0:${PORT}..."

exec uvicorn backend.app.main:app \
    --host "0.0.0.0" \
    --port "${PORT}" \
    --workers 1 \
    --proxy-headers \
    --forwarded-allow-ips "*"
