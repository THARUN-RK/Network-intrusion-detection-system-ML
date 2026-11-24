#!/bin/bash
set -e

# Render Deployment Startup Script for NIDS
# Starts FastAPI on internal port 8000 and Streamlit on Render's external $PORT

echo "=== Initializing Network Intrusion Detection System ==="

# Initialize database schema and verify trained model
python -c "import database; database.init_db()"

# Safety check: train baseline model if missing
if [ ! -f "models/svm_model.pkl" ]; then
    echo "[Notice] Model file not found. Generating baseline model..."
    python dataset_utility.py
    python train_model.py
fi

# Start FastAPI backend in background on internal port 8000
echo "[1/2] Starting FastAPI REST Backend on http://127.0.0.1:8000..."
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 &

# Wait 3 seconds for backend to start up
sleep 3

# Determine external port assigned by Render (defaults to 10000 if not set)
PORT="${PORT:-10000}"
echo "[2/2] Starting Streamlit SOC Dashboard on external port ${PORT}..."

# Launch Streamlit with headless settings for cloud hosting
exec streamlit run frontend/dashboard.py \
    --server.port "${PORT}" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
