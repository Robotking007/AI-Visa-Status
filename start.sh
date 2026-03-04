#!/usr/bin/env bash
# start.sh — run both Flask API and Streamlit in one container
set -e

API_PORT=${PORT:-5000}
UI_PORT=${STREAMLIT_PORT:-8501}

# ── Download model if missing and MODEL_URL is set ────────────────────────────
if [ ! -f "models/best_model.joblib" ]; then
  if [ -n "$MODEL_URL" ]; then
    echo "[start.sh] Downloading model from MODEL_URL ..."
    python download_model.py
  else
    echo "[start.sh] WARNING: models/best_model.joblib not found and MODEL_URL is not set."
    echo "[start.sh] Mount the model file or set MODEL_URL to a download URL."
    echo "[start.sh] Starting anyway — API /health will work but /api/predict will fail."
  fi
fi

echo "[start.sh] Starting Flask API on port $API_PORT ..."
gunicorn app:app \
  --workers 2 \
  --timeout 120 \
  --bind 0.0.0.0:$API_PORT \
  --access-logfile - \
  --error-logfile - \
  --daemon

echo "[start.sh] Starting Streamlit on port $UI_PORT ..."
exec streamlit run streamlit_app.py \
  --server.port=$UI_PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
