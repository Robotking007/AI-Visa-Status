# start.sh — run both Flask API and Streamlit in one container (for single-process hosts)
#!/usr/bin/env bash
set -e

API_PORT=${PORT:-5000}
UI_PORT=${STREAMLIT_PORT:-8501}

echo "Starting Flask API on port $API_PORT ..."
gunicorn app:app \
  --workers 2 \
  --timeout 120 \
  --bind 0.0.0.0:$API_PORT \
  --access-logfile - \
  --error-logfile - \
  --daemon

echo "Starting Streamlit on port $UI_PORT ..."
streamlit run streamlit_app.py \
  --server.port=$UI_PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
