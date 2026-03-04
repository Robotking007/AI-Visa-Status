# ── Dockerfile ────────────────────────────────────────────────────────────────# ── Dockerfile ────────────────────────────────────────────────────────────────



















































CMD ["bash", "start.sh"]# Runs both Flask (daemonized) and Streamlit (foreground)# ── Default command ───────────────────────────────────────────────────────────  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \# ── Health check ──────────────────────────────────────────────────────────────EXPOSE 5000 8501# ── Expose ports ──────────────────────────────────────────────────────────────USER appuserRUN useradd -m appuser && chown -R appuser /app# Create a non-root user for securityCOPY data/FY2025-ceac-2025-10-01.csv   data/COPY data/FY2024-ceac-2024-10-01.csv   data/COPY data/FY2023-ceac-2023-06-24.csv   data/COPY data/FY2022-ceac-current.csv      data/COPY data/FY2021-ceac-current.csv      data/COPY data/FY2020-ceac-current.csv      data/COPY data/processed_visa_dataset.csv   data/COPY data/engineered_visa_dataset.csv  data/COPY models/       models/COPY predictor.py  app.py  streamlit_app.py  start.sh  ./# ── Copy application source ───────────────────────────────────────────────────    pip install --no-cache-dir -r requirements.txtRUN pip install --upgrade pip && \# Upgrade pip first to avoid resolver issuesCOPY requirements.txt .# ── Install Python dependencies ───────────────────────────────────────────────WORKDIR /app    && rm -rf /var/lib/apt/lists/*    libgomp1 \    build-essential \RUN apt-get update && apt-get install -y --no-install-recommends \# System dependenciesFROM python:3.10-slim AS base# ─────────────────────────────────────────────────────────────────────────────# Run:     docker run -p 5000:5000 -p 8501:8501 visa-estimator# Build:   docker build -t visa-estimator .##   - Streamlit frontend      → port 8501#   - Flask API  (gunicorn)   → port 5000# Multi-stage build that produces a slim production image containing:# Multi-stage build for the Visa Processing Time Estimator
#
# Stage 1 – base runtime
# Stage 2 – dependency installation  
# Stage 3 – application image
#
# Build:
#   docker build -t visa-estimator .
#
# Run (both services via start.sh):
#   docker run -p 5000:5000 -p 8501:8501 visa-estimator
#
# Run (Flask API only):
#   docker run -p 5000:5000 -e PORT=5000 visa-estimator gunicorn app:app --bind 0.0.0.0:5000
#
# Run (Streamlit only, with direct predictor):
#   docker run -p 8501:8501 visa-estimator streamlit run streamlit_app.py --server.port 8501
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim AS base

# System deps for LightGBM / XGBoost / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY predictor.py   ./
COPY app.py         ./
COPY streamlit_app.py ./
COPY start.sh       ./
COPY models/        ./models/

# Copy only the CEAC data files needed for label-encoder reconstruction
# (the full engineered dataset is large; copy only what predictor.py needs)
COPY data/FY2020-ceac-current.csv   ./data/
COPY data/FY2021-ceac-current.csv   ./data/
COPY data/FY2022-ceac-current.csv   ./data/
COPY data/FY2023-ceac-2023-06-24.csv ./data/
COPY data/FY2024-ceac-2024-10-01.csv ./data/
COPY data/FY2025-ceac-2025-10-01.csv ./data/
COPY data/engineered_visa_dataset.csv ./data/

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PORT=5000 \
    STREAMLIT_PORT=8501 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose Flask API + Streamlit ports
EXPOSE 5000 8501

RUN chmod +x start.sh

# Default: launch both services
CMD ["bash", "start.sh"]
