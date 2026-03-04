# ── Dockerfile ────────────────────────────────────────────────────────────────
# NOTE: The trained model binary (best_model.joblib, ~837 MB) is NOT stored in
# git and is NOT copied into this image. Provide it at runtime via ONE of:
#
#   Option A — Docker volume / bind mount (recommended for local/self-hosted):
#     docker run -v /host/path/to/models:/app/models -p 5000:5000 -p 8501:8501 visa-estimator
#
#   Option B — Environment variable download (recommended for cloud):
#     docker run -e MODEL_URL="https://your-storage/best_model.joblib" ...
#     The app will auto-download via download_model.py on first start.
#
#   Option C — Azure File Share / AWS EFS mounted at /app/models
#
# Build:   docker build -t visa-estimator .
# Run:     docker run -p 5000:5000 -p 8501:8501 -v $(pwd)/models:/app/models visa-estimator
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source (NO model binary) ─────────────────────────────────
COPY predictor.py app.py streamlit_app.py download_model.py start.sh ./
COPY models/feature_names.json   models/
COPY models/training_report.json models/

# CEAC data files (needed for label-encoder reconstruction ~few MB each)
COPY data/FY2020-ceac-current.csv      data/
COPY data/FY2021-ceac-current.csv      data/
COPY data/FY2022-ceac-current.csv      data/
COPY data/FY2023-ceac-2023-06-24.csv   data/
COPY data/FY2024-ceac-2024-10-01.csv   data/
COPY data/FY2025-ceac-2025-10-01.csv   data/
COPY data/engineered_visa_dataset.csv  data/

# models/ dir exists but the .joblib is injected at runtime
RUN mkdir -p models

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PORT=5000 \
    STREAMLIT_PORT=8501 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
# MODEL_URL: set to auto-download model on first start
# e.g. -e MODEL_URL="https://myblob.blob.core.windows.net/models/best_model.joblib?sv=..."

# ── Expose ports ──────────────────────────────────────────────────────────────
EXPOSE 5000 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

RUN chmod +x start.sh

# Default: download model if needed, then start both services
CMD ["bash", "start.sh"]
