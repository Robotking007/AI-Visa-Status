# ── Heroku Procfile ───────────────────────────────────────────────────────────
# Two dynos:
#   web     → Flask REST API (serves /api/* routes)
#   worker  → Streamlit UI (on port $PORT+1, proxied via Heroku)
#
# For a single-dyno setup (free tier), use only the `web` line and point
# STREAMLIT_SERVER_PORT to the $PORT variable — run both processes via a
# start script instead.
web: gunicorn app:app --workers 2 --timeout 120 --bind 0.0.0.0:$PORT
