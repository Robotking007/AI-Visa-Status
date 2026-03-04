"""
app.py — Flask REST API for Visa Processing Time Predictor
===========================================================
Endpoints
---------
GET  /health          → liveness probe
GET  /api/consulates  → list of available consulates with stats
POST /api/predict     → predict processing time
GET  /api/model-info  → model metadata

Run locally
-----------
    python app.py

Environment variables
---------------------
PORT      : port to listen on (default 5000)
DEBUG     : "true" to enable Flask debug mode
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from predictor import get_predictor

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)  # allow Streamlit front-end to call the API cross-origin

# Warm up predictor at import time so first request is fast
_predictor = get_predictor()


# ── Health ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  _predictor.model_name,
        "consulates_loaded": len(_predictor.consulate_stats),
    })


# ── Consulates ───────────────────────────────────────────────────────────────

@app.route("/api/consulates", methods=["GET"])
def list_consulates():
    """Return all consulates the model knows about."""
    consulates = _predictor.list_consulates()
    return jsonify({"consulates": consulates, "count": len(consulates)})


# ── Predict ──────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Request body (JSON)
    -------------------
    {
        "consulate":   "KGL",
        "submit_date": "2024-06-15",
        "case_number": 1234          (optional, default 1000)
    }

    Response
    --------
    See predictor.VisaPredictor.predict() docstring.
    """
    data = request.get_json(force=True, silent=True) or {}

    # --- Validate ---
    errors = {}
    consulate = data.get("consulate", "").strip().upper()
    submit_date = data.get("submit_date", "").strip()
    case_number = data.get("case_number", 1000)

    if not consulate:
        errors["consulate"] = "Required field."
    if not submit_date:
        errors["submit_date"] = "Required field (YYYY-MM-DD)."
    else:
        try:
            from datetime import datetime
            datetime.strptime(submit_date, "%Y-%m-%d")
        except ValueError:
            errors["submit_date"] = "Must be YYYY-MM-DD format."

    try:
        case_number = int(case_number)
        if case_number < 1:
            errors["case_number"] = "Must be a positive integer."
    except (TypeError, ValueError):
        errors["case_number"] = "Must be a positive integer."

    if errors:
        return jsonify({"errors": errors}), 400

    # --- Predict ---
    try:
        result = _predictor.predict(
            consulate=consulate,
            submit_date=submit_date,
            case_number=case_number,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Model info ───────────────────────────────────────────────────────────────

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model metadata and training metrics."""
    return jsonify({
        "model_name":    _predictor.model_name,
        "n_features":    len(_predictor.feature_names),
        "feature_names": _predictor.feature_names,
        "test_metrics":  _predictor.test_metrics,
    })


# ── Sample cases endpoint (for testing) ──────────────────────────────────────

@app.route("/api/sample-cases", methods=["GET"])
def sample_cases():
    """Return a set of sample test cases for quick validation."""
    cases = [
        {"consulate": "KGL", "submit_date": "2024-06-15", "case_number": 1234},
        {"consulate": "ADD", "submit_date": "2023-10-01", "case_number": 500},
        {"consulate": "FRN", "submit_date": "2024-01-20", "case_number": 2000},
        {"consulate": "SGP", "submit_date": "2024-09-05", "case_number": 800},
        {"consulate": "LGS", "submit_date": "2023-12-25", "case_number": 300},
    ]
    results = []
    for case in cases:
        try:
            res = _predictor.predict(**case)
            results.append({
                "input":  case,
                "result": {
                    "predicted_days":   res["predicted_days"],
                    "predicted_months": res["predicted_months"],
                    "category":         res["category"],
                    "lower_bound":      res["lower_bound"],
                    "upper_bound":      res["upper_bound"],
                },
            })
        except Exception as exc:
            results.append({"input": case, "error": str(exc)})
    return jsonify({"sample_cases": results})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"Starting Flask API on port {port}  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
