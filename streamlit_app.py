"""
streamlit_app.py — Visa Processing Time Estimator (Frontend)
=============================================================
Run locally:
    streamlit run streamlit_app.py

The app can work in two modes:
  1. DIRECT mode  — calls predictor.py directly (no Flask needed)
  2. API mode     — calls the Flask REST API (set API_URL env var)

Environment variables
---------------------
API_URL : URL of Flask backend, e.g. http://localhost:5000
          If not set, the app uses the predictor directly.
"""

import os
import math
import json
from datetime import date, datetime, timedelta

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Visa Processing Time Estimator",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI-powered visa processing time predictor using historical CEAC data."
    },
)

# ── Mode detection ────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "").rstrip("/")
USE_API  = bool(API_URL)

if not USE_API:
    # Import predictor directly
    try:
        from predictor import get_predictor
        _predictor = get_predictor()
        DIRECT_OK = True
    except Exception as e:
        DIRECT_OK = False
        DIRECT_ERR = str(e)
else:
    DIRECT_OK = False
    DIRECT_ERR = ""


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main header */
.main-header {
    background: linear-gradient(135deg, #1a3c5e 0%, #2d6a9f 100%);
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
    color: #ffffff;
}
.main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
.main-header p  { font-size: 1rem; opacity: 0.9; margin-top: 0.4rem; }

/* Metric cards */
.metric-card {
    background: #f7faff;
    border: 1px solid #d0e3f5;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.8rem;
}
.metric-card .label  { font-size: 0.82rem; color: #555; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-card .value  { font-size: 2rem; font-weight: 700; color: #1a3c5e; line-height: 1.2; }
.metric-card .sub    { font-size: 0.82rem; color: #888; }

/* Result banner */
.result-fast   { background:#e8f5e9; border-left:5px solid #43a047; border-radius:8px; padding:1rem; }
.result-normal { background:#fff8e1; border-left:5px solid #fb8c00; border-radius:8px; padding:1rem; }
.result-slow   { background:#fce4ec; border-left:5px solid #e53935; border-radius:8px; padding:1rem; }

/* Sidebar */
.sidebar-note { font-size:0.78rem; color:#888; margin-top:0.5rem; }

/* Info boxes */
.info-box { background:#e3f2fd; border-radius:8px; padding:0.8rem 1rem; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── Helper to call backend ────────────────────────────────────────────────────

def api_predict(consulate, submit_date, case_number):
    if USE_API:
        resp = requests.post(
            f"{API_URL}/api/predict",
            json={
                "consulate":   consulate,
                "submit_date": str(submit_date),
                "case_number": case_number,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    elif DIRECT_OK:
        return _predictor.predict(
            consulate=consulate,
            submit_date=str(submit_date),
            case_number=case_number,
        )
    else:
        raise RuntimeError(f"No backend available. Error: {DIRECT_ERR}")


def api_consulates():
    if USE_API:
        resp = requests.get(f"{API_URL}/api/consulates", timeout=15)
        resp.raise_for_status()
        return resp.json()["consulates"]
    elif DIRECT_OK:
        return _predictor.list_consulates()
    else:
        return []


# ── Load consulate list (cached) ──────────────────────────────────────────────

@st.cache_data(show_spinner="Loading consulate data …", ttl=3600)
def load_consulates():
    try:
        return api_consulates()
    except Exception as e:
        st.error(f"Could not load consulates: {e}")
        return []


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_gauge(predicted, lower, upper, mean_pt, title="Processing Time"):
    """Horizontal bar gauge showing prediction vs historical mean."""
    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.set_xlim(0, max(upper * 1.15, mean_pt * 1.4))
    ax.set_ylim(-0.5, 0.5)

    # Historical range band
    hist_lo = max(0, mean_pt - 98)
    hist_hi = mean_pt + 98
    ax.barh(0, hist_hi - hist_lo, left=hist_lo, height=0.35,
            color="#bbdefb", label="Historical ± 1σ", zorder=1)

    # Confidence interval
    ax.barh(0, upper - lower, left=lower, height=0.25,
            color="#1976d2", alpha=0.45, label="Prediction CI", zorder=2)

    # Predicted point
    ax.scatter([predicted], [0], color="#e53935", s=160, zorder=5,
               label=f"Predicted: {predicted:.0f}d")

    # Mean line
    ax.axvline(mean_pt, color="#43a047", ls="--", lw=1.5,
               label=f"Consulate Mean: {mean_pt:.0f}d")

    ax.set_yticks([])
    ax.set_xlabel("Days", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(90))
    fig.tight_layout()
    return fig


def plot_consulate_breakdown(stats):
    """Stacked bar showing approval / AP / refusal rates."""
    approval = stats["approval_rate_pct"]
    ap_rate  = stats["ap_rate_pct"]
    refusal  = stats["refusal_rate_pct"]
    r221g    = stats["refusal_221g_pct"]

    labels  = ["Issued", "Admin. Processing", "Refused", "Refused 221g"]
    values  = [approval, ap_rate, refusal, r221g]
    colors  = ["#43a047", "#fb8c00", "#e53935", "#8e24aa"]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    left = 0.0
    for lbl, val, col in zip(labels, values, colors):
        ax.barh(0, val, left=left, color=col, height=0.5, label=f"{lbl} ({val:.1f}%)")
        if val > 3:
            ax.text(left + val / 2, 0, f"{val:.1f}%",
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += val

    ax.set_xlim(0, max(100, left + 5))
    ax.set_yticks([])
    ax.set_xlabel("Percentage (%)", fontsize=9)
    ax.set_title("Consulate Outcome Rates", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_month_heatmap():
    """Illustrative heatmap of typical processing load by month (seasonal)."""
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    # Relative load index (1.2 = peak, 0.9 = off-peak)
    loads  = [1.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 1.2, 1.2, 1.2]

    fig, ax = plt.subplots(figsize=(7, 1.8))
    data = np.array(loads).reshape(1, -1)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0.8, vmax=1.25)
    ax.set_xticks(range(12))
    ax.set_xticklabels(months, fontsize=9)
    ax.set_yticks([])
    ax.set_title("Seasonal Load Index (Higher = Longer Wait)", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.35,
                 label="Relative Load", fraction=0.04)
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛂 About")
    st.markdown(
        "This tool estimates the **visa processing time** for immigrant visa "
        "applications tracked via the CEAC (Consular Electronic Application Center) "
        "system, using a machine-learning model trained on historical data from "
        "FY 2020–2025."
    )
    st.markdown("---")
    st.markdown("### Model Info")
    if DIRECT_OK:
        st.markdown(f"- **Model:** `{_predictor.model_name}`")
        mae = _predictor.test_metrics.get("MAE", 0)
        r2  = _predictor.test_metrics.get("R2", 0)
        st.markdown(f"- **Test MAE:** `{mae:.1f}` days")
        st.markdown(f"- **R²:** `{r2:.4f}`")
        st.markdown(f"- **Features:** `{len(_predictor.feature_names)}`")
    else:
        st.markdown("_(Backend not connected — run the Flask API first)_")
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.markdown(
        "<span class='sidebar-note'>"
        "Predictions are probabilistic estimates based on historical patterns. "
        "Actual processing times may differ due to policy changes, case complexity, "
        "and administrative factors outside the model's training data."
        "</span>",
        unsafe_allow_html=True,
    )


# ── Main Header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
  <h1>🛂 AI Visa Processing Time Estimator</h1>
  <p>Enter your consulate and application details to get a data-driven processing time estimate.</p>
</div>
""", unsafe_allow_html=True)

# ── Load consulate data ───────────────────────────────────────────────────────

consulates_data = load_consulates()

if not consulates_data:
    st.error("⚠️ Could not load consulate data. Make sure the model files are in the `models/` folder.")
    st.stop()

consulate_options = {
    f"{c['code']} — {c['label']}": c["code"]
    for c in consulates_data
}
consulate_display_list = list(consulate_options.keys())

# ── Input Form ────────────────────────────────────────────────────────────────

st.markdown("## 📋 Application Details")

col_form, col_info = st.columns([3, 2], gap="large")

with col_form:
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("### Select Your Consulate")
        selected_display = st.selectbox(
            "Consulate / Embassy",
            options=consulate_display_list,
            index=0,
            help="Select the US consulate where your visa interview is scheduled.",
        )
        selected_code = consulate_options[selected_display]

        st.markdown("### Application Date")
        submit_date = st.date_input(
            "Visa Application Submission Date",
            value=date.today() - timedelta(days=30),
            min_value=date(2020, 1, 1),
            max_value=date(2030, 12, 31),
            help="The date on which your DS-260 was submitted or your interview was scheduled.",
        )

        st.markdown("### Case Details")
        case_number = st.number_input(
            "Case Number (within fiscal year)",
            min_value=1,
            max_value=50000,
            value=1000,
            step=100,
            help="Your sequential case number. Higher numbers may indicate later processing.",
        )

        submitted = st.form_submit_button(
            "🔮  Estimate Processing Time",
            use_container_width=True,  # still valid for form buttons
            type="primary",
        )

with col_info:
    st.markdown("### 📌 How It Works")
    st.markdown("""
1. **Select your consulate** from the dropdown — each location has unique historical patterns.
2. **Enter your submission date** — seasonal factors (peak/off-peak months) significantly affect wait times.
3. **Provide your case number** — higher numbers may indicate a later position in the queue.
4. Click **Estimate Processing Time** to get an AI-powered prediction.
""")
    st.markdown("---")
    st.markdown("### 📅 Seasonal Patterns")
    st.pyplot(plot_month_heatmap(), width="stretch")
    st.caption("October–January tend to have higher application volumes and longer waits.")


# ── Prediction Results ────────────────────────────────────────────────────────

if submitted:
    with st.spinner("Running prediction model …"):
        try:
            result = api_predict(selected_code, submit_date, case_number)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # Category banner
    cat            = result["category"]
    cat_color      = result["category_color"]
    css_class      = f"result-{cat.lower()}"
    cat_icons      = {"Fast": "🟢", "Normal": "🟡", "Slow": "🔴"}
    cat_icon       = cat_icons.get(cat, "⚪")
    predicted_days = result["predicted_days"]
    lower_bound    = result["lower_bound"]
    upper_bound    = result["upper_bound"]
    months         = result["predicted_months"]

    st.markdown(
        f"""<div class="{css_class}">
        <strong style="font-size:1.1rem">{cat_icon} Category: {cat}</strong><br>
        <span>Your application is expected to be processed in approximately 
        <strong>{predicted_days:.0f} days ({months:.1f} months)</strong>, 
        which is <strong>{cat.lower()}</strong> relative to the historical average 
        for <em>{result['consulate_label']}</em>.</span>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Main metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Predicted Time</div>
            <div class="value">{predicted_days:.0f}d</div>
            <div class="sub">{months:.1f} months</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Lower Bound (−1σ)</div>
            <div class="value">{lower_bound:.0f}d</div>
            <div class="sub">{lower_bound/30.44:.1f} months</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Upper Bound (+1σ)</div>
            <div class="value">{upper_bound:.0f}d</div>
            <div class="sub">{upper_bound/30.44:.1f} months</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        peak_label = "🔴 Peak Season" if result["is_peak_season"] else "🟢 Off-Peak"
        st.markdown(f"""<div class="metric-card">
            <div class="label">Season</div>
            <div class="value" style="font-size:1.2rem">{peak_label}</div>
            <div class="sub">{submit_date.strftime('%B %Y')}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Charts
    chart_col, stats_col = st.columns([3, 2], gap="large")

    with chart_col:
        cstats = result["consulate_stats"]
        mean_pt = cstats["mean_processing_days"]
        st.pyplot(
            plot_gauge(predicted_days, lower_bound, upper_bound, mean_pt,
                       f"Processing Time Estimate — {result['consulate_label']}"),
            width="stretch",
        )
        st.pyplot(plot_consulate_breakdown(cstats), width="stretch")

    with stats_col:
        st.markdown("### 📍 Consulate Historical Stats")
        st.markdown(f"**{result['consulate_label']}** (`{result['consulate_code']}`)")
        st.markdown("")

        metrics_table = {
            "Metric": [
                "Average Processing Time",
                "Median Processing Time",
                "Std Deviation",
                "Total Cases (Historical)",
                "Approval Rate",
                "Admin. Processing Rate",
                "Refusal Rate",
                "Refused 221g Rate",
            ],
            "Value": [
                f"{cstats['mean_processing_days']:.0f} days",
                f"{cstats['median_processing_days']:.0f} days",
                f"± {cstats['std_processing_days']:.0f} days",
                f"{cstats['total_cases']:,}",
                f"{cstats['approval_rate_pct']:.1f}%",
                f"{cstats['ap_rate_pct']:.1f}%",
                f"{cstats['refusal_rate_pct']:.1f}%",
                f"{cstats['refusal_221g_pct']:.1f}%",
            ],
        }
        st.dataframe(
            pd.DataFrame(metrics_table),
            width="stretch",
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("### 🤖 Model Info")
        st.markdown(f"- **Model:** `{result['model_name']}`")
        st.markdown(f"- **Training MAE:** `{result['model_mae_days']}` days")
        st.markdown(f"- **R²:** `{result['model_r2']}`")
        st.markdown(f"- **Features used:** `{len(result['features_used'])}`")

    # Estimated date range
    st.markdown("---")
    st.markdown("### 📅 Estimated Completion Date Range")
    est_low  = submit_date + timedelta(days=int(lower_bound))
    est_mid  = submit_date + timedelta(days=int(predicted_days))
    est_high = submit_date + timedelta(days=int(upper_bound))

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.info(f"**Optimistic (−1σ)**\n\n{est_low.strftime('%B %d, %Y')}")
    with dc2:
        st.success(f"**Most Likely**\n\n{est_mid.strftime('%B %d, %Y')}")
    with dc3:
        st.warning(f"**Conservative (+1σ)**\n\n{est_high.strftime('%B %d, %Y')}")

    # Raw JSON expander
    with st.expander("🔍 Full Prediction Details (JSON)"):
        display_result = {k: v for k, v in result.items() if k != "features_used"}
        st.json(display_result)

    # Tips
    st.markdown("---")
    st.markdown("### 💡 Tips")
    tips = []
    if result["is_peak_season"]:
        tips.append("📌 You applied during **peak season** (Oct–Jan) — consider that wait times may be longer than average.")
    if cstats["ap_rate_pct"] > 15:
        tips.append(f"⚠️ This consulate has a relatively **high administrative processing rate** ({cstats['ap_rate_pct']:.1f}%) — prepare for a possible AP hold.")
    if cstats["refusal_rate_pct"] > 20:
        tips.append(f"⚠️ This consulate's **refusal rate** ({cstats['refusal_rate_pct']:.1f}%) is notable — ensure all documentation is complete.")
    if cat == "Slow":
        tips.append("🐢 Your predicted processing time is **above average** for this consulate. Stay informed via CEAC and NVC portals.")
    if cat == "Fast":
        tips.append("🚀 Your predicted processing time is **below average** — things look favorable!")
    if not tips:
        tips.append("✅ Everything looks typical for this consulate. Monitor your case status regularly on CEAC.")
    for tip in tips:
        st.markdown(tip)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center><small>AI Visa Processing Time Estimator • Built with Streamlit & scikit-learn • "
    "Data: CEAC FY2020–FY2025</small></center>",
    unsafe_allow_html=True,
)
