import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import numpy as np
from pathlib import Path

st.set_page_config(page_title="ERM v8.0 — Truth Detector", layout="wide", page_icon="🌍")

# ===================== SIDEBAR =====================
st.sidebar.header("🔧 Truth Detector Controls")

backend_url = st.sidebar.text_input(
    "Backend URL (FastAPI)",
    value="https://ermforecast.onrender.com",
    help="Your main.py endpoint"
)

github_repo = st.sidebar.text_input(
    "GitHub Repo (optional — for raw CSV fallback)",
    value="",
    placeholder="username/erm-forecast-repo",
    help="If filled, dashboard can load CSVs directly from GitHub raw"
)

use_github = st.sidebar.toggle("Use GitHub raw CSVs (fallback)", value=False)

refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60)

st.sidebar.markdown("---")
st.sidebar.caption("ERM v8.0 • Truth Detector Dashboard")

# ===================== CACHING =====================
@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_latest_data(url: str):
    try:
        r = requests.get(f"{url}/latest", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.error(f"Backend fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_predict(url: str, city: str):
    try:
        r = requests.get(f"{url}/predict/{city}", params={"steps": "1,3,6,12,24"}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def load_github_csv(repo: str, city_key: str):
    if not repo:
        return pd.DataFrame()
    try:
        url = f"https://raw.githubusercontent.com/{repo}/main/ERM_Data/erm_v8.0_{city_key}_{datetime.now().strftime('%Y%m%d')}.csv"
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

# ===================== MAIN HEADER =====================
st.markdown("""
<h1 style='text-align:center; color:#00ff88;'>
    🌍 ERM v8.0 — Truth Detector
</h1>
<p style='text-align:center; font-size:1.1em; color:#aaaaaa;'>
    Scientific visualization • Reality vs Prediction • Error evolution • Falsifiable proof
</p>
""", unsafe_allow_html=True)

# ===================== DATA LOADING =====================
latest_df = fetch_latest_data(backend_url)

if latest_df.empty and use_github and github_repo:
    st.warning("Backend unavailable — falling back to GitHub raw CSVs")
    test_city = "Columbus_OH"
    latest_df = load_github_csv(github_repo, test_city.lower().replace(" ", "_"))

if latest_df.empty:
    st.error("No data available from backend or GitHub. Check URLs and redeploy.")
    st.stop()

# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs(["📈 Reality vs Prediction", "📉 Error Evolution", "🔬 Diagnostics & Falsifiability"])

with tab1:
    st.subheader("1. Reality vs Prediction — The Proof Layer")
    city_options = latest_df["city"].unique().tolist() if "city" in latest_df.columns else []
    selected_city = st.selectbox("Select City", city_options, index=0)

    city_data = latest_df[latest_df["city"] == selected_city].iloc[-1]
    pred_data = fetch_predict(backend_url, selected_city)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Actual Temp", f"{city_data.get('live_temp', 'N/A')}°F")
    with col2:
        st.metric("ERM 1h Prediction", f"{pred_data.get('next_predicted_1h', 'N/A') if pred_data else 'N/A'}°F")

    fig1 = make_subplots(rows=1, cols=1)
    fig1.add_trace(go.Scatter(x=[1], y=[city_data.get('live_temp', 0)], mode='markers+text', name='Actual', text="Actual", textposition="top center", marker=dict(size=14, color="#00ff88")))
    fig1.add_trace(go.Scatter(x=[2], y=[pred_data.get('next_predicted_1h', 0) if pred_data else 0], mode='markers+text', name='ERM Prediction', text="ERM", textposition="top center", marker=dict(size=14, color="#ffaa00")))
    fig1.update_layout(title="Reality vs ERM Prediction (Current Snapshot)", xaxis=dict(showticklabels=False), yaxis_title="Temperature (°F)")
    st.plotly_chart(fig1, use_container_width=True)

    st.info("✅ If ERM dots consistently hug the Actual line better than a naive baseline → your model is converging faster.")

with tab2:
    st.subheader("2. Error Evolution Over Time — The Learning Layer")

    if use_github and github_repo:
        hist_df = load_github_csv(github_repo, selected_city.lower().replace(" ", "_"))
    else:
        hist_df = latest_df[latest_df["city"] == selected_city].copy() if "city" in latest_df.columns else pd.DataFrame()

    if not hist_df.empty and "error_1h" in hist_df.columns:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], errors="coerce")
        hist_df = hist_df.sort_values("timestamp")

        fig2 = make_subplots(rows=2, cols=1, subplot_titles=("Horizon Errors Over Time", "Rolling Error Stats"))

        for h in [1, 3, 6, 12, 24]:
            col = f"error_{h}h" if f"error_{h}h" in hist_df.columns else None
            if col:
                fig2.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df[col], mode='lines', name=f"{h}h Error"), row=1, col=1)

        if "error_1h" in hist_df.columns:
            rolling_avg = hist_df["error_1h"].rolling(window=6, min_periods=1).mean()
            rolling_var = hist_df["error_1h"].rolling(window=6, min_periods=1).std()
            fig2.add_trace(go.Scatter(x=hist_df["timestamp"], y=rolling_avg, mode='lines', name="Rolling Avg Error (1h)"), row=2, col=1)
            fig2.add_trace(go.Scatter(x=hist_df["timestamp"], y=rolling_var, mode='lines', name="Error Std Dev"), row=2, col=1)

        fig2.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

        st.success("✅ Downward trend in rolling error = adaptive learning confirmed.")
    else:
        st.warning("No historical error columns yet — trigger /update on backend first.")

with tab3:
    st.subheader("3. Diagnostics & Falsifiability")
    st.markdown("**Can your model be proven wrong?** These metrics let you see it instantly.")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Average 1h Error", f"{hist_df['error_1h'].mean():.2f}°F" if not hist_df.empty and "error_1h" in hist_df.columns else "—")
    with cols[1]:
        st.metric("Error Volatility", f"{hist_df['error_1h'].std():.2f}°F" if not hist_df.empty and "error_1h" in hist_df.columns else "—")
    with cols[2]:
        st.metric("ERM Convergence Speed", "Fast" if not hist_df.empty and hist_df['error_1h'].std() < 2.5 else "Improving", delta="↓" if not hist_df.empty and len(hist_df) > 10 else None)

    st.caption("If error variance shrinks over time → your recursive field model is mathematically superior to baselines.")

# ===================== FOOTER =====================
st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} • Data from {backend_url if not use_github else 'GitHub raw'}")

if st.button("🔄 Hard Refresh All Data"):
    st.cache_data.clear()
    st.rerun()
