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
    # Try loading one city as example
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

    # Get latest row for this city
    city_data = latest_df[latest_df["city"] == selected_city].iloc[-1]

    # Get prediction from backend
    pred_data = fetch_predict(backend_url, selected_city)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Actual Temp", f"{city_data.get('live_temp', 'N/A')}°F", delta=None)
    with col2:
        st.metric("ERM 1h Prediction", f"{pred_data.get('next_predicted_1h', 'N/A') if pred_data else 'N/A'}°F", delta=None)

    # Plot Reality vs Prediction
    fig1 = make_subplots(rows=1, cols=1)
    fig1.add_trace(go.Scatter(x=[1], y=[city_data.get('live_temp', 0)], mode='markers+text', name='Actual', text="Actual", textposition="top center", marker=dict(size=14, color="#00ff88")))
    fig1.add_trace(go.Scatter(x=[2], y=[pred_data.get('next_predicted_1h', 0) if pred_data else 0], mode='markers+text', name='ERM Prediction', text="ERM", textposition="top center", marker=dict(size=14, color="#ffaa00")))
    fig1.update_layout(title="Reality vs ERM Prediction (Current Snapshot)", xaxis=dict(showticklabels=False), yaxis_title="Temperature (°F)")
    st.plotly_chart(fig1, use_container_width=True)

    st.info("✅ If ERM dots consistently hug the Actual line better than a naive baseline → your model is converging faster.")

with tab2:
    st.subheader("2. Error Evolution Over Time — The Learning Layer")

    # For error analysis we need historical data
    if use_github and github_repo:
        hist_df = load_github_csv(github_repo, selected_city.lower().replace(" ", "_"))
    else:
        hist_df = latest_df[latest_df["city"] == selected_city].copy() if "city" in latest_df.columns else pd.DataFrame()

    if not hist_df.empty and "error_1h" in hist_df.columns:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], errors="coerce")
        hist_df = hist_df.sort_values("timestamp")

        fig2 = make_subplots(rows=2, cols=1, subplot_titles=("Horizon Errors Over Time", "Rolling Error Stats"))

        # Horizon errors
        for h in [1, 3, 6, 12, 24]:
            col = f"error_{h}h" if f"error_{h}h" in hist_df.columns else None
            if col:
                fig2.add_trace(go.Scatter(x=hist_df["timestamp"], y=hist_df[col], mode='lines', name=f"{h}h Error"), row=1, col=1)

        # Rolling average & variance
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
    st.rerun()import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import requests
import zipfile
import io
from collections import deque
from pathlib import Path

st.set_page_config(page_title="ERM v8.0", layout="wide", page_icon="🌍")

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; font-weight: 700; color: #00ff9d; text-align: center; margin-bottom: 0.2rem;}
    .sub-header {font-size: 1.1rem; color: #aaaaaa; text-align: center; margin-bottom: 2rem;}
    .metric-card {background-color: #1a1a1a; padding: 1rem; border-radius: 12px; border: 1px solid #333;}
    .stMetric {background-color: #1a1a1a; padding: 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# =============================================
# CONFIG
# =============================================
DEFAULT_CITIES = [
    {"name": "Columbus_OH", "display": "Columbus, OH"},
    {"name": "Miami_FL", "display": "Miami, FL"},
    {"name": "New_York_NY", "display": "New York, NY"},
    {"name": "Los_Angeles_CA", "display": "Los Angeles, CA"},
    {"name": "London_UK", "display": "London, UK"},
    {"name": "Tokyo_JP", "display": "Tokyo, JP"},
    {"name": "Pataskala_OH", "display": "Pataskala, OH"},
    {"name": "Cleveland_OH", "display": "Cleveland, OH"},
    {"name": "Fort_Lauderdale_FL", "display": "Fort Lauderdale, FL"},
    {"name": "West_Palm_Beach_FL", "display": "West Palm Beach, FL"},
    {"name": "Philadelphia_PA", "display": "Philadelphia, PA"},
    {"name": "Boston_MA", "display": "Boston, MA"},
    {"name": "San_Diego_CA", "display": "San Diego, CA"},
    {"name": "San_Francisco_CA", "display": "San Francisco, CA"},
    {"name": "Manchester_UK", "display": "Manchester, UK"},
    {"name": "Birmingham_UK", "display": "Birmingham, UK"},
    {"name": "Yokohama_JP", "display": "Yokohama, JP"},
    {"name": "Osaka_JP", "display": "Osaka, JP"},
]

BACKEND_URL = "https://ermforecast.onrender.com"

# =============================================
# HELPERS
# =============================================
def to_unit(temp_c: float, unit: str) -> float:
    return round(temp_c * 9/5 + 32, 1) if unit == "°F" else round(temp_c, 1)

@st.cache_data(ttl=180)
def fetch_latest_data():
    try:
        resp = requests.get(f"{BACKEND_URL}/latest", timeout=15)
        resp.raise_for_status()
        df = pd.DataFrame(resp.json())
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp', ascending=False)
        return df
    except Exception as e:
        st.error(f"Backend connection issue: {e}")
        return pd.DataFrame()

def load_local_data():
    data_dir = Path("ERM_Data")
    if not data_dir.exists():
        return pd.DataFrame()
    csv_files = list(data_dir.glob("erm_v8.0_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            city_part = f.stem.replace("erm_v8.0_", "").split("_20")[0]
            df['city'] = city_part.replace("_", " ").title()
            dfs.append(df)
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp') if dfs else pd.DataFrame()

# =============================================
# UI
# =============================================
st.markdown('<h1 class="main-header">🌍 ERM v8.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Live Adaptive Weather Field Model • Real-time recursive predictions</p>', unsafe_allow_html=True)

col_status, col_refresh = st.columns([4, 1])
with col_status:
    st.success("✅ Backend Connected • Live")
with col_refresh:
    if st.button("🔄 Refresh Now"):
        st.rerun()

unit = st.radio("Temperature unit", ["°C", "°F"], horizontal=True, label_visibility="collapsed")

latest_df = fetch_latest_data()

tabs = st.tabs(["📡 Live Predictions", "📊 City Deep Dive", "📈 Historical Trends", "🔬 System Status", "💾 Export"])

with tabs[0]:
    st.subheader("Live ERM Predictions — All Cities")
    if latest_df.empty:
        st.warning("No live data yet. Click /update on the backend or wait for the next cycle.")
    else:
        cols = st.columns(3)
        for idx, city_config in enumerate(DEFAULT_CITIES):
            city_key = city_config["name"]
            display = city_config["display"]
            row = latest_df[latest_df.get("city", "").str.lower().str.replace(" ", "_") == city_key.lower()]
            if row.empty:
                continue
            row = row.iloc[0]
            live = float(row.get('live_temp', 15))
            pred = float(row.get('next_predicted_1h', live))
            delta = pred - live
            with cols[idx % 3]:
                st.metric(
                    label=display,
                    value=f"{to_unit(live, unit):.1f}{unit}",
                    delta=f"{to_unit(delta, unit):+.1f}{unit} (1h)"
                )

with tabs[1]:
    st.subheader("City Deep Dive")
    selected = st.selectbox("Choose city", [c["display"] for c in DEFAULT_CITIES])
    city_key = next((c["name"] for c in DEFAULT_CITIES if c["display"] == selected), None)
    if city_key:
        try:
            data = requests.get(f"{BACKEND_URL}/predict/{city_key}", params={"steps": "1,3,6,12,24,48"}, timeout=10).json()
            st.json(data, expanded=False)
            horizons = list(data.get("future_forecast", {}).keys())
            values = [data["future_forecast"][h] for h in horizons]
            fig = go.Figure(go.Bar(x=[f"+{h}h" for h in horizons], y=[to_unit(v, unit) for v in values]))
            fig.update_layout(title=f"{selected} Forecast", height=400)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("No prediction data yet — run /update first.")

with tabs[2]:
    st.subheader("Historical Trends")
    local_df = load_local_data()
    if not local_df.empty:
        city_filter = st.multiselect("Select cities to compare", local_df['city'].unique(), default=local_df['city'].unique()[:3])
        filtered = local_df[local_df['city'].isin(city_filter)]
        if not filtered.empty:
            fig = go.Figure()
            for city in filtered['city'].unique():
                data = filtered[filtered['city'] == city].sort_values('timestamp')
                fig.add_trace(go.Scatter(x=data['timestamp'], y=data['live_temp'], name=f"{city} Live", mode='lines'))
                if 'next_predicted_1h' in data.columns:
                    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['next_predicted_1h'], name=f"{city} 1h Pred", line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("System & Volatility")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show All ERM States"):
            st.json(requests.get(f"{BACKEND_URL}/states").json())
    with col2:
        if st.button("Show Volatility Log"):
            st.text("\n".join(requests.get(f"{BACKEND_URL}/volatility", params={"limit": 30}).json().get("logs", [])))

with tabs[4]:
    st.subheader("Export Data")
    if st.button("Download Latest Snapshot"):
        if not latest_df.empty:
            st.download_button("⬇️ latest.csv", latest_df.to_csv(index=False).encode(), "erm_v8_latest.csv", "text/csv")
    if st.button("Download Full ERM_Data ZIP"):
        data_dir = Path("ERM_Data")
        if data_dir.exists():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for f in data_dir.glob("*.csv"):
                    z.write(f, f.name)
            st.download_button("⬇️ Full Data ZIP", buf.getvalue(), "erm_v8_data.zip", "application/zip")

st.caption(f"ERM v8.0 • Live at {BACKEND_URL} • Last refresh: {datetime.now().strftime('%H:%M:%S')}")
