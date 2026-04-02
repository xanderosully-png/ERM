import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import numpy as np

st.set_page_config(page_title="ERM v8.0 — Truth Detector", layout="wide", page_icon="🌍")

st.sidebar.header("🔧 Truth Detector Controls")
backend_url = st.sidebar.text_input("Backend URL", value="https://ermforecast.onrender.com")
github_repo = st.sidebar.text_input("GitHub Repo (optional fallback)", value="", placeholder="username/repo")
use_github = st.sidebar.toggle("Use GitHub raw CSVs", value=False)
refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60)

st.sidebar.markdown("---")
st.sidebar.caption("ERM v8.0 • Truth Detector Dashboard")

# ===================== FETCH WITH ERROR HANDLING =====================
@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_latest_data(url: str):
    try:
        r = requests.get(f"{url}/latest", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 500:
            st.error("Backend 500 error on /latest — this is the known CSV parsing issue in main.py")
            st.info("Click the button below to trigger an update")
            return pd.DataFrame()
        st.error(f"Backend error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_predict(url: str, city: str):
    try:
        r = requests.get(f"{url}/predict/{city}", params={"steps": "1,3,6,12,24"}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ===================== MAIN HEADER =====================
st.markdown("""
<h1 style='text-align:center; color:#00ff88;'>
    🌍 ERM v8.0 — Truth Detector
</h1>
""", unsafe_allow_html=True)

# ===================== DATA =====================
latest_df = fetch_latest_data(backend_url)

if latest_df.empty:
    st.warning("No data from backend yet. Click the button below to force an update.")

# Big manual update button
if st.button("🚀 Trigger Backend /update Now", type="primary", use_container_width=True):
    try:
        r = requests.get(f"{backend_url}/update", timeout=30)
        if r.status_code == 200:
            st.success("✅ Update triggered successfully! Refreshing dashboard...")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"Update failed with status {r.status_code}")
    except Exception as e:
        st.error(f"Could not trigger update: {e}")

# ===================== TABS (defensive) =====================
tab1, tab2, tab3 = st.tabs(["📈 Reality vs Prediction", "📉 Error Evolution", "🔬 Diagnostics"])

with tab1:
    st.subheader("1. Reality vs Prediction")
    if latest_df.empty:
        st.info("Waiting for data...")
    else:
        city_options = latest_df["city"].unique().tolist() if "city" in latest_df.columns else []
        selected_city = st.selectbox("Select City", city_options, index=0)
        city_data = latest_df[latest_df["city"] == selected_city].iloc[-1]
        pred_data = fetch_predict(backend_url, selected_city)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Actual Temp", f"{city_data.get('live_temp', 'N/A')}°F")
        with col2:
            st.metric("ERM 1h Prediction", f"{pred_data.get('next_predicted_1h', 'N/A') if pred_data else 'N/A'}°F")

        # Simple snapshot chart
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=[1], y=[city_data.get('live_temp', 0)], mode='markers+text', name='Actual', text="Actual", marker=dict(color="#00ff88", size=14)))
        fig.add_trace(go.Scatter(x=[2], y=[pred_data.get('next_predicted_1h', 0) if pred_data else 0], mode='markers+text', name='ERM', text="ERM", marker=dict(color="#ffaa00", size=14)))
        fig.update_layout(title="Reality vs ERM Prediction", xaxis=dict(showticklabels=False), yaxis_title="Temperature (°F)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("2. Error Evolution")
    st.info("Historical error data appears after /update has run at least once.")

with tab3:
    st.subheader("3. Diagnostics")
    st.info("Full diagnostics will populate once data is flowing.")

st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

if st.button("🔄 Hard Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()
