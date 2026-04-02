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
refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 30, 300, 60)

st.sidebar.markdown("---")
st.sidebar.caption("ERM v8.0 • Truth Detector Dashboard")

# Sidebar explainer
st.sidebar.markdown("### How the Truth Detector works")
st.sidebar.info(
    "• Regime-aware forecasting\n"
    "• Real-time benchmark vs baselines\n"
    "• Horizon-specific confidence\n"
    "• Neighbor feedback learning\n"
    "• Live matplotlib dashboard"
)

# ===================== FETCH HELPERS =====================
@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_latest_data(url: str):
    try:
        r = requests.get(f"{url}/latest", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
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

@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_visualization(url: str, city: str):
    try:
        r = requests.get(f"{url}/visualize/{city}", timeout=15)
        r.raise_for_status()
        return r.json()["visualization"]
    except Exception as e:
        return {"error": str(e)}

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

# Global city selector (used by all tabs)
city_options = latest_df["city"].unique().tolist() if not latest_df.empty and "city" in latest_df.columns else []
selected_city = st.selectbox("🌍 Select City", city_options, index=0) if city_options else None

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

# ===================== TABS =====================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Reality vs Prediction", "📉 Error Evolution", "🔬 Diagnostics", "📊 Visualizations"])

with tab1:
    st.subheader("1. Reality vs Prediction")
    if not selected_city or latest_df.empty:
        st.info("Select a city above")
    else:
        city_data = latest_df[latest_df["city"] == selected_city].iloc[-1]
        pred_data = fetch_predict(backend_url, selected_city)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Actual Temp", f"{city_data.get('live_temp', 'N/A')}°F")
        with col2:
            st.metric("ERM 1h Prediction", f"{pred_data.get('next_predicted_1h', 'N/A') if pred_data else 'N/A'}°F")
        with col3:
            st.metric("Confidence", f"{pred_data.get('confidence_1h', 'N/A') if pred_data else 'N/A'}%")

        # Full-horizon prediction table
        if pred_data:
            horizons = ["1h", "3h", "6h", "12h", "24h"]
            preds = [pred_data.get(f"next_predicted_{h}", None) for h in horizons]
            confs = [pred_data.get(f"confidence_{h}", None) for h in horizons]
            df_pred = pd.DataFrame({"Horizon": horizons, "Prediction (°F)": preds, "Confidence (%)": confs})
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

        # Simple snapshot chart
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=[1], y=[city_data.get('live_temp', 0)], mode='markers+text', name='Actual', text="Actual", marker=dict(color="#00ff88", size=16)))
        fig.add_trace(go.Scatter(x=[2], y=[pred_data.get('next_predicted_1h', 0) if pred_data else 0], mode='markers+text', name='ERM', text="ERM", marker=dict(color="#ffaa00", size=16)))
        fig.update_layout(title="Reality vs ERM Prediction", xaxis=dict(showticklabels=False), yaxis_title="Temperature (°F)")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("2. Error Evolution")
    st.info("Historical error data (MAE, RMSE) will appear automatically after several /update cycles.")

with tab3:
    st.subheader("3. Diagnostics")
    st.info("Regime detection, neighbor feedback, and full performance metrics will populate here once the model has run for a while.")

with tab4:
    st.subheader("4. Live Visualizations (matplotlib)")
    if not selected_city:
        st.info("Select a city above")
    else:
        viz_data = fetch_visualization(backend_url, selected_city)
        if "error" in viz_data:
            st.error(viz_data["error"])
        elif "dashboard_png_base64" in viz_data:
            st.image(f"data:image/png;base64,{viz_data['dashboard_png_base64']}", use_container_width=True)
            st.caption("📊 Real-time ERM dashboard: history, regime performance, benchmark MAE, rolling confidence")
        else:
            st.warning("Visualization not available yet – run a few updates first.")

st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")

if st.button("🔄 Hard Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()
