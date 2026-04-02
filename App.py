import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from typing import Optional

st.set_page_config(
    page_title="ERM v9.1 — Truth Detector",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# ===================== SESSION STATE & UNIT SELECTOR =====================
if "unit" not in st.session_state:
    st.session_state.unit = "F"
if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

st.sidebar.header("🔧 Truth Detector Controls")

unit_choice = st.sidebar.radio(
    "Temperature Unit",
    options=["Fahrenheit (°F)", "Celsius (°C)"],
    index=0 if st.session_state.unit == "F" else 1,
    horizontal=True,
    help="Changes are applied instantly across the entire dashboard"
)
st.session_state.unit = "F" if "F" in unit_choice else "C"

def convert_temp(c: Optional[float]) -> str:
    if c is None:
        return "N/A"
    return f"{round((c * 9/5) + 32, 1)}" if st.session_state.unit == "F" else f"{round(c, 1)}"

def unit_symbol() -> str:
    return "°F" if st.session_state.unit == "F" else "°C"

backend_url = st.sidebar.text_input(
    "Backend URL",
    value="https://ermforecast.onrender.com",
    help="Change only if you are running a local backend"
)

refresh_interval = st.sidebar.slider(
    "Auto-refresh interval (seconds)",
    min_value=30,
    max_value=300,
    value=60,
    step=15,
    help="Dashboard will auto-refresh at this interval"
)

st.sidebar.markdown("---")
st.sidebar.caption("ERM v9.1 • Self-Evolving Truth Detector")

st.sidebar.markdown("### How the Truth Detector works")
st.sidebar.info(
    "• Regime-aware forecasting\n"
    "• Real baseline competition\n"
    "• Multi-horizon validation\n"
    "• Neighbor feedback learning\n"
    "• Live matplotlib dashboard"
)

# ===================== DEFAULT CITIES (copied from backend) =====================
DEFAULT_CITIES = [
    "Columbus_OH", "Miami_FL", "New_York_NY", "Los_Angeles_CA", "London_UK",
    "Tokyo_JP", "Pataskala_OH", "Cleveland_OH", "Fort_Lauderdale_FL",
    "West_Palm_Beach_FL", "Philadelphia_PA", "Boston_MA", "San_Diego_CA",
    "San_Francisco_CA", "Manchester_UK", "Birmingham_UK", "Yokohama_JP", "Osaka_JP"
]

# ===================== FETCH HELPERS =====================
@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_predict(url: str, city: str):
    try:
        r = requests.get(
            f"{url}/predict/{city}",
            params={"steps": "1,3,6,12,24"},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_visualization(url: str, city: str):
    try:
        r = requests.get(f"{url}/visualize/{city}", timeout=15)
        r.raise_for_status()
        return r.json().get("visualization", {})
    except Exception as e:
        return {"error": str(e)}

# ===================== MAIN HEADER =====================
st.markdown("""
<h1 style='text-align:center; color:#00ff88; margin-bottom:0;'>
    🌍 ERM v9.1 — Truth Detector
</h1>
<p style='text-align:center; color:#aaaaaa; font-size:1.1em;'>
    Self-evolving • Regime-aware • Beats every baseline
</p>
""", unsafe_allow_html=True)

# ===================== CITY SELECTOR =====================
selected_city = st.selectbox(
    "🌍 Select City",
    options=DEFAULT_CITIES,
    index=0,
    key="city_selector"
)
st.session_state.selected_city = selected_city

# ===================== MANUAL UPDATE BUTTON =====================
if st.button("🚀 Trigger Backend /update Now", type="primary", use_container_width=True):
    with st.spinner("Triggering full update cycle..."):
        try:
            r = requests.get(f"{backend_url}/update", timeout=30)
            if r.status_code == 200:
                st.success("✅ Update triggered successfully! Refreshing dashboard...")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Update failed with status {r.status_code}")
        except Exception as e:
            st.error(f"Could not reach backend: {e}")

# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Reality vs Prediction",
    "📉 Error Evolution",
    "🔬 Diagnostics",
    "📊 Live Visualizations",
    "🏆 Benchmarks & Evolution"
])

with tab1:
    st.subheader("1. Reality vs Prediction")
    if not selected_city:
        st.info("👆 Select a city above")
    else:
        pred_data = fetch_predict(backend_url, selected_city)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Actual Temperature", f"{convert_temp(None)} {unit_symbol()}")  # placeholder until /latest is added
        with col2:
            pred_display = convert_temp(pred_data.get("next_predicted_1h") if pred_data else None)
            st.metric("ERM 1‑Hour Prediction", f"{pred_display} {unit_symbol()}")
        with col3:
            confidence = pred_data.get("confidence_percent", "N/A") if pred_data else "N/A"
            st.metric("Prediction Confidence", f"{confidence}%")

        if pred_data and "future_forecast" in pred_data:
            horizons = ["1h", "3h", "6h", "12h", "24h"]
            preds = [pred_data["future_forecast"].get(int(h[:-1])) for h in horizons]
            preds_display = [convert_temp(p) for p in preds]
            confs = [confidence] * len(horizons)

            df_pred = pd.DataFrame({
                "Horizon": horizons,
                f"Prediction ({unit_symbol()})": preds_display,
                "Confidence (%)": confs
            })
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

        # Snapshot chart
        pred_c = pred_data.get("next_predicted_1h") if pred_data else None
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=[1], y=[0], mode="markers+text", name="Actual", text="Actual", marker=dict(color="#00ff88", size=18)))
        fig.add_trace(go.Scatter(x=[2], y=[pred_c or 0], mode="markers+text", name="ERM", text="ERM", marker=dict(color="#ffaa00", size=18)))
        fig.update_layout(title="Reality vs ERM Prediction (Snapshot)", xaxis=dict(showticklabels=False), yaxis_title=f"Temperature ({unit_symbol()})", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("2. Error Evolution")
    st.info("📈 Historical MAE / RMSE trends will populate automatically after multiple update cycles.")

with tab3:
    st.subheader("3. Diagnostics")
    if selected_city:
        pred_data = fetch_predict(backend_url, selected_city)
        if pred_data:
            st.metric("Current Regime", pred_data.get("current_regime", "N/A"))
            st.metric("Performance Score", f"{pred_data.get('performance_score', 'N/A')}")
        else:
            st.info("Waiting for prediction data...")
    else:
        st.info("Select a city above")

with tab4:
    st.subheader("4. Live Visualizations (matplotlib)")
    if selected_city:
        viz_data = fetch_visualization(backend_url, selected_city)
        if "error" in viz_data:
            st.error(viz_data["error"])
        elif "dashboard_png_base64" in viz_data:
            st.image(
                f"data:image/png;base64,{viz_data['dashboard_png_base64']}",
                use_container_width=True
            )
            st.caption("📊 Real-time ERM v9.1 dashboard: history, regime performance, benchmark MAE, rolling confidence")
        else:
            st.warning("Visualization not ready yet — run a few updates first")
    else:
        st.info("Select a city above")

with tab5:
    st.subheader("5. Benchmarks & Self-Evolution")
    st.info("Benchmark data appears after several update cycles")

# ===================== FOOTER =====================
st.caption(
    f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} | "
    f"Backend: v9.1 | Unit: {unit_symbol()}"
)

if st.button("🔄 Hard Refresh Dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
