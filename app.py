import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime
from typing import Optional

st.set_page_config(
    page_title="ERM v10.1 — Truth Detector",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# ===================== SESSION STATE =====================
if "unit" not in st.session_state:
    st.session_state.unit = "F"
if "selected_city" not in st.session_state:
    st.session_state.selected_city = None

st.sidebar.header("🔧 Truth Detector Controls")

unit_choice = st.sidebar.radio(
    "Temperature Unit",
    options=["Fahrenheit (°F)", "Celsius (°C)"],
    index=0 if st.session_state.unit == "F" else 1,
    horizontal=True
)
st.session_state.unit = "F" if "F" in unit_choice else "C"

def convert_temp(c: Optional[float]) -> str:
    if c is None:
        return "N/A"
    if st.session_state.unit == "F":
        return f"{round((c * 9/5) + 32, 1)}"
    return f"{round(c, 1)}"

def unit_symbol() -> str:
    return "°F" if st.session_state.unit == "F" else "°C"

# ===================== BACKEND URL FROM SECRETS =====================
try:
    backend_url = st.secrets["BACKEND_URL"]
except Exception:
    backend_url = st.sidebar.text_input(
        "Backend URL",
        value="https://erm-live.onrender.com",
        help="Fallback — update .streamlit/secrets.toml for production"
    )

refresh_interval = st.sidebar.slider(
    "Auto-refresh interval (seconds)",
    min_value=30,
    max_value=300,
    value=60,
    step=15
)

st.sidebar.markdown("---")
st.sidebar.caption("ERM v10.1 • Self-Evolving Truth Detector")

# ===================== DEFAULT CITIES =====================
DEFAULT_CITIES = [
    "Columbus_OH", "Miami_FL", "New_York_NY", "Los_Angeles_CA", "London_UK",
    "Tokyo_JP", "Pataskala_OH", "Cleveland_OH", "Fort_Lauderdale_FL",
    "West_Palm_Beach_FL", "Philadelphia_PA", "Boston_MA", "San_Diego_CA",
    "San_Francisco_CA", "Manchester_UK", "Birmingham_UK", "Yokohama_JP", "Osaka_JP"
]

# ===================== FETCH HELPERS =====================
@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_latest(url: str, city: str):
    try:
        r = requests.get(f"{url}/latest/{city}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_predict(url: str, city: str):
    try:
        r = requests.get(f"{url}/predict/{city}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=refresh_interval, show_spinner=False)
def fetch_visualization(url: str, city: str):
    try:
        r = requests.get(f"{url}/visualize/{city}", timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return f"<h3 style='color:#ffaa00'>Could not load chart (backend may be waking up): {e}</h3>"

# ===================== HEADER =====================
st.markdown("""
<h1 style='text-align:center; color:#00ff88; margin-bottom:0;'>
    🌍 ERM v10.1 — Truth Detector
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
    latest = fetch_latest(backend_url, selected_city)
    pred_data = fetch_predict(backend_url, selected_city)

    col1, col2, col3 = st.columns(3)
    with col1:
        current_temp = latest.get("current_temp") if latest else None
        st.metric("Current Actual Temperature", f"{convert_temp(current_temp)} {unit_symbol()}")
    with col2:
        one_hour_pred = pred_data.get("predictions", {}).get("1h") if pred_data and "predictions" in pred_data else None
        st.metric("ERM 1‑Hour Prediction", f"{convert_temp(one_hour_pred)} {unit_symbol()}")
    with col3:
        confidence = pred_data.get("confidence", "N/A") if pred_data else "N/A"
        st.metric("Prediction Confidence", f"{confidence}%")

    if pred_data and "predictions" in pred_data:
        horizons = ["1h", "3h", "6h", "12h", "24h"]
        preds = [pred_data["predictions"].get(h) for h in horizons]
        preds_display = [convert_temp(p) for p in preds]
        confs = [confidence] * len(horizons)

        df_pred = pd.DataFrame({
            "Horizon": horizons,
            f"Prediction ({unit_symbol()})": preds_display,
            "Confidence (%)": confs
        })
        st.dataframe(df_pred, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("2. Error Evolution")
    st.info("Historical error trends will appear after multiple update cycles.")

with tab3:
    st.subheader("3. Diagnostics")
    if latest:
        st.metric("Current Regime", latest.get("current_regime", "N/A"))
        st.metric("Performance Score", f"{latest.get('performance_score', 'N/A')}")
    else:
        st.info("No data yet — run an update first.")

with tab4:
    st.subheader("4. Live Visualizations (Plotly)")
    if selected_city:
        viz_html = fetch_visualization(backend_url, selected_city)
        # Fixed: use st.iframe instead of deprecated st.components.v1.html
        st.iframe(viz_html, height=700, scrolling=True)
    else:
        st.info("Select a city above")

with tab5:
    st.subheader("5. Benchmarks & Self-Evolution")
    st.info("Benchmark data appears after several update cycles.")

# ===================== FOOTER =====================
st.caption(
    f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} | "
    f"Backend: v10.1 | Unit: {unit_symbol()} | "
    f"Connected to: {backend_url}"
)

if st.button("🔄 Hard Refresh Dashboard", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
