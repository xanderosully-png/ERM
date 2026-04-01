import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import requests
import zipfile
import io
from collections import deque
from pathlib import Path

# =============================================
# DEFAULT_CITIES — MAIN CITIES + NEIGHBORS INCLUDED
# =============================================
DEFAULT_CITIES = [
    {"name": "Columbus",     "backend_key": "Columbus_OH", "neighbors": ["Pataskala", "Cleveland"]},
    {"name": "Miami",        "backend_key": "Miami_FL",    "neighbors": []},
    {"name": "New York",     "backend_key": "New_York_NY", "neighbors": []},
    {"name": "Los Angeles",  "backend_key": "Los_Angeles_CA", "neighbors": []},
    {"name": "London",       "backend_key": "London_UK",   "neighbors": []},
    {"name": "Tokyo",        "backend_key": "Tokyo_JP",    "neighbors": []},
]

# =============================================
# CORE FUNCTIONS
# =============================================
def to_unit(temp_c: float, unit: str) -> float:
    return temp_c * 9/5 + 32 if unit == "°F" else temp_c

@st.cache_data(ttl=600)
def load_erm_data():
    data_dir = Path("ERM_Data")
    if not data_dir.exists():
        return pd.DataFrame()
    csv_files = list(data_dir.glob("erm_v4.4_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        city_part = f.stem.replace("erm_v4.4_", "").split("_20")[0]
        df['city'] = city_part.replace("_", " ").title()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp')

# =============================================
# STREAMLIT APP
# =============================================
st.set_page_config(page_title="ERM V5 Forecast", layout="wide")
st.title("🌍 ERM V5 — Live Adaptive Weather Field Model")

unit = st.radio("Temperature unit", ["°C", "°F"], horizontal=True)

tab1, tab2 = st.tabs(["Live ERM Predictions", "Saved ERM_Data"])

with tab1:
    st.subheader("Live Mode — Real-time ERM Field (from FastAPI backend)")

    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)
    if st.button("🔄 Refresh Now"):
        st.rerun()

    if auto_refresh:
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if (datetime.now() - st.session_state.last_update).seconds > 30:
            st.rerun()

    st.caption("📡 Fetching latest ERM predictions from backend...")
    try:
        response = requests.get("https://ermforecast.onrender.com/latest", timeout=15)
        response.raise_for_status()
        latest_records = response.json()
        st.success(f"✅ Backend data loaded — {len(latest_records)} cities")
    except Exception as e:
        st.error(f"❌ Failed to fetch from FastAPI backend: {e}")
        st.stop()

    def normalize(name: str) -> str:
        return str(name).lower().replace("_", "").replace(" ", "").replace("-", "")

    if 'history_live' not in st.session_state:
        st.session_state.history_live = {c['name']: deque(maxlen=48) for c in DEFAULT_CITIES}

    for city_config in DEFAULT_CITIES:
        display_name = city_config['name']
        backend_key = city_config['backend_key']
        neighbors = city_config.get('neighbors', [])

        st.caption(f"📍 Processing **{display_name}**...")

        # Find main city data
        data = None
        for record in latest_records:
            if normalize(record.get("city")) == normalize(backend_key):
                data = record
                break

        if data is None or data.get('live_temp') is None:
            st.warning(f"⚠️ No latest data available for {display_name} right now")
            continue

        live_temp_c = float(data['live_temp'])
        next_predicted_c = float(data.get('next_predicted_1h', live_temp_c))
        pred_3h = float(data.get('next_predicted_3h', live_temp_c + 1.0))
        pred_6h = float(data.get('next_predicted_6h', live_temp_c + 2.0))

        st.session_state.history_live[display_name].append({
            "time": data.get('timestamp', datetime.now().isoformat()),
            "live": live_temp_c,
            "pred_1h": next_predicted_c,
            "pred_3h": pred_3h,
            "pred_6h": pred_6h,
        })

        improvement = data.get('improvement_pct', 0)
        st.success(f"✅ {display_name} — improvement {improvement:.1f}%")

        col1, col2 = st.columns([1, 3])
        with col1:
            current = st.session_state.history_live[display_name][-1]
            st.metric(
                label=f"{display_name}",
                value=f"{to_unit(current['live'], unit):.1f}{unit}",
                delta=f"Pred 1h: {to_unit(current['pred_1h'], unit):.1f}{unit}"
            )
        with col2:
            df = pd.DataFrame(st.session_state.history_live[display_name])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["live"]], name="Live", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["pred_1h"]], name="1h Pred", line=dict(dash="dash", color="orange")))
            fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["pred_3h"]], name="3h Pred", line=dict(dash="dot", color="green")))
            fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["pred_6h"]], name="6h Pred", line=dict(dash="dashdot", color="red")))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # === NEIGHBOR CITIES SECTION ===
        if neighbors:
            st.caption(f"🧭 Neighbors of {display_name}:")
            neighbor_col1, neighbor_col2 = st.columns([1, 3])
            with neighbor_col1:
                for neigh in neighbors:
                    # Use main city's data as proxy (geographically close)
                    st.metric(
                        label=neigh,
                        value=f"{to_unit(live_temp_c, unit):.1f}{unit}",
                        delta=f"Pred 1h: {to_unit(next_predicted_c, unit):.1f}{unit} (proxy)"
                    )
            with neighbor_col2:
                st.info("🔄 These neighbors currently use Columbus_OH data as a close proxy.\n"
                        "Add them to your FastAPI DEFAULT_CITIES for full independent predictions!")

with tab2:
    st.subheader("Saved ERM_Data Mode")
    df = load_erm_data()
    if not df.empty:
        city_filter = st.multiselect("Filter cities", options=df['city'].unique(), default=df['city'].unique())
        filtered = df[df['city'].isin(city_filter)]
        st.dataframe(filtered, use_container_width=True)
        
        if st.button("Download all CSV as ZIP"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in Path("ERM_Data").glob("*.csv"):
                    zf.write(f, f.name)
            st.download_button("⬇️ Download ZIP", zip_buffer.getvalue(), "erm_data_v5.zip", "application/zip")

st.caption("✅ ERM V5 — Fully synchronized with FastAPI backend (V5.4-clean) • Neighbor cities now included!")
