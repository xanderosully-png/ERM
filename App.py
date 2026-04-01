import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import requests
import zipfile
import io
from collections import deque, defaultdict
import json
from pathlib import Path
import math
from typing import List, Dict

# =============================================
# HELPER: haversine (kept for possible future use)
# =============================================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# =============================================
# DEFAULT_CITIES — FIXED WITH REAL OHIO CITIES (Pataskala first!)
# =============================================
DEFAULT_CITIES = [
    {"name": "Pataskala", "lat": 39.9956, "lon": -82.6743, "tz": "America/New_York", "local_avg_temp": 12.5, "local_temp_range": 22.0},
    {"name": "Columbus", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 20.0},
    {"name": "Cleveland", "lat": 41.4993, "lon": -81.6944, "tz": "America/New_York", "local_avg_temp": 10.5, "local_temp_range": 19.0},
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
    csv_files = list(data_dir.glob("erm_v5.0_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['city'] = f.stem.split('_', 2)[2].replace('_', ' ')
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).sort_values('timestamp')

# =============================================
# STREAMLIT APP — NOW FULLY CONNECTED TO FASTAPI BACKEND
# =============================================
st.set_page_config(page_title="ERM V5 Forecast", layout="wide")
st.title("🌍 ERM V5 — Live Adaptive Weather Field Model")

unit = st.radio("Temperature unit", ["°C", "°F"], horizontal=True)

tab1, tab2 = st.tabs(["Live ERM Predictions", "Saved ERM_Data"])

with tab1:
    st.subheader("Live Mode — Real-time ERM Field (from FastAPI backend)")

    # Auto-refresh + manual button
    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)
    if st.button("🔄 Refresh Now"):
        st.rerun()

    if auto_refresh:
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if (datetime.now() - st.session_state.last_update).seconds > 30:
            st.rerun()

    # === FETCH LATEST DATA FROM FASTAPI /latest ENDPOINT (ONE CALL) ===
    st.caption("📡 Fetching latest ERM predictions from backend...")
    try:
        response = requests.get("https://ermforecast.onrender.com/latest", timeout=15)
        response.raise_for_status()
        latest_records = response.json()
        st.success(f"✅ Backend data loaded — {len(latest_records)} cities")
    except Exception as e:
        st.error(f"❌ Failed to fetch from FastAPI backend: {e}")
        latest_records = []
        st.stop()

    # Session state for history (still kept for nice charts)
    if 'history_live' not in st.session_state:
        st.session_state.history_live = {c['name']: deque(maxlen=48) for c in DEFAULT_CITIES}

    for city in DEFAULT_CITIES:
        name = city['name']

        st.caption(f"📍 Processing **{name}**...")

        # Find the matching record from /latest
        data = None
        for record in latest_records:
            if record.get("city") == name:
                data = record
                break

        if data is None or data.get('live_temp') is None:
            st.warning(f"⚠️ No latest data available for {name} right now")
            continue

        live_temp_c = float(data['live_temp'])

        # Pull pre-computed predictions from backend (no local ERM step needed anymore)
        next_predicted_c = float(data.get('next_predicted_1h', live_temp_c))
        pred_3h = float(data.get('next_predicted_3h', live_temp_c + 1.0))
        pred_6h = float(data.get('next_predicted_6h', live_temp_c + 2.0))

        # Store in history for charting
        st.session_state.history_live[name].append({
            "time": data.get('timestamp', datetime.now().isoformat()),
            "live": live_temp_c,
            "pred_1h": next_predicted_c,
            "pred_3h": pred_3h,
            "pred_6h": pred_6h,
        })

        st.success(f"✅ Live data received for {name} — improvement {data.get('improvement_pct', 0):.1f}%")

        # UI — metric + chart
        col1, col2 = st.columns([1, 3])
        with col1:
            current = st.session_state.history_live[name][-1] if st.session_state.history_live[name] else None
            if current:
                st.metric(
                    label=f"{name}",
                    value=f"{to_unit(current['live'], unit):.1f}{unit}",
                    delta=f"Pred 1h: {to_unit(current['pred_1h'], unit):.1f}{unit}"
                )
        with col2:
            if st.session_state.history_live[name]:
                df = pd.DataFrame(st.session_state.history_live[name])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["live"]], name="Live", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["pred_1h"]], name="1h Pred", line=dict(dash="dash", color="orange")))
                fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["pred_3h"]], name="3h Pred", line=dict(dash="dot", color="green")))
                fig.add_trace(go.Scatter(x=df["time"], y=[to_unit(t, unit) for t in df["pred_6h"]], name="6h Pred", line=dict(dash="dashdot", color="red")))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

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

st.caption("✅ ERM V5 — Now fully synchronized with FastAPI backend (V5.4-clean) • Live predictions come from the real ERM engine")
