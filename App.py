import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests
import zipfile
import io
from collections import deque
from pathlib import Path
import time

# =============================================
# CONFIG & DEFAULT CITIES (v8.0 SYNC)
# =============================================
st.set_page_config(page_title="ERM v8.0 Dashboard", layout="wide", page_icon="🌍")

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

CSV_PREFIX = "erm_v8.0"
BACKEND_DEFAULT = "https://ermforecast.onrender.com"   # ← Updated to your live URL

# =============================================
# HELPERS
# =============================================
def to_unit(temp_c: float, unit: str) -> float:
    return round(temp_c * 9/5 + 32, 1) if unit == "°F" else round(temp_c, 1)

@st.cache_data(ttl=300)
def fetch_latest_data(base_url: str):
    try:
        resp = requests.get(f"{base_url}/latest", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp', ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to fetch /latest: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_predict(base_url: str, city_key: str, steps: str = "1,3,6,12,24,48"):
    try:
        resp = requests.get(f"{base_url}/predict/{city_key}", params={"steps": steps, "dry_run": True}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def load_local_data():
    data_dir = Path("ERM_Data")
    if not data_dir.exists():
        return pd.DataFrame()
    csv_files = list(data_dir.glob(f"{CSV_PREFIX}_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            city_part = f.stem.replace(f"{CSV_PREFIX}_", "").split("_20")[0]
            df['city'] = city_part.replace("_", " ").title()
            dfs.append(df)
        except Exception:
            continue
    if dfs:
        return pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    return pd.DataFrame()

# =============================================
# STREAMLIT UI
# =============================================
st.title("🌍 ERM v8.0 — Live Adaptive Weather Field Model")
st.caption("Full visualization dashboard • Cross-sourced from FastAPI v8.0 endpoints + local ERM_Data")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    backend_url = st.text_input("FastAPI Backend URL", value=BACKEND_DEFAULT, help="Your live ERM v8.0 backend")
    unit = st.radio("Temperature unit", ["°C", "°F"], horizontal=True, index=0)
    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)
    st.divider()
    st.markdown("**Endpoints used**")
    st.code("/latest\n/predict/{city}\n/hotfixes\n/states\n/volatility", language="http")
    if st.button("🔄 Force Full Refresh"):
        st.rerun()

# Auto-refresh logic
if auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if (datetime.now() - st.session_state.last_refresh).seconds > 30:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

latest_df = fetch_latest_data(backend_url)

tabs = st.tabs(["📡 Live Dashboard", "📊 City Deep Dive", "📈 Historical Trends", "🔬 System & Volatility", "💾 Data Export"])

with tabs[0]:
    st.subheader("Live ERM Predictions — All Cities")
    if latest_df.empty:
        st.warning("No live data available yet. Run /update on the backend.")
    else:
        cols = st.columns(3)
        for idx, city_config in enumerate(DEFAULT_CITIES):
            city_key = city_config["name"]
            display_name = city_config["display"]
            row = latest_df[latest_df.get("city", "").str.lower().str.replace(" ", "_") == city_key.lower()]
            if row.empty:
                with cols[idx % 3]:
                    st.metric(label=display_name, value="No data", delta="—")
                continue
            row = row.iloc[0]
            live_c = float(row.get('live_temp', 15.0))
            pred_1h_c = float(row.get('next_predicted_1h', live_c))
            er = float(row.get('Er_value', 0.0))
            beta = float(row.get('beta', 0.6))
            neigh = float(row.get('neighbor_influence', 0.0))
            with cols[idx % 3]:
                st.metric(
                    label=display_name,
                    value=f"{to_unit(live_c, unit):.1f}{unit}",
                    delta=f"1h: {to_unit(pred_1h_c, unit):.1f}{unit}"
                )
                st.caption(f"Er: {er:.1f} | β: {beta:.2f} | Neighbor: {neigh:.2f}")

with tabs[1]:
    st.subheader("City Deep Dive")
    selected_city = st.selectbox("Select city", options=[c["display"] for c in DEFAULT_CITIES])
    city_key = next((c["name"] for c in DEFAULT_CITIES if c["display"] == selected_city), None)
    if city_key:
        pred_data = fetch_predict(backend_url, city_key)
        if pred_data:
            st.json(pred_data, expanded=False)
            horizons = list(pred_data.get("future_forecast", {}).keys())
            values = [pred_data["future_forecast"][h] for h in horizons]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"+{h}h" for h in horizons], y=[to_unit(v, unit) for v in values], name="Forecast"))
            fig.update_layout(title=f"{selected_city} Multi-Horizon Forecast", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Predict endpoint not responding — using latest only")

with tabs[2]:
    st.subheader("Historical Trends (Local ERM_Data)")
    local_df = load_local_data()
    if not local_df.empty:
        city_filter = st.multiselect("Filter cities", options=local_df['city'].unique(), default=local_df['city'].unique()[:3])
        filtered = local_df[local_df['city'].isin(city_filter)]
        if not filtered.empty:
            fig = go.Figure()
            for city in filtered['city'].unique():
                city_data = filtered[filtered['city'] == city].sort_values('timestamp')
                fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['live_temp'], name=f"{city} Live", mode='lines'))
                if 'next_predicted_1h' in city_data.columns:
                    fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['next_predicted_1h'], name=f"{city} 1h Pred", line=dict(dash='dash')))
            fig.update_layout(title="Temperature History + Predictions", height=500, xaxis_title="Time", yaxis_title=f"Temperature ({unit})")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered[['timestamp', 'city', 'live_temp', 'next_predicted_1h', 'Er_value', 'beta']].tail(50), use_container_width=True)

with tabs[3]:
    st.subheader("System State & Volatility")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Show All ERM States"):
            try:
                states = requests.get(f"{backend_url}/states", timeout=10).json()
                st.json(states)
            except Exception as e:
                st.error(e)
    with col_b:
        if st.button("Show Volatility Log"):
            try:
                vol = requests.get(f"{backend_url}/volatility", params={"limit": 20}, timeout=10).json()
                st.text("\n".join(vol.get("logs", ["No events yet"])))
            except Exception as e:
                st.error(e)
    st.caption("Hotfixes status")
    try:
        hf = requests.get(f"{backend_url}/hotfixes", timeout=10).json()
        st.success(f"✅ v{hf.get('version')} — Cutoffs eliminated: {hf.get('cutoffs_eliminated')}")
    except Exception:
        st.info("Hotfixes endpoint not reachable")

with tabs[4]:
    st.subheader("Export All Data")
    if st.button("Download latest snapshot as CSV"):
        if not latest_df.empty:
            csv = latest_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download latest.csv", csv, "erm_v8_latest.csv", "text/csv")
    if st.button("Download full ERM_Data as ZIP"):
        data_dir = Path("ERM_Data")
        if data_dir.exists():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in data_dir.glob("*.csv"):
                    zf.write(f, f.name)
            st.download_button("⬇️ Download full ZIP", zip_buffer.getvalue(), "erm_v8_full_data.zip", "application/zip")
        else:
            st.info("No local ERM_Data folder found")

st.caption(f"ERM v8.0 Dashboard • Backend: {backend_url} • Last refresh: {datetime.now().strftime('%H:%M:%S')}")import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests
import zipfile
import io
from collections import deque
from pathlib import Path
import time

# =============================================
# CONFIG & DEFAULT CITIES (FULL v8.0 SYNC)
# =============================================
st.set_page_config(page_title="ERM v8.0 Dashboard", layout="wide", page_icon="🌍")

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

CSV_PREFIX = "erm_v8.0"
BACKEND_DEFAULT = "https://erm-v8.onrender.com"

# =============================================
# HELPERS
# =============================================
def to_unit(temp_c: float, unit: str) -> float:
    return round(temp_c * 9/5 + 32, 1) if unit == "°F" else round(temp_c, 1)

@st.cache_data(ttl=300)
def fetch_latest_data(base_url: str):
    try:
        resp = requests.get(f"{base_url}/latest", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp', ascending=False)
        return df
    except Exception as e:
        st.error(f"Failed to fetch /latest: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_predict(base_url: str, city_key: str, steps: str = "1,3,6,12,24,48"):
    try:
        resp = requests.get(f"{base_url}/predict/{city_key}", params={"steps": steps, "dry_run": True}, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

def load_local_data():
    data_dir = Path("ERM_Data")
    if not data_dir.exists():
        return pd.DataFrame()
    csv_files = list(data_dir.glob(f"{CSV_PREFIX}_*.csv"))
    if not csv_files:
        return pd.DataFrame()
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            city_part = f.stem.replace(f"{CSV_PREFIX}_", "").split("_20")[0]
            df['city'] = city_part.replace("_", " ").title()
            dfs.append(df)
        except Exception:
            continue
    if dfs:
        return pd.concat(dfs, ignore_index=True).sort_values('timestamp')
    return pd.DataFrame()

# =============================================
# STREAMLIT UI
# =============================================
st.title("🌍 ERM v8.0 — Live Adaptive Weather Field Model")
st.caption("Full visualization dashboard • Cross-sourced from FastAPI v8.0 endpoints + local ERM_Data")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    backend_url = st.text_input("FastAPI Backend URL", value=BACKEND_DEFAULT, help="e.g. https://erm-v8.onrender.com")
    unit = st.radio("Temperature unit", ["°C", "°F"], horizontal=True, index=0)
    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)
    st.divider()
    st.markdown("**Endpoints used**")
    st.code("/latest\n/predict/{city}\n/hotfixes\n/states\n/volatility", language="http")
    if st.button("🔄 Force Full Refresh"):
        st.rerun()

# Auto-refresh logic
if auto_refresh:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if (datetime.now() - st.session_state.last_refresh).seconds > 30:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

latest_df = fetch_latest_data(backend_url)

tabs = st.tabs(["📡 Live Dashboard", "📊 City Deep Dive", "📈 Historical Trends", "🔬 System & Volatility", "💾 Data Export"])

with tabs[0]:
    st.subheader("Live ERM Predictions — All Cities")
    if latest_df.empty:
        st.warning("No live data available yet. Run /update on the backend.")
    else:
        cols = st.columns(3)
        for idx, city_config in enumerate(DEFAULT_CITIES):
            city_key = city_config["name"]
            display_name = city_config["display"]
            row = latest_df[latest_df.get("city", "").str.lower().str.replace(" ", "_") == city_key.lower()]
            if row.empty:
                with cols[idx % 3]:
                    st.metric(label=display_name, value="No data", delta="—")
                continue
            row = row.iloc[0]
            live_c = float(row.get('live_temp', 15.0))
            pred_1h_c = float(row.get('next_predicted_1h', live_c))
            er = float(row.get('Er_value', 0.0))
            beta = float(row.get('beta', 0.6))
            neigh = float(row.get('neighbor_influence', 0.0))
            vol = "N/A"
            with cols[idx % 3]:
                st.metric(
                    label=display_name,
                    value=f"{to_unit(live_c, unit):.1f}{unit}",
                    delta=f"1h: {to_unit(pred_1h_c, unit):.1f}{unit}"
                )
                st.caption(f"Er: {er:.1f} | β: {beta:.2f} | Neighbor: {neigh:.2f}")

with tabs[1]:
    st.subheader("City Deep Dive")
    selected_city = st.selectbox("Select city", options=[c["display"] for c in DEFAULT_CITIES])
    city_key = next((c["name"] for c in DEFAULT_CITIES if c["display"] == selected_city), None)
    if city_key:
        pred_data = fetch_predict(backend_url, city_key)
        if pred_data:
            st.json(pred_data, expanded=False)
            # Mini forecast chart
            horizons = list(pred_data.get("future_forecast", {}).keys())
            values = [pred_data["future_forecast"][h] for h in horizons]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"+{h}h" for h in horizons], y=[to_unit(v, unit) for v in values], name="Forecast"))
            fig.update_layout(title=f"{selected_city} Multi-Horizon Forecast", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Predict endpoint not responding — using latest only")

with tabs[2]:
    st.subheader("Historical Trends (Local ERM_Data)")
    local_df = load_local_data()
    if not local_df.empty:
        city_filter = st.multiselect("Filter cities", options=local_df['city'].unique(), default=local_df['city'].unique()[:3])
        filtered = local_df[local_df['city'].isin(city_filter)]
        if not filtered.empty:
            fig = go.Figure()
            for city in filtered['city'].unique():
                city_data = filtered[filtered['city'] == city].sort_values('timestamp')
                fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['live_temp'], name=f"{city} Live", mode='lines'))
                if 'next_predicted_1h' in city_data.columns:
                    fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['next_predicted_1h'], name=f"{city} 1h Pred", line=dict(dash='dash')))
            fig.update_layout(title="Temperature History + Predictions", height=500, xaxis_title="Time", yaxis_title=f"Temperature ({unit})")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered[['timestamp', 'city', 'live_temp', 'next_predicted_1h', 'Er_value', 'beta']].tail(50), use_container_width=True)

with tabs[3]:
    st.subheader("System State & Volatility")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Show All ERM States"):
            try:
                states = requests.get(f"{backend_url}/states", timeout=10).json()
                st.json(states)
            except Exception as e:
                st.error(e)
    with col_b:
        if st.button("Show Volatility Log"):
            try:
                vol = requests.get(f"{backend_url}/volatility", params={"limit": 20}, timeout=10).json()
                st.text("\n".join(vol.get("logs", ["No events yet"])))
            except Exception as e:
                st.error(e)
    st.caption("Hotfixes status")
    try:
        hf = requests.get(f"{backend_url}/hotfixes", timeout=10).json()
        st.success(f"✅ v{hf.get('version')} — Cutoffs eliminated: {hf.get('cutoffs_eliminated')}")
    except Exception:
        st.info("Hotfixes endpoint not reachable")

with tabs[4]:
    st.subheader("Export All Data")
    if st.button("Download latest snapshot as CSV"):
        if not latest_df.empty:
            csv = latest_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download latest.csv", csv, "erm_v8_latest.csv", "text/csv")
    if st.button("Download full ERM_Data as ZIP"):
        data_dir = Path("ERM_Data")
        if data_dir.exists():
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in data_dir.glob("*.csv"):
                    zf.write(f, f.name)
            st.download_button("⬇️ Download full ZIP", zip_buffer.getvalue(), "erm_v8_full_data.zip", "application/zip")
        else:
            st.info("No local ERM_Data folder found")

st.caption(f"ERM v8.0 Dashboard • Backend: {backend_url} • Last refresh: {datetime.now().strftime('%H:%M:%S')}")
