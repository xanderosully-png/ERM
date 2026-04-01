import streamlit as st
import numpy as np
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from collections import deque
from io import StringIO, BytesIO
import zipfile
import re

VERSION = "4.4"
GITHUB_REPO = "xanderosully-png/ERM"

DEFAULT_CITIES = [ ... ]  # (unchanged — your list)

class ERM_Live_Adaptive:
    # (Your full upgraded class with dynamic field scaling, self-scaling bias, regime detection, etc. — unchanged from last version)
    # ... paste your current ERM_Live_Adaptive class here ...

def fetch_data(lat, lon, tz):
    """Robust fetch with logging and safe key handling."""
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": tz
    }
    for attempt in range(3):
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            current = data.get("current", {})
            daily = data.get("daily", {})
            return {
                'temp': current.get('temperature_2m'),
                'humidity': current.get('relative_humidity_2m'),
                'wind': current.get('wind_speed_10m'),
                'pressure': current.get('surface_pressure'),
                'time': datetime.now().isoformat(),
                'tomorrow_max': daily.get("temperature_2m_max", [None, None])[1],
                'tomorrow_min': daily.get("temperature_2m_min", [None, None])[1]
            }
        except Exception as e:
            print(f"⚠️ fetch_data attempt {attempt+1}/3 failed for ({lat}, {lon}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # exponential backoff
    print(f"❌ fetch_data failed after 3 attempts for ({lat}, {lon})")
    return None

# (normalize_city_key, load_erm_data, to_unit — unchanged except for the robust regex you already have)

# ====================== STREAMLIT UI ======================
st.set_page_config(page_title=f"ERM v{VERSION} Live", page_icon="🌡️", layout="wide")
st.title("🌍 ERM v4.4 — Live Adaptive Weather Predictor + Saved ERM_Data")

# ... (banner, sidebar, session_state setup — unchanged) ...

if mode == "Live":
    cols = st.columns(min(len(active_cities), 4))
    for idx, city in enumerate(active_cities):
        name = city["name"]
        data = fetch_data(city["lat"], city["lon"], city["tz"])
        if data:
            # ... (error feedback, baseline, step call — unchanged) ...

            # Richer live chart: Live + 1h + 3h + 6h
            if st.session_state.history_live[name]:
                df = pd.DataFrame(st.session_state.history_live[name])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["time"], y=df["live"], name="Live", line=dict(color="#1f77b4"),
                                        hovertemplate="Live: %{y}°<br>%{x}<extra></extra>"))
                fig.add_trace(go.Scatter(x=df["time"], y=df["pred_1h"], name="1h Pred", line=dict(dash="dash"),
                                        hovertemplate="1h Pred: %{y}°<br>%{x}<extra></extra>"))
                # Add 3h and 6h if you store them in history_live (optional extension)
                fig.update_layout(height=220, margin=dict(l=0,r=0,t=0,b=0), showlegend=True)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_live_{name}")

        else:
            with cols[idx % len(cols)]:
                st.error(f"❌ {name} — API unavailable")

else:  # Saved ERM_Data mode
    # ... (existing saved mode with MAE/RMSE table + error trend chart — unchanged) ...

# ... (downloads, auto-refresh, caption — unchanged) ...

st.caption("🚀 ERM v4.4 • Retry logging + richer live chart + dynamic bias tied to prediction error")
