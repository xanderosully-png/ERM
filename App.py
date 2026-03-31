import streamlit as st
import numpy as np
import requests
import csv
import time
import logging
import json
from datetime import datetime
from collections import deque
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from io import StringIO, BytesIO
import zipfile

# ====================== ORIGINAL ERM ENGINE (unchanged) ======================
VERSION = "4.4"

DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York",
     "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL",    "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York",
     "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York",
     "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles",
     "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK",   "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London",
     "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP",    "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo",
     "local_avg_temp": 16.0, "local_temp_range": 28.0},
]

def load_cities() -> list:
    return [c.copy() for c in DEFAULT_CITIES]  # you can add cities.json later if you want

class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 10, gamma: float = 0.935,
                 lambda_damp: float = 0.28, alpha: float = 0.75):
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self.humidity_history: deque = deque(maxlen=history_size)
        self.wind_history: deque = deque(maxlen=history_size)
        self.pressure_history: deque = deque(maxlen=history_size)
        self.Er_history: deque = deque(maxlen=history_size)
        self.gamma = gamma
        self.lambda_damp = lambda_damp
        self.alpha = alpha

    def _derive_variables(self, current_temp, current_humidity, current_wind,
                          current_pressure, previous_temp, hour_of_day,
                          local_avg_temp, local_temp_range):
        self.history.append(current_temp)
        self.humidity_history.append(current_humidity)
        self.wind_history.append(current_wind)
        self.pressure_history.append(current_pressure)

        if len(self.history) < 2:
            return 4.0, 0.85, 0.0, 1.0, 1.05, 0.97

        recent_t = np.array(self.history, dtype=np.float32)
        diffs = np.diff(recent_t)
        Nr = len(recent_t) * (1 + np.var(recent_t) / 10)
        Tr = max(0.6, 1 - np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6)) * (1 - np.mean(self.humidity_history) / 200)
        dphi = current_temp - previous_temp if previous_temp is not None else 0.0
        k = 0.8 + np.mean(np.abs(diffs)) / 5 + np.mean(self.wind_history) / 50
        rhoE = 1.0 + ((np.mean(recent_t) - local_avg_temp) / local_temp_range) + (np.mean(self.pressure_history) - 1013) / 1000
        tauE = 0.95 + (hour_of_day / 48)
        return Nr, Tr, dphi, k, rhoE, tauE

    def step(self, current_temp, current_humidity, current_wind, current_pressure,
             previous_temp, hour_of_day, local_avg_temp, local_temp_range):
        Nr, Tr, dphi, k, rhoE, tauE = self._derive_variables(
            current_temp, current_humidity, current_wind, current_pressure,
            previous_temp, hour_of_day, local_avg_temp, local_temp_range)

        base = (Nr * Tr * dphi) / max(k, 1e-8)
        f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)

        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            recursive = np.sum(decayed) ** self.alpha

        Er_new = f_field + (self.lambda_damp * recursive)
        Er_new = np.clip(Er_new, -200, 200)
        self.Er_history.append(Er_new)

        beta_raw = np.std(self.history) / (np.std(self.Er_history) + 1e-6)
        beta = np.clip(beta_raw, max(0.01, np.std(self.history)/50), max(1.0, np.std(self.history)/2))

        next_predicted = current_temp + (Er_new * beta)
        return Er_new, next_predicted, beta

    def predict_future(self, steps_list=[1, 3, 6, 12, 24]):
        if len(self.Er_history) < 3:
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}
        x = np.arange(len(self.Er_history), dtype=np.float32)
        y = np.array(self.Er_history, dtype=np.float32)
        slope, intercept = np.polyfit(x, y, 1)
        return {s: float(np.clip(slope * (len(x) + s) + intercept, -200, 200)) for s in steps_list}

def fetch_data(lat, lon, tz):
    url = (f"https://api.open-meteo.com/v1/current?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
           f"&timezone={tz.replace('/', '%2F')}")
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()['current']
        return {
            'temp': data['temperature_2m'],
            'humidity': data['relative_humidity_2m'],
            'wind': data['wind_speed_10m'],
            'pressure': data['surface_pressure'],
            'time': datetime.now().isoformat()
        }
    except Exception:
        return None

# ====================== STREAMLIT UI ======================
st.set_page_config(page_title=f"ERM v{VERSION} Live Dashboard", page_icon="🌡️", layout="wide")
st.title(f"🌍 ERM v{VERSION} — Live Adaptive Weather Predictor")
st.caption("Real-time Open-Meteo + ERM engine • Works on phone & desktop")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    available = load_cities()
    selected_cities = st.multiselect(
        "Choose cities to monitor",
        options=[c["name"] for c in available],
        default=[c["name"] for c in available[:3]]
    )
    update_interval = st.slider("Update interval (minutes)", 1, 60, 5)
    auto_refresh = st.toggle("Auto-refresh live", value=True)

    if st.button("🔄 Update Now", type="primary", use_container_width=True):
        st.session_state["force_update"] = True

# Session state
if "erms" not in st.session_state:
    st.session_state.erms = {name: ERM_Live_Adaptive() for name in selected_cities}
    st.session_state.previous = {name: None for name in selected_cities}
    st.session_state.history = {name: [] for name in selected_cities}  # for charts

# Filter to only selected cities
active_cities = [c for c in available if c["name"] in selected_cities]

# Main dashboard
cols = st.columns(len(active_cities) if len(active_cities) <= 4 else 4)

for idx, city in enumerate(active_cities):
    name = city["name"]
    data = fetch_data(city["lat"], city["lon"], city["tz"])

    if data:
        live_temp = data["temp"]
        hour = datetime.now().hour
        erm = st.session_state.erms[name]
        prev = st.session_state.previous[name]

        if prev:
            erm_err = abs(live_temp - prev["next_predicted"])
            baseline_err = abs(live_temp - prev["live_temp"])
            improvement = 100 * (baseline_err - erm_err) / max(baseline_err, 0.01)
        else:
            improvement = 0.0

        Er_flux, next_predicted, beta = erm.step(
            live_temp, data["humidity"], data["wind"], data["pressure"],
            prev["live_temp"] if prev else None, hour,
            city["local_avg_temp"], city["local_temp_range"]
        )

        future = erm.predict_future()
        st.session_state.history[name].append({
            "time": datetime.now(),
            "live": live_temp,
            "pred_1h": next_predicted + future[1] * beta,
            "pred_3h": next_predicted + future[3] * beta
        })
        if len(st.session_state.history[name]) > 20:
            st.session_state.history[name] = st.session_state.history[name][-20:]

        st.session_state.previous[name] = {"live_temp": live_temp, "next_predicted": next_predicted}

        # Fancy card
        with cols[idx % len(cols)]:
            st.subheader(f"📍 {name.replace('_', ' ')}")
            st.metric("Current Temp", f"{live_temp:.1f}°C", f"β={beta:.3f}")
            st.metric("Next 1h", f"{next_predicted + future[1]*beta:.1f}°C", f"Imp: {improvement:.1f}%")
            st.metric("Next 3h", f"{next_predicted + future[3]*beta:.1f}°C")
            st.metric("Next 6h", f"{next_predicted + future[6]*beta:.1f}°C")

            # Small chart
            if st.session_state.history[name]:
                df = pd.DataFrame(st.session_state.history[name])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["time"], y=df["live"], name="Live", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df["time"], y=df["pred_1h"], name="Pred 1h", line=dict(dash="dash")))
                fig.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    else:
        with cols[idx % len(cols)]:
            st.error(f"{name} — API unavailable")

# Log & downloads
with st.expander("📋 Full Log & CSV Downloads"):
    st.info("Last update: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if st.button("Download all CSVs as ZIP"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name in selected_cities:
                if st.session_state.history.get(name):
                    df = pd.DataFrame(st.session_state.history[name])
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr(f"erm_{name.lower()}.csv", csv_buffer.getvalue())
        st.download_button("⬇️ Download ZIP", zip_buffer.getvalue(), "ERM_data.zip", "application/zip")

st.caption("Built with your exact ERM v4.4 engine • Auto-refreshes every selected interval")

# Auto-refresh
if auto_refresh and "last_update" in st.session_state and (time.time() - st.session_state.last_update) > update_interval * 60:
    st.rerun()
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()