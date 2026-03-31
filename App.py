import streamlit as st
import numpy as np
import requests
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import deque
from io import StringIO, BytesIO
import zipfile
from pathlib import Path

VERSION = "4.4"

DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL",    "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK",   "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP",    "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 28.0},
]

class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self.humidity_history: deque = deque(maxlen=history_size)
        self.wind_history: deque = deque(maxlen=history_size)
        self.pressure_history: deque = deque(maxlen=history_size)
        self.Er_history: deque = deque(maxlen=history_size)
        self.gamma = 0.935
        self.lambda_damp = 0.28
        self.alpha = 0.75

    def step(self, current_temp, current_humidity, current_wind, current_pressure,
             previous_temp, hour_of_day, local_avg_temp, local_temp_range):
        self.history.append(current_temp)
        self.humidity_history.append(current_humidity)
        self.wind_history.append(current_wind)
        self.pressure_history.append(current_pressure)

        if len(self.history) < 2:
            return 0.0, current_temp, 1.0

        recent_t = np.array(self.history, dtype=np.float32)
        diffs = np.diff(recent_t)
        Nr = len(recent_t) * (1 + np.var(recent_t) / 10)
        Tr = max(0.6, 1 - np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6)) * (1 - np.mean(self.humidity_history) / 200)
        dphi = current_temp - previous_temp if previous_temp is not None else 0.0
        k = 0.8 + np.mean(np.abs(diffs)) / 5 + np.mean(self.wind_history) / 50
        rhoE = 1.0 + ((np.mean(recent_t) - local_avg_temp) / local_temp_range) + (np.mean(self.pressure_history) - 1013) / 1000
        tauE = 0.95 + (hour_of_day / 48)

        base = (Nr * Tr * dphi) / max(k, 1e-8)
        f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)

        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            recursive = np.sum(decayed) ** self.alpha

        Er_new = np.clip(f_field + (self.lambda_damp * recursive), -200, 200)
        self.Er_history.append(Er_new)

        beta = np.clip(np.std(self.history) / (np.std(self.Er_history) + 1e-6),
                       max(0.01, np.std(self.history)/50),
                       max(1.0, np.std(self.history)/2))

        next_predicted = current_temp + (Er_new * beta)
        return Er_new, next_predicted, beta

    def predict_future(self, steps_list=[1, 3, 6, 12, 24, 48]):
        if len(self.Er_history) < 3:
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}
        x = np.arange(len(self.Er_history), dtype=np.float32)
        y = np.array(self.Er_history, dtype=np.float32)
        slope, intercept = np.polyfit(x, y, 1)
        return {s: float(np.clip(slope * (len(x) + s) + intercept, -200, 200)) for s in steps_list}

def fetch_data(lat, lon, tz):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure&daily=temperature_2m_max,temperature_2m_min&timezone={tz.replace('/', '%2F')}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data["current"]
        daily = data["daily"]
        return {
            'temp': current['temperature_2m'],
            'humidity': current['relative_humidity_2m'],
            'wind': current['wind_speed_10m'],
            'pressure': current['surface_pressure'],
            'time': datetime.now().isoformat(),
            'tomorrow_max': daily["temperature_2m_max"][1],
            'tomorrow_min': daily["temperature_2m_min"][1]
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def load_erm_data(github_repo: str):
    data_dir_url = f"https://api.github.com/repos/{github_repo}/contents/ERM_Data"
    try:
        resp = requests.get(data_dir_url)
        resp.raise_for_status()
        files = resp.json()
        city_data = {}
        for file in files:
            if file["type"] == "file" and file["name"].endswith(".csv") and file["name"].startswith("erm_v"):
                raw_url = file["download_url"]
                df = pd.read_csv(raw_url)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                stem = file["name"].replace(".csv", "")
                parts = stem.split("_")
                city_key = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]
                if city_key not in city_data:
                    city_data[city_key] = []
                city_data[city_key].append(df)
        for city in city_data:
            city_data[city] = pd.concat(city_data[city], ignore_index=True).drop_duplicates(subset='timestamp').sort_values('timestamp')
        return city_data
    except Exception:
        return {}

st.set_page_config(page_title=f"ERM v{VERSION} Live", page_icon="🌡️", layout="wide")
st.title("🌍 ERM v4.4 — Live Adaptive Weather Predictor + Saved ERM_Data")

with st.sidebar:
    st.header("Controls")
    github_repo = st.text_input("GitHub Repo (username/repo)", value="YOURUSERNAME/YOURREPO", help="Enter your exact GitHub repo so Saved mode can load ERM_Data")
    
    available = [c["name"] for c in DEFAULT_CITIES]
    selected = st.multiselect("Cities (Live mode)", available, default=["Columbus_OH"])

    mode = st.radio("Mode", ["Live", "Saved ERM_Data"], horizontal=True, index=0)

    if mode == "Live":
        unit = st.radio("Temperature unit", ["°F", "°C"], index=0, horizontal=True)
        interval_min = st.slider("Update every (minutes)", 1, 60, 5)
        auto_refresh = st.toggle("Auto-refresh", value=True)
        if st.button("🔄 Update Now", type="primary", use_container_width=True):
            st.session_state.force_update =
