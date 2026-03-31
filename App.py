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
    def __init__(self):
        self.history: deque = deque(maxlen=10)
        self.humidity_history: deque = deque(maxlen=10)
        self.wind_history: deque = deque(maxlen=10)
        self.pressure_history: deque = deque(maxlen=10)
        self.Er_history: deque = deque(maxlen=10)
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
    # Current + daily forecast for tomorrow
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure&daily=temperature_2m_max,temperature_2m_min&timezone={tz.replace('/', '%2F')}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data["current"]
        daily = data["daily"]
        tomorrow_max = daily["temperature_2m_max"][1]   # index 1 = tomorrow
        tomorrow_min = daily["temperature_2m_min"][1]
        return {
            'temp': current['temperature_2m'],
            'humidity': current['relative_humidity_2m'],
            'wind': current['wind_speed_10m'],
            'pressure': current['surface_pressure'],
            'time': datetime.now().isoformat(),
            'tomorrow_max': tomorrow_max,
            'tomorrow_min': tomorrow_min
        }
    except Exception:
        return None

# ====================== STREAMLIT UI ======================
st.set_page_config(page_title=f"ERM v{VERSION} Live", page_icon="🌡️", layout="wide")
st.title("🌍 ERM v4.4 — Live Adaptive Weather Predictor + Next-Day Forecast")

with st.sidebar:
    st.header("Controls")
    available = [c["name"] for c in DEFAULT_CITIES]
    selected = st.multiselect("Cities", available, default=["Columbus_OH"])
    unit = st.radio("Temperature unit", ["°F", "°C"], index=0, horizontal=True)
    interval_min = st.slider("Update every (minutes)", 1, 60, 5)
    auto_refresh = st.toggle("Auto-refresh", value=True)

    if st.button("🔄 Update Now", type="primary", use_container_width=True):
        st.session_state.force_update = True

def to_unit(temp_c):
    return round(temp_c * 9/5 + 32, 1) if unit == "°F" else round(temp_c, 1)

if "erms" not in st.session_state:
    st.session_state.erms = {name: ERM_Live_Adaptive() for name in selected}
    st.session_state.previous = {name: None for name in selected}
    st.session_state.history = {name: [] for name in selected}

active_cities = [c for c in DEFAULT_CITIES if c["name"] in selected]

cols = st.columns(min(len(active_cities), 4))

for idx, city in enumerate(active_cities):
    name = city["name"]
    data = fetch_data(city["lat"], city["lon"], city["tz"])

    if data:
        live_temp_c = data["temp"]
        hour = datetime.now().hour
        erm = st.session_state.erms[name]
        prev = st.session_state.previous.get(name)

        if prev:
            erm_err = abs(live_temp_c - prev["next_predicted"])
            baseline_err = abs(live_temp_c - prev["live_temp"])
            improvement = 100 * (baseline_err - erm_err) / max(baseline_err, 0.01)
        else:
            improvement = 0.0

        Er_flux, next_predicted_c, beta = erm.step(
            live_temp_c, data["humidity"], data["wind"], data["pressure"],
            prev["live_temp"] if prev else None, hour,
            city["local_avg_temp"], city["local_temp_range"]
        )

        future = erm.predict_future([1, 3, 6, 12, 24, 48])  # added 48h for tomorrow
        live_f = to_unit(live_temp_c)
        pred_1h = to_unit(next_predicted_c + future[1] * beta)
        pred_3h = to_unit(next_predicted_c + future[3] * beta)
        pred_6h = to_unit(next_predicted_c + future[6] * beta)
        pred_tomorrow = to_unit(next_predicted_c + future[48] * beta)  # next-day prediction

        st.session_state.history[name].append({"time": datetime.now(), "live": live_f, "pred_1h": pred_1h})
        if len(st.session_state.history[name]) > 20:
            st.session_state.history[name] = st.session_state.history[name][-20:]

        st.session_state.previous[name] = {"live_temp": live_temp_c, "next_predicted": next_predicted_c}

        with cols[idx % len(cols)]:
            st.subheader(f"📍 {name.replace('_', ' ')}")
            st.metric("Current", f"{live_f}°{unit[-1]}", f"β={beta:.3f}")
            st.metric("Next 1h", f"{pred_1h}°{unit[-1]}", f"Imp: {improvement:.1f}%")
            st.metric("Next 3h", f"{pred_3h}°{unit[-1]}")
            st.metric("Next 6h", f"{pred_6h}°{unit[-1]}")
            st.metric("🌅 Tomorrow (ERM)", f"{pred_tomorrow}°{unit[-1]}")
            st.caption(f"Open-Meteo daily: {to_unit(data['tomorrow_max'])}° / {to_unit(data['tomorrow_min'])}°")

            if st.session_state.history[name]:
                df = pd.DataFrame(st.session_state.history[name])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["time"], y=df["live"], name="Live", line=dict(color="#1f77b4")))
                fig.add_trace(go.Scatter(x=df["time"], y=df["pred_1h"], name="1h Pred", line=dict(dash="dash")))
                fig.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{name}")

    else:
        with cols[idx % len(cols)]:
            st.error(f"❌ {name} — API unavailable")

with st.expander("📥 Downloads & Log"):
    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if st.button("Download all data as ZIP"):
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name in selected:
                if st.session_state.history.get(name):
                    df = pd.DataFrame(st.session_state.history[name])
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
                    zf.writestr(f"erm_{name.lower()}.csv", csv_buffer.getvalue())
        st.download_button("⬇️ Download ZIP", zip_buffer.getvalue(), "ERM_data.zip", "application/zip")

if auto_refresh:
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    if time.time() - st.session_state.last_update > interval_min * 60 or st.session_state.get("force_update"):
        st.session_state.last_update = time.time()
        st.session_state.force_update = False
        st.rerun()

st.caption("🚀 ERM v4.4 with next-day predictions • Fully mobile-friendly")
