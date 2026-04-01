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
# MISSING HELPER: haversine (used for neighbor influence)
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
# DEFAULT CITIES — REPLACE WITH YOUR RECORDER'S EXACT LIST
# =============================================
DEFAULT_CITIES = [
    # ←←← PASTE YOUR FULL CITY LIST FROM THE RECORDER SCRIPT HERE ←←←
    # Example format (remove or replace):
    # {"name": "Columbus", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 20.0},
]

# =============================================
# V5 ERM_Live_Adaptive — EXACT MATCH TO RECORDER
# =============================================
class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self.humidity_history: deque = deque(maxlen=history_size)
        self.wind_history: deque = deque(maxlen=history_size)
        self.pressure_history: deque = deque(maxlen=history_size)
        self.Er_history: deque = deque(maxlen=history_size)
        self.error_history: deque = deque(maxlen=20)
        self.systematic_bias: deque = deque(maxlen=20)
        self.noise_error: deque = deque(maxlen=20)
        self.bias_offset = 0.0
        self.hourly_bias = defaultdict(float)
        self.gamma = 0.935
        self.lambda_damp = 0.28
        self.alpha = 0.75

    def save_state(self, filepath: str):
        state = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, deque):
                state[k] = list(v)
            elif isinstance(v, defaultdict):
                state[k] = dict(v)          # convert for JSON
            else:
                state[k] = v
        state["last_update_timestamp"] = datetime.now().isoformat()
        Path(filepath).write_text(json.dumps(state))

    def load_state(self, filepath: str):
        if Path(filepath).exists():
            try:
                state = json.loads(Path(filepath).read_text())
                for k, v in state.items():
                    if k in self.__dict__:
                        if isinstance(self.__dict__[k], deque):
                            self.__dict__[k].extend(v)
                        elif k == "hourly_bias" and isinstance(v, dict):
                            self.hourly_bias = defaultdict(float, v)
                        else:
                            self.__dict__[k] = v
            except Exception as e:
                st.warning(f"Failed to load state {filepath}: {e}")

    def record_error(self, realized_error: float, predicted: float):
        error = realized_error
        systematic = np.mean(self.error_history) if self.error_history else 0.0
        noise = error - systematic
        self.error_history.append(error)
        self.systematic_bias.append(systematic)
        self.noise_error.append(noise)

    def step(self, current_temp, current_humidity, current_wind, current_pressure,
             current_rain_prob, current_cloud_cover, current_solar,
             previous_temp, hour_of_day, local_avg_temp, local_temp_range,
             neighbor_influence: float = 0.0, city_name: str = "Unknown"):
        self.history.append(current_temp)
        self.humidity_history.append(current_humidity)
        self.wind_history.append(current_wind)
        self.pressure_history.append(current_pressure)

        if len(self.history) < 2:
            return 0.0, current_temp, 0.5, 0.0

        recent_t = np.array(self.history, dtype=np.float32)
        diffs = np.diff(recent_t)
        Nr = len(recent_t) * (1 + np.var(recent_t) / 10)
        Tr = max(0.6, 1 - np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6)) * (1 - np.mean(self.humidity_history) / 200)

        recent_error = 0.0
        if self.error_history:
            ewma_alpha = 0.3
            recent_error = self.error_history[-1]
            for e in reversed(list(self.error_history)[-10:]):
                recent_error = ewma_alpha * e + (1 - ewma_alpha) * recent_error

        error_factor = np.tanh(abs(recent_error) / 5.0)
        learning_rate = 0.05 + 0.25 * error_factor

        if len(self.error_history) > 1 and np.sign(self.error_history[-1]) != np.sign(self.error_history[-2]):
            learning_rate *= 0.5

        volatility = np.std(self.history) if len(self.history) > 1 else 0.0

        if volatility > 3.0:
            self.alpha = 0.65
            self.gamma = 0.92
            learning_rate *= 1.5
        else:
            self.alpha = 0.75
            self.gamma = 0.935

        Tr = Tr * (1 - 0.3 * error_factor)
        correction = learning_rate * recent_error

        pressure_trend = np.mean(np.diff(self.pressure_history)) if len(self.pressure_history) > 1 else 0.0
        dphi = np.mean(diffs) if len(diffs) > 0 else (current_temp - previous_temp if previous_temp is not None else 0.0)
        dphi += pressure_trend * 0.3

        k = 0.8 + np.mean(np.abs(diffs)) / 5 + np.mean(self.wind_history) / 50
        rhoE = 1.0 + ((np.mean(recent_t) - local_avg_temp) / local_temp_range) + (np.mean(self.pressure_history) - 1013) / 1000
        tauE = 0.95 + (hour_of_day / 48)

        base = (Nr * Tr * dphi) / max(k, 1e-8)
        f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)

        f_field += neighbor_influence * 0.4

        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            recursive = np.sum(decayed) ** self.alpha

        field_limit = max(50, np.std(self.history) * 3) if len(self.history) > 1 else 200
        Er_new = np.clip(f_field + (self.lambda_damp * recursive) + correction, -field_limit, field_limit)
        self.Er_history.append(Er_new)

        beta = np.clip(np.std(self.history) / (np.std(self.Er_history) + 1e-6),
                       max(0.01, np.std(self.history)/50),
                       max(1.0, np.std(self.history)/2))
        beta = beta * (1 - 0.2 * error_factor)

        decay_rate = 0.995 if volatility < 3.0 else 0.99
        self.bias_offset *= decay_rate
        self.bias_offset += learning_rate * recent_error * 0.08
        bias_limit = max(2.0, np.std(self.history) * 1.5) if len(self.history) > 1 else 5.0
        self.bias_offset = np.clip(self.bias_offset, -bias_limit, bias_limit)

        next_predicted = current_temp + (Er_new * beta) + self.bias_offset + self.hourly_bias[hour_of_day]
        next_predicted = np.clip(next_predicted, current_temp - 50, current_temp + 50)

        return Er_new, next_predicted, beta, pressure_trend

    def predict_future(self, steps_list: List[int] = [1, 3, 6, 12, 24, 48]) -> Dict[int, float]:
        if len(self.Er_history) < 5:
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}
        try:
            x = np.arange(len(self.Er_history), dtype=np.float32)
            y = np.array(self.Er_history, dtype=np.float32)
            slope, intercept = np.polyfit(x, y, 1)
            return {s: float(np.clip(slope * (len(x) + s) + intercept, -200, 200)) for s in steps_list}
        except Exception:
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}

# =============================================
# CORE FUNCTIONS FROM ORIGINAL APP.PY
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

def fetch_multi_variable_data(city):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": city['lat'],
        "longitude": city['lon'],
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation_probability,cloud_cover,shortwave_radiation",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": city['tz']
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data.get('current', {})
        daily = data.get('daily', {})
        return {
            'temp': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'wind': current.get('wind_speed_10m'),
            'pressure': current.get('surface_pressure'),
            'rain_prob': current.get('precipitation_probability'),
            'cloud_cover': current.get('cloud_cover'),
            'solar': current.get('shortwave_radiation'),
            'time': datetime.now().isoformat(),
            'tomorrow_max': daily.get('temperature_2m_max', [None, None])[1],
            'tomorrow_min': daily.get('temperature_2m_min', [None, None])[1]
        }
    except Exception:
        return None

# =============================================
# STREAMLIT APP — ALL BUGS FIXED
# =============================================
st.set_page_config(page_title="ERM V5 Forecast", layout="wide")
st.title("🌍 ERM V5 — Live Adaptive Weather Field Model")

unit = st.radio("Temperature unit", ["°C", "°F"], horizontal=True)

tab1, tab2 = st.tabs(["Live ERM Predictions", "Saved ERM_Data"])

with tab1:
    st.subheader("Live Mode — Real-time ERM Field")

    if not DEFAULT_CITIES:
        st.error("DEFAULT_CITIES list is empty. Paste your city list from the recorder script above.")
        st.stop()

    if 'history_live' not in st.session_state:
        st.session_state.history_live = {c['name']: deque(maxlen=48) for c in DEFAULT_CITIES}
    if 'previous_live' not in st.session_state:
        st.session_state.previous_live = {}
    if 'erms' not in st.session_state:
        st.session_state.erms = {c['name']: ERM_Live_Adaptive() for c in DEFAULT_CITIES}

    auto_refresh = st.toggle("Auto-refresh every 30 seconds", value=True)
    if auto_refresh:
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if (datetime.now() - st.session_state.last_update).seconds > 30:
            st.rerun()

    for city in DEFAULT_CITIES:
        name = city['name']
        erm = st.session_state.erms[name]

        try:
            data = fetch_multi_variable_data(city)
            if not data or data['temp'] is None:
                continue

            live_temp_c = data['temp']
            hour_of_day = datetime.now().hour

            # V5 neighbor influence (exact match to recorder)
            neighbor_influence = 0.0
            decay_factor = 1000.0
            neighbor_distances = [haversine(city['lat'], city['lon'], other['lat'], other['lon'])
                                  for other in DEFAULT_CITIES if other['name'] != name]
            avg_dist = np.mean(neighbor_distances) or 1.0
            for other in DEFAULT_CITIES:
                if other['name'] == name:
                    continue
                dist = haversine(city['lat'], city['lon'], other['lat'], other['lon'])
                pressure_diff = data.get('pressure', 1013) - 1013
                raw = (pressure_diff / max(dist, 0.1)) * 0.1
                normalized = raw / avg_dist
                influence = normalized * math.exp(-dist / decay_factor)
                neighbor_influence += influence

            # V5 step — exact match to recorder
            prev_temp = erm.history[-1] if len(erm.history) > 0 else None
            Er_flux, next_predicted_c, beta, _ = erm.step(
                live_temp_c, data['humidity'], data['wind'], data['pressure'],
                data.get('rain_prob', 0), data.get('cloud_cover', 50), data.get('solar', 0),
                prev_temp, hour_of_day, city['local_avg_temp'], city['local_temp_range'],
                neighbor_influence=neighbor_influence, city_name=name
            )

            # FIXED: Use previous prediction for realized error (not the new one)
            prev_pred = st.session_state.previous_live.get(name, {}).get("predicted", live_temp_c)
            realized_error = live_temp_c - prev_pred
            erm.record_error(realized_error, next_predicted_c)

            # Store for chart & baseline comparison
            st.session_state.history_live[name].append({
                "time": data['time'],
                "live": live_temp_c,
                "pred_1h": next_predicted_c,
                "pred_3h": next_predicted_c + erm.predict_future([3])[3],
                "pred_6h": next_predicted_c + erm.predict_future([6])[6],
            })

            # Update previous for next cycle
            st.session_state.previous_live[name] = {
                "live_temp": live_temp_c,
                "predicted": next_predicted_c
            }

        except Exception as e:
            st.warning(f"Failed to process {name}: {e}")
            continue

        # Original UI
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

st.caption("ERM V5 — Fully synchronized with backend recorder • Auto-refreshes every 30s in Live mode")
