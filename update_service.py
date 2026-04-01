from fastapi import FastAPI, BackgroundTasks
import uvicorn
import os
import subprocess
import numpy as np
import httpx
import asyncio
import pandas as pd
import logging
import threading
from datetime import datetime
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Dict, Optional
import math
from zoneinfo import ZoneInfo
import json  # ← ADDED: required for save_state / load_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="ERM Live Update Service — V5")

VERSION = "5.0"

DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL",    "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK",   "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP",    "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 28.0},
]

http_client: Optional[httpx.AsyncClient] = None
failure_counter = defaultdict(int)

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

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

    def save_state(self, filepath: Path):
        """Patched: now safely converts defaultdict and uses pretty-print JSON"""
        state = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, deque):
                state[k] = list(v)
            elif isinstance(v, defaultdict):
                state[k] = dict(v)          # ← fix for hourly_bias
            else:
                state[k] = v
        state["last_update_timestamp"] = datetime.now().isoformat()
        filepath.write_text(json.dumps(state, indent=2))

    def load_state(self, filepath: Path):
        if filepath.exists():
            try:
                state = json.loads(filepath.read_text())
                for k, v in state.items():
                    if k in self.__dict__:
                        if isinstance(self.__dict__[k], deque):
                            self.__dict__[k].extend(v)
                        elif k == "hourly_bias":
                            # Restore as defaultdict so code that does hourly_bias[hour] still works
                            self.hourly_bias = defaultdict(float, v)
                        else:
                            self.__dict__[k] = v
            except Exception as e:
                logger.warning(f"Failed to load state {filepath}: {e}")

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
            return 0.0, current_temp, 0.5

        recent_t = np.array(self.history, dtype=np.float32)
        diffs = np.diff(recent_t)
        Nr = len(recent_t) * (1 + np.var(recent_t) / 10)
        Tr = max(0.6, 1 - np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6)) * (1 - np.mean(self.humidity_history) / 200)

        # V5 EWMA multi-step error feedback
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

        # V5 volatility-adaptive gamma & alpha
        if volatility > 3.0:
            self.alpha = 0.65   # less aggressive recursion
            self.gamma = 0.92   # slower decay
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

        # V5 multivariate neighbor influence
        f_field += neighbor_influence * 0.4

        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            recursive = np.sum(decayed) ** self.alpha

        # V5 adaptive field_limit
        field_limit = max(50, np.std(self.history) * 3) if len(self.history) > 1 else 200
        Er_new = np.clip(f_field + (self.lambda_damp * recursive) + correction, -field_limit, field_limit)
        self.Er_history.append(Er_new)

        if abs(Er_new) > field_limit * 0.8:
            logger.warning(f"⚠️ Extreme Er_new detected for {city_name}: {abs(Er_new):.1f}")

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
        except Exception as e:
            logger.warning(f"polyfit failed in predict_future: {e}")
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}

# normalize_city_key, get_yesterday_baseline, fetch_multi_variable_data, backfill_realized_errors, git_backup unchanged from previous version

update_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global http_client
    http_client = httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=10, max_keepalive_connections=5))
    base_dir = Path(__file__).parent
    (base_dir / "ERM_Data").mkdir(parents=True, exist_ok=True)
    (base_dir / "ERM_State").mkdir(parents=True, exist_ok=True)
    logger.info("🚀 V5 HTTP client initialized and directories ready")

@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()
    logger.info("🛑 V5 HTTP client closed")

# /update endpoint remains the same as previous version (with refined neighbor influence and missing-data fallback already applied)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "V5 Context-Aware Relational (final)"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
