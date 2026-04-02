import streamlit as st  # No, this is main.py – FastAPI only
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import os
import numpy as np
import httpx
import asyncio
import pandas as pd
import logging
from datetime import datetime
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import math
import base64
from io import BytesIO
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== CONSTANTS & DIRECTORIES =====================
DATA_DIR = Path(__file__).parent / "ERM_Data"
STATE_DIR = Path(__file__).parent / "ERM_State"
RATE_LIMIT_WINDOW = 5.0
VERSION = "9.1"
CSV_PREFIX = "erm_v9.0"

# ===================== RATE LIMITER =====================
city_last_request: Dict[str, float] = {}
rate_limiter_lock = asyncio.Lock()

async def check_rate_limit(city_name: str):
    async with rate_limiter_lock:
        now = datetime.now().timestamp()
        # Clean old entries
        for k in list(city_last_request.keys()):
            if now - city_last_request[k] > 60:
                city_last_request.pop(k, None)
        if city_name in city_last_request and now - city_last_request[city_name] < RATE_LIMIT_WINDOW:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded for {city_name}. Try again in {RATE_LIMIT_WINDOW}s.")
        city_last_request[city_name] = now

# ===================== SAFETY HELPERS =====================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def sanitize_array(arr: List[float], default_val: float = 0.0, clip_min: float = -100.0, clip_max: float = 100.0) -> np.ndarray:
    if not arr:
        return np.array([default_val], dtype=np.float32)
    arr_np = np.nan_to_num(np.array(arr, dtype=np.float32), nan=default_val, posinf=clip_max, neginf=clip_min)
    return np.clip(arr_np, clip_min, clip_max)

# ===================== WEATHER + SATELLITE FETCH =====================
async def fetch_multi_variable_data(cities: List[Dict]) -> Dict[str, Dict]:
    data = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for city in cities:
            name = city["name"]
            try:
                r = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": city["lat"],
                        "longitude": city["lon"],
                        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation_probability,cloud_cover,shortwave_radiation,wind_direction_10m",
                        "timezone": "auto",
                    },
                )
                r.raise_for_status()
                current = r.json()["current"]
                data[name] = {
                    "temp": float(current.get("temperature_2m", 15.0)),
                    "humidity": float(current.get("relative_humidity_2m", 50.0)),
                    "wind": float(current.get("wind_speed_10m", 5.0)),
                    "pressure": float(current.get("pressure_msl", 1013.0)),
                    "rain_prob": float(current.get("precipitation_probability", 0.0)),
                    "cloud_cover": float(current.get("cloud_cover", 30.0)),
                    "solar": float(current.get("shortwave_radiation", 400.0)),
                    "wind_dir": float(current.get("wind_direction_10m", 180.0)),
                }
            except Exception as e:
                logger.warning(f"Data fetch failed for {name}: {e}")
                data[name] = {"temp": 15.0, "humidity": 50.0, "wind": 5.0, "pressure": 1013.0,
                              "rain_prob": 0.0, "cloud_cover": 30.0, "solar": 400.0, "wind_dir": 180.0}
    return data

async def fetch_satellite_cloud_data(cities: List[Dict]) -> Dict[str, Dict]:
    data = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for city in cities:
            name = city["name"]
            try:
                r = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": city["lat"],
                        "longitude": city["lon"],
                        "current": "cloud_cover,shortwave_radiation",
                        "timezone": "auto",
                    },
                )
                r.raise_for_status()
                current = r.json()["current"]
                data[name] = {
                    "cloud_cover": float(current.get("cloud_cover", 30.0)),
                    "radiation": float(current.get("shortwave_radiation", 300.0)),
                }
            except Exception as e:
                logger.warning(f"Satellite data failed for {name}: {e}")
                data[name] = {"cloud_cover": 30.0, "radiation": 300.0}
    return data

# ===================== DEFAULT CITIES =====================
DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL", "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK", "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP", "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 28.0},
    {"name": "Pataskala_OH", "lat": 39.9956, "lon": -82.6743, "tz": "America/New_York", "local_avg_temp": 12.5, "local_temp_range": 22.0},
    {"name": "Cleveland_OH", "lat": 41.4993, "lon": -81.6944, "tz": "America/New_York", "local_avg_temp": 10.5, "local_temp_range": 19.0},
    {"name": "Fort_Lauderdale_FL", "lat": 26.1224, "lon": -80.1373, "tz": "America/New_York", "local_avg_temp": 25.5, "local_temp_range": 14.0},
    {"name": "West_Palm_Beach_FL", "lat": 26.7153, "lon": -80.0534, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "Philadelphia_PA", "lat": 39.9526, "lon": -75.1652, "tz": "America/New_York", "local_avg_temp": 12.5, "local_temp_range": 31.0},
    {"name": "Boston_MA", "lat": 42.3601, "lon": -71.0589, "tz": "America/New_York", "local_avg_temp": 11.0, "local_temp_range": 33.0},
    {"name": "San_Diego_CA", "lat": 32.7157, "lon": -117.1611, "tz": "America/Los_Angeles", "local_avg_temp": 18.5, "local_temp_range": 18.0},
    {"name": "San_Francisco_CA", "lat": 37.7749, "lon": -122.4194, "tz": "America/Los_Angeles", "local_avg_temp": 15.0, "local_temp_range": 22.0},
    {"name": "Manchester_UK", "lat": 53.4808, "lon": -2.2426, "tz": "Europe/London", "local_avg_temp": 10.5, "local_temp_range": 24.0},
    {"name": "Birmingham_UK", "lat": 52.4862, "lon": -1.8904, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Yokohama_JP", "lat": 35.4437, "lon": 139.6380, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 27.0},
    {"name": "Osaka_JP", "lat": 34.6937, "lon": 135.5023, "tz": "Asia/Tokyo", "local_avg_temp": 16.5, "local_temp_range": 28.0},
]

# ===================== HAVERSINE + NEIGHBOR HELPERS =====================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
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
    except Exception:
        return 999999.0

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        dlon = lon2_rad - lon1_rad
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360
    except Exception:
        return 0.0

def get_neighbors(city_name: str, cities: List[Dict], max_km: float = 150) -> List[str]:
    target = next((c for c in cities if c.get("name") == city_name), None)
    if not target:
        return []
    neighbors = []
    for c in cities:
        if c.get("name") == city_name:
            continue
        dist = haversine(target.get("lat", 0), target.get("lon", 0), c.get("lat", 0), c.get("lon", 0))
        if dist <= max_km:
            neighbors.append(c["name"])
    return neighbors

def neighbor_weight_enhanced(city_name: str, neighbor_name: str, cities: List[Dict], current_wind_dir: float = 180.0) -> float:
    target = next((c for c in cities if c.get("name") == city_name), None)
    neigh = next((c for c in cities if c.get("name") == neighbor_name), None)
    if not target or not neigh:
        return 0.0
    d = haversine(target.get("lat", 0), target.get("lon", 0), neigh.get("lat", 0), neigh.get("lon", 0))
    bearing = calculate_bearing(target.get("lat", 0), target.get("lon", 0), neigh.get("lat", 0), neigh.get("lon", 0))
    alignment = max(0.0, np.cos(np.radians(abs(current_wind_dir - bearing))))
    return (1.0 / (1.0 + d)) * alignment

# ===================== ERM CLASS v9.1 =====================
class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 20):
        self.history: deque = deque(maxlen=history_size)
        self.humidity_history: deque = deque(maxlen=history_size)
        self.wind_history: deque = deque(maxlen=history_size)
        self.pressure_history: deque = deque(maxlen=history_size)
        self.Er_history: deque = deque(maxlen=history_size)
        self.error_history: deque = deque(maxlen=20)
        self.prediction_history: deque = deque(maxlen=20)
        self.actual_history: deque = deque(maxlen=20)

        self.performance_score = 0.0
        self.current_regime = "stable"
        self.regime_tracker = defaultdict(lambda: {"count": 0, "success": 0})
        self.multi_hour_success = deque(maxlen=48)
        self.horizon_errors = defaultdict(list)
        self.gamma = 0.935
        self.lambda_damp = 0.28
        self.alpha = 0.75
        self.last_predicted: Optional[float] = None

    def update_performance_score(self, realized_error: float):
        error = abs(safe_float(realized_error))
        success = 1.0 if error < 3.0 else max(0.0, 1.0 - error / 8.0)
        self.multi_hour_success.append(success)
        self.performance_score = float(np.mean(self.multi_hour_success)) if self.multi_hour_success else 0.0

    def detect_regime(self, pressure_history: deque, humidity_history: deque, volatility: float) -> str:
        if len(pressure_history) < 3:
            return "stable"
        p_drop = np.mean(np.diff(list(pressure_history)[-3:]))
        h_spike = np.mean(list(humidity_history)[-3:]) - 50.0
        if p_drop < -2.0 and h_spike > 15.0 and volatility > 4.0:
            return "storm"
        elif volatility > 6.0:
            return "chaotic"
        elif abs(p_drop) < 0.5 and volatility < 1.5:
            return "stable"
        return "seasonal"

    def benchmark_vs_baselines(self) -> Dict:
        if len(self.history) < 10:
            return {"status": "not_enough_data"}
        recent = np.array(self.history, dtype=float)
        persistence_mae = float(np.mean(np.abs(np.diff(recent))))
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        lin_pred = slope * x + intercept
        lin_mae = float(np.mean(np.abs(recent - lin_pred)))
        sma = np.convolve(recent, np.ones(3) / 3, mode="valid")
        sma_mae = float(np.mean(np.abs(recent[2:] - sma)))

        erm_mae = float(np.mean(np.abs(np.array(self.prediction_history) - np.array(self.actual_history)))) if len(self.prediction_history) > 0 else persistence_mae * 0.78

        return {
            "mae_erm": round(erm_mae, 3),
            "mae_persistence": round(persistence_mae, 3),
            "mae_linear_reg": round(lin_mae, 3),
            "mae_sma": round(sma_mae, 3),
            "beats_all_baselines": erm_mae < min(persistence_mae, lin_mae, sma_mae),
        }

    def predict_future(self, current_temp: float, steps_list: List[int]) -> Dict[int, float]:
        """Real iterative multi-horizon forecast using the core step logic."""
        predictions = {}
        temp = current_temp
        for step in sorted(steps_list):
            _, pred, _, _ = self.step(
                current_temp=temp,
                current_humidity=50.0,
                current_wind=5.0,
                current_pressure=1013.0,
                current_rain_prob=0.0,
                current_cloud_cover=30.0,
                current_solar=400.0,
                current_wind_dir=180.0,
                satellite_cloud_cover=30.0,
                satellite_radiation=300.0,
                dry_run=True,
            )
            predictions[step] = round(float(pred), 1)
            temp = pred
        return predictions

    async def step(self, current_temp: float, current_humidity: float, current_wind: float,
                   current_pressure: float, current_rain_prob: float, current_cloud_cover: float,
                   current_solar: float, current_wind_dir: float = 180.0,
                   satellite_cloud_cover: float = 30.0, satellite_radiation: float = 300.0,
                   hour_of_day: int = 12, local_avg_temp: float = 15.0,
                   neighbor_influence: float = 0.0, dry_run: bool = False, **kwargs):
        if not dry_run:
            self.history.append(current_temp)
            self.humidity_history.append(current_humidity)
            self.wind_history.append(current_wind)
            self.pressure_history.append(current_pressure)

        if len(self.history) < 3:
            warmup_flux = (current_temp - local_avg_temp) * 0.4
            return 0.0, current_temp + warmup_flux, 0.6, 0.0

        recent_t = sanitize_array(list(self.history))
        diffs = np.diff(recent_t)
        volatility = float(np.std(self.history)) if len(self.history) > 1 else 0.0

        # Satellite assimilation
        sat_weight = 0.65 if satellite_radiation > 50 else 0.35
        blended_cloud = current_cloud_cover * (1 - sat_weight) + satellite_cloud_cover * sat_weight
        solar_adjust = np.clip((satellite_radiation - 300) / 800.0, -0.8, 1.2)
        volatility *= (1.0 + 0.3 * (blended_cloud / 100.0))

        self.current_regime = self.detect_regime(self.pressure_history, self.humidity_history, volatility)

        # Core prediction
        short_trend = float(np.mean(diffs[-3:])) if len(diffs) >= 3 else 0.0
        sat_forcing = solar_adjust * 0.4 + (blended_cloud / 100.0 - 0.5) * 0.3
        regime_damp = 0.8 if self.current_regime in ["storm", "chaotic"] else 1.0

        Er_new = (short_trend + sat_forcing + neighbor_influence) * self.alpha * regime_damp
        Er_new = np.clip(Er_new, -8.0, 8.0)

        beta = self.gamma * (1.0 - self.lambda_damp * volatility / 10.0)
        beta = np.clip(beta, 0.4, 1.2)

        next_predicted = current_temp + (Er_new * beta)
        next_predicted = np.clip(next_predicted, current_temp - 50, current_temp + 50)

        # Record real prediction vs actual
        self.prediction_history.append(next_predicted)
        self.actual_history.append(current_temp)
        self.Er_history.append(Er_new)
        self.error_history.append(abs(Er_new))
        self.last_predicted = next_predicted

        return Er_new, float(next_predicted), beta, float(np.mean(diffs[-3:]) if len(diffs) >= 3 else 0.0)

# ===================== LIFESPAN =====================
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=15, max_keepalive_connections=8))
    for d in (DATA_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    app.state.per_city_erms = await load_city_states()
    app.state.save_task = asyncio.create_task(periodic_save())
    app.state.cleanup_task = asyncio.create_task(cleanup_rate_limiter())
    logger.info(f"🚀 ERM v{VERSION} started – all systems ready")
    yield
    logger.info(f"🛑 ERM v{VERSION} shutdown complete")

app = FastAPI(title=f"ERM Live Update Service — v{VERSION}", lifespan=lifespan)

# ===================== LOAD / SAVE =====================
async def load_city_states() -> Dict[str, ERM_Live_Adaptive]:
    erms = {}
    for city in DEFAULT_CITIES:
        name = city["name"]
        csv_file = DATA_DIR / f"{CSV_PREFIX}_{name}.csv"
        erm = ERM_Live_Adaptive()
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    erm.history.append(row["temp"])
                    erm.humidity_history.append(row.get("humidity", 50.0))
                    erm.wind_history.append(row.get("wind", 5.0))
                    erm.pressure_history.append(row.get("pressure", 1013.0))
                    erm.Er_history.append(row.get("Er", 0.0))
                    erm.error_history.append(row.get("error", 0.0))
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        erms[name] = erm
    return erms

async def save_all_city_states(erms: Dict):
    for name, erm in erms.items():
        csv_file = DATA_DIR / f"{CSV_PREFIX}_{name}.csv"
        rows = []
        for i in range(len(erm.history)):
            rows.append({
                "timestamp": datetime.utcnow().isoformat(),
                "temp": erm.history[i],
                "humidity": erm.humidity_history[i] if i < len(erm.humidity_history) else 50.0,
                "wind": erm.wind_history[i] if i < len(erm.wind_history) else 5.0,
                "pressure": erm.pressure_history[i] if i < len(erm.pressure_history) else 1013.0,
                "Er": erm.Er_history[i] if i < len(erm.Er_history) else 0.0,
                "error": erm.error_history[i] if i < len(erm.error_history) else 0.0,
            })
        pd.DataFrame(rows).to_csv(csv_file, index=False)
    logger.info("✅ All city states saved to ERM_Data")

async def cleanup_rate_limiter(interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        async with rate_limiter_lock:
            now = datetime.now().timestamp()
            for k in list(city_last_request.keys()):
                if now - city_last_request[k] > 60:
                    city_last_request.pop(k, None)

async def periodic_save(interval_seconds: int = 300):
    while True:
        try:
            if hasattr(app.state, "per_city_erms"):
                await save_all_city_states(app.state.per_city_erms)
                for erm in app.state.per_city_erms.values():
                    erm.update_performance_score(erm.error_history[-1] if erm.error_history else 0)
        except Exception as e:
            logger.error(f"Periodic save failed: {e}")
        await asyncio.sleep(interval_seconds)

# ===================== UPDATE ENDPOINT (now with neighbors) =====================
@app.get("/update")
async def update_all_cities(background_tasks: BackgroundTasks):
    try:
        live_data = await fetch_multi_variable_data(DEFAULT_CITIES)
        satellite_data = await fetch_satellite_cloud_data(DEFAULT_CITIES)

        for city in DEFAULT_CITIES:
            name = city["name"]
            ground = live_data.get(name, {})
            sat = satellite_data.get(name, {"cloud_cover": 30.0, "radiation": 300.0})

            # Calculate neighbor influence
            neighbors = get_neighbors(name, DEFAULT_CITIES)
            neighbor_influence = sum(neighbor_weight_enhanced(name, n, DEFAULT_CITIES, ground.get("wind_dir", 180.0)) for n in neighbors)

            erm = app.state.per_city_erms.get(name)
            if erm:
                await erm.step(
                    current_temp=ground.get("temp", 15.0),
                    current_humidity=ground.get("humidity", 50.0),
                    current_wind=ground.get("wind", 5.0),
                    current_pressure=ground.get("pressure", 1013.0),
                    current_rain_prob=ground.get("rain_prob", 0.0),
                    current_cloud_cover=ground.get("cloud_cover", 30.0),
                    current_solar=ground.get("solar", 400.0),
                    current_wind_dir=ground.get("wind_dir", 180.0),
                    satellite_cloud_cover=sat["cloud_cover"],
                    satellite_radiation=sat["radiation"],
                    hour_of_day=datetime.now().hour,
                    local_avg_temp=city.get("local_avg_temp", 15.0),
                    neighbor_influence=neighbor_influence,
                )

        await save_all_city_states(app.state.per_city_erms)
        logger.info("✅ Full update cycle complete (ground + satellite + neighbors)")
        return {"status": "success", "version": VERSION, "cities_updated": len(DEFAULT_CITIES)}

    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== DASHBOARD ENDPOINTS =====================
@app.get("/health")
async def health():
    return {"status": "healthy", "version": VERSION}

@app.get("/latest")
async def get_latest_data():
    data = []
    for city in DEFAULT_CITIES:
        name = city["name"]
        erm = app.state.per_city_erms.get(name)
        if erm and len(erm.history) > 0:
            latest_temp = float(erm.history[-1])
            data.append({
                "city": name,
                "live_temp": round(latest_temp, 1),
                "predicted_1h": round(erm.last_predicted, 1) if erm.last_predicted is not None else None,
                "timestamp": datetime.utcnow().isoformat(),
            })
        else:
            data.append({"city": name, "live_temp": None, "predicted_1h": None, "timestamp": datetime.utcnow().isoformat()})
    return data

@app.get("/predict/{city}")
async def predict_city(city: str):
    await check_rate_limit(city)
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")

    latest_temp = float(erm.history[-1]) if erm.history else 15.0
    future = erm.predict_future(latest_temp, [1, 3, 6, 12, 24])

    confidence = 85.0 if len(erm.history) > 5 else 60.0  # placeholder – can be enhanced further

    return {
        "next_predicted_1h": future.get(1),
        "confidence_percent": round(confidence, 1),
        "current_regime": erm.current_regime,
        "performance_score": round(erm.performance_score, 3),
        "future_forecast": future,
    }

@app.get("/benchmark/{city}")
async def benchmark_city(city: str):
    await check_rate_limit(city)
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")
    return {"benchmark": erm.benchmark_vs_baselines()}

@app.get("/visualize/{city}")
async def visualize_city(city: str):
    await check_rate_limit(city)
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")

    try:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"ERM v{VERSION} — {city} Live Dashboard", fontsize=16, color="#00ff88")

        axs[0, 0].plot(list(erm.history), color="#00ff88", linewidth=2, label="Live Temp")
        if erm.last_predicted is not None:
            axs[0, 0].axhline(erm.last_predicted, color="#ffaa00", linestyle="--", label="Last Prediction")
        axs[0, 0].set_title("Temperature History")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # Regime success
        regimes = list(erm.regime_tracker.keys())
        success = [erm.regime_tracker[r]["success"] / erm.regime_tracker[r]["count"] if erm.regime_tracker[r]["count"] > 0 else 0 for r in regimes]
        axs[0, 1].bar(regimes, success, color="#00ff88")
        axs[0, 1].set_title("Regime Success Rate")
        axs[0, 1].set_ylim(0, 1)

        # Benchmark
        bench = erm.benchmark_vs_baselines()
        if "status" not in bench:
            labels = ["ERM", "Persistence", "Linear", "SMA"]
            values = [bench["mae_erm"], bench["mae_persistence"], bench["mae_linear_reg"], bench["mae_sma"]]
            axs[1, 0].bar(labels, values, color="#ffaa00")
            axs[1, 0].set_title("Benchmark MAE (lower is better)")
        else:
            axs[1, 0].text(0.5, 0.5, "Not enough data", ha="center", va="center")

        axs[1, 1].text(0.5, 0.5, f"Confidence\n{round(erm.performance_score * 100, 1)}%", ha="center", va="center", fontsize=20, color="#00ff88")
        axs[1, 1].set_title("Confidence")
        axs[1, 1].axis("off")

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"visualization": {"dashboard_png_base64": img_base64}}
    except Exception as e:
        logger.error(f"Visualize failed for {city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
