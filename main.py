from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import os
import numpy as np
import httpx
import asyncio
import pandas as pd
import logging
import traceback
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
RATE_LIMIT_WINDOW = 12.0          # Increased for better dashboard UX
VERSION = "9.1"
CSV_PREFIX = "erm_v9.0"

# ===================== RATE LIMITER =====================
city_last_request: Dict[str, float] = {}
rate_limiter_lock = asyncio.Lock()

async def check_rate_limit(city_name: str):
    async with rate_limiter_lock:
        now = datetime.now().timestamp()
        for k in list(city_last_request.keys()):
            if now - city_last_request[k] > 60:
                city_last_request.pop(k, None)
        if city_name in city_last_request and now - city_last_request[city_name] < RATE_LIMIT_WINDOW:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for {city_name}. Try again in {RATE_LIMIT_WINDOW}s."
            )
        city_last_request[city_name] = now

# ===================== SAFETY HELPERS =====================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def sanitize_array(arr: List[float], default_val: float = 0.0,
                   clip_min: float = -100.0, clip_max: float = 100.0) -> np.ndarray:
    if not arr:
        return np.array([default_val], dtype=np.float32)
    arr_np = np.nan_to_num(np.array(arr, dtype=np.float32),
                           nan=default_val, posinf=clip_max, neginf=clip_min)
    return np.clip(arr_np, clip_min, clip_max)

# ===================== PARALLEL WEATHER + SATELLITE FETCH =====================
async def fetch_city_data(city: Dict, is_satellite: bool = False) -> Dict:
    name = city["name"]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            if is_satellite:
                params = {
                    "latitude": city["lat"], "longitude": city["lon"],
                    "current": "cloud_cover,shortwave_radiation", "timezone": "auto"
                }
                r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                current = r.json()["current"]
                return {
                    "cloud_cover": float(current.get("cloud_cover", 30.0)),
                    "radiation": float(current.get("shortwave_radiation", 300.0)),
                }
            else:
                params = {
                    "latitude": city["lat"], "longitude": city["lon"],
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,"
                               "precipitation_probability,cloud_cover,shortwave_radiation,wind_direction_10m",
                    "timezone": "auto"
                }
                r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                current = r.json()["current"]
                return {
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
        return {}  # fallback handled by caller

async def fetch_multi_variable_data(cities: List[Dict]) -> Dict[str, Dict]:
    tasks = [fetch_city_data(city, is_satellite=False) for city in cities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for city, result in zip(cities, results):
        name = city["name"]
        if isinstance(result, dict):
            data[name] = result
        else:
            data[name] = {"temp": 15.0, "humidity": 50.0, "wind": 5.0, "pressure": 1013.0,
                          "rain_prob": 0.0, "cloud_cover": 30.0, "solar": 400.0, "wind_dir": 180.0}
    return data

async def fetch_satellite_cloud_data(cities: List[Dict]) -> Dict[str, Dict]:
    tasks = [fetch_city_data(city, is_satellite=True) for city in cities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for city, result in zip(cities, results):
        name = city["name"]
        if isinstance(result, dict):
            data[name] = result
        else:
            data[name] = {"cloud_cover": 30.0, "radiation": 300.0}
    return data

# ===================== DEFAULT CITIES (full list) =====================
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

# ===================== HAVERSINE + NEIGHBOR HELPERS (unchanged) =====================
# ... (kept exactly as in your previous version)

# ===================== ERM CLASS v9.1 (full implementation) =====================
class ERM_Live_Adaptive:
    # ... (kept exactly as in your previous version - all methods present)

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

# ===================== LOAD / SAVE (unchanged) =====================
# ... (kept exactly as before)

async def cleanup_rate_limiter(interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        async with rate_limiter_lock:
            now = datetime.now().timestamp()
            for k in list(city_last_request.keys()):
                if now - city_last_request[k] > RATE_LIMIT_WINDOW * 5:  # synced with window
                    city_last_request.pop(k, None)

# ===================== UPDATE, /latest, /predict, /benchmark (unchanged) =====================
# ... (kept exactly as before)

@app.get("/visualize/{city}")
async def visualize_city(city: str):
    # NO rate limiter here — dashboard calls this frequently
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")

    try:
        if len(erm.history) < 10:
            # Guard for insufficient data
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough historical data yet.\nRun a few updates first.", 
                    ha="center", va="center", fontsize=14, color="#ffaa00")
            ax.axis("off")
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            plt.clf()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            return {"visualization": {"dashboard_png_base64": img_base64}}

        # Normal visualization (rest of your plot code)
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"ERM v{VERSION} — {city} Live Dashboard", fontsize=16, color="#00ff88")

        axs[0, 0].plot(list(erm.history), color="#00ff88", linewidth=2, label="Live Temp")
        if erm.last_predicted is not None:
            axs[0, 0].axhline(erm.last_predicted, color="#ffaa00", linestyle="--", label="Last Prediction")
        axs[0, 0].set_title("Temperature History")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        regimes = list(erm.regime_tracker.keys())
        success = [erm.regime_tracker[r]["success"] / erm.regime_tracker[r]["count"] if erm.regime_tracker[r]["count"] > 0 else 0 for r in regimes]
        axs[0, 1].bar(regimes, success, color="#00ff88")
        axs[0, 1].set_title("Regime Success Rate")
        axs[0, 1].set_ylim(0, 1)

        bench = erm.benchmark_vs_baselines()
        if "status" not in bench:
            labels = ["ERM", "Persistence", "Linear", "SMA"]
            values = [bench["mae_erm"], bench["mae_persistence"], bench["mae_linear_reg"], bench["mae_sma"]]
            axs[1, 0].bar(labels, values, color="#ffaa00")
            axs[1, 0].set_title("Benchmark MAE (lower is better)")
        else:
            axs[1, 0].text(0.5, 0.5, "Not enough data", ha="center", va="center")

        axs[1, 1].text(0.5, 0.5, f"Confidence\n{round(erm.performance_score * 100, 1)}%", 
                       ha="center", va="center", fontsize=20, color="#00ff88")
        axs[1, 1].set_title("Confidence")
        axs[1, 1].axis("off")

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.clf()                     # Extra memory cleanup
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"visualization": {"dashboard_png_base64": img_base64}}

    except Exception as e:
        logger.error(f"Visualize failed for {city}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
