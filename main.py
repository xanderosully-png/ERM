from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
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

# NEW: Import the modular ERM v3 model
from erm_model import ERM_Live_Adaptive   # ← Phase 1: Clean, improved v3 model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
class Settings:
    DATA_DIR = Path(__file__).parent / "ERM_Data"
    STATE_DIR = Path(__file__).parent / "ERM_State"
    RATE_LIMIT_WINDOW = 45.0
    VERSION = "9.4"                                      # ← Bumped for v3 model integration
    CSV_PREFIX = "erm_v9.0"
    HISTORY_SIZE = 24                                    # ← Matches v3 default
    SAVE_INTERVAL_SEC = 300
    AUTO_UPDATE_INTERVAL_MIN = 10
    MAX_CSV_LOAD_RECORDS = 100

settings = Settings()

# ===================== LOCKS =====================
city_last_request: Dict[str, float] = {}
rate_limiter_lock = asyncio.Lock()
git_backup_lock = asyncio.Lock()
csv_write_lock = asyncio.Lock()

# ===================== SHARED HTTPX CLIENT =====================
http_client: Optional[httpx.AsyncClient] = None

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

# ===================== RESPONSE MODELS =====================
class CityData(BaseModel):
    city: str
    current_temp: float
    last_prediction: Optional[float] = None
    current_regime: str
    performance_score: float
    timestamp: str

class PredictionResponse(BaseModel):
    predictions: Dict[str, float]
    current_regime: str
    confidence: float

class StatusResponse(BaseModel):
    version: str
    cities: int
    total_records: int
    per_city: Dict[str, int]
    avg_performance: float
    build_phase: str

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: str = "running"

# ===================== MERGED FETCH =====================
async def fetch_city_data(city: Dict) -> Dict:
    name = city["name"]
    try:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation_probability,cloud_cover,shortwave_radiation,wind_direction_10m",
            "timezone": "auto",
        }
        r = await http_client.get("https://api.open-meteo.com/v1/forecast", params=params)
        r.raise_for_status()
        current = r.json()["current"]
        logger.info(f"✅ Fetched live data for {name}")
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
        return {}

async def fetch_multi_variable_data(cities: List[Dict]) -> Dict[str, Dict]:
    tasks = [fetch_city_data(city) for city in cities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for city, result in zip(cities, results):
        name = city["name"]
        data[name] = result if isinstance(result, dict) else {"temp": 15.0, "humidity": 50.0, "wind": 5.0, "pressure": 1013.0, "rain_prob": 0.0, "cloud_cover": 30.0, "solar": 400.0, "wind_dir": 180.0}
    return data

# ===================== DEFAULT CITIES =====================
# (unchanged - will be made configurable in Phase 2)
DEFAULT_CITIES = [ ... ]   # ← your original list stays exactly the same

# ===================== NEIGHBOR HELPERS =====================
# (unchanged)
def haversine(...): ...
def calculate_bearing(...): ...
def get_neighbors(...): ...
def neighbor_weight_enhanced(...): ...

# ===================== LIFESPAN =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    for d in (settings.DATA_DIR, settings.STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    http_client = httpx.AsyncClient(timeout=8.0, limits=httpx.Limits(max_connections=20))

    app.state.per_city_erms = await load_city_states()
    app.state.save_task = asyncio.create_task(periodic_save())
    app.state.cleanup_task = asyncio.create_task(cleanup_rate_limiter())
    app.state.auto_update_task = asyncio.create_task(periodic_auto_update())

    logger.info(f"🚀 ERM v{settings.VERSION} (with ERM v3 model) started")
    yield

    # ... cleanup code stays the same ...

app = FastAPI(title=f"ERM Live Update Service — v{settings.VERSION}", lifespan=lifespan)

# ===================== UPDATED LOAD (v3 compatible) =====================
async def load_city_states() -> Dict[str, ERM_Live_Adaptive]:
    erms = {}
    for city in DEFAULT_CITIES:
        name = city["name"]
        erm = ERM_Live_Adaptive(city_name=name)   # v3 constructor

        csv_file = settings.DATA_DIR / f"{settings.CSV_PREFIX}_{name}.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file).tail(settings.MAX_CSV_LOAD_RECORDS)
                logger.info(f"✅ Loaded last {len(df)} records for {name}")

                for _, row in df.iterrows():
                    erm.history.append(row["temp"])
                    erm.humidity_history.append(row.get("humidity", 50.0))
                    erm.wind_history.append(row.get("wind", 5.0))
                    erm.pressure_history.append(row.get("pressure", 1013.0))
                    erm.Er_history.append(row.get("Er", 0.0))
                    erm.smoothed_er = float(row.get("smoothed_er", 0.0))
                    erm.performance_score = float(row.get("performance_score", 0.0))
                    erm.current_regime = row.get("current_regime", "stable")
                    erm.local_climatology = float(row.get("local_climatology", 15.0))  # v3 field

                if len(df) > 0:
                    erm.last_predicted = float(df.iloc[-1]["temp"])
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        else:
            logger.info(f"No CSV yet for {name} — starting fresh")

        erms[name] = erm
    return erms

# ===================== INCREMENTAL SAVE (updated for v3) =====================
async def save_all_city_states(erms: Dict):
    logger.info("💾 Starting incremental save...")
    async with csv_write_lock:
        saved_count = 0
        for name, erm in erms.items():
            csv_file = settings.DATA_DIR / f"{settings.CSV_PREFIX}_{name}.csv"
            try:
                if len(erm.history) == 0:
                    continue

                row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "temp": erm.history[-1],
                    "humidity": erm.humidity_history[-1] if erm.humidity_history else 50.0,
                    "wind": erm.wind_history[-1] if erm.wind_history else 5.0,
                    "pressure": erm.pressure_history[-1] if erm.pressure_history else 1013.0,
                    "Er": erm.Er_history[-1] if erm.Er_history else 0.0,
                    "smoothed_er": erm.smoothed_er,
                    "performance_score": erm.performance_score,
                    "current_regime": erm.current_regime,
                    "local_climatology": erm.local_climatology,          # ← NEW for v3
                }

                # ... rest of duplicate-check + write logic stays exactly the same ...

                if should_write:
                    df = pd.DataFrame([row])
                    df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)
                    logger.info(f"✅ Appended record for {name}")
                    saved_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to save {name}: {e}")
        logger.info(f"💾 Incremental save finished — {saved_count} cities updated")

    await async_git_backup(settings.DATA_DIR, settings.STATE_DIR)

# ===================== PERIODIC TASKS (unchanged) =====================
# ... periodic_save, cleanup_rate_limiter, periodic_auto_update stay the same ...

# ===================== UPDATED UPDATE ENDPOINT =====================
@app.get("/update")
async def update_all_cities(background_tasks: BackgroundTasks):
    try:
        logger.info("🚀 Starting full /update cycle — ALL cities fetched and stepped")
        live_data = await fetch_multi_variable_data(DEFAULT_CITIES)

        for city in DEFAULT_CITIES:
            name = city["name"]
            ground = live_data.get(name, {})

            # ... extract ground_temp, humidity, etc. (same as before) ...

            now = datetime.now().timestamp()
            async with rate_limiter_lock:
                if now - city_last_request.get(name, 0) < settings.RATE_LIMIT_WINDOW:
                    logger.info(f"⏭️ Skipped {name} — rate limited")
                    continue
                city_last_request[name] = now

            neighbors = get_neighbors(name, DEFAULT_CITIES)
            neighbor_influence = sum(neighbor_weight_enhanced(name, n, DEFAULT_CITIES, ground_wind_dir) for n in neighbors)

            erm = app.state.per_city_erms.get(name)
            if erm:
                # v3 step is now synchronous (clean & fast)
                erm.step(
                    current_temp=ground_temp,
                    current_humidity=ground_humidity,
                    current_wind=ground_wind,
                    current_pressure=ground_pressure,
                    current_rain_prob=ground_rain_prob,
                    current_cloud_cover=ground_cloud_cover,
                    current_solar=ground_solar,
                    current_wind_dir=ground_wind_dir,
                    satellite_cloud_cover=ground_cloud_cover,
                    satellite_radiation=ground_solar,
                    hour_of_day=datetime.now().hour,
                    local_avg_temp=city.get("local_avg_temp", 15.0),
                    neighbor_influence=neighbor_influence,
                    dry_run=False
                )
                logger.info(f"✅ Stepped {name} (ERM v3)")

        await save_all_city_states(app.state.per_city_erms)
        logger.info("✅ Full update cycle completed")
        return {"status": "updated", "timestamp": datetime.utcnow().isoformat(), "model": "ERM_v3"}

    except Exception as e:
        logger.error(f"Update failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== REST OF ENDPOINTS (unchanged) =====================
# /health, /status, /latest, /predict, /visualize stay exactly the same
# (they already work perfectly with the new model)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
