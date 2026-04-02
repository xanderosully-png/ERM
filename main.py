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
RATE_LIMIT_WINDOW = 12.0
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

# ===================== PARALLEL FETCHES =====================
async def fetch_city_data(city: Dict, is_satellite: bool = False) -> Dict:
    name = city["name"]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            if is_satellite:
                params = {"latitude": city["lat"], "longitude": city["lon"], "current": "cloud_cover,shortwave_radiation", "timezone": "auto"}
                r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
                current = r.json()["current"]
                return {"cloud_cover": float(current.get("cloud_cover", 30.0)), "radiation": float(current.get("shortwave_radiation", 300.0))}
            else:
                params = {"latitude": city["lat"], "longitude": city["lon"], "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation_probability,cloud_cover,shortwave_radiation,wind_direction_10m", "timezone": "auto"}
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
        return {}

async def fetch_multi_variable_data(cities: List[Dict]) -> Dict[str, Dict]:
    tasks = [fetch_city_data(city, is_satellite=False) for city in cities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for city, result in zip(cities, results):
        name = city["name"]
        data[name] = result if isinstance(result, dict) else {"temp": 15.0, "humidity": 50.0, "wind": 5.0, "pressure": 1013.0, "rain_prob": 0.0, "cloud_cover": 30.0, "solar": 400.0, "wind_dir": 180.0}
    return data

async def fetch_satellite_cloud_data(cities: List[Dict]) -> Dict[str, Dict]:
    tasks = [fetch_city_data(city, is_satellite=True) for city in cities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for city, result in zip(cities, results):
        name = city["name"]
        data[name] = result if isinstance(result, dict) else {"cloud_cover": 30.0, "radiation": 300.0}
    return data

# ===================== DEFAULT CITIES =====================
DEFAULT_CITIES = [ ... ]   # ← your full list of 18 cities (unchanged)

# ===================== HAVERSINE + NEIGHBOR HELPERS =====================
# (unchanged - kept exactly as before)

# ===================== ERM CLASS v9.1 =====================
class ERM_Live_Adaptive:
    # (unchanged - kept exactly as before)

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
    logger.info(f"🚀 ERM v{VERSION} started")
    yield
    logger.info(f"🛑 ERM v{VERSION} shutdown complete")

app = FastAPI(title=f"ERM Live Update Service — v{VERSION}", lifespan=lifespan)

# ===================== LOAD / SAVE (HEAVILY LOGGED) =====================
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
                logger.info(f"✅ Loaded {len(df)} records for {name}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}\n{traceback.format_exc()}")
        else:
            logger.info(f"No CSV yet for {name} — starting fresh")
        erms[name] = erm
    return erms

async def save_all_city_states(erms: Dict):
    logger.info("💾 Starting save_all_city_states...")
    for name, erm in erms.items():
        csv_file = DATA_DIR / f"{CSV_PREFIX}_{name}.csv"
        try:
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
            logger.info(f"✅ Saved {len(rows)} records for {name} → {csv_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save {name}: {e}\n{traceback.format_exc()}")
    await async_git_backup(DATA_DIR, STATE_DIR)   # commit new CSVs to Git
    logger.info("✅ All cities saved and backed up to Git")

# ===================== GIT BACKUP =====================
async def async_git_backup(data_dir: Path, state_dir: Path):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    if not token or not repo:
        logger.warning("GITHUB_TOKEN or GITHUB_REPO not set — skipping git backup")
        return
    try:
        remote_url = f"https://{token}@github.com/{repo}.git"
        cwd = Path(__file__).parent
        await asyncio.create_subprocess_exec("git", "add", str(data_dir), str(state_dir), cwd=cwd)
        proc = await asyncio.create_subprocess_exec("git", "commit", "-m", f"🚀 Auto-save {datetime.utcnow().isoformat()}", cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        await asyncio.create_subprocess_exec("git", "push", remote_url, "main", cwd=cwd)
        logger.info("✅ GitHub backup completed")
    except Exception as e:
        logger.error(f"GitHub backup failed: {e}")

# ===================== PERIODIC & CLEANUP =====================
async def cleanup_rate_limiter(interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        async with rate_limiter_lock:
            now = datetime.now().timestamp()
            for k in list(city_last_request.keys()):
                if now - city_last_request[k] > RATE_LIMIT_WINDOW * 5:
                    city_last_request.pop(k, None)

async def periodic_save(interval_seconds: int = 300):
    while True:
        try:
            if hasattr(app.state, 'per_city_erms'):
                await save_all_city_states(app.state.per_city_erms)
                for erm in app.state.per_city_erms.values():
                    erm.update_performance_score(erm.error_history[-1] if erm.error_history else 0)
        except Exception as e:
            logger.error(f"Periodic save failed: {e}\n{traceback.format_exc()}")
        await asyncio.sleep(interval_seconds)

# ===================== UPDATE & DASHBOARD ENDPOINTS =====================
# (all other endpoints unchanged from previous version)

@app.get("/status")
async def status():
    """Quick diagnostics endpoint"""
    if not hasattr(app.state, 'per_city_erms'):
        return {"status": "not_initialized"}
    counts = {name: len(erm.history) for name, erm in app.state.per_city_erms.items()}
    total_records = sum(counts.values())
    return {
        "version": VERSION,
        "cities_loaded": len(counts),
        "cities_with_data": sum(1 for v in counts.values() if v > 0),
        "total_records": total_records,
        "per_city_records": counts,
        "rate_limit_window": RATE_LIMIT_WINDOW
    }

# ===================== VISUALIZE (unchanged except guard) =====================
@app.get("/visualize/{city}")
async def visualize_city(city: str):
    # (your current visualize code with the <10 record guard — unchanged)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
