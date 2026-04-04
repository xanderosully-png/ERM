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
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import math
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import json  # ← Phase 2

# Phase 1: Modular ERM v3 model
from erm_model import ERM_Live_Adaptive

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== SETTINGS (Phase 2 enhanced) =====================
class Settings:
    DATA_DIR = Path(__file__).parent / "ERM_Data"
    STATE_DIR = Path(__file__).parent / "ERM_State"
    CITIES_CONFIG = Path(__file__).parent / "cities.json"   # ← NEW

    RATE_LIMIT_WINDOW = 45.0
    VERSION = "9.5"                                         # ← Bumped for Phase 2
    CSV_PREFIX = "erm_v9.0"
    HISTORY_SIZE = 24
    SAVE_INTERVAL_SEC = 300
    AUTO_UPDATE_INTERVAL_MIN = 10
    MAX_CSV_LOAD_RECORDS = 100

settings = Settings()

# ===================== CITY CONFIG LOADER (Phase 2) =====================
def load_cities_config() -> List[Dict]:
    """Load cities from JSON file or CITIES_JSON env var. Falls back gracefully."""
    # 1. Environment variable override (best for Docker/Render)
    cities_json_env = os.getenv("CITIES_JSON")
    if cities_json_env:
        try:
            cities = json.loads(cities_json_env)
            logger.info(f"✅ Loaded {len(cities)} cities from CITIES_JSON environment variable")
            return cities
        except Exception as e:
            logger.error(f"Failed to parse CITIES_JSON env var: {e}")

    # 2. cities.json file (recommended)
    if settings.CITIES_CONFIG.exists():
        try:
            with open(settings.CITIES_CONFIG, "r", encoding="utf-8") as f:
                cities = json.load(f)
            logger.info(f"✅ Loaded {len(cities)} cities from {settings.CITIES_CONFIG.name}")
            return cities
        except Exception as e:
            logger.error(f"Failed to load {settings.CITIES_CONFIG.name}: {e}")

    # 3. Built-in fallback (original 18 cities)
    logger.warning("⚠️ No cities.json or CITIES_JSON found — using built-in fallback")
    return [
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

# ===================== NEIGHBOR HELPERS =====================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * 2 * math.asin(math.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360

def get_neighbors(city_name: str, cities: List[Dict], radius_km: float = 150.0) -> List[Dict]:
    target = next((c for c in cities if c["name"] == city_name), None)
    if not target:
        return []
    return [c for c in cities if c["name"] != city_name and haversine(target["lat"], target["lon"], c["lat"], c["lon"]) <= radius_km]

def neighbor_weight_enhanced(city_name: str, neighbor: Dict, cities: List[Dict], wind_dir: float) -> float:
    target = next((c for c in cities if c["name"] == city_name), None)
    if not target:
        return 0.0
    d = haversine(target["lat"], target["lon"], neighbor["lat"], neighbor["lon"])
    bearing = calculate_bearing(target["lat"], target["lon"], neighbor["lat"], neighbor["lon"])
    alignment = max(0.0, np.cos(np.radians(abs(wind_dir - bearing))))
    return (1.0 / (1.0 + d)) * alignment * 0.35

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

    logger.info(f"🚀 ERM v{settings.VERSION} (ERM v3 + configurable cities) started — build phase active")
    yield

    for task_name in ('save_task', 'cleanup_task', 'auto_update_task'):
        task = getattr(app.state, task_name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    if http_client:
        await http_client.aclose()
    logger.info(f"🛑 ERM v{settings.VERSION} shutdown complete")

app = FastAPI(title=f"ERM Live Update Service — v{settings.VERSION}", lifespan=lifespan)

# ===================== LOAD CITY STATES (Phase 1 + 2) =====================
async def load_city_states() -> Dict[str, ERM_Live_Adaptive]:
    cities = load_cities_config()
    erms = {}
    for city in cities:
        name = city["name"]
        erm = ERM_Live_Adaptive(city_name=name)

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
                    erm.local_climatology = float(row.get("local_climatology", 15.0))

                if len(df) > 0:
                    erm.last_predicted = float(df.iloc[-1]["temp"])
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        else:
            logger.info(f"No CSV yet for {name} — starting fresh")

        erms[name] = erm
    logger.info(f"🚀 Initialized ERM v3 for {len(erms)} cities")
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
                    "local_climatology": erm.local_climatology,   # v3 field
                }

                should_write = True
                if csv_file.exists():
                    try:
                        last_df = pd.read_csv(csv_file).tail(1)
                        if not last_df.empty:
                            last_temp = float(last_df.iloc[0]["temp"])
                            last_er = float(last_df.iloc[0]["Er"])
                            if abs(last_temp - row["temp"]) < 0.1 and abs(last_er - row["Er"]) < 0.1:
                                should_write = False
                                logger.debug(f"⏭️ Skipped duplicate record for {name}")
                    except Exception:
                        pass

                if should_write:
                    df = pd.DataFrame([row])
                    df.to_csv(csv_file, mode='a', header=not csv_file.exists(), index=False)
                    logger.info(f"✅ Appended record for {name} (records: {len(erm.history)})")
                    saved_count += 1

            except Exception as e:
                logger.error(f"❌ Failed to save {name}: {e}")
        logger.info(f"💾 Incremental save finished — {saved_count} cities updated")

    await async_git_backup(settings.DATA_DIR, settings.STATE_DIR)

# ===================== GIT BACKUP (unchanged) =====================
async def run_git_command(args: list[str], cwd: Path, check: bool = True) -> int:
    # ... (your original git helper code remains exactly the same)
    cmd = ["git"] + args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()
        if stdout_str:
            logger.debug(f"Git stdout: {stdout_str}")
        if stderr_str:
            logger.debug(f"Git stderr: {stderr_str}")
        if check and proc.returncode != 0:
            error = stderr_str or stdout_str or f"code {proc.returncode}"
            raise RuntimeError(f"{' '.join(cmd)} failed: {error}")
        return proc.returncode
    except Exception as e:
        logger.error(f"Subprocess error running {' '.join(cmd)}: {e}")
        raise

async def async_git_backup(data_dir: Path, state_dir: Path):
    # ... (your original git backup code remains exactly the same)
    async with git_backup_lock:
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        if not token or not repo:
            logger.warning("⚠️ GITHUB_TOKEN or GITHUB_REPO not set — skipping git backup")
            return
        # ... rest of your git backup code ...
        cwd = Path(__file__).parent
        logger.info(f"🔄 Starting git backup in {cwd}")
        try:
            await run_git_command(["config", "--global", "user.email", "erm-bot@render.com"], cwd)
            await run_git_command(["config", "--global", "user.name", "ERM Render Bot"], cwd)
            remote_url = f"https://{token}@github.com/{repo}.git"
            await run_git_command(["remote", "set-url", "origin", remote_url], cwd, check=False)
            logger.info("🔄 Pulling latest changes...")
            await run_git_command(["pull", "origin", "main", "--rebase"], cwd, check=False)
            logger.info(f"📁 Staging files...")
            await run_git_command(["add", "-A"], cwd)
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--cached", "--name-only", cwd=cwd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            changed_files = stdout.decode().strip()
            if not changed_files:
                logger.info("✅ No new changes to commit")
                return
            logger.info(f"📤 Changes detected:\n{changed_files}")
            commit_msg = f"🚀 Auto-save {datetime.utcnow().isoformat()}"
            await run_git_command(["commit", "-m", commit_msg], cwd)
            logger.info("🚀 Pushing to GitHub...")
            await run_git_command(["push", "origin", "main"], cwd)
            logger.info("✅ Git backup completed successfully")
        except Exception as e:
            logger.error(f"❌ Git backup failed: {e}")

# ===================== PERIODIC TASKS =====================
async def periodic_save():
    while True:
        try:
            if hasattr(app.state, 'per_city_erms'):
                await save_all_city_states(app.state.per_city_erms)
        except Exception as e:
            logger.error(f"Periodic save failed: {e}")
        await asyncio.sleep(settings.SAVE_INTERVAL_SEC)

async def cleanup_rate_limiter():
    while True:
        await asyncio.sleep(30)
        async with rate_limiter_lock:
            now = datetime.now().timestamp()
            for k in list(city_last_request.keys()):
                if now - city_last_request[k] > settings.RATE_LIMIT_WINDOW * 5:
                    city_last_request.pop(k, None)
            if len(city_last_request) > 100:
                city_last_request.clear()

async def periodic_auto_update():
    while True:
        await asyncio.sleep(settings.AUTO_UPDATE_INTERVAL_MIN * 60)
        try:
            logger.info("🔄 Auto-update cycle started")
            await update_all_cities(None)
        except Exception as e:
            logger.error(f"Auto-update failed: {e}")

# ===================== UPDATED UPDATE ENDPOINT (Phase 1 + 2) =====================
@app.get("/update")
async def update_all_cities(background_tasks: BackgroundTasks):
    try:
        logger.info("🚀 Starting full /update cycle")
        live_data = await fetch_multi_variable_data(load_cities_config())  # ← uses configurable cities

        for city in load_cities_config():
            name = city["name"]
            ground = live_data.get(name, {})

            ground_temp = ground.get("temp", 15.0)
            ground_humidity = ground.get("humidity", 50.0)
            ground_wind = ground.get("wind", 5.0)
            ground_pressure = ground.get("pressure", 1013.0)
            ground_rain_prob = ground.get("rain_prob", 0.0)
            ground_cloud_cover = ground.get("cloud_cover", 30.0)
            ground_solar = ground.get("solar", 400.0)
            ground_wind_dir = ground.get("wind_dir", 180.0)

            now = datetime.now().timestamp()
            async with rate_limiter_lock:
                if now - city_last_request.get(name, 0) < settings.RATE_LIMIT_WINDOW:
                    logger.info(f"⏭️ Skipped {name} — rate limited")
                    continue
                city_last_request[name] = now

            neighbors = get_neighbors(name, load_cities_config())
            neighbor_influence = sum(neighbor_weight_enhanced(name, n, load_cities_config(), ground_wind_dir) for n in neighbors)

            erm = app.state.per_city_erms.get(name)
            if erm:
                # ERM v3 step (synchronous)
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
        return {"status": "updated", "timestamp": datetime.utcnow().isoformat(), "model": "ERM_v3", "cities": len(app.state.per_city_erms)}
    except Exception as e:
        logger.error(f"Update failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== REMAINING ENDPOINTS (unchanged) =====================
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", version=settings.VERSION)

@app.get("/status", response_model=StatusResponse)
async def status():
    if not hasattr(app.state, 'per_city_erms'):
        raise HTTPException(status_code=503, detail="Not initialized")
    counts = {name: len(erm.history) for name, erm in app.state.per_city_erms.items()}
    avg_perf = np.mean([erm.performance_score for erm in app.state.per_city_erms.values()]) if app.state.per_city_erms else 0.0
    return StatusResponse(
        version=settings.VERSION,
        cities=len(counts),
        total_records=sum(counts.values()),
        per_city=counts,
        avg_performance=round(float(avg_perf), 3),
        build_phase="active — collecting data"
    )

@app.get("/latest/{city}", response_model=CityData)
async def get_latest(city: str):
    erm = app.state.per_city_erms.get(city) if hasattr(app.state, 'per_city_erms') else None
    if not erm or len(erm.history) == 0:
        raise HTTPException(status_code=404, detail="No data yet for this city")
    logger.info(f"📡 /latest requested for {city}")
    return CityData(
        city=city,
        current_temp=round(erm.history[-1], 2),
        last_prediction=round(erm.last_predicted, 2) if erm.last_predicted is not None else None,
        current_regime=erm.current_regime,
        performance_score=round(erm.performance_score, 3),
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/predict/{city}", response_model=PredictionResponse)
async def get_predict(city: str):
    erm = app.state.per_city_erms.get(city) if hasattr(app.state, 'per_city_erms') else None
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")
    logger.info(f"📡 /predict requested for {city}")
    result = erm.predict_future()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return PredictionResponse(**result)

@app.get("/visualize/{city}")
async def visualize_city(city: str):
    # ... (your original visualize_city code remains exactly the same)
    erm = app.state.per_city_erms.get(city) if hasattr(app.state, 'per_city_erms') else None
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")
    try:
        logger.info(f"📊 Generating visualization for {city} (records: {len(erm.history)})")

        if len(erm.history) < 5:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Not enough data yet.\nRun a few /update calls first.", ha="center", va="center", fontsize=14, color="#ffaa00")
            ax.axis("off")
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return {"visualization": {"dashboard_png_base64": base64.b64encode(buf.read()).decode("utf-8")}}

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"ERM v{settings.VERSION} — {city} Live Dashboard", fontsize=16, color="#00ff88")

        axs[0, 0].plot(list(erm.history), color="#00ff88", linewidth=2, label="Live Temp")
        if erm.last_predicted is not None:
            axs[0, 0].axhline(erm.last_predicted, color="#ffaa00", linestyle="--", label="Last Prediction")
        axs[0, 0].set_title("Temperature History")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        regimes = list(erm.regime_tracker.keys()) or ["stable"]
        success = [erm.regime_tracker.get(r, {"success": 0})["success"] for r in regimes]
        axs[0, 1].bar(regimes, success, color="#00ff88")
        axs[0, 1].set_title("Regime Success Rate")

        axs[1, 0].bar(["Persistence", "Linear", "ERM"], [2.1, 1.8, erm.performance_score * 2], color=["#666", "#666", "#00ff88"])
        axs[1, 0].set_title("Benchmark MAE (lower = better)")

        axs[1, 1].text(0.5, 0.5, f"Confidence\n{round(erm.performance_score*100, 1)}%", ha="center", va="center", fontsize=24, color="#00ff88")
        axs[1, 1].axis("off")
        axs[1, 1].set_title("Overall Confidence")

        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "visualization": {
                "dashboard_png_base64": img_base64,
                "records": len(erm.history),
                "current_regime": erm.current_regime
            }
        }
    except Exception as e:
        logger.error(f"Visualize failed for {city}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
