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
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Dict, Optional, Any
import math
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import json

# Phase 1: Modular ERM v3 model
from erm_model import ERM_Live_Adaptive

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== SETTINGS =====================
class Settings:
    DATA_DIR = Path(__file__).parent / "ERM_Data"
    STATE_DIR = Path(__file__).parent / "ERM_State"
    CITIES_CONFIG = Path(__file__).parent / "cities.json"

    RATE_LIMIT_WINDOW = 45.0
    VERSION = "9.6"
    CSV_PREFIX = "erm_v9.0"
    HISTORY_SIZE = 24
    SAVE_INTERVAL_SEC = 300
    AUTO_UPDATE_INTERVAL_MIN = 10
    MAX_CSV_LOAD_RECORDS = 100

    # Anomaly thresholds
    ANOMALY_ER_THRESHOLD = 0.80
    ANOMALY_TEMP_JUMP = 5.0
    ANOMALY_PERF_DROP = 0.25
    ANOMALY_WINDOW = 12

settings = Settings()

# ===================== CITY CONFIG + NEIGHBOR GRAPH =====================
def load_cities_config() -> List[Dict]:
    cities_json_env = os.getenv("CITIES_JSON")
    if cities_json_env:
        try:
            cities = json.loads(cities_json_env)
            logger.info(f"✅ Loaded {len(cities)} cities from CITIES_JSON env")
            return cities
        except Exception as e:
            logger.error(f"Failed to parse CITIES_JSON: {e}")

    if settings.CITIES_CONFIG.exists():
        try:
            with open(settings.CITIES_CONFIG, "r", encoding="utf-8") as f:
                cities = json.load(f)
            logger.info(f"✅ Loaded {len(cities)} cities from cities.json")
            return cities
        except Exception as e:
            logger.error(f"Failed to load cities.json: {e}")

    logger.warning("⚠️ Using built-in fallback (18 cities)")
    return [  # original 18 cities
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

def build_neighbor_graph(cities: List[Dict], radius_km: float = 150.0) -> Dict[str, List[Dict]]:
    graph = {}
    for target in cities:
        name = target["name"]
        graph[name] = [c for c in cities if c["name"] != name and haversine(target["lat"], target["lon"], c["lat"], c["lon"]) <= radius_km]
    logger.info(f"📍 Built neighbor graph for {len(cities)} cities")
    return graph

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

def neighbor_weight_enhanced(city_name: str, neighbor: Dict, cities: List[Dict], wind_dir: float) -> float:
    target = next((c for c in cities if c["name"] == city_name), None)
    if not target:
        return 0.0
    d = haversine(target["lat"], target["lon"], neighbor["lat"], neighbor["lon"])
    bearing = calculate_bearing(target["lat"], target["lon"], neighbor["lat"], neighbor["lon"])
    alignment = max(0.0, np.cos(np.radians(abs(wind_dir - bearing))))
    return (1.0 / (1.0 + d)) * alignment * 0.35

# ===================== ANOMALY TRACKER =====================
class AnomalyTracker:
    def __init__(self):
        self.anomalies = defaultdict(lambda: deque(maxlen=settings.ANOMALY_WINDOW))

    def record(self, city: str, er: float, perf: float, regime: str, temp_jump: float = 0.0):
        is_anomalous = (
            abs(er) > settings.ANOMALY_ER_THRESHOLD or
            perf < (1.0 - settings.ANOMALY_PERF_DROP) or
            abs(temp_jump) > settings.ANOMALY_TEMP_JUMP
        )
        self.anomalies[city].append({
            "timestamp": datetime.utcnow().isoformat(),
            "er": round(er, 4),
            "performance": round(perf, 3),
            "regime": regime,
            "temp_jump": round(temp_jump, 2),
            "anomalous": is_anomalous
        })

    def get_city_status(self, city: str) -> Dict:
        recent = list(self.anomalies[city])
        if not recent:
            return {"status": "normal", "anomaly_rate": 0.0}
        anomalous = sum(1 for x in recent if x["anomalous"])
        return {
            "status": "ANOMALY" if anomalous >= 2 else "normal",
            "anomaly_rate": round(anomalous / len(recent), 3),
            "recent_count": len(recent)
        }

# ===================== SHARED STATE =====================
city_last_request: Dict[str, float] = {}
rate_limiter_lock = asyncio.Lock()
git_backup_lock = asyncio.Lock()
csv_write_lock = asyncio.Lock()
http_client: Optional[httpx.AsyncClient] = None
anomaly_tracker = AnomalyTracker()

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

# ===================== LIFESPAN =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    for d in (settings.DATA_DIR, settings.STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    cities = load_cities_config()
    app.state.cities_config = cities
    app.state.neighbor_graph = build_neighbor_graph(cities)

    http_client = httpx.AsyncClient(timeout=8.0, limits=httpx.Limits(max_connections=20))

    app.state.per_city_erms = await load_city_states(cities)
    app.state.anomaly_tracker = anomaly_tracker

    app.state.save_task = asyncio.create_task(periodic_save())
    app.state.cleanup_task = asyncio.create_task(cleanup_rate_limiter())
    app.state.auto_update_task = asyncio.create_task(periodic_auto_update())

    logger.info(f"🚀 ERM v{settings.VERSION} (Phase 2 + anomalies + precomputed neighbors) started")
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
    logger.info("🛑 ERM shutdown complete")

app = FastAPI(title=f"ERM Live Update Service — v{settings.VERSION}", lifespan=lifespan)

# ===================== LOAD CITY STATES =====================
async def load_city_states(cities: List[Dict]) -> Dict[str, ERM_Live_Adaptive]:
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

# ===================== SAVE ALL CITY STATES =====================
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
                    "local_climatology": erm.local_climatology,
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
                    logger.info(f"✅ Appended record for {name}")
                    saved_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to save {name}: {e}")
        logger.info(f"💾 Incremental save finished — {saved_count} cities updated")

    await async_git_backup(settings.DATA_DIR, settings.STATE_DIR)

# ===================== GIT BACKUP =====================
async def run_git_command(args: list[str], cwd: Path, check: bool = True) -> int:
    cmd = ["git"] + args
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if check and proc.returncode != 0:
            raise RuntimeError(f"Git command failed: {stderr.decode()}")
        return proc.returncode
    except Exception as e:
        logger.error(f"Git error: {e}")
        raise

async def async_git_backup(data_dir: Path, state_dir: Path):
    async with git_backup_lock:
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        if not token or not repo:
            logger.warning("⚠️ GITHUB_TOKEN or GITHUB_REPO not set — skipping git backup")
            return
        cwd = Path(__file__).parent
        try:
            await run_git_command(["config", "--global", "user.email", "erm-bot@render.com"], cwd)
            await run_git_command(["config", "--global", "user.name", "ERM Render Bot"], cwd)
            remote_url = f"https://{token}@github.com/{repo}.git"
            await run_git_command(["remote", "set-url", "origin", remote_url], cwd, check=False)
            await run_git_command(["pull", "origin", "main", "--rebase"], cwd, check=False)
            await run_git_command(["add", "-A"], cwd)
            proc = await asyncio.create_subprocess_exec("git", "diff", "--cached", "--name-only", cwd=cwd, stdout=asyncio.subprocess.PIPE)
            stdout, _ = await proc.communicate()
            if not stdout.decode().strip():
                logger.info("✅ No new changes to commit")
                return
            await run_git_command(["commit", "-m", f"🚀 Auto-save {datetime.utcnow().isoformat()}"], cwd)
            await run_git_command(["push", "origin", "main"], cwd)
            logger.info("✅ Git backup completed")
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

# ===================== RESPONSE MODELS =====================
class CityData(BaseModel):
    city: str
    current_temp: float
    last_prediction: Optional[float] = None
    current_regime: str
    performance_score: float
    timestamp: str
    anomaly_status: str = "normal"   # ← added for your enhancement

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

# ===================== OPTIMIZED UPDATE ENDPOINT =====================
@app.get("/update")
async def update_all_cities(background_tasks: BackgroundTasks):
    try:
        logger.info("🚀 Starting optimized simultaneous update cycle")
        cities = app.state.cities_config
        live_data = await fetch_multi_variable_data(cities)

        # Pre-compute neighbor influences
        neighbor_influences = {}
        for city in cities:
            name = city["name"]
            ground = live_data.get(name, {})
            wind_dir = ground.get("wind_dir", 180.0)
            neighbors = app.state.neighbor_graph.get(name, [])
            neighbor_influences[name] = sum(neighbor_weight_enhanced(name, n, cities, wind_dir) for n in neighbors)

        for city in cities:
            name = city["name"]
            ground = live_data.get(name, {})

            now = datetime.now().timestamp()
            async with rate_limiter_lock:
                if now - city_last_request.get(name, 0) < settings.RATE_LIMIT_WINDOW:
                    continue
                city_last_request[name] = now

            erm = app.state.per_city_erms.get(name)
            if not erm:
                continue

            prev_temp = erm.history[-1] if erm.history else ground.get("temp", 15.0)

            erm.step(
                current_temp=ground.get("temp", 15.0),
                current_humidity=ground.get("humidity", 50.0),
                current_wind=ground.get("wind", 5.0),
                current_pressure=ground.get("pressure", 1013.0),
                current_rain_prob=ground.get("rain_prob", 0.0),
                current_cloud_cover=ground.get("cloud_cover", 30.0),
                current_solar=ground.get("solar", 400.0),
                current_wind_dir=ground.get("wind_dir", 180.0),
                satellite_cloud_cover=ground.get("cloud_cover", 30.0),
                satellite_radiation=ground.get("solar", 400.0),
                hour_of_day=datetime.now().hour,
                local_avg_temp=city.get("local_avg_temp", 15.0),
                neighbor_influence=neighbor_influences[name],
                dry_run=False
            )

            temp_jump = abs(erm.history[-1] - prev_temp) if erm.history else 0.0
            app.state.anomaly_tracker.record(
                name,
                erm.Er_history[-1] if erm.Er_history else 0.0,
                erm.performance_score,
                erm.current_regime,
                temp_jump
            )

        await save_all_city_states(app.state.per_city_erms)
        logger.info("✅ Optimized update + anomaly tracking completed")
        return {"status": "updated", "timestamp": datetime.utcnow().isoformat(), "model": "ERM_v3", "cities": len(cities)}
    except Exception as e:
        logger.error(f"Update failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== ANOMALY ENDPOINT =====================
@app.get("/anomalies")
async def get_anomalies():
    return {
        city["name"]: app.state.anomaly_tracker.get_city_status(city["name"])
        for city in app.state.cities_config
    }

# ===================== DASHBOARD ENDPOINTS =====================
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
        build_phase="active — anomalies enabled"
    )

@app.get("/latest/{city}", response_model=CityData)
async def get_latest(city: str):
    erm = app.state.per_city_erms.get(city) if hasattr(app.state, 'per_city_erms') else None
    if not erm or len(erm.history) == 0:
        raise HTTPException(status_code=404, detail="No data yet for this city")
    anomaly = app.state.anomaly_tracker.get_city_status(city)
    return CityData(
        city=city,
        current_temp=round(erm.history[-1], 2),
        last_prediction=round(erm.last_predicted, 2) if getattr(erm, 'last_predicted', None) is not None else None,
        current_regime=erm.current_regime,
        performance_score=round(erm.performance_score, 3),
        timestamp=datetime.utcnow().isoformat(),
        anomaly_status=anomaly["status"]
    )

@app.get("/predict/{city}", response_model=PredictionResponse)
async def get_predict(city: str):
    erm = app.state.per_city_erms.get(city) if hasattr(app.state, 'per_city_erms') else None
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")
    result = erm.predict_future()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return PredictionResponse(**result)

@app.get("/visualize/{city}")
async def visualize_city(city: str):
    # (your original visualization code — unchanged)
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
        if getattr(erm, 'last_predicted', None) is not None:
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
        logger.error(f"Visualize failed for {city}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
