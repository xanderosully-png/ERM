from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
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
import json
import sqlite3
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import plotly.graph_objects as go
import plotly.io as pio

# Phase 1: Modular ERM v3 model
from erm_model import ERM_Live_Adaptive

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== SETTINGS =====================
class Settings:
    DATA_DIR = Path(__file__).parent / "ERM_Data"
    STATE_DIR = Path(__file__).parent / "ERM_State"
    CITIES_CONFIG = Path(__file__).parent / "cities.json"
    DB_PATH = Path(__file__).parent / "ERM_Data" / "erm_data.db"
    VISUALIZATION_CACHE_DIR = Path(__file__).parent / "ERM_Data" / "visualizations"

    RATE_LIMIT_WINDOW = 45.0
    VERSION = "10.1"
    CSV_PREFIX = "erm_v10.0"
    HISTORY_SIZE = 24
    SAVE_INTERVAL_SEC = 300
    AUTO_UPDATE_INTERVAL_MIN = 10
    MAX_CSV_LOAD_RECORDS = 100

    ANOMALY_ER_THRESHOLD = 0.80
    ANOMALY_TEMP_JUMP = 5.0
    ANOMALY_PERF_DROP = 0.25
    ANOMALY_WINDOW = 12

    MAX_RETRIES = 5
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 8
    CIRCUIT_BREAKER_RESET_TIMEOUT_SEC = 30   # ← changed to 30 seconds for faster recovery

    VISUALIZATION_CACHE_TTL_MIN = 5

settings = Settings()

# Create cache directory
settings.VISUALIZATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ===================== CIRCUIT BREAKER =====================
class CircuitBreaker:
    def __init__(self):
        self.failures = 0
        self.last_failure_time = 0
        self.open = False

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now().timestamp()
        if self.failures >= settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            self.open = True
            logger.warning("🚨 Circuit breaker OPEN — Open-Meteo calls paused")

    def record_success(self):
        self.failures = 0
        if self.open:
            self.open = False
            logger.info("✅ Circuit breaker CLOSED — Open-Meteo calls resumed")

    def is_open(self):
        if not self.open:
            return False
        if datetime.now().timestamp() - self.last_failure_time > settings.CIRCUIT_BREAKER_RESET_TIMEOUT_SEC:
            self.open = False
            logger.info("🔄 Circuit breaker auto-reset")
        return self.open

circuit_breaker = CircuitBreaker()

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
            raise

    logger.error("❌ No cities configuration found. Place cities.json in the project root or set CITIES_JSON environment variable.")
    raise FileNotFoundError("cities.json or CITIES_JSON env var is required")

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
http_client: Optional[httpx.AsyncClient] = None
anomaly_tracker = AnomalyTracker()

# ===================== SQLITE HELPERS =====================
def init_database():
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            temp REAL,
            humidity REAL,
            wind REAL,
            pressure REAL,
            Er REAL,
            smoothed_er REAL,
            performance_score REAL,
            current_regime TEXT,
            local_climatology REAL,
            UNIQUE(city, timestamp)
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"✅ SQLite database initialized at {settings.DB_PATH}")

async def migrate_from_csvs():
    logger.info("🔄 Checking for CSV → SQLite migration...")
    migrated = 0
    for csv_file in settings.DATA_DIR.glob(f"{settings.CSV_PREFIX}_*.csv"):
        city = csv_file.stem.replace(f"{settings.CSV_PREFIX}_", "")
        try:
            df = pd.read_csv(csv_file)
            conn = sqlite3.connect(settings.DB_PATH)
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR IGNORE INTO records 
                    (city, timestamp, temp, humidity, wind, pressure, Er, smoothed_er, 
                     performance_score, current_regime, local_climatology)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    city,
                    row.get("timestamp", datetime.utcnow().isoformat()),
                    row.get("temp"),
                    row.get("humidity", 50.0),
                    row.get("wind", 5.0),
                    row.get("pressure", 1013.0),
                    row.get("Er", 0.0),
                    row.get("smoothed_er", 0.0),
                    row.get("performance_score", 0.0),
                    row.get("current_regime", "stable"),
                    row.get("local_climatology", 15.0)
                ))
            conn.commit()
            conn.close()
            migrated += 1
        except Exception as e:
            logger.error(f"Migration failed for {city}: {e}")
    if migrated:
        logger.info(f"✅ Migration complete — {migrated} cities moved to SQLite")

# ===================== RESILIENT FETCH =====================
@retry(
    stop=stop_after_attempt(settings.MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(lambda e: isinstance(e, (httpx.RequestError, httpx.HTTPStatusError))),
    reraise=True
)
async def fetch_city_data(city: Dict) -> Dict:
    if circuit_breaker.is_open():
        logger.warning(f"⛔ Circuit breaker open — skipping fetch for {city['name']}")
        return {}
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
        circuit_breaker.record_success()
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
        circuit_breaker.record_failure()
        logger.warning(f"❌ Fetch failed for {name}: {e}")
        raise

async def fetch_multi_variable_data(cities: List[Dict]) -> Dict[str, Dict]:
    tasks = [fetch_city_data(city) for city in cities]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = {}
    for city, result in zip(cities, results):
        name = city["name"]
        data[name] = result if isinstance(result, dict) else {"temp": 15.0, "humidity": 50.0, "wind": 5.0, "pressure": 1013.0, "rain_prob": 0.0, "cloud_cover": 30.0, "solar": 400.0, "wind_dir": 180.0}
    return data

# ===================== CORE UPDATE LOGIC =====================
async def _perform_city_updates(force_update: bool = False):
    logger.info("🚀 Starting optimized simultaneous update cycle")
    cities = app.state.cities_config
    live_data = await fetch_multi_variable_data(cities)

    neighbor_influences = {}
    for city in cities:
        name = city["name"]
        ground = live_data.get(name, {})
        wind_dir = ground.get("wind_dir", 180.0)
        neighbors = app.state.neighbor_graph.get(name, [])
        neighbor_influences[name] = sum(neighbor_weight_enhanced(name, n, cities, wind_dir) for n in neighbors)

    updated_count = 0
    for city in cities:
        name = city["name"]
        ground = live_data.get(name, {})

        if not force_update:
            now = datetime.now().timestamp()
            async with rate_limiter_lock:
                if now - city_last_request.get(name, 0) < settings.RATE_LIMIT_WINDOW:
                    logger.info(f"⏭️ Rate-limited skip for {name}")
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
        updated_count += 1

    await save_all_city_states(app.state.per_city_erms)
    logger.info(f"✅ Update completed — {updated_count}/{len(cities)} cities refreshed")
    return {
        "status": "updated",
        "timestamp": datetime.utcnow().isoformat(),
        "model": "ERM_v3",
        "cities_updated": updated_count,
        "total_cities": len(cities),
        "version": settings.VERSION
    }

# ===================== LIFESPAN =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    for d in (settings.DATA_DIR, settings.STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    init_database()
    await migrate_from_csvs()

    cities = load_cities_config()
    app.state.cities_config = cities
    app.state.neighbor_graph = build_neighbor_graph(cities)

    http_client = httpx.AsyncClient(timeout=8.0, limits=httpx.Limits(max_connections=20))

    app.state.per_city_erms = await load_city_states(cities)
    app.state.anomaly_tracker = anomaly_tracker

    app.state.save_task = asyncio.create_task(periodic_save())
    app.state.cleanup_task = asyncio.create_task(cleanup_rate_limiter())
    app.state.auto_update_task = asyncio.create_task(periodic_auto_update())

    logger.info(f"🚀 ERM v{settings.VERSION} started successfully")
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

# ===================== LOAD / SAVE CITY STATES =====================
async def load_city_states(cities: List[Dict]) -> Dict[str, ERM_Live_Adaptive]:
    erms = {}
    conn = sqlite3.connect(settings.DB_PATH)
    for city in cities:
        name = city["name"]
        erm = ERM_Live_Adaptive(city_name=name)
        df = pd.read_sql_query(
            "SELECT * FROM records WHERE city = ? ORDER BY timestamp ASC LIMIT ?",
            conn, params=(name, settings.MAX_CSV_LOAD_RECORDS)
        )
        if not df.empty:
            logger.info(f"✅ Loaded {len(df)} records for {name} from SQLite")
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

        else:
            logger.info(f"No records yet for {name} — starting fresh")

        erms[name] = erm

    conn.close()
    return erms

async def save_all_city_states(erms: Dict):
    logger.info("💾 Saving to SQLite...")
    conn = sqlite3.connect(settings.DB_PATH)
    saved_count = 0
    for name, erm in erms.items():
        if len(erm.history) == 0:
            continue
        try:
            conn.execute("""
                INSERT OR IGNORE INTO records 
                (city, timestamp, temp, humidity, wind, pressure, Er, smoothed_er, 
                 performance_score, current_regime, local_climatology)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                datetime.utcnow().isoformat(),
                erm.history[-1],
                erm.humidity_history[-1] if erm.humidity_history else 50.0,
                erm.wind_history[-1] if erm.wind_history else 5.0,
                erm.pressure_history[-1] if erm.pressure_history else 1013.0,
                erm.Er_history[-1] if erm.Er_history else 0.0,
                erm.smoothed_er,
                erm.performance_score,
                erm.current_regime,
                erm.local_climatology
            ))
            saved_count += 1
        except Exception as e:
            logger.error(f"SQLite save failed for {name}: {e}")
    conn.commit()
    conn.close()
    logger.info(f"✅ Saved {saved_count} cities to SQLite")
    await async_git_backup(settings.DATA_DIR, settings.STATE_DIR)

# ===================== GIT BACKUP =====================
async def run_git_command(args: list[str], cwd: Path, check: bool = True) -> int:
    cmd = ["git"] + args
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()
        if check and proc.returncode != 0:
            raise RuntimeError(f"Git failed: {stderr_str or stdout_str}")
        return proc.returncode
    except Exception as e:
        logger.error(f"Git subprocess error: {e}")
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
            await _perform_city_updates()
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
    anomaly_status: str = "normal"

class PredictionResponse(BaseModel):
    predictions: Dict[str, Optional[float]]
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

# ===================== ENDPOINTS =====================
@app.get("/update")
async def update_all_cities():
    try:
        return await _perform_city_updates(force_update=True)
    except Exception as e:
        logger.error(f"Update failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anomalies")
async def get_anomalies():
    return {
        city["name"]: app.state.anomaly_tracker.get_city_status(city["name"])
        for city in app.state.cities_config
    }

@app.get("/backtest/{city}")
async def backtest_city(city: str):
    erm = app.state.per_city_erms.get(city) if hasattr(app.state, 'per_city_erms') else None
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")
    if len(erm.history) < 10:
        raise HTTPException(status_code=400, detail="Not enough historical data for back-testing")
    result = erm.replay_history(list(erm.history))
    return {
        "city": city,
        "mae": round(result["mae"], 3),
        "steps": result["steps"],
        "final_regime": result["final_regime"],
        "performance_score": round(result["performance_score"], 3),
        "message": "Back-test completed using historical temperature series"
    }

@app.get("/metrics")
async def metrics():
    if not hasattr(app.state, 'per_city_erms'):
        return {"status": "not_ready"}
    total_cities = len(app.state.per_city_erms)
    total_records = sum(len(erm.history) for erm in app.state.per_city_erms.values())
    avg_perf = round(np.mean([erm.performance_score for erm in app.state.per_city_erms.values()]), 3) if total_cities > 0 else 0.0
    return {
        "version": settings.VERSION,
        "cities": total_cities,
        "total_records": total_records,
        "avg_performance": avg_perf,
        "anomaly_count": sum(len(app.state.anomaly_tracker.anomalies[city]) for city in app.state.cities_config),
        "uptime": "running"
    }

@app.get("/visualize/{city}")
async def visualize_city(city: str):
    erm = app.state.per_city_erms.get(city) if hasattr(app.state, 'per_city_erms') else None
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")

    cache_file = settings.VISUALIZATION_CACHE_DIR / f"{city}.html"
    if cache_file.exists():
        cache_age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 60
        if cache_age < settings.VISUALIZATION_CACHE_TTL_MIN:
            logger.info(f"✅ Serving cached visualization for {city}")
            return HTMLResponse(cache_file.read_text(encoding="utf-8"))

    logger.info(f"📊 Generating fresh Plotly visualization for {city}")

    if len(erm.history) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data yet.<br>Run a few /update calls first.", showarrow=False, font_size=18)
        fig.update_layout(height=400, title=f"ERM v{settings.VERSION} — {city}")
        html_str = pio.to_html(fig, full_html=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=list(erm.history), mode='lines', name='Live Temp', line=dict(color='#00ff88', width=3)))
        if getattr(erm, 'last_predicted', None) is not None:
            fig.add_hline(y=erm.last_predicted, line_dash="dash", line_color="#ffaa00", annotation_text="Last Prediction")
        fig.update_layout(
            title=f"ERM v{settings.VERSION} — {city} Live Dashboard",
            xaxis_title="Time Steps",
            yaxis_title="Temperature (°C)",
            template="plotly_dark",
            height=600
        )
        fig.add_annotation(text=f"Performance: {round(erm.performance_score*100, 1)}%", x=0.5, y=0.9, showarrow=False, font_size=24, font_color="#00ff88")
        html_str = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")

    cache_file.write_text(html_str, encoding="utf-8")
    return HTMLResponse(html_str)

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
        build_phase="active — full Phase 6 (back-testing + metrics + Plotly)"
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
    
    # Fixed: graceful handling when model has no data yet
    if erm.last_predicted is None or len(erm.history) < 4:
        return PredictionResponse(
            predictions={"1h": None, "3h": None, "6h": None, "12h": None, "24h": None},
            current_regime=erm.current_regime,
            confidence=0.0
        )
    
    result = erm.predict_future()
    return PredictionResponse(**result)

# New helper endpoint to manually reset circuit breaker if it gets stuck
@app.get("/reset-circuit")
async def reset_circuit_breaker():
    circuit_breaker.failures = 0
    circuit_breaker.open = False
    logger.info("🔄 Circuit breaker manually reset")
    return {"status": "circuit breaker reset", "open": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
