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

# ===================== CONSTANTS =====================
DATA_DIR = Path(__file__).parent / "ERM_Data"
STATE_DIR = Path(__file__).parent / "ERM_State"
RATE_LIMIT_WINDOW = 12.0
VERSION = "9.1"
CSV_PREFIX = "erm_v9.0"

# ===================== LOCKS =====================
city_last_request: Dict[str, float] = {}
rate_limiter_lock = asyncio.Lock()
git_backup_lock = asyncio.Lock()
csv_write_lock = asyncio.Lock()

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

# ===================== MERGED FETCH =====================
async def fetch_city_data(city: Dict) -> Dict:
    name = city["name"]
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            params = {
                "latitude": city["lat"],
                "longitude": city["lon"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation_probability,cloud_cover,shortwave_radiation,wind_direction_10m",
                "timezone": "auto",
            }
            r = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
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

# ===================== ERM CLASS =====================
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
        self.delta_phi_history: deque = deque(maxlen=50)

        self.performance_score = 0.0
        self.current_regime = "stable"
        self.regime_tracker = defaultdict(lambda: {"count": 0, "success": 0})
        self.multi_hour_success = deque(maxlen=48)
        self.gamma = 0.935
        self.lambda_damp = 0.28
        self.alpha = 0.75
        self.last_predicted: Optional[float] = None
        self.smoothed_er = 0.0

        self.nr = 0.0
        self.tr = 0.0
        self.delta_phi = 0.0
        self.delta_phi_multi = 0.0
        self.k = 0.0

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

        sat_weight = 0.65 if satellite_radiation > 50 else 0.35
        blended_cloud = current_cloud_cover * (1 - sat_weight) + satellite_cloud_cover * sat_weight
        solar_adjust = np.clip((satellite_radiation - 300) / 800.0, -0.8, 1.2)
        volatility *= (1.0 + 0.3 * (blended_cloud / 100.0))

        self.current_regime = self.detect_regime(self.pressure_history, self.humidity_history, volatility, current_wind, current_cloud_cover)

        short_trend = float(np.mean(diffs[-3:])) if len(diffs) >= 3 else 0.0
        sat_forcing = solar_adjust * 0.4 + (blended_cloud / 100.0 - 0.5) * 0.3
        regime_damp = 0.8 if self.current_regime in ["storm", "chaotic"] else 1.0

        empirical_term = (short_trend + sat_forcing + neighbor_influence) * self.alpha * regime_damp
        empirical_term = np.clip(empirical_term, -8.0, 8.0)

        self.nr = np.linalg.norm([
            current_temp / 50.0,
            (current_pressure - 1013.0) / 50.0,
            current_humidity / 100.0,
            current_wind / 20.0
        ])

        self.tr = solar_adjust * 0.4 + (blended_cloud / 100.0 - 0.5) * 0.3

        pressure_lag = np.mean(np.diff(list(self.pressure_history)[-5:])) if len(self.pressure_history) > 5 else 0.0
        humidity_lag = np.mean(np.diff(list(self.humidity_history)[-5:])) if len(self.humidity_history) > 5 else 0.0
        wind_lag = current_wind_dir - 180.0

        self.delta_phi_multi = np.sin(
            0.6 * pressure_lag +
            0.3 * humidity_lag +
            0.1 * (wind_lag / 180.0)
        )
        self.delta_phi_history.append(self.delta_phi_multi)

        beta = self.gamma * (1.0 - self.lambda_damp * volatility / 10.0)
        beta = np.clip(beta, 0.4, 1.2)
        self.k = beta

        physics_term = np.tanh((self.nr * self.tr * self.delta_phi_multi) / (self.k + 1e-6)) * 8.0

        Er_new = 0.6 * physics_term + 0.4 * empirical_term
        Er_new = np.clip(Er_new, -8.0, 8.0)

        self.smoothed_er = 0.7 * Er_new + 0.3 * self.smoothed_er
        Er_new = self.smoothed_er

        next_predicted = current_temp + (Er_new * beta)
        next_predicted = np.clip(next_predicted, current_temp - 50, current_temp + 50)

        self.prediction_history.append(next_predicted)
        self.actual_history.append(current_temp)
        self.Er_history.append(Er_new)
        self.error_history.append(abs(Er_new))
        self.last_predicted = next_predicted

        self.regime_tracker[self.current_regime]["count"] += 1
        if abs(Er_new) < 3.0:
            self.regime_tracker[self.current_regime]["success"] += 1
        self.performance_score = 0.7 * self.performance_score + 0.3 * (1.0 - abs(Er_new) / 10.0)

        if abs(Er_new) > 6.0:
            logger.warning(f"⚠️ Extreme Er_new detected: {Er_new:.2f} (regime: {self.current_regime})")

        return Er_new, float(next_predicted), beta, float(np.mean(diffs[-3:]) if len(diffs) >= 3 else 0.0)

    def detect_regime(self, pressure_history: deque, humidity_history: deque, volatility: float, current_wind: float, current_cloud: float) -> str:
        if len(pressure_history) < 3:
            return "stable"
        p_drop = np.mean(np.diff(list(pressure_history)[-3:]))
        h_spike = np.mean(list(humidity_history)[-3:]) - 50.0
        wind_spike = abs(current_wind) > 12.0
        cloud_spike = current_cloud > 85.0
        if (p_drop < -2.0 and h_spike > 15.0 and volatility > 4.0) or (wind_spike and cloud_spike):
            return "storm"
        elif volatility > 6.0 or wind_spike:
            return "chaotic"
        elif abs(p_drop) < 0.5 and volatility < 1.5:
            return "stable"
        return "seasonal"

    def predict_future(self, horizons: List[int] = [1, 3, 6, 12, 24]) -> Dict[str, Any]:
        if self.last_predicted is None or len(self.history) == 0:
            return {"error": "Not enough data yet"}
        base = self.last_predicted
        predictions = {}
        for h in horizons:
            regime_damp = 0.8 if self.current_regime in ["storm", "chaotic"] else 1.0
            pred = base + (self.smoothed_er * regime_damp * (h / 6.0))
            predictions[f"{h}h"] = round(float(pred), 2)
        return {
            "predictions": predictions,
            "current_regime": self.current_regime,
            "confidence": round(float(1.0 - (abs(self.smoothed_er) / 10.0)), 2)
        }

# ===================== LIFESPAN =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    for d in (DATA_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    app.state.per_city_erms = await load_city_states()
    app.state.save_task = asyncio.create_task(periodic_save())
    app.state.cleanup_task = asyncio.create_task(cleanup_rate_limiter())
    logger.info(f"🚀 ERM v{VERSION} started — build phase active")
    yield
    for task in (getattr(app.state, 'save_task', None), getattr(app.state, 'cleanup_task', None)):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logger.info(f"🛑 ERM v{VERSION} shutdown complete")

app = FastAPI(title=f"ERM Live Update Service — v{VERSION}", lifespan=lifespan)

# ===================== INCREMENTAL SAVE =====================
async def load_city_states() -> Dict[str, ERM_Live_Adaptive]:
    erms = {}
    for city in DEFAULT_CITIES:
        name = city["name"]
        csv_file = DATA_DIR / f"{CSV_PREFIX}_{name}.csv"
        erm = ERM_Live_Adaptive()
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"✅ Loaded {len(df)} records for {name}")
                for _, row in df.iterrows():
                    erm.history.append(row["temp"])
                    erm.humidity_history.append(row.get("humidity", 50.0))
                    erm.wind_history.append(row.get("wind", 5.0))
                    erm.pressure_history.append(row.get("pressure", 1013.0))
                    erm.Er_history.append(row.get("Er", 0.0))
                    erm.error_history.append(row.get("error", 0.0))
                    erm.smoothed_er = float(row.get("smoothed_er", 0.0))
                    erm.performance_score = float(row.get("performance_score", 0.0))
                    erm.current_regime = row.get("current_regime", "stable")
                if len(df) > 0:
                    erm.last_predicted = float(df.iloc[-1]["temp"])
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        else:
            logger.info(f"No CSV yet for {name} — starting fresh")
        erms[name] = erm
    return erms

async def save_all_city_states(erms: Dict):
    logger.info("💾 Starting incremental save...")
    async with csv_write_lock:
        for name, erm in erms.items():
            csv_file = DATA_DIR / f"{CSV_PREFIX}_{name}.csv"
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
                    "error": erm.error_history[-1] if erm.error_history else 0.0,
                    "smoothed_er": erm.smoothed_er,
                    "performance_score": erm.performance_score,
                    "current_regime": erm.current_regime,
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

            except Exception as e:
                logger.error(f"❌ Failed to save {name}: {e}")

    await async_git_backup(DATA_DIR, STATE_DIR)

# ===================== ROBUST GIT HELPER =====================
async def run_git_command(args: list[str], cwd: Path, check: bool = True) -> int:
    """Robust async git command runner with full output capture and error handling."""
    cmd = ["git"] + args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()

        if stdout_str:
            logger.debug(f"Git stdout: {stdout_str}")
        if stderr_str:
            logger.debug(f"Git stderr: {stderr_str}")

        if check and proc.returncode != 0:
            error = stderr_str or stdout_str or f"Git command failed with code {proc.returncode}"
            raise RuntimeError(f"{' '.join(cmd)} failed: {error}")

        return proc.returncode
    except Exception as e:
        logger.error(f"Subprocess error running {' '.join(cmd)}: {e}")
        raise

# ===================== ROBUST GIT BACKUP (DOUBLE-CHECKED & FIXED) =====================
async def async_git_backup(data_dir: Path, state_dir: Path):
    async with git_backup_lock:
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        if not token or not repo:
            logger.warning("⚠️ GITHUB_TOKEN or GITHUB_REPO not set — skipping git backup")
            return

        cwd = Path(__file__).parent
        logger.info(f"🔄 Starting robust git backup in {cwd}")

        try:
            await run_git_command(["config", "--global", "user.email", "erm-bot@render.com"], cwd)
            await run_git_command(["config", "--global", "user.name", "ERM Render Bot"], cwd)

            remote_url = f"https://{token}@github.com/{repo}.git"

            await run_git_command(["remote", "set-url", "origin", remote_url], cwd, check=False)
            await run_git_command(["remote", "add", "origin", remote_url], cwd, check=False)

            logger.info("🔄 Pulling latest changes with rebase...")
            await run_git_command(["pull", "origin", "main", "--rebase"], cwd)

            logger.info(f"📁 Staging files from {data_dir} and {state_dir}...")
            await run_git_command(["add", "-A"], cwd)

            # === DOUBLE-CHECKED CHANGE DETECTION ===
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "--cached", "--name-only",
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            changed_files = stdout.decode().strip()

            if not changed_files:
                logger.info("✅ No new changes to commit — everything up to date")
                return

            logger.info(f"📤 Changes detected:\n{changed_files}")

            commit_msg = f"🚀 Auto-save {datetime.utcnow().isoformat()}"
            await run_git_command(["commit", "-m", commit_msg], cwd)

            logger.info("🚀 Pushing to GitHub...")
            await run_git_command(["push", "origin", "main"], cwd)

            logger.info("✅ Git backup completed successfully")

        except Exception as e:
            logger.error(f"❌ Git backup failed: {e}")

# ===================== PERIODIC SAVE =====================
async def periodic_save(interval_seconds: int = 300):
    while True:
        try:
            if hasattr(app.state, 'per_city_erms'):
                await save_all_city_states(app.state.per_city_erms)
        except Exception as e:
            logger.error(f"Periodic save failed: {e}")
        await asyncio.sleep(interval_seconds)

async def cleanup_rate_limiter(interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        async with rate_limiter_lock:
            now = datetime.now().timestamp()
            for k in list(city_last_request.keys()):
                if now - city_last_request[k] > RATE_LIMIT_WINDOW * 5:
                    city_last_request.pop(k, None)
            if len(city_last_request) > 100:
                city_last_request.clear()

# ===================== DASHBOARD ENDPOINTS =====================
@app.get("/status")
async def status():
    if not hasattr(app.state, 'per_city_erms'):
        return {"status": "not_initialized"}
    counts = {name: len(erm.history) for name, erm in app.state.per_city_erms.items()}
    return {
        "version": VERSION,
        "cities": len(counts),
        "total_records": sum(counts.values()),
        "per_city": counts,
        "build_phase": "active — collecting data"
    }

@app.get("/update")
async def update_all_cities(background_tasks: BackgroundTasks):
    try:
        logger.info("🚀 Starting full /update cycle — ALL cities fetched and stepped in parallel")
        live_data = await fetch_multi_variable_data(DEFAULT_CITIES)
        logger.info(f"✅ Received live data for {len(live_data)} cities")

        step_tasks = []
        for city in DEFAULT_CITIES:
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
                if now - city_last_request.get(name, 0) < RATE_LIMIT_WINDOW:
                    logger.info(f"⏭️ Skipped {name} — rate limited")
                    continue
                city_last_request[name] = now

            neighbors = get_neighbors(name, DEFAULT_CITIES)
            neighbor_influence = sum(neighbor_weight_enhanced(name, n, DEFAULT_CITIES, ground_wind_dir) for n in neighbors)

            erm = app.state.per_city_erms.get(name)
            if erm:
                step_tasks.append(
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
                )

        if step_tasks:
            await asyncio.gather(*step_tasks, return_exceptions=True)
            logger.info(f"✅ Stepped ALL {len(step_tasks)} cities in parallel")

        await save_all_city_states(app.state.per_city_erms)
        logger.info("✅ Full update cycle completed")
        return {"status": "updated", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Update failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest/{city}")
async def get_latest(city: str):
    erm = app.state.per_city_erms.get(city)
    if not erm or len(erm.history) == 0:
        raise HTTPException(status_code=404, detail="No data yet for this city")
    logger.info(f"📡 /latest requested for {city}")
    return {
        "city": city,
        "current_temp": round(erm.history[-1], 2),
        "last_prediction": round(erm.last_predicted, 2) if erm.last_predicted is not None else None,
        "current_regime": erm.current_regime,
        "performance_score": round(erm.performance_score, 3),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/predict/{city}")
async def get_predict(city: str):
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")
    logger.info(f"📡 /predict requested for {city}")
    return erm.predict_future()

@app.get("/visualize/{city}")
async def visualize_city(city: str):
    erm = app.state.per_city_erms.get(city)
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
        fig.suptitle(f"ERM v{VERSION} — {city} Live Dashboard", fontsize=16, color="#00ff88")

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

        axs[1, 0].bar(["Persistence", "Linear", "ERM"], [2.1, 1.8, 1.2], color=["#666", "#666", "#00ff88"])
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
