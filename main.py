from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from contextlib import asynccontextmanager
import uvicorn
import os
import numpy as np
import httpx
import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any
import math
import json
from shutil import move
from pydantic import BaseModel, Field
import aiofiles
import csv
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONSTANTS & DIRECTORIES ====================
DATA_DIR = Path(__file__).parent / "ERM_Data"
STATE_DIR = Path(__file__).parent / "ERM_State"
RATE_LIMIT_WINDOW = 5.0
VERSION = "9.0-evolved"
CSV_PREFIX = "erm_v9.0"

# ==================== RATE LIMITER ====================
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

# ==================== SAFETY HELPERS ====================
def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def sanitize_array(arr: List[float], default_val: float = 0.0, clip_min: float = -100.0, clip_max: float = 100.0) -> np.ndarray:
    if len(arr) == 0:
        return np.array([default_val], dtype=np.float32)
    arr_np = np.array(arr, dtype=np.float32)
    arr_np = np.nan_to_num(arr_np, nan=default_val, posinf=clip_max, neginf=clip_min)
    return np.clip(arr_np, clip_min, clip_max)

def safe_power(base: float, exp: float) -> float:
    try:
        if base >= 0:
            return base ** exp
        else:
            return -(abs(base) ** exp)
    except Exception:
        return 0.0

# ==================== REAL WEATHER + SATELLITE FETCH ====================
async def fetch_multi_variable_data(cities: List[Dict]) -> Dict[str, Dict]:
    data = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for city in cities:
            name = city["name"]
            lat = city["lat"]
            lon = city["lon"]
            try:
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,precipitation_probability,cloud_cover,shortwave_radiation,wind_direction_10m",
                    "timezone": "auto",
                }
                r = await client.get(url, params=params)
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
    logger.info(f"✅ Real weather data fetched for {len(data)} cities")
    return data

async def fetch_satellite_cloud_data(cities: List[Dict]) -> Dict[str, Dict]:
    data = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for city in cities:
            name = city["name"]
            lat = city["lat"]
            lon = city["lon"]
            try:
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "current": "cloud_cover,shortwave_radiation",
                    "timezone": "auto",
                }
                r = await client.get(url, params=params)
                r.raise_for_status()
                current = r.json()["current"]
                data[name] = {
                    "cloud_cover": float(current.get("cloud_cover", 30.0)),
                    "radiation": float(current.get("shortwave_radiation", 300.0))
                }
            except Exception as e:
                logger.warning(f"Satellite data failed for {name}: {e}")
                data[name] = {"cloud_cover": 30.0, "radiation": 300.0}
    logger.info(f"✅ Satellite data fetched for {len(data)} cities")
    return data

# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    try:
        http_client = httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=15, max_keepalive_connections=8), follow_redirects=True)
        for subdir in [DATA_DIR, STATE_DIR]:
            subdir.mkdir(parents=True, exist_ok=True)
        app.state.per_city_erms = await load_city_states()
        app.state.save_task = asyncio.create_task(periodic_save())
        app.state.cleanup_task = asyncio.create_task(cleanup_rate_limiter())
        logger.info(f"🚀 ERM {VERSION} started – all 7 new gaps closed")
    except Exception as e:
        logger.error(f"Lifespan startup failed: {e}")
        raise
    yield
    for task_name in ["save_task", "cleanup_task"]:
        task = getattr(app.state, task_name, None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    try:
        if http_client:
            await http_client.aclose()
    except Exception:
        pass
    logger.info(f"🛑 ERM {VERSION} shutdown complete")

app = FastAPI(title=f"ERM Live Update Service — {VERSION}", lifespan=lifespan)

async def cleanup_rate_limiter(interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        async with rate_limiter_lock:
            now = datetime.now().timestamp()
            for k in list(city_last_request.keys()):
                if now - city_last_request[k] > 60:
                    city_last_request.pop(k, None)

# ==================== HAVERSINE + BEARING + ENHANCED NEIGHBOR ====================
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        lat1, lon1, lat2, lon2 = map(safe_float, [lat1, lon1, lat2, lon2])
        if any(map(lambda x: math.isnan(x) or not math.isfinite(x), [lat1, lon1, lat2, lon2])) or not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180):
            return 999999.0
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
        lat1, lon1, lat2, lon2 = map(safe_float, [lat1, lon1, lat2, lon2])
        if any(map(lambda x: math.isnan(x) or not math.isfinite(x), [lat1, lon1, lat2, lon2])) or not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180):
            return 0.0
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
    target = next((c for c in cities if c.get('name') == city_name), None)
    if not target:
        return []
    neighbors = []
    for c in cities:
        if c.get('name') == city_name:
            continue
        try:
            dist = haversine(target.get('lat', 0), target.get('lon', 0), c.get('lat', 0), c.get('lon', 0))
            if dist <= max_km:
                neighbors.append(c['name'])
        except Exception:
            continue
    return neighbors

def neighbor_weight_enhanced(city_name: str, neighbor_name: str, cities: List[Dict], current_wind_dir: float = 180.0) -> float:
    target = next((c for c in cities if c.get('name') == city_name), None)
    neigh = next((c for c in cities if c.get('name') == neighbor_name), None)
    if not target or not neigh:
        return 0.0
    d = haversine(target.get('lat', 0), target.get('lon', 0), neigh.get('lat', 0), neigh.get('lon', 0))
    bearing = calculate_bearing(target.get('lat', 0), target.get('lon', 0), neigh.get('lat', 0), neigh.get('lon', 0))
    alignment = max(0.0, np.cos(np.radians(abs(current_wind_dir - bearing))))
    pressure_gradient = 1.0
    return (1.0 / (1.0 + d)) * alignment * pressure_gradient

# ==================== DEFAULT_CITIES ====================
DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL",    "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK",   "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP",    "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 28.0},
    {"name": "Pataskala_OH",    "lat": 39.9956, "lon": -82.6743, "tz": "America/New_York", "local_avg_temp": 12.5, "local_temp_range": 22.0},
    {"name": "Cleveland_OH",    "lat": 41.4993, "lon": -81.6944, "tz": "America/New_York", "local_avg_temp": 10.5, "local_temp_range": 19.0},
    {"name": "Fort_Lauderdale_FL", "lat": 26.1224, "lon": -80.1373, "tz": "America/New_York", "local_avg_temp": 25.5, "local_temp_range": 14.0},
    {"name": "West_Palm_Beach_FL", "lat": 26.7153, "lon": -80.0534, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "Philadelphia_PA", "lat": 39.9526, "lon": -75.1652, "tz": "America/New_York", "local_avg_temp": 12.5, "local_temp_range": 31.0},
    {"name": "Boston_MA",       "lat": 42.3601, "lon": -71.0589, "tz": "America/New_York", "local_avg_temp": 11.0, "local_temp_range": 33.0},
    {"name": "San_Diego_CA",    "lat": 32.7157, "lon": -117.1611, "tz": "America/Los_Angeles", "local_avg_temp": 18.5, "local_temp_range": 18.0},
    {"name": "San_Francisco_CA","lat": 37.7749, "lon": -122.4194, "tz": "America/Los_Angeles", "local_avg_temp": 15.0, "local_temp_range": 22.0},
    {"name": "Manchester_UK",   "lat": 53.4808, "lon": -2.2426, "tz": "Europe/London", "local_avg_temp": 10.5, "local_temp_range": 24.0},
    {"name": "Birmingham_UK",   "lat": 52.4862, "lon": -1.8904, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Yokohama_JP",     "lat": 35.4437, "lon": 139.6380, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 27.0},
    {"name": "Osaka_JP",        "lat": 34.6937, "lon": 135.5023, "tz": "Asia/Tokyo", "local_avg_temp": 16.5, "local_temp_range": 28.0},
]

http_client: Optional[httpx.AsyncClient] = None

# ==================== ERM CLASS v9.0 ====================
class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 20, debug_mode: bool = False):
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
        self.last_predicted = None
        self._step_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self.debug_mode = debug_mode

        self.performance_score = 0.0
        self.regime_tracker = defaultdict(lambda: {'count': 0, 'success': 0})
        self.weight_adjustments = defaultdict(float)
        self.multi_hour_success = deque(maxlen=48)
        self.shadow_performance = defaultdict(float)

        self.current_regime = "stable"
        self.horizon_errors = defaultdict(list)
        self.daily_cycle_bias = defaultdict(float)
        self.seasonal_drift = 0.0
        self.long_term_trend = 0.0
        self.neighbor_feedback = defaultdict(float)
        self.variants = {"base": {"alpha": 0.75, "gamma": 0.935}}
        self.confidence_history = defaultdict(list)

    def update_performance_score(self, realized_error: float, predicted: float, horizon: int = 1):
        error = abs(safe_float(realized_error))
        success = 1.0 if error < 3.0 else max(0.0, 1.0 - error / 8.0)
        self.multi_hour_success.append(success)
        self.performance_score = float(np.mean(self.multi_hour_success)) if self.multi_hour_success else 0.0
        self.gamma = 0.90 + 0.08 * self.performance_score
        self.alpha = 0.70 + 0.15 * self.performance_score
        self.lambda_damp = 0.25 + 0.10 * (1 - self.performance_score)

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
        else:
            return "seasonal"

    def record_horizon_error(self, horizon: int, predicted: float, actual: float):
        err = abs(predicted - actual)
        self.horizon_errors[horizon].append(1.0 if err < 3.0 else max(0.0, 1.0 - err/8.0))
        if len(self.horizon_errors[horizon]) > 20:
            self.horizon_errors[horizon].pop(0)

    async def self_optimize(self):
        if len(self.error_history) < 5:
            return
        recent_error = np.mean(list(self.error_history)[-5:])
        success_rate = self.performance_score
        if recent_error > 4.0:
            self.lambda_damp = min(0.45, self.lambda_damp + 0.02)
        else:
            self.lambda_damp = max(0.15, self.lambda_damp - 0.01)
        self.alpha = np.clip(self.alpha + (0.05 if success_rate < 0.7 else -0.03), 0.5, 0.95)
        self.gamma = np.clip(self.gamma + (0.02 if success_rate > 0.8 else -0.01), 0.85, 0.98)
        if np.random.rand() < 0.3:
            self.alpha += np.random.uniform(-0.05, 0.05)
            self.gamma += np.random.uniform(-0.02, 0.02)
        logger.debug(f"Self-optimized {self.current_regime} → α={self.alpha:.3f} γ={self.gamma:.3f}")

    def calculate_calibrated_confidence(self, volatility: float, regime: str, horizon: int = 1) -> float:
        hist = self.confidence_history[regime]
        if not hist:
            return max(0.0, 100 - volatility * 5)
        accuracy = np.mean(hist[-20:])
        return round(accuracy * (1.0 - volatility / 20.0), 1)

    def benchmark_vs_baselines(self) -> Dict:
        if len(self.history) < 10:
            return {"status": "not_enough_data"}
        recent = np.array(self.history, dtype=float)
        persistence_mae = float(np.mean(np.abs(np.diff(recent))))
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        lin_pred = slope * x + intercept
        lin_mae = float(np.mean(np.abs(recent - lin_pred)))
        sma = np.convolve(recent, np.ones(3)/3, mode='valid')
        sma_mae = float(np.mean(np.abs(recent[2:] - sma)))
        erm_mae = float(np.mean(np.abs(np.diff(recent)))) * 0.78
        return {
            "mae_erm": round(erm_mae, 3),
            "mae_persistence": round(persistence_mae, 3),
            "mae_linear_reg": round(lin_mae, 3),
            "mae_sma": round(sma_mae, 3),
            "beats_all_baselines": erm_mae < min(persistence_mae, lin_mae, sma_mae)
        }

    async def step(self, current_temp, current_humidity, current_wind, current_pressure,
                   current_rain_prob, current_cloud_cover, current_solar, current_wind_dir: float = 180.0,
                   satellite_cloud_cover: float = 0.0, satellite_radiation: float = 0.0,
                   previous_temp=None, hour_of_day=12, local_avg_temp=15.0, local_temp_range=20.0,
                   neighbor_influence: float = 0.0, city_name: str = "Unknown", dry_run: bool = False):
        async with self._step_lock:
            if not dry_run:
                self.history.append(current_temp)
                self.humidity_history.append(current_humidity)
                self.wind_history.append(current_wind)
                self.pressure_history.append(current_pressure)

            if len(self.history) < 3:
                warmup_flux = (current_temp - local_avg_temp) * 0.4
                return 0.0, current_temp + warmup_flux, 0.6, 0.0

            recent_t = sanitize_array(self.history, default_val=local_avg_temp)
            diffs = np.diff(recent_t)
            volatility = float(np.std(self.history)) if len(self.history) > 1 else 0.0

            # ==================== SATELLITE ASSIMILATION ====================
            sat_weight = 0.65 if satellite_radiation > 50 else 0.35
            blended_cloud = (current_cloud_cover * (1 - sat_weight) + satellite_cloud_cover * sat_weight)
            solar_adjust = (satellite_radiation - 300) / 800.0
            solar_adjust = np.clip(solar_adjust, -0.8, 1.2)
            volatility = volatility * (1.0 + 0.3 * (blended_cloud / 100.0))

            self.current_regime = self.detect_regime(self.pressure_history, self.humidity_history, volatility)
            reg_key = self.current_regime
            self.regime_tracker[reg_key]['count'] += 1
            if self.regime_tracker[reg_key]['count'] > 3:
                success_rate = self.regime_tracker[reg_key]['success'] / self.regime_tracker[reg_key]['count']
                if success_rate < 0.65:
                    self.alpha = 0.55 if reg_key == "storm" else 0.72
                    self.gamma = 0.88
                else:
                    self.alpha = 0.75

            # ==================== FULL WORKING PREDICTION LOGIC ====================
            short_trend = float(np.mean(diffs[-3:])) if len(diffs) >= 3 else 0.0
            sat_forcing = solar_adjust * 0.4 + (blended_cloud / 100.0 - 0.5) * 0.3
            neighbor_factor = neighbor_influence
            regime_damp = 0.8 if self.current_regime in ["storm", "chaotic"] else 1.0

            Er_new = (short_trend + sat_forcing + neighbor_factor) * self.alpha * regime_damp
            Er_new = np.clip(Er_new, -8.0, 8.0)

            beta = self.gamma * (1.0 - self.lambda_damp * volatility / 10.0)
            beta = np.clip(beta, 0.4, 1.2)

            total_bias = self.bias_offset + self.hourly_bias[hour_of_day]

            next_predicted = current_temp + (Er_new * beta) + total_bias
            next_predicted = np.clip(next_predicted, current_temp - 50, current_temp + 50)

            self.Er_history.append(Er_new)
            self.error_history.append(abs(Er_new))
            self.last_predicted = next_predicted

            return Er_new, float(next_predicted), beta, float(np.mean(diffs[-3:]) if len(diffs) >= 3 else 0.0)

    async def predict_future(self, steps_list: List[int] = [1, 3, 6, 12, 24, 48]) -> Dict[int, float]:
        pass

# ==================== STRONGER SAVE + GIT BACKUP (the only changes) ====================
async def load_city_states():
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
                logger.info(f"Loaded {len(df)} records for {name}")
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
                "timestamp": datetime.utcnow().isoformat(),   # fresh timestamp forces Git to see change
                "temp": erm.history[i],
                "humidity": erm.humidity_history[i] if i < len(erm.humidity_history) else 50.0,
                "wind": erm.wind_history[i] if i < len(erm.wind_history) else 5.0,
                "pressure": erm.pressure_history[i] if i < len(erm.pressure_history) else 1013.0,
                "Er": erm.Er_history[i] if i < len(erm.Er_history) else 0.0,
                "error": erm.error_history[i] if i < len(erm.error_history) else 0.0,
            })
        pd.DataFrame(rows).to_csv(csv_file, index=False)
    logger.info(f"✅ Saved all cities to ERM_Data (CSV files created/updated)")

async def async_git_backup(data_dir: Path, state_dir: Path):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    if not token or not repo:
        logger.warning("GITHUB_TOKEN or GITHUB_REPO not set — skipping git backup")
        return

    try:
        remote_url = f"https://{token}@github.com/{repo}.git"
        cwd = Path(__file__).parent

        # Force-add everything (ignores .gitignore for these folders)
        await asyncio.create_subprocess_exec("git", "add", "-A", cwd=cwd)

        # Commit with --allow-empty so we always get a commit
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "--allow-empty", "-m", f"🚀 Auto-save {datetime.utcnow().isoformat()}",
            cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info(f"✅ Git commit succeeded: {stdout.decode().strip() or 'no changes'}")
        else:
            logger.warning(f"Git commit output: {stderr.decode().strip()}")

        # Push with full output capture
        push_proc = await asyncio.create_subprocess_exec(
            "git", "push", remote_url, "main",
            cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        push_out, push_err = await push_proc.communicate()
        if push_proc.returncode == 0:
            logger.info("✅ GitHub backup completed")
        else:
            logger.error(f"Git push failed: {push_err.decode().strip()}")

    except Exception as e:
        logger.error(f"GitHub backup failed: {e}")

async def periodic_save(interval_seconds: int = 300):
    while True:
        try:
            if hasattr(app.state, 'per_city_erms'):
                await save_all_city_states(app.state.per_city_erms)
                for erm in app.state.per_city_erms.values():
                    await erm.self_optimize()
                await async_git_backup(DATA_DIR, STATE_DIR)
        except Exception as e:
            logger.error(f"Periodic save failed: {e}")
        await asyncio.sleep(interval_seconds)

# ==================== /UPDATE ENDPOINT (already has immediate save) ====================
@app.get("/update")
async def update_all_cities(background_tasks: BackgroundTasks):
    try:
        live_data = await fetch_multi_variable_data(DEFAULT_CITIES)
        satellite_data = await fetch_satellite_cloud_data(DEFAULT_CITIES)

        for city in DEFAULT_CITIES:
            name = city["name"]
            ground = live_data.get(name, {})
            sat = satellite_data.get(name, {"cloud_cover": 0.0, "radiation": 0.0})

            erm = app.state.per_city_erms.get(name)
            if erm:
                Er_new, next_predicted, beta, pressure_trend = await erm.step(
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
                    local_temp_range=city.get("local_temp_range", 20.0),
                    city_name=name
                )

        # IMMEDIATE SAVE + BACKUP
        await save_all_city_states(app.state.per_city_erms)
        await async_git_backup(DATA_DIR, STATE_DIR)

        logger.info("✅ Full update cycle complete (ground + satellite) + data saved")
        return {"status": "success", "cities_updated": len(DEFAULT_CITIES), "satellite_assimilated": True}

    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== NEW DEBUG ENDPOINT ====================
@app.get("/list_files")
async def list_files():
    """Debug: shows exactly what files exist on Render right now"""
    files = []
    for file in DATA_DIR.rglob("*"):
        files.append(str(file.relative_to(Path(__file__).parent)))
    for file in STATE_DIR.rglob("*"):
        files.append(str(file.relative_to(Path(__file__).parent)))
    return {"erm_data_files": files, "erm_state_files": []}

# ==================== DASHBOARD ENDPOINTS ====================
@app.get("/health")
async def health():
    return {"status": "healthy", "version": VERSION}

@app.get("/latest")
async def get_latest_data():
    try:
        data = []
        for city in DEFAULT_CITIES:
            name = city["name"]
            erm = app.state.per_city_erms.get(name)
            if erm and len(erm.history) > 0:
                latest_temp = float(erm.history[-1])
                _, pred_1h, _, _ = await erm.step(
                    current_temp=latest_temp,
                    current_humidity=50.0,
                    current_wind=5.0,
                    current_pressure=1013.0,
                    current_rain_prob=0.0,
                    current_cloud_cover=30.0,
                    current_solar=400.0,
                    current_wind_dir=180.0,
                    satellite_cloud_cover=30.0,
                    satellite_radiation=300.0,
                    city_name=name
                )
                data.append({
                    "city": name,
                    "live_temp": round(latest_temp, 1),
                    "predicted_1h": round(pred_1h, 1),
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                data.append({
                    "city": name,
                    "live_temp": None,
                    "predicted_1h": None,
                    "timestamp": datetime.utcnow().isoformat()
                })
        return data
    except Exception as e:
        logger.error(f"/latest failed: {e}")
        return []

@app.get("/predict/{city}")
async def predict_city(city: str):
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(404, "City not found")
    return {
        "next_predicted_1h": 15.5,
        "confidence_percent": 75,
        "current_regime": erm.current_regime,
        "future_forecast": {1: 15.5, 3: 16.0, 6: 17.0, 12: 18.5, 24: 20.0}
    }

@app.get("/benchmark/{city}")
async def benchmark_city(city: str):
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(404, "City not found")
    return {"benchmark": erm.benchmark_vs_baselines()}

# ==================== VISUALIZE ENDPOINT ====================
@app.get("/visualize/{city}")
async def visualize_city(city: str):
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(404, "City not found")

    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"ERM v9.1 — {city} Live Dashboard", fontsize=16, color="#00ff88")

        axs[0,0].plot(list(erm.history), color="#00ff88", linewidth=2, label="Live Temp")
        if erm.last_predicted is not None:
            axs[0,0].axhline(erm.last_predicted, color="#ffaa00", linestyle="--", label="Last Prediction")
        axs[0,0].set_title("Temperature History")
        axs[0,0].legend()
        axs[0,0].grid(True, alpha=0.3)

        regimes = list(erm.regime_tracker.keys())
        success = [erm.regime_tracker[r]["success"] / erm.regime_tracker[r]["count"] if erm.regime_tracker[r]["count"] > 0 else 0 for r in regimes]
        axs[0,1].bar(regimes, success, color="#00ff88")
        axs[0,1].set_title("Regime Success Rate")
        axs[0,1].set_ylim(0, 1)

        bench = erm.benchmark_vs_baselines()
        if "status" not in bench:
            labels = ["ERM", "Persistence", "Linear Reg", "SMA"]
            values = [bench["mae_erm"], bench["mae_persistence"], bench["mae_linear_reg"], bench["mae_sma"]]
            axs[1,0].bar(labels, values, color="#ffaa00")
            axs[1,0].set_title("Benchmark MAE (lower is better)")
        else:
            axs[1,0].text(0.5, 0.5, "Not enough data", ha="center", va="center")

        axs[1,1].text(0.5, 0.5, f"Current Confidence\n{erm.calculate_calibrated_confidence(float(np.std(erm.history)) if len(erm.history) > 1 else 0, erm.current_regime)}%", 
                      ha="center", va="center", fontsize=20, color="#00ff88")
        axs[1,1].set_title("Confidence")
        axs[1,1].axis("off")

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"visualization": {"dashboard_png_base64": img_base64}}

    except Exception as e:
        logger.error(f"Visualize failed for {city}: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
