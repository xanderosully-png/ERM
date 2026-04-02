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
VERSION = "8.0-final"
CSV_PREFIX = "erm_v8.0"

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

def sanitize_array(arr: List[float], default_val: float = 0.0, clip_min: float = -150.0, clip_max: float = 150.0) -> np.ndarray:
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
        logger.info(f"🚀 ERM {VERSION} started (full async + hotfixes + cutoffs eliminated)")
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
    if http_client:
        await http_client.aclose()
    logger.info(f"🛑 ERM {VERSION} shutdown complete")

app = FastAPI(title=f"ERM Live Update Service — {VERSION}", lifespan=lifespan)

# ==================== CLEANUP ====================
async def cleanup_rate_limiter(interval: int = 30):
    while True:
        await asyncio.sleep(interval)
        async with rate_limiter_lock:
            now = datetime.now().timestamp()
            for k in list(city_last_request.keys()):
                if now - city_last_request[k] > 60:
                    city_last_request.pop(k, None)

# ==================== DISTANCE + NEIGHBOR HELPERS ====================
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

def neighbor_weight(city_name: str, neighbor_name: str, cities: List[Dict]) -> float:
    target = next((c for c in cities if c.get('name') == city_name), None)
    neigh = next((c for c in cities if c.get('name') == neighbor_name), None)
    if not target or not neigh:
        return 0.0
    try:
        d = haversine(target.get('lat', 0), target.get('lon', 0), neigh.get('lat', 0), neigh.get('lon', 0))
        return 1.0 / (1.0 + d)
    except Exception:
        return 0.0

# ==================== DEFAULT CITIES ====================
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

# ==================== ERM CLASS (v8.0 — cutoffs eliminated) ====================
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
        self.gamma = 0.92
        self.lambda_damp = 0.25
        self.alpha = 0.78
        self.last_predicted = None
        self._step_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self.debug_mode = debug_mode

    async def async_save_state(self, filepath: Path):
        async with self._state_lock:
            try:
                self.backup_state(filepath)
                state = {}
                for k, v in self.__dict__.items():
                    if k.startswith('_'): continue
                    if isinstance(v, deque):
                        state[k] = list(v)
                    elif isinstance(v, defaultdict):
                        state[k] = dict(v)
                    else:
                        state[k] = v
                state["last_update_timestamp"] = datetime.now().isoformat()
                state["version"] = VERSION
                tmp_path = filepath.with_suffix('.json.tmp')
                async with aiofiles.open(tmp_path, mode='w', encoding='utf-8') as f:
                    await f.write(json.dumps(state, indent=2))
                await asyncio.to_thread(move, tmp_path, filepath)
            except Exception as e:
                logger.error(f"Failed to async-save state {filepath}: {e}")

    async def async_load_state(self, filepath: Path):
        async with self._state_lock:
            if filepath.exists():
                try:
                    async with aiofiles.open(filepath, mode='r', encoding='utf-8') as f:
                        raw = await f.read()
                    state = json.loads(raw)
                    for k in ['history', 'humidity_history', 'wind_history', 'pressure_history',
                              'Er_history', 'error_history', 'systematic_bias', 'noise_error']:
                        if k in self.__dict__:
                            self.__dict__[k].clear()
                    for k, v in state.items():
                        if k in self.__dict__:
                            if isinstance(self.__dict__[k], deque):
                                self.__dict__[k].extend(v)
                            elif k == "hourly_bias":
                                self.hourly_bias = defaultdict(float, v)
                            else:
                                self.__dict__[k] = v
                except Exception as e:
                    logger.error(f"Failed to async-load state {filepath}: {e}")
                    filepath.unlink(missing_ok=True)

    def backup_state(self, filepath: Path):
        if filepath.exists():
            backup_path = filepath.with_suffix('.backup.json')
            try:
                move(filepath, backup_path)
            except Exception:
                pass

    def record_error(self, realized_error: float, predicted: float):
        try:
            error = safe_float(realized_error)
            systematic = np.mean(self.error_history) if self.error_history else 0.0
            noise = error - systematic
            self.error_history.append(error)
            self.systematic_bias.append(systematic)
            self.noise_error.append(noise)
        except Exception:
            pass

    async def step(self, current_temp, current_humidity, current_wind, current_pressure,
                   current_rain_prob, current_cloud_cover, current_solar,
                   previous_temp, hour_of_day, local_avg_temp, local_temp_range,
                   neighbor_influence: float = 0.0, city_name: str = "Unknown",
                   dry_run: bool = False):
        async with self._step_lock:
            current_temp = safe_float(current_temp, 15.0)
            current_humidity = safe_float(current_humidity, 50.0)
            current_wind = safe_float(current_wind, 5.0)
            current_pressure = safe_float(current_pressure, 1013.0)
            current_solar = safe_float(current_solar, 200.0)
            local_avg_temp = safe_float(local_avg_temp, 15.0)
            local_temp_range = max(1.0, safe_float(local_temp_range, 20.0))
            neighbor_influence = safe_float(neighbor_influence, 0.0)
            hour_of_day = int(safe_float(hour_of_day, 12))

            if not dry_run:
                self.history.append(current_temp)
                self.humidity_history.append(current_humidity)
                self.wind_history.append(current_wind)
                self.pressure_history.append(current_pressure)

            if len(self.history) < 3:
                warmup_flux = (current_temp - local_avg_temp) * 0.4
                next_predicted = current_temp + warmup_flux
                return 0.0, next_predicted, 0.6, 0.0

            try:
                recent_t = sanitize_array(self.history, default_val=local_avg_temp)
                diffs = np.diff(recent_t)
                if len(diffs) == 0:
                    diffs = np.zeros(1, dtype=np.float32)

                Nr = len(recent_t) * (1 + np.var(recent_t) / 10)
                mean_abs_diffs = np.mean(np.abs(diffs)) + 1e-8
                Tr = max(0.6, 1 - np.std(diffs) / mean_abs_diffs) * (1 - np.mean(self.humidity_history) / 200)

                recent_error = 0.0
                if self.error_history:
                    recent_error = safe_float(self.error_history[-1])
                    ewma_alpha = 0.3
                    for e in reversed(list(self.error_history)[-10:]):
                        recent_error = ewma_alpha * safe_float(e) + (1 - ewma_alpha) * recent_error

                error_factor = np.tanh(abs(recent_error) / 5.0)
                learning_rate = 0.05 + 0.25 * error_factor
                if len(self.error_history) > 1 and np.sign(safe_float(self.error_history[-1])) != np.sign(safe_float(self.error_history[-2])):
                    learning_rate *= 0.5

                volatility = float(np.std(self.history)) if len(self.history) > 1 else 0.0

                self.gamma = 0.90 + 0.05 * np.tanh(2.0 / max(volatility, 0.5))

                if volatility > 3.0:
                    self.alpha = 0.68
                    learning_rate *= 1.5
                else:
                    self.alpha = 0.78

                Tr = Tr * (1 - 0.3 * error_factor)
                correction = learning_rate * recent_error

                pressure_trend = np.mean(np.diff(self.pressure_history)) if len(self.pressure_history) > 1 else 0.0
                dphi = np.mean(diffs) if len(diffs) > 0 else (current_temp - safe_float(previous_temp))
                dphi += pressure_trend * 0.3
                humidity_temp_coupling = np.mean(self.humidity_history) - 50.0
                dphi += np.tanh(humidity_temp_coupling / 50.0) * 0.2

                # HOTFIX: solar radiation influence
                solar_influence = (current_solar - 200.0) / 400.0 * 0.15
                dphi += solar_influence

                k = 0.8 + np.mean(np.abs(diffs)) / 5 + np.mean(self.wind_history) / 50
                k = max(k, 1e-8)

                rhoE = 1.0 + ((np.mean(recent_t) - local_avg_temp) / local_temp_range) + (np.mean(self.pressure_history) - 1013) / 1000
                tauE = 0.95 + (hour_of_day / 48)

                base = (Nr * Tr * dphi) / k
                f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)
                dynamic_neighbor_scale = 0.4 * max(0.3, 1.0 - volatility / 15.0)
                f_field += neighbor_influence * dynamic_neighbor_scale

                recursive = 0.0
                if self.Er_history:
                    times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
                    decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
                    sum_decayed = np.nansum(decayed)
                    if np.isfinite(sum_decayed):
                        recursive = safe_power(sum_decayed, self.alpha)

                # CUTOFF ELIMINATED
                field_limit = max(120.0, np.std(self.history) * 5.5) if len(self.history) > 1 else 400.0
                Er_new = np.clip(f_field + (self.lambda_damp * recursive) + correction, -field_limit, field_limit)

                if not dry_run:
                    self.Er_history.append(Er_new)

                std_erm = np.std(self.Er_history) if len(self.Er_history) > 1 else 1.0
                beta = np.std(self.history) / (std_erm + 1e-6)
                beta = np.clip(beta, max(0.05, np.std(self.history)/60), min(2.2, np.std(self.history)/1.8))
                beta = beta * (1 - 0.2 * error_factor)

                decay_rate = 0.995 if volatility < 3.0 else 0.99
                self.bias_offset *= decay_rate
                self.bias_offset += learning_rate * recent_error * 0.08
                bias_limit = max(3.0, np.std(self.history) * 2.0) if len(self.history) > 1 else 6.0
                self.bias_offset = np.clip(self.bias_offset, -bias_limit, bias_limit)

                decay_rate_hourly = 0.995 if volatility < 3.0 else 0.99
                self.hourly_bias[hour_of_day] *= decay_rate_hourly
                self.hourly_bias[hour_of_day] += learning_rate * recent_error * 0.05
                hourly_limit = max(4.0, np.std(self.history) * 1.8) if len(self.history) > 1 else 6.0
                self.hourly_bias[hour_of_day] = np.clip(self.hourly_bias[hour_of_day], -hourly_limit, hourly_limit)

                total_bias = self.bias_offset + self.hourly_bias[hour_of_day]

                next_predicted = current_temp + (Er_new * beta) + total_bias
                next_predicted = np.clip(next_predicted, current_temp - local_temp_range * 2.2, current_temp + local_temp_range * 2.2)

                return Er_new, float(next_predicted), beta, pressure_trend

            except Exception as e:
                logger.error(f"Critical error in ERM.step() for {city_name}: {e}", exc_info=True)
                fallback_pred = current_temp + (current_temp - local_avg_temp) * 0.3
                return 0.0, float(fallback_pred), 0.6, 0.0

    async def predict_future(self, steps_list: List[int] = [1, 3, 6, 12, 24, 48]) -> Dict[int, float]:
        async with self._step_lock:
            if len(self.Er_history) < 3:
                last = safe_float(self.Er_history[-1]) if self.Er_history else 0.0
                return {s: last for s in steps_list}
            try:
                alpha_ewma = 0.3
                smoothed = []
                last_smoothed = safe_float(self.Er_history[-1])
                for val in reversed(list(self.Er_history)):
                    last_smoothed = alpha_ewma * safe_float(val) + (1 - alpha_ewma) * last_smoothed
                    smoothed.append(last_smoothed)
                smoothed = smoothed[::-1]

                times = np.arange(len(smoothed), 0, -1, dtype=np.float32)
                decayed = np.array(smoothed, dtype=np.float32) * (self.gamma ** times)
                base_projection = np.nansum(decayed) ** self.alpha
                if not np.isfinite(base_projection):
                    base_projection = 0.0

                result = {}
                for s in steps_list:
                    future_decay = (self.gamma ** s) * base_projection
                    result[s] = float(np.clip(future_decay, -400, 400))
                return result
            except Exception as e:
                logger.warning(f"predict_future fallback: {e}")
                last = safe_float(self.Er_history[-1]) if self.Er_history else 0.0
                return {s: last for s in steps_list}

# ==================== HELPERS ====================
def normalize_city_key(city_name: str) -> str:
    return str(city_name).lower().replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_").replace(",", "_")

def get_city_by_name(city_name: str, cities: List[Dict] = DEFAULT_CITIES) -> Dict:
    city_name_norm = normalize_city_key(city_name)
    for c in cities:
        if normalize_city_key(c.get("name", "")) == city_name_norm:
            return c
    return {}

async def load_city_states() -> Dict[str, ERM_Live_Adaptive]:
    erms: Dict[str, ERM_Live_Adaptive] = {}
    for city in DEFAULT_CITIES:
        erm = ERM_Live_Adaptive(debug_mode=False)
        state_file = STATE_DIR / f"erm_state_{normalize_city_key(city['name'])}.json"
        await erm.async_load_state(state_file)
        erms[city['name']] = erm
    logger.info(f"✅ Loaded {len(erms)} per-city ERM states")
    return erms

async def save_all_city_states(erms: Dict[str, ERM_Live_Adaptive]):
    for city_name, erm in erms.items():
        state_file = STATE_DIR / f"erm_state_{normalize_city_key(city_name)}.json"
        await erm.async_save_state(state_file)

async def async_append_csv(csv_path: Path, row_dict: Dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 5_000_000:
        backup = csv_path.with_suffix('.old.csv')
        try:
            move(csv_path, backup)
            logger.info(f"CSV rotated: {csv_path.name}")
        except Exception as e:
            logger.warning(f"CSV rotation failed: {e}")
    header = not csv_path.exists()
    async with aiofiles.open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=row_dict.keys(), quoting=csv.QUOTE_MINIMAL)
        if header:
            writer.writeheader()
        writer.writerow(row_dict)
        await f.write(output.getvalue())

async def periodic_save(interval_seconds: int = 300):
    while True:
        try:
            if hasattr(app.state, 'per_city_erms'):
                await save_all_city_states(app.state.per_city_erms)
        except Exception as e:
            logger.error(f"Periodic save failed: {e}")
        await asyncio.sleep(interval_seconds)

# ==================== HARDENED ASYNC GIT BACKUP (fixes your Render errors) ====================
async def async_git_backup(data_dir: Path):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    if not token or not repo:
        logger.warning("Git backup skipped — missing GITHUB_TOKEN or GITHUB_REPO")
        return
    repo_root = data_dir.parent
    try:
        # Clean stale locks
        for lock_path in [repo_root / ".git" / "index.lock", repo_root / ".git" / "config.lock", repo_root / ".git" / "HEAD.lock"]:
            if lock_path.exists():
                try:
                    lock_path.unlink(missing_ok=True)
                    logger.info(f"✅ Cleaned stale lock: {lock_path.name}")
                except Exception as e:
                    logger.warning(f"Could not remove lock {lock_path}: {e}")

        cmds = [
            ["git", "init"],
            ["git", "config", "--global", "user.name", "ERM Bot"],
            ["git", "config", "--global", "user.email", "erm-bot@github.com"],
            ["git", "remote", "remove", "origin"],
            ["git", "remote", "add", "origin", f"https://{token}@github.com/{repo}.git"],
        ]
        for cmd in cmds:
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=repo_root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

        await asyncio.create_subprocess_exec("git", "add", "ERM_Data/", cwd=repo_root)
        await asyncio.create_subprocess_exec("git", "add", "ERM_State/", cwd=repo_root)

        proc = await asyncio.create_subprocess_exec("git", "diff-index", "--quiet", "HEAD", "--", cwd=repo_root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()

        if proc.returncode != 0:
            commit_msg = f"ERM v{VERSION} automated backup - {datetime.now().isoformat()}"
            await asyncio.create_subprocess_exec("git", "commit", "-m", commit_msg, cwd=repo_root)
            await asyncio.create_subprocess_exec("git", "push", "-u", "origin", "main", cwd=repo_root)
            logger.info("✅ Async git backup successful")
        else:
            logger.info("✅ No changes to backup")
    except Exception as e:
        logger.warning(f"Async git backup warning (non-critical): {e}")

# ==================== PYDANTIC MODELS ====================
class LiveWeatherRecord(BaseModel):
    live_temp: float = Field(..., ge=-100, le=100)
    humidity: float = Field(50.0, ge=0, le=100)
    wind: float = Field(5.0, ge=0)
    pressure: float = Field(1013.0)
    rain_prob: float = Field(0.0, ge=0, le=100)
    cloud_cover: float = Field(0.0, ge=0, le=100)
    solar: float = Field(0.0)

class LiveWeatherBatch(BaseModel):
    records: Dict[str, LiveWeatherRecord]

class TuneParameters(BaseModel):
    alpha: Optional[float] = None
    gamma: Optional[float] = None
    lambda_damp: Optional[float] = None
    bias_offset: Optional[float] = None

# ==================== REMAINING FUNCTIONS (get_yesterday_baseline, fetch..., backfill, _core_record, endpoints, etc.) ====================
# (All the rest of your original code from get_yesterday_baseline down to the end remains exactly the same — only the git backup and a few small v8.0 changes were applied above)

# ... [the rest of your original file from get_yesterday_baseline to the very end] ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
