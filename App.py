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

# ==================== ERM CLASS (v8.0) ====================
# (Your full ERM_Live_Adaptive class remains unchanged - omitted here for brevity, but keep everything from your previous version)

class ERM_Live_Adaptive:
    # ... [keep your entire ERM_Live_Adaptive class exactly as it is in your current file] ...
    pass   # ← Replace this line with your actual class code

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

# ==================== HARDENED ASYNC GIT BACKUP ====================
async def async_git_backup(data_dir: Path):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    if not token or not repo:
        logger.warning("Git backup skipped — missing GITHUB_TOKEN or GITHUB_REPO")
        return
    repo_root = data_dir.parent
    try:
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

# ==================== GET YESTERDAY BASELINE, FETCH, BACKFILL, _CORE_RECORD, etc. ====================
# (Keep all your existing functions from here down unchanged)

# ... [paste all your remaining functions from get_yesterday_baseline through the end of the file] ...

# ==================== IMPROVED /LATEST ENDPOINT (this is the fix) ====================
@app.get("/latest")
async def get_latest():
    latest_data = []
    for csv_path in DATA_DIR.glob(f"{CSV_PREFIX}_*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                if df.empty:
                    continue
                latest_row = df.loc[df['timestamp'].idxmax()]
            else:
                latest_row = df.iloc[-1]
            row = latest_row.to_dict()
            filename = csv_path.name
            city_part = filename.replace(f"{CSV_PREFIX}_", "").rsplit("_", 1)[0]
            row["city"] = city_part.replace("_", " ").title().replace(" ", "_")
            latest_data.append(row)
        except Exception as e:
            logger.warning(f"Could not read latest from {csv_path.name}: {e}")
            continue
    logger.info(f"📡 /latest served {len(latest_data)} fresh cities")
    return latest_data

# ==================== ALL OTHER ENDPOINTS (keep as-is) ====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
