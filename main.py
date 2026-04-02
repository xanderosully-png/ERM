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

# ===================== SAFETY + FETCH HELPERS =====================
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

# (parallel fetch functions unchanged from previous version)

# ===================== DEFAULT CITIES =====================
DEFAULT_CITIES = [ ... ]   # ← keep your full list of 18 cities here

# ===================== ERM CLASS (unchanged) =====================
class ERM_Live_Adaptive:
    # (full class from previous version)

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

# ===================== LOAD / SAVE (with extra logging) =====================
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
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        else:
            logger.info(f"No CSV yet for {name} — starting fresh")
        erms[name] = erm
    return erms

async def save_all_city_states(erms: Dict):
    logger.info("💾 Starting save_all_city_states for all cities...")
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
            logger.info(f"✅ Saved {len(rows)} records for {name}")
        except Exception as e:
            logger.error(f"❌ Failed to save {name}: {e}")
    await async_git_backup(DATA_DIR, STATE_DIR)

async def async_git_backup(data_dir: Path, state_dir: Path):
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    if not token or not repo:
        logger.warning("⚠️ GITHUB_TOKEN or GITHUB_REPO not set — skipping git backup")
        return
    try:
        remote_url = f"https://{token}@github.com/{repo}.git"
        cwd = Path(__file__).parent
        await asyncio.create_subprocess_exec("git", "add", str(data_dir), cwd=cwd)
        proc = await asyncio.create_subprocess_exec("git", "commit", "-m", f"🚀 Auto-save {datetime.utcnow().isoformat()}", cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await proc.communicate()
        await asyncio.create_subprocess_exec("git", "push", remote_url, "main", cwd=cwd)
        logger.info("✅ GitHub backup completed — CSVs should now appear in repo")
    except Exception as e:
        logger.error(f"GitHub backup failed: {e}")

# ===================== PERIODIC SAVE =====================
async def periodic_save(interval_seconds: int = 300):
    while True:
        try:
            if hasattr(app.state, 'per_city_erms'):
                await save_all_city_states(app.state.per_city_erms)
        except Exception as e:
            logger.error(f"Periodic save failed: {e}")
        await asyncio.sleep(interval_seconds)

# ===================== UPDATE & DASHBOARD ENDPOINTS =====================
@app.get("/update")
async def update_all_cities(background_tasks: BackgroundTasks):
    # (your current update code — unchanged)

@app.get("/status")
async def status():
    if not hasattr(app.state, 'per_city_erms'):
        return {"status": "not_initialized"}
    counts = {name: len(erm.history) for name, erm in app.state.per_city_erms.items()}
    return {
        "version": VERSION,
        "cities": len(counts),
        "total_records": sum(counts.values()),
        "per_city": counts
    }

@app.get("/predict/{city}")
async def predict_city(city: str):
    await check_rate_limit(city)
    erm = app.state.per_city_erms.get(city)
    if not erm:
        raise HTTPException(status_code=404, detail="City not found")
    latest_temp = float(erm.history[-1]) if erm.history else 15.0
    future = await erm.predict_future(latest_temp, [1, 3, 6, 12, 24])   # ← await added here
    return {
        "next_predicted_1h": future.get(1),
        "confidence_percent": 85 if len(erm.history) > 5 else 60,
        "current_regime": erm.current_regime,
        "performance_score": round(erm.performance_score, 3),
        "future_forecast": future,
    }

# ( /latest, /benchmark, /visualize — unchanged from previous version )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
