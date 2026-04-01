from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import uvicorn
import os
import subprocess
import numpy as np
import httpx
import asyncio
import pandas as pd
import logging
import threading
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Dict, Optional
import math
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=10, max_keepalive_connections=5))
    base_dir = Path(__file__).parent
    (base_dir / "ERM_Data").mkdir(parents=True, exist_ok=True)
    (base_dir / "ERM_State").mkdir(parents=True, exist_ok=True)
    logger.info("🚀 V5.3 HTTP client initialized and directories ready")
    yield
    if http_client:
        await http_client.aclose()
    logger.info("🛑 V5.3 HTTP client closed")

app = FastAPI(title="ERM Live Update Service — V5.3", lifespan=lifespan)

VERSION = "5.3"

DEFAULT_CITIES = [ ... ]  # (same as before - kept unchanged)

http_client: Optional[httpx.AsyncClient] = None

# (haversine, ERM_Live_Adaptive class, step(), predict_future, normalize_city_key, get_yesterday_baseline, fetch_multi_variable_data, backfill_realized_errors - all exactly the same as V5.2)

# ==================== IMPROVED GIT BACKUP ====================
def git_backup(data_dir: Path):
    def do_backup():
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        if not token or not repo:
            logger.warning("Git backup skipped — missing GITHUB_TOKEN or GITHUB_REPO")
            return
        try:
            repo_root = data_dir.parent
            if not (repo_root / ".git").exists():
                subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
                logger.info("Git repo initialized on Render")
            # More robust remote handling
            remote_url = f"https://{token}@github.com/{repo}.git"
            result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo_root, capture_output=True, text=True)
            if result.returncode != 0:
                subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=repo_root, check=True, capture_output=True)
            else:
                subprocess.run(["git", "remote", "set-url", "origin", remote_url], cwd=repo_root, check=True, capture_output=True)
            # ... rest of your original git config / add / commit / push code ...
            # (kept exactly as before, just with the safer remote logic)
            # (full block is in the code I gave you earlier - I shortened it here for brevity)
        except Exception as e:
            logger.error(f"Git backup failed: {e}")

    threading.Thread(target=do_backup, daemon=True).start()

# ==================== /update WITH DEBUG LOGGING ====================
update_lock = asyncio.Lock()

@app.get("/update")
@app.head("/update")
async def update_data(background_tasks: BackgroundTasks):
    async with update_lock:
        data_dir = Path(__file__).parent / "ERM_Data"
        state_dir = Path(__file__).parent / "ERM_State"

        cities = DEFAULT_CITIES
        erms = {}
        for city in cities:
            erm = ERM_Live_Adaptive()
            state_file = state_dir / f"erm_state_{normalize_city_key(city['name'])}.json"
            erm.load_state(state_file)
            erms[city['name']] = erm

        now = datetime.now()
        today_str = now.strftime('%Y%m%d')
        hour_of_day = now.hour

        sem_limit = int(os.getenv("ERM_SEMAPHORE_LIMIT", 5))
        sem = asyncio.Semaphore(sem_limit)

        async def fetch_and_update(city):
            async with sem:
                data = await fetch_multi_variable_data(city['lat'], city['lon'], city['tz'])
                
                # === DEBUG LOGGING ===
                logger.info(f"DEBUG {city['name']}: data received → temp={data.get('temp') if data else None}")
                if not data:
                    logger.error(f"DEBUG {city['name']}: No data returned from API")
                    return city['name'], False
                if data.get('temp') is None:
                    logger.warning(f"DEBUG {city['name']}: temp is None")
                    return city['name'], False

                try:
                    live_temp = data['temp']
                    erm = erms[city['name']]
                    baseline_temp = get_yesterday_baseline(city['name'], hour_of_day, data_dir)

                    Er_flux, next_predicted, beta, _ = erm.step(
                        live_temp, data['humidity'], data['wind'], data['pressure'],
                        0.0, 50.0, 0.0,
                        erm.history[-1] if len(erm.history) > 0 else None,
                        hour_of_day, city.get('local_avg_temp', 15.0), city.get('local_temp_range', 30.0),
                        city_name=city['name']
                    )

                    # ... rest of your original processing (realized_error, record_error, improvement, future, row, save CSV, save state) ...

                    logger.info(f"✅ Updated {city['name']} → improvement {improvement:.1f}% | error {realized_error:.3f}")
                    return city['name'], True

                except Exception as e:
                    logger.error(f"ERROR processing {city['name']}: {e}")
                    return city['name'], False

        results = await asyncio.gather(*(fetch_and_update(city) for city in cities), return_exceptions=True)
        successful = sum(1 for r in results if not isinstance(r, Exception) and r[1] is True)

    backfill_realized_errors(data_dir)
    background_tasks.add_task(git_backup, data_dir)
    return {"status": "success", "updated": len(cities), "successful": successful, "time": datetime.now().isoformat()}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "V5.3 Context-Aware Relational (final)"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
