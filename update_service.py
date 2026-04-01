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

# ==================== LIFESPAN (modern FastAPI) ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_connections=10, max_keepalive_connections=5))
    base_dir = Path(__file__).parent
    (base_dir / "ERM_Data").mkdir(parents=True, exist_ok=True)
    (base_dir / "ERM_State").mkdir(parents=True, exist_ok=True)
    logger.info("🚀 V5.2 HTTP client initialized and directories ready")
    yield
    if http_client:
        await http_client.aclose()
    logger.info("🛑 V5.2 HTTP client closed")

app = FastAPI(title="ERM Live Update Service — V5.2", lifespan=lifespan)

VERSION = "5.2"

DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL",    "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK",   "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP",    "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 28.0},
]

http_client: Optional[httpx.AsyncClient] = None
failure_counter = defaultdict(int)

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 10):
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

    def save_state(self, filepath: Path):
        state = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, deque):
                state[k] = list(v)
            elif isinstance(v, defaultdict):
                state[k] = dict(v)
            else:
                state[k] = v
        state["last_update_timestamp"] = datetime.now().isoformat()
        filepath.write_text(json.dumps(state, indent=2))

    def load_state(self, filepath: Path):
        if filepath.exists():
            try:
                state = json.loads(filepath.read_text())
                for k, v in state.items():
                    if k in self.__dict__:
                        if isinstance(self.__dict__[k], deque):
                            self.__dict__[k].extend(v)
                        elif k == "hourly_bias":
                            self.hourly_bias = defaultdict(float, v)
                        else:
                            self.__dict__[k] = v
            except Exception as e:
                logger.warning(f"Failed to load state {filepath}: {e}")

    def record_error(self, realized_error: float, predicted: float):
        error = realized_error
        systematic = np.mean(self.error_history) if self.error_history else 0.0
        noise = error - systematic
        self.error_history.append(error)
        self.systematic_bias.append(systematic)
        self.noise_error.append(noise)

    def step(self, current_temp, current_humidity, current_wind, current_pressure,
             current_rain_prob, current_cloud_cover, current_solar,
             previous_temp, hour_of_day, local_avg_temp, local_temp_range,
             neighbor_influence: float = 0.0, city_name: str = "Unknown"):
        self.history.append(current_temp)
        self.humidity_history.append(current_humidity)
        self.wind_history.append(current_wind)
        self.pressure_history.append(current_pressure)

        if len(self.history) < 2:
            return 0.0, current_temp, 0.5

        recent_t = np.array(self.history, dtype=np.float32)
        diffs = np.diff(recent_t)
        Nr = len(recent_t) * (1 + np.var(recent_t) / 10)
        Tr = max(0.6, 1 - np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6)) * (1 - np.mean(self.humidity_history) / 200)

        recent_error = 0.0
        if self.error_history:
            ewma_alpha = 0.3
            recent_error = self.error_history[-1]
            for e in reversed(list(self.error_history)[-10:]):
                recent_error = ewma_alpha * e + (1 - ewma_alpha) * recent_error

        error_factor = np.tanh(abs(recent_error) / 5.0)
        learning_rate = 0.05 + 0.25 * error_factor

        if len(self.error_history) > 1 and np.sign(self.error_history[-1]) != np.sign(self.error_history[-2]):
            learning_rate *= 0.5

        volatility = np.std(self.history) if len(self.history) > 1 else 0.0

        if volatility > 3.0:
            self.alpha = 0.65
            self.gamma = 0.92
            learning_rate *= 1.5
        else:
            self.alpha = 0.75
            self.gamma = 0.935

        Tr = Tr * (1 - 0.3 * error_factor)
        correction = learning_rate * recent_error

        pressure_trend = np.mean(np.diff(self.pressure_history)) if len(self.pressure_history) > 1 else 0.0
        dphi = np.mean(diffs) if len(diffs) > 0 else (current_temp - previous_temp if previous_temp is not None else 0.0)
        dphi += pressure_trend * 0.3

        k = 0.8 + np.mean(np.abs(diffs)) / 5 + np.mean(self.wind_history) / 50
        rhoE = 1.0 + ((np.mean(recent_t) - local_avg_temp) / local_temp_range) + (np.mean(self.pressure_history) - 1013) / 1000
        tauE = 0.95 + (hour_of_day / 48)

        base = (Nr * Tr * dphi) / max(k, 1e-8)
        f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)
        f_field += neighbor_influence * 0.4

        # FIXED: prevent NaN on negative power
        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            sum_decayed = np.sum(decayed)
            if sum_decayed >= 0:
                recursive = sum_decayed ** self.alpha
            else:
                recursive = -(np.abs(sum_decayed) ** self.alpha)

        field_limit = max(50, np.std(self.history) * 3) if len(self.history) > 1 else 200
        Er_new = np.clip(f_field + (self.lambda_damp * recursive) + correction, -field_limit, field_limit)
        self.Er_history.append(Er_new)

        beta = np.clip(np.std(self.history) / (np.std(self.Er_history) + 1e-6),
                       max(0.01, np.std(self.history)/50),
                       max(1.0, np.std(self.history)/2))
        beta = beta * (1 - 0.2 * error_factor)

        decay_rate = 0.995 if volatility < 3.0 else 0.99
        self.bias_offset *= decay_rate
        self.bias_offset += learning_rate * recent_error * 0.08
        bias_limit = max(2.0, np.std(self.history) * 1.5) if len(self.history) > 1 else 5.0
        self.bias_offset = np.clip(self.bias_offset, -bias_limit, bias_limit)

        next_predicted = current_temp + (Er_new * beta) + self.bias_offset + self.hourly_bias[hour_of_day]
        next_predicted = np.clip(next_predicted, current_temp - 50, current_temp + 50)

        return Er_new, next_predicted, beta, pressure_trend

    def predict_future(self, steps_list: List[int] = [1, 3, 6, 12, 24, 48]) -> Dict[int, float]:
        if len(self.Er_history) < 5:
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}
        try:
            x = np.arange(len(self.Er_history), dtype=np.float32)
            y = np.array(self.Er_history, dtype=np.float32)
            slope, intercept = np.polyfit(x, y, 1)
            return {s: float(np.clip(slope * (len(x) + s) + intercept, -200, 200)) for s in steps_list}
        except Exception as e:
            logger.warning(f"polyfit failed in predict_future: {e}")
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}

# ==================== HELPERS (with ISO8601 fix) ====================
def normalize_city_key(city_name: str) -> str:
    return city_name.lower().replace(" ", "_").replace("-", "_")

def get_yesterday_baseline(city_name: str, hour: int, data_dir: Path) -> float:
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    csv_path = data_dir / f"erm_v4.4_{normalize_city_key(city_name)}_{yesterday_str}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
            yesterday_same_hour = df[(df['timestamp'].dt.date == (datetime.now() - timedelta(days=1)).date()) &
                                     (df['timestamp'].dt.hour == hour)]
            if not yesterday_same_hour.empty:
                return float(yesterday_same_hour['live_temp'].mean())
        except Exception:
            pass
    for city in DEFAULT_CITIES:
        if normalize_city_key(city['name']) == normalize_city_key(city_name):
            return city.get('local_avg_temp', 15.0)
    return 15.0

async def fetch_multi_variable_data(lat: float, lon: float, timezone: str) -> Optional[Dict]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone
    }
    for attempt in range(3):
        try:
            resp = await http_client.get("https://api.open-meteo.com/v1/forecast", params=params)
            resp.raise_for_status()
            data = resp.json()
            current = data.get('current', {})
            daily = data.get('daily', {})
            return {
                'temp': current.get('temperature_2m'),
                'humidity': current.get('relative_humidity_2m'),
                'wind': current.get('wind_speed_10m'),
                'pressure': current.get('surface_pressure'),
                'time': datetime.now().isoformat(),
                'tomorrow_max': daily.get('temperature_2m_max', [None, None])[1],
                'tomorrow_min': daily.get('temperature_2m_min', [None, None])[1]
            }
        except Exception as e:
            logger.warning(f"Failed fetching data for {lat},{lon} (attempt {attempt+1}): {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return None

def backfill_realized_errors(data_dir: Path):
    logger.info("🔄 Back-filling realized prediction errors...")
    today_str = datetime.now().strftime('%Y%m%d')
    for csv_path in data_dir.glob(f"erm_v4.4_*_{today_str}.csv"):
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
            df = df.sort_values('timestamp').reset_index(drop=True)
            horizons = [1, 3, 6, 12, 24, 48]
            for h in horizons:
                pred_col = f'next_predicted_{h}h'
                error_col = f'error_{h}h'
                if pred_col not in df.columns:
                    df[error_col] = np.nan
                    continue
                df['shifted_time'] = df['timestamp'] + pd.Timedelta(hours=h)
                merged = pd.merge_asof(df[['shifted_time']], df[['timestamp', 'live_temp']],
                                       left_on='shifted_time', right_on='timestamp',
                                       direction='forward', tolerance=pd.Timedelta('5min'))
                df[error_col] = merged['live_temp'] - df[pred_col]
            df.drop(columns=['shifted_time'], errors='ignore').to_csv(csv_path, index=False)
            logger.info(f"✅ Back-filled errors for {csv_path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Backfill skipped for {csv_path.name}: {e}")

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
            remote_url = f"https://{token}@github.com/{repo}.git"
            subprocess.run(["git", "remote", "set-url", "origin", remote_url], cwd=repo_root, check=True, capture_output=True)
            subprocess.run(["git", "config", "--global", "user.name", "ERM Bot"], cwd=repo_root, check=True, capture_output=True)
            subprocess.run(["git", "config", "--global", "user.email", "erm-bot@github.com"], cwd=repo_root, check=True, capture_output=True)
            today_str = datetime.now().strftime('%Y%m%d')
            subprocess.run(["git", "add", f"ERM_Data/*_{today_str}.csv"], cwd=repo_root, check=True, capture_output=True)
            subprocess.run(["git", "add", f"ERM_State/*"], cwd=repo_root, check=True, capture_output=True)
            diff_check = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=repo_root, capture_output=True)
            if diff_check.returncode != 0:
                subprocess.run(["git", "checkout", "-B", "main"], cwd=repo_root, check=True, capture_output=True)
                subprocess.run(["git", "commit", "-m", f"ERM V5 live update {datetime.now().isoformat()}"], cwd=repo_root, check=True, capture_output=True)
                subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repo_root, check=True, capture_output=True)
                logger.info("✅ Git backup successful")
            else:
                logger.info("No changes to commit")
        except Exception as e:
            logger.error(f"Git backup failed: {e}")

    threading.Thread(target=do_backup, daemon=True).start()

# ==================== /update ENDPOINT ====================
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
                if not data or data['temp'] is None:
                    logger.warning(f"Skipped {city['name']} due to fetch error")
                    return city['name'], False

                live_temp = data['temp']
                erm = erms[city['name']]
                baseline_temp = get_yesterday_baseline(city['name'], hour_of_day, data_dir)

                Er_flux, next_predicted, beta, _ = erm.step(
                    live_temp, data['humidity'], data['wind'], data['pressure'],
                    0.0, 50.0, 0.0,                                      # rain, cloud, solar (Open-Meteo fallback)
                    erm.history[-1] if len(erm.history) > 0 else None,
                    hour_of_day, city.get('local_avg_temp', 15.0), city.get('local_temp_range', 30.0),
                    city_name=city['name']
                )

                realized_error = live_temp - next_predicted
                erm.record_error(realized_error, next_predicted)

                erm_err = abs(live_temp - next_predicted)
                baseline_err = abs(live_temp - baseline_temp)
                improvement = 100 * (baseline_err - erm_err) / max(baseline_err, 0.01) if baseline_err > 0 else 0.0

                if improvement < -50:
                    logger.warning(f"🚨 ANOMALY ALERT: {city['name']} improvement {improvement:.1f}%")

                future = erm.predict_future([1, 3, 6, 12, 24, 48])

                row = {
                    'timestamp': data['time'],
                    'timestamp_utc': datetime.utcnow().isoformat(),
                    'live_temp': live_temp,
                    'humidity': data['humidity'],
                    'wind': data['wind'],
                    'pressure': data['pressure'],
                    'erm_flux': Er_flux,
                    'beta': beta,
                    'baseline_temp': baseline_temp,
                    **{f'next_predicted_{h}h': next_predicted + future.get(h, 0.0) for h in [1, 3, 6, 12, 24, 48]},
                    'tomorrow_max': data.get('tomorrow_max'),
                    'tomorrow_min': data.get('tomorrow_min'),
                    'improvement_pct': improvement,
                    'error_1h': np.nan, 'error_3h': np.nan, 'error_6h': np.nan,
                    'error_12h': np.nan, 'error_24h': np.nan, 'error_48h': np.nan
                }

                csv_path = data_dir / f"erm_v4.4_{normalize_city_key(city['name'])}_{today_str}.csv"
                file_exists = csv_path.exists()
                df_row = pd.DataFrame([row])
                df_row.to_csv(csv_path, mode='a', header=not file_exists, index=False)

                logger.info(f"✅ Updated {city['name']} → improvement {improvement:.1f}% | error {realized_error:.3f}")

                state_file = state_dir / f"erm_state_{normalize_city_key(city['name'])}.json"
                erm.save_state(state_file)
                return city['name'], True

        results = await asyncio.gather(*(fetch_and_update(city) for city in cities), return_exceptions=True)
        successful = sum(1 for r in results if not isinstance(r, Exception) and r[1] is True)

    backfill_realized_errors(data_dir)
    background_tasks.add_task(git_backup, data_dir)
    return {"status": "success", "updated": len(cities), "successful": successful, "time": datetime.now().isoformat()}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "V5.2 Context-Aware Relational (final)"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
