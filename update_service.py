from fastapi import FastAPI, BackgroundTasks
import uvicorn
import os
import subprocess
import numpy as np
import httpx
import asyncio
import csv
import json
import pandas as pd
import logging
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="ERM Live Update Service")

VERSION = "4.4"

DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York", "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL",    "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York", "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York", "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles", "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK",   "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London", "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP",    "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo", "local_avg_temp": 16.0, "local_temp_range": 28.0},
]

http_client: Optional[httpx.AsyncClient] = None

class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 10):
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self.humidity_history: deque = deque(maxlen=history_size)
        self.wind_history: deque = deque(maxlen=history_size)
        self.pressure_history: deque = deque(maxlen=history_size)
        self.Er_history: deque = deque(maxlen=history_size)
        self.error_history: deque = deque(maxlen=20)
        self.bias_offset = 0.0
        self.gamma = 0.935
        self.lambda_damp = 0.28
        self.alpha = 0.75

    def save_state(self, filepath: Path):
        state = {
            "history": list(self.history), "humidity_history": list(self.humidity_history),
            "wind_history": list(self.wind_history), "pressure_history": list(self.pressure_history),
            "Er_history": list(self.Er_history), "error_history": list(self.error_history),
            "bias_offset": self.bias_offset,
            "gamma": self.gamma, "lambda_damp": self.lambda_damp, "alpha": self.alpha
        }
        filepath.write_text(json.dumps(state))

    def load_state(self, filepath: Path):
        if filepath.exists():
            try:
                state = json.loads(filepath.read_text())
                self.history.extend(state.get("history", []))
                self.humidity_history.extend(state.get("humidity_history", []))
                self.wind_history.extend(state.get("wind_history", []))
                self.pressure_history.extend(state.get("pressure_history", []))
                self.Er_history.extend(state.get("Er_history", []))
                self.error_history.extend(state.get("error_history", []))
                self.bias_offset = state.get("bias_offset", 0.0)
                self.gamma = state.get("gamma", self.gamma)
                self.lambda_damp = state.get("lambda_damp", self.lambda_damp)
                self.alpha = state.get("alpha", self.alpha)
            except Exception as e:
                logger.warning(f"Failed to load state {filepath}: {e}")

    def record_error(self, error: float):
        self.error_history.append(error)

    def step(self, current_temp, current_humidity, current_wind, current_pressure,
             previous_temp, hour_of_day, local_avg_temp, local_temp_range):
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
            weights = np.linspace(1.0, 0.2, len(self.error_history))
            recent_error = np.average(self.error_history, weights=weights)

        error_factor = np.tanh(abs(recent_error) / 5.0)
        learning_rate = 0.05 + 0.25 * error_factor

        if len(self.error_history) > 1 and np.sign(self.error_history[-1]) != np.sign(self.error_history[-2]):
            learning_rate *= 0.5

        volatility = np.std(self.history) if len(self.history) > 1 else 0.0
        if volatility > 3.0:
            learning_rate *= 1.5
        else:
            learning_rate *= 0.7

        Tr = Tr * (1 - 0.3 * error_factor)
        correction = learning_rate * recent_error

        dphi = np.mean(diffs) if len(diffs) > 0 else (current_temp - previous_temp if previous_temp is not None else 0.0)

        k = 0.8 + np.mean(np.abs(diffs)) / 5 + np.mean(self.wind_history) / 50
        rhoE = 1.0 + ((np.mean(recent_t) - local_avg_temp) / local_temp_range) + (np.mean(self.pressure_history) - 1013) / 1000
        tauE = 0.95 + (hour_of_day / 48)

        base = (Nr * Tr * dphi) / max(k, 1e-8)
        f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)

        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            recursive = np.sum(np.abs(decayed)) ** self.alpha

        field_limit = max(50, np.std(self.history) * 10) if len(self.history) > 1 else 200
        Er_new = np.clip(f_field + (self.lambda_damp * recursive) + correction, -field_limit, field_limit)
        self.Er_history.append(Er_new)

        if abs(Er_new) > field_limit * 0.8:
            logger.warning(f"⚠️ Extreme Er_new detected: {abs(Er_new):.1f}")

        beta = np.clip(np.std(self.history) / (np.std(self.Er_history) + 1e-6),
                       max(0.01, np.std(self.history)/50),
                       max(1.0, np.std(self.history)/2))
        beta = beta * (1 - 0.2 * error_factor)

        self.bias_offset *= 0.995
        self.bias_offset += learning_rate * recent_error * 0.08
        bias_limit = max(2.0, np.std(self.history) * 1.5) if len(self.history) > 1 else 5.0
        self.bias_offset = np.clip(self.bias_offset, -bias_limit, bias_limit)

        next_predicted = current_temp + (Er_new * beta) + self.bias_offset
        return Er_new, next_predicted, beta

    def predict_future(self, steps_list: List[int] = [1, 3, 6, 12, 24, 48]):
        if len(self.Er_history) < 5:
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}
        x = np.arange(len(self.Er_history), dtype=np.float32)
        y = np.array(self.Er_history, dtype=np.float32)
        slope, intercept = np.polyfit(x, y, 1)
        return {s: float(np.clip(slope * (len(x) + s) + intercept, -200, 200)) for s in steps_list}


def normalize_city_key(city_name: str) -> str:
    return city_name.lower().replace(" ", "_").replace("-", "_")


def get_yesterday_baseline(city_name: str, hour: int, data_dir: Path) -> float:
    yesterday = datetime.now() - pd.Timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y%m%d')
    csv_path = data_dir / f"erm_v{VERSION}_{normalize_city_key(city_name)}_{yesterday_str}.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            yesterday_same_hour = df[(df['timestamp'].dt.date == yesterday.date()) & (df['timestamp'].dt.hour == hour)]
            if not yesterday_same_hour.empty:
                return float(yesterday_same_hour['live_temp'].mean())
        except Exception:
            pass
    for city in DEFAULT_CITIES:
        if normalize_city_key(city['name']) == normalize_city_key(city_name):
            return city['local_avg_temp']
    return 15.0


async def fetch_multi_variable_data(lat: float, lon: float, timezone: str) -> Optional[Dict]:
    params = {"latitude": lat, "longitude": lon,
              "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
              "daily": "temperature_2m_max,temperature_2m_min", "timezone": timezone}
    for attempt in range(3):
        try:
            resp = await http_client.get("https://api.open-meteo.com/v1/forecast", params=params)
            resp.raise_for_status()
            data = resp.json()
            current = data['current']
            daily = data['daily']
            return {
                'temp': current['temperature_2m'],
                'humidity': current['relative_humidity_2m'],
                'wind': current['wind_speed_10m'],
                'pressure': current['surface_pressure'],
                'time': datetime.now().isoformat(),
                'tomorrow_max': daily.get('temperature_2m_max', [None, None])[1],
                'tomorrow_min': daily.get('temperature_2m_min', [None, None])[1]
            }
        except Exception as e:
            logger.warning(f"Failed fetching data for {lat},{lon} (attempt {attempt+1}): {e}")
            if attempt < 2:
                await asyncio.sleep(3)
    return None


def backfill_realized_errors(data_dir: Path):
    logger.info("🔄 Back-filling realized prediction errors...")
    today_str = datetime.now().strftime('%Y%m%d')
    for csv_path in data_dir.glob(f"erm_v{VERSION}_*_{today_str}.csv"):
        try:
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            horizons = [1, 3, 6, 12, 24, 48]
            for h in horizons:
                pred_col = f'next_predicted_{h}h'
                error_col = f'error_{h}h'
                if pred_col not in df.columns:
                    df[error_col] = np.nan
                    continue
                df['shifted_time'] = df['timestamp'] + pd.Timedelta(hours=h)
                merged = pd.merge_asof(df[['shifted_time']], df[['timestamp', 'live_temp']], left_on='shifted_time', right_on='timestamp', direction='forward')
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
            subprocess.run(["git", "checkout", "-B", "main"], cwd=repo_root, check=True, capture_output=True)
            commit_result = subprocess.run(["git", "commit", "-m", f"ERM live update {datetime.now().isoformat()}"], cwd=repo_root, capture_output=True, text=True)
            if commit_result.returncode == 0:
                subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=repo_root, check=True, capture_output=True, text=True)
                logger.info("✅ Git backup successful")
            else:
                logger.info("No changes to commit")
        except Exception as e:
            logger.error(f"Git backup failed: {e}")

    threading.Thread(target=do_backup, daemon=True).start()


update_lock = asyncio.Lock()

@app.on_event("startup")
async def startup_event():
    global http_client
    http_client = httpx.AsyncClient(timeout=10.0)
    logger.info("🚀 HTTP client initialized")


@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()
    logger.info("🛑 HTTP client closed")


@app.get("/update")
@app.head("/update")
async def update_data(background_tasks: BackgroundTasks):
    async with update_lock:
        base_dir = Path(__file__).parent
        data_dir = base_dir / "ERM_Data"
        state_dir = base_dir / "ERM_State"
        data_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

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

        sem = asyncio.Semaphore(5)

        async def fetch_city(city):
            async with sem:
                return city, await fetch_multi_variable_data(city['lat'], city['lon'], city['tz'])

        results = await asyncio.gather(*(fetch_city(city) for city in cities), return_exceptions=True)

        for city, data in results:
            if isinstance(data, Exception) or not data:
                logger.warning(f"Skipped {city['name']} due to fetch error")
                continue

            live_temp = data['temp']
            erm = erms[city['name']]

            baseline_temp = get_yesterday_baseline(city['name'], hour_of_day, data_dir)

            Er_flux, next_predicted, beta = erm.step(
                live_temp, data['humidity'], data['wind'], data['pressure'],
                erm.history[-1] if len(erm.history) > 0 else None,
                hour_of_day, city['local_avg_temp'], city['local_temp_range']
            )

            realized_error = live_temp - next_predicted
            erm.record_error(realized_error)

            erm_err = abs(live_temp - next_predicted)
            baseline_err = abs(live_temp - baseline_temp)
            improvement = 100 * (baseline_err - erm_err) / max(baseline_err, 0.01) if baseline_err > 0 else 0.0

            if improvement < -50:
                logger.warning(f"🚨 ANOMALY ALERT: {city['name']} improvement {improvement:.1f}% — check data/API")

            future = erm.predict_future([1, 3, 6, 12, 24, 48])

            row = {
                'timestamp': data['time'],
                'timestamp_utc': datetime.utcnow().isoformat(),
                'live_temp': live_temp,
                'humidity': data['humidity'], 'wind': data['wind'], 'pressure': data['pressure'],
                'erm_flux': Er_flux, 'beta': beta, 'baseline_temp': baseline_temp,
                'next_predicted_1h': next_predicted + future[1],
                'next_predicted_3h': next_predicted + future[3],
                'next_predicted_6h': next_predicted + future[6],
                'next_predicted_12h': next_predicted + future[12],
                'next_predicted_24h': next_predicted + future[24],
                'next_predicted_48h': next_predicted + future[48],
                'tomorrow_max': data.get('tomorrow_max'), 'tomorrow_min': data.get('tomorrow_min'),
                'improvement_pct': improvement,
                'error_1h': np.nan, 'error_3h': np.nan, 'error_6h': np.nan,
                'error_12h': np.nan, 'error_24h': np.nan, 'error_48h': np.nan
            }

            csv_path = data_dir / f"erm_v{VERSION}_{normalize_city_key(city['name'])}_{today_str}.csv"
            file_exists = csv_path.exists()
            df_row = pd.DataFrame([row])
            df_row.to_csv(csv_path, mode='a', header=not file_exists, index=False)

            logger.info(f"✅ Updated {city['name']} → improvement {improvement:.1f}% | error {realized_error:.3f}")

            state_file = state_dir / f"erm_state_{normalize_city_key(city['name'])}.json"
            erm.save_state(state_file)

    backfill_realized_errors(data_dir)
    background_tasks.add_task(git_backup, data_dir)
    return {"status": "success", "updated": len(cities), "time": datetime.now().isoformat()}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "last_update": datetime.now().isoformat(),
        "cities_tracked": len(DEFAULT_CITIES)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
