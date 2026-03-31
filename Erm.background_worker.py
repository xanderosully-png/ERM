import os
import sys
import time
import subprocess
import numpy as np
import requests
import csv
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque
from pathlib import Path
import signal
from logging.handlers import RotatingFileHandler

# =============================================
# ERM v4.4 Background Worker + GitHub Backup
# =============================================
# Runs 24/7 as Render Background Worker
# Logs live data + predictions to per-city CSVs in ERM_Data/
# Automatically commits & pushes CSVs to your GitHub repo every 4 hours
# Uses GITHUB_PAT environment variable for authentication

VERSION = "4.4"

# Default cities
DEFAULT_CITIES = [
    {"name": "Columbus_OH", "lat": 39.9612, "lon": -82.9988, "tz": "America/New_York",
     "local_avg_temp": 11.5, "local_temp_range": 35.0},
    {"name": "Miami_FL",    "lat": 25.7617, "lon": -80.1918, "tz": "America/New_York",
     "local_avg_temp": 25.0, "local_temp_range": 15.0},
    {"name": "New_York_NY", "lat": 40.7128, "lon": -74.0060, "tz": "America/New_York",
     "local_avg_temp": 12.0, "local_temp_range": 32.0},
    {"name": "Los_Angeles_CA", "lat": 34.0522, "lon": -118.2437, "tz": "America/Los_Angeles",
     "local_avg_temp": 18.0, "local_temp_range": 20.0},
    {"name": "London_UK",   "lat": 51.5074, "lon": -0.1278, "tz": "Europe/London",
     "local_avg_temp": 11.0, "local_temp_range": 25.0},
    {"name": "Tokyo_JP",    "lat": 35.6895, "lon": 139.6917, "tz": "Asia/Tokyo",
     "local_avg_temp": 16.0, "local_temp_range": 28.0},
]

def load_cities(base_dir: Path) -> List[Dict]:
    for candidate in [base_dir / "cities.json", base_dir / "ERM_Data" / "cities.json"]:
        if candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    loaded = json.load(f)
                logging.info(f"✅ Loaded {len(loaded)} cities from {candidate.name}")
                return loaded
            except Exception as e:
                logging.warning(f"Failed to load {candidate}: {e}")
    logging.info("Using built-in default cities")
    return [c.copy() for c in DEFAULT_CITIES]

class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 10, gamma: float = 0.935,
                 lambda_damp: float = 0.28, alpha: float = 0.75):
        self.history_size = history_size
        self.history: deque[float] = deque(maxlen=history_size)
        self.humidity_history: deque[float] = deque(maxlen=history_size)
        self.wind_history: deque[float] = deque(maxlen=history_size)
        self.pressure_history: deque[float] = deque(maxlen=history_size)
        self.Er_history: deque[float] = deque(maxlen=history_size)
        self.gamma = gamma
        self.lambda_damp = lambda_damp
        self.alpha = alpha

    def _derive_variables(self, current_temp: float, current_humidity: float,
                          current_wind: float, current_pressure: float,
                          previous_temp: Optional[float], hour_of_day: int,
                          local_avg_temp: float, local_temp_range: float):
        self.history.append(current_temp)
        self.humidity_history.append(current_humidity)
        self.wind_history.append(current_wind)
        self.pressure_history.append(current_pressure)

        if len(self.history) < 2:
            return 4.0, 0.85, 0.0, 1.0, 1.05, 0.97

        recent_t = np.array(self.history, dtype=np.float32)
        diffs = np.diff(recent_t)
        Nr = len(recent_t) * (1 + np.var(recent_t) / 10)
        Tr = max(0.6, 1 - np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-6)) * (1 - np.mean(self.humidity_history) / 200)
        dphi = current_temp - previous_temp if previous_temp is not None else 0.0
        k = 0.8 + np.mean(np.abs(diffs)) / 5 + np.mean(self.wind_history) / 50
        rhoE = 1.0 + ((np.mean(recent_t) - local_avg_temp) / local_temp_range) + (np.mean(self.pressure_history) - 1013) / 1000
        tauE = 0.95 + (hour_of_day / 48)

        return Nr, Tr, dphi, k, rhoE, tauE

    def step(self, current_temp: float, current_humidity: float, current_wind: float,
             current_pressure: float, previous_temp: Optional[float], hour_of_day: int,
             local_avg_temp: float, local_temp_range: float):
        Nr, Tr, dphi, k, rhoE, tauE = self._derive_variables(
            current_temp, current_humidity, current_wind, current_pressure,
            previous_temp, hour_of_day, local_avg_temp, local_temp_range)

        base = (Nr * Tr * dphi) / max(k, 1e-8)
        f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)

        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            recursive = np.sum(decayed) ** self.alpha

        Er_new = f_field + (self.lambda_damp * recursive)
        Er_new = np.clip(Er_new, -200, 200)
        self.Er_history.append(Er_new)

        beta_raw = np.std(self.history) / (np.std(self.Er_history) + 1e-6)
        beta = np.clip(beta_raw, max(0.01, np.std(self.history)/50), max(1.0, np.std(self.history)/2))

        next_predicted = current_temp + (Er_new * beta)
        return Er_new, next_predicted, beta

    def predict_future(self, steps_list: List[int] = [1, 3, 6, 12, 24, 48]):
        if len(self.Er_history) < 3:
            last = float(self.Er_history[-1]) if self.Er_history else 0.0
            return {s: last for s in steps_list}
        x = np.arange(len(self.Er_history), dtype=np.float32)
        y = np.array(self.Er_history, dtype=np.float32)
        slope, intercept = np.polyfit(x, y, 1)
        return {s: float(np.clip(slope * (len(x) + s) + intercept, -200, 200)) for s in steps_list}

def fetch_multi_variable_data(lat: float, lon: float, timezone: str) -> Optional[Dict]:
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"
           f"&daily=temperature_2m_max,temperature_2m_min&timezone={timezone.replace('/', '%2F')}")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=10)
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
                'tomorrow_max': daily['temperature_2m_max'][1],
                'tomorrow_min': daily['temperature_2m_min'][1]
            }
        except Exception as e:
            logging.warning(f"Open-Meteo fetch failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None

def setup_logging(data_dir: Path):
    log_path = data_dir / "erm_daemon.log"
    log_handler = RotatingFileHandler(str(log_path), maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[log_handler, logging.StreamHandler(sys.stdout)],
        force=True
    )

def git_backup(data_dir: Path, repo_path: Path):
    """Commit and push ERM_Data folder to GitHub using GITHUB_PAT"""
    pat = os.getenv("GITHUB_PAT")
    repo_url = os.getenv("GITHUB_REPO_URL")  # e.g. https://github.com/USERNAME/REPO.git
    if not pat or not repo_url:
        return  # silent fail if not configured

    try:
        # Ensure we are in the repo root
        os.chdir(repo_path)

        # Configure git with PAT
        remote_url = repo_url.replace("https://", f"https://{pat}@")
        subprocess.run(["git", "remote", "set-url", "origin", remote_url], check=True, capture_output=True)

        # Add only ERM_Data changes
        subprocess.run(["git", "add", str(data_dir)], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"ERM auto-backup {datetime.now().isoformat()}"], check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)  # change 'main' if your default branch is different
        logging.info("✅ GitHub backup successful")
    except subprocess.CalledProcessError as e:
        logging.warning(f"Git backup failed (non-fatal): {e.stderr.decode().strip()}")
    except Exception as e:
        logging.warning(f"Git backup error: {e}")

def run_background_worker():
    # EXE-aware / Render-aware data directory
    if getattr(sys, 'frozen', False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent
    data_dir = base_dir / "ERM_Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(data_dir)
    logging.info(f"🚀 ERM v{VERSION} Background Worker STARTED (GitHub backup enabled)")

    cities = load_cities(base_dir)
    erms = {city['name']: ERM_Live_Adaptive() for city in cities}
    previous_data = {city['name']: None for city in cities}

    # Signal handler
    def signal_handler(sig, frame):
        logging.info("🛑 Shutdown signal received. Final GitHub backup...")
        git_backup(data_dir, base_dir)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    backup_counter = 0
    update_interval_seconds = int(os.getenv("ERM_INTERVAL_SECONDS", 3600))

    while True:
        cycle_start = time.perf_counter()
        now = datetime.now()
        today_str = now.strftime('%Y%m%d')

        logging.info(f"=== LIVE UPDATE at {now.strftime('%Y-%m-%d %H:%M:%S')} ===")

        for city in cities:
            data = fetch_multi_variable_data(city['lat'], city['lon'], city['tz'])
            if not data:
                logging.warning(f"{city['name']}: API unavailable — skipping")
                continue

            live_temp = data['temp']
            hour_of_day = now.hour
            erm = erms[city['name']]
            prev = previous_data[city['name']]

            if prev is not None:
                erm_err = abs(live_temp - prev['next_predicted'])
                baseline_err = abs(live_temp - prev['live_temp'])
                improvement = 100 * (baseline_err - erm_err) / max(baseline_err, 0.01)
            else:
                improvement = 0.0

            Er_flux, next_predicted, beta = erm.step(
                live_temp, data['humidity'], data['wind'], data['pressure'],
                prev['live_temp'] if prev else None,
                hour_of_day, city['local_avg_temp'], city['local_temp_range']
            )

            future = erm.predict_future([1, 3, 6, 12, 24, 48])

            row = {
                'timestamp': data['time'],
                'live_temp': live_temp,
                'humidity': data['humidity'],
                'wind': data['wind'],
                'pressure': data['pressure'],
                'erm_flux': Er_flux,
                'beta': beta,
                'next_predicted_1h': next_predicted + (future[1] * beta),
                'next_predicted_3h': next_predicted + (future[3] * beta),
                'next_predicted_6h': next_predicted + (future[6] * beta),
                'next_predicted_12h': next_predicted + (future[12] * beta),
                'next_predicted_24h': next_predicted + (future[24] * beta),
                'next_predicted_48h': next_predicted + (future[48] * beta),
                'tomorrow_max': data.get('tomorrow_max'),
                'tomorrow_min': data.get('tomorrow_min'),
                'improvement_pct': improvement
            }

            csv_path = data_dir / f"erm_v{VERSION}_{city['name'].lower().replace(' ', '_')}_{today_str}.csv"
            file_exists = csv_path.exists()
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()

            logging.info(
                f"  {city['name']:15} | Temp: {live_temp:5.1f}°C | β: {beta:.3f} | "
                f"Next 1h: {next_predicted + (future[1] * beta):5.1f}°C | Imp: {improvement:5.1f}%"
            )

            previous_data[city['name']] = {'live_temp': live_temp, 'next_predicted': next_predicted}

        logging.info("✅ Cycle complete.")

        # GitHub backup every 4 hours
        backup_counter += 1
        if backup_counter >= 4:
            git_backup(data_dir, base_dir)
            backup_counter = 0

        # Dynamic sleep
        elapsed = time.perf_counter() - cycle_start
        sleep_seconds = max(0.0, update_interval_seconds - elapsed)
        logging.info(f"Next update in {sleep_seconds/60:.1f} minutes...")
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    # Render / env configuration
    logging.info(f"ERM Background Worker v{VERSION} starting with interval = {os.getenv('ERM_INTERVAL_SECONDS', 3600)}s")
    run_background_worker()
