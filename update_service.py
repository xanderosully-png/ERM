from fastapi import FastAPI
import uvicorn
import os
import subprocess
import numpy as np
import requests
import csv
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional

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

class ERM_Live_Adaptive:
    def __init__(self, history_size: int = 10, gamma: float = 0.935, lambda_damp: float = 0.28, alpha: float = 0.75):
        self.history_size = history_size
        self.history: deque[float] = deque(maxlen=history_size)
        self.humidity_history: deque[float] = deque(maxlen=history_size)
        self.wind_history: deque[float] = deque(maxlen=history_size)
        self.pressure_history: deque[float] = deque(maxlen=history_size)
        self.Er_history: deque[float] = deque(maxlen=history_size)
        self.gamma = gamma
        self.lambda_damp = lambda_damp
        self.alpha = alpha

    def _derive_variables(self, current_temp: float, current_humidity: float, current_wind: float, current_pressure: float, previous_temp: Optional[float], hour_of_day: int, local_avg_temp: float, local_temp_range: float):
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

    def step(self, current_temp: float, current_humidity: float, current_wind: float, current_pressure: float, previous_temp: Optional[float], hour_of_day: int, local_avg_temp: float, local_temp_range: float):
        Nr, Tr, dphi, k, rhoE, tauE = self._derive_variables(current_temp, current_humidity, current_wind, current_pressure, previous_temp, hour_of_day, local_avg_temp, local_temp_range)
        base = (Nr * Tr * dphi) / max(k, 1e-8)
        f_field = base * (rhoE ** 0.5) * (tauE ** 0.5)
        recursive = 0.0
        if self.Er_history:
            times = np.arange(len(self.Er_history), 0, -1, dtype=np.float32)
            decayed = np.array(self.Er_history, dtype=np.float32) * (self.gamma ** times)
            recursive = np.sum(decayed) ** self.alpha
        Er_new = np.clip(f_field + (self.lambda_damp * recursive), -200, 200)
        self.Er_history.append(Er_new)
        beta = np.clip(np.std(self.history) / (np.std(self.Er_history) + 1e-6), max(0.01, np.std(self.history)/50), max(1.0, np.std(self.history)/2))
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
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone
    }
    for attempt in range(3):
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
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
                'tomorrow_max': daily.get('temperature_2m_max', [None])[1] if len(daily.get('temperature_2m_max', [])) > 1 else None,
                'tomorrow_min': daily.get('temperature_2m_min', [None])[1] if len(daily.get('temperature_2m_min', [])) > 1 else None
            }
        except Exception as e:
            if "429" in str(e):
                print(f"⏳ Rate limit hit for {lat},{lon} – waiting longer")
                time.sleep(5)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None

def git_backup(data_dir: Path):
    print("=== GIT BACKUP STARTED ===")
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    print(f"GITHUB_TOKEN present: {'YES' if token else 'NO'}")
    print(f"GITHUB_REPO: {repo}")
    if not token or not repo:
        print("ERROR: GITHUB_TOKEN or GITHUB_REPO is missing!")
        return
    try:
        repo_root = data_dir.parent
        if not (repo_root / ".git").exists():
            subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
            print("✅ Git repo initialized")
        # Ensure origin remote exists
        remotes = subprocess.run(["git", "remote"], cwd=repo_root, capture_output=True, text=True).stdout
        if "origin" not in remotes:
            remote_url = f"https://{token}@github.com/{repo}.git"
            subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=repo_root, check=True, capture_output=True)
            print("✅ Remote origin added")
        else:
            remote_url = f"https://{token}@github.com/{repo}.git"
            subprocess.run(["git", "remote", "set-url", "origin", remote_url], cwd=repo_root, check=True, capture_output=True)
            print("✅ Remote URL updated")
        subprocess.run(["git", "config", "--global", "user.name", "ERM Bot"], cwd=repo_root, check=True, capture_output=True)
        subprocess.run(["git", "config", "--global", "user.email", "erm-bot@github.com"], cwd=repo_root, check=True, capture_output=True)
        print("✅ Git config set")
        subprocess.run(["git", "add", str(data_dir)], cwd=repo_root, check=True, capture_output=True)
        print("✅ Git add completed")
        # Ensure we are on main branch
        subprocess.run(["git", "checkout", "-B", "main"], cwd=repo_root, check=True, capture_output=True)
        print("✅ Switched to main branch")
        commit_result = subprocess.run(["git", "commit", "-m", f"ERM live update {datetime.now().isoformat()}"], cwd=repo_root, capture_output=True, text=True)
        print(f"Commit result: {commit_result.returncode} - {commit_result.stdout.strip() or commit_result.stderr.strip()}")
        if commit_result.returncode == 0:
            push_result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd=repo_root, capture_output=True, text=True)
            print(f"Push result: {push_result.returncode} - {push_result.stdout.strip() or push_result.stderr.strip()}")
            print("✅ GitHub backup SUCCESSFUL")
        else:
            print("No changes to commit (normal)")
    except Exception as e:
        print(f"Git backup FAILED: {e}")

@app.get("/update")
async def update_data():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "ERM_Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Data directory ready: {data_dir}")

    cities = DEFAULT_CITIES
    erms = {city['name']: ERM_Live_Adaptive() for city in cities}
    previous_data = {city['name']: None for city in cities}

    now = datetime.now()
    today_str = now.strftime('%Y%m%d')

    for city in cities:
        data = fetch_multi_variable_data(city['lat'], city['lon'], city['tz'])
        if not data:
            print(f"⚠️ Failed to fetch data for {city['name']}")
            time.sleep(3)
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

        print(f"✅ CSV written for {city['name']} → {csv_path} ({csv_path.stat().st_size} bytes)")

        previous_data[city['name']] = {'live_temp': live_temp, 'next_predicted': next_predicted}
        time.sleep(3)  # ← Rate-limit safety

    print("✅ All cities processed. Starting git backup...")
    git_backup(data_dir)
    return {"status": "success", "updated": len(cities), "time": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
