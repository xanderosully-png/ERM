from fastapi import FastAPI
import uvicorn
import os
import subprocess
import numpy as np
import requests
import csv
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
