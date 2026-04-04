# erm_model.py (v3 – Mature, Modular, Diurnal-Aware Edition)
from collections import deque
import numpy as np
import math
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class ERM_Live_Adaptive:
    # Tunable constants (all exposed)
    GAMMA = 0.932
    LAMBDA_DAMP = 0.225
    ALPHA = 0.76
    PHYSICS_WEIGHT = 0.61
    EMPIRICAL_WEIGHT = 0.39
    SMOOTHING = 0.67
    EXTREME_ER_THRESHOLD = 6.5
    HISTORY_SIZE = 24

    # Nr coefficients (quadratic + cross terms)
    NR_TEMP_WT = 1.18
    NR_PRESS_WT = 0.83
    NR_HUM_WT = 1.13
    NR_WIND_WT = 0.91
    NR_TH_CROSS = 0.67
    NR_PW_CROSS = 0.47
    NR_TP_CROSS = -0.37
    NR_PH_CROSS = -0.58

    # Phase-space
    DELTA_PHI_LAG_WINDOW = 6
    PHASE_VOLATILITY_GAIN = 0.44

    # Diurnal & climatology
    DIURNAL_AMPLITUDE = 1.75          # °C in stable regimes
    CLIMATOLOGY_SMOOTH = 0.86

    # Adaptation
    ADAPTATION_RATE = 0.075

    def __init__(self, city_name: str = "Unknown"):
        self.city_name = city_name
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.humidity_history = deque(maxlen=self.HISTORY_SIZE)
        self.wind_history = deque(maxlen=self.HISTORY_SIZE)
        self.pressure_history = deque(maxlen=self.HISTORY_SIZE)
        self.Er_history = deque(maxlen=self.HISTORY_SIZE)
        self.delta_phi_history = deque(maxlen=60)

        self.performance_score = 0.0
        self.current_regime = "stable"
        self.regime_tracker: Dict[str, Dict] = {}
        self.gamma = self.GAMMA
        self.lambda_damp = self.LAMBDA_DAMP
        self.alpha = self.ALPHA
        self.smoothed_er = 0.0
        self.last_predicted: Optional[float] = None
        self.local_climatology = 15.0

    # ── Private modular helpers (much cleaner now) ──
    def _diurnal_factor(self, hour: int) -> float:
        """Sinusoidal diurnal cycle – strongest effect in stable regimes."""
        return self.DIURNAL_AMPLITUDE * math.sin(2 * math.pi * (hour - 3) / 24.0)

    def _compute_nr(self, temp: float, pressure: float, humidity: float, wind: float) -> float:
        t_n = np.clip(temp / 50.0, -2.0, 2.0)
        p_n = np.clip((pressure - 1013.0) / 50.0, -2.0, 2.0)
        h_n = np.clip(humidity / 100.0, 0.0, 1.2)
        w_n = np.clip(wind / 20.0, -2.0, 2.0)

        t_w = 1.35 if self.current_regime == "stable" else 1.0
        h_w = 1.30 if self.current_regime in ["storm", "chaotic"] else 0.90

        quad = (t_w * self.NR_TEMP_WT * t_n**2 +
                self.NR_PRESS_WT * p_n**2 +
                h_w * self.NR_HUM_WT * h_n**2 +
                self.NR_WIND_WT * w_n**2)

        cross = (self.NR_TH_CROSS * t_n * h_n +
                 self.NR_PW_CROSS * p_n * w_n +
                 self.NR_TP_CROSS * t_n * p_n +
                 self.NR_PH_CROSS * p_n * h_n)

        nr_val = math.sqrt(max(quad + cross + 0.08, 0.05))
        return float(np.clip(nr_val * 1.13, 0.25, 4.2))

    def _compute_delta_phi(self, pressure_lag: float, humidity_lag: float,
                           wind_lag: float, volatility: float, short_trend: float) -> float:
        phase_input = (0.52 * pressure_lag +
                       0.31 * humidity_lag +
                       0.12 * (wind_lag / 180.0) +
                       0.05 * short_trend)
        coupling = 1.0 + self.PHASE_VOLATILITY_GAIN * math.tanh(volatility / 3.5)
        phi = math.sin(phase_input) * coupling
        self.delta_phi_history.append(phi)
        return float(phi)

    def _detect_regime(self, volatility: float, p_drop: float, h_spike: float,
                       wind: float, cloud: float) -> str:
        if volatility > 7.5 or abs(wind) > 14.0:
            return "chaotic"
        if p_drop < -2.2 and h_spike > 18.0 and volatility > 4.0:
            return "storm"
        if abs(p_drop) < 0.6 and volatility < 1.4:
            return "stable"
        return "seasonal"

    def _light_adapt(self):
        """Tiny online adaptation based on recent regime success."""
        for regime, data in self.regime_tracker.items():
            if data["count"] >= 8:
                success_rate = data["success"] / data["count"]
                if success_rate < 0.45:
                    self.alpha = max(0.65, self.alpha * (1 - self.ADAPTATION_RATE))
                elif success_rate > 0.75:
                    self.alpha = min(0.85, self.alpha * (1 + self.ADAPTATION_RATE * 0.5))

    # ── Main step (now synchronous, modular, and cleaner) ──
    def step(self, current_temp: float, current_humidity: float, current_wind: float,
             current_pressure: float, current_rain_prob: float = 20.0,
             current_cloud_cover: float = 50.0, current_solar: float = 300.0,
             current_wind_dir: float = 180.0, satellite_cloud_cover: float = 30.0,
             satellite_radiation: float = 300.0, hour_of_day: int = 12,
             local_avg_temp: float = 15.0, neighbor_influence: float = 0.0,
             dry_run: bool = False, **kwargs) -> tuple:

        self._update_histories(current_temp, current_humidity, current_wind, current_pressure, dry_run)

        if len(self.history) < 4:
            warmup = (current_temp - local_avg_temp) * 0.4
            return 0.0, current_temp + warmup, 0.65, 0.0

        # Core features
        recent_t = np.array(self.history)
        volatility = float(np.std(recent_t))
        diffs = np.diff(recent_t)
        short_trend = float(np.mean(diffs[-4:])) if len(diffs) >= 4 else 0.0

        # Satellite + cloud blending
        sat_weight = 0.7 if satellite_radiation > 80 else 0.35
        blended_cloud = (1 - sat_weight) * current_cloud_cover + sat_weight * satellite_cloud_cover
        solar_adjust = np.clip((satellite_radiation - 300) / 750.0, -0.85, 1.25)
        volatility *= (1.0 + 0.28 * (blended_cloud / 100.0))

        # Lags
        p_drop = np.mean(np.diff(list(self.pressure_history)[-4:])) if len(self.pressure_history) >= 4 else 0.0
        h_spike = np.mean(list(self.humidity_history)[-4:]) - 50.0

        self.current_regime = self._detect_regime(volatility, p_drop, h_spike, current_wind, blended_cloud)

        # Update climatology
        self.local_climatology = self.CLIMATOLOGY_SMOOTH * self.local_climatology + (1 - self.CLIMATOLOGY_SMOOTH) * local_avg_temp

        # Empirical term (now uses diurnal + rain prob lightly)
        sat_forcing = solar_adjust * 0.42 + (blended_cloud / 100.0 - 0.5) * 0.32
        regime_damp = 0.78 if self.current_regime in ["storm", "chaotic"] else 1.0
        empirical = (short_trend + sat_forcing + neighbor_influence) * self.alpha * regime_damp

        if self.current_regime == "stable":
            empirical += (current_temp - self.local_climatology) * 0.085
            empirical += 0.35 * self._diurnal_factor(hour_of_day)
            empirical += current_rain_prob * -0.008  # light negative bias when rainy

        # Physics core
        self.nr = self._compute_nr(current_temp, current_pressure, current_humidity, current_wind)
        self.tr = solar_adjust * 0.38 + (blended_cloud / 100.0 - 0.5) * 0.28

        pressure_lag = np.mean(np.diff(list(self.pressure_history)[-self.DELTA_PHI_LAG_WINDOW:])) if len(self.pressure_history) >= self.DELTA_PHI_LAG_WINDOW else 0.0
        humidity_lag = np.mean(np.diff(list(self.humidity_history)[-self.DELTA_PHI_LAG_WINDOW:])) if len(self.humidity_history) >= self.DELTA_PHI_LAG_WINDOW else 0.0
        wind_lag = current_wind_dir - 180.0

        self.delta_phi_multi = self._compute_delta_phi(pressure_lag, humidity_lag, wind_lag, volatility, short_trend)

        beta = np.clip(self.gamma * (1.0 - self.lambda_damp * volatility / 9.5), 0.42, 1.18)

        physics_term = math.tanh((self.nr * self.tr * self.delta_phi_multi) / (beta + 1e-6)) * 8.2

        Er_new = self.PHYSICS_WEIGHT * physics_term + self.EMPIRICAL_WEIGHT * empirical
        Er_new = np.clip(Er_new, -8.5, 8.5)

        self.smoothed_er = self.SMOOTHING * Er_new + (1 - self.SMOOTHING) * self.smoothed_er
        Er_new = self.smoothed_er

        next_predicted = current_temp + Er_new * beta
        next_predicted = np.clip(next_predicted, current_temp - 45, current_temp + 45)

        # Tracking & light adaptation
        logger.info(f"ERM.v3 [{self.city_name}] T={current_temp:.2f}°C Er={Er_new:.3f} regime={self.current_regime} "
                    f"nr={self.nr:.2f} pred={next_predicted:.2f} vol={volatility:.2f}")

        self.Er_history.append(Er_new)
        self.last_predicted = next_predicted
        self.performance_score = 0.72 * self.performance_score + 0.28 * (1 - abs(Er_new) / 11.0)

        # Regime tracking
        self.regime_tracker.setdefault(self.current_regime, {"count": 0, "success": 0})
        self.regime_tracker[self.current_regime]["count"] += 1
        if abs(Er_new) < 3.0:
            self.regime_tracker[self.current_regime]["success"] += 1

        self._light_adapt()

        if abs(Er_new) > self.EXTREME_ER_THRESHOLD:
            logger.warning(f"⚠️ Extreme Er in {self.city_name}: {Er_new:.2f}")

        return Er_new, float(next_predicted), beta, short_trend

    def _update_histories(self, temp, hum, wind, pressure, dry_run):
        if not dry_run:
            self.history.append(temp)
            self.humidity_history.append(hum)
            self.wind_history.append(wind)
            self.pressure_history.append(pressure)

    # ── Improved future prediction (light recursive simulation) ──
    def predict_future(self, horizons: List[int] = [1, 3, 6, 12, 24]) -> Dict[str, Any]:
        if self.last_predicted is None:
            return {"error": "Not enough data yet"}

        predictions = {}
        base = self.last_predicted
        temp = base
        current_er = self.smoothed_er
        regime_damp = 0.75 if self.current_regime in ["storm", "chaotic"] else 1.0
        reversion = 0.038 if self.current_regime == "stable" else 0.014

        for h in sorted(set(horizons)):
            # Mini recursive step
            step_er = current_er * (0.89 ** (h - 1)) * regime_damp
            temp += step_er
            # Mean-reversion pull toward climatology
            temp += reversion * (self.local_climatology - temp)
            predictions[f"{h}h"] = round(float(np.clip(temp, base - 22, base + 22)), 2)

        return {
            "predictions": predictions,
            "current_regime": self.current_regime,
            "confidence": round(max(0.18, 1.0 - abs(self.smoothed_er) / 9.5), 2),
            "method": "damped_recursive_with_reversion"
        }

    def replay_history(self, temp_series: List[float], **kwargs) -> Dict:
        """Robust replay with sensible defaults."""
        defaults = {
            "current_humidity": 60.0, "current_wind": 8.0, "current_pressure": 1013.0,
            "current_cloud_cover": 50.0, "current_solar": 300.0, "hour_of_day": 12,
            "local_avg_temp": 15.0, "dry_run": False
        }
        for k, v in defaults.items():
            if k not in kwargs:
                kwargs[k] = v

        results = []
        for i, temp in enumerate(temp_series):
            _, pred, _, _ = self.step(current_temp=temp, **kwargs)
            if i > 0:
                results.append({"actual": temp, "predicted": pred, "error": abs(temp - pred)})

        return {
            "mae": float(np.mean([r["error"] for r in results])) if results else 0.0,
            "final_regime": self.current_regime,
            "steps": len(results),
            "performance_score": round(self.performance_score, 3)
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Handy overview for debugging / monitoring."""
        return {
            "current_regime": self.current_regime,
            "performance_score": round(self.performance_score, 3),
            "regime_tracker": self.regime_tracker,
            "smoothed_er": round(self.smoothed_er, 3),
            "nr": round(self.nr, 3),
            "last_delta_phi": round(self.delta_phi_history[-1], 3) if self.delta_phi_history else None,
            "local_climatology": round(self.local_climatology, 2),
            "history_length": len(self.history)
      }
