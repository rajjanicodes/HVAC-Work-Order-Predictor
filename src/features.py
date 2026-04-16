"""Extracts 23 statistical features from a temperature time-series. Same function used at train and inference time."""

import numpy as np
from scipy import stats

from src.config import TEMP_BAND_HIGH, TEMP_BAND_LOW


def extract_features(timeseries_data: list[dict]) -> dict[str, float]:
    """Turns a list of {timestamp, temperature} dicts into a flat feature dict."""

    # Sort by timestamp so all derived features are temporally consistent
    ts = sorted(timeseries_data, key=lambda d: d["timestamp"])
    times = np.array([d["timestamp"] for d in ts])
    temps = np.array([d["temperature"] for d in ts])
    n = len(temps)

    # Quartiles for IQR
    q25, q75 = np.percentile(temps, [25, 75])

    # Gaps between consecutive readings (seconds)
    intervals = np.diff(times)

    # Fit a line to temp vs hours elapsed to capture trend direction
    hours = (times - times[0]) / 3600.0
    slope, _, r, _, _ = stats.linregress(hours, temps)

    # Jump size between back-to-back readings
    jumps = np.abs(np.diff(temps))

    # Readings outside the normal band
    above = int(np.sum(temps > TEMP_BAND_HIGH))
    below = int(np.sum(temps < TEMP_BAND_LOW))
    mean_t = float(np.mean(temps))

    return {
        "mean_temp":        mean_t,
        "std_temp":         float(np.std(temps)),
        "min_temp":         float(np.min(temps)),
        "max_temp":         float(np.max(temps)),
        "median_temp":      float(np.median(temps)),
        "iqr_temp":         float(q75 - q25),
        "temp_range":       float(np.max(temps) - np.min(temps)),
        "skew_temp":        float(stats.skew(temps)),
        "kurtosis_temp":    float(stats.kurtosis(temps)),
        "duration_sec":     float(times[-1] - times[0]),
        "num_readings":     n,
        "mean_interval":    float(np.mean(intervals)),
        "std_interval":     float(np.std(intervals)),
        "slope":            float(slope),
        "r_squared":        float(r ** 2),
        "mean_consec_diff": float(np.mean(jumps)),
        "max_consec_diff":  float(np.max(jumps)),
        "std_consec_diff":  float(np.std(jumps)),
        "count_above_80":   above,
        "count_below_20":   below,
        "frac_above_80":    above / n,
        "frac_below_20":    below / n,
        "cv_temp":          float(np.std(temps) / mean_t) if mean_t != 0 else 0.0,
    }


# Ordered feature names derived by calling extract_features on a dummy record
FEATURE_NAMES: list[str] = list(extract_features([
    {"timestamp": 0, "temperature": 50.0},
    {"timestamp": 1, "temperature": 51.0},
]).keys())
