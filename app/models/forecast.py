"""Lightweight offline forecasting and time-to-threshold estimation."""

from __future__ import annotations

import numpy as np


def short_forecast(series_values, steps: int = 5) -> float | None:
    """Linear-trend forecast clipped to [0, 1] (for normalized scores)."""
    y = np.asarray(series_values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return None
    x = np.arange(len(y), dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
        future = slope * np.arange(len(y), len(y) + steps, dtype=float) + intercept
        return float(np.clip(future[-1], 0.0, 1.0))
    except Exception:
        return None


def forecast_series(series_values, steps: int = 10) -> tuple[list[float], list[float]] | None:
    """Return (future_index, forecast_values) for plotting a raw telemetry trend."""
    y = np.asarray(series_values, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 5:
        return None
    x = np.arange(len(y), dtype=float)
    try:
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(y), len(y) + steps, dtype=float)
        future_y = slope * future_x + intercept
        return future_x.tolist(), future_y.tolist()
    except Exception:
        return None


def estimate_ttf_hours(score_history, warning_threshold: float, critical_threshold: float = 0.9) -> float | None:
    """Estimate hours until the score trend crosses the critical threshold."""
    hist = np.asarray(score_history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) < 20:
        return None
    x = np.arange(len(hist), dtype=float)
    try:
        slope, intercept = np.polyfit(x, hist, 1)
        if slope <= 1e-9:
            return None
        current = slope * (len(hist) - 1) + intercept
        if current >= critical_threshold:
            return 0.0
        steps_to_critical = (critical_threshold - current) / slope
        # Each step ~ one monitoring cycle; report as relative units (cycles).
        return float(max(0.0, steps_to_critical))
    except Exception:
        return None
