"""Concept drift detection using KS test + Population Stability Index (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats

    _SCIPY = True
except Exception:  # pragma: no cover
    _SCIPY = False


def compute_psi(reference_col, current_col, bins: int = 10) -> float:
    ref = np.asarray(reference_col, dtype=float)
    cur = np.asarray(current_col, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-8, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-8, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _ks_pvalue(ref: np.ndarray, cur: np.ndarray) -> float:
    if not _SCIPY:
        return 1.0
    try:
        _, p_value = scipy_stats.ks_2samp(ref, cur)
        return float(p_value)
    except Exception:
        return 1.0


class DriftDetector:
    def __init__(
        self,
        reference_window_size: int = 200,
        ks_pvalue_threshold: float = 0.01,
        psi_threshold: float = 0.2,
        feature_ratio_threshold: float = 0.3,
    ):
        self.reference_window_size = reference_window_size
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.psi_threshold = psi_threshold
        self.feature_ratio_threshold = feature_ratio_threshold
        self.reference_window: pd.DataFrame | None = None

    def check(self, feature_df: pd.DataFrame) -> dict:
        if feature_df is None or feature_df.empty:
            return {"is_drift": False, "drift_score": 0.0, "drifted_features": []}

        if self.reference_window is None or self.reference_window.empty:
            self.reference_window = feature_df.tail(self.reference_window_size).copy()
            return {"is_drift": False, "drift_score": 0.0, "drifted_features": [], "status": "reference_initialized"}

        common = [c for c in self.reference_window.columns if c in feature_df.columns]
        if not common:
            self.reference_window = feature_df.tail(self.reference_window_size).copy()
            return {"is_drift": False, "drift_score": 0.0, "drifted_features": [], "status": "reference_reset"}

        current = feature_df[common].tail(self.reference_window_size)
        reference = self.reference_window[common].tail(self.reference_window_size)

        drifted: list[str] = []
        details: dict[str, dict] = {}
        for col in common:
            ref_col = reference[col].to_numpy(dtype=float)
            cur_col = current[col].to_numpy(dtype=float)
            ref_col = ref_col[np.isfinite(ref_col)]
            cur_col = cur_col[np.isfinite(cur_col)]
            if len(ref_col) < 20 or len(cur_col) < 20:
                continue
            p_value = _ks_pvalue(ref_col, cur_col)
            psi = compute_psi(ref_col, cur_col)
            is_drift = (p_value < self.ks_pvalue_threshold) or (psi > self.psi_threshold)
            if is_drift:
                drifted.append(col)
            details[col] = {"ks_pvalue": p_value, "psi": psi, "is_drift": is_drift}

        ratio = (len(drifted) / len(common)) if common else 0.0
        self.reference_window = current.copy()
        return {
            "is_drift": ratio >= self.feature_ratio_threshold,
            "drift_score": ratio,
            "drift_ratio": ratio,
            "drifted_features": drifted,
            "feature_details": details,
        }


class TabConceptDriftChecker:
    """Stable-reference drift check used by legacy CustomMonitoringTab."""

    def __init__(
        self,
        reference_window_size: int = 200,
        ks_pvalue_threshold: float = 0.01,
        psi_threshold: float = 0.2,
        feature_ratio_threshold: float = 0.3,
    ):
        self.reference_window_size = reference_window_size
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.psi_threshold = psi_threshold
        self.feature_ratio_threshold = feature_ratio_threshold
        self.reference_window: pd.DataFrame | None = None

    def set_reference(self, feature_df: pd.DataFrame) -> bool:
        if feature_df is None or feature_df.empty:
            return False
        self.reference_window = feature_df.tail(self.reference_window_size).copy()
        return True

    def check(self, feature_df: pd.DataFrame, *, auto_rebaseline_on_drift: bool = False) -> dict:
        if feature_df is None or feature_df.empty:
            return {"is_drift": False, "drift_score": 0.0, "drifted_features": []}

        if self.reference_window is None or self.reference_window.empty:
            self.reference_window = feature_df.tail(self.reference_window_size).copy()
            return {
                "is_drift": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "status": "reference_initialized",
            }

        common_cols = [c for c in self.reference_window.columns if c in feature_df.columns]
        if not common_cols:
            self.reference_window = feature_df.tail(self.reference_window_size).copy()
            return {
                "is_drift": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "status": "reference_reset",
            }

        drifted: list[str] = []
        feature_details: dict[str, dict] = {}
        current_window = feature_df[common_cols].tail(self.reference_window_size).copy()
        reference_window = self.reference_window[common_cols].tail(self.reference_window_size).copy()

        for col in common_cols:
            ref_col = reference_window[col].to_numpy(dtype=float)
            cur_col = current_window[col].to_numpy(dtype=float)
            ref_col = ref_col[np.isfinite(ref_col)]
            cur_col = cur_col[np.isfinite(cur_col)]
            if len(ref_col) < 20 or len(cur_col) < 20:
                continue

            p_value = _ks_pvalue(ref_col, cur_col)
            psi_score = compute_psi(ref_col, cur_col)
            is_feature_drift = (p_value < self.ks_pvalue_threshold) or (psi_score > self.psi_threshold)
            if is_feature_drift:
                drifted.append(col)
            feature_details[col] = {
                "ks_pvalue": float(p_value),
                "psi": float(psi_score),
                "is_drift": bool(is_feature_drift),
            }

        drift_ratio = (len(drifted) / len(common_cols)) if common_cols else 0.0
        is_drift = drift_ratio >= self.feature_ratio_threshold
        result = {
            "is_drift": bool(is_drift),
            "drift_score": float(drift_ratio),
            "drift_ratio": float(drift_ratio),
            "drifted_features": drifted,
            "feature_details": feature_details,
        }

        if is_drift and auto_rebaseline_on_drift:
            self.reference_window = current_window.copy()
            result["status"] = "reference_rebaselined"
        return result


def compute_adaptive_threshold(
    score_history: list[float],
    latest_score: float,
    *,
    base_threshold: float = 0.6,
    threshold_sigma: float = 3.0,
    threshold_min: float = 0.35,
    threshold_max: float = 0.995,
    score_history_maxlen: int = 500,
    min_samples: int = 20,
) -> tuple[float, list[float]]:
    """Rolling median + sigma*std threshold; returns (threshold, updated_history)."""
    history = list(score_history)
    history.append(float(latest_score))
    if len(history) > score_history_maxlen:
        history = history[-score_history_maxlen:]

    if len(history) < min_samples:
        return max(base_threshold, threshold_min), history

    hist = np.asarray(history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) < min_samples:
        return max(base_threshold, threshold_min), history

    median = float(np.median(hist))
    std = float(np.std(hist))
    threshold = median + (threshold_sigma * std)
    threshold = max(threshold_min, min(threshold_max, threshold))
    return threshold, history
