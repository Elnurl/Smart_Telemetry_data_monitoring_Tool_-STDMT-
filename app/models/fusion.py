"""Multi-model fusion, adaptive thresholding, health state, and XAI reasons."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FusionEngine:
    def __init__(
        self,
        base_threshold: float = 0.6,
        threshold_sigma: float = 3.0,
        threshold_min: float = 0.35,
        threshold_max: float = 0.995,
        score_history_maxlen: int = 500,
    ):
        self.base_threshold = base_threshold
        self.threshold_sigma = threshold_sigma
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.score_history_maxlen = score_history_maxlen
        self.score_history: list[float] = []

    def fuse(self, model_scores: dict[str, float], weights: dict[str, float] | None = None) -> float:
        if not model_scores:
            return 0.0
        weights = weights or {}
        weight_sum = sum(weights.get(name, 1.0) for name in model_scores)
        fused = 0.0
        for name, score in model_scores.items():
            w = weights.get(name, 1.0) / max(weight_sum, 1e-9)
            fused += w * score
        return float(np.clip(fused, 0.0, 1.0))

    def adaptive_threshold(self, latest_score: float) -> float:
        self.score_history.append(float(latest_score))
        if len(self.score_history) > self.score_history_maxlen:
            self.score_history = self.score_history[-self.score_history_maxlen:]

        hist = np.asarray(self.score_history, dtype=float)
        hist = hist[np.isfinite(hist)]
        if len(hist) < 20:
            return max(self.base_threshold, self.threshold_min)

        median = float(np.median(hist))
        std = float(np.std(hist))
        threshold = median + self.threshold_sigma * std
        return float(max(self.threshold_min, min(self.threshold_max, threshold)))

    @staticmethod
    def classify(fused_score: float, decision_threshold: float) -> tuple[str, bool, bool]:
        critical_threshold = max(0.9, decision_threshold + 0.1)
        is_critical = fused_score >= critical_threshold
        is_warning = (fused_score >= decision_threshold) and not is_critical
        if is_critical:
            return "Critical", is_warning, is_critical
        if is_warning:
            return "Warning", is_warning, is_critical
        return "Healthy", is_warning, is_critical

    @staticmethod
    def build_xai_reasons(
        feature_df: pd.DataFrame | None,
        fused_score: float,
        threshold: float,
        drift_result: dict,
        model_scores: dict[str, float],
    ) -> list[str]:
        reasons: list[str] = []
        if fused_score > threshold:
            reasons.append(f"Fusion score crossed adaptive threshold ({fused_score:.3f} > {threshold:.3f}).")
        if drift_result.get("is_drift"):
            drifted = drift_result.get("drifted_features", [])
            reasons.append(
                f"Concept drift detected (ratio={drift_result.get('drift_ratio', 0.0):.2f}); "
                f"top features: {', '.join(drifted[:3]) or 'N/A'}."
            )
        if feature_df is not None and not feature_df.empty:
            recent = feature_df.tail(min(30, len(feature_df)))
            rolling_mean = recent.mean()
            rolling_std = recent.std().replace(0, np.nan)
            latest = recent.iloc[-1]
            z = ((latest - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).dropna()
            if not z.empty and float(z.abs().max()) > 3.0:
                reasons.append(f"Sudden spike on feature '{z.abs().idxmax()}' (|z|>3).")
        if model_scores:
            top = max(model_scores.items(), key=lambda x: x[1])[0]
            reasons.append(f"Dominant contributing model: {top}.")
        if not reasons:
            reasons.append("No strong root-cause rule matched; continue observation.")
        return reasons
