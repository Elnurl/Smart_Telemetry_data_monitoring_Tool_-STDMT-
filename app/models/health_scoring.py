from __future__ import annotations


def compute_health_score(reconstruction_error: float, threshold: float) -> float:
    safe_threshold = float(threshold) + 1e-6
    score = 1.0 - (float(reconstruction_error) / safe_threshold)
    return max(0.0, min(1.0, score))

