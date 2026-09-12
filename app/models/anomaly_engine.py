"""Local, offline anomaly detection engine.

Clean modular port of the working detectors from the legacy monolith.
Supports scikit-learn detectors plus lightweight statistical methods.
Advanced/unavailable model names gracefully fall back to a robust backend.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger("STDMS.Anomaly")


# Display name -> internal key
MODEL_DISPLAY_TO_INTERNAL = {
    "Isolation Forest": "isolation_forest",
    "Local Outlier Factor": "lof",
    "One-Class SVM": "ocsvm",
    "Random Forest": "random_forest",
    "Z-Score": "z_score",
    "IQR (Interquartile Range)": "iqr",
    "Prophet": "prophet",
    "LSTM": "lstm",
    "Autoencoder": "autoencoder",
}

# Internal key -> executable runtime backend (fallback for unimplemented)
RUNTIME_FALLBACK = {
    "prophet": "z_score",
    "lstm": "isolation_forest",
    "autoencoder": "isolation_forest",
}

SUPPORTED_BACKENDS = {"isolation_forest", "lof", "ocsvm", "random_forest", "z_score", "iqr"}


def to_internal(display_name: str) -> str:
    if display_name in MODEL_DISPLAY_TO_INTERNAL:
        return MODEL_DISPLAY_TO_INTERNAL[display_name]
    return str(display_name).strip().lower().replace(" ", "_")


def resolve_backend(internal: str) -> str:
    backend = RUNTIME_FALLBACK.get(internal, internal)
    if backend not in SUPPORTED_BACKENDS:
        return "isolation_forest"
    return backend


class AnomalyModel:
    """Single anomaly detector with a uniform train/predict interface."""

    def __init__(self, model_type: str = "Isolation Forest", params: dict[str, Any] | None = None):
        self.model_type = model_type
        self.internal_type = to_internal(model_type)
        self.backend = resolve_backend(self.internal_type)
        self.params = params or {}
        self.model: Any = None
        self.scaler: StandardScaler | None = None
        self.feature_columns: list[str] | None = None
        self.iqr_bounds: dict[str, dict[str, float]] = {}
        self.z_stats: dict[str, dict[str, float]] = {}
        self.z_threshold: float = 3.0
        self.contamination: float = float(self.params.get("contamination", 0.1))
        self.trained: bool = False

    # ---- feature handling -------------------------------------------------
    def _numeric_frame(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        numeric = data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if fit:
            self.feature_columns = list(numeric.columns)
            return numeric
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in numeric.columns:
                    numeric[col] = 0.0
            numeric = numeric[self.feature_columns]
        return numeric

    # ---- training ---------------------------------------------------------
    def train(self, data: pd.DataFrame) -> tuple[bool, str]:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return False, "Training data is empty."

        numeric = self._numeric_frame(data, fit=True)
        if numeric.empty:
            return False, "No numeric columns found for training."

        backend = self.backend
        try:
            if backend == "iqr":
                return self._train_iqr(numeric)
            if backend == "z_score":
                return self._train_zscore(numeric)

            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(numeric)

            if backend == "isolation_forest":
                self.model = IsolationForest(
                    n_estimators=int(self.params.get("n_estimators", 100)),
                    contamination=self.contamination,
                    random_state=42,
                )
                self.model.fit(X)
            elif backend == "lof":
                self.model = LocalOutlierFactor(
                    n_neighbors=int(self.params.get("n_neighbors", 20)),
                    contamination=self.contamination,
                    novelty=True,
                )
                self.model.fit(X)
            elif backend == "ocsvm":
                self.model = OneClassSVM(
                    nu=float(self.params.get("nu", 0.05)),
                    kernel=self.params.get("kernel", "rbf"),
                    gamma=self.params.get("gamma", "scale"),
                )
                self.model.fit(X)
            elif backend == "random_forest":
                rng = np.random.default_rng(42)
                n_synth = max(8, int(len(X) * max(self.contamination, 0.05)))
                mean = np.mean(X, axis=0)
                std = np.where(np.std(X, axis=0) < 1e-8, 1.0, np.std(X, axis=0))
                signs = rng.choice(np.array([-1.0, 1.0]), size=(n_synth, X.shape[1]))
                synth = mean + signs * (8.0 + rng.random((n_synth, X.shape[1])) * 4.0) * std
                X_aug = np.vstack([X, synth])
                y = np.concatenate([np.zeros(len(X), dtype=int), np.ones(n_synth, dtype=int)])
                self.model = RandomForestClassifier(
                    n_estimators=int(self.params.get("n_estimators", 200)),
                    random_state=42,
                    class_weight="balanced_subsample",
                )
                self.model.fit(X_aug, y)
                train_proba = self.model.predict_proba(X)
                train_scores = train_proba[:, 1] if train_proba.shape[1] > 1 else train_proba[:, 0]
                self.score_threshold = float(max(0.5, np.percentile(train_scores, (1 - self.contamination) * 100)))
            else:
                return False, f"Unsupported backend: {backend}"

            self.trained = True
            return True, f"{self.model_type} trained on {len(numeric):,} rows × {len(numeric.columns)} features."
        except Exception as exc:
            logger.exception("Training failed for %s", self.model_type)
            return False, f"Training error: {exc}"

    def _train_iqr(self, numeric: pd.DataFrame) -> tuple[bool, str]:
        factor = float(self.params.get("iqr_factor", 1.5))
        self.iqr_bounds = {}
        for col in numeric.columns:
            q1 = numeric[col].quantile(0.25)
            q3 = numeric[col].quantile(0.75)
            iqr = q3 - q1
            self.iqr_bounds[col] = {"lower": q1 - factor * iqr, "upper": q3 + factor * iqr}
        self.model = "iqr_trained"
        self.trained = True
        return True, f"IQR model trained (factor={factor})."

    def _train_zscore(self, numeric: pd.DataFrame) -> tuple[bool, str]:
        self.z_threshold = float(self.params.get("threshold", 3.0))
        self.z_stats = {}
        for col in numeric.columns:
            self.z_stats[col] = {"mean": float(numeric[col].mean()), "std": float(numeric[col].std())}
        self.model = "zscore_trained"
        self.trained = True
        return True, f"Z-Score model trained (threshold={self.z_threshold})."

    # ---- prediction -------------------------------------------------------
    def predict(self, data: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.trained or self.model is None:
            return None, None

        backend = self.backend
        try:
            if backend == "iqr":
                return self._predict_iqr(data)
            if backend == "z_score":
                return self._predict_zscore(data)

            numeric = self._numeric_frame(data, fit=False)
            if numeric.empty or self.scaler is None:
                return None, None
            X = self.scaler.transform(numeric)

            if backend in ("isolation_forest", "lof", "ocsvm"):
                scores = self.model.decision_function(X)
                predictions = self.model.predict(X)
                anomalies = predictions == -1
                # invert so higher score = more anomalous
                return -scores, anomalies
            if backend == "random_forest":
                proba = self.model.predict_proba(X)
                clf = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                distance = np.max(np.abs(X), axis=1)
                scores = np.maximum(clf, np.clip(distance / 6.0, 0.0, 1.0))
                threshold = getattr(self, "score_threshold", None)
                if threshold is None:
                    threshold = np.percentile(scores, (1 - self.contamination) * 100)
                return scores, (scores > threshold) | (distance > 3.5)
            return None, None
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", self.model_type, exc)
            return None, None

    def _predict_iqr(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cols = [c for c in (self.feature_columns or []) if c in data.columns]
        pred = data[cols] if cols else data.select_dtypes(include=[np.number])
        scores = np.zeros(len(pred))
        violation_count = np.zeros(len(pred))
        for col in pred.columns:
            if col in self.iqr_bounds:
                b = self.iqr_bounds[col]
                below = np.maximum(0, b["lower"] - pred[col])
                above = np.maximum(0, pred[col] - b["upper"])
                scores += (below + above).to_numpy()
                violation_count += ((pred[col] < b["lower"]) | (pred[col] > b["upper"])).astype(int).to_numpy()
        if len(pred.columns) > 0:
            scores = scores / len(pred.columns)
        threshold = max(1, len(pred.columns) * 0.25)
        return scores, violation_count >= threshold

    def _predict_zscore(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cols = [c for c in (self.feature_columns or []) if c in data.columns]
        pred = data[cols] if cols else data.select_dtypes(include=[np.number])
        z = np.zeros((len(pred), len(pred.columns)))
        exceed = np.zeros(len(pred))
        for i, col in enumerate(pred.columns):
            stat = self.z_stats.get(col)
            if stat and stat["std"] > 0:
                z[:, i] = np.abs((pred[col] - stat["mean"]) / stat["std"])
                exceed += (z[:, i] > self.z_threshold).astype(int)
        scores = np.max(z, axis=1) if z.size else np.zeros(len(pred))
        threshold = max(1, len(pred.columns) * 0.25)
        return scores, exceed >= threshold

    # ---- persistence payload ---------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "internal_type": self.internal_type,
            "backend": self.backend,
            "params": self.params,
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "iqr_bounds": self.iqr_bounds,
            "z_stats": self.z_stats,
            "z_threshold": self.z_threshold,
            "contamination": self.contamination,
            "score_threshold": getattr(self, "score_threshold", None),
            "trained": self.trained,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "AnomalyModel":
        obj = cls(state.get("model_type", "Isolation Forest"), state.get("params"))
        obj.internal_type = state.get("internal_type", obj.internal_type)
        obj.backend = state.get("backend", obj.backend)
        obj.model = state.get("model")
        obj.scaler = state.get("scaler")
        obj.feature_columns = state.get("feature_columns")
        obj.iqr_bounds = state.get("iqr_bounds", {})
        obj.z_stats = state.get("z_stats", {})
        obj.z_threshold = state.get("z_threshold", 3.0)
        obj.contamination = state.get("contamination", 0.1)
        obj.score_threshold = state.get("score_threshold")
        obj.trained = state.get("trained", False)
        return obj
