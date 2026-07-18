from __future__ import annotations

import datetime
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ingestion.folder_loader import (
    apply_dataset_mode,
    load_latest_from_folder,
    numeric_features,
)
from app.models.anomaly_engine import AnomalyModel
from app.models.drift import DriftDetector
from app.models.forecast import estimate_ttf_hours, short_forecast
from app.models.fusion import FusionEngine
from app.models.fsm import apply_threshold_scale, ensure_fsm_fields, resolve_current_mode
from app.models.model_store import ModelStore
from app.monitoring.obs_limits import evaluate_obs_limits

logger = logging.getLogger("STDMS.TabRuntime")


@dataclass
class TabSnapshot:
    tab_id: str
    title: str = ""
    subsystem: str = "—"
    monitoring_active: bool = False
    watch_status: str = "Idle"
    health_state: str = "Idle"
    fusion_score: float | None = None
    last_file: str = "—"
    last_file_rows: int = 0
    dataset_mode: str = "Full Latest File"
    obs_ok: bool = True
    obs_violations: int = 0
    drift: bool = False
    alert_count: int = 0
    trained_models: int = 0
    ttf: float | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tab_id": self.tab_id,
            "title": self.title,
            "subsystem": self.subsystem,
            "monitoring_active": self.monitoring_active,
            "watch_status": self.watch_status,
            "health_state": self.health_state,
            "fusion_score": self.fusion_score,
            "last_file": self.last_file,
            "last_file_rows": self.last_file_rows,
            "dataset_mode": self.dataset_mode,
            "obs_ok": self.obs_ok,
            "obs_violations": self.obs_violations,
            "drift": self.drift,
            "alert_count": self.alert_count,
            "trained_models": self.trained_models,
            "ttf": self.ttf,
            "updated_at": self.updated_at,
        }


class TabRuntimeState:
    """Independent per-tab monitoring pipeline (offline/local)."""

    def __init__(self, tab_id: str, config: dict[str, Any], data_dir: Path | None = None):
        self.tab_id = tab_id
        self.config = ensure_fsm_fields(dict(config or {}))
        self.data_dir = Path(data_dir) if data_dir else Path("data")

        self.dataframe: pd.DataFrame | None = None
        self.watched_path: str | None = None
        self.watched_mtime: float | None = None
        self.monitoring_active = False
        self.watch_status = "Idle"
        self.obs_result: dict[str, Any] = {"ok": True, "violations": [], "mode_normal": []}

        self.models: dict[str, AnomalyModel] = {}
        self.model_store = ModelStore(self.data_dir, tab_id)
        self.drift_detector = DriftDetector()
        self.fusion = FusionEngine()
        self.last_drift_result: dict[str, Any] = {}
        self.last_xai_reasons: list[str] = []
        self.anomaly_events: deque = deque(maxlen=500)
        self.last_fusion_score: float | None = None
        self.last_health_state = "Idle"
        self.last_ttf: float | None = None

        self.snapshot = TabSnapshot(tab_id=tab_id)
        self.snapshot.title = config.get("title", "")
        self.snapshot.subsystem = config.get("subsystem_name") or "—"
        self.snapshot.dataset_mode = config.get("dataset_mode", "Full Latest File")

        self._bootstrap_models()

    # ---- model bootstrap --------------------------------------------------
    def _bootstrap_models(self) -> None:
        try:
            self.models = self.model_store.load_all()
            if self.models:
                logger.info("Tab %s: loaded %d saved model(s)", self.tab_id, len(self.models))
        except Exception as exc:
            logger.warning("Tab %s: model bootstrap failed: %s", self.tab_id, exc)

    # ---- data -------------------------------------------------------------
    def reload_data(self, force: bool = False) -> tuple[bool, str | None]:
        if self.config.get("input_mode", "CSV Polling") != "CSV Polling":
            return False, "Stream mode is not enabled in this phase."

        folder = self.config.get("data_folder", "")
        if not folder:
            return False, "No data folder configured."

        file_type = self.config.get("data_file_type", "CSV")
        ok, message, loaded = load_latest_from_folder(folder, file_type)
        if not ok or loaded is None:
            self.watch_status = "Error"
            return False, message

        if (
            not force
            and self.watched_path == loaded.source_path
            and self.watched_mtime == loaded.source_mtime
            and self.dataframe is not None
        ):
            return True, None

        df = apply_dataset_mode(
            loaded.dataframe,
            self.config.get("dataset_mode", "Full Latest File"),
            int(self.config.get("monitoring_window_rows", 500)),
        )
        window = int(self.config.get("monitoring_window_rows", 500))
        if len(df) > window:
            df = df.tail(window).copy()

        self.dataframe = df
        self.watched_path = loaded.source_path
        self.watched_mtime = loaded.source_mtime
        mode = resolve_current_mode(self.config)
        self.obs_result = evaluate_obs_limits(
            df,
            threshold_scale=mode.threshold_scale,
            mission_mode=mode.name,
        )
        self.watch_status = "OK" if self.obs_result.get("ok", True) else "OBS Violation"
        self._publish_snapshot()
        return True, None

    def _feature_frame(self) -> pd.DataFrame:
        if self.dataframe is None:
            return pd.DataFrame()
        selected = self.config.get("selected_features") or []
        cols = numeric_features(self.dataframe, selected if selected else None)
        return self.dataframe[cols].copy() if cols else pd.DataFrame()

    def feature_names(self) -> list[str]:
        if self.dataframe is None:
            return []
        selected = self.config.get("selected_features") or []
        return numeric_features(self.dataframe, selected if selected else None)

    def get_plot_data(self, feature: str):
        if self.dataframe is None or feature not in self.dataframe.columns:
            return None
        df = self.dataframe
        y = pd.to_numeric(df[feature], errors="coerce")
        for col in df.columns:
            if str(col).lower() in ("time", "timestamp", "datetime", "date", "ds"):
                return pd.to_datetime(df[col], errors="coerce"), y
        return range(len(y)), y

    # ---- training ---------------------------------------------------------
    def train_models(self) -> tuple[bool, str]:
        """Train all configured models on the current data window."""
        ok, err = self.reload_data(force=True)
        if not ok:
            return False, err or "Failed to load data."

        feature_df = self._feature_frame()
        if feature_df.empty:
            return False, "No numeric features available for training."

        max_rows = int(self.config.get("max_training_rows", 100_000))
        train_df = feature_df.tail(max_rows) if len(feature_df) > max_rows else feature_df

        configured = self.config.get("models", [])
        if not configured:
            return False, "No models configured for this tab."

        trained = 0
        messages = []
        new_models: dict[str, AnomalyModel] = {}
        for entry in configured:
            model_type = entry.get("model_type", "Isolation Forest")
            params = entry.get("model_parameters", {}) or {}
            model = AnomalyModel(model_type, params)
            success, msg = model.train(train_df)
            messages.append(f"{model_type}: {msg}")
            if success:
                model_id = self.model_store.save(model)
                new_models[model_id] = model
                trained += 1

        if trained == 0:
            return False, "No models trained. " + " | ".join(messages)

        self.models.update(new_models)
        self._publish_snapshot()
        return True, f"Trained {trained}/{len(configured)} model(s). " + " | ".join(messages)

    def count_trained_models(self) -> int:
        return sum(1 for m in self.models.values() if getattr(m, "trained", False))

    # ---- monitoring cycle -------------------------------------------------
    def run_monitoring_cycle(self) -> dict[str, Any]:
        """Full pipeline: data → OBS → drift → model scores → fusion → health."""
        result: dict[str, Any] = {"ok": False}

        ok, err = self.reload_data(force=False)
        if not ok:
            self.last_health_state = "Data Error"
            self._publish_snapshot()
            result["error"] = err
            return result

        if self.dataframe is None or len(self.dataframe) == 0:
            self.last_health_state = "No Data"
            self._publish_snapshot()
            result["error"] = "No data available"
            return result

        mode = resolve_current_mode(self.config)
        self.obs_result = evaluate_obs_limits(
            self.dataframe,
            threshold_scale=mode.threshold_scale,
            mission_mode=mode.name,
        )

        feature_df = self._feature_frame()
        if feature_df.empty:
            self.last_health_state = "No Numeric Features"
            self._publish_snapshot()
            result["error"] = "No numeric features"
            return result

        # Drift
        drift_result = self.drift_detector.check(feature_df)
        self.last_drift_result = drift_result

        # Model scoring
        trained_models = {mid: m for mid, m in self.models.items() if getattr(m, "trained", False)}
        if not trained_models:
            self.last_health_state = "No Trained Model"
            self.watch_status = "Needs Training"
            self._publish_snapshot()
            result["error"] = "No trained models. Train first."
            return result

        model_scores: dict[str, float] = {}
        anomaly_counts: dict[str, int] = {}
        for model in trained_models.values():
            scores, anomalies = model.predict(self.dataframe)
            if scores is None:
                continue
            score_arr = np.asarray(scores, dtype=float).ravel()
            score_arr = score_arr[np.isfinite(score_arr)]
            if len(score_arr) == 0:
                score_arr = np.array([0.0])
            anomaly_arr = np.asarray(anomalies).ravel() if anomalies is not None else np.array([])
            count = int(np.sum(anomaly_arr)) if anomaly_arr.dtype == bool else int(np.sum(anomaly_arr == -1))
            ratio = count / max(len(self.dataframe), 1)
            norm = float(np.mean(np.abs(score_arr)))
            norm = norm / (norm + 1.0)
            model_scores[model.model_type] = max(norm, ratio)
            anomaly_counts[model.model_type] = count

        if not model_scores:
            self.last_health_state = "Prediction Unavailable"
            self._publish_snapshot()
            result["error"] = "No valid predictions"
            return result

        fused = self.fusion.fuse(model_scores)
        threshold = apply_threshold_scale(
            self.fusion.adaptive_threshold(fused),
            mode.threshold_scale,
            lo=self.fusion.threshold_min,
            hi=self.fusion.threshold_max,
        )
        health_state, is_warning, is_critical = self.fusion.classify(fused, threshold)
        xai = self.fusion.build_xai_reasons(feature_df, fused, threshold, drift_result, model_scores)
        ttf = estimate_ttf_hours(self.fusion.score_history, threshold)

        self.last_fusion_score = fused
        self.last_health_state = health_state
        self.last_xai_reasons = xai
        self.last_ttf = ttf

        if is_warning or is_critical or not self.obs_result.get("ok", True):
            self.anomaly_events.appendleft(
                {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "health_state": health_state,
                    "fused_score": fused,
                    "xai": "; ".join(xai),
                }
            )

        self.watch_status = health_state if self.obs_result.get("ok", True) else "OBS Violation"
        self._publish_snapshot()

        result.update(
            {
                "ok": True,
                "fused_score": fused,
                "threshold": threshold,
                "health_state": health_state,
                "model_scores": model_scores,
                "anomaly_counts": anomaly_counts,
                "drift": drift_result,
                "xai": xai,
                "ttf": ttf,
                "obs": self.obs_result,
            }
        )
        return result

    # ---- snapshot ---------------------------------------------------------
    def set_monitoring(self, active: bool) -> None:
        self.monitoring_active = active
        if not active and self.last_health_state in ("Idle", "Monitoring"):
            self.last_health_state = "Idle"
        self._publish_snapshot()

    def _publish_snapshot(self) -> None:
        self.snapshot.monitoring_active = self.monitoring_active
        self.snapshot.watch_status = self.watch_status
        self.snapshot.health_state = self.last_health_state
        self.snapshot.fusion_score = self.last_fusion_score
        self.snapshot.last_file = self.watched_path or "—"
        self.snapshot.last_file_rows = len(self.dataframe) if self.dataframe is not None else 0
        self.snapshot.obs_ok = bool(self.obs_result.get("ok", True))
        self.snapshot.obs_violations = len(self.obs_result.get("violations", []))
        self.snapshot.drift = bool(self.last_drift_result.get("is_drift", False))
        self.snapshot.alert_count = len(self.anomaly_events)
        self.snapshot.trained_models = self.count_trained_models()
        self.snapshot.ttf = self.last_ttf
        self.snapshot.title = self.config.get("title", "")
        self.snapshot.dataset_mode = self.config.get("dataset_mode", "Full Latest File")
        self.snapshot.updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_snapshot(self) -> dict[str, Any]:
        self._publish_snapshot()
        return self.snapshot.to_dict()
