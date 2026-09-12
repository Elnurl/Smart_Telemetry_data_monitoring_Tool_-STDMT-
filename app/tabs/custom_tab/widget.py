"""Custom monitoring tab widget (extracted from monolith)."""

from __future__ import annotations

import datetime
import inspect
import json
import logging
import os
import pickle
import smtplib
import threading
import types
import uuid
from collections import deque
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
    QListWidgetItem,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QTextEdit,
    QSizePolicy,
    QFileDialog,
    QAbstractItemView,
    QGridLayout,
    QFrame,
    QSplitter,
    QLineEdit,
    QDialog,
)

from app.models.model_types import catalog_display_name
from app.models.drift import TabConceptDriftChecker, compute_adaptive_threshold
from app.models.fsm import (
    MissionModeStore,
    apply_threshold_scale,
    ensure_fsm_fields,
    resolve_current_mode,
)
from app.monitoring.obs_limits import evaluate_obs_limits
from app.reports.mission_report import export_mission_report as write_mission_report
from app.tabs.custom_tab.model_ops import (
    autosave_model,
    bootstrap_from_manifest,
    build_loaded_model_entry,
    create_model_instance,
    internal_to_display_type,
    model_types_equivalent,
    load_model_from_file,
    resolve_training_target,
    save_model_to_file,
)
from app.tabs.custom_tab.panels import (
    build_models_page,
    build_nav_shell,
    build_quick_actions_page,
    connect_nav_pages,
)
from app.tabs.custom_tab.panels.analysis_ml import build_analysis_ml_panel
from app.tabs.custom_tab.panels.data_import import build_data_import_panel
from app.tabs.custom_tab.panels.slots import assert_panel_slots_complete
from app.tabs.custom_tab.panels.visualization import build_visualization_panel
from app.tabs.custom_tab.stream_connector import StreamConnector, StreamRecordBuffer
from app.monitoring.async_pipeline import get_async_pipeline
from app.monitoring.pipeline_bridge import (
    ensure_pipeline,
    get_pipeline_bridge,
    make_on_result_callback,
)
from app.tabs.custom_tab.workers import TabTrainWorker

logger = logging.getLogger("STDMS.CustomMonitoringTab")

_CONFIG_KEYS = (
    "DATA_DIR",
    "REPORTS_DIR",
    "SecureAnomalyDetectionTool",
    "DataProcessor",
    "AnomalyDetectionModel",
    "EnhancedAnomalyDetectionModel",
    "MplCanvas",
    "format_registry_error",
    "to_internal_model_type",
    "AddModelDialog",
    "TabConfigurationDialog",
    "safe_pickle_load",
    "build_legacy_panel_slots",
)
_CONFIGURED = False

# Legacy monolith bindings (set via configure_custom_tab before use).
DATA_DIR = None
REPORTS_DIR = None
SecureAnomalyDetectionTool = None
DataProcessor = None
AnomalyDetectionModel = None
EnhancedAnomalyDetectionModel = None
MplCanvas = None
format_registry_error = None
to_internal_model_type = None
AddModelDialog = None
TabConfigurationDialog = None
safe_pickle_load = None
build_legacy_panel_slots = None


def _missing_config_keys() -> list[str]:
    g = globals()
    return [key for key in _CONFIG_KEYS if g.get(key) is None]


def _assert_custom_tab_configured() -> None:
    missing = _missing_config_keys()
    if missing or not _CONFIGURED:
        detail = ", ".join(missing) if missing else "configure_custom_tab() was not called"
        raise RuntimeError(f"call configure_custom_tab() first — missing: {detail}")


def configure_custom_tab(**kwargs) -> None:
    """Inject monolith symbols required by the extracted tab widget.

    A2 guard (``_assert_custom_tab_configured``, incomplete-config RuntimeError) is live;
    covered by ``test_custom_tab_configure.py``. ``build_legacy_panel_slots`` factory
    required for legacy Data Import / Analysis / Visualization panels (B6 Phase 3).
    """
    global _CONFIGURED
    unknown = set(kwargs) - set(_CONFIG_KEYS)
    if unknown:
        raise TypeError(f"Unknown custom tab dependency: {', '.join(sorted(unknown))}")

    g = globals()
    for key, value in kwargs.items():
        g[key] = value

    missing = _missing_config_keys()
    if missing:
        _CONFIGURED = False
        raise RuntimeError(f"configure_custom_tab() incomplete — missing: {', '.join(missing)}")

    _CONFIGURED = True


class CustomMonitoringTab(QWidget):
    """Custom monitoring tab widget with independent monitoring capability"""

    # Keep tab-native implementations; legacy panel handlers are invoked via _invoke_tab_*_method.
    _TAB_NATIVE_METHODS = frozenset({
        "train_model", "save_model", "load_model",
        "show_recent_models", "show_model_metrics", "update_data_preview",
    })
    
    def __init__(self, tab_id, config, parent=None):
        _assert_custom_tab_configured()
        super().__init__(parent)
        self.tab_id = tab_id
        self.config = ensure_fsm_fields(dict(config or {}))
        self.parent_window = parent
        self.data_processor = None
        self.model = None  # Keep for backward compatibility
        self.models = {}  # Dictionary to store multiple models: {model_id: model_object}
        # Legacy QTimer kept for compatibility; Continuous/Scheduled use AsyncMonitoringPipeline.
        self.monitoring_timer = QTimer(self)
        self.monitoring_active = False
        self._monitoring_worker = None  # deprecated: monitoring uses async pipeline
        self._monitoring_cycle_pending = False
        self._pipeline_registered = False
        self._train_worker = None
        self._setup_pipeline_bridge()
        # Per-tab independent pipeline state
        self.reference_window_size = 200
        self.score_history = []
        self.score_history_maxlen = 500
        self.base_threshold = 0.6
        self.threshold_sigma = 3.0
        self.threshold_min = 0.35
        self.threshold_max = 0.995
        self.ks_pvalue_threshold = 0.01
        self.psi_threshold = 0.2
        self.drift_feature_ratio_threshold = 0.3
        self._drift_checker = TabConceptDriftChecker(
            reference_window_size=self.reference_window_size,
            ks_pvalue_threshold=self.ks_pvalue_threshold,
            psi_threshold=self.psi_threshold,
            feature_ratio_threshold=self.drift_feature_ratio_threshold,
        )
        self.last_drift_result = {}
        self.last_xai_reasons = []
        self.last_retrain_signal_id = None
        self.health_trend_history = deque(maxlen=300)
        self._stream_buffer = StreamRecordBuffer(maxlen=2000)
        self._stream_connector = StreamConnector(self._stream_buffer)
        self.last_scheduled_utc_key = None
        self._watched_data_path = None
        self._watched_data_mtime = None
        self._watched_csv_path = None  # backward compatibility
        self._watched_csv_mtime = None
        self._watched_csv_row_count = 0
        self.anomaly_events = deque(maxlen=500)
        self.last_snapshot = {}
        self.last_obs_result = {"ok": True, "violations": [], "mode_normal": []}
        self.last_mllm_summary = ""
        self.last_mllm_analysis = {}
        self._fsm_store = MissionModeStore()
        try:
            self._fsm_store.sync_tab_modes(self.tab_id, self.config.get("mission_modes") or [])
        except Exception:
            pass
        self.initUI()
        self._bootstrap_saved_models()
        self._publish_snapshot(health_state="Idle")

    def __getattr__(self, name):
        """Lazy-bind legacy SecureAnomalyDetectionTool panel methods onto custom tabs."""
        if name.startswith("__") or name in self._TAB_NATIVE_METHODS:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        parent = object.__getattribute__(self, "parent_window")
        tool_method = getattr(SecureAnomalyDetectionTool, name, None)
        if tool_method is None or not callable(tool_method):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        parent._ensure_tab_tool_context(self)
        bound = types.MethodType(tool_method, self)
        object.__setattr__(self, name, bound)
        return bound

    def _tab_models_dir(self):
        path = os.path.join(DATA_DIR, "custom_tabs", self.tab_id, "models")
        os.makedirs(path, exist_ok=True)
        return path

    def _get_max_training_rows(self):
        return max(1000, int(self.config.get("max_training_rows", 100000)))

    def _get_monitoring_window_rows(self):
        return max(50, int(self.config.get("monitoring_window_rows", 500)))

    def _ensure_data_processor(self):
        if not self.data_processor:
            self.data_processor = DataProcessor()
        return self.data_processor

    def _preprocess_loaded_data(self):
        dp = self._ensure_data_processor()
        if dp.data is None or len(dp.data) == 0:
            return False, "No data loaded"
        ts_col = dp.timestamp_column
        if not ts_col:
            for col in dp.data.columns:
                if str(col).lower() in ("time", "timestamp", "datetime", "date", "ds"):
                    ts_col = col
                    dp.timestamp_column = col
                    break
        ok, msg = dp.preprocess_data(timestamp_col=ts_col, normalize=False, remove_outliers=False)
        if ok:
            logger.info(
                f"Tab {self.config.get('title', self.tab_id)}: preprocessed "
                f"{len(dp.preprocessed_data) if dp.preprocessed_data is not None else len(dp.data):,} rows, "
                f"{len(dp.feature_columns)} features"
            )
        return ok, msg

    def _prepare_training_data(self, data):
        """Cap training size so large CSV tabs remain operable."""
        if data is None or len(data) == 0:
            return data
        max_rows = self._get_max_training_rows()
        if len(data) <= max_rows:
            return data
        recent_n = max_rows // 2
        sample_n = max_rows - recent_n
        recent = data.tail(recent_n)
        sample = data.sample(n=min(sample_n, len(data)), random_state=42)
        combined = pd.concat([sample, recent]).drop_duplicates()
        if len(combined) > max_rows:
            combined = combined.tail(max_rows)
        logger.info(
            f"Tab {self.config.get('title', self.tab_id)}: training on "
            f"{len(combined):,}/{len(data):,} rows (max_training_rows={max_rows:,})"
        )
        return combined

    def _monitoring_window(self, data):
        if data is None or len(data) == 0:
            return data
        window = self._get_monitoring_window_rows()
        if len(data) <= window:
            return data.copy()
        logger.info(
            f"Tab {self.config.get('title', self.tab_id)}: monitoring window "
            f"{window:,}/{len(data):,} latest rows"
        )
        return data.tail(window).copy()

    def _reload_data_if_needed(self, force=False):
        """Auto-fetch latest CSV or JSON file from configured folder."""
        data_folder = Path(self.config.get("data_folder", ""))
        if not data_folder.exists():
            return False, f"Data folder does not exist: {data_folder}"

        file_type = self.config.get("data_file_type", "CSV")
        pattern = "*.csv" if file_type == "CSV" else "*.json"
        data_files = list(data_folder.glob(pattern))
        if not data_files:
            return False, f"No {file_type} files found in: {data_folder}"

        # Prefer agent-pinned source_file when present
        pinned = str(self.config.get("source_file") or "").strip()
        if pinned:
            pin_path = Path(pinned)
            if pin_path.is_file():
                latest_file = pin_path
            else:
                latest_file = max(data_files, key=os.path.getmtime)
        else:
            latest_file = max(data_files, key=os.path.getmtime)
        latest_path = str(latest_file)
        latest_mtime = os.path.getmtime(latest_file)

        unchanged = (
            not force
            and self._watched_data_path == latest_path
            and self._watched_data_mtime == latest_mtime
            and self.data_processor is not None
            and self.data_processor.data is not None
        )
        if unchanged:
            return True, None

        dp = self._ensure_data_processor()
        if file_type == "JSON":
            success, message = dp.load_json(latest_path)
        else:
            success, message = dp.load_csv(latest_path)
        if not success:
            return False, message

        ok, prep_msg = self._preprocess_loaded_data()
        if not ok:
            return False, prep_msg

        self._watched_data_path = latest_path
        self._watched_data_mtime = latest_mtime
        self._watched_csv_path = latest_path
        self._watched_csv_mtime = latest_mtime
        self._watched_csv_row_count = len(dp.data)
        return True, None

    def _reload_csv_if_needed(self, force=False):
        """Backward-compatible alias."""
        return self._reload_data_if_needed(force=force)

    def _timestamp_column(self, df):
        ts_col = getattr(self.data_processor, "timestamp_column", None) if self.data_processor else None
        if ts_col and ts_col in df.columns:
            return ts_col
        for col in df.columns:
            if str(col).lower() in ("time", "timestamp", "datetime", "date", "ds"):
                return col
        return None

    def _apply_dataset_mode(self, data):
        """Filter dataset by daily / weekly / monthly window when configured."""
        if data is None or len(data) == 0:
            return data
        mode = self.config.get("dataset_mode", "Full Latest File")
        if mode == "Full Latest File":
            return data
        ts_col = self._timestamp_column(data)
        if not ts_col:
            return data
        ts = pd.to_datetime(data[ts_col], errors="coerce")
        if not ts.notna().any():
            return data
        anchor = ts.max()
        if mode == "Daily":
            start = anchor - pd.Timedelta(days=1)
        elif mode == "Weekly":
            start = anchor - pd.Timedelta(days=7)
        elif mode == "Monthly":
            start = anchor - pd.Timedelta(days=30)
        else:
            return data
        filtered = data.loc[ts >= start].copy()
        return filtered if len(filtered) > 0 else data.tail(self._get_monitoring_window_rows()).copy()

    def _current_mission_mode(self):
        return resolve_current_mode(self.config)

    def _check_obs_limits(self, data):
        """Operational bounds check using rule_config.json + mission-mode scaling."""
        mode = self._current_mission_mode()
        self.last_obs_result = evaluate_obs_limits(
            data,
            threshold_scale=mode.threshold_scale,
            mission_mode=mode.name,
        )
        return self.last_obs_result

    def set_mission_mode(self, mode_name: str, *, trigger_source: str = "manual") -> None:
        """Switch active mission mode, persist config, and record FSM history."""
        ensure_fsm_fields(self.config)
        mode_name = str(mode_name or "nominal").strip().lower()
        names = {m.get("name") for m in self.config.get("mission_modes") or []}
        if mode_name not in names:
            mode_name = "nominal" if "nominal" in names else next(iter(names), "nominal")
        prev = str(self.config.get("current_mission_mode") or "nominal")
        self.config["current_mission_mode"] = mode_name
        mode = self._current_mission_mode()
        try:
            self._fsm_store.sync_tab_modes(self.tab_id, self.config.get("mission_modes") or [])
            if prev != mode_name:
                self._fsm_store.set_mode(
                    self.tab_id,
                    mode_name,
                    threshold_scale=mode.threshold_scale,
                    trigger_source=trigger_source,
                )
        except Exception as exc:
            logger.warning("FSM mode persist failed for %s: %s", self.tab_id, exc)
        if hasattr(self.parent_window, "tab_config_manager"):
            self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)
        if hasattr(self, "mission_mode_combo") and self.mission_mode_combo.currentText() != mode_name:
            idx = self.mission_mode_combo.findText(mode_name)
            if idx >= 0:
                self.mission_mode_combo.blockSignals(True)
                self.mission_mode_combo.setCurrentIndex(idx)
                self.mission_mode_combo.blockSignals(False)
        if hasattr(self, "header_mission_mode_label"):
            self.header_mission_mode_label.setText(
                f"{mode.name} (×{mode.threshold_scale:g})"
            )
        self._publish_snapshot()

    def _on_mission_mode_changed(self, mode_name: str) -> None:
        self.set_mission_mode(mode_name, trigger_source="manual")

    def browse_log_file(self):
        """Browse for a mission/event .log or .txt file (Data Import)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Log File",
            "",
            "Log files (*.log *.txt);;All files (*.*)",
        )
        if not path:
            return
        if hasattr(self, "log_path_input"):
            self.log_path_input.setText(path)
        self.config["log_file"] = path
        if hasattr(self.parent_window, "tab_config_manager"):
            self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)

    def load_log_file(self):
        """Parse selected log file into LogMonitor (tab_log_events) and preview."""
        path = ""
        if hasattr(self, "log_path_input"):
            path = self.log_path_input.text().strip()
        if not path:
            path = str(self.config.get("log_file") or "").strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Load Logs", "Please select a valid .log or .txt file.")
            return
        try:
            from app.agent.log_monitor import get_log_monitor

            monitor = get_log_monitor()
            count = monitor.ingest_file(self.tab_id, path)
            self.config["log_file"] = path
            if hasattr(self.parent_window, "tab_config_manager"):
                self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)
            # Immediate rule-based analysis (LLM optional on next agent cycle)
            analysis = monitor.analyze(
                self.tab_id,
                self._fsm_store,
                tab_config=self.config,
                apply_transitions=False,
                use_llm=False,
            )
            self.last_mllm_summary = analysis.summary
            self.last_mllm_analysis = analysis.to_dict()
            if analysis.mode_transitions:
                to_mode = analysis.mode_transitions[-1].get("to_mode")
                if to_mode:
                    self.set_mission_mode(str(to_mode), trigger_source="log_event")
            else:
                self._publish_snapshot()
            self._refresh_log_preview(monitor.get_window(self.tab_id))
            if hasattr(self, "log_status_label"):
                self.log_status_label.setText(
                    f"Logs: loaded {count} event(s). {analysis.summary}"
                )
            QMessageBox.information(
                self,
                "Load Logs",
                f"Ingested {count} log event(s).\n\n{analysis.summary}",
            )
        except Exception as exc:
            logger.exception("load_log_file failed")
            QMessageBox.warning(self, "Load Logs", f"Failed to load logs:\n{exc}")

    def _refresh_log_preview(self, events) -> None:
        if not hasattr(self, "log_preview_table"):
            return
        rows = list(events or [])[-50:]
        self.log_preview_table.setRowCount(len(rows))
        for i, ev in enumerate(rows):
            ts = ev.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ev, "timestamp") else ""
            self.log_preview_table.setItem(i, 0, QTableWidgetItem(ts))
            self.log_preview_table.setItem(i, 1, QTableWidgetItem(str(getattr(ev, "level", ""))))
            self.log_preview_table.setItem(i, 2, QTableWidgetItem(str(getattr(ev, "source", ""))))
            self.log_preview_table.setItem(i, 3, QTableWidgetItem(str(getattr(ev, "message", ""))))

    def get_snapshot(self):
        return dict(self.last_snapshot or {})

    def _publish_snapshot(self, **fields):
        mode = self._current_mission_mode()
        snap = {
            "tab_id": self.tab_id,
            "title": self.config.get("title", ""),
            "subsystem": self.config.get("subsystem_name") or "—",
            "monitoring_active": bool(self.monitoring_active),
            "watch_status": self.status_label.text() if hasattr(self, "status_label") else "",
            "health_state": fields.get("health_state", self.last_snapshot.get("health_state", "Idle")),
            "fusion_score": fields.get("fusion_score", self.last_snapshot.get("fusion_score")),
            "last_file": self._watched_data_path or "—",
            "last_file_rows": self._watched_csv_row_count,
            "dataset_mode": self.config.get("dataset_mode", "Full Latest File"),
            "mission_mode": mode.name,
            "threshold_scale": mode.threshold_scale,
            "obs_ok": self.last_obs_result.get("ok", True),
            "obs_violations": len(self.last_obs_result.get("violations", [])),
            "obs_mode_normal": len(self.last_obs_result.get("mode_normal", [])),
            "mllm_summary": self.last_mllm_summary or "",
            "mllm_anomalies": len((self.last_mllm_analysis or {}).get("anomalies") or []),
            "mllm_filtered": int((self.last_mllm_analysis or {}).get("false_positives_filtered") or 0),
            "drift": bool(self.last_drift_result.get("is_drift", False)),
            "alert_count": len(self.anomaly_events),
            "trained_models": self._count_trained_models(),
            "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.last_snapshot = snap
        if hasattr(self, "snapshot_health_label"):
            self.snapshot_health_label.setText(str(snap["health_state"]))
        if hasattr(self, "snapshot_fusion_label"):
            fs = snap.get("fusion_score")
            self.snapshot_fusion_label.setText("—" if fs is None else f"{float(fs):.4f}")
        if hasattr(self, "snapshot_file_label"):
            self.snapshot_file_label.setText(str(snap["last_file"]))
        if hasattr(self, "snapshot_watch_label"):
            self.snapshot_watch_label.setText(str(snap["watch_status"]))
        if hasattr(self, "snapshot_obs_label"):
            if snap["obs_ok"]:
                mn = snap.get("obs_mode_normal") or 0
                self.snapshot_obs_label.setText(
                    "OK" if not mn else f"OK ({mn} mode-normal)"
                )
            else:
                self.snapshot_obs_label.setText(f"{snap['obs_violations']} violation(s)")
        if hasattr(self, "header_mission_mode_label"):
            self.header_mission_mode_label.setText(
                f"{snap.get('mission_mode', 'nominal')} (×{float(snap.get('threshold_scale', 1.0)):g})"
            )
        if hasattr(self, "snapshot_mllm_label"):
            summary = (snap.get("mllm_summary") or "").strip()
            self.snapshot_mllm_label.setText(summary if summary else "—")
        parent = self.parent_window
        if parent and hasattr(parent, "refresh_fleet_dashboard"):
            parent.refresh_fleet_dashboard()

    def _run_short_forecast(self, series_values, steps=5):
        """Local offline forecast using recent score trend."""
        y = np.asarray(series_values, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 5:
            return None
        x = np.arange(len(y), dtype=float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            future_x = np.arange(len(y), len(y) + steps, dtype=float)
            forecast = slope * future_x + intercept
            return float(np.clip(forecast[-1], 0.0, 1.0))
        except Exception:
            return None

    def export_mission_report(self):
        """Export HTML mission report for this tab (offline/local)."""
        return write_mission_report(
            Path(REPORTS_DIR),
            self.config.get("title", "tab"),
            self.get_snapshot(),
            list(self.anomaly_events),
        )

    def _normalize_stream_batch(self, df):
        if df is None or df.empty:
            return None, "Empty stream batch"
        dp = self._ensure_data_processor()
        dp.data = df.copy()
        dp.timestamp_column = None
        for col in df.columns:
            if str(col).lower() in ("time", "timestamp", "datetime", "date", "ds"):
                dp.timestamp_column = col
                break
        ok, msg = dp.preprocess_data(timestamp_col=dp.timestamp_column, normalize=False, remove_outliers=False)
        if ok:
            return dp.preprocessed_data if dp.preprocessed_data is not None else dp.data, None
        return df.copy(), None

    def _autosave_model(self, model_id, model_obj, model_type):
        try:
            autosave_model(self._tab_models_dir(), model_id, model_obj, model_type)
        except Exception as e:
            logger.warning(f"Tab model autosave skipped for {model_id}: {e}")

    def _register_trained_model_for_cross_tab_load(self, model_id, model_type, model_config=None):
        """Register autosaved trained model in global registry for other tabs."""
        try:
            if not hasattr(self.parent_window, "model_registry"):
                return
            filepath = os.path.join(self._tab_models_dir(), f"{model_id}.pkl")
            if not os.path.exists(filepath):
                return
            model_name = (
                f"{self.config.get('title', 'tab')}::{model_type}::"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            metadata = {
                "filepath": filepath,
                "tab_id": self.tab_id,
                "tab_title": self.config.get("title", ""),
                "model_id": model_id,
                "auto_registered_from_training": True,
                "parameters": (model_config or {}).get("model_parameters", {}),
            }
            self.parent_window.model_registry.register_model(model_name, model_type, metadata=metadata)
        except Exception as e:
            logger.warning(f"Cross-tab trained model registry skipped: {e}")

    def _bootstrap_saved_models(self):
        models_list, loaded_models, loaded = bootstrap_from_manifest(
            self._tab_models_dir(),
            self.config.get("models", []),
            anomaly_detection_model_cls=AnomalyDetectionModel,
            enhanced_anomaly_detection_model_cls=EnhancedAnomalyDetectionModel,
            safe_pickle_load=safe_pickle_load,
        )
        if not loaded:
            return

        self.models.update(loaded_models)
        self.config["models"] = models_list
        if hasattr(self.parent_window, "tab_config_manager"):
            self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)
        self.update_models_table()
        if hasattr(self, "results_text"):
            self.results_text.append(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Restored {loaded} saved model(s) for this tab"
            )

    def _count_trained_models(self):
        return sum(1 for model_obj in self.models.values() if model_obj is not None)

    def _record_anomaly_event(self, fused_score, threshold, all_anomaly_counts, xai_reasons, drift_result, health_state):
        event = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "health_state": health_state,
            "fused_score": float(fused_score),
            "threshold": float(threshold),
            "anomalies": dict(all_anomaly_counts),
            "xai": xai_reasons[0] if xai_reasons else "",
            "drift": bool(drift_result.get("is_drift", False)),
        }
        self.anomaly_events.appendleft(event)
        self._update_anomaly_history_table()

    def _update_anomaly_history_table(self):
        if not hasattr(self, "anomaly_history_table"):
            return
        events = list(self.anomaly_events)[:100]
        self.anomaly_history_table.setRowCount(len(events))
        for row_idx, event in enumerate(events):
            self.anomaly_history_table.setItem(row_idx, 0, QTableWidgetItem(event.get("timestamp", "")))
            self.anomaly_history_table.setItem(row_idx, 1, QTableWidgetItem(event.get("health_state", "")))
            self.anomaly_history_table.setItem(row_idx, 2, QTableWidgetItem(f"{event.get('fused_score', 0.0):.4f}"))
            self.anomaly_history_table.setItem(row_idx, 3, QTableWidgetItem(f"{event.get('threshold', 0.0):.4f}"))
            anomaly_total = sum(event.get("anomalies", {}).values())
            self.anomaly_history_table.setItem(row_idx, 4, QTableWidgetItem(str(anomaly_total)))
            self.anomaly_history_table.setItem(row_idx, 5, QTableWidgetItem(event.get("xai", "")))

    def _numeric_feature_frame(self, df):
        """Return numeric feature frame excluding timestamp column."""
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        ts_col = getattr(self.data_processor, "timestamp_column", None) if self.data_processor else None
        if ts_col and ts_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[ts_col], errors='ignore')
        selected = self.config.get("selected_features") or []
        if selected:
            keep = [c for c in selected if c in numeric_df.columns]
            if keep:
                numeric_df = numeric_df[keep]
        return numeric_df

    @property
    def reference_window(self):
        return self._drift_checker.reference_window

    @reference_window.setter
    def reference_window(self, value):
        self._drift_checker.reference_window = value

    def _check_concept_drift(self, feature_df):
        """Window-based drift detection using KS test + PSI."""
        return self._drift_checker.check(
            feature_df,
            auto_rebaseline_on_drift=bool(self.config.get("auto_rebaseline_on_drift", False)),
        )

    def rebaseline_drift_reference(self, feature_df=None):
        """Manually reset drift reference window (e.g. after confirmed maintenance)."""
        source = feature_df
        if source is None and self.data_processor is not None:
            source = self.data_processor.preprocessed_data or self.data_processor.data
        if source is None or source.empty:
            return False
        return self._drift_checker.set_reference(source)

    def _compute_adaptive_threshold(self, latest_score):
        """Rolling median + sigma*std threshold, scaled by mission mode."""
        threshold, self.score_history = compute_adaptive_threshold(
            self.score_history,
            latest_score,
            base_threshold=self.base_threshold,
            threshold_sigma=self.threshold_sigma,
            threshold_min=self.threshold_min,
            threshold_max=self.threshold_max,
            score_history_maxlen=self.score_history_maxlen,
        )
        mode = self._current_mission_mode()
        return apply_threshold_scale(
            threshold,
            mode.threshold_scale,
            lo=self.threshold_min,
            hi=self.threshold_max,
        )

    def _build_xai_reasons(self, feature_df, final_score, threshold, drift_result, agent_scores):
        """Lightweight rule-based explanation builder."""
        reasons = []
        if final_score > threshold:
            reasons.append(f"Fusion score crossed adaptive threshold ({final_score:.3f} > {threshold:.3f}).")

        if drift_result.get("is_drift"):
            drifted = drift_result.get("drifted_features", [])
            reasons.append(
                f"Concept drift detected (ratio={drift_result.get('drift_ratio', 0.0):.2f}); top features: {', '.join(drifted[:3]) or 'N/A'}."
            )

        if feature_df is not None and not feature_df.empty:
            recent = feature_df.tail(min(30, len(feature_df)))
            rolling_mean = recent.mean()
            rolling_std = recent.std().replace(0, np.nan)
            latest = recent.iloc[-1]
            z_scores = ((latest - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).dropna()
            if not z_scores.empty and float(z_scores.abs().max()) > 3.0:
                spike_feature = z_scores.abs().idxmax()
                reasons.append(f"Sudden spike on feature '{spike_feature}' (|z|>{3.0}).")

            variance = recent.var().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            low_var_features = [c for c, v in variance.items() if float(v) < 1e-10]
            if low_var_features:
                reasons.append(f"Variance collapse observed on {len(low_var_features)} feature(s).")

        if agent_scores:
            top_model = max(agent_scores.items(), key=lambda x: x[1])[0]
            reasons.append(f"Dominant contributing model: {top_model}.")

        if not reasons:
            reasons.append("No strong root-cause rule matched; continue observation.")
        return reasons

    def _estimate_ttf_hours(self, warning_threshold, critical_threshold=0.9):
        """Estimate time-to-threshold from score trend (simple linear slope)."""
        if len(self.score_history) < 20:
            return None
        y = np.asarray(self.score_history[-50:], dtype=float)
        x = np.arange(len(y), dtype=float)
        try:
            slope = np.polyfit(x, y, 1)[0]
        except Exception:
            return None
        if slope <= 1e-9:
            return None
        remaining_steps = (critical_threshold - float(y[-1])) / float(slope)
        if remaining_steps <= 0:
            return 0.0
        interval_hours = (self.config.get("interval_ms", 300000) / 1000.0) / 3600.0
        return float(remaining_steps * interval_hours)

    def _update_health_panel(self, health_state=None, fused_score=None, threshold=None, drift_result=None, ttf_hours=None, xai_reasons=None):
        """Update tab-local system health indicators."""
        if health_state is not None and hasattr(self, "health_state_value"):
            self.health_state_value.setText(str(health_state))

        if hasattr(self, "fusion_value_label"):
            if fused_score is None:
                self.fusion_value_label.setText("N/A")
            else:
                self.fusion_value_label.setText(f"{float(fused_score):.4f}")

        if hasattr(self, "threshold_value_label"):
            if threshold is None:
                self.threshold_value_label.setText("N/A")
            else:
                self.threshold_value_label.setText(f"{float(threshold):.4f}")

        if hasattr(self, "drift_value_label"):
            if drift_result is None:
                self.drift_value_label.setText("N/A")
            else:
                is_drift = drift_result.get("is_drift", False)
                drift_score = float(drift_result.get("drift_score", 0.0))
                drift_feats = drift_result.get("drifted_features", [])
                drift_text = (
                    f"{'Detected' if is_drift else 'Stable'} ({drift_score:.2f})"
                    f" — {', '.join(drift_feats[:3]) or 'no shift'}"
                )
                self.drift_value_label.setText(drift_text)

        if hasattr(self, "ttf_value_label"):
            if ttf_hours is None:
                self.ttf_value_label.setText("N/A")
            else:
                self.ttf_value_label.setText(f"{float(ttf_hours):.2f} h")

        if hasattr(self, "xai_value_label"):
            reasons = xai_reasons or []
            self.xai_value_label.setText(reasons[0] if reasons else "N/A")

        if hasattr(self, "forecast_value_label") and len(self.score_history) >= 5:
            fc = self._run_short_forecast(self.score_history)
            if fc is not None:
                self.forecast_value_label.setText(f"{fc:.4f}")
            else:
                self.forecast_value_label.setText("N/A")

    def _update_health_trend_plot(self, fused_score, threshold, drift_score):
        """Append and redraw local health trend chart."""
        self.health_trend_history.append(
            {
                "fused": float(fused_score),
                "threshold": float(threshold),
                "drift": float(drift_score),
            }
        )
        if not hasattr(self, "health_trend_canvas"):
            return

        ax = self.health_trend_canvas.figure.gca()
        ax.clear()
        data = list(self.health_trend_history)
        x = np.arange(len(data))
        fused = [d["fused"] for d in data]
        thr = [d["threshold"] for d in data]
        drift = [d["drift"] for d in data]

        ax.plot(x, fused, label="Fusion", color="#e74c3c", linewidth=1.8)
        ax.plot(x, thr, label="Threshold", color="#3498db", linestyle="--", linewidth=1.6)
        ax.plot(x, drift, label="Drift Score", color="#f39c12", linewidth=1.4)
        ax.set_ylim(0, max(1.0, max(fused + thr + drift) if data else 1.0))
        ax.set_title("Real-Time Health Trend")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        self.health_trend_canvas.draw_idle()

    def _stream_status(self, message: str) -> None:
        if hasattr(self, "results_text"):
            self.results_text.append(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}"
            )

    def _append_stream_record(self, record):
        """Store one incoming stream record (dict)."""
        self._stream_buffer.append(record)

    def _consume_stream_batch(self, max_records=500):
        """Consume pending stream records into DataFrame."""
        return self._stream_buffer.consume_batch(max_records=max_records)

    def _start_stream_connector(self):
        """Start active connector based on tab configuration."""
        return self._stream_connector.start(self.config, status_callback=self._stream_status)

    def _stop_stream_connector(self):
        """Stop active stream connector."""
        self._stream_connector.stop()

    def _make_scrollable_page(self, content_widget):
        """Wrap a page in a vertical scroll area so sections are not squeezed."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        scroll.setWidget(content_widget)
        return scroll

    def initUI(self):
        """Initialize the custom monitoring tab UI with internal navigation."""
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        build_nav_shell(self, root_layout)

        models_list = self.config.get("models", [])
        if not models_list and "model_type" in self.config:
            models_list = [{"model_type": self.config["model_type"]}]

        self.custom_pages.addWidget(
            self._make_scrollable_page(build_quick_actions_page(self, models_list))
        )

        # ---------------- Page 2: Model Development (tab-native train path — see panels/) ----------------
        self.custom_pages.addWidget(self._make_scrollable_page(build_models_page(self)))

        # ---------------- Page 3: Analysis / ML (legacy train path — see panels/__init__.py) ----------------
        page_analysis = QWidget()
        page_analysis_layout = QVBoxLayout(page_analysis)
        page_analysis_layout.setContentsMargins(0, 0, 0, 0)
        slots = self._legacy_panel_slots()
        build_analysis_ml_panel(self, page_analysis_layout, slots=slots)
        data_folder = self.config.get("data_folder", "")
        if data_folder and hasattr(self, "auto_folder_input"):
            self.auto_folder_input.setText(str(data_folder))
        self.custom_pages.addWidget(page_analysis)

        # ---------------- Page 4: Data Import ----------------
        page_data = QWidget()
        page_data_layout = QVBoxLayout(page_data)
        page_data_layout.setContentsMargins(0, 0, 0, 0)
        build_data_import_panel(self, page_data_layout, slots=slots)
        data_folder = self.config.get("data_folder", "")
        file_type = self.config.get("data_file_type", "CSV")
        if data_folder and hasattr(self, "file_path_input") and not self.file_path_input.text():
            self.file_path_input.setText(str(data_folder))
        if hasattr(self, "file_type_combo"):
            idx = self.file_type_combo.findText(str(file_type).upper())
            if idx >= 0:
                self.file_type_combo.setCurrentIndex(idx)
        self.custom_pages.addWidget(page_data)

        # ---------------- Page 5: Visualization ----------------
        page_viz = QWidget()
        page_viz_layout = QVBoxLayout(page_viz)
        page_viz_layout.setContentsMargins(0, 0, 0, 0)
        build_visualization_panel(self, page_viz_layout, slots=slots)
        try:
            if self.data_processor and (
                self.data_processor.data is not None
                or self.data_processor.preprocessed_data is not None
            ):
                slots.invoke_visualization_method(self, "update_visualization_features")
            slots.invoke_visualization_method(self, "update_viz_model_info")
        except Exception:
            pass
        self.custom_pages.addWidget(page_viz)

        connect_nav_pages(self)

    def _legacy_panel_slots(self):
        if self.parent_window is None:
            raise RuntimeError("custom tab has no parent_window for legacy panel slots")
        return assert_panel_slots_complete(build_legacy_panel_slots(self.parent_window))

    def update_models_table(self):
        """Update the models table with current models"""
        models_list = self.config.get('models', [])
        if not models_list and 'model_type' in self.config:
            # Backward compatibility
            models_list = [{'model_type': self.config['model_type'], 
                          'model_parameters': self.config.get('model_parameters', {}),
                          'model_id': str(uuid.uuid4())}]
        
        self.models_table.setRowCount(len(models_list))
        
        for i, model_config in enumerate(models_list):
            model_id = model_config.get('model_id', str(uuid.uuid4()))
            if 'model_id' not in model_config:
                model_config['model_id'] = model_id
            
            # Model Type
            self.models_table.setItem(i, 0, QTableWidgetItem(model_config.get('model_type', 'Unknown')))
            
            # Status
            if model_id in self.models:
                status = "Trained" if self.models[model_id] is not None else "Not Trained"
            else:
                status = "Not Trained"
            self.models_table.setItem(i, 1, QTableWidgetItem(status))
            
            # Parameters (summary)
            params = model_config.get('model_parameters', {})
            params_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:2]])
            if len(params) > 2:
                params_str += "..."
            self.models_table.setItem(i, 2, QTableWidgetItem(params_str or "Default"))
            
            # Actions button
            actions_btn = QPushButton("Select")
            actions_btn.clicked.connect(lambda checked, mid=model_id: self.select_model(mid))
            self.models_table.setCellWidget(i, 3, actions_btn)
            
            # Remove button
            remove_btn = QPushButton("Remove")
            remove_btn.setStyleSheet("background-color: #dc3545; color: white;")
            remove_btn.clicked.connect(lambda checked, mid=model_id: self.remove_model(mid))
            self.models_table.setCellWidget(i, 4, remove_btn)
    
    def select_model(self, model_id):
        """Select a model for operations"""
        self.selected_model_id = model_id
        models_list = self.config.get('models', [])
        model_config = next((m for m in models_list if m.get('model_id') == model_id), None)
        
        if model_config:
            status = "Trained" if model_id in self.models and self.models[model_id] is not None else "Not Trained"
            self.models_selection_label.setText(
                f"Selected: {model_config.get('model_type', 'Unknown')} - Status: {status}"
            )
            self.save_model_btn.setEnabled(model_id in self.models and self.models[model_id] is not None)
            self.model_metrics_btn.setEnabled(model_id in self.models and self.models[model_id] is not None)
    
    def add_model(self):
        """Add a new model to this tab with inline parameter configuration."""
        dialog = AddModelDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        model_type = dialog.model_combo.currentText()
        model_params = dialog.get_model_parameters()

        new_model = {
            'model_type': model_type,
            'model_parameters': model_params,
            'model_id': str(uuid.uuid4())
        }

        models_list = self.config.get('models', [])
        if not models_list and 'model_type' in self.config:
            models_list = [{
                'model_type': self.config['model_type'],
                'model_parameters': self.config.get('model_parameters', {}),
                'model_id': str(uuid.uuid4())
            }]

        models_list.append(new_model)
        self.config['models'] = models_list

        if hasattr(self.parent_window, 'tab_config_manager'):
            self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)

        self.update_models_table()
        QMessageBox.information(
            self, "Model Added",
            f"New model '{model_type}' added with configured parameters."
        )
    
    def remove_model(self, model_id, silent=False):
        """Remove a model from the tab. silent=True skips the confirm dialog (agent Approve path)."""
        if not silent:
            reply = QMessageBox.question(
                self,
                "Remove Model",
                "Are you sure you want to remove this model?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return False

        models_list = self.config.get('models', [])
        models_list = [m for m in models_list if m.get('model_id') != model_id]
        self.config['models'] = models_list
            
        # Remove from active models
        if model_id in self.models:
            del self.models[model_id]
            
        # Save config
        if hasattr(self.parent_window, 'tab_config_manager'):
            self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)
            
        # Update table
        self.update_models_table()
            
        # Clear selection if removed model was selected
        if hasattr(self, 'selected_model_id') and self.selected_model_id == model_id:
            self.models_selection_label.setText("No model selected")
            self.save_model_btn.setEnabled(False)
            self.model_metrics_btn.setEnabled(False)
        return True
    
    def train_selected_model(self):
        """Train the currently selected model"""
        if not hasattr(self, 'selected_model_id'):
            QMessageBox.warning(self, "No Model Selected", "Please select a model from the table first.")
            return
        
        self.train_model(self.selected_model_id)

    def _normalize_model_type_key(self, model_type: str) -> str:
        from app.tabs.custom_tab.model_ops import normalize_model_type_key
        return normalize_model_type_key(model_type, to_internal_model_type)

    def _model_types_equivalent(self, left: str, right: str) -> bool:
        return model_types_equivalent(left, right, to_internal_model_type)

    def _find_model_config_by_type(self, display_model_type: str):
        for model_cfg in self.config.get("models", []):
            if self._model_types_equivalent(model_cfg.get("model_type", ""), display_model_type):
                return model_cfg
        return None

    def _analysis_params_for_type(self, display_model_type: str) -> dict:
        values = getattr(self, "model_param_values", None) or {}
        params = values.get(display_model_type)
        return dict(params) if isinstance(params, dict) else {}

    def _ensure_model_config_for_analysis_train(self, display_model_type: str):
        """Resolve tab config entry; auto-add to Model Development when missing."""
        model_cfg = self._find_model_config_by_type(display_model_type)
        if model_cfg and model_cfg.get("model_id"):
            return model_cfg["model_id"], model_cfg

        models_list = list(self.config.get("models", []))
        if not models_list and self.config.get("model_type"):
            models_list = [{
                "model_type": self.config["model_type"],
                "model_parameters": self.config.get("model_parameters", {}) or {},
                "model_id": str(uuid.uuid4()),
            }]
            self.config["models"] = models_list
            model_cfg = self._find_model_config_by_type(display_model_type)
            if model_cfg and model_cfg.get("model_id"):
                return model_cfg["model_id"], model_cfg

        model_id = str(uuid.uuid4())
        model_cfg = {
            "model_type": display_model_type,
            "model_parameters": self._analysis_params_for_type(display_model_type),
            "model_id": model_id,
        }
        models_list.append(model_cfg)
        self.config["models"] = models_list

        if self.parent_window and hasattr(self.parent_window, "tab_config_manager"):
            self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)

        self.update_models_table()
        self.results_text.append(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"Added '{display_model_type}' to Model Development for training."
        )
        return model_id, model_cfg

    def train_from_analysis_panel(self):
        """Analysis panel Train → canonical tab.models path (B6 Phase 4 redirect)."""
        checked_types = []
        if hasattr(self, "model_list_widget"):
            for i in range(self.model_list_widget.count()):
                item = self.model_list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    stored = item.data(Qt.UserRole)
                    checked_types.append(catalog_display_name(stored or item.text()))

        if checked_types:
            matched = []
            for model_type in checked_types:
                model_id, model_cfg = self._ensure_model_config_for_analysis_train(model_type)
                matched.append((model_id, model_cfg.get("model_type", model_type)))
            silent_batch = len(matched) > 1
            for model_id, model_type in matched:
                self.train_model(model_id, silent=silent_batch)
            return

        self.train_selected_model()

    def train_all_models(self):
        """Train all configured models for this tab."""
        models_list = self.config.get('models', [])
        if not models_list:
            QMessageBox.warning(self, "No Models", "No models configured in this tab.")
            return

        self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting batch training for {len(models_list)} model(s)...")
        trained_count = 0
        failed_count = 0

        data, error = self.load_latest_data(for_monitoring=False, force_reload=True)
        if error or data is None or len(data) == 0:
            QMessageBox.warning(self, "Training Error", error or "No data available for batch training.")
            return

        for model_cfg in models_list:
            model_id = model_cfg.get('model_id')
            if not model_id:
                model_id = str(uuid.uuid4())
                model_cfg['model_id'] = model_id
            try:
                ok = self.train_model(model_id, silent=True, preloaded_data=data, _sync=True)
                if ok and model_id in self.models and self.models[model_id] is not None:
                    trained_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

        self.results_text.append(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Batch training completed: success={trained_count}, failed={failed_count}"
        )

    def _param_schema_for_model(self, model_type):
        """Return editable parameter schema for model type."""
        if model_type in ["Isolation Forest", "Enhanced Isolation Forest"]:
            return [
                ("n_estimators", "n_estimators", "int", 10, 1000, 100),
                ("contamination", "contamination", "float", 0.01, 0.5, 0.1),
            ]
        if model_type in ["Local Outlier Factor"]:
            return [
                ("n_neighbors", "n_neighbors", "int", 5, 100, 20),
                ("contamination", "contamination", "float", 0.01, 0.5, 0.1),
            ]
        if model_type in ["One-Class SVM"]:
            return [("nu", "nu", "float", 0.001, 0.5, 0.05)]
        if model_type in ["LSTM", "GRU", "Autoencoder", "LSTM-AE", "TCN-AE", "Transformer-AE", "VAE", "Mamba"]:
            return [
                ("epochs", "epochs", "int", 1, 500, 50),
                ("batch_size", "batch_size", "int", 8, 256, 32),
                ("sequence_length", "sequence_length", "int", 5, 200, 10),
            ]
        if model_type in ["XGBoost", "XGBoost RUL", "Random Forest", "Random Forest RUL", "DeepHit", "DRSA"]:
            return [
                ("n_estimators", "n_estimators", "int", 10, 1000, 200),
                ("max_depth", "max_depth", "int", 1, 20, 6),
                ("learning_rate", "learning_rate", "float", 0.01, 1.0, 0.1),
            ]
        if model_type in ["IQR (Interquartile Range)"]:
            return [("iqr_factor", "iqr_factor", "float", 0.5, 5.0, 1.5)]
        if model_type in ["Z-Score"]:
            return [("threshold", "threshold", "float", 1.0, 6.0, 3.0)]
        return []

    def edit_selected_model_params(self):
        """Edit parameters for selected model in current tab."""
        if not hasattr(self, 'selected_model_id'):
            QMessageBox.warning(self, "No Model Selected", "Please select a model first.")
            return

        models_list = self.config.get('models', [])
        model_config = next((m for m in models_list if m.get('model_id') == self.selected_model_id), None)
        if not model_config:
            QMessageBox.warning(self, "Model Not Found", "Selected model configuration not found.")
            return

        model_type = model_config.get("model_type", "Unknown")
        current_params = dict(model_config.get("model_parameters", {}))
        schema = self._param_schema_for_model(model_type)
        if not schema:
            QMessageBox.information(self, "No Editable Params", f"No explicit parameter schema for {model_type}.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Parameters - {model_type}")
        dialog.setMinimumWidth(420)
        vbox = QVBoxLayout(dialog)
        form = QFormLayout()
        widgets = {}

        for key, label, ptype, min_v, max_v, default_v in schema:
            if ptype == "int":
                w = QSpinBox()
                w.setRange(int(min_v), int(max_v))
                w.setValue(int(current_params.get(key, default_v)))
            else:
                w = QDoubleSpinBox()
                w.setRange(float(min_v), float(max_v))
                w.setDecimals(4)
                w.setSingleStep(0.01)
                w.setValue(float(current_params.get(key, default_v)))
            form.addRow(label, w)
            widgets[key] = w

        vbox.addLayout(form)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        vbox.addLayout(btn_row)

        if dialog.exec_() == QDialog.Accepted:
            new_params = {}
            for key, w in widgets.items():
                new_params[key] = w.value()
            model_config["model_parameters"] = new_params
            self.config["updated_at"] = datetime.datetime.now().isoformat()
            if hasattr(self.parent_window, "tab_config_manager"):
                self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)
            self.update_models_table()
            self.results_text.append(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Updated parameters for {model_type}: {new_params}"
            )
    
    def save_selected_model(self):
        """Save the currently selected model"""
        if not hasattr(self, 'selected_model_id'):
            QMessageBox.warning(self, "No Model Selected", "Please select a model from the table first.")
            return
        
        model_id = self.selected_model_id
        if model_id not in self.models or self.models[model_id] is None:
            QMessageBox.warning(self, "Save Model", "Selected model is not trained yet.")
            return
        
        self.save_model(model_id)
    
    def load_latest_data(self, for_monitoring=False, force_reload=False):
        """Load CSV/JSON source, preprocess, and return training or monitoring frame."""
        try:
            ok, err = self._reload_data_if_needed(force=force_reload)
            if not ok:
                return None, err

            dp = self._ensure_data_processor()
            full = dp.preprocessed_data if dp.preprocessed_data is not None else dp.data
            if full is None or len(full) == 0:
                return None, "No data available after preprocessing"

            full = self._apply_dataset_mode(full)

            if for_monitoring:
                data = self._monitoring_window(full)
            else:
                data = self._prepare_training_data(full)

            self.update_data_preview()
            return data, None

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, str(e)
    
    def update_data_preview(self):
        """Update the data preview table"""
        if not self.data_processor:
            return
        df = self.data_processor.preprocessed_data if self.data_processor.preprocessed_data is not None else self.data_processor.data
        if df is None:
            return
        
        preview = df.tail(min(100, len(df)))
        self.data_table.setRowCount(len(preview))
        self.data_table.setColumnCount(len(preview.columns))
        self.data_table.setHorizontalHeaderLabels([str(c) for c in preview.columns.tolist()])
        
        for i in range(len(preview)):
            for j, col in enumerate(preview.columns):
                value = str(preview.iloc[i, j])
                self.data_table.setItem(i, j, QTableWidgetItem(value[:50]))

        if hasattr(self, "timestamp_combo"):
            self.timestamp_combo.clear()
            self.timestamp_combo.addItems([str(c) for c in df.columns.tolist()])
            for candidate in ("time", "timestamp", "datetime", "date"):
                idx = self.timestamp_combo.findText(candidate)
                if idx >= 0:
                    self.timestamp_combo.setCurrentIndex(idx)
                    break

        if hasattr(self, "column_stats_label"):
            self.column_stats_label.setText(
                f"Dataset Info: {len(df):,} rows x {len(df.columns)} columns"
            )

        if hasattr(self, "feature_list"):
            self.feature_list.setRowCount(len(df.columns))
            for i, col in enumerate(df.columns):
                self.feature_list.setItem(i, 0, QTableWidgetItem(str(col)))
                use = "Yes" if str(col) == "value" or str(col).startswith("value") else "No"
                self.feature_list.setItem(i, 1, QTableWidgetItem(use))

        if hasattr(self, "feature_combo") and self.parent_window is not None:
            try:
                self._legacy_panel_slots().invoke_visualization_method(
                    self, "update_visualization_features"
                )
            except Exception:
                pass
    
    def _on_export_mission_report(self):
        try:
            path = self.export_mission_report()
            QMessageBox.information(self, "Mission Report", f"Report exported:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def train_model(self, model_id=None, silent=False, preloaded_data=None, _sync=False):
        """Train a model with loaded data (background worker by default)."""
        try:
            if self._train_worker and self._train_worker.isRunning():
                if not silent:
                    QMessageBox.information(self, "Training", "Training is already in progress for this tab.")
                return False

            if preloaded_data is not None:
                data = preloaded_data
                error = None
            else:
                data, error = self.load_latest_data(for_monitoring=False, force_reload=True)
            if error:
                if not silent:
                    QMessageBox.warning(self, "Training Error", f"Cannot train model: {error}")
                self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training error: {error}")
                return False

            if data is None or len(data) == 0:
                if not silent:
                    QMessageBox.warning(self, "Training Error", "No data available for training.")
                return False

            if model_id is None:
                if hasattr(self, 'selected_model_id'):
                    model_id = self.selected_model_id

            try:
                model_id, model_config, model_type, model_params = resolve_training_target(
                    self.config,
                    model_id=model_id,
                )
            except ValueError:
                if not silent:
                    QMessageBox.warning(self, "Training Error", "Model configuration not found.")
                return False

            self.results_text.append(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting training for {model_type}..."
            )

            model_obj, _internal_model_type = create_model_instance(
                model_type,
                model_params,
                to_internal_model_type=to_internal_model_type,
                anomaly_detection_model_cls=AnomalyDetectionModel,
                enhanced_anomaly_detection_model_cls=EnhancedAnomalyDetectionModel,
            )

            if _sync:
                from app.ui.training_console import (
                    TrainingConsoleBridge,
                    append_training_header,
                    capture_stdout,
                    clear_training_console,
                )

                clear_training_console(self)
                append_training_header(self, f"Training {model_type} ({model_id})")
                bridge = TrainingConsoleBridge()
                bridge.chunk.connect(self._on_train_log_chunk)
                with capture_stdout(bridge):
                    success, message = model_obj.train(data, **model_params)
                return self._finalize_train_result(
                    success, message, model_id, model_obj, model_type, model_params, model_config, len(data), silent
                )

            self._pending_train_context = {
                "model_id": model_id,
                "model_type": model_type,
                "model_params": model_params,
                "model_config": model_config,
                "data_len": len(data),
                "silent": silent,
            }
            if hasattr(self, "train_btn"):
                self.train_btn.setEnabled(False)
            self._train_worker = TabTrainWorker(self, model_id, data, model_obj, model_type, model_params)
            self._train_worker.finished.connect(self._on_tab_train_finished)
            self._train_worker.log_chunk.connect(self._on_train_log_chunk)
            from app.ui.training_console import append_training_header, clear_training_console

            clear_training_console(self)
            append_training_header(self, f"Training {model_type} ({model_id})")
            self._train_worker.start()
            return True

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training error: {str(e)}")
            if not silent:
                QMessageBox.critical(self, "Training Error", f"Error during training: {str(e)}")
            if hasattr(self, "train_btn"):
                self.train_btn.setEnabled(True)
            return False

    def _on_train_log_chunk(self, text: str) -> None:
        """Mirror Keras stdout into Training Console (or results_text fallback)."""
        from app.ui.training_console import append_console_chunk

        console = getattr(self, "training_console", None)
        if console is not None:
            append_console_chunk(console, text)
            return
        if hasattr(self, "results_text") and text.strip():
            # Fallback: only append complete lines to avoid flooding with \r updates
            if "\n" in text:
                for line in text.splitlines():
                    if line.strip():
                        self.results_text.append(line)

    def _on_tab_train_finished(self, success, message, model_id, model_obj):
        self._train_worker = None
        if hasattr(self, "train_btn"):
            self.train_btn.setEnabled(True)
        ctx = getattr(self, "_pending_train_context", {}) or {}
        self._finalize_train_result(
            success,
            message,
            model_id,
            model_obj,
            ctx.get("model_type", "Unknown"),
            ctx.get("model_params", {}),
            ctx.get("model_config"),
            ctx.get("data_len", 0),
            ctx.get("silent", False),
        )
        self._pending_train_context = None

    def _finalize_train_result(self, success, message, model_id, model_obj, model_type, model_params, model_config, data_len, silent):
        if success and model_obj is not None:
            self.models[model_id] = model_obj
            self.model = model_obj
            self._autosave_model(model_id, model_obj, model_type)
            self._register_trained_model_for_cross_tab_load(model_id, model_type, model_config=model_config)

            if hasattr(self, 'selected_model_id') and self.selected_model_id == model_id:
                self.models_selection_label.setText(f"Model: {model_type} (Trained)\nParameters: {model_params}")
                self.save_model_btn.setEnabled(True)
                self.model_metrics_btn.setEnabled(True)

            self.update_models_table()
            try:
                self._legacy_panel_slots().invoke_visualization_method(
                    self, "update_viz_model_info"
                )
            except Exception:
                pass
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model type: {model_type}")
            self.results_text.append(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training completed successfully ({data_len:,} rows)"
            )
            if not silent:
                QMessageBox.information(self, "Training Complete", f"Model training completed successfully for {model_type}!")
            # Agent Approve chain: enqueue start_monitoring only after a real train success
            hook = getattr(self, "_agent_post_train_hook", None)
            if callable(hook):
                self._agent_post_train_hook = None
                try:
                    hook()
                except Exception as exc:
                    logger.warning("agent post-train hook failed: %s", exc)
            return True

        self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training failed: {message}")
        if not silent:
            QMessageBox.warning(self, "Training Error", f"Training failed: {message}")
        if getattr(self, "_agent_post_train_hook", None) is not None:
            self._agent_post_train_hook = None
        return False

    def test_models_on_latest_window(self):
        """Run all trained models on the latest monitoring window (real inference test)."""
        if self._count_trained_models() == 0:
            QMessageBox.warning(self, "No Trained Models", "Train or load at least one model first.")
            return

        active_device = bool(self.config.get("active_device_monitoring_enabled", False))
        if not active_device:
            data, error = self.load_latest_data(for_monitoring=True, force_reload=True)
        else:
            raw, error = self._consume_stream_batch()
            data, error = self._normalize_stream_batch(raw) if raw is not None else (None, error)
            if data is not None:
                data = self._monitoring_window(data)

        if error or data is None or len(data) == 0:
            QMessageBox.warning(self, "Test Error", error or "No data available for testing.")
            return

        log = self.results_text
        log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Testing {self._count_trained_models()} model(s) on {len(data):,} rows...")

        summary = []
        for model_id, model_obj in self.models.items():
            if model_obj is None:
                continue
            models_list = self.config.get("models", [])
            model_config = next((m for m in models_list if m.get("model_id") == model_id), None)
            model_type = model_config.get("model_type", "Unknown") if model_config else "Unknown"
            try:
                scores, anomalies = model_obj.predict(data)
                anomaly_arr = np.asarray(anomalies).ravel() if anomalies is not None else np.array([], dtype=bool)
                if anomaly_arr.dtype != bool and anomaly_arr.size > 0:
                    anomaly_count = int(np.sum(anomaly_arr == -1))
                else:
                    anomaly_count = int(np.sum(anomaly_arr)) if anomaly_arr.size > 0 else 0
                score_mean = float(np.nanmean(np.abs(np.asarray(scores, dtype=float)))) if scores is not None else 0.0
                line = f"{model_type}: {anomaly_count}/{len(data)} anomalies, mean|score|={score_mean:.4f}"
                summary.append(line)
                log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {line}")
            except Exception as test_err:
                log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {model_type}: test failed -> {test_err}")

        log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model test complete.")
        QMessageBox.information(self, "Model Test Complete", "\n".join(summary) if summary else "No model produced results.")

    def _is_scheduled_due_now(self):
        """Return True when current UTC time matches configured scheduled time."""
        now_utc = datetime.datetime.utcnow()
        target_hour = int(self.config.get("schedule_utc_hour", 0))
        target_minute = int(self.config.get("schedule_utc_minute", 0))

        if now_utc.hour != target_hour or now_utc.minute != target_minute:
            return False

        slot_key = now_utc.strftime("%Y-%m-%d %H:%M")
        if self.last_scheduled_utc_key == slot_key:
            return False

        self.last_scheduled_utc_key = slot_key
        return True
    
    def start_monitoring(self):
        """Start monitoring with configured schedule"""
        if self._count_trained_models() == 0:
            QMessageBox.warning(
                self,
                "No Trained Models",
                "Train or load at least one model in Model Management before starting monitoring."
            )
            return

        schedule_type = self.config['schedule_type']
        input_mode = self.config.get("input_mode", "CSV Polling")
        active_device = bool(self.config.get("active_device_monitoring_enabled", False))

        # Stream connector is needed only for active serial ingestion.
        if active_device and schedule_type in ("Continuous", "Scheduled"):
            ok, stream_error = self._start_stream_connector()
            if not ok:
                QMessageBox.warning(self, "Stream Connector Error", stream_error or "Unable to start stream connector.")
                self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Stream connector error: {stream_error}")
                self._update_health_panel(health_state="Stream Connector Error")
                return
            source_label = "Serial Device"
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Stream connector active: {source_label}")

        if schedule_type == "On-Demand":
            # Manual one-shot run via shared async pipeline (off UI thread).
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] On-Demand run requested")
            self.monitor_data()
            if active_device:
                self._stop_stream_connector()
            self.status_label.setText("Status: On-Demand (manual run completed)")
            self._update_health_panel(health_state="On-Demand")
            self._publish_snapshot(health_state="On-Demand")
            return

        if schedule_type == "Continuous":
            interval_ms = int(self.config.get('interval_ms', 300000))
            if interval_ms <= 0:
                interval_ms = 300000
            self.monitoring_active = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText(f"Status: Monitoring (checking every {interval_ms//1000}s)")
            self._update_health_panel(health_state="Monitoring")
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Continuous monitoring started")
            self._publish_snapshot(health_state="Monitoring")
            self._register_with_pipeline(interval_ms=interval_ms, schedule_type="Continuous")
            return

        if schedule_type == "Scheduled":
            # Pipeline polls every 30s; due_fn gates execution to configured UTC HH:MM.
            self.last_scheduled_utc_key = None
            self.monitoring_active = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            utc_h = int(self.config.get("schedule_utc_hour", 0))
            utc_m = int(self.config.get("schedule_utc_minute", 0))
            self.status_label.setText(f"Status: Scheduled (daily UTC {utc_h:02d}:{utc_m:02d})")
            self._update_health_panel(health_state="Monitoring (Scheduled)")
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Scheduled monitoring armed for UTC {utc_h:02d}:{utc_m:02d}")
            self._publish_snapshot(health_state="Monitoring (Scheduled)")
            self._register_with_pipeline(
                interval_ms=30000,
                schedule_type="Scheduled",
                due_fn=self._is_scheduled_due_now,
                run_immediately=bool(self._is_scheduled_due_now()),
            )
            return

    def _setup_pipeline_bridge(self) -> None:
        bridge = get_pipeline_bridge()
        bridge.set_handler(self.tab_id, self._on_pipeline_cycle_result)

    def _register_with_pipeline(
        self,
        *,
        interval_ms: int,
        schedule_type: str,
        due_fn=None,
        run_immediately: bool = True,
    ) -> None:
        pipe = ensure_pipeline()
        bridge = get_pipeline_bridge()
        bridge.set_handler(self.tab_id, self._on_pipeline_cycle_result)
        pipe.register_tab(
            self.tab_id,
            interval_ms=interval_ms,
            compute_fn=self._compute_monitoring_cycle,
            on_result=make_on_result_callback(bridge),
            schedule_type=schedule_type,
            due_fn=due_fn,
            run_immediately=run_immediately,
        )
        self._pipeline_registered = True
        self.results_text.append(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Async pipeline registered"
        )

    def _unregister_from_pipeline(self) -> None:
        if not self._pipeline_registered and self.tab_id not in get_async_pipeline().active_tab_ids():
            get_pipeline_bridge().clear_handler(self.tab_id)
            return
        try:
            get_async_pipeline().unregister_tab(self.tab_id)
        except Exception:
            pass
        get_pipeline_bridge().clear_handler(self.tab_id)
        self._pipeline_registered = False

    def _on_pipeline_cycle_result(self, result, metrics=None) -> None:
        """GUI-thread apply path for async pipeline cycles."""
        if isinstance(result, dict):
            self._apply_monitoring_cycle_result(result)

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_timer.stop()
        self._monitoring_cycle_pending = False
        self._unregister_from_pipeline()
        if self._monitoring_worker and self._monitoring_worker.isRunning():
            self._monitoring_worker.requestInterruption()
            self._monitoring_worker.wait(5000)
        self._monitoring_worker = None
        self._stop_stream_connector()
        self.monitoring_active = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Inactive")
        self._update_health_panel(health_state="Inactive")
        self._publish_snapshot(health_state="Inactive")
        self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Monitoring stopped")

    def monitor_data(self):
        """Queue a monitoring cycle on the shared async pipeline (keeps UI responsive)."""
        schedule_type = self.config.get("schedule_type", "Continuous")
        if schedule_type == "Scheduled" and self.monitoring_active and not self._is_scheduled_due_now():
            return
        self._submit_monitoring_once(schedule_type=schedule_type)

    def _submit_monitoring_once(self, *, schedule_type: str = "On-Demand") -> None:
        self._monitoring_cycle_pending = False
        self.results_text.append(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Checking for new data..."
        )
        pipe = ensure_pipeline()
        bridge = get_pipeline_bridge()
        bridge.set_handler(self.tab_id, self._on_pipeline_cycle_result)
        pipe.submit_once(
            self.tab_id,
            self._compute_monitoring_cycle,
            on_result=make_on_result_callback(bridge),
            schedule_type=schedule_type,
        )

    def _monitoring_log(self, log_lines, message):
        log_lines.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

    def _compute_monitoring_cycle(self):
        """Heavy monitoring pipeline (runs off the UI thread)."""
        log_lines = []
        try:
            input_mode = self.config.get("input_mode", "CSV Polling")
            active_device = bool(self.config.get("active_device_monitoring_enabled", False))
            schedule_type = self.config.get("schedule_type", "Continuous")

            if schedule_type == "Scheduled" and self.monitoring_active:
                if not self._is_scheduled_due_now():
                    return {"status": "skipped", "log_lines": log_lines}
                self._monitoring_log(log_lines, "Scheduled UTC slot matched. Executing monitoring cycle.")

            if QThread.currentThread().isInterruptionRequested():
                return {"status": "cancelled", "log_lines": log_lines}

            if not active_device and input_mode == "CSV Polling":
                ok, err = self._reload_data_if_needed(force=False)
                if not ok:
                    data, error = None, err
                else:
                    dp = self._ensure_data_processor()
                    full = dp.preprocessed_data if dp.preprocessed_data is not None else dp.data
                    full = self._apply_dataset_mode(full)
                    data = self._monitoring_window(full)
                    error = None
            else:
                raw, error = self._consume_stream_batch()
                data, error = self._normalize_stream_batch(raw) if raw is not None else (None, error)
                if data is not None:
                    data = self._monitoring_window(data)

            if error:
                health_state = "Waiting Stream Data" if "No stream data received yet" in str(error) else "Data Error"
                self._monitoring_log(log_lines, f"Error: {error}")
                return {"status": "data_error", "log_lines": log_lines, "health_state": health_state}

            if data is None or len(data) == 0:
                self._monitoring_log(log_lines, "No data available")
                return {"status": "no_data", "log_lines": log_lines, "health_state": "No Data"}

            obs_result = self._check_obs_limits(data)
            if not obs_result.get("ok"):
                for v in obs_result.get("violations", []):
                    self._monitoring_log(
                        log_lines,
                        f"OBS violation: {v.get('name')} ({v.get('parameter')}={v.get('value')})",
                    )

            if not self.models:
                self._monitoring_log(log_lines, "Warning: No models trained. Please train at least one model first.")
                return {"status": "no_models", "log_lines": log_lines, "health_state": "No Trained Model"}

            feature_df = self._numeric_feature_frame(data)
            predict_df = data
            if feature_df.empty:
                self._monitoring_log(log_lines, "No numeric features available for monitoring.")
                return {"status": "no_features", "log_lines": log_lines, "health_state": "No Numeric Features"}

            all_anomaly_counts = {}
            all_scores = {}
            model_quality_weights = {}
            drift_result = self._check_concept_drift(feature_df)
            self.last_drift_result = drift_result

            if drift_result.get("is_drift"):
                self._monitoring_log(
                    log_lines,
                    f"Drift detected: score={drift_result.get('drift_score', 0.0):.2f}, "
                    f"features={', '.join(drift_result.get('drifted_features', [])[:5])}",
                )

            for model_id, model_obj in self.models.items():
                if QThread.currentThread().isInterruptionRequested():
                    return {"status": "cancelled", "log_lines": log_lines}
                if model_obj is None:
                    continue

                models_list = self.config.get("models", [])
                model_config = next((m for m in models_list if m.get("model_id") == model_id), None)
                model_type = model_config.get("model_type", "Unknown") if model_config else "Unknown"

                try:
                    scores, anomalies = model_obj.predict(predict_df)
                    if scores is None:
                        err = anomalies if isinstance(anomalies, str) else "predict returned no scores"
                        self._monitoring_log(log_lines, f"{model_type} prediction failed: {err}")
                        continue
                    score_arr = np.asarray(scores, dtype=float).ravel()
                    score_arr = score_arr[np.isfinite(score_arr)]
                    if len(score_arr) == 0:
                        self._monitoring_log(log_lines, f"{model_type} prediction failed: empty/non-finite scores")
                        continue

                    anomaly_arr = np.asarray(anomalies).ravel() if anomalies is not None else np.array([], dtype=float)
                    if anomaly_arr.size > 0:
                        if anomaly_arr.dtype == bool:
                            anomaly_count = int(np.sum(anomaly_arr))
                        else:
                            anomaly_count = int(np.sum(anomaly_arr == -1))
                    else:
                        anomaly_count = 0

                    anomaly_ratio = anomaly_count / max(len(predict_df), 1)
                    score_norm = float(np.mean(np.abs(score_arr)))
                    score_norm = score_norm / (score_norm + 1.0)
                    model_score = max(score_norm, anomaly_ratio)

                    all_anomaly_counts[model_type] = anomaly_count
                    all_scores[model_type] = model_score

                    metrics = getattr(model_obj, "metrics", {}) if hasattr(model_obj, "metrics") else {}
                    perf = metrics.get("f1_score", metrics.get("roc_auc", metrics.get("auc", 1.0)))
                    try:
                        perf = float(perf)
                    except Exception:
                        perf = 1.0
                    model_quality_weights[model_type] = max(0.05, perf)

                    self._monitoring_log(
                        log_lines,
                        f"{model_type}: {anomaly_count} anomalies, score={model_score:.3f}",
                    )
                except Exception as pred_error:
                    logger.error("Error in %s prediction: %s", model_type, pred_error)
                    self._monitoring_log(log_lines, f"{model_type} prediction error: {str(pred_error)}")

            if not all_scores:
                self._monitoring_log(log_lines, "No valid model predictions in this cycle.")
                return {
                    "status": "prediction_unavailable",
                    "log_lines": log_lines,
                    "health_state": "Prediction Unavailable",
                    "drift_result": drift_result,
                }

            weight_sum = sum(model_quality_weights.get(name, 1.0) for name in all_scores.keys())
            fused_score = 0.0
            for name, score in all_scores.items():
                w = model_quality_weights.get(name, 1.0) / max(weight_sum, 1e-9)
                fused_score += (w * score)

            decision_threshold = self._compute_adaptive_threshold(fused_score)
            critical_threshold = max(0.9, decision_threshold + 0.1)
            is_critical = fused_score >= critical_threshold
            is_warning = (fused_score >= decision_threshold) and not is_critical

            xai_reasons = self._build_xai_reasons(
                feature_df=feature_df,
                final_score=fused_score,
                threshold=decision_threshold,
                drift_result=drift_result,
                agent_scores=all_scores,
            )
            self.last_xai_reasons = xai_reasons
            ttf_hours = self._estimate_ttf_hours(decision_threshold, critical_threshold)
            total_anomalies = sum(all_anomaly_counts.values())

            self._monitoring_log(
                log_lines,
                f"Fusion={fused_score:.3f}, threshold={decision_threshold:.3f}, critical={critical_threshold:.3f}",
            )
            self._monitoring_log(
                log_lines,
                f"Processed {len(data)} records with {len(all_scores)} active models",
            )

            payload = {
                "status": "success",
                "log_lines": log_lines,
                "drift_result": drift_result,
                "all_anomaly_counts": all_anomaly_counts,
                "fused_score": fused_score,
                "decision_threshold": decision_threshold,
                "critical_threshold": critical_threshold,
                "is_critical": is_critical,
                "is_warning": is_warning,
                "xai_reasons": xai_reasons,
                "ttf_hours": ttf_hours,
                "total_anomalies": total_anomalies,
                "data_len": len(data),
                "update_preview": not (is_warning or is_critical),
            }

            if is_critical:
                payload["health_state"] = "Critical"
                payload["log_lines"].append(
                    f"[{datetime.datetime.now().strftime('%H:%M:%S')}] CRITICAL: Immediate alarm path activated."
                )
            elif is_warning:
                payload["health_state"] = "Warning"
                if ttf_hours is not None:
                    payload["log_lines"].append(
                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] WARNING: Estimated TTF to critical ~ {ttf_hours:.2f}h"
                    )
                else:
                    payload["log_lines"].append(
                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] WARNING: Elevated anomaly trend detected."
                    )
            else:
                payload["health_state"] = "Healthy"

            if (is_warning or is_critical) and self.config.get("email_alerts_enabled", False):
                details = f"Pipeline alert in {self.config['title']}.\n"
                details += f"Total records: {len(data)}\n"
                details += f"Anomalies by model: {all_anomaly_counts}\n"
                details += f"Fusion score: {fused_score:.4f}\n"
                details += f"Adaptive threshold: {decision_threshold:.4f}\n"
                details += f"Critical threshold: {critical_threshold:.4f}\n"
                details += f"Drift: {drift_result.get('is_drift', False)} (score={drift_result.get('drift_score', 0.0):.3f})\n"
                details += f"XAI: {' | '.join(xai_reasons[:3])}\n"
                if ttf_hours is not None:
                    details += f"Estimated TTF to critical: {ttf_hours:.2f}h\n"
                payload["send_email"] = True
                payload["email_details"] = details

            return payload

        except Exception as exc:
            logger.error("Error during monitoring compute: %s", exc)
            self._monitoring_log(log_lines, f"Error: {str(exc)}")
            return {"status": "error", "log_lines": log_lines, "health_state": "Runtime Error", "error": str(exc)}

    def _apply_monitoring_cycle_result(self, result):
        """Apply monitoring results on the UI thread."""
        status = result.get("status", "error")
        if status == "skipped":
            return

        for line in result.get("log_lines", []):
            self.results_text.append(line)

        if status == "cancelled":
            return

        health_state = result.get("health_state")
        drift_result = result.get("drift_result", {})

        if status == "data_error":
            self._update_health_panel(health_state=health_state or "Data Error")
            return
        if status == "no_data":
            self._update_health_panel(health_state="No Data")
            self._publish_snapshot(health_state="No Data")
            return
        if status in ("no_models", "no_features", "prediction_unavailable", "error"):
            if health_state:
                self._update_health_panel(health_state=health_state, drift_result=drift_result)
            return

        if status != "success":
            return

        if drift_result.get("is_drift"):
            model_registry = getattr(self.parent_window, "model_registry", None)
            if model_registry:
                try:
                    signal_id = model_registry.mark_retrain_needed(
                        model_name=f"{self.config.get('title', 'custom_tab')}::fusion",
                        drift_score=drift_result.get("drift_score", 0.0),
                        reason="concept_drift",
                        drifted_features=drift_result.get("drifted_features", []),
                        tab_id=self.tab_id,
                        metadata={"tab_id": self.tab_id, "tab_title": self.config.get("title", "N/A")},
                    )
                    self.last_retrain_signal_id = signal_id
                    drifted = drift_result.get("drifted_features", [])
                    drifted_text = ", ".join(drifted) if drifted else "n/a"
                    self.results_text.append(
                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                        f"Retrain signal recorded (id={signal_id}): "
                        f"drift_score={drift_result.get('drift_score', 0.0):.3f}, "
                        f"features=[{drifted_text}]"
                    )
                    parent = getattr(self, "parent_window", None)
                    if parent is not None and hasattr(parent, "refresh_retrain_signals_panel"):
                        parent.refresh_retrain_signals_panel()
                except Exception as registry_error:
                    logger.error("Failed to record retrain signal: %s", registry_error)
                    err_text = (
                        format_registry_error(registry_error)
                        if format_registry_error
                        else "Retrain signal could not be saved."
                    )
                    self.results_text.append(
                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                        f"ERROR: {err_text}"
                    )

        fused_score = result.get("fused_score", 0.0)
        decision_threshold = result.get("decision_threshold", 0.0)
        critical_threshold = result.get("critical_threshold", 0.9)
        is_critical = result.get("is_critical", False)
        is_warning = result.get("is_warning", False)
        xai_reasons = result.get("xai_reasons", [])
        ttf_hours = result.get("ttf_hours")
        all_anomaly_counts = result.get("all_anomaly_counts", {})

        self._update_health_trend_plot(
            fused_score=fused_score,
            threshold=decision_threshold,
            drift_score=drift_result.get("drift_score", 0.0),
        )

        if is_critical:
            self.status_label.setText("Status: Monitoring - Critical Alarm")
            self._update_health_panel(
                health_state="Critical",
                fused_score=fused_score,
                threshold=decision_threshold,
                drift_result=drift_result,
                ttf_hours=ttf_hours,
                xai_reasons=xai_reasons,
            )
            self._publish_snapshot(health_state="Critical", fusion_score=fused_score)
            self._record_anomaly_event(
                fused_score, decision_threshold, all_anomaly_counts, xai_reasons, drift_result, "Critical"
            )
        elif is_warning:
            self.status_label.setText("Status: Monitoring - Warning")
            self._update_health_panel(
                health_state="Warning",
                fused_score=fused_score,
                threshold=decision_threshold,
                drift_result=drift_result,
                ttf_hours=ttf_hours,
                xai_reasons=xai_reasons,
            )
            self._publish_snapshot(health_state="Warning", fusion_score=fused_score)
            self._record_anomaly_event(
                fused_score, decision_threshold, all_anomaly_counts, xai_reasons, drift_result, "Warning"
            )
        else:
            self.status_label.setText(f"Status: Monitoring (healthy, score={fused_score:.3f})")
            self._update_health_panel(
                health_state="Healthy",
                fused_score=fused_score,
                threshold=decision_threshold,
                drift_result=drift_result,
                ttf_hours=ttf_hours,
                xai_reasons=xai_reasons,
            )
            self._publish_snapshot(health_state="Healthy", fusion_score=fused_score)
            if result.get("update_preview"):
                self.update_data_preview()

        if result.get("send_email"):
            self.send_email_alert(result.get("total_anomalies", 0), result.get("email_details", ""))

    def edit_configuration(self):
        """Edit tab configuration"""
        dialog = TabConfigurationDialog(self.parent_window, self.config)
        if dialog.exec_() == QDialog.Accepted:
            # Store existing trained models before update
            existing_models = self.models.copy()
            existing_data_processor = self.data_processor
            
            # Update config and save
            old_config = self.config.copy()
            self.config = ensure_fsm_fields(dict(dialog.config or {}))
            self.config['updated_at'] = datetime.datetime.now().isoformat()
            try:
                self._fsm_store.sync_tab_modes(self.tab_id, self.config.get("mission_modes") or [])
                new_mode = str(self.config.get("current_mission_mode") or "nominal")
                old_mode = str(old_config.get("current_mission_mode") or "nominal")
                if new_mode != old_mode:
                    mode = self._current_mission_mode()
                    self._fsm_store.set_mode(
                        self.tab_id,
                        new_mode,
                        threshold_scale=mode.threshold_scale,
                        trigger_source="manual",
                    )
            except Exception:
                pass
            if hasattr(self, "mission_mode_combo"):
                self.mission_mode_combo.blockSignals(True)
                self.mission_mode_combo.clear()
                for m in self.config.get("mission_modes") or []:
                    self.mission_mode_combo.addItem(str(m.get("name", "nominal")))
                idx = self.mission_mode_combo.findText(
                    str(self.config.get("current_mission_mode") or "nominal")
                )
                if idx >= 0:
                    self.mission_mode_combo.setCurrentIndex(idx)
                self.mission_mode_combo.blockSignals(False)

            # Preserve existing models (by matching model_ids)
            if 'models' in self.config:
                # Update model_ids in new config to match existing trained models
                for new_model_config in self.config['models']:
                    new_model_id = new_model_config.get('model_id')
                    # Try to find matching model from old config
                    if 'models' in old_config:
                        for old_model_config in old_config['models']:
                            old_model_id = old_model_config.get('model_id')
                            if (old_model_config.get('model_type') == new_model_config.get('model_type') and
                                old_model_id in existing_models):
                                # Reuse the old model_id and trained model
                                new_model_config['model_id'] = old_model_id
                                self.models[old_model_id] = existing_models[old_model_id]
                                break
            
            # Restore data processor
            self.data_processor = existing_data_processor
            
            # Save to manager
            if hasattr(self.parent_window, 'tab_config_manager'):
                self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)
            
            # Update tab title
            if hasattr(self.parent_window, 'tabs'):
                for i in range(self.parent_window.tabs.count()):
                    if self.parent_window.tabs.widget(i) == self:
                        self.parent_window.tabs.setTabText(i, self.config['title'])
                        break
            
            # Update UI elements without full rebuild
            self.update_ui_after_config_change()
            
            QMessageBox.information(self, "Configuration Updated", 
                                  "Tab configuration has been updated. Trained models preserved.")
    
    def update_ui_after_config_change(self):
        """Update UI elements after configuration change without full rebuild"""
        try:
            # Update models table (most important)
            if hasattr(self, 'models_table'):
                self.update_models_table()
            
            # Update header labels by searching through widgets
            self.update_header_labels()
            
            # Refresh results/log
            if hasattr(self, 'results_text'):
                self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Configuration updated")
                
        except Exception as e:
            logger.error(f"Error updating UI after config change: {str(e)}")
            # If update fails, just log it - don't break the tab
    
    def update_header_labels(self):
        """Helper to update header labels with current config values"""
        try:
            title = self.config.get("title", "")
            subsystem = self.config.get("subsystem_name") or "—"
            schedule = self.config.get("schedule_type", "")
            input_mode = self.config.get("input_mode", "CSV Polling")
            data_source = self.config.get("data_folder", "—")

            if hasattr(self, "header_tab_label"):
                self.header_tab_label.setText(title)
            if hasattr(self, "header_schedule_label"):
                self.header_schedule_label.setText(schedule)
            if hasattr(self, "header_input_label"):
                self.header_input_label.setText(input_mode)
            if hasattr(self, "header_file_type_label"):
                self.header_file_type_label.setText(self.config.get("data_file_type", "CSV"))
            if hasattr(self, "header_dataset_mode_label"):
                self.header_dataset_mode_label.setText(self.config.get("dataset_mode", "Full Latest File"))
            if hasattr(self, "header_data_source_label"):
                self.header_data_source_label.setText(data_source)
            if hasattr(self, "header_subsystem_label"):
                self.header_subsystem_label.setText(subsystem)

            models_list = self.config.get("models", [])
            if not models_list and "model_type" in self.config:
                models_list = [{"model_type": self.config["model_type"]}]
            count = len(models_list)
            if hasattr(self, "header_models_count_label"):
                self.header_models_count_label.setText(str(count))
        except Exception as e:
            logger.error(f"Error updating header labels: {str(e)}")
    
    def _update_labels_recursive(self, layout):
        """Recursively find and update QLabel widgets"""
        try:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item:
                    if item.widget():
                        widget = item.widget()
                        if isinstance(widget, QLabel):
                            text = widget.text()
                            if "<b>Tab:</b>" in text:
                                widget.setText(f"<b>Tab:</b> {self.config['title']}")
                            elif "<b>Schedule:</b>" in text:
                                widget.setText(f"<b>Schedule:</b> {self.config['schedule_type']}")
                            elif "<b>Input:</b>" in text:
                                widget.setText(f"<b>Input:</b> {self.config.get('input_mode', 'CSV Polling')}")
                            elif "<b>Data Source:</b>" in text:
                                widget.setText(f"<b>Data Source:</b> {self.config['data_folder']}")
                            elif "<b>Subsystem:</b>" in text:
                                subsystem_name = self.config.get('subsystem_name')
                                if subsystem_name:
                                    widget.setText(f"<b>Subsystem:</b> {subsystem_name}")
                            elif "<b>Models Configured:</b>" in text:
                                models_list = self.config.get('models', [])
                                if not models_list and 'model_type' in self.config:
                                    models_list = [{'model_type': self.config['model_type']}]
                                widget.setText(f"<b>Models Configured:</b> {len(models_list)}")
                    elif item.layout():
                        # Recursively check nested layouts
                        self._update_labels_recursive(item.layout())
        except Exception as e:
            logger.error(f"Error in recursive label update: {str(e)}")
    
    def clear_layout(self, layout):
        """Helper to clear a layout recursively"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
    
    def save_model(self, model_id=None):
        """Save a trained model"""
        if model_id is None:
            if hasattr(self, 'selected_model_id'):
                model_id = self.selected_model_id
            else:
                QMessageBox.warning(self, "Save Model", "Please select a model to save.")
                return
        
        if model_id not in self.models or self.models[model_id] is None:
            QMessageBox.warning(self, "Save Model", "Selected model is not trained. Please train it first.")
            return
        
        model_obj = self.models[model_id]
        
        # Get model config for name
        models_list = self.config.get('models', [])
        model_config = next((m for m in models_list if m.get('model_id') == model_id), None)
        model_type = model_config.get('model_type', 'Unknown') if model_config else 'Unknown'
        
        try:
            filepath, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Model", 
                os.path.join(DATA_DIR, f"{self.config['title']}_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"),
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if filepath:
                save_model_to_file(model_obj, filepath)
                
                # Register in model registry
                if hasattr(self.parent_window, 'model_registry'):
                    model_name = f"{self.config['title']}_{model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.parent_window.model_registry.register_model(
                        model_name,
                        model_type,
                        metadata={
                            'filepath': filepath,
                            'tab_id': self.tab_id,
                            'model_id': model_id,
                            'parameters': model_config.get('model_parameters', {}) if model_config else {}
                        }
                    )
                
                self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model {model_type} saved to {filepath}")
                QMessageBox.information(self, "Save Model", f"Model saved successfully to:\n{filepath}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            QMessageBox.critical(self, "Save Error", f"Error saving model: {str(e)}")
    
    def load_model(self):
        """Load a saved model and add to models list"""
        try:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Load Model",
                DATA_DIR,
                "Pickle Files (*.pkl);;All Files (*)"
            )
            
            if filepath and os.path.exists(filepath):
                loaded_model = load_model_from_file(
                    filepath,
                    anomaly_detection_model_cls=AnomalyDetectionModel,
                    enhanced_anomaly_detection_model_cls=EnhancedAnomalyDetectionModel,
                    safe_pickle_load=safe_pickle_load,
                )

                if hasattr(loaded_model, 'model_type'):
                    display_type = internal_to_display_type(str(loaded_model.model_type))
                else:
                    display_type = "Unknown"

                new_model_id, new_model_config = build_loaded_model_entry(loaded_model)
                
                models_list = self.config.get('models', [])
                models_list.append(new_model_config)
                self.config['models'] = models_list
                
                # Store loaded model
                self.models[new_model_id] = loaded_model
                
                # Save config
                if hasattr(self.parent_window, 'tab_config_manager'):
                    self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)
                
                # Update UI
                self.update_models_table()
                self.select_model(new_model_id)
                self.save_model_btn.setEnabled(True)
                self.model_metrics_btn.setEnabled(True)
                
                self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model {display_type} loaded from {filepath}")
                QMessageBox.information(self, "Load Model", f"Model {display_type} loaded successfully and added to models list!")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            QMessageBox.critical(self, "Load Error", f"Error loading model: {str(e)}")
    
    def show_recent_models(self):
        """Show recent models dialog"""
        if not hasattr(self.parent_window, 'model_registry'):
            QMessageBox.warning(self, "Recent Models", "Model registry not available.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Recent Models")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout(dialog)
        
        # Model list
        model_table = QTableWidget()
        model_table.setColumnCount(4)
        model_table.setHorizontalHeaderLabels(["Model Name", "Type", "Created", "Actions"])
        model_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Get recent models from registry
        recent_models = self.parent_window.model_registry.get_recent_models(20)
        model_table.setRowCount(len(recent_models))
        
        for i, (model_id, name, model_type, created_at, use_count, filepath) in enumerate(recent_models):
            model_table.setItem(i, 0, QTableWidgetItem(name))
            model_table.setItem(i, 1, QTableWidgetItem(model_type))
            model_table.setItem(i, 2, QTableWidgetItem(created_at[:19] if created_at else "N/A"))
            
            # Load button
            load_btn = QPushButton("Load")
            load_btn.clicked.connect(
                lambda checked, fp=filepath, mid=model_id: self.load_model_from_path(fp, dialog, registry_model_id=mid)
            )
            model_table.setCellWidget(i, 3, load_btn)
        
        layout.addWidget(model_table)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def load_model_from_path(self, filepath, dialog, registry_model_id=None):
        """Load model from filepath and close dialog"""
        try:
            if not os.path.exists(filepath):
                QMessageBox.warning(self, "Load Error", "Model file not found.")
                return

            loaded_model = load_model_from_file(
                filepath,
                anomaly_detection_model_cls=AnomalyDetectionModel,
                enhanced_anomaly_detection_model_cls=EnhancedAnomalyDetectionModel,
                safe_pickle_load=safe_pickle_load,
            )

            if hasattr(loaded_model, 'model_type'):
                display_type = internal_to_display_type(str(loaded_model.model_type))
            else:
                display_type = "Unknown"

            new_model_id = str(uuid.uuid4())
            models_list = self.config.get('models', [])
            models_list.append({
                'model_type': display_type,
                'model_parameters': {},
                'model_id': new_model_id,
            })
            self.config['models'] = models_list
            self.models[new_model_id] = loaded_model
            self.model = loaded_model

            if hasattr(self.parent_window, 'tab_config_manager'):
                self.parent_window.tab_config_manager.add_config(self.tab_id, self.config)

            self.update_models_table()
            self.select_model(new_model_id)
            self.results_text.append(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Model {display_type} loaded from {filepath}"
            )
            if registry_model_id and hasattr(self.parent_window, "model_registry"):
                try:
                    self.parent_window.model_registry.record_model_usage(registry_model_id)
                except Exception:
                    pass
            dialog.accept()
            QMessageBox.information(self, "Load Model", f"Model {display_type} loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading model: {str(e)}")
    
    def show_model_metrics(self):
        """Show model performance metrics"""
        if not hasattr(self, 'selected_model_id'):
            QMessageBox.warning(self, "Model Metrics", "Please select a model from the table first.")
            return
        
        model_id = self.selected_model_id
        if model_id not in self.models or self.models[model_id] is None:
            QMessageBox.warning(self, "Model Metrics", "Selected model is not trained yet.")
            return
        
        model_obj = self.models[model_id]
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Metrics")
        dialog.setMinimumSize(500, 400)
        layout = QVBoxLayout(dialog)
        
        metrics_text = QTextEdit()
        metrics_text.setReadOnly(True)
        
        # Get model config for name
        models_list = self.config.get('models', [])
        model_config = next((m for m in models_list if m.get('model_id') == model_id), None)
        model_type = model_config.get('model_type', 'Unknown') if model_config else 'Unknown'
        
        # Get metrics from model
        metrics_info = "Model Information:\n"
        metrics_info += f"Type: {getattr(model_obj, 'model_type', model_type)}\n"
        metrics_info += f"Model ID: {model_id}\n\n"
        
        if hasattr(model_obj, 'metrics'):
            metrics_info += "Performance Metrics:\n"
            for metric_name, metric_value in model_obj.metrics.items():
                metrics_info += f"{metric_name}: {metric_value}\n"
        else:
            metrics_info += "No metrics available for this model.\n"
        
        metrics_text.setText(metrics_info)
        layout.addWidget(metrics_text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec_()
    
    def send_email_alert(self, anomaly_count, details=""):
        """Send email alert when anomalies detected"""
        if not self.config.get('email_alerts_enabled', False):
            return
        
        try:
            # Load email config from admin tab
            email_config_file = os.path.join(DATA_DIR, "email_config.json")
            if not os.path.exists(email_config_file):
                logger.warning("Email configuration not found. Please configure in Administration tab.")
                return
            
            with open(email_config_file, 'r', encoding='utf-8') as f:
                email_config = json.load(f)
            
            if not all([email_config.get('smtp_server'), email_config.get('smtp_username'), email_config.get('from_email')]):
                logger.warning("Email configuration incomplete. Please configure in Administration tab.")
                return
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            recipients = self.config.get('alert_recipient_emails', [])
            if not recipients:
                raw_recipients = str(self.config.get('alert_recipient_email', '')).strip()
                if raw_recipients:
                    recipients = [e.strip() for e in raw_recipients.split(',') if e.strip()]
            if not recipients:
                recipients = [email_config['smtp_username']]
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"Alert: {anomaly_count} Anomalies Detected - {self.config['title']}"
            
            body = f"""
Anomaly Alert from Monitoring Tab: {self.config['title']}

Number of Anomalies Detected: {anomaly_count}
Detection Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model Type: {self.config['model_type']}
Data Source: {self.config['data_folder']}

{details}

This is an automated alert from SDA v4.0.
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config.get('smtp_port', 587))
            server.starttls()
            server.login(email_config['smtp_username'], email_config['smtp_password'])
            server.sendmail(email_config['from_email'], recipients, msg.as_string())
            server.quit()
            
            logger.info(f"Email alert sent for {self.config['title']}: {anomaly_count} anomalies")
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Email alert sent ({anomaly_count} anomalies)")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            self.results_text.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Email alert failed: {str(e)}")
    
    def delete_tab(self):
        """Delete this custom tab"""
        if not self.parent_window.service_layer.authorize(
                self.parent_window.current_username, "delete_tab", resource=f"tab:{self.tab_id}"):
            QMessageBox.warning(self, "Permission Denied", 
                              "Only administrators can delete custom tabs.")
            return
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Tab",
            f"Are you sure you want to delete the tab '{self.config['title']}'?\n\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        # Stop monitoring if active
        if self.monitoring_active:
            self.stop_monitoring()
        
        # Remove from parent window
        if hasattr(self.parent_window, 'remove_custom_tab'):
            success = self.parent_window.remove_custom_tab(self.tab_id)
            if success:
                QMessageBox.information(self, "Tab Deleted", 
                                      f"Tab '{self.config['title']}' has been deleted successfully.")
            else:
                QMessageBox.warning(self, "Delete Error", 
                                  "Failed to delete tab. Please try again.")
    
    def clear_layout(self, layout):
        """Helper to clear a layout recursively"""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())
