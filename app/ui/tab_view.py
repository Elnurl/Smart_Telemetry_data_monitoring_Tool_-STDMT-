from __future__ import annotations

from pathlib import Path

from app.monitoring.tab_runtime import TabRuntimeState
from app.reports.mission_report import export_mission_report
from app.ui.qt_compat import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from app.ui.telemetry_plot import TelemetryPlotWidget


def _stretch(table: QTableWidget) -> None:
    mode = QHeaderView.ResizeMode.Stretch if hasattr(QHeaderView, "ResizeMode") else QHeaderView.Stretch
    table.horizontalHeader().setSectionResizeMode(mode)


class MonitoringTabView(QWidget):
    """Single monitoring channel view with Live / Models / Health / Drift."""

    def __init__(self, runtime: TabRuntimeState, reports_dir: Path, parent=None):
        super().__init__(parent)
        self.runtime = runtime
        self.reports_dir = reports_dir
        self._build_ui()
        self.runtime.reload_data(force=True)
        self.refresh_view(None)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QHBoxLayout()
        self.title_label = QLabel(self.runtime.config.get("title", "Monitoring Tab"))
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        header.addWidget(self.title_label)
        header.addStretch()
        self.reload_btn = QPushButton("Reload Data")
        self.train_btn = QPushButton("Train Models")
        self.config_btn = QPushButton("Configure")
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerBtn")
        for btn in (self.reload_btn, self.train_btn, self.config_btn, self.start_btn, self.stop_btn):
            header.addWidget(btn)
        layout.addLayout(header)

        status_row = QHBoxLayout()
        self.health_label = QLabel("Health: Idle")
        self.fusion_label = QLabel("Fusion: —")
        self.file_label = QLabel("File: —")
        self.obs_label = QLabel("OBS: —")
        self.models_label = QLabel("Models: 0")
        for w in (self.health_label, self.fusion_label, self.models_label, self.obs_label, self.file_label):
            w.setStyleSheet("padding: 6px 10px; background: #ffffff; border: 1px solid #dfe3eb; border-radius: 6px;")
            status_row.addWidget(w)
        status_row.addStretch()
        layout.addLayout(status_row)

        self.inner_tabs = QTabWidget()
        self.inner_tabs.addTab(self._build_live_tab(), "Live Data")
        self.inner_tabs.addTab(self._build_models_tab(), "Models")
        self.inner_tabs.addTab(self._build_health_tab(), "Health & Alerts")
        self.inner_tabs.addTab(self._build_drift_tab(), "Drift")
        layout.addWidget(self.inner_tabs, 1)

    def _build_live_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        feature_row = QHBoxLayout()
        feature_row.addWidget(QLabel("Feature"))
        self.feature_combo = QComboBox()
        self.feature_combo.currentTextChanged.connect(self._update_plot)
        feature_row.addWidget(self.feature_combo, 1)
        v.addLayout(feature_row)
        self.plot = TelemetryPlotWidget()
        v.addWidget(self.plot, 1)
        return w

    def _build_models_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        info = QLabel("Configured models are trained on the latest data window and saved locally for offline inference.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #5c6478;")
        v.addWidget(info)
        self.models_table = QTableWidget(0, 4)
        self.models_table.setHorizontalHeaderLabels(["Model", "Backend", "Trained", "Saved At"])
        _stretch(self.models_table)
        v.addWidget(self.models_table, 1)
        return w

    def _build_health_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        metrics_group = QGroupBox("Health Status")
        mg = QHBoxLayout(metrics_group)
        self.health_state_big = QLabel("Idle")
        self.health_state_big.setStyleSheet("font-size: 22px; font-weight: 700;")
        self.ttf_label = QLabel("TTF: —")
        self.alert_count_label = QLabel("Alerts: 0")
        mg.addWidget(self.health_state_big)
        mg.addStretch()
        mg.addWidget(self.ttf_label)
        mg.addWidget(self.alert_count_label)
        v.addWidget(metrics_group)

        xai_group = QGroupBox("Explanation (XAI)")
        xg = QVBoxLayout(xai_group)
        self.xai_text = QTextEdit()
        self.xai_text.setReadOnly(True)
        self.xai_text.setMaximumHeight(90)
        xg.addWidget(self.xai_text)
        v.addWidget(xai_group)

        events_group = QGroupBox("Anomaly Events")
        eg = QVBoxLayout(events_group)
        self.events_table = QTableWidget(0, 4)
        self.events_table.setHorizontalHeaderLabels(["Time", "State", "Fusion", "Explanation"])
        _stretch(self.events_table)
        eg.addWidget(self.events_table)
        report_btn = QPushButton("Export Mission Report")
        report_btn.clicked.connect(self._export_report)
        eg.addWidget(report_btn)
        v.addWidget(events_group, 1)
        return w

    def _build_drift_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.drift_status = QLabel("Drift: not evaluated yet")
        self.drift_status.setStyleSheet("font-size: 15px; font-weight: 600;")
        v.addWidget(self.drift_status)
        self.drift_table = QTableWidget(0, 4)
        self.drift_table.setHorizontalHeaderLabels(["Feature", "KS p-value", "PSI", "Drift"])
        _stretch(self.drift_table)
        v.addWidget(self.drift_table, 1)
        return w

    # ---- refresh ----------------------------------------------------------
    def refresh_view(self, cycle_result: dict | None) -> None:
        snap = self.runtime.get_snapshot()
        self.health_label.setText(f"Health: {snap.get('health_state', 'Idle')}")
        fs = snap.get("fusion_score")
        self.fusion_label.setText("Fusion: —" if fs is None else f"Fusion: {float(fs):.3f}")
        self.models_label.setText(f"Models: {snap.get('trained_models', 0)}")
        self.obs_label.setText("OBS: OK" if snap.get("obs_ok", True) else f"OBS: {snap.get('obs_violations', 0)} viol.")
        last_file = str(snap.get("last_file", "—"))
        if len(last_file) > 50:
            last_file = "…" + last_file[-47:]
        self.file_label.setText(f"File: {last_file}")

        self.health_state_big.setText(snap.get("health_state", "Idle"))
        ttf = snap.get("ttf")
        self.ttf_label.setText("TTF: —" if ttf is None else f"TTF: {float(ttf):.1f} cycles")
        self.alert_count_label.setText(f"Alerts: {snap.get('alert_count', 0)}")

        self._refresh_features()
        self._update_plot()
        self._refresh_models_table()
        self._refresh_events_table()
        self._refresh_xai()
        self._refresh_drift_table(cycle_result)

    def _refresh_features(self) -> None:
        features = self.runtime.feature_names()
        current = self.feature_combo.currentText()
        self.feature_combo.blockSignals(True)
        self.feature_combo.clear()
        self.feature_combo.addItems(features)
        if current in features:
            self.feature_combo.setCurrentText(current)
        self.feature_combo.blockSignals(False)

    def _update_plot(self) -> None:
        feature = self.feature_combo.currentText()
        if not feature:
            self.plot.clear()
            return
        series = self.runtime.get_plot_data(feature)
        if series is None:
            self.plot.clear()
            return
        x, y = series
        self.plot.plot_series(x, y, feature)

    def _refresh_models_table(self) -> None:
        manifest = self.runtime.model_store.list_models()
        self.models_table.setRowCount(len(manifest))
        for row, (_mid, info) in enumerate(manifest.items()):
            self.models_table.setItem(row, 0, QTableWidgetItem(info.get("model_type", "—")))
            self.models_table.setItem(row, 1, QTableWidgetItem(info.get("backend", "—")))
            self.models_table.setItem(row, 2, QTableWidgetItem("Yes"))
            self.models_table.setItem(row, 3, QTableWidgetItem(info.get("saved_at", "—")[:19]))

    def _refresh_events_table(self) -> None:
        events = list(self.runtime.anomaly_events)[:100]
        self.events_table.setRowCount(len(events))
        for row, ev in enumerate(events):
            self.events_table.setItem(row, 0, QTableWidgetItem(str(ev.get("timestamp", ""))))
            self.events_table.setItem(row, 1, QTableWidgetItem(str(ev.get("health_state", ""))))
            self.events_table.setItem(row, 2, QTableWidgetItem(f"{float(ev.get('fused_score', 0) or 0):.4f}"))
            self.events_table.setItem(row, 3, QTableWidgetItem(str(ev.get("xai", ""))))

    def _refresh_xai(self) -> None:
        reasons = self.runtime.last_xai_reasons
        self.xai_text.setPlainText("\n".join(f"• {r}" for r in reasons) if reasons else "No analysis yet.")

    def _refresh_drift_table(self, cycle_result: dict | None) -> None:
        drift = self.runtime.last_drift_result or {}
        is_drift = drift.get("is_drift", False)
        ratio = drift.get("drift_ratio", drift.get("drift_score", 0.0))
        status = drift.get("status")
        if status == "reference_initialized":
            self.drift_status.setText("Drift: reference window initialized — needs another cycle")
        else:
            self.drift_status.setText(
                f"Drift: {'DETECTED' if is_drift else 'stable'} (ratio={ratio:.2f})"
            )
        details = drift.get("feature_details", {})
        self.drift_table.setRowCount(len(details))
        for row, (feat, d) in enumerate(details.items()):
            self.drift_table.setItem(row, 0, QTableWidgetItem(str(feat)))
            self.drift_table.setItem(row, 1, QTableWidgetItem(f"{d.get('ks_pvalue', 1.0):.4f}"))
            self.drift_table.setItem(row, 2, QTableWidgetItem(f"{d.get('psi', 0.0):.4f}"))
            self.drift_table.setItem(row, 3, QTableWidgetItem("Yes" if d.get("is_drift") else "No"))

    def _export_report(self) -> None:
        snap = self.runtime.get_snapshot()
        events = list(self.runtime.anomaly_events)
        try:
            path = export_mission_report(self.reports_dir, self.runtime.config.get("title", "tab"), snap, events)
            QMessageBox.information(self, "Mission Report", f"Report saved:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Report Error", f"Failed to export report:\n{exc}")

    def set_monitoring_active(self, active: bool) -> None:
        self.runtime.set_monitoring(active)
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)
        self.refresh_view(None)
