"""Quick Actions page panel builders (Page 1 — runtime hub widgets on tab)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.tabs.custom_tab.panels.header_config import build_header_config_group

if TYPE_CHECKING:
    from app.tabs.custom_tab.widget import CustomMonitoringTab


def build_snapshot_group(tab: CustomMonitoringTab) -> QGroupBox:
    group = QGroupBox("Tab Snapshot (runtime status)")
    layout = QFormLayout()
    tab.snapshot_health_label = QLabel("Idle")
    tab.snapshot_fusion_label = QLabel("—")
    tab.snapshot_file_label = QLabel("—")
    tab.snapshot_watch_label = QLabel("Status: Inactive")
    tab.snapshot_obs_label = QLabel("OK")
    tab.snapshot_mllm_label = QLabel("—")
    tab.snapshot_mllm_label.setWordWrap(True)
    layout.addRow("Health:", tab.snapshot_health_label)
    layout.addRow("Fusion score:", tab.snapshot_fusion_label)
    layout.addRow("Last file:", tab.snapshot_file_label)
    layout.addRow("Watch status:", tab.snapshot_watch_label)
    layout.addRow("OBS limits:", tab.snapshot_obs_label)
    layout.addRow("M-LLM logs:", tab.snapshot_mllm_label)
    group.setLayout(layout)
    return group


def build_overview_actions(tab: CustomMonitoringTab) -> QHBoxLayout:
    layout = QHBoxLayout()
    export_report_btn = QPushButton("Export Mission Report")
    export_report_btn.clicked.connect(tab._on_export_mission_report)
    layout.addWidget(export_report_btn)
    layout.addStretch()
    return layout


def build_monitoring_controls_group(tab: CustomMonitoringTab) -> QGroupBox:
    group = QGroupBox("Monitoring Controls")
    actions_layout = QVBoxLayout()
    controls_layout = QHBoxLayout()

    tab.start_btn = QPushButton("Start Monitoring")
    tab.start_btn.clicked.connect(tab.start_monitoring)
    tab.stop_btn = QPushButton("Stop")
    tab.stop_btn.clicked.connect(tab.stop_monitoring)
    tab.stop_btn.setEnabled(False)
    tab.edit_btn = QPushButton("Edit Config")
    tab.edit_btn.clicked.connect(tab.edit_configuration)
    if not tab.parent_window.service_layer.authorize(
        tab.parent_window.current_username,
        "configure_system",
        resource=f"tab:{tab.tab_id}",
    ):
        tab.edit_btn.setEnabled(False)

    tab.delete_tab_btn = QPushButton("Delete Tab")
    tab.delete_tab_btn.clicked.connect(tab.delete_tab)
    if not tab.parent_window.service_layer.authorize(
        tab.parent_window.current_username,
        "delete_tab",
        resource=f"tab:{tab.tab_id}",
    ):
        tab.delete_tab_btn.setEnabled(False)
        tab.delete_tab_btn.setToolTip("Only administrators can delete tabs")

    controls_layout.addWidget(tab.start_btn)
    controls_layout.addWidget(tab.stop_btn)
    controls_layout.addWidget(tab.edit_btn)
    controls_layout.addStretch()
    controls_layout.addWidget(tab.delete_tab_btn)
    actions_layout.addLayout(controls_layout)

    mode_row = QHBoxLayout()
    mode_row.addWidget(QLabel("Mission mode:"))
    tab.mission_mode_combo = QComboBox()
    for mode in tab.config.get("mission_modes") or []:
        name = str(mode.get("name") or "").strip()
        if name:
            tab.mission_mode_combo.addItem(name)
    if tab.mission_mode_combo.count() == 0:
        for name in ("nominal", "eclipse", "maneuver", "safe_mode"):
            tab.mission_mode_combo.addItem(name)
    current = str(tab.config.get("current_mission_mode") or "nominal")
    idx = tab.mission_mode_combo.findText(current)
    if idx >= 0:
        tab.mission_mode_combo.setCurrentIndex(idx)
    tab.mission_mode_combo.currentTextChanged.connect(tab._on_mission_mode_changed)
    mode_row.addWidget(tab.mission_mode_combo)
    mode_row.addStretch()
    actions_layout.addLayout(mode_row)

    tab.status_label = QLabel("Status: Inactive")
    tab.status_label.setWordWrap(True)
    actions_layout.addWidget(tab.status_label)
    group.setLayout(actions_layout)
    return group


def build_monitoring_log_group(tab: CustomMonitoringTab) -> QGroupBox:
    group = QGroupBox("Monitoring Log")
    layout = QVBoxLayout()
    tab.results_text = QTextEdit()
    tab.results_text.setReadOnly(True)
    tab.results_text.setMinimumHeight(100)
    tab.results_text.setPlaceholderText("Monitoring events will appear here.")
    layout.addWidget(tab.results_text)
    group.setLayout(layout)
    return group


def build_health_status_group(tab: CustomMonitoringTab) -> QGroupBox:
    group = QGroupBox("Health Status")
    layout = QFormLayout()
    tab.health_state_value = QLabel("Idle")
    tab.fusion_value_label = QLabel("N/A")
    tab.threshold_value_label = QLabel("N/A")
    tab.ttf_value_label = QLabel("N/A")
    layout.addRow("State:", tab.health_state_value)
    layout.addRow("Fusion score:", tab.fusion_value_label)
    layout.addRow("Threshold:", tab.threshold_value_label)
    layout.addRow("Est. TTF:", tab.ttf_value_label)
    tab.forecast_value_label = QLabel("N/A")
    layout.addRow("Forecast (next score):", tab.forecast_value_label)
    group.setLayout(layout)
    return group


def build_diagnostics_group(tab: CustomMonitoringTab) -> QGroupBox:
    group = QGroupBox("Diagnostics")
    layout = QFormLayout()
    tab.drift_value_label = QLabel("N/A")
    tab.xai_value_label = QLabel("N/A")
    tab.drift_value_label.setWordWrap(True)
    tab.xai_value_label.setWordWrap(True)
    layout.addRow("Concept drift:", tab.drift_value_label)
    layout.addRow("Root cause (XAI):", tab.xai_value_label)
    group.setLayout(layout)
    return group


def build_health_trend_group(tab: CustomMonitoringTab) -> QGroupBox:
    group = QGroupBox("Health Trend")
    layout = QVBoxLayout()
    tab.health_trend_canvas = FigureCanvas(Figure(figsize=(8, 3.0)))
    tab.health_trend_canvas.setMinimumHeight(180)
    layout.addWidget(tab.health_trend_canvas)
    group.setLayout(layout)
    return group


def build_alert_history_group(tab: CustomMonitoringTab) -> QGroupBox:
    group = QGroupBox("Alert History")
    layout = QVBoxLayout()
    tab.anomaly_history_table = QTableWidget(0, 6)
    tab.anomaly_history_table.setHorizontalHeaderLabels(
        ["Time", "State", "Fusion", "Threshold", "Anomalies", "XAI"]
    )
    tab.anomaly_history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tab.anomaly_history_table.setAlternatingRowColors(True)
    tab.anomaly_history_table.verticalHeader().setVisible(False)
    tab.anomaly_history_table.setMinimumHeight(180)
    layout.addWidget(tab.anomaly_history_table)
    group.setLayout(layout)
    return group


def build_quick_actions_page(tab: CustomMonitoringTab, models_list: List[dict]) -> QWidget:
    """Assemble Quick Actions page; all widgets attached to *tab*."""
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setSpacing(12)
    layout.setContentsMargins(10, 10, 10, 10)

    layout.addWidget(build_header_config_group(tab, models_list))
    layout.addWidget(build_snapshot_group(tab))
    layout.addLayout(build_overview_actions(tab))
    layout.addWidget(build_monitoring_controls_group(tab))
    layout.addWidget(build_monitoring_log_group(tab))
    layout.addWidget(build_health_status_group(tab))
    layout.addWidget(build_diagnostics_group(tab))
    layout.addWidget(build_health_trend_group(tab))
    layout.addWidget(build_alert_history_group(tab))

    return page
