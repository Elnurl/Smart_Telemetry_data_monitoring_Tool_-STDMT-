from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.monitoring.tab_config import DEFAULT_MODELS
from app.ui.qt_compat import (
    Checked,
    ItemIsUserCheckable,
    Unchecked,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    exec_dialog,
)
from app.monitoring.scheduler import interval_ms_from_parts


class TabConfigDialog(QDialog):
    """Per-tab configuration dialog for monitoring channels."""

    def __init__(self, parent=None, existing_config: dict | None = None):
        super().__init__(parent)
        self.existing_config = existing_config
        self.result_config: dict | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        is_edit = self.existing_config is not None
        self.setWindowTitle("Edit Monitoring Tab" if is_edit else "New Monitoring Tab")
        self.setMinimumSize(640, 720)

        root = QVBoxLayout(self)

        # Tab info
        info_group = QGroupBox("Tab Information")
        info_layout = QFormLayout(info_group)
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("e.g., EPS Power Monitoring")
        if is_edit:
            self.title_input.setText(self.existing_config.get("title", ""))
        info_layout.addRow("Tab Title", self.title_input)
        self.subsystem_input = QLineEdit()
        self.subsystem_input.setPlaceholderText("e.g., EPS, Thermal")
        if is_edit:
            self.subsystem_input.setText(self.existing_config.get("subsystem_name", ""))
        info_layout.addRow("Subsystem", self.subsystem_input)
        root.addWidget(info_group)

        # Data source
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_group)

        folder_row = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        if is_edit:
            self.folder_input.setText(self.existing_config.get("data_folder", ""))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(QLabel("Data Folder"))
        folder_row.addWidget(self.folder_input, 1)
        folder_row.addWidget(browse_btn)
        source_layout.addLayout(folder_row)

        type_row = QHBoxLayout()
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["CSV", "JSON"])
        if is_edit:
            idx = self.file_type_combo.findText(self.existing_config.get("data_file_type", "CSV"))
            if idx >= 0:
                self.file_type_combo.setCurrentIndex(idx)
        self.file_type_combo.currentTextChanged.connect(lambda _: self._refresh_features())
        type_row.addWidget(QLabel("File Type"))
        type_row.addWidget(self.file_type_combo)

        self.dataset_mode_combo = QComboBox()
        self.dataset_mode_combo.addItems(["Full Latest File", "Daily", "Weekly", "Monthly"])
        if is_edit:
            idx = self.dataset_mode_combo.findText(self.existing_config.get("dataset_mode", "Full Latest File"))
            if idx >= 0:
                self.dataset_mode_combo.setCurrentIndex(idx)
        type_row.addWidget(QLabel("Dataset Window"))
        type_row.addWidget(self.dataset_mode_combo)
        type_row.addStretch()
        source_layout.addLayout(type_row)

        source_layout.addWidget(QLabel("Features (unchecked = use all numeric columns)"))
        self.features_list = QListWidget()
        self.features_list.setMaximumHeight(120)
        source_layout.addWidget(self.features_list)
        if is_edit and self.existing_config.get("data_folder"):
            self._refresh_features()
        root.addWidget(source_group)

        # Models
        model_group = QGroupBox("ML Models (Phase 1 — selection saved for training)")
        model_layout = QVBoxLayout(model_group)
        self.models_list = QListWidget()
        self.models_list.setMaximumHeight(140)
        existing_models = []
        if is_edit:
            existing_models = [m.get("model_type") for m in self.existing_config.get("models", [])]
        for name in DEFAULT_MODELS:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | ItemIsUserCheckable)
            checked = name in existing_models if existing_models else name == "Isolation Forest"
            item.setCheckState(Checked if checked else Unchecked)
            self.models_list.addItem(item)
        model_layout.addWidget(self.models_list)
        root.addWidget(model_group)

        # Schedule
        schedule_group = QGroupBox("Monitoring Schedule")
        schedule_layout = QVBoxLayout(schedule_group)
        sched_row = QHBoxLayout()
        self.schedule_combo = QComboBox()
        self.schedule_combo.addItems(["Continuous", "On-Demand", "Scheduled"])
        if is_edit:
            idx = self.schedule_combo.findText(self.existing_config.get("schedule_type", "Continuous"))
            if idx >= 0:
                self.schedule_combo.setCurrentIndex(idx)
        self.schedule_combo.currentTextChanged.connect(self._on_schedule_changed)
        sched_row.addWidget(QLabel("Schedule Type"))
        sched_row.addWidget(self.schedule_combo)
        sched_row.addStretch()
        schedule_layout.addLayout(sched_row)

        interval_row = QHBoxLayout()
        self.interval_hours = QSpinBox()
        self.interval_hours.setRange(0, 23)
        self.interval_hours.setSuffix(" h")
        self.interval_minutes = QSpinBox()
        self.interval_minutes.setRange(0, 59)
        self.interval_minutes.setSuffix(" m")
        self.interval_seconds = QSpinBox()
        self.interval_seconds.setRange(0, 59)
        self.interval_seconds.setSuffix(" s")
        if is_edit:
            ms = int(self.existing_config.get("interval_ms", 300_000))
            self.interval_hours.setValue(ms // 3_600_000)
            self.interval_minutes.setValue((ms // 60_000) % 60)
            self.interval_seconds.setValue((ms // 1_000) % 60)
        else:
            self.interval_minutes.setValue(5)
        interval_row.addWidget(QLabel("Interval"))
        interval_row.addWidget(self.interval_hours)
        interval_row.addWidget(self.interval_minutes)
        interval_row.addWidget(self.interval_seconds)
        interval_row.addStretch()
        self.interval_widget = QWidget()
        self.interval_widget.setLayout(interval_row)
        schedule_layout.addWidget(self.interval_widget)

        utc_row = QHBoxLayout()
        self.utc_hour = QSpinBox()
        self.utc_hour.setRange(0, 23)
        self.utc_minute = QSpinBox()
        self.utc_minute.setRange(0, 59)
        if is_edit:
            self.utc_hour.setValue(int(self.existing_config.get("schedule_utc_hour", 0)))
            self.utc_minute.setValue(int(self.existing_config.get("schedule_utc_minute", 0)))
        utc_row.addWidget(QLabel("Run at UTC"))
        utc_row.addWidget(self.utc_hour)
        utc_row.addWidget(self.utc_minute)
        utc_row.addStretch()
        self.utc_widget = QWidget()
        self.utc_widget.setLayout(utc_row)
        schedule_layout.addWidget(self.utc_widget)
        self._on_schedule_changed(self.schedule_combo.currentText())
        root.addWidget(schedule_group)

        # Ops
        ops_group = QGroupBox("Operational Settings")
        ops_layout = QFormLayout(ops_group)
        self.max_training_rows = QSpinBox()
        self.max_training_rows.setRange(1000, 2_000_000)
        self.max_training_rows.setSingleStep(10_000)
        self.max_training_rows.setValue(int(self.existing_config.get("max_training_rows", 100_000)) if is_edit else 100_000)
        self.monitoring_window = QSpinBox()
        self.monitoring_window.setRange(50, 500_000)
        self.monitoring_window.setSingleStep(100)
        self.monitoring_window.setValue(int(self.existing_config.get("monitoring_window_rows", 500)) if is_edit else 500)
        ops_layout.addRow("Max training rows", self.max_training_rows)
        ops_layout.addRow("Monitoring window rows", self.monitoring_window)
        root.addWidget(ops_group)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save Tab")
        save.setObjectName("primaryBtn")
        save.clicked.connect(self._accept)
        btn_row.addWidget(cancel)
        btn_row.addWidget(save)
        root.addLayout(btn_row)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.folder_input.setText(folder)
            self._refresh_features()

    def _refresh_features(self) -> None:
        self.features_list.clear()
        folder = self.folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        ext = ".csv" if self.file_type_combo.currentText() == "CSV" else ".json"
        files = list(Path(folder).glob(f"*{ext}"))
        if not files:
            return
        latest = max(files, key=os.path.getmtime)
        try:
            if ext == ".csv":
                sample = pd.read_csv(latest, nrows=200)
            else:
                sample = pd.read_json(latest)
        except Exception:
            return
        saved = set(self.existing_config.get("selected_features", []) if self.existing_config else [])
        use_saved = bool(saved)
        for col in sample.select_dtypes(include=[np.number]).columns:
            item = QListWidgetItem(str(col))
            item.setFlags(item.flags() | ItemIsUserCheckable)
            item.setCheckState(Checked if (col in saved if use_saved else True) else Unchecked)
            self.features_list.addItem(item)

    def _on_schedule_changed(self, schedule_type: str) -> None:
        self.interval_widget.setVisible(schedule_type == "Continuous")
        self.utc_widget.setVisible(schedule_type == "Scheduled")

    def _selected_models(self) -> list[str]:
        names = []
        for i in range(self.models_list.count()):
            item = self.models_list.item(i)
            if item.checkState() == Checked:
                names.append(item.text())
        return names

    def _selected_features(self) -> list[str]:
        features = []
        for i in range(self.features_list.count()):
            item = self.features_list.item(i)
            if item.checkState() == Checked:
                features.append(item.text())
        return features

    def _accept(self) -> None:
        title = self.title_input.text().strip()
        if not title:
            QMessageBox.warning(self, "Validation", "Enter a tab title.")
            return
        folder = self.folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Validation", "Select a valid data folder.")
            return
        models = self._selected_models()
        if not models:
            QMessageBox.warning(self, "Validation", "Select at least one ML model.")
            return

        schedule_type = self.schedule_combo.currentText()
        interval_ms = interval_ms_from_parts(
            self.interval_hours.value(),
            self.interval_minutes.value(),
            self.interval_seconds.value(),
        )
        if schedule_type == "Continuous" and interval_ms <= 0:
            QMessageBox.warning(self, "Validation", "Set a positive interval for continuous monitoring.")
            return
        if schedule_type in ("On-Demand", "Scheduled"):
            interval_ms = 0

        self.result_config = {
            "title": title,
            "data_folder": folder,
            "data_file_type": self.file_type_combo.currentText(),
            "dataset_mode": self.dataset_mode_combo.currentText(),
            "input_mode": "CSV Polling",
            "selected_features": self._selected_features(),
            "subsystem_name": self.subsystem_input.text().strip(),
            "models": [{"model_type": m, "model_parameters": {}} for m in models],
            "model_type": models[0],
            "schedule_type": schedule_type,
            "interval_ms": interval_ms,
            "schedule_utc_hour": self.utc_hour.value(),
            "schedule_utc_minute": self.utc_minute.value(),
            "max_training_rows": self.max_training_rows.value(),
            "monitoring_window_rows": self.monitoring_window.value(),
            "email_alerts_enabled": False,
            "alert_recipients": [],
        }
        self.accept()

    @staticmethod
    def get_config(parent=None, existing: dict | None = None) -> dict | None:
        dialog = TabConfigDialog(parent, existing)
        accepted_code = QDialog.DialogCode.Accepted if hasattr(QDialog, "DialogCode") else QDialog.Accepted
        if exec_dialog(dialog) != accepted_code:
            return None
        return dialog.result_config
