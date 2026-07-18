"""Tab configuration header group (read-only labels from tab config)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from PyQt5.QtWidgets import QFormLayout, QGroupBox, QLabel

if TYPE_CHECKING:
    from app.tabs.custom_tab.widget import CustomMonitoringTab


def build_header_config_group(tab: CustomMonitoringTab, models_list: List[dict]) -> QGroupBox:
    """Build header labels on *tab*; returns the group box for layout assembly."""
    header_group = QGroupBox("Tab Configuration")
    header_layout = QFormLayout()

    tab.header_tab_label = QLabel(tab.config.get("title", ""))
    tab.header_subsystem_label = QLabel(tab.config.get("subsystem_name") or "—")
    tab.header_schedule_label = QLabel(tab.config.get("schedule_type", ""))
    tab.header_input_label = QLabel(tab.config.get("input_mode", "CSV Polling"))
    tab.header_file_type_label = QLabel(tab.config.get("data_file_type", "CSV"))
    tab.header_dataset_mode_label = QLabel(tab.config.get("dataset_mode", "Full Latest File"))
    tab.header_data_source_label = QLabel(tab.config.get("data_folder", "—"))
    tab.header_data_source_label.setWordWrap(True)
    tab.header_models_count_label = QLabel(str(len(models_list)))
    current_mode = str(tab.config.get("current_mission_mode") or "nominal")
    scale = 1.0
    for mode in tab.config.get("mission_modes") or []:
        if str(mode.get("name", "")).lower() == current_mode.lower():
            try:
                scale = float(mode.get("threshold_scale", 1.0))
            except (TypeError, ValueError):
                scale = 1.0
            break
    tab.header_mission_mode_label = QLabel(f"{current_mode} (×{scale:g})")

    header_layout.addRow("Tab:", tab.header_tab_label)
    header_layout.addRow("Subsystem:", tab.header_subsystem_label)
    header_layout.addRow("Mission mode:", tab.header_mission_mode_label)
    header_layout.addRow("Schedule:", tab.header_schedule_label)
    header_layout.addRow("Input:", tab.header_input_label)
    header_layout.addRow("File type:", tab.header_file_type_label)
    header_layout.addRow("Dataset window:", tab.header_dataset_mode_label)
    header_layout.addRow("Data source:", tab.header_data_source_label)
    header_layout.addRow("Models:", tab.header_models_count_label)

    header_group.setLayout(header_layout)
    return header_group
