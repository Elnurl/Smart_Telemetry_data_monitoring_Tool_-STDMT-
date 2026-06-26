"""Legacy panel UI (extracted from monolith)."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from app.tabs.custom_tab.panels.slots import LegacyPanelSlots

def build_data_import_panel(host: Any, parent_layout, *, slots: LegacyPanelSlots) -> None:
    """Build full Data Import UI on host (main window or custom tab)."""
    if host is not slots.tool_window:
        slots.ensure_panel_helpers(host)
    h = host
    wrapper = QWidget()
    panel_layout = QVBoxLayout(wrapper)
    panel_layout.setContentsMargins(0, 0, 0, 0)

    # Create scroll area
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    # Container widget for scroll area
    container_widget = QWidget()
    layout = QVBoxLayout(container_widget)
    layout.setSpacing(10)
    layout.setContentsMargins(10, 10, 10, 10)
    
    # Add monitoring control buttons at the top
    monitoring_group = QGroupBox("Monitoring Controls")
    monitoring_widget = QWidget()
    monitoring_layout = QHBoxLayout(monitoring_widget)
    monitoring_layout.setContentsMargins(0, 0, 0, 0)
    
    h.stop_monitoring_btn = QPushButton("Stop Monitoring and Exit")
    h.stop_monitoring_btn.clicked.connect(slots.data_import_stop_slot(h))
    h.stop_monitoring_btn.setEnabled(False)
    h.stop_monitoring_btn.setMinimumWidth(150)
    
    h.clean_data_btn = QPushButton("Clean Data")
    h.clean_data_btn.clicked.connect(slots.data_slot(h, 'clean_data'))
    h.clean_data_btn.setMinimumWidth(100)
    
    monitoring_layout.addWidget(h.stop_monitoring_btn)
    monitoring_layout.addWidget(h.clean_data_btn)
    monitoring_layout.addStretch()
    
    monitoring_group_layout = QVBoxLayout()
    monitoring_group_layout.addWidget(monitoring_widget)
    monitoring_group.setLayout(monitoring_group_layout)
    layout.addWidget(monitoring_group)
    
    # Data source section
    source_group = QGroupBox("Data Source")
    source_layout = QVBoxLayout()
    
    # File selection - responsive
    file_layout = QHBoxLayout()
    h.file_path_input = QLineEdit()
    h.file_path_input.setReadOnly(True)
    h.file_path_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    browse_button = QPushButton("Browse...")
    browse_button.clicked.connect(slots.data_slot(h, 'browse_data_file'))
    browse_button.setMinimumWidth(80)
    file_layout.addWidget(QLabel("File:"))
    file_layout.addWidget(h.file_path_input, 1)
    file_layout.addWidget(browse_button)
    source_layout.addLayout(file_layout)
    
    # File type selection
    type_layout = QHBoxLayout()
    h.file_type_combo = QComboBox()
    h.file_type_combo.addItems(["CSV", "JSON"])
    h.file_type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    h.file_type_combo.setMinimumWidth(100)
    type_layout.addWidget(QLabel("Type:"))
    type_layout.addWidget(h.file_type_combo)
    type_layout.addStretch(1)
    source_layout.addLayout(type_layout)
    
    # Load button
    load_button = QPushButton("Load Data")
    load_button.clicked.connect(slots.data_slot(h, 'load_data'))
    load_button.setMinimumWidth(100)
    source_layout.addWidget(load_button)
    
    source_group.setLayout(source_layout)
    layout.addWidget(source_group)
    
    # Data preview section
    preview_group = QGroupBox("Data Preview")
    preview_layout = QVBoxLayout()
    
    h.data_table = QTableWidget(0, 0)
    h.data_table.setMinimumHeight(150)
    h.data_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    preview_layout.addWidget(h.data_table)
    
    preview_group.setLayout(preview_layout)
    layout.addWidget(preview_group, 1)
    
    # Preprocessing options
    preprocess_group = QGroupBox("Advanced Preprocessing Options")
    preprocess_layout = QVBoxLayout()
    
    # Legacy feature list (kept for compatibility)
    h.feature_list = QTableWidget(0, 2)
    h.feature_list.setHorizontalHeaderLabels(["Column", "Use"])
    h.feature_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    h.feature_list.hide()  # Hide legacy widget but keep for compatibility
    
    # Normalization and outlier options
    options_layout = QHBoxLayout()
    h.normalize_check = QCheckBox("Normalize Data")
    h.remove_outliers_check = QCheckBox("Remove Extreme Outliers")
    options_layout.addWidget(h.normalize_check)
    options_layout.addWidget(h.remove_outliers_check)
    options_layout.addStretch()
    preprocess_layout.addLayout(options_layout)
    
    # Preprocess button
    preprocess_button = QPushButton("Preprocess Data")
    preprocess_button.clicked.connect(slots.data_slot(h, 'preprocess_data'))
    preprocess_button.setMinimumWidth(120)
    preprocess_layout.addWidget(preprocess_button)
    
    preprocess_group.setLayout(preprocess_layout)
    layout.addWidget(preprocess_group)
    
    # Add basic column configuration (minimal required elements)
    basic_config_group = QGroupBox("Basic Configuration")
    basic_config_layout = QVBoxLayout()
    
    # Timestamp column selection
    timestamp_layout = QHBoxLayout()
    timestamp_layout.addWidget(QLabel("Timestamp Column:"))
    h.timestamp_combo = QComboBox()
    h.timestamp_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    h.timestamp_combo.setMinimumWidth(100)
    timestamp_layout.addWidget(h.timestamp_combo)
    timestamp_layout.addStretch()
    basic_config_layout.addLayout(timestamp_layout)
    
    # Dataset info label
    h.column_stats_label = QLabel("Dataset Info: No data loaded")
    h.column_stats_label.setWordWrap(True)
    basic_config_layout.addWidget(h.column_stats_label)
    
    basic_config_group.setLayout(basic_config_layout)
    layout.addWidget(basic_config_group)
    
    # Status message
    h.data_status_label = QLabel()
    h.data_status_label.setWordWrap(True)
    layout.addWidget(h.data_status_label)
    
    # Add stretch at the end
    layout.addStretch()
    
    # Set container widget and add to scroll area
    scroll_area.setWidget(container_widget)
    panel_layout.addWidget(scroll_area)
    parent_layout.addWidget(wrapper)
    
