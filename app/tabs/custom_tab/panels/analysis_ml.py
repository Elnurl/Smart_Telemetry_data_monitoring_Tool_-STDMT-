"""Legacy panel UI (extracted from monolith)."""

from __future__ import annotations

from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from app.models.model_types import format_model_catalog_label
from app.tabs.custom_tab.panels.slots import LegacyPanelSlots


def _wire_analysis_train_button(train_button, host, slots: LegacyPanelSlots) -> None:
    """Custom tabs: canonical tab.models train; main window keeps legacy analysis path."""
    if hasattr(host, "train_from_analysis_panel"):
        train_button.clicked.connect(host.train_from_analysis_panel)
    else:
        train_button.clicked.connect(slots.analysis_slot(host, "train_model"))


def build_analysis_ml_panel(host: Any, parent_layout, *, slots: LegacyPanelSlots) -> None:
    """Build full Analysis/ML UI on host (main window or custom tab)."""
    if host is not slots.tool_window:
        slots.ensure_panel_helpers(host)
    h = host
    if not hasattr(h, "trained_models"):
        h.trained_models = {}
    if not hasattr(h, "monitoring_channels"):
        h.monitoring_channels = {}
    if not hasattr(h, "channel_processors"):
        h.channel_processors = {}
    if not hasattr(h, "model_param_widgets"):
        h.model_param_widgets = {}

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
    
    # Add multi-channel monitoring section at the top
    monitoring_group = QGroupBox("Multi-Channel Data Monitoring")
    monitoring_layout = QVBoxLayout()
    
    # Monitoring mode selection
    mode_layout = QHBoxLayout()
    h.monitoring_mode_combo = QComboBox()
    h.monitoring_mode_combo.addItems(["Single Source", "Multi-Channel"])
    h.monitoring_mode_combo.currentTextChanged.connect(slots.analysis_slot(h, 'on_monitoring_mode_changed'))
    mode_layout.addWidget(QLabel("Monitoring Mode:"))
    mode_layout.addWidget(h.monitoring_mode_combo)
    mode_layout.addStretch()
    monitoring_layout.addLayout(mode_layout)
    
    # Single source monitoring (original)
    h.single_source_widget = QWidget()
    single_source_layout = QVBoxLayout(h.single_source_widget)
    
    folder_layout = QHBoxLayout()
    h.auto_folder_input = QLineEdit()
    h.auto_folder_input.setReadOnly(True)
    browse_folder_button = QPushButton("Select Monitoring Folder...")
    browse_folder_button.clicked.connect(slots.analysis_slot(h, 'browse_auto_folder'))
    folder_layout.addWidget(QLabel("Monitor Folder:"))
    folder_layout.addWidget(h.auto_folder_input, 1)
    folder_layout.addWidget(browse_folder_button)
    single_source_layout.addLayout(folder_layout)
    
    # Multi-channel monitoring (new)
    h.multi_channel_widget = QWidget()
    multi_channel_layout = QVBoxLayout(h.multi_channel_widget)
    
    # Channel management - make buttons responsive
    channel_mgmt_layout = QHBoxLayout()
    add_channel_button = QPushButton("Add Channel")
    add_channel_button.clicked.connect(slots.analysis_slot(h, 'add_monitoring_channel'))
    remove_channel_button = QPushButton("Remove")
    remove_channel_button.clicked.connect(slots.analysis_slot(h, 'remove_monitoring_channel'))
    clear_channels_button = QPushButton("Clear All")
    clear_channels_button.clicked.connect(slots.analysis_slot(h, 'clear_monitoring_channels'))
    
    channel_mgmt_layout.addWidget(add_channel_button)
    channel_mgmt_layout.addWidget(remove_channel_button)
    channel_mgmt_layout.addWidget(clear_channels_button)
    channel_mgmt_layout.addStretch()
    multi_channel_layout.addLayout(channel_mgmt_layout)
    
    # Channels list
    h.channels_table = QTableWidget(0, 4)
    h.channels_table.setHorizontalHeaderLabels(["Channel Name", "Source Path", "Type", "Status"])
    h.channels_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    h.channels_table.setMinimumHeight(100)
    h.channels_table.setMaximumHeight(150)
    multi_channel_layout.addWidget(h.channels_table)
    
    # Initialize multi-channel data storage
    h.monitoring_channels = {}
    h.channel_processors = {}
    
    # Add both widgets to monitoring layout
    monitoring_layout.addWidget(h.single_source_widget)
    monitoring_layout.addWidget(h.multi_channel_widget)
    
    # Initially hide multi-channel widget
    h.multi_channel_widget.setVisible(False)
    
    # Auto-load controls - more compact
    auto_load_layout = QVBoxLayout()
    
    # Checkbox
    h.auto_load_check = QCheckBox("Enable Auto Loading")
    h.auto_load_check.stateChanged.connect(slots.analysis_slot(h, 'toggle_auto_loading'))
    auto_load_layout.addWidget(h.auto_load_check)
    
    # Interval controls - responsive grid
    interval_widget = QWidget()
    interval_layout = QHBoxLayout(interval_widget)
    interval_layout.setContentsMargins(0, 0, 0, 0)
    interval_layout.addWidget(QLabel("Check Interval:"))
    
    # Hours
    h.load_hours_spin = QSpinBox()
    h.load_hours_spin.setRange(0, 23)
    h.load_hours_spin.setValue(0)
    h.load_hours_spin.setSuffix(" h")
    h.load_hours_spin.setMinimumWidth(60)
    h.load_hours_spin.setMaximumWidth(80)
    interval_layout.addWidget(h.load_hours_spin)
    
    # Minutes
    h.load_minutes_spin = QSpinBox()
    h.load_minutes_spin.setRange(0, 59)
    h.load_minutes_spin.setValue(5)
    h.load_minutes_spin.setSuffix(" m")
    h.load_minutes_spin.setMinimumWidth(60)
    h.load_minutes_spin.setMaximumWidth(80)
    interval_layout.addWidget(h.load_minutes_spin)
    
    # Seconds
    h.load_seconds_spin = QSpinBox()
    h.load_seconds_spin.setRange(0, 59)
    h.load_seconds_spin.setValue(0)
    h.load_seconds_spin.setSuffix(" s")
    h.load_seconds_spin.setMinimumWidth(60)
    h.load_seconds_spin.setMaximumWidth(80)
    interval_layout.addWidget(h.load_seconds_spin)
    
    interval_layout.addStretch()
    auto_load_layout.addWidget(interval_widget)
    
    monitoring_layout.addLayout(auto_load_layout)
    monitoring_group.setLayout(monitoring_layout)
    layout.addWidget(monitoring_group)
    
    # Model selection
    model_group = QGroupBox("Anomaly Detection Model")
    model_layout = QVBoxLayout()
    
    # Multi-model selection header with Select All/Clear buttons
    model_header_layout = QHBoxLayout()
    model_header_layout.addWidget(QLabel("Select Models (multiple):"))
    
    select_all_btn = QPushButton("Select All")
    select_all_btn.setMaximumWidth(80)
    select_all_btn.clicked.connect(slots.analysis_slot(h, 'select_all_models'))
    model_header_layout.addWidget(select_all_btn)
    
    clear_all_btn = QPushButton("Clear All")
    clear_all_btn.setMaximumWidth(80)
    clear_all_btn.clicked.connect(slots.analysis_slot(h, 'clear_all_models'))
    model_header_layout.addWidget(clear_all_btn)
    
    model_header_layout.addStretch()
    model_layout.addLayout(model_header_layout)
    
    # Model selection with checkboxes (replacing dropdown)
    h.model_list_widget = QListWidget()
    h.model_list_widget.setMaximumHeight(200)
    h.model_list_widget.setSelectionMode(QAbstractItemView.NoSelection)
    
    # Add all available models as checkable items
    h.available_models = slots.get_supported_model_names()
    
    for model_name in h.available_models:
        item = QListWidgetItem(format_model_catalog_label(model_name))
        item.setData(Qt.UserRole, model_name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        h.model_list_widget.addItem(item)
    
    # Connect to update parameters when selection changes
    h.model_list_widget.itemChanged.connect(slots.analysis_slot(h, 'update_model_parameters'))
    
    model_layout.addWidget(h.model_list_widget)
    
    # Keep the combo box hidden for backward compatibility
    h.model_type_combo = QComboBox()
    h.model_type_combo.addItems(h.available_models)
    h.model_type_combo.setVisible(False)
    
    # Add current model display
    h.selected_model_label = QLabel("No models selected")
    h.selected_model_label.setWordWrap(True)
    model_layout.addWidget(h.selected_model_label)
    
    # Button to configure parameters (opens dialog)
    configure_params_button = QPushButton("Configure Parameters")
    configure_params_button.clicked.connect(slots.analysis_slot(h, 'open_parameter_dialog'))
    model_layout.addWidget(configure_params_button)
    
    # Dictionary to store parameter widgets for each model
    h.model_param_widgets = {}
    
    # Model file operations - responsive grid
    model_file_widget = QWidget()
    model_file_layout = QHBoxLayout(model_file_widget)
    model_file_layout.setContentsMargins(0, 0, 0, 0)
    
    # Train button (custom tab → tab.models via train_from_analysis_panel)
    train_button = QPushButton("Train")
    _wire_analysis_train_button(train_button, h, slots)
    train_button.setMinimumWidth(60)
    
    # Save model button
    save_model_button = QPushButton("Save")
    save_model_button.clicked.connect(slots.analysis_slot(h, 'save_model'))
    save_model_button.setMinimumWidth(60)
    
    # Load model button
    load_model_button = QPushButton("Load")
    load_model_button.clicked.connect(slots.analysis_slot(h, 'load_model'))
    load_model_button.setMinimumWidth(60)
    
    # Recent models button
    recent_models_button = QPushButton("Recent...")
    recent_models_button.clicked.connect(slots.analysis_slot(h, 'show_recent_models'))
    recent_models_button.setMinimumWidth(60)
    
    model_file_layout.addWidget(train_button)
    model_file_layout.addWidget(save_model_button)
    model_file_layout.addWidget(load_model_button)
    model_file_layout.addWidget(recent_models_button)
    model_file_layout.addStretch()
    
    model_layout.addWidget(model_file_widget)
    
    model_group.setLayout(model_layout)
    layout.addWidget(model_group)
    
    # Prediction section
    prediction_group = QGroupBox("Prediction")
    prediction_layout = QVBoxLayout()
    
    # Prediction options - responsive
    prediction_options_layout = QHBoxLayout()
    h.prediction_file_input = QLineEdit()
    h.prediction_file_input.setReadOnly(True)
    h.prediction_file_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    browse_prediction_button = QPushButton("Browse...")
    browse_prediction_button.clicked.connect(slots.analysis_slot(h, 'browse_prediction_file'))
    browse_prediction_button.setMinimumWidth(80)
    
    prediction_options_layout.addWidget(QLabel("File:"))
    prediction_options_layout.addWidget(h.prediction_file_input, 1)
    prediction_options_layout.addWidget(browse_prediction_button)
    prediction_layout.addLayout(prediction_options_layout)
    
    # Button layout
    button_layout = QHBoxLayout()
    
    # Predict button for new data
    predict_button = QPushButton("Test on New Data")
    predict_button.clicked.connect(slots.analysis_slot(h, 'predict_anomalies'))
    predict_button.setMinimumWidth(100)
    button_layout.addWidget(predict_button)
    
    # Test on training data button
    test_training_button = QPushButton("Test on Training Data")
    test_training_button.clicked.connect(slots.analysis_slot(h, 'test_on_training_data'))
    test_training_button.setMinimumWidth(100)
    test_training_button.setToolTip("Test models on the same data they were trained on")
    button_layout.addWidget(test_training_button)
    
    prediction_layout.addLayout(button_layout)
    
    prediction_group.setLayout(prediction_layout)
    layout.addWidget(prediction_group)
    
    # Add auto processing controls
    auto_process_group = QGroupBox("Automatic Analysis")
    auto_process_layout = QVBoxLayout()
    
    # Checkbox
    h.auto_process_check = QCheckBox("Enable Auto Processing")
    h.auto_process_check.stateChanged.connect(slots.analysis_slot(h, 'toggle_auto_processing'))
    auto_process_layout.addWidget(h.auto_process_check)
    
    # Process interval controls - responsive
    interval_widget = QWidget()
    process_interval_layout = QHBoxLayout(interval_widget)
    process_interval_layout.setContentsMargins(0, 0, 0, 0)
    process_interval_layout.addWidget(QLabel("Interval:"))
    
    # Hours
    h.process_hours_spin = QSpinBox()
    h.process_hours_spin.setRange(0, 23)
    h.process_hours_spin.setValue(0)
    h.process_hours_spin.setSuffix(" h")
    h.process_hours_spin.setMinimumWidth(60)
    h.process_hours_spin.setMaximumWidth(80)
    process_interval_layout.addWidget(h.process_hours_spin)
    
    # Minutes
    h.process_minutes_spin = QSpinBox()
    h.process_minutes_spin.setRange(0, 59)
    h.process_minutes_spin.setValue(10)
    h.process_minutes_spin.setSuffix(" m")
    h.process_minutes_spin.setMinimumWidth(60)
    h.process_minutes_spin.setMaximumWidth(80)
    process_interval_layout.addWidget(h.process_minutes_spin)
    
    # Seconds
    h.process_seconds_spin = QSpinBox()
    h.process_seconds_spin.setRange(0, 59)
    h.process_seconds_spin.setValue(0)
    h.process_seconds_spin.setSuffix(" s")
    h.process_seconds_spin.setMinimumWidth(60)
    h.process_seconds_spin.setMaximumWidth(80)
    process_interval_layout.addWidget(h.process_seconds_spin)
    
    process_interval_layout.addStretch()
    auto_process_layout.addWidget(interval_widget)
    
    auto_process_group.setLayout(auto_process_layout)
    layout.addWidget(auto_process_group)
    
    # Status message
    h.analysis_status_label = QLabel()
    h.analysis_status_label.setWordWrap(True)
    layout.addWidget(h.analysis_status_label)

    # Live Keras / training stdout (same bars as the IDE terminal)
    console_group = QGroupBox("Training Console")
    console_layout = QVBoxLayout(console_group)
    console_layout.setContentsMargins(6, 6, 6, 6)

    h.training_console = QPlainTextEdit()
    h.training_console.setReadOnly(True)
    h.training_console.setMinimumHeight(140)
    h.training_console.setMaximumHeight(180)
    h.training_console.setPlaceholderText("Keras/training output appears here…")
    mono = QFont("Consolas")
    mono.setStyleHint(QFont.Monospace)
    mono.setPointSize(9)
    h.training_console.setFont(mono)
    h.training_console.setStyleSheet(
        "QPlainTextEdit {"
        " background-color: #1e1e1e;"
        " color: #d4d4d4;"
        " border: 1px solid #3c3c3c;"
        "}"
    )
    console_layout.addWidget(h.training_console)

    console_btn_row = QHBoxLayout()
    clear_console_btn = QPushButton("Clear")
    clear_console_btn.setMaximumWidth(80)
    clear_console_btn.clicked.connect(h.training_console.clear)
    console_btn_row.addWidget(clear_console_btn)
    console_btn_row.addStretch()
    console_layout.addLayout(console_btn_row)
    layout.addWidget(console_group)
    
    # Add buttons for metrics and logs - responsive
    buttons_widget = QWidget()
    buttons_layout = QHBoxLayout(buttons_widget)
    buttons_layout.setContentsMargins(0, 0, 0, 0)
    
    show_metrics_btn = QPushButton("Model Metrics")
    show_metrics_btn.clicked.connect(slots.analysis_slot(h, 'show_model_metrics'))
    show_metrics_btn.setMinimumWidth(80)
    
    show_log_btn = QPushButton("Action Log")
    show_log_btn.clicked.connect(slots.analysis_slot(h, 'show_action_log'))
    show_log_btn.setMinimumWidth(80)
    
    buttons_layout.addWidget(show_metrics_btn)
    buttons_layout.addWidget(show_log_btn)
    buttons_layout.addStretch()
    
    layout.addWidget(buttons_widget)
    
    # Add stretch at the end to push content to the top
    layout.addStretch()
    
    # Set container widget and add to scroll area
    scroll_area.setWidget(container_widget)
    panel_layout.addWidget(scroll_area)
    parent_layout.addWidget(wrapper)

    slots.invoke_analysis_method(h, "update_model_parameters")
    
