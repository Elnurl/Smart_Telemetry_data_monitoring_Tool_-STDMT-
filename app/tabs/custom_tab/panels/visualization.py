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
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from app.tabs.custom_tab.panels.slots import LegacyPanelSlots

def build_visualization_panel(host: Any, parent_layout, *, slots: LegacyPanelSlots) -> None:
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
    
    # Main visualization options
    main_options_group = QGroupBox("Data Visualization")
    main_options_layout = QVBoxLayout()
    
    # First row: Feature and visualization type selection - responsive
    selection_layout = QHBoxLayout()
    
    # Feature selection
    h.feature_combo = QComboBox()
    h.feature_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    h.feature_combo.setMinimumWidth(100)
    selection_layout.addWidget(QLabel("Feature:"))
    selection_layout.addWidget(h.feature_combo, 1)
    
    # Visualization type selection  
    h.viz_type_combo = QComboBox()
    h.viz_type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    h.viz_type_combo.setMinimumWidth(100)
    h.viz_type_combo.addItems([
        "Time Series with Anomalies", 
        "Anomaly Score Distribution", 
        "Feature Correlation Heatmap",
        "Statistical Summary",
        "Anomaly Timeline"
    ])
    selection_layout.addWidget(QLabel("Type:"))
    selection_layout.addWidget(h.viz_type_combo, 1)
    
    main_options_layout.addLayout(selection_layout)
    
    # Second row: Plot customization options - responsive
    customization_layout = QHBoxLayout()
    
    # Time range selection
    h.time_range_combo = QComboBox()
    h.time_range_combo.addItems(["All Data", "Last 100 Points", "Last 500 Points", "Last 1000 Points", "Custom Range"])
    h.time_range_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    h.time_range_combo.setMinimumWidth(100)
    customization_layout.addWidget(QLabel("Range:"))
    customization_layout.addWidget(h.time_range_combo)
    
    # Anomaly highlighting options
    h.highlight_anomalies_check = QCheckBox("Highlight Anomalies")
    h.highlight_anomalies_check.setChecked(True)
    customization_layout.addWidget(h.highlight_anomalies_check)
    
    # Smooth lines option
    h.smooth_lines_check = QCheckBox("Smooth Lines")
    h.smooth_lines_check.setChecked(False)
    customization_layout.addWidget(h.smooth_lines_check)
    
    customization_layout.addStretch()
    main_options_layout.addLayout(customization_layout)
    
    # Third row: Plot controls - responsive
    controls_widget = QWidget()
    controls_layout = QHBoxLayout(controls_widget)
    controls_layout.setContentsMargins(0, 0, 0, 0)
    
    # Plot button
    plot_button = QPushButton("Generate")
    plot_button.clicked.connect(slots.viz_slot(h, 'plot_feature'))
    plot_button.setMinimumWidth(80)
    controls_layout.addWidget(plot_button)
    
    # Clear plot button
    clear_button = QPushButton("Clear")
    clear_button.clicked.connect(slots.viz_slot(h, 'clear_plot'))
    clear_button.setMinimumWidth(80)
    controls_layout.addWidget(clear_button)
    
    # Export plot button
    export_button = QPushButton("Export")
    export_button.clicked.connect(slots.viz_slot(h, 'export_plot'))
    export_button.setMinimumWidth(80)
    controls_layout.addWidget(export_button)
    
    controls_layout.addStretch()
    main_options_layout.addWidget(controls_widget)
    
    main_options_group.setLayout(main_options_layout)
    layout.addWidget(main_options_group)
    
    # Multi-Model Comparison Section
    multi_model_group = QGroupBox("Multi-Model Comparison")
    multi_model_layout = QVBoxLayout()
    
    # Model info label
    h.viz_model_info_label = QLabel("No models trained yet")
    h.viz_model_info_label.setWordWrap(True)
    h.viz_model_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
    multi_model_layout.addWidget(h.viz_model_info_label)
    
    # Comparison visualization options
    comparison_layout = QHBoxLayout()
    
    h.viz_comparison_type = QComboBox()
    h.viz_comparison_type.addItems([
        "Model Detection Counts",
        "Model Agreement Matrix",
        "Detection Overlap Venn",
        "Individual Model Predictions",
        "Score Distributions",
        "Confidence Comparison",
        "Performance Metrics"
    ])
    comparison_layout.addWidget(QLabel("Comparison Type:"))
    comparison_layout.addWidget(h.viz_comparison_type, 1)
    
    compare_button = QPushButton("Compare Models")
    compare_button.clicked.connect(slots.viz_slot(h, 'plot_model_comparison'))
    comparison_layout.addWidget(compare_button)
    
    multi_model_layout.addLayout(comparison_layout)
    multi_model_group.setLayout(multi_model_layout)
    layout.addWidget(multi_model_group)
    
    # Enhanced plot area with toolbar
    plot_group = QGroupBox("Visualization")
    plot_layout = QVBoxLayout()
    
    # Create matplotlib canvas with navigation toolbar
    h.plot_canvas = slots.mpl_canvas_cls(h, width=12, height=8, dpi=100)
    
    # Add navigation toolbar for zoom, pan, etc.
    try:
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    except ImportError:
        from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
    h.plot_toolbar = NavigationToolbar(h.plot_canvas, h)
    
    plot_layout.addWidget(h.plot_toolbar)
    plot_layout.addWidget(h.plot_canvas)
    
    plot_group.setLayout(plot_layout)
    layout.addWidget(plot_group)
    
    # Add metrics visualization section
    metrics_group = QGroupBox("Model Metrics Visualization")
    metrics_layout = QVBoxLayout()
    
    # Add metric type selection - responsive
    metric_type_layout = QHBoxLayout()
    h.metric_type_combo = QComboBox()
    h.metric_type_combo.addItems(["Error Metrics", "Model Performance", "Anomaly Distribution"])
    h.metric_type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    h.metric_type_combo.setMinimumWidth(100)
    metric_type_layout.addWidget(QLabel("Metric:"))
    metric_type_layout.addWidget(h.metric_type_combo)
    metric_type_layout.addStretch()
    metrics_layout.addLayout(metric_type_layout)
    
    # Add plot type selection - responsive
    plot_type_layout = QHBoxLayout()
    h.plot_type_combo = QComboBox()
    h.plot_type_combo.addItems(["Line Plot", "Bar Chart", "Box Plot", "Histogram", 
                                    "3D Surface", "3D Waterfall"])
    h.plot_type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    h.plot_type_combo.setMinimumWidth(100)
    plot_type_layout.addWidget(QLabel("Plot:"))
    plot_type_layout.addWidget(h.plot_type_combo)
    plot_type_layout.addStretch()
    metrics_layout.addLayout(plot_type_layout)
    
    # Add plot metrics button
    plot_metrics_button = QPushButton("Plot Metrics")
    plot_metrics_button.clicked.connect(slots.viz_slot(h, 'plot_metrics'))
    plot_metrics_button.setMinimumWidth(100)
    metrics_layout.addWidget(plot_metrics_button)
    
    metrics_group.setLayout(metrics_layout)
    layout.addWidget(metrics_group)
    
    # Status message
    h.visualization_status_label = QLabel()
    h.visualization_status_label.setWordWrap(True)
    layout.addWidget(h.visualization_status_label)
    
    # Add stretch at the end
    layout.addStretch()
    
    # Set container widget and add to scroll area
    scroll_area.setWidget(container_widget)
    panel_layout.addWidget(scroll_area)
    parent_layout.addWidget(wrapper)
    

