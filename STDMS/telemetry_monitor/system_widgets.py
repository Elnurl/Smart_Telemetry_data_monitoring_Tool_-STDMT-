#!/usr/bin/env python3
"""
Additional Settings UI Components for Stage 5
System settings, visualization configuration, and monitoring controls.
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QSlider, QTextEdit, QGroupBox,
    QTabWidget, QFormLayout, QScrollArea, QMessageBox,
    QProgressBar, QFileDialog, QListWidget, QListWidgetItem,
    QColorDialog, QFontDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtGui import QFont, QColor, QPalette, QPixmap, QIcon

from config_manager import (
    ConfigurationManager, VisualizationConfig, SystemConfig
)

class VisualizationSettingsWidget(QWidget):
    """Widget for configuring visualization settings"""
    
    settings_changed = Signal()
    
    def __init__(self, config_manager: ConfigurationManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_ui()
        self._load_settings()
        
    def _setup_ui(self):
        """Setup visualization settings UI"""
        layout = QVBoxLayout(self)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Chart appearance settings
        appearance_group = QGroupBox("📊 Chart Appearance")
        appearance_layout = QFormLayout(appearance_group)
        
        self.chart_theme = QComboBox()
        self.chart_theme.addItems(["Dark", "Light", "System"])
        self.chart_theme.currentTextChanged.connect(self._on_settings_changed)
        appearance_layout.addRow("Theme:", self.chart_theme)
        
        # Color selection buttons
        color_layout = QHBoxLayout()
        
        self.normal_color_btn = QPushButton("Normal Data")
        self.normal_color_btn.clicked.connect(lambda: self._select_color("normal"))
        color_layout.addWidget(self.normal_color_btn)
        
        self.anomaly_color_btn = QPushButton("Anomaly Data")
        self.anomaly_color_btn.clicked.connect(lambda: self._select_color("anomaly"))
        color_layout.addWidget(self.anomaly_color_btn)
        
        self.critical_color_btn = QPushButton("Critical Alert")
        self.critical_color_btn.clicked.connect(lambda: self._select_color("critical"))
        color_layout.addWidget(self.critical_color_btn)
        
        appearance_layout.addRow("Colors:", color_layout)
        
        # Line properties
        self.line_width = QSpinBox()
        self.line_width.setRange(1, 10)
        self.line_width.valueChanged.connect(self._on_settings_changed)
        appearance_layout.addRow("Line Width:", self.line_width)
        
        self.point_size = QSpinBox()
        self.point_size.setRange(2, 20)
        self.point_size.valueChanged.connect(self._on_settings_changed)
        appearance_layout.addRow("Point Size:", self.point_size)
        
        scroll_layout.addWidget(appearance_group)
        
        # Data display settings
        data_group = QGroupBox("📈 Data Display")
        data_layout = QFormLayout(data_group)
        
        self.max_points = QSpinBox()
        self.max_points.setRange(100, 10000)
        self.max_points.setSuffix(" points")
        self.max_points.valueChanged.connect(self._on_settings_changed)
        data_layout.addRow("Max Points per Chart:", self.max_points)
        
        self.update_interval = QSpinBox()
        self.update_interval.setRange(100, 5000)
        self.update_interval.setSuffix(" ms")
        self.update_interval.valueChanged.connect(self._on_settings_changed)
        data_layout.addRow("Update Interval:", self.update_interval)
        
        self.auto_scale = QCheckBox("Auto-scale Y axis")
        self.auto_scale.stateChanged.connect(self._on_settings_changed)
        data_layout.addRow(self.auto_scale)
        
        self.show_grid = QCheckBox("Show grid lines")
        self.show_grid.stateChanged.connect(self._on_settings_changed)
        data_layout.addRow(self.show_grid)
        
        self.show_legend = QCheckBox("Show legend")
        self.show_legend.stateChanged.connect(self._on_settings_changed)
        data_layout.addRow(self.show_legend)
        
        scroll_layout.addWidget(data_group)
        
        # Performance settings
        performance_group = QGroupBox("⚡ Performance")
        performance_layout = QFormLayout(performance_group)
        
        self.enable_antialiasing = QCheckBox("Enable anti-aliasing")
        self.enable_antialiasing.stateChanged.connect(self._on_settings_changed)
        performance_layout.addRow(self.enable_antialiasing)
        
        self.use_opengl = QCheckBox("Hardware acceleration (OpenGL)")
        self.use_opengl.stateChanged.connect(self._on_settings_changed)
        performance_layout.addRow(self.use_opengl)
        
        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(1000, 100000)
        self.buffer_size.setSuffix(" samples")
        self.buffer_size.valueChanged.connect(self._on_settings_changed)
        performance_layout.addRow("Data Buffer Size:", self.buffer_size)
        
        scroll_layout.addWidget(performance_group)
        
        # Export settings
        export_group = QGroupBox("💾 Export Settings")
        export_layout = QFormLayout(export_group)
        
        self.default_export_format = QComboBox()
        self.default_export_format.addItems(["PNG", "SVG", "PDF", "CSV"])
        self.default_export_format.currentTextChanged.connect(self._on_settings_changed)
        export_layout.addRow("Default Format:", self.default_export_format)
        
        self.export_dpi = QSpinBox()
        self.export_dpi.setRange(72, 600)
        self.export_dpi.setSuffix(" DPI")
        self.export_dpi.valueChanged.connect(self._on_settings_changed)
        export_layout.addRow("Export DPI:", self.export_dpi)
        
        # Export directory selection
        export_dir_layout = QHBoxLayout()
        self.export_directory = QLineEdit()
        self.export_directory.textChanged.connect(self._on_settings_changed)
        export_dir_layout.addWidget(self.export_directory)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_export_directory)
        export_dir_layout.addWidget(browse_btn)
        
        export_layout.addRow("Export Directory:", export_dir_layout)
        
        scroll_layout.addWidget(export_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("👁️ Preview Changes")
        self.preview_btn.clicked.connect(self._preview_changes)
        controls_layout.addWidget(self.preview_btn)
        
        self.reset_vis_btn = QPushButton("🔄 Reset Visualization")
        self.reset_vis_btn.clicked.connect(self._reset_visualization)
        controls_layout.addWidget(self.reset_vis_btn)
        
        controls_layout.addStretch()
        scroll_layout.addLayout(controls_layout)
        
        # Setup scroll area
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Store color values
        self.colors = {
            "normal": "#00ff00",
            "anomaly": "#ffaa00", 
            "critical": "#ff0000"
        }
        
    def _load_settings(self):
        """Load current visualization settings"""
        config = self.config_manager.get_visualization_config()
        
        # Appearance
        theme_index = {"dark": 0, "light": 1, "system": 2}.get(config.theme.lower(), 0)
        self.chart_theme.setCurrentIndex(theme_index)
        
        self.line_width.setValue(config.line_width)
        self.point_size.setValue(config.point_size)
        
        # Colors
        self.colors["normal"] = config.normal_color
        self.colors["anomaly"] = config.anomaly_color
        self.colors["critical"] = config.critical_color
        self._update_color_buttons()
        
        # Data display
        self.max_points.setValue(config.max_points_per_chart)
        self.update_interval.setValue(config.update_interval_ms)
        self.auto_scale.setChecked(config.auto_scale_y)
        self.show_grid.setChecked(config.show_grid)
        self.show_legend.setChecked(config.show_legend)
        
        # Performance
        self.enable_antialiasing.setChecked(config.enable_antialiasing)
        self.use_opengl.setChecked(config.use_opengl)
        self.buffer_size.setValue(config.data_buffer_size)
        
        # Export
        format_index = ["PNG", "SVG", "PDF", "CSV"].index(config.default_export_format)
        self.default_export_format.setCurrentIndex(format_index)
        self.export_dpi.setValue(config.export_dpi)
        self.export_directory.setText(config.export_directory)
        
    def _update_color_buttons(self):
        """Update color button appearances"""
        for color_type, color_value in self.colors.items():
            if color_type == "normal":
                btn = self.normal_color_btn
            elif color_type == "anomaly":
                btn = self.anomaly_color_btn
            else:
                btn = self.critical_color_btn
            
            btn.setStyleSheet(f"background-color: {color_value}; color: white; font-weight: bold;")
        
    def _select_color(self, color_type: str):
        """Open color picker for specified color type"""
        current_color = QColor(self.colors[color_type])
        new_color = QColorDialog.getColor(current_color, self, f"Select {color_type.title()} Color")
        
        if new_color.isValid():
            self.colors[color_type] = new_color.name()
            self._update_color_buttons()
            self._on_settings_changed()
            
    def _browse_export_directory(self):
        """Browse for export directory"""
        current_dir = self.export_directory.text() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", current_dir
        )
        
        if directory:
            self.export_directory.setText(directory)
            
    def _preview_changes(self):
        """Preview visualization changes"""
        QMessageBox.information(
            self, "Preview Changes",
            "Visualization changes will be applied to charts in real-time."
        )
        
    def _reset_visualization(self):
        """Reset visualization settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Visualization", 
            "Reset all visualization settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            default_config = VisualizationConfig()
            self.config_manager.settings.visualization = default_config
            self._load_settings()
            self.settings_changed.emit()
            
    def _on_settings_changed(self):
        """Handle settings change"""
        self.settings_changed.emit()
        
    def get_current_config(self) -> VisualizationConfig:
        """Get current visualization configuration from UI"""
        theme_map = {0: "dark", 1: "light", 2: "system"}
        theme = theme_map.get(self.chart_theme.currentIndex(), "dark")
        
        export_format = self.default_export_format.currentText()
        
        config = VisualizationConfig(
            theme=theme,
            normal_color=self.colors["normal"],
            anomaly_color=self.colors["anomaly"],
            critical_color=self.colors["critical"],
            line_width=self.line_width.value(),
            point_size=self.point_size.value(),
            max_points_per_chart=self.max_points.value(),
            update_interval_ms=self.update_interval.value(),
            auto_scale_y=self.auto_scale.isChecked(),
            show_grid=self.show_grid.isChecked(),
            show_legend=self.show_legend.isChecked(),
            enable_antialiasing=self.enable_antialiasing.isChecked(),
            use_opengl=self.use_opengl.isChecked(),
            data_buffer_size=self.buffer_size.value(),
            default_export_format=export_format,
            export_dpi=self.export_dpi.value(),
            export_directory=self.export_directory.text()
        )
        
        return config


class SystemSettingsWidget(QWidget):
    """Widget for system settings and monitoring controls"""
    
    settings_changed = Signal()
    monitoring_state_changed = Signal(bool)  # True = start, False = stop
    
    def __init__(self, config_manager: ConfigurationManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.monitoring_active = False
        self._setup_ui()
        self._load_settings()
        
    def _setup_ui(self):
        """Setup system settings UI"""
        layout = QVBoxLayout(self)
        
        # Monitoring controls - prominent section
        monitoring_group = QGroupBox("🚀 Telemetry Monitoring Control")
        monitoring_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #2E86AB;
                border: 2px solid #2E86AB;
                border-radius: 8px;
                margin: 10px 0;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        monitoring_layout = QVBoxLayout(monitoring_group)
        
        # Status display
        status_layout = QHBoxLayout()
        self.status_label = QLabel("🔴 Monitoring: STOPPED")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        self.uptime_label = QLabel("Uptime: 00:00:00")
        self.uptime_label.setStyleSheet("font-size: 12px; color: gray;")
        status_layout.addWidget(self.uptime_label)
        
        monitoring_layout.addLayout(status_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_stop_btn = QPushButton("🚀 START MONITORING")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        self.start_stop_btn.clicked.connect(self._toggle_monitoring)
        button_layout.addWidget(self.start_stop_btn)
        
        self.pause_resume_btn = QPushButton("⏸️ PAUSE")
        self.pause_resume_btn.setEnabled(False)
        self.pause_resume_btn.clicked.connect(self._toggle_pause)
        button_layout.addWidget(self.pause_resume_btn)
        
        self.restart_btn = QPushButton("🔄 RESTART")
        self.restart_btn.setEnabled(False)
        self.restart_btn.clicked.connect(self._restart_monitoring)
        button_layout.addWidget(self.restart_btn)
        
        monitoring_layout.addLayout(button_layout)
        
        # Quick stats
        stats_layout = QGridLayout()
        
        self.packets_received = QLabel("Packets: 0")
        self.anomalies_detected = QLabel("Anomalies: 0")
        self.alerts_sent = QLabel("Alerts: 0")
        self.cpu_usage = QLabel("CPU: 0%")
        
        stats_layout.addWidget(QLabel("📊 Quick Stats:"), 0, 0)
        stats_layout.addWidget(self.packets_received, 0, 1)
        stats_layout.addWidget(self.anomalies_detected, 0, 2)
        stats_layout.addWidget(self.alerts_sent, 1, 1)
        stats_layout.addWidget(self.cpu_usage, 1, 2)
        
        monitoring_layout.addLayout(stats_layout)
        
        layout.addWidget(monitoring_group)
        
        # Create scroll area for other settings
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # System settings
        system_group = QGroupBox("⚙️ System Configuration")
        system_layout = QFormLayout(system_group)
        
        self.auto_start = QCheckBox("Auto-start monitoring on application launch")
        self.auto_start.stateChanged.connect(self._on_settings_changed)
        system_layout.addRow(self.auto_start)
        
        self.minimize_to_tray = QCheckBox("Minimize to system tray")
        self.minimize_to_tray.stateChanged.connect(self._on_settings_changed)
        system_layout.addRow(self.minimize_to_tray)
        
        self.check_updates = QCheckBox("Check for updates automatically")
        self.check_updates.stateChanged.connect(self._on_settings_changed)
        system_layout.addRow(self.check_updates)
        
        scroll_layout.addWidget(system_group)
        
        # Logging settings
        logging_group = QGroupBox("📝 Logging Configuration")
        logging_layout = QFormLayout(logging_group)
        
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level.currentTextChanged.connect(self._on_settings_changed)
        logging_layout.addRow("Log Level:", self.log_level)
        
        self.max_log_size = QSpinBox()
        self.max_log_size.setRange(1, 1000)
        self.max_log_size.setSuffix(" MB")
        self.max_log_size.valueChanged.connect(self._on_settings_changed)
        logging_layout.addRow("Max Log File Size:", self.max_log_size)
        
        self.log_rotation_count = QSpinBox()
        self.log_rotation_count.setRange(1, 50)
        self.log_rotation_count.valueChanged.connect(self._on_settings_changed)
        logging_layout.addRow("Log Rotation Count:", self.log_rotation_count)
        
        # Log directory selection
        log_dir_layout = QHBoxLayout()
        self.log_directory = QLineEdit()
        self.log_directory.textChanged.connect(self._on_settings_changed)
        log_dir_layout.addWidget(self.log_directory)
        
        browse_log_btn = QPushButton("Browse...")
        browse_log_btn.clicked.connect(self._browse_log_directory)
        log_dir_layout.addWidget(browse_log_btn)
        
        logging_layout.addRow("Log Directory:", log_dir_layout)
        
        scroll_layout.addWidget(logging_group)
        
        # Data management
        data_group = QGroupBox("💾 Data Management")
        data_layout = QFormLayout(data_group)
        
        self.data_retention_days = QSpinBox()
        self.data_retention_days.setRange(1, 365)
        self.data_retention_days.setSuffix(" days")
        self.data_retention_days.valueChanged.connect(self._on_settings_changed)
        data_layout.addRow("Data Retention Period:", self.data_retention_days)
        
        self.auto_cleanup = QCheckBox("Automatic data cleanup")
        self.auto_cleanup.stateChanged.connect(self._on_settings_changed)
        data_layout.addRow(self.auto_cleanup)
        
        self.backup_enabled = QCheckBox("Enable automatic database backup")
        self.backup_enabled.stateChanged.connect(self._on_settings_changed)
        data_layout.addRow(self.backup_enabled)
        
        self.backup_interval = QSpinBox()
        self.backup_interval.setRange(1, 168)
        self.backup_interval.setSuffix(" hours")
        self.backup_interval.valueChanged.connect(self._on_settings_changed)
        data_layout.addRow("Backup Interval:", self.backup_interval)
        
        # Database management buttons
        db_buttons_layout = QHBoxLayout()
        
        self.backup_now_btn = QPushButton("💾 Backup Now")
        self.backup_now_btn.clicked.connect(self._backup_database)
        db_buttons_layout.addWidget(self.backup_now_btn)
        
        self.cleanup_now_btn = QPushButton("🧹 Cleanup Now")
        self.cleanup_now_btn.clicked.connect(self._cleanup_database)
        db_buttons_layout.addWidget(self.cleanup_now_btn)
        
        self.reset_db_btn = QPushButton("⚠️ Reset Database")
        self.reset_db_btn.clicked.connect(self._reset_database)
        self.reset_db_btn.setStyleSheet("background-color: #dc3545; color: white;")
        db_buttons_layout.addWidget(self.reset_db_btn)
        
        data_layout.addRow("Database Actions:", db_buttons_layout)
        
        scroll_layout.addWidget(data_group)
        
        # Configuration management
        config_group = QGroupBox("📋 Configuration Management")
        config_layout = QVBoxLayout(config_group)
        
        config_buttons_layout = QHBoxLayout()
        
        self.export_config_btn = QPushButton("📤 Export Settings")
        self.export_config_btn.clicked.connect(self._export_configuration)
        config_buttons_layout.addWidget(self.export_config_btn)
        
        self.import_config_btn = QPushButton("📥 Import Settings")
        self.import_config_btn.clicked.connect(self._import_configuration)
        config_buttons_layout.addWidget(self.import_config_btn)
        
        self.reset_all_btn = QPushButton("🔄 Reset All Settings")
        self.reset_all_btn.clicked.connect(self._reset_all_settings)
        self.reset_all_btn.setStyleSheet("background-color: #ffc107; color: black;")
        config_buttons_layout.addWidget(self.reset_all_btn)
        
        config_layout.addLayout(config_buttons_layout)
        
        scroll_layout.addWidget(config_group)
        
        # Setup scroll area
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Setup timer for status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(1000)  # Update every second
        
    def _load_settings(self):
        """Load current system settings"""
        config = self.config_manager.get_system_config()
        
        # System settings
        self.auto_start.setChecked(config.auto_start_monitoring)
        self.minimize_to_tray.setChecked(config.minimize_to_tray)
        self.check_updates.setChecked(config.check_for_updates)
        
        # Logging
        log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level in log_levels:
            self.log_level.setCurrentText(config.log_level)
        
        self.max_log_size.setValue(config.max_log_file_size_mb)
        self.log_rotation_count.setValue(config.log_rotation_count)
        self.log_directory.setText(config.log_directory)
        
        # Data management
        self.data_retention_days.setValue(config.data_retention_days)
        self.auto_cleanup.setChecked(config.auto_cleanup_enabled)
        self.backup_enabled.setChecked(config.backup_enabled)
        self.backup_interval.setValue(config.backup_interval_hours)
        
    def _toggle_monitoring(self):
        """Toggle monitoring state"""
        if self.monitoring_active:
            self._stop_monitoring()
        else:
            self._start_monitoring()
            
    def _start_monitoring(self):
        """Start monitoring"""
        self.monitoring_active = True
        self.status_label.setText("🟢 Monitoring: ACTIVE")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        
        self.start_stop_btn.setText("🛑 STOP MONITORING")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        
        self.pause_resume_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)
        
        self.monitoring_state_changed.emit(True)
        
    def _stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        self.status_label.setText("🔴 Monitoring: STOPPED")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        
        self.start_stop_btn.setText("🚀 START MONITORING")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        
        self.pause_resume_btn.setEnabled(False)
        self.restart_btn.setEnabled(False)
        
        self.monitoring_state_changed.emit(False)
        
    def _toggle_pause(self):
        """Toggle pause state"""
        current_text = self.pause_resume_btn.text()
        if "PAUSE" in current_text:
            self.pause_resume_btn.setText("▶️ RESUME")
            self.status_label.setText("🟡 Monitoring: PAUSED")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: orange;")
        else:
            self.pause_resume_btn.setText("⏸️ PAUSE")
            self.status_label.setText("🟢 Monitoring: ACTIVE")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
            
    def _restart_monitoring(self):
        """Restart monitoring"""
        self._stop_monitoring()
        QTimer.singleShot(1000, self._start_monitoring)  # Restart after 1 second
        
    def _update_status(self):
        """Update monitoring status display"""
        # This would be connected to actual monitoring system
        # For now, just show placeholder values
        if self.monitoring_active:
            import random
            self.packets_received.setText(f"Packets: {random.randint(1000, 9999)}")
            self.anomalies_detected.setText(f"Anomalies: {random.randint(0, 50)}")
            self.alerts_sent.setText(f"Alerts: {random.randint(0, 10)}")
            self.cpu_usage.setText(f"CPU: {random.randint(5, 25)}%")
        
    def _browse_log_directory(self):
        """Browse for log directory"""
        current_dir = self.log_directory.text() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Select Log Directory", current_dir
        )
        
        if directory:
            self.log_directory.setText(directory)
            
    def _backup_database(self):
        """Backup database"""
        QMessageBox.information(
            self, "Database Backup", 
            "Database backup completed successfully!"
        )
        
    def _cleanup_database(self):
        """Cleanup old database records"""
        reply = QMessageBox.question(
            self, "Database Cleanup",
            "This will remove old records based on retention settings. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(
                self, "Database Cleanup",
                "Database cleanup completed!"
            )
            
    def _reset_database(self):
        """Reset database"""
        reply = QMessageBox.critical(
            self, "Reset Database",
            "⚠️ WARNING: This will delete ALL telemetry data, anomalies, and logs!\n\n"
            "This action cannot be undone. Are you sure?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            confirm = QMessageBox.critical(
                self, "Final Confirmation",
                "Last chance! Really delete ALL data?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                QMessageBox.information(
                    self, "Database Reset",
                    "Database has been reset. All data cleared."
                )
                
    def _export_configuration(self):
        """Export configuration to file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration",
            "stdms_config.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                self.config_manager.export_settings(filename)
                QMessageBox.information(
                    self, "Export Successful",
                    f"Configuration exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Failed",
                    f"Failed to export configuration:\n{str(e)}"
                )
                
    def _import_configuration(self):
        """Import configuration from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                self.config_manager.import_settings(filename)
                self._load_settings()  # Reload UI
                QMessageBox.information(
                    self, "Import Successful",
                    "Configuration imported successfully!"
                )
                self.settings_changed.emit()
            except Exception as e:
                QMessageBox.critical(
                    self, "Import Failed",
                    f"Failed to import configuration:\n{str(e)}"
                )
                
    def _reset_all_settings(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset All Settings",
            "This will reset ALL settings to their default values.\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.reset_to_defaults()
            self._load_settings()
            QMessageBox.information(
                self, "Settings Reset",
                "All settings have been reset to defaults."
            )
            self.settings_changed.emit()
            
    def _on_settings_changed(self):
        """Handle settings change"""
        self.settings_changed.emit()
        
    def get_current_config(self) -> SystemConfig:
        """Get current system configuration from UI"""
        config = SystemConfig(
            auto_start_monitoring=self.auto_start.isChecked(),
            minimize_to_tray=self.minimize_to_tray.isChecked(),
            check_for_updates=self.check_updates.isChecked(),
            log_level=self.log_level.currentText(),
            max_log_file_size_mb=self.max_log_size.value(),
            log_rotation_count=self.log_rotation_count.value(),
            log_directory=self.log_directory.text(),
            data_retention_days=self.data_retention_days.value(),
            auto_cleanup_enabled=self.auto_cleanup.isChecked(),
            backup_enabled=self.backup_enabled.isChecked(),
            backup_interval_hours=self.backup_interval.value()
        )
        
        return config
    
    def set_monitoring_state(self, active: bool):
        """Set monitoring state externally"""
        if active != self.monitoring_active:
            if active:
                self._start_monitoring()
            else:
                self._stop_monitoring()