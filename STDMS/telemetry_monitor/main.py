#!/usr/bin/env python3
"""
Telemetry Monitor - Main Application Entry Point
PySide6-based Windows application for satellite telemetry monitoring with comprehensive Stage 5 settings
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, 
                               QVBoxLayout, QHBoxLayout, QWidget, 
                               QTextEdit, QLabel, QPushButton, QGridLayout,
                               QStatusBar, QMenuBar, QMessageBox, QSplitter)
from PySide6.QtCore import QTimer, QThread, Signal, QObject, Qt
from PySide6.QtGui import QIcon, QFont, QTextCursor
import logging
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from ingestion import TelemetrySimulator
from visualization import RealTimeVisualizationWidget
from config_manager import ConfigurationManager, get_config_manager
from settings_widgets import MLModelSettingsWidget, AlertSettingsWidget
from system_widgets import VisualizationSettingsWidget, SystemSettingsWidget


class ComprehensiveSettingsWidget(QWidget):
    """Comprehensive settings widget with all configuration options"""
    
    def __init__(self, config_manager: ConfigurationManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Setup comprehensive settings UI with tabbed interface"""
        layout = QVBoxLayout(self)
        
        # Create settings tab widget
        self.settings_tabs = QTabWidget()
        
        # System & Monitoring tab (most important - first)
        self.system_widget = SystemSettingsWidget(self.config_manager)
        self.settings_tabs.addTab(self.system_widget, "🚀 System & Monitoring")
        
        # ML Models configuration
        self.ml_widget = MLModelSettingsWidget(self.config_manager)
        self.settings_tabs.addTab(self.ml_widget, "🤖 ML Models")
        
        # Alert system configuration
        self.alert_widget = AlertSettingsWidget(self.config_manager)
        self.settings_tabs.addTab(self.alert_widget, "⚠️ Alerts")
        
        # Visualization configuration
        self.viz_widget = VisualizationSettingsWidget(self.config_manager)
        self.settings_tabs.addTab(self.viz_widget, "📊 Visualization")
        
        layout.addWidget(self.settings_tabs)
        
        # Status bar for settings
        self.status_label = QLabel("Settings loaded successfully")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        # Auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self._auto_save_settings)
        self.auto_save_timer.start(5000)  # Auto-save every 5 seconds
        
    def _connect_signals(self):
        """Connect settings change signals"""
        # Connect all widget signals to save handler
        self.system_widget.settings_changed.connect(self._on_settings_changed)
        self.ml_widget.settings_changed.connect(self._on_settings_changed)
        self.alert_widget.settings_changed.connect(self._on_settings_changed)
        self.viz_widget.settings_changed.connect(self._on_settings_changed)
        
        # Connect monitoring state changes
        self.system_widget.monitoring_state_changed.connect(self._on_monitoring_state_changed)
        
    def _on_settings_changed(self):
        """Handle settings changes from any widget"""
        try:
            # Update configuration from all widgets
            self.config_manager.settings.system = self.system_widget.get_current_config()
            self.config_manager.settings.ml_models = self.ml_widget.get_current_config()
            self.config_manager.settings.alerts = self.alert_widget.get_current_config()
            self.config_manager.settings.visualization = self.viz_widget.get_current_config()
            
            self.status_label.setText("Settings updated (will auto-save)")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")
            self.status_label.setText(f"Settings update error: {e}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
    
    def _auto_save_settings(self):
        """Auto-save settings periodically"""
        try:
            self.config_manager.save_settings()
            if "will auto-save" in self.status_label.text():
                self.status_label.setText("Settings saved automatically")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
            
    def _on_monitoring_state_changed(self, active: bool):
        """Handle monitoring state changes"""
        self.logger.info(f"Monitoring state changed: {'ACTIVE' if active else 'STOPPED'}")
        
        # This signal could be connected to parent to start/stop actual monitoring
        if hasattr(self.parent(), 'set_monitoring_active'):
            self.parent().set_monitoring_active(active)


class TelemetryMonitorApp(QMainWindow):
    """Main application window with comprehensive Stage 5 settings integration"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STDMS - Satellite Telemetry Data Management System v2.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize configuration manager first
        self.config_manager = get_config_manager()
        
        # Initialize logging first
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting STDMS - Satellite Telemetry Data Management System")
        
        # Apply system configuration
        self._apply_system_config()
        
        # Initialize simulator with database storage
        self.simulator = TelemetrySimulator(enable_database_storage=True)
        
        # Initialize database for direct queries
        try:
            from storage import get_database
            self.database = get_database()
        except ImportError:
            self.database = None
            self.logger.warning("Database not available")
        
        # Initialize ML anomaly detection
        try:
            from anomaly import get_anomaly_detector
            self.ml_detector = get_anomaly_detector()
        except ImportError:
            self.ml_detector = None
            self.logger.warning("ML anomaly detection not available")
        
        # Initialize alert system
        try:
            from alerts import get_alert_system
            self.alert_system = get_alert_system()
            self.alert_system.start()
        except ImportError:
            self.alert_system = None
            self.logger.warning("Alert system not available")
        
        # Setup UI
        self.setup_ui()
        self.setup_status_bar()
        self.setup_menu_bar()
        
        # Start data updates
        self.setup_timers()
        
        # Initialize monitoring state
        self.monitoring_active = False
        
        # Auto-start monitoring if configured
        if self.config_manager.get_system_config().auto_start_monitoring:
            self.set_monitoring_active(True)
        
    def _apply_system_config(self):
        """Apply system configuration settings"""
        try:
            system_config = self.config_manager.get_system_config()
            
            # Apply window state based on minimize to tray setting
            if system_config.minimize_to_tray:
                self.logger.info("System tray minimization enabled")
                
            # Setup logging based on configuration
            log_level = getattr(logging, system_config.log_level.upper(), logging.INFO)
            logging.getLogger().setLevel(log_level)
            
            self.logger.info(f"Applied system configuration: log_level={system_config.log_level}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply system configuration: {e}")
    
    def set_monitoring_active(self, active: bool):
        """Set monitoring system active/inactive state"""
        try:
            if active and not self.monitoring_active:
                self.logger.info("Starting telemetry monitoring system")
                self.monitoring_active = True
                
                # Start simulator if not already running
                if not self.simulator.is_running:
                    self.start_simulator()
                    
                # Start ML detection if available
                if self.ml_detector:
                    self.logger.info("ML anomaly detection activated")
                    
                # Start alert system if available
                if self.alert_system and not self.alert_system.is_running:
                    self.alert_system.start()
                    
                self.status_bar.showMessage("🟢 Monitoring System: ACTIVE")
                
            elif not active and self.monitoring_active:
                self.logger.info("Stopping telemetry monitoring system")
                self.monitoring_active = False
                
                # Stop simulator
                if self.simulator.is_running:
                    self.stop_simulator()
                    
                # Stop alert system if running
                if self.alert_system and self.alert_system.is_running:
                    self.alert_system.stop()
                    
                self.status_bar.showMessage("🔴 Monitoring System: STOPPED")
                
        except Exception as e:
            self.logger.error(f"Failed to change monitoring state: {e}")
        
    def setup_logging(self):
        """Configure application logging with settings integration"""
        try:
            # Get logging configuration
            system_config = self.config_manager.get_system_config()
            
            # Ensure logs directory exists
            log_dir = Path(system_config.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure logging level
            log_level = getattr(logging, system_config.log_level.upper(), logging.INFO)
            
            # Configure logging with rotation
            from logging.handlers import RotatingFileHandler
            
            log_file = log_dir / "system.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=system_config.max_log_file_size_mb * 1024 * 1024,  # Convert MB to bytes
                backupCount=system_config.log_rotation_count
            )
            
            # Setup logging configuration
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    file_handler,
                    logging.StreamHandler()  # Console output
                ]
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Logging configured: level={system_config.log_level}, dir={log_dir}")
            
        except Exception as e:
            # Fallback to basic logging if configuration fails
            logs_dir = project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            log_file = logs_dir / "system.log"
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
            self.logger.error(f"Failed to configure logging from settings: {e}")
        
    def setup_ui(self):
        """Setup the main user interface with tabs"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_charts_tab()
        self.create_logs_tab()
        self.create_settings_tab()
        
    def create_dashboard_tab(self):
        """Create the dashboard tab with charts placeholder"""
        dashboard_widget = QWidget()
        layout = QVBoxLayout(dashboard_widget)
        
        # Header
        header = QLabel("Telemetry Dashboard")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(header)
        
        # Stats grid
        stats_widget = QWidget()
        stats_layout = QGridLayout(stats_widget)
        
        # Placeholder stats
        self.active_sats_label = QLabel("Active Satellites: 0")
        self.data_points_label = QLabel("Data Points: 0")
        self.anomalies_label = QLabel("Anomalies: 0")
        self.last_update_label = QLabel("Last Update: Never")
        
        # Database stats
        self.db_records_label = QLabel("DB Records: 0")
        self.db_size_label = QLabel("DB Size: 0.0 MB")
        self.db_queue_label = QLabel("DB Queue: 0")
        self.db_status_label = QLabel("DB Status: Disconnected")
        
        # ML Model stats
        self.ml_models_label = QLabel("ML Models: 0/3")
        self.ml_threshold_label = QLabel("ML Threshold: 0.6")
        self.alerts_active_label = QLabel("Active Alerts: 0")
        self.alerts_unack_label = QLabel("Unacknowledged: 0")
        
        stats_layout.addWidget(self.active_sats_label, 0, 0)
        stats_layout.addWidget(self.data_points_label, 0, 1)
        stats_layout.addWidget(self.anomalies_label, 1, 0)
        stats_layout.addWidget(self.last_update_label, 1, 1)
        stats_layout.addWidget(self.db_records_label, 2, 0)
        stats_layout.addWidget(self.db_size_label, 2, 1)
        stats_layout.addWidget(self.db_queue_label, 3, 0)
        stats_layout.addWidget(self.db_status_label, 3, 1)
        stats_layout.addWidget(self.ml_models_label, 4, 0)
        stats_layout.addWidget(self.ml_threshold_label, 4, 1)
        stats_layout.addWidget(self.alerts_active_label, 5, 0)
        stats_layout.addWidget(self.alerts_unack_label, 5, 1)
        
        layout.addWidget(stats_widget)
        
        # Chart placeholder
        chart_placeholder = QLabel("📊 Charts will be displayed here\n\nFeatures coming:\n• Real-time telemetry plots\n• Satellite health status\n• Anomaly indicators")
        chart_placeholder.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 50px;
                text-align: center;
                font-size: 14px;
                color: #666;
            }
        """)
        layout.addWidget(chart_placeholder)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        self.start_sim_btn = QPushButton("Start Simulator")
        self.stop_sim_btn = QPushButton("Stop Simulator")
        self.clear_data_btn = QPushButton("Clear Data")
        self.view_db_btn = QPushButton("View DB Records")
        self.export_data_btn = QPushButton("Export Data")
        self.train_ml_btn = QPushButton("Train ML Models")
        self.view_alerts_btn = QPushButton("View Alerts")
        
        self.start_sim_btn.clicked.connect(self.start_simulator)
        self.stop_sim_btn.clicked.connect(self.stop_simulator)
        self.clear_data_btn.clicked.connect(self.clear_data)
        self.view_db_btn.clicked.connect(self.view_database_records)
        self.export_data_btn.clicked.connect(self.export_telemetry_data)
        self.train_ml_btn.clicked.connect(self.train_ml_models)
        self.view_alerts_btn.clicked.connect(self.view_alerts)
        
        controls_layout.addWidget(self.start_sim_btn)
        controls_layout.addWidget(self.stop_sim_btn)
        controls_layout.addWidget(self.clear_data_btn)
        controls_layout.addWidget(self.view_db_btn)
        controls_layout.addWidget(self.export_data_btn)
        controls_layout.addWidget(self.train_ml_btn)
        controls_layout.addWidget(self.view_alerts_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        self.tab_widget.addTab(dashboard_widget, "Dashboard")
    
    def create_charts_tab(self):
        """Create the charts tab with real-time visualization"""
        # Initialize visualization widget
        self.visualization_widget = RealTimeVisualizationWidget()
        
        # Add to tab widget
        self.tab_widget.addTab(self.visualization_widget, "📊 Charts")
        
        # Set anomaly threshold if ML detector is available
        if hasattr(self, 'ml_detector') and self.ml_detector:
            self.visualization_widget.set_anomaly_threshold(0.5)
        
    def create_logs_tab(self):
        """Create the logs tab to display system logs"""
        logs_widget = QWidget()
        layout = QVBoxLayout(logs_widget)
        
        # Header
        header = QLabel("System Logs")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(header)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log_display)
        
        # Controls
        controls_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Logs")
        clear_logs_btn = QPushButton("Clear Display")
        
        refresh_btn.clicked.connect(self.refresh_logs)
        clear_logs_btn.clicked.connect(self.clear_log_display)
        
        controls_layout.addWidget(refresh_btn)
        controls_layout.addWidget(clear_logs_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        self.tab_widget.addTab(logs_widget, "Logs")
        
    def create_settings_tab(self):
        """Create the comprehensive Stage 5 settings tab"""
        # Create comprehensive settings widget
        self.settings_widget = ComprehensiveSettingsWidget(self.config_manager, self)
        
        # Connect monitoring state changes to actual system control
        self.settings_widget.system_widget.monitoring_state_changed.connect(self.set_monitoring_active)
        
        self.tab_widget.addTab(self.settings_widget, "⚙️ Settings")
        
    def setup_status_bar(self):
        """Setup the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
    def setup_menu_bar(self):
        """Setup the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        # Help menu
        help_menu = menubar.addMenu('&Help')
        
    def setup_timers(self):
        """Setup timers for periodic updates"""
        # Update dashboard every 2 seconds
        self.dashboard_timer = QTimer()
        self.dashboard_timer.timeout.connect(self.update_dashboard)
        self.dashboard_timer.start(2000)  # 2 seconds
        
        # Update logs every 5 seconds
        self.logs_timer = QTimer()
        self.logs_timer.timeout.connect(self.update_logs)
        self.logs_timer.start(5000)  # 5 seconds
        
    def start_simulator(self):
        """Start the telemetry simulator"""
        try:
            self.simulator.start()
            self.logger.info("Telemetry simulator started")
            self.status_bar.showMessage("Simulator running...")
            self.start_sim_btn.setEnabled(False)
            self.stop_sim_btn.setEnabled(True)
        except Exception as e:
            self.logger.error(f"Failed to start simulator: {e}")
            QMessageBox.warning(self, "Error", f"Failed to start simulator: {e}")
            
    def stop_simulator(self):
        """Stop the telemetry simulator"""
        try:
            self.simulator.stop()
            self.logger.info("Telemetry simulator stopped")
            self.status_bar.showMessage("Simulator stopped")
            self.start_sim_btn.setEnabled(True)
            self.stop_sim_btn.setEnabled(False)
        except Exception as e:
            self.logger.error(f"Failed to stop simulator: {e}")
            
    def clear_data(self):
        """Clear all data"""
        try:
            self.simulator.clear_data()
            self.logger.info("All data cleared")
            self.status_bar.showMessage("Data cleared")
            self.update_dashboard()
        except Exception as e:
            self.logger.error(f"Failed to clear data: {e}")
            
    def update_dashboard(self):
        """Update dashboard statistics and visualization"""
        try:
            # Get simulator stats
            stats = self.simulator.get_stats()
            self.active_sats_label.setText(f"Active Satellites: {stats.get('satellites', 0)}")
            self.data_points_label.setText(f"Data Points: {stats.get('data_points', 0)}")
            self.anomalies_label.setText(f"Anomalies: {stats.get('anomalies', 0)}")
            self.last_update_label.setText(f"Last Update: {stats.get('last_update', 'Never')}")
            
            # Update visualization with latest data
            self.update_visualization()
            
            # Update database stats
            if self.database:
                db_stats = self.database.get_database_stats()
                self.db_records_label.setText(f"DB Records: {db_stats.total_records}")
                self.db_size_label.setText(f"DB Size: {db_stats.database_size_mb:.2f} MB")
                self.db_status_label.setText("DB Status: Connected")
                
                # Show database queue status
                db_queue_size = stats.get('db_queue_size', 0)
                self.db_queue_label.setText(f"DB Queue: {db_queue_size}")
            else:
                self.db_status_label.setText("DB Status: Unavailable")
            
            # Update ML model stats
            if self.ml_detector:
                try:
                    ml_status = self.ml_detector.get_model_status()
                    trained_models = sum(ml_status['models_trained'].values())
                    self.ml_models_label.setText(f"ML Models: {trained_models}/3")
                    self.ml_threshold_label.setText(f"ML Threshold: {ml_status['ensemble_threshold']:.2f}")
                except Exception as e:
                    self.logger.error(f"Failed to get ML status: {e}")
            else:
                self.ml_models_label.setText("ML Models: N/A")
            
            # Update alert stats
            if self.alert_system:
                try:
                    alert_stats = self.alert_system.get_stats()
                    self.alerts_active_label.setText(f"Active Alerts: {alert_stats['active_alerts']}")
                    self.alerts_unack_label.setText(f"Unacknowledged: {alert_stats['unacknowledged_alerts']}")
                except Exception as e:
                    self.logger.error(f"Failed to get alert stats: {e}")
            else:
                self.alerts_active_label.setText("Active Alerts: N/A")
                
        except Exception as e:
            self.logger.error(f"Failed to update dashboard: {e}")
    
    def update_visualization(self):
        """Update real-time visualization with latest telemetry data"""
        try:
            if not hasattr(self, 'visualization_widget'):
                return
                
            # Get latest telemetry data from simulator
            latest_data = self.simulator.get_latest_data()
            if not latest_data:
                return
            
            # Get corresponding anomaly score if ML detector is available
            anomaly_score = None
            if self.ml_detector and latest_data:
                try:
                    anomaly_score = self.ml_detector.detect_anomaly(latest_data)
                except Exception as e:
                    self.logger.debug(f"Could not get anomaly score: {e}")
            
            # Update visualization widget
            self.visualization_widget.add_telemetry_data(latest_data, anomaly_score)
            
        except Exception as e:
            self.logger.debug(f"Failed to update visualization: {e}")
            
    def update_logs(self):
        """Update log display with recent entries"""
        try:
            # Read recent log entries
            log_file = project_root / "logs" / "system.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Show last 50 lines
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    self.log_display.setPlainText(''.join(recent_lines))
                    # Scroll to bottom
                    cursor = self.log_display.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    self.log_display.setTextCursor(cursor)
        except Exception as e:
            self.logger.error(f"Failed to update logs: {e}")
            
    def refresh_logs(self):
        """Manually refresh log display"""
        self.update_logs()
        self.status_bar.showMessage("Logs refreshed", 2000)
        
    def clear_log_display(self):
        """Clear the log display (not the log file)"""
        self.log_display.clear()
        self.status_bar.showMessage("Log display cleared", 2000)
        
    def view_database_records(self):
        """View recent database records"""
        if not self.database:
            QMessageBox.warning(self, "Database Error", "Database not available")
            return
            
        try:
            recent_data = self.database.get_recent_telemetry(limit=20)
            
            if not recent_data:
                QMessageBox.information(self, "Database", "No records found in database")
                return
            
            # Create a simple dialog to show recent records
            dialog = QWidget()
            dialog.setWindowTitle("Recent Database Records")
            dialog.setGeometry(200, 200, 800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Create text display
            text_display = QTextEdit()
            text_display.setReadOnly(True)
            text_display.setFont(QFont("Consolas", 10))
            
            # Format data for display
            display_text = "Recent Telemetry Records (Last 20)\n"
            display_text += "=" * 80 + "\n\n"
            
            for record in recent_data:
                display_text += f"Satellite: {record['satellite_id']}\n"
                display_text += f"Time: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
                display_text += f"Temperature: {record['temperature']:.1f}°C\n"
                display_text += f"Battery: {record['battery_voltage']:.2f}V\n"
                display_text += f"Solar Power: {record['solar_power']:.1f}W\n"
                display_text += f"Signal: {record['signal_strength']:.1f}dBm\n"
                display_text += f"Altitude: {record['orbit_altitude']:.1f}km\n"
                
                if record['anomaly_flag']:
                    display_text += f"🚨 ANOMALY: {record['anomaly_type']}\n"
                
                display_text += "-" * 40 + "\n"
            
            text_display.setPlainText(display_text)
            layout.addWidget(text_display)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.show()
            
        except Exception as e:
            self.logger.error(f"Failed to view database records: {e}")
            QMessageBox.critical(self, "Error", f"Failed to view database records: {e}")
            
    def export_telemetry_data(self):
        """Export telemetry data to JSON file"""
        if not self.database:
            QMessageBox.warning(self, "Database Error", "Database not available")
            return
            
        try:
            # Get recent data
            recent_data = self.database.get_recent_telemetry(limit=1000)
            
            if not recent_data:
                QMessageBox.information(self, "Export", "No data to export")
                return
            
            # Convert to JSON serializable format
            export_data = []
            for record in recent_data:
                export_data.append({
                    'satellite_id': record['satellite_id'],
                    'timestamp': record['timestamp'].isoformat(),
                    'temperature': record['temperature'],
                    'battery_voltage': record['battery_voltage'],
                    'solar_power': record['solar_power'],
                    'attitude_x': record['attitude_x'],
                    'attitude_y': record['attitude_y'],
                    'attitude_z': record['attitude_z'],
                    'orbit_altitude': record['orbit_altitude'],
                    'signal_strength': record['signal_strength'],
                    'anomaly_flag': record['anomaly_flag'],
                    'anomaly_type': record['anomaly_type']
                })
            
            # Save to file
            from datetime import datetime
            import json
            
            filename = f"telemetry_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = project_root / "data" / filename
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            QMessageBox.information(self, "Export Complete", 
                                  f"Exported {len(export_data)} records to:\n{filepath}")
            self.logger.info(f"Exported {len(export_data)} records to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")
    
    def train_ml_models(self):
        """Train ML models for anomaly detection"""
        if not self.ml_detector:
            QMessageBox.warning(self, "ML Error", "ML anomaly detection not available")
            return
        
        if not self.database:
            QMessageBox.warning(self, "Database Error", "Database not available for training")
            return
        
        try:
            # Check if we have enough data
            db_stats = self.database.get_database_stats()
            if db_stats.total_records < 50:
                QMessageBox.warning(self, "Insufficient Data", 
                                   f"Need at least 50 records for training. Currently have {db_stats.total_records}")
                return
            
            # Show progress dialog
            from PySide6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Training ML models...", "Cancel", 0, 100, self)
            progress.setWindowModality(2)  # ApplicationModal
            progress.show()
            
            def update_progress(value):
                progress.setValue(value)
                QApplication.processEvents()
            
            # Train models in background (simplified for demo)
            update_progress(20)
            self.logger.info("Starting ML model training...")
            
            update_progress(50)
            training_results = self.ml_detector.train_models(retrain=True)
            
            update_progress(90)
            # Give time for training to complete
            import time
            time.sleep(1)
            
            update_progress(100)
            progress.close()
            
            # Show results
            result_text = "ML Model Training Results:\n\n"
            for model, results in training_results.items():
                result_text += f"{model.title()}:\n"
                result_text += f"  Training time: {results.get('training_time', 0):.2f}s\n"
                result_text += f"  Threshold: {results.get('threshold', 0):.4f}\n"
                result_text += f"  Samples: {results.get('n_samples', results.get('n_sequences', 0))}\n\n"
            
            QMessageBox.information(self, "Training Complete", result_text)
            self.logger.info("ML model training completed successfully")
            
        except Exception as e:
            progress.close() if 'progress' in locals() else None
            self.logger.error(f"ML model training failed: {e}")
            QMessageBox.critical(self, "Training Error", f"ML model training failed: {e}")
    
    def view_alerts(self):
        """View recent alerts and anomalies"""
        if not self.alert_system:
            QMessageBox.warning(self, "Alert Error", "Alert system not available")
            return
        
        try:
            # Get recent alerts
            alerts = self.alert_system.get_alert_history(limit=20)
            
            if not alerts:
                QMessageBox.information(self, "Alerts", "No alerts found")
                return
            
            # Create alert viewer dialog
            dialog = QWidget()
            dialog.setWindowTitle("Recent Alerts & Anomalies")
            dialog.setGeometry(150, 150, 900, 700)
            
            layout = QVBoxLayout(dialog)
            
            # Create text display
            text_display = QTextEdit()
            text_display.setReadOnly(True)
            text_display.setFont(QFont("Consolas", 10))
            
            # Format alerts for display
            display_text = "Recent Alerts & ML Anomalies (Last 20)\n"
            display_text += "=" * 80 + "\n\n"
            
            for alert in reversed(alerts):  # Show newest first
                severity_emoji = {
                    'LOW': '🟡',
                    'MEDIUM': '🟠', 
                    'HIGH': '🔴',
                    'CRITICAL': '🚨'
                }
                
                emoji = severity_emoji.get(alert.severity, '⚠️')
                status = "✅" if alert.resolved else "❌" if alert.acknowledged else "🔔"
                
                display_text += f"{emoji} {alert.severity} - {alert.satellite_id} {status}\n"
                display_text += f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                display_text += f"Rule: {alert.rule_name}\n"
                display_text += f"Type: {alert.anomaly_score.anomaly_type}\n"
                display_text += f"ML Scores:\n"
                display_text += f"  Combined: {alert.anomaly_score.combined_score:.3f}\n"
                display_text += f"  Isolation Forest: {alert.anomaly_score.isolation_forest_score:.3f}\n"
                display_text += f"  Autoencoder: {alert.anomaly_score.autoencoder_score:.3f}\n"
                display_text += f"  LSTM: {alert.anomaly_score.lstm_score:.3f}\n"
                display_text += f"  Confidence: {alert.anomaly_score.confidence:.3f}\n"
                display_text += f"Message: {alert.message}\n"
                display_text += "-" * 60 + "\n"
            
            text_display.setPlainText(display_text)
            layout.addWidget(text_display)
            
            # Control buttons
            controls_layout = QHBoxLayout()
            
            close_btn = QPushButton("Close")
            refresh_btn = QPushButton("Refresh")
            export_alerts_btn = QPushButton("Export Alerts")
            
            close_btn.clicked.connect(dialog.close)
            refresh_btn.clicked.connect(lambda: self.view_alerts())
            export_alerts_btn.clicked.connect(lambda: self.export_alerts(alerts))
            
            controls_layout.addWidget(refresh_btn)
            controls_layout.addWidget(export_alerts_btn)
            controls_layout.addStretch()
            controls_layout.addWidget(close_btn)
            
            layout.addLayout(controls_layout)
            
            dialog.show()
            
        except Exception as e:
            self.logger.error(f"Failed to view alerts: {e}")
            QMessageBox.critical(self, "Error", f"Failed to view alerts: {e}")
    
    def export_alerts(self, alerts):
        """Export alerts to JSON file"""
        try:
            # Convert alerts to JSON serializable format
            export_data = []
            for alert in alerts:
                export_data.append({
                    'id': alert.id,
                    'timestamp': alert.timestamp.isoformat(),
                    'satellite_id': alert.satellite_id,
                    'severity': alert.severity,
                    'rule_name': alert.rule_name,
                    'message': alert.message,
                    'anomaly_type': alert.anomaly_score.anomaly_type,
                    'combined_score': alert.anomaly_score.combined_score,
                    'isolation_forest_score': alert.anomaly_score.isolation_forest_score,
                    'autoencoder_score': alert.anomaly_score.autoencoder_score,
                    'lstm_score': alert.anomaly_score.lstm_score,
                    'confidence': alert.anomaly_score.confidence,
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved
                })
            
            # Save to file
            import json
            filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = project_root / "data" / filename
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            QMessageBox.information(self, "Export Complete", 
                                   f"Exported {len(export_data)} alerts to:\n{filepath}")
            self.logger.info(f"Exported {len(export_data)} alerts to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export alerts: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export alerts: {e}")
        
    def closeEvent(self, event):
        """Handle application closing"""
        self.logger.info("Closing Telemetry Monitor Application")
        self.simulator.stop()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Telemetry Monitor")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = TelemetryMonitorApp()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()