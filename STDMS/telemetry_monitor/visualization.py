#!/usr/bin/env python3
"""
Visualization components for Stage 4 - Real-time telemetry and anomaly plotting
Uses PyQtGraph for high-performance real-time charts and graphs.
"""

import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pyqtgraph as pg
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QSplitter, QTabWidget
)
from PySide6.QtCore import QTimer, Signal, Qt
from PySide6.QtGui import QFont

from models import TelemetryData, AnomalyScore

# Configure PyQtGraph for better performance
pg.setConfigOptions(antialias=True, useOpenGL=True)

@dataclass
class PlotConfiguration:
    """Configuration for plot appearance and behavior"""
    max_points: int = 500
    update_interval: int = 1000  # milliseconds
    line_width: int = 2
    anomaly_marker_size: int = 8
    normal_color: str = '#00ff00'  # Green
    anomaly_color: str = '#ff0000'  # Red
    warning_color: str = '#ffaa00'  # Orange
    background_color: str = '#2b2b2b'  # Dark gray
    grid_alpha: float = 0.3

class TelemetryPlotWidget(pg.PlotWidget):
    """Custom plot widget for telemetry data with anomaly highlighting"""
    
    def __init__(self, title: str, y_label: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.y_label = y_label
        self.unit = unit
        self.config = PlotConfiguration()
        
        # Data storage
        self.timestamps = deque(maxlen=self.config.max_points)
        self.values = deque(maxlen=self.config.max_points)
        self.anomaly_flags = deque(maxlen=self.config.max_points)
        self.anomaly_scores = deque(maxlen=self.config.max_points)
        
        # Plot items
        self.normal_curve = None
        self.anomaly_scatter = None
        self.warning_scatter = None
        
        self._setup_plot()
        
    def _setup_plot(self):
        """Setup plot appearance and properties"""
        # Set background and labels
        self.setBackground(self.config.background_color)
        self.setLabel('left', self.y_label, units=self.unit)
        self.setLabel('bottom', 'Time')
        self.setTitle(self.title)
        
        # Configure grid
        self.showGrid(x=True, y=True, alpha=self.config.grid_alpha)
        
        # Create plot items
        self.normal_curve = self.plot(
            pen=pg.mkPen(color=self.config.normal_color, width=self.config.line_width),
            name='Normal Data'
        )
        
        self.anomaly_scatter = self.plot(
            pen=None,
            symbol='o',
            symbolBrush=self.config.anomaly_color,
            symbolSize=self.config.anomaly_marker_size,
            name='Anomalies'
        )
        
        self.warning_scatter = self.plot(
            pen=None,
            symbol='t',
            symbolBrush=self.config.warning_color,
            symbolSize=self.config.anomaly_marker_size,
            name='Warnings'
        )
        
        # Enable auto-range
        self.enableAutoRange()
        
    def add_data_point(self, timestamp: datetime, value: float, 
                      anomaly_score: Optional[AnomalyScore] = None):
        """Add a new data point to the plot"""
        # Convert timestamp to seconds since epoch for plotting
        time_val = timestamp.timestamp()
        
        # Add to data storage
        self.timestamps.append(time_val)
        self.values.append(value)
        
        # Determine anomaly status
        is_anomaly = False
        is_warning = False
        score = 0.0
        
        if anomaly_score:
            score = anomaly_score.combined_score
            if score >= 0.7:  # High anomaly threshold
                is_anomaly = True
            elif score >= 0.4:  # Warning threshold
                is_warning = True
        
        self.anomaly_flags.append(is_anomaly or is_warning)
        self.anomaly_scores.append(score)
        
        # Update plots
        self._update_plots()
        
    def _update_plots(self):
        """Update plot curves with current data"""
        if not self.timestamps:
            return
            
        # Convert to numpy arrays for better performance
        times = np.array(self.timestamps)
        values = np.array(self.values)
        scores = np.array(self.anomaly_scores)
        
        # Plot normal data (all points as line)
        self.normal_curve.setData(times, values)
        
        # Plot anomalies (high score points)
        anomaly_mask = scores >= 0.7
        if np.any(anomaly_mask):
            anomaly_times = times[anomaly_mask]
            anomaly_values = values[anomaly_mask]
            self.anomaly_scatter.setData(anomaly_times, anomaly_values)
        else:
            self.anomaly_scatter.clear()
            
        # Plot warnings (medium score points)  
        warning_mask = (scores >= 0.4) & (scores < 0.7)
        if np.any(warning_mask):
            warning_times = times[warning_mask]
            warning_values = values[warning_mask]
            self.warning_scatter.setData(warning_times, warning_values)
        else:
            self.warning_scatter.clear()
    
    def clear_data(self):
        """Clear all plot data"""
        self.timestamps.clear()
        self.values.clear()
        self.anomaly_flags.clear()
        self.anomaly_scores.clear()
        
        self.normal_curve.clear()
        self.anomaly_scatter.clear()
        self.warning_scatter.clear()
    
    def set_time_range(self, minutes: int):
        """Set the time range for x-axis display"""
        if self.timestamps:
            latest_time = max(self.timestamps)
            earliest_time = latest_time - (minutes * 60)
            self.setXRange(earliest_time, latest_time)

class AnomalyScorePlotWidget(pg.PlotWidget):
    """Specialized plot widget for ML anomaly scores"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = PlotConfiguration()
        
        # Data storage for different ML models
        self.timestamps = deque(maxlen=self.config.max_points)
        self.combined_scores = deque(maxlen=self.config.max_points)
        self.isolation_scores = deque(maxlen=self.config.max_points)
        self.autoencoder_scores = deque(maxlen=self.config.max_points)
        self.lstm_scores = deque(maxlen=self.config.max_points)
        
        # Plot curves
        self.combined_curve = None
        self.isolation_curve = None
        self.autoencoder_curve = None
        self.lstm_curve = None
        self.threshold_line = None
        
        self._setup_plot()
        
    def _setup_plot(self):
        """Setup anomaly score plot"""
        self.setBackground(self.config.background_color)
        self.setLabel('left', 'Anomaly Score', units='')
        self.setLabel('bottom', 'Time')
        self.setTitle('ML Anomaly Scores')
        self.setYRange(0, 1)  # Anomaly scores are 0-1
        
        # Configure grid
        self.showGrid(x=True, y=True, alpha=self.config.grid_alpha)
        
        # Create curves for different models
        self.combined_curve = self.plot(
            pen=pg.mkPen(color='#ffffff', width=self.config.line_width + 1),
            name='Combined Score'
        )
        
        self.isolation_curve = self.plot(
            pen=pg.mkPen(color='#ff6b6b', width=self.config.line_width),
            name='Isolation Forest'
        )
        
        self.autoencoder_curve = self.plot(
            pen=pg.mkPen(color='#4ecdc4', width=self.config.line_width),
            name='Autoencoder'
        )
        
        self.lstm_curve = self.plot(
            pen=pg.mkPen(color='#45b7d1', width=self.config.line_width),
            name='LSTM'
        )
        
        # Add threshold line
        self.threshold_line = pg.InfiniteLine(
            pos=0.5, angle=0, pen=pg.mkPen(color='#ffaa00', width=2, style=Qt.DashLine),
            label='Threshold'
        )
        self.addItem(self.threshold_line)
        
        # Add legend
        self.addLegend()
        
    def add_anomaly_score(self, timestamp: datetime, anomaly_score: AnomalyScore):
        """Add new anomaly score data point"""
        time_val = timestamp.timestamp()
        
        self.timestamps.append(time_val)
        self.combined_scores.append(anomaly_score.combined_score)
        self.isolation_scores.append(anomaly_score.isolation_forest_score)
        self.autoencoder_scores.append(anomaly_score.autoencoder_score)
        self.lstm_scores.append(anomaly_score.lstm_score)
        
        self._update_plots()
        
    def _update_plots(self):
        """Update all anomaly score curves"""
        if not self.timestamps:
            return
            
        times = np.array(self.timestamps)
        
        # Update all curves
        self.combined_curve.setData(times, np.array(self.combined_scores))
        self.isolation_curve.setData(times, np.array(self.isolation_scores))
        self.autoencoder_curve.setData(times, np.array(self.autoencoder_scores))
        self.lstm_curve.setData(times, np.array(self.lstm_scores))
    
    def set_threshold(self, threshold: float):
        """Update the anomaly threshold line"""
        self.threshold_line.setPos(threshold)
    
    def clear_data(self):
        """Clear all anomaly score data"""
        self.timestamps.clear()
        self.combined_scores.clear()
        self.isolation_scores.clear()
        self.autoencoder_scores.clear()
        self.lstm_scores.clear()
        
        self.combined_curve.clear()
        self.isolation_curve.clear()
        self.autoencoder_curve.clear()
        self.lstm_curve.clear()

class VisualizationControlPanel(QWidget):
    """Control panel for visualization settings"""
    
    # Signals
    pause_resume_clicked = Signal(bool)  # True = pause, False = resume
    time_range_changed = Signal(int)     # minutes
    clear_plots_clicked = Signal()
    max_points_changed = Signal(int)     # max data points
    threshold_changed = Signal(float)    # anomaly threshold (0-1)
    grid_toggled = Signal(bool)          # show/hide grid
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_paused = False
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup control panel UI"""
        layout = QHBoxLayout(self)
        
        # Pause/Resume button
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.setFixedSize(100, 30)
        self.pause_btn.clicked.connect(self._on_pause_resume)
        layout.addWidget(self.pause_btn)
        
        # Time range selector
        layout.addWidget(QLabel("Time Range:"))
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItems(["1 min", "5 min", "10 min", "30 min", "1 hour"])
        self.time_range_combo.setCurrentText("5 min")
        self.time_range_combo.currentTextChanged.connect(self._on_time_range_changed)
        layout.addWidget(self.time_range_combo)
        
        # Chart settings
        layout.addWidget(QLabel("Max Points:"))
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(100, 2000)
        self.max_points_spin.setValue(500)
        self.max_points_spin.setSuffix(" pts")
        self.max_points_spin.valueChanged.connect(self._on_max_points_changed)
        layout.addWidget(self.max_points_spin)
        
        # Anomaly threshold
        layout.addWidget(QLabel("Anomaly Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(10, 90)
        self.threshold_spin.setValue(50)
        self.threshold_spin.setSuffix("%")
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_spin)
        
        # Show grid checkbox
        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self._on_grid_toggled)
        layout.addWidget(self.grid_check)
        
        # Clear plots button
        self.clear_btn = QPushButton("🗑️ Clear Plots")
        self.clear_btn.setFixedSize(100, 30)
        self.clear_btn.clicked.connect(self.clear_plots_clicked.emit)
        layout.addWidget(self.clear_btn)
        
        # Statistics display
        self.stats_label = QLabel("Points: 0 | Anomalies: 0")
        self.stats_label.setFont(QFont("Consolas", 9))
        layout.addWidget(self.stats_label)
        
        layout.addStretch()
        
    def _on_pause_resume(self):
        """Handle pause/resume button click"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText("▶️ Resume")
            self.pause_btn.setStyleSheet("background-color: #ff6b6b;")
        else:
            self.pause_btn.setText("⏸️ Pause")
            self.pause_btn.setStyleSheet("")
            
        self.pause_resume_clicked.emit(self.is_paused)
        
    def _on_time_range_changed(self, text: str):
        """Handle time range selection change"""
        # Convert text to minutes
        if "min" in text:
            minutes = int(text.split()[0])
        elif "hour" in text:
            minutes = int(text.split()[0]) * 60
        else:
            minutes = 5  # default
            
        self.time_range_changed.emit(minutes)
    
    def _on_max_points_changed(self, value: int):
        """Handle max points change"""
        self.max_points_changed.emit(value)
    
    def _on_threshold_changed(self, value: int):
        """Handle anomaly threshold change"""
        # Convert percentage to 0-1 range
        threshold = value / 100.0
        self.threshold_changed.emit(threshold)
    
    def _on_grid_toggled(self, checked: bool):
        """Handle grid toggle"""
        self.grid_toggled.emit(checked)
    
    def update_statistics(self, total_points: int, anomaly_count: int):
        """Update statistics display"""
        self.stats_label.setText(f"Points: {total_points} | Anomalies: {anomaly_count}")

class RealTimeVisualizationWidget(QWidget):
    """Main visualization widget with real-time telemetry and anomaly plots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.config = PlotConfiguration()
        self.is_paused = False
        self.time_range_minutes = 5
        
        # Telemetry plot widgets
        self.telemetry_plots: Dict[str, TelemetryPlotWidget] = {}
        self.anomaly_plot = None
        self.control_panel = None
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_time_ranges)
        self.update_timer.start(5000)  # Update time ranges every 5 seconds
        
        self._setup_ui()
        self._connect_signals()
        
    def _setup_ui(self):
        """Setup the visualization UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        self.control_panel = VisualizationControlPanel()
        layout.addWidget(self.control_panel)
        
        # Create splitter for plots
        splitter = QSplitter(Qt.Vertical)
        
        # Create telemetry plots tabs
        telemetry_tabs = QTabWidget()
        
        # Battery and Power tab
        power_widget = QWidget()
        power_layout = QVBoxLayout(power_widget)
        
        power_splitter = QSplitter(Qt.Horizontal)
        self.telemetry_plots['battery_voltage'] = TelemetryPlotWidget(
            "Battery Voltage", "Voltage", "V"
        )
        self.telemetry_plots['solar_power'] = TelemetryPlotWidget(
            "Solar Power", "Power", "W"
        )
        power_splitter.addWidget(self.telemetry_plots['battery_voltage'])
        power_splitter.addWidget(self.telemetry_plots['solar_power'])
        power_layout.addWidget(power_splitter)
        
        telemetry_tabs.addTab(power_widget, "⚡ Power")
        
        # Attitude and Orbit tab
        attitude_widget = QWidget()
        attitude_layout = QVBoxLayout(attitude_widget)
        
        attitude_splitter = QSplitter(Qt.Horizontal)
        self.telemetry_plots['attitude_x'] = TelemetryPlotWidget(
            "Attitude X", "Angle", "°"
        )
        self.telemetry_plots['orbit_altitude'] = TelemetryPlotWidget(
            "Orbit Altitude", "Altitude", "km"
        )
        attitude_splitter.addWidget(self.telemetry_plots['attitude_x'])
        attitude_splitter.addWidget(self.telemetry_plots['orbit_altitude'])
        attitude_layout.addWidget(attitude_splitter)
        
        telemetry_tabs.addTab(attitude_widget, "�️ Attitude & Orbit")
        
        # Environmental tab
        env_widget = QWidget()
        env_layout = QVBoxLayout(env_widget)
        
        env_splitter = QSplitter(Qt.Horizontal)
        self.telemetry_plots['temperature'] = TelemetryPlotWidget(
            "Temperature", "Temperature", "°C"
        )
        self.telemetry_plots['signal_strength'] = TelemetryPlotWidget(
            "Signal Strength", "Signal", "dBm"
        )
        env_splitter.addWidget(self.telemetry_plots['temperature'])
        env_splitter.addWidget(self.telemetry_plots['signal_strength'])
        env_layout.addWidget(env_splitter)
        
        telemetry_tabs.addTab(env_widget, "🌡️ Environment")
        
        # Add telemetry tabs to splitter
        splitter.addWidget(telemetry_tabs)
        
        # Anomaly scores plot
        self.anomaly_plot = AnomalyScorePlotWidget()
        anomaly_group = QGroupBox("🤖 ML Anomaly Scores")
        anomaly_layout = QVBoxLayout(anomaly_group)
        anomaly_layout.addWidget(self.anomaly_plot)
        
        splitter.addWidget(anomaly_group)
        
        # Set splitter proportions (telemetry: 60%, anomaly: 40%)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
    def _connect_signals(self):
        """Connect control panel signals"""
        self.control_panel.pause_resume_clicked.connect(self._on_pause_resume)
        self.control_panel.time_range_changed.connect(self._on_time_range_changed)
        self.control_panel.clear_plots_clicked.connect(self._on_clear_plots)
        self.control_panel.max_points_changed.connect(self._on_max_points_changed)
        self.control_panel.threshold_changed.connect(self._on_threshold_changed)
        self.control_panel.grid_toggled.connect(self._on_grid_toggled)
        
    def _on_pause_resume(self, is_paused: bool):
        """Handle pause/resume of live plotting"""
        self.is_paused = is_paused
        self.logger.info(f"Live plotting {'paused' if is_paused else 'resumed'}")
        
    def _on_time_range_changed(self, minutes: int):
        """Handle time range change"""
        self.time_range_minutes = minutes
        self._update_time_ranges()
        self.logger.info(f"Time range changed to {minutes} minutes")
        
    def _on_clear_plots(self):
        """Clear all plot data"""
        for plot in self.telemetry_plots.values():
            plot.clear_data()
        self.anomaly_plot.clear_data()
        self.logger.info("All plots cleared")
    
    def _on_max_points_changed(self, max_points: int):
        """Handle max points configuration change"""
        # Update configuration for all plots
        for plot in self.telemetry_plots.values():
            plot.config.max_points = max_points
            plot.timestamps = deque(plot.timestamps, maxlen=max_points)
            plot.values = deque(plot.values, maxlen=max_points)
            plot.anomaly_flags = deque(plot.anomaly_flags, maxlen=max_points)
            plot.anomaly_scores = deque(plot.anomaly_scores, maxlen=max_points)
        
        # Update anomaly plot
        self.anomaly_plot.config.max_points = max_points
        self.anomaly_plot.timestamps = deque(self.anomaly_plot.timestamps, maxlen=max_points)
        self.anomaly_plot.combined_scores = deque(self.anomaly_plot.combined_scores, maxlen=max_points)
        self.anomaly_plot.isolation_scores = deque(self.anomaly_plot.isolation_scores, maxlen=max_points)
        self.anomaly_plot.autoencoder_scores = deque(self.anomaly_plot.autoencoder_scores, maxlen=max_points)
        self.anomaly_plot.lstm_scores = deque(self.anomaly_plot.lstm_scores, maxlen=max_points)
        
        self.logger.info(f"Max points updated to {max_points}")
    
    def _on_threshold_changed(self, threshold: float):
        """Handle anomaly threshold change"""
        self.anomaly_plot.set_threshold(threshold)
        self.logger.info(f"Anomaly threshold updated to {threshold:.2f}")
    
    def _on_grid_toggled(self, show_grid: bool):
        """Handle grid visibility toggle"""
        for plot in self.telemetry_plots.values():
            plot.showGrid(x=show_grid, y=show_grid, alpha=plot.config.grid_alpha)
        self.anomaly_plot.showGrid(x=show_grid, y=show_grid, alpha=self.anomaly_plot.config.grid_alpha)
        self.logger.info(f"Grid {'shown' if show_grid else 'hidden'}")
        
    def _update_time_ranges(self):
        """Update time ranges for all plots"""
        for plot in self.telemetry_plots.values():
            plot.set_time_range(self.time_range_minutes)
            
    def add_telemetry_data(self, data: TelemetryData, anomaly_score: Optional[AnomalyScore] = None):
        """Add new telemetry data point to plots"""
        if self.is_paused:
            return
            
        # Add data to telemetry plots
        telemetry_values = {
            'battery_voltage': data.battery_voltage,
            'solar_power': data.solar_power,
            'temperature': data.temperature,
            'attitude_x': data.attitude_x,
            'orbit_altitude': data.orbit_altitude,
            'signal_strength': data.signal_strength
        }
        
        for param_name, value in telemetry_values.items():
            if param_name in self.telemetry_plots:
                self.telemetry_plots[param_name].add_data_point(
                    data.timestamp, value, anomaly_score
                )
        
        # Add anomaly score if available
        if anomaly_score:
            self.anomaly_plot.add_anomaly_score(data.timestamp, anomaly_score)
            
        # Update statistics
        self._update_statistics()
        
    def _update_statistics(self):
        """Update control panel statistics"""
        # Count total points and anomalies from any plot
        if self.telemetry_plots:
            sample_plot = next(iter(self.telemetry_plots.values()))
            total_points = len(sample_plot.timestamps)
            anomaly_count = sum(1 for score in sample_plot.anomaly_scores if score >= 0.4)
            
            self.control_panel.update_statistics(total_points, anomaly_count)
    
    def set_anomaly_threshold(self, threshold: float):
        """Update anomaly threshold visualization"""
        self.anomaly_plot.set_threshold(threshold)
        
    def get_plot_statistics(self) -> Dict[str, Any]:
        """Get current plotting statistics"""
        stats = {
            'is_paused': self.is_paused,
            'time_range_minutes': self.time_range_minutes,
            'total_plots': len(self.telemetry_plots) + 1,  # +1 for anomaly plot
        }
        
        if self.telemetry_plots:
            sample_plot = next(iter(self.telemetry_plots.values()))
            stats['total_points'] = len(sample_plot.timestamps)
            stats['anomaly_count'] = sum(1 for score in sample_plot.anomaly_scores if score >= 0.4)
            
        return stats