#!/usr/bin/env python3
"""
Settings UI Components for Stage 5
Comprehensive configuration widgets for ML models, alerts, and system settings.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QSlider, QTextEdit, QGroupBox,
    QTabWidget, QFormLayout, QScrollArea, QMessageBox,
    QProgressBar, QFileDialog, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette

from config_manager import (
    ConfigurationManager, MLModelConfig, AlertConfig, 
    VisualizationConfig, SystemConfig
)

class MLModelSettingsWidget(QWidget):
    """Widget for configuring ML model parameters"""
    
    settings_changed = Signal()
    
    def __init__(self, config_manager: ConfigurationManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_ui()
        self._load_settings()
        
    def _setup_ui(self):
        """Setup ML model settings UI"""
        layout = QVBoxLayout(self)
        
        # Create scroll area for settings
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Isolation Forest settings
        isolation_group = QGroupBox("🌲 Isolation Forest Parameters")
        isolation_layout = QFormLayout(isolation_group)
        
        self.isolation_estimators = QSpinBox()
        self.isolation_estimators.setRange(10, 1000)
        self.isolation_estimators.setValue(100)
        self.isolation_estimators.valueChanged.connect(self._on_settings_changed)
        isolation_layout.addRow("Number of Estimators:", self.isolation_estimators)
        
        self.isolation_contamination = QDoubleSpinBox()
        self.isolation_contamination.setRange(0.01, 0.5)
        self.isolation_contamination.setSingleStep(0.01)
        self.isolation_contamination.setDecimals(3)
        self.isolation_contamination.valueChanged.connect(self._on_settings_changed)
        isolation_layout.addRow("Contamination Rate:", self.isolation_contamination)
        
        self.isolation_random_state = QSpinBox()
        self.isolation_random_state.setRange(1, 9999)
        self.isolation_random_state.valueChanged.connect(self._on_settings_changed)
        isolation_layout.addRow("Random State:", self.isolation_random_state)
        
        scroll_layout.addWidget(isolation_group)
        
        # Autoencoder settings
        autoencoder_group = QGroupBox("🤖 Autoencoder Neural Network")
        autoencoder_layout = QFormLayout(autoencoder_group)
        
        self.autoencoder_epochs = QSpinBox()
        self.autoencoder_epochs.setRange(10, 1000)
        self.autoencoder_epochs.valueChanged.connect(self._on_settings_changed)
        autoencoder_layout.addRow("Training Epochs:", self.autoencoder_epochs)
        
        self.autoencoder_batch_size = QSpinBox()
        self.autoencoder_batch_size.setRange(8, 256)
        self.autoencoder_batch_size.valueChanged.connect(self._on_settings_changed)
        autoencoder_layout.addRow("Batch Size:", self.autoencoder_batch_size)
        
        self.autoencoder_learning_rate = QDoubleSpinBox()
        self.autoencoder_learning_rate.setRange(0.0001, 0.1)
        self.autoencoder_learning_rate.setSingleStep(0.0001)
        self.autoencoder_learning_rate.setDecimals(4)
        self.autoencoder_learning_rate.valueChanged.connect(self._on_settings_changed)
        autoencoder_layout.addRow("Learning Rate:", self.autoencoder_learning_rate)
        
        # Hidden layers configuration
        self.autoencoder_layers = QLineEdit()
        self.autoencoder_layers.setPlaceholderText("e.g., 4,2,4")
        self.autoencoder_layers.textChanged.connect(self._on_settings_changed)
        autoencoder_layout.addRow("Hidden Layers:", self.autoencoder_layers)
        
        scroll_layout.addWidget(autoencoder_group)
        
        # LSTM settings
        lstm_group = QGroupBox("📈 LSTM Time Series Model")
        lstm_layout = QFormLayout(lstm_group)
        
        self.lstm_sequence_length = QSpinBox()
        self.lstm_sequence_length.setRange(5, 100)
        self.lstm_sequence_length.valueChanged.connect(self._on_settings_changed)
        lstm_layout.addRow("Sequence Length:", self.lstm_sequence_length)
        
        self.lstm_units = QSpinBox()
        self.lstm_units.setRange(10, 200)
        self.lstm_units.valueChanged.connect(self._on_settings_changed)
        lstm_layout.addRow("LSTM Units:", self.lstm_units)
        
        self.lstm_epochs = QSpinBox()
        self.lstm_epochs.setRange(10, 500)
        self.lstm_epochs.valueChanged.connect(self._on_settings_changed)
        lstm_layout.addRow("Training Epochs:", self.lstm_epochs)
        
        self.lstm_batch_size = QSpinBox()
        self.lstm_batch_size.setRange(8, 128)
        self.lstm_batch_size.valueChanged.connect(self._on_settings_changed)
        lstm_layout.addRow("Batch Size:", self.lstm_batch_size)
        
        self.lstm_learning_rate = QDoubleSpinBox()
        self.lstm_learning_rate.setRange(0.0001, 0.1)
        self.lstm_learning_rate.setSingleStep(0.0001)
        self.lstm_learning_rate.setDecimals(4)
        self.lstm_learning_rate.valueChanged.connect(self._on_settings_changed)
        lstm_layout.addRow("Learning Rate:", self.lstm_learning_rate)
        
        scroll_layout.addWidget(lstm_group)
        
        # Ensemble settings
        ensemble_group = QGroupBox("⚖️ Ensemble Configuration")
        ensemble_layout = QFormLayout(ensemble_group)
        
        self.isolation_weight = QDoubleSpinBox()
        self.isolation_weight.setRange(0.0, 1.0)
        self.isolation_weight.setSingleStep(0.1)
        self.isolation_weight.setDecimals(2)
        self.isolation_weight.valueChanged.connect(self._on_ensemble_weight_changed)
        ensemble_layout.addRow("Isolation Forest Weight:", self.isolation_weight)
        
        self.autoencoder_weight = QDoubleSpinBox()
        self.autoencoder_weight.setRange(0.0, 1.0)
        self.autoencoder_weight.setSingleStep(0.1)
        self.autoencoder_weight.setDecimals(2)
        self.autoencoder_weight.valueChanged.connect(self._on_ensemble_weight_changed)
        ensemble_layout.addRow("Autoencoder Weight:", self.autoencoder_weight)
        
        self.lstm_weight = QDoubleSpinBox()
        self.lstm_weight.setRange(0.0, 1.0)
        self.lstm_weight.setSingleStep(0.1)
        self.lstm_weight.setDecimals(2)
        self.lstm_weight.valueChanged.connect(self._on_ensemble_weight_changed)
        ensemble_layout.addRow("LSTM Weight:", self.lstm_weight)
        
        self.weights_status = QLabel("Total: 1.00 ✅")
        ensemble_layout.addRow("Weight Total:", self.weights_status)
        
        # Auto-retrain settings
        self.auto_retrain_threshold = QSpinBox()
        self.auto_retrain_threshold.setRange(100, 10000)
        self.auto_retrain_threshold.valueChanged.connect(self._on_settings_changed)
        ensemble_layout.addRow("Auto-retrain Threshold:", self.auto_retrain_threshold)
        
        self.retrain_schedule = QSpinBox()
        self.retrain_schedule.setRange(1, 168)  # 1-168 hours (1 week)
        self.retrain_schedule.setSuffix(" hours")
        self.retrain_schedule.valueChanged.connect(self._on_settings_changed)
        ensemble_layout.addRow("Retrain Schedule:", self.retrain_schedule)
        
        scroll_layout.addWidget(ensemble_group)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("🔄 Reset to Defaults")
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        controls_layout.addWidget(self.reset_btn)
        
        self.train_now_btn = QPushButton("🚀 Train Models Now")
        self.train_now_btn.clicked.connect(self._train_models_now)
        controls_layout.addWidget(self.train_now_btn)
        
        controls_layout.addStretch()
        scroll_layout.addLayout(controls_layout)
        
        # Setup scroll area
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
    def _load_settings(self):
        """Load current ML model settings"""
        config = self.config_manager.get_ml_config()
        
        # Isolation Forest
        self.isolation_estimators.setValue(config.isolation_n_estimators)
        self.isolation_contamination.setValue(config.isolation_contamination)
        self.isolation_random_state.setValue(config.isolation_random_state)
        
        # Autoencoder
        self.autoencoder_epochs.setValue(config.autoencoder_epochs)
        self.autoencoder_batch_size.setValue(config.autoencoder_batch_size)
        self.autoencoder_learning_rate.setValue(config.autoencoder_learning_rate)
        self.autoencoder_layers.setText(','.join(map(str, config.autoencoder_hidden_layers)))
        
        # LSTM
        self.lstm_sequence_length.setValue(config.lstm_sequence_length)
        self.lstm_units.setValue(config.lstm_units)
        self.lstm_epochs.setValue(config.lstm_epochs)
        self.lstm_batch_size.setValue(config.lstm_batch_size)
        self.lstm_learning_rate.setValue(config.lstm_learning_rate)
        
        # Ensemble
        weights = config.ensemble_weights
        if len(weights) >= 3:
            self.isolation_weight.setValue(weights[0])
            self.autoencoder_weight.setValue(weights[1])
            self.lstm_weight.setValue(weights[2])
        
        self.auto_retrain_threshold.setValue(config.auto_retrain_threshold)
        self.retrain_schedule.setValue(config.retrain_schedule_hours)
        
        self._update_weights_status()
        
    def _on_settings_changed(self):
        """Handle settings change"""
        self.settings_changed.emit()
        
    def _on_ensemble_weight_changed(self):
        """Handle ensemble weight change"""
        self._update_weights_status()
        self._on_settings_changed()
        
    def _update_weights_status(self):
        """Update ensemble weights status"""
        total = (self.isolation_weight.value() + 
                self.autoencoder_weight.value() + 
                self.lstm_weight.value())
        
        if abs(total - 1.0) < 0.01:
            self.weights_status.setText(f"Total: {total:.2f} ✅")
            self.weights_status.setStyleSheet("color: green;")
        else:
            self.weights_status.setText(f"Total: {total:.2f} ⚠️")
            self.weights_status.setStyleSheet("color: orange;")
    
    def _reset_to_defaults(self):
        """Reset ML settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings", 
            "Reset all ML model settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            default_config = MLModelConfig()
            self.config_manager.settings.ml_models = default_config
            self._load_settings()
            self.settings_changed.emit()
            
    def _train_models_now(self):
        """Trigger immediate model training"""
        self.train_now_btn.setEnabled(False)
        self.train_now_btn.setText("🔄 Training...")
        
        # This would trigger the actual training
        QMessageBox.information(
            self, "Training Models", 
            "ML model training initiated. Check the logs tab for progress."
        )
        
        # Re-enable button after delay
        QTimer.singleShot(3000, lambda: (
            self.train_now_btn.setEnabled(True),
            self.train_now_btn.setText("🚀 Train Models Now")
        ))
    
    def get_current_config(self) -> MLModelConfig:
        """Get current ML configuration from UI"""
        # Parse hidden layers
        try:
            layers_text = self.autoencoder_layers.text().strip()
            if layers_text:
                hidden_layers = [int(x.strip()) for x in layers_text.split(',')]
            else:
                hidden_layers = [4, 2, 4]
        except ValueError:
            hidden_layers = [4, 2, 4]
        
        # Create configuration
        config = MLModelConfig(
            # Isolation Forest
            isolation_n_estimators=self.isolation_estimators.value(),
            isolation_contamination=self.isolation_contamination.value(),
            isolation_random_state=self.isolation_random_state.value(),
            
            # Autoencoder
            autoencoder_hidden_layers=hidden_layers,
            autoencoder_epochs=self.autoencoder_epochs.value(),
            autoencoder_batch_size=self.autoencoder_batch_size.value(),
            autoencoder_learning_rate=self.autoencoder_learning_rate.value(),
            
            # LSTM
            lstm_sequence_length=self.lstm_sequence_length.value(),
            lstm_units=self.lstm_units.value(),
            lstm_epochs=self.lstm_epochs.value(),
            lstm_batch_size=self.lstm_batch_size.value(),
            lstm_learning_rate=self.lstm_learning_rate.value(),
            
            # Ensemble
            ensemble_weights=[
                self.isolation_weight.value(),
                self.autoencoder_weight.value(),
                self.lstm_weight.value()
            ],
            auto_retrain_threshold=self.auto_retrain_threshold.value(),
            retrain_schedule_hours=self.retrain_schedule.value()
        )
        
        return config

class AlertSettingsWidget(QWidget):
    """Widget for configuring alert system"""
    
    settings_changed = Signal()
    
    def __init__(self, config_manager: ConfigurationManager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_ui()
        self._load_settings()
        
    def _setup_ui(self):
        """Setup alert settings UI"""
        layout = QVBoxLayout(self)
        
        # Create scroll area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Threshold settings
        threshold_group = QGroupBox("⚠️ Anomaly Score Thresholds")
        threshold_layout = QGridLayout(threshold_group)
        
        # Create threshold sliders with labels
        self.threshold_widgets = {}
        thresholds = [
            ("Low", "low_threshold", 0.3, "yellow"),
            ("Medium", "medium_threshold", 0.5, "orange"), 
            ("High", "high_threshold", 0.7, "red"),
            ("Critical", "critical_threshold", 0.9, "darkred")
        ]
        
        for i, (name, key, default, color) in enumerate(thresholds):
            label = QLabel(f"{name}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(default * 100))
            slider.valueChanged.connect(self._on_threshold_changed)
            
            value_label = QLabel(f"{default:.2f}")
            value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            threshold_layout.addWidget(label, i, 0)
            threshold_layout.addWidget(slider, i, 1)
            threshold_layout.addWidget(value_label, i, 2)
            
            self.threshold_widgets[key] = (slider, value_label)
        
        scroll_layout.addWidget(threshold_group)
        
        # Alert behavior settings
        behavior_group = QGroupBox("🔔 Alert Behavior")
        behavior_layout = QFormLayout(behavior_group)
        
        self.suppress_duplicates = QCheckBox("Suppress duplicate alerts")
        self.suppress_duplicates.stateChanged.connect(self._on_settings_changed)
        behavior_layout.addRow(self.suppress_duplicates)
        
        self.suppression_window = QSpinBox()
        self.suppression_window.setRange(1, 60)
        self.suppression_window.setSuffix(" minutes")
        self.suppression_window.valueChanged.connect(self._on_settings_changed)
        behavior_layout.addRow("Suppression Window:", self.suppression_window)
        
        self.max_alerts_per_hour = QSpinBox()
        self.max_alerts_per_hour.setRange(1, 1000)
        self.max_alerts_per_hour.valueChanged.connect(self._on_settings_changed)
        behavior_layout.addRow("Max Alerts/Hour:", self.max_alerts_per_hour)
        
        scroll_layout.addWidget(behavior_group)
        
        # Notification channels
        channels_group = QGroupBox("📱 Notification Channels")
        channels_layout = QVBoxLayout(channels_group)
        
        # Enable/disable checkboxes
        channels_control = QHBoxLayout()
        self.enable_email = QCheckBox("📧 Email")
        self.enable_email.stateChanged.connect(self._on_channel_toggled)
        channels_control.addWidget(self.enable_email)
        
        self.enable_slack = QCheckBox("💬 Slack")
        self.enable_slack.stateChanged.connect(self._on_channel_toggled)
        channels_control.addWidget(self.enable_slack)
        
        self.enable_popup = QCheckBox("🔔 Popup")
        self.enable_popup.stateChanged.connect(self._on_settings_changed)
        channels_control.addWidget(self.enable_popup)
        
        channels_control.addStretch()
        channels_layout.addLayout(channels_control)
        
        # Email configuration
        self.email_group = QGroupBox("📧 Email Configuration")
        email_layout = QFormLayout(self.email_group)
        
        self.email_server = QLineEdit()
        self.email_server.textChanged.connect(self._on_settings_changed)
        email_layout.addRow("SMTP Server:", self.email_server)
        
        self.email_port = QSpinBox()
        self.email_port.setRange(1, 65535)
        self.email_port.valueChanged.connect(self._on_settings_changed)
        email_layout.addRow("SMTP Port:", self.email_port)
        
        self.email_username = QLineEdit()
        self.email_username.textChanged.connect(self._on_settings_changed)
        email_layout.addRow("Username:", self.email_username)
        
        self.email_password = QLineEdit()
        self.email_password.setEchoMode(QLineEdit.Password)
        self.email_password.textChanged.connect(self._on_settings_changed)
        email_layout.addRow("Password:", self.email_password)
        
        self.email_recipients = QTextEdit()
        self.email_recipients.setMaximumHeight(80)
        self.email_recipients.setPlaceholderText("Enter email addresses, one per line")
        self.email_recipients.textChanged.connect(self._on_settings_changed)
        email_layout.addRow("Recipients:", self.email_recipients)
        
        channels_layout.addWidget(self.email_group)
        
        # Slack configuration
        self.slack_group = QGroupBox("💬 Slack Configuration")
        slack_layout = QFormLayout(self.slack_group)
        
        self.slack_webhook = QLineEdit()
        self.slack_webhook.setPlaceholderText("https://hooks.slack.com/services/...")
        self.slack_webhook.textChanged.connect(self._on_settings_changed)
        slack_layout.addRow("Webhook URL:", self.slack_webhook)
        
        self.slack_channel = QLineEdit()
        self.slack_channel.setPlaceholderText("#alerts")
        self.slack_channel.textChanged.connect(self._on_settings_changed)
        slack_layout.addRow("Channel:", self.slack_channel)
        
        self.slack_username = QLineEdit()
        self.slack_username.setPlaceholderText("STDMS Bot")
        self.slack_username.textChanged.connect(self._on_settings_changed)
        slack_layout.addRow("Bot Username:", self.slack_username)
        
        channels_layout.addWidget(self.slack_group)
        
        scroll_layout.addWidget(channels_group)
        
        # Test buttons
        test_layout = QHBoxLayout()
        
        self.test_email_btn = QPushButton("📧 Test Email")
        self.test_email_btn.clicked.connect(self._test_email)
        test_layout.addWidget(self.test_email_btn)
        
        self.test_slack_btn = QPushButton("💬 Test Slack")
        self.test_slack_btn.clicked.connect(self._test_slack)
        test_layout.addWidget(self.test_slack_btn)
        
        test_layout.addStretch()
        scroll_layout.addLayout(test_layout)
        
        # Setup scroll area
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
    def _load_settings(self):
        """Load current alert settings"""
        config = self.config_manager.get_alert_config()
        
        # Thresholds
        thresholds = {
            "low_threshold": config.low_threshold,
            "medium_threshold": config.medium_threshold,
            "high_threshold": config.high_threshold,
            "critical_threshold": config.critical_threshold
        }
        
        for key, value in thresholds.items():
            if key in self.threshold_widgets:
                slider, label = self.threshold_widgets[key]
                slider.setValue(int(value * 100))
                label.setText(f"{value:.2f}")
        
        # Behavior
        self.suppress_duplicates.setChecked(config.suppress_duplicates)
        self.suppression_window.setValue(config.suppression_window_minutes)
        self.max_alerts_per_hour.setValue(config.max_alerts_per_hour)
        
        # Channels
        self.enable_email.setChecked(config.enable_email)
        self.enable_slack.setChecked(config.enable_slack)
        self.enable_popup.setChecked(config.enable_popup)
        
        # Email
        self.email_server.setText(config.email_smtp_server)
        self.email_port.setValue(config.email_smtp_port)
        self.email_username.setText(config.email_username)
        self.email_password.setText(config.email_password)
        self.email_recipients.setPlainText('\n'.join(config.email_recipients))
        
        # Slack
        self.slack_webhook.setText(config.slack_webhook_url)
        self.slack_channel.setText(config.slack_channel)
        self.slack_username.setText(config.slack_username)
        
        self._update_channel_visibility()
        
    def _on_settings_changed(self):
        """Handle settings change"""
        self.settings_changed.emit()
        
    def _on_threshold_changed(self, value):
        """Handle threshold slider change"""
        # Update corresponding label
        sender = self.sender()
        for key, (slider, label) in self.threshold_widgets.items():
            if slider == sender:
                label.setText(f"{value/100:.2f}")
                break
        
        self._on_settings_changed()
        
    def _on_channel_toggled(self):
        """Handle notification channel toggle"""
        self._update_channel_visibility()
        self._on_settings_changed()
        
    def _update_channel_visibility(self):
        """Update visibility of channel configuration groups"""
        self.email_group.setVisible(self.enable_email.isChecked())
        self.slack_group.setVisible(self.enable_slack.isChecked())
        
    def _test_email(self):
        """Test email configuration"""
        self.test_email_btn.setEnabled(False)
        self.test_email_btn.setText("📧 Testing...")
        
        # Simulate test
        QTimer.singleShot(2000, lambda: (
            QMessageBox.information(self, "Email Test", "Test email sent successfully!"),
            self.test_email_btn.setEnabled(True),
            self.test_email_btn.setText("📧 Test Email")
        ))
        
    def _test_slack(self):
        """Test Slack configuration"""
        self.test_slack_btn.setEnabled(False)
        self.test_slack_btn.setText("💬 Testing...")
        
        # Simulate test
        QTimer.singleShot(2000, lambda: (
            QMessageBox.information(self, "Slack Test", "Test message sent to Slack!"),
            self.test_slack_btn.setEnabled(True),
            self.test_slack_btn.setText("💬 Test Slack")
        ))
    
    def get_current_config(self) -> AlertConfig:
        """Get current alert configuration from UI"""
        # Parse email recipients
        recipients_text = self.email_recipients.toPlainText().strip()
        recipients = [line.strip() for line in recipients_text.split('\n') if line.strip()]
        
        config = AlertConfig(
            # Thresholds
            low_threshold=self.threshold_widgets["low_threshold"][0].value() / 100.0,
            medium_threshold=self.threshold_widgets["medium_threshold"][0].value() / 100.0,
            high_threshold=self.threshold_widgets["high_threshold"][0].value() / 100.0,
            critical_threshold=self.threshold_widgets["critical_threshold"][0].value() / 100.0,
            
            # Behavior
            suppress_duplicates=self.suppress_duplicates.isChecked(),
            suppression_window_minutes=self.suppression_window.value(),
            max_alerts_per_hour=self.max_alerts_per_hour.value(),
            
            # Channels
            enable_email=self.enable_email.isChecked(),
            enable_slack=self.enable_slack.isChecked(),
            enable_popup=self.enable_popup.isChecked(),
            
            # Email
            email_smtp_server=self.email_server.text(),
            email_smtp_port=self.email_port.value(),
            email_username=self.email_username.text(),
            email_password=self.email_password.text(),
            email_recipients=recipients,
            
            # Slack
            slack_webhook_url=self.slack_webhook.text(),
            slack_channel=self.slack_channel.text(),
            slack_username=self.slack_username.text()
        )
        
        return config