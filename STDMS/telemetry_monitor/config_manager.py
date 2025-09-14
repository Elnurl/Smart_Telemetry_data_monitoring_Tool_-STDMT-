#!/usr/bin/env python3
"""
Configuration Management System for Stage 5
Handles system settings, ML parameters, alert configuration, and persistence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime

@dataclass
class MLModelConfig:
    """Configuration for ML model parameters"""
    # Isolation Forest parameters
    isolation_n_estimators: int = 100
    isolation_contamination: float = 0.1
    isolation_random_state: int = 42
    
    # Autoencoder parameters
    autoencoder_hidden_layers: List[int] = field(default_factory=lambda: [4, 2, 4])
    autoencoder_epochs: int = 100
    autoencoder_batch_size: int = 32
    autoencoder_learning_rate: float = 0.001
    
    # LSTM parameters
    lstm_sequence_length: int = 10
    lstm_units: int = 50
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 0.001
    
    # Ensemble parameters
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    auto_retrain_threshold: int = 1000  # Retrain after N new samples
    retrain_schedule_hours: int = 24    # Auto-retrain every N hours

@dataclass
class AlertConfig:
    """Configuration for alert system"""
    # Anomaly score thresholds
    low_threshold: float = 0.3
    medium_threshold: float = 0.5
    high_threshold: float = 0.7
    critical_threshold: float = 0.9
    
    # Alert suppression
    suppress_duplicates: bool = True
    suppression_window_minutes: int = 10
    max_alerts_per_hour: int = 50
    
    # Notification channels
    enable_email: bool = True
    enable_slack: bool = False
    enable_popup: bool = True
    
    # Email configuration
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    email_subject_prefix: str = "[STDMS Alert]"
    
    # Slack configuration
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    slack_username: str = "STDMS Bot"

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    # Theme and appearance
    theme: str = "dark"
    line_width: int = 2
    point_size: int = 8
    
    # Chart appearance
    max_data_points: int = 500
    max_points_per_chart: int = 500
    update_interval_ms: int = 1000
    default_time_range_minutes: int = 5
    
    # Colors (hex format)
    normal_color: str = "#00ff00"
    normal_data_color: str = "#00ff00"
    warning_color: str = "#ffaa00"
    anomaly_color: str = "#ff0000"
    critical_color: str = "#ff0000"
    background_color: str = "#2b2b2b"
    
    # Chart behavior
    auto_scale: bool = True
    auto_scale_y: bool = True
    show_grid: bool = True
    show_legend: bool = True
    marker_size: int = 8
    
    # Performance settings
    enable_antialiasing: bool = True
    use_opengl: bool = False
    data_buffer_size: int = 10000
    
    # Export settings
    default_export_format: str = "PNG"
    export_dpi: int = 300
    export_directory: str = "exports"

@dataclass
class SystemConfig:
    """Overall system configuration"""
    # Monitoring control
    monitoring_enabled: bool = False
    auto_start_simulator: bool = False
    auto_start_monitoring: bool = False
    simulator_interval_seconds: float = 1.0
    
    # Database settings
    database_path: str = "data/telemetry.db"
    backup_interval_hours: int = 24
    max_database_size_mb: int = 1000
    data_retention_days: int = 90
    
    # Logging configuration
    log_level: str = "INFO"
    log_directory: str = "logs"
    log_rotation_size_mb: int = 10
    log_rotation_count: int = 5
    max_log_files: int = 5
    max_log_file_size_mb: int = 10
    
    # UI settings
    minimize_to_tray: bool = False
    show_splash_screen: bool = True
    remember_window_state: bool = True
    check_for_updates: bool = True
    
    # Data management
    auto_cleanup_enabled: bool = True
    backup_enabled: bool = True
    
    # Performance settings
    max_memory_usage_mb: int = 512
    cpu_usage_limit_percent: int = 50
    enable_gpu_acceleration: bool = True

@dataclass
class AppSettings:
    """Complete application settings"""
    version: str = "1.0.0"
    last_modified: str = ""
    
    # Component configurations
    ml_models: MLModelConfig = field(default_factory=MLModelConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

class ConfigurationManager:
    """Manages application configuration with persistence"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set default config path
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "app_settings.json"
        
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize settings
        self.settings = AppSettings()
        self.original_settings = None  # For change detection
        
        # Load existing configuration
        self.load_settings()
        
    def load_settings(self) -> bool:
        """Load settings from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                # Convert dict to dataclass
                self.settings = self._dict_to_settings(data)
                self.original_settings = self._dict_to_settings(data)
                
                self.logger.info(f"Settings loaded from {self.config_path}")
                return True
            else:
                self.logger.info("No existing settings file found, using defaults")
                self.save_settings()  # Create default settings file
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            return False
    
    def save_settings(self) -> bool:
        """Save settings to JSON file"""
        try:
            # Update modification timestamp
            self.settings.last_modified = datetime.now().isoformat()
            
            # Convert to dict and save
            data = asdict(self.settings)
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update original settings for change detection
            self.original_settings = self._dict_to_settings(data)
            
            self.logger.info(f"Settings saved to {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
            return False
    
    def _dict_to_settings(self, data: Dict[str, Any]) -> AppSettings:
        """Convert dictionary to AppSettings dataclass"""
        # Handle nested dataclasses
        if 'ml_models' in data and isinstance(data['ml_models'], dict):
            data['ml_models'] = MLModelConfig(**data['ml_models'])
        if 'alerts' in data and isinstance(data['alerts'], dict):
            data['alerts'] = AlertConfig(**data['alerts'])
        if 'visualization' in data and isinstance(data['visualization'], dict):
            data['visualization'] = VisualizationConfig(**data['visualization'])
        if 'system' in data and isinstance(data['system'], dict):
            data['system'] = SystemConfig(**data['system'])
            
        return AppSettings(**data)
    
    def has_changes(self) -> bool:
        """Check if settings have been modified"""
        if self.original_settings is None:
            return True
        return asdict(self.settings) != asdict(self.original_settings)
    
    def reset_to_defaults(self):
        """Reset all settings to default values"""
        self.settings = AppSettings()
        self.logger.info("Settings reset to defaults")
    
    def validate_settings(self) -> List[str]:
        """Validate current settings and return list of errors"""
        errors = []
        
        # Validate ML model parameters
        ml = self.settings.ml_models
        if ml.isolation_contamination < 0.01 or ml.isolation_contamination > 0.5:
            errors.append("Isolation Forest contamination must be between 0.01 and 0.5")
        
        if ml.autoencoder_learning_rate <= 0 or ml.autoencoder_learning_rate > 0.1:
            errors.append("Autoencoder learning rate must be between 0 and 0.1")
        
        if ml.lstm_sequence_length < 5 or ml.lstm_sequence_length > 100:
            errors.append("LSTM sequence length must be between 5 and 100")
        
        # Validate alert thresholds
        alert = self.settings.alerts
        thresholds = [alert.low_threshold, alert.medium_threshold, 
                     alert.high_threshold, alert.critical_threshold]
        
        if not all(0 <= t <= 1 for t in thresholds):
            errors.append("All alert thresholds must be between 0 and 1")
        
        if not (thresholds == sorted(thresholds)):
            errors.append("Alert thresholds must be in ascending order")
        
        # Validate email settings if enabled
        if alert.enable_email:
            if not alert.email_username or not alert.email_password:
                errors.append("Email username and password required when email alerts enabled")
            
            if not alert.email_recipients:
                errors.append("At least one email recipient required when email alerts enabled")
        
        # Validate Slack settings if enabled
        if alert.enable_slack:
            if not alert.slack_webhook_url:
                errors.append("Slack webhook URL required when Slack alerts enabled")
        
        # Validate visualization settings
        viz = self.settings.visualization
        if viz.max_data_points < 100 or viz.max_data_points > 10000:
            errors.append("Max data points must be between 100 and 10,000")
        
        if viz.update_interval_ms < 100 or viz.update_interval_ms > 10000:
            errors.append("Update interval must be between 100ms and 10s")
        
        # Validate system settings
        sys = self.settings.system
        if sys.simulator_interval_seconds < 0.1 or sys.simulator_interval_seconds > 60:
            errors.append("Simulator interval must be between 0.1s and 60s")
        
        return errors
    
    def export_settings(self, export_path: Path) -> bool:
        """Export current settings to specified path"""
        try:
            data = asdict(self.settings)
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Settings exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, import_path: Path) -> bool:
        """Import settings from specified path"""
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            # Validate imported settings
            temp_settings = self._dict_to_settings(data)
            old_settings = self.settings
            self.settings = temp_settings
            
            errors = self.validate_settings()
            if errors:
                self.settings = old_settings  # Restore original
                self.logger.error(f"Invalid imported settings: {errors}")
                return False
            
            self.logger.info(f"Settings imported from {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import settings: {e}")
            return False
    
    def get_ml_config(self) -> MLModelConfig:
        """Get ML model configuration"""
        return self.settings.ml_models
    
    def get_alert_config(self) -> AlertConfig:
        """Get alert configuration"""
        return self.settings.alerts
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration"""
        return self.settings.visualization
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.settings.system
    
    def update_ml_config(self, **kwargs):
        """Update ML model configuration"""
        for key, value in kwargs.items():
            if hasattr(self.settings.ml_models, key):
                setattr(self.settings.ml_models, key, value)
    
    def update_alert_config(self, **kwargs):
        """Update alert configuration"""
        for key, value in kwargs.items():
            if hasattr(self.settings.alerts, key):
                setattr(self.settings.alerts, key, value)
    
    def update_visualization_config(self, **kwargs):
        """Update visualization configuration"""
        for key, value in kwargs.items():
            if hasattr(self.settings.visualization, key):
                setattr(self.settings.visualization, key, value)
    
    def update_system_config(self, **kwargs):
        """Update system configuration"""
        for key, value in kwargs.items():
            if hasattr(self.settings.system, key):
                setattr(self.settings.system, key, value)

# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def get_ml_config() -> MLModelConfig:
    """Get ML model configuration"""
    return get_config_manager().get_ml_config()

def get_alert_config() -> AlertConfig:
    """Get alert configuration"""
    return get_config_manager().get_alert_config()

def get_visualization_config() -> VisualizationConfig:
    """Get visualization configuration"""
    return get_config_manager().get_visualization_config()

def get_system_config() -> SystemConfig:
    """Get system configuration"""
    return get_config_manager().get_system_config()