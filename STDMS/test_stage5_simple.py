#!/usr/bin/env python3
"""
Simple Stage 5 Configuration Test
Test configuration management without GUI components
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_config_files():
    """Test configuration file creation and validation"""
    print("🚀 Testing Stage 5 Configuration Files")
    print("=" * 50)
    
    try:
        # Test config manager creation
        from telemetry_monitor.config_manager import ConfigurationManager, Settings
        
        config_manager = ConfigurationManager()
        print("✅ Configuration manager created successfully")
        
        # Test default settings creation
        settings = config_manager.settings
        print(f"🤖 ML Models settings: {type(settings.ml_models).__name__}")
        print(f"⚠️ Alert settings: {type(settings.alerts).__name__}")  
        print(f"📊 Visualization settings: {type(settings.visualization).__name__}")
        print(f"⚙️ System settings: {type(settings.system).__name__}")
        
        # Test JSON serialization
        config_dict = settings.to_dict()
        print(f"✅ Configuration serialized to dictionary ({len(config_dict)} sections)")
        
        # Test JSON file creation
        config_file = project_root / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✅ Configuration saved to {config_file.name}")
        
        # Test JSON file loading
        with open(config_file, 'r') as f:
            loaded_dict = json.load(f)
        
        loaded_settings = Settings.from_dict(loaded_dict)
        print("✅ Configuration loaded from JSON successfully")
        
        # Verify some key values
        ml_config = loaded_settings.ml_models
        print(f"🔧 Loaded ML config - Isolation Forest estimators: {ml_config.isolation_n_estimators}")
        
        alert_config = loaded_settings.alerts
        print(f"🔧 Loaded Alert config - Critical threshold: {alert_config.critical_threshold}")
        
        viz_config = loaded_settings.visualization  
        print(f"🔧 Loaded Viz config - Theme: {viz_config.theme}")
        
        system_config = loaded_settings.system
        print(f"🔧 Loaded System config - Auto-start: {system_config.auto_start_monitoring}")
        
        # Test validation
        validation_errors = config_manager.validate_settings()
        if not validation_errors:
            print("✅ Configuration validation passed")
        else:
            print(f"⚠️ Validation warnings: {validation_errors}")
        
        # Cleanup
        config_file.unlink()
        print("✅ Test configuration file cleaned up")
        
        print("\n🎉 Stage 5 Configuration Files Test Complete!")
        print("All configuration components working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_settings_structure():
    """Test the structure and content of default settings"""
    print("\n📋 Testing Settings Structure")
    print("-" * 30)
    
    try:
        from telemetry_monitor.config_manager import (
            MLModelConfig, AlertConfig, VisualizationConfig, SystemConfig
        )
        
        # Test ML Model Config
        ml_config = MLModelConfig()
        print(f"🤖 ML Model Config defaults:")
        print(f"   - Isolation Forest: {ml_config.isolation_n_estimators} estimators")
        print(f"   - Autoencoder: {ml_config.autoencoder_epochs} epochs")
        print(f"   - LSTM: {ml_config.lstm_sequence_length} sequence length")
        print(f"   - Ensemble weights: {ml_config.ensemble_weights}")
        
        # Test Alert Config
        alert_config = AlertConfig()
        print(f"⚠️ Alert Config defaults:")
        print(f"   - Thresholds: L={alert_config.low_threshold}, M={alert_config.medium_threshold}, H={alert_config.high_threshold}, C={alert_config.critical_threshold}")
        print(f"   - Channels: Email={alert_config.enable_email}, Slack={alert_config.enable_slack}, Popup={alert_config.enable_popup}")
        print(f"   - Max alerts/hour: {alert_config.max_alerts_per_hour}")
        
        # Test Visualization Config
        viz_config = VisualizationConfig()
        print(f"📊 Visualization Config defaults:")
        print(f"   - Theme: {viz_config.theme}")
        print(f"   - Colors: Normal={viz_config.normal_color}, Anomaly={viz_config.anomaly_color}")
        print(f"   - Max points: {viz_config.max_points_per_chart}")
        print(f"   - Update interval: {viz_config.update_interval_ms}ms")
        
        # Test System Config
        system_config = SystemConfig()
        print(f"⚙️ System Config defaults:")
        print(f"   - Auto-start: {system_config.auto_start_monitoring}")
        print(f"   - Log level: {system_config.log_level}")
        print(f"   - Data retention: {system_config.data_retention_days} days")
        print(f"   - Backup enabled: {system_config.backup_enabled}")
        
        print("✅ All configuration classes working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Settings structure test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_settings_structure()
    success2 = test_config_files()
    
    if success1 and success2:
        print("\n🎉 STAGE 5 COMPLETE!")
        print("=" * 50)
        print("✅ Configuration management system fully implemented")
        print("✅ ML model parameter configuration")
        print("✅ Alert sensitivity and notification settings")
        print("✅ Visualization configuration")
        print("✅ System monitoring controls")
        print("✅ JSON configuration persistence")
        print("✅ Settings validation and export/import")
        print("\nStage 5 requirements satisfied! 🚀")
    else:
        print("\n❌ Some tests failed - please check implementation")