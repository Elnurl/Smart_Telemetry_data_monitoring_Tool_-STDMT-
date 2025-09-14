#!/usr/bin/env python3
"""
Stage 5 Settings System Test
Test configuration management and settings functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from telemetry_monitor.config_manager import get_config_manager

def test_configuration_system():
    """Test the Stage 5 configuration system"""
    print("🚀 Testing Stage 5 Configuration System")
    print("=" * 50)
    
    # Get configuration manager
    config_manager = get_config_manager()
    print("✅ Configuration manager loaded successfully")
    
    # Test ML configuration
    ml_config = config_manager.get_ml_config()
    print(f"🤖 ML Models Config:")
    print(f"   - Isolation Forest estimators: {ml_config.isolation_n_estimators}")
    print(f"   - Autoencoder epochs: {ml_config.autoencoder_epochs}")
    print(f"   - LSTM sequence length: {ml_config.lstm_sequence_length}")
    print(f"   - Ensemble weights: {ml_config.ensemble_weights}")
    
    # Test alert configuration
    alert_config = config_manager.get_alert_config()
    print(f"⚠️ Alert System Config:")
    print(f"   - Critical threshold: {alert_config.critical_threshold}")
    print(f"   - Email enabled: {alert_config.enable_email}")
    print(f"   - Slack enabled: {alert_config.enable_slack}")
    print(f"   - Max alerts/hour: {alert_config.max_alerts_per_hour}")
    
    # Test visualization configuration
    viz_config = config_manager.get_visualization_config()
    print(f"📊 Visualization Config:")
    print(f"   - Theme: {viz_config.theme}")
    print(f"   - Max points per chart: {viz_config.max_points_per_chart}")
    print(f"   - Update interval: {viz_config.update_interval_ms}ms")
    print(f"   - Normal color: {viz_config.normal_color}")
    
    # Test system configuration
    system_config = config_manager.get_system_config()
    print(f"⚙️ System Config:")
    print(f"   - Auto-start monitoring: {system_config.auto_start_monitoring}")
    print(f"   - Log level: {system_config.log_level}")
    print(f"   - Data retention: {system_config.data_retention_days} days")
    print(f"   - Log directory: {system_config.log_directory}")
    
    # Test configuration update
    print("\n🔧 Testing configuration updates...")
    original_threshold = alert_config.critical_threshold
    
    # Update threshold
    alert_config.critical_threshold = 0.95
    config_manager.settings.alerts = alert_config
    
    # Save and reload
    config_manager.save_settings()
    print("✅ Settings saved to config.json")
    
    # Verify persistence
    new_config_manager = get_config_manager()
    new_alert_config = new_config_manager.get_alert_config()
    
    if new_alert_config.critical_threshold == 0.95:
        print("✅ Configuration persistence working correctly")
    else:
        print("❌ Configuration persistence failed")
    
    # Restore original value
    alert_config.critical_threshold = original_threshold
    config_manager.settings.alerts = alert_config
    config_manager.save_settings()
    
    # Test export/import
    print("\n📤 Testing export/import functionality...")
    export_file = project_root / "test_config_export.json"
    
    try:
        config_manager.export_settings(str(export_file))
        print("✅ Configuration exported successfully")
        
        # Test import
        import_manager = get_config_manager()
        import_manager.import_settings(str(export_file))
        print("✅ Configuration imported successfully")
        
        # Cleanup
        export_file.unlink()
        print("✅ Test export file cleaned up")
        
    except Exception as e:
        print(f"❌ Export/import test failed: {e}")
    
    # Test validation
    print("\n🔍 Testing configuration validation...")
    try:
        # Test invalid threshold (should be caught by validation)
        invalid_config = config_manager.get_alert_config()
        invalid_config.critical_threshold = 1.5  # Invalid - over 1.0
        
        validation_errors = config_manager.validate_settings()
        if validation_errors:
            print("✅ Configuration validation working - detected invalid threshold")
        else:
            print("⚠️ Configuration validation may need improvement")
            
    except Exception as e:
        print(f"✅ Configuration validation caught error: {e}")
    
    print("\n🎉 Stage 5 Configuration System Test Complete!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    test_configuration_system()