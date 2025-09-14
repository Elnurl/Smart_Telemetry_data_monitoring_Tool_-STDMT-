#!/usr/bin/env python3
"""
Simple test for Stage 3 ML components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Stage 3 components can be imported"""
    try:
        print("Testing Stage 3 imports...")
        
        from telemetry_monitor.anomaly import SmartAnomalyDetector
        print("✅ SmartAnomalyDetector imported successfully")
        
        from telemetry_monitor.alerts import SmartAlertSystem  
        print("✅ SmartAlertSystem imported successfully")
        
        from telemetry_monitor.models import AnomalyScore, TelemetryData
        print("✅ Models imported successfully")
        
        print("\n🎉 All Stage 3 components imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring database"""
    try:
        print("\nTesting basic functionality...")
        
        from telemetry_monitor.models import AnomalyScore, TelemetryData
        from datetime import datetime
        
        # Create test data
        test_data = TelemetryData(
            satellite_id="TEST-001",
            timestamp=datetime.now(),
            battery_voltage=12.5,
            solar_current=2.3,
            temperature=20.0,
            memory_usage=45.2,
            cpu_usage=35.8,
            signal_strength=-75.2
        )
        print("✅ Test telemetry data created")
        
        # Create test anomaly score
        anomaly_score = AnomalyScore(
            combined_score=0.15,
            isolation_forest_score=0.12,
            autoencoder_score=0.18,
            lstm_score=0.14,
            confidence=0.85,
            anomaly_type="NORMAL"
        )
        print("✅ Test anomaly score created")
        
        print(f"  Combined Score: {anomaly_score.combined_score}")
        print(f"  Anomaly Type: {anomaly_score.anomaly_type}")
        print(f"  Confidence: {anomaly_score.confidence}")
        
        print("\n🎉 Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_ml_detector_creation():
    """Test creating ML detector without training"""
    try:
        print("\nTesting ML detector creation...")
        
        from telemetry_monitor.anomaly import SmartAnomalyDetector
        
        # Create detector without database (for testing)
        detector = SmartAnomalyDetector(
            database=None,
            storage_service=None
        )
        print("✅ SmartAnomalyDetector created successfully")
        
        # Check if models are initialized
        print(f"  Isolation Forest: {'✅' if hasattr(detector, 'isolation_forest') else '❌'}")
        print(f"  Autoencoder: {'✅' if hasattr(detector, 'autoencoder') else '❌'}")
        print(f"  LSTM: {'✅' if hasattr(detector, 'lstm_model') else '❌'}")
        
        print("\n🎉 ML detector creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ML detector creation test failed: {e}")
        return False

def test_alert_system_creation():
    """Test creating alert system"""
    try:
        print("\nTesting alert system creation...")
        
        from telemetry_monitor.alerts import SmartAlertSystem
        
        # Create alert system without database (for testing)
        alert_system = SmartAlertSystem(
            database=None,
            config_path=None  # Will use default config
        )
        print("✅ SmartAlertSystem created successfully")
        
        # Check notifiers
        print(f"  Email Notifier: {'✅' if hasattr(alert_system, 'email_notifier') else '❌'}")
        print(f"  Slack Notifier: {'✅' if hasattr(alert_system, 'slack_notifier') else '❌'}")
        print(f"  Popup Notifier: {'✅' if hasattr(alert_system, 'popup_notifier') else '❌'}")
        
        print("\n🎉 Alert system creation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Alert system creation test failed: {e}")
        return False

def main():
    """Run all simple tests"""
    print("=" * 60)
    print("STAGE 3 - SIMPLE FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("ML Detector Creation", test_ml_detector_creation),
        ("Alert System Creation", test_alert_system_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed ({passed/len(results)*100:.1f}%)")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Stage 3 components are ready.")
        return 0
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)