#!/usr/bin/env python3
"""
Test script to verify the telemetry monitor application components
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from ingestion import TelemetrySimulator

def test_simulator():
    """Test the telemetry simulator"""
    print("🔧 Testing Telemetry Simulator...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create simulator
    simulator = TelemetrySimulator()
    
    # Test basic functionality
    print(f"📊 Initial stats: {simulator.get_stats()}")
    
    # Start simulator
    print("▶️  Starting simulator...")
    simulator.start()
    
    # Run for a few seconds
    print("⏱️  Running for 8 seconds...")
    time.sleep(8)
    
    # Get stats
    stats = simulator.get_stats()
    print(f"📈 Final stats: {stats}")
    
    # Get recent data
    recent_data = simulator.get_recent_data(5)
    print(f"📋 Recent data points: {len(recent_data)}")
    
    if recent_data:
        latest = recent_data[-1]
        print(f"🛰️  Latest data from {latest.satellite_id}:")
        print(f"   Temperature: {latest.temperature:.1f}°C")
        print(f"   Battery: {latest.battery_voltage:.2f}V")
        print(f"   Signal: {latest.signal_strength:.1f}dBm")
    
    # Stop simulator
    print("⏹️  Stopping simulator...")
    simulator.stop()
    
    # Save test data
    filepath = simulator.save_data_to_file("test_run.json")
    print(f"💾 Data saved to: {filepath}")
    
    print("✅ Simulator test completed successfully!")
    return True

def test_logging():
    """Test the logging system"""
    print("\n🔧 Testing Logging System...")
    
    # Ensure logs directory exists
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Test log file
    log_file = logs_dir / "system.log"
    
    if log_file.exists():
        print(f"📄 Log file exists: {log_file}")
        
        # Read last few lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"📝 Log file has {len(lines)} lines")
            
            if lines:
                print("📋 Last 3 log entries:")
                for line in lines[-3:]:
                    print(f"   {line.strip()}")
    else:
        print("❌ Log file not found")
        return False
    
    print("✅ Logging test completed!")
    return True

def test_directories():
    """Test directory structure"""
    print("\n🔧 Testing Directory Structure...")
    
    required_dirs = ["data", "models", "logs", "ui"]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            return False
    
    required_files = ["main.py", "ingestion.py", "storage.py", "anomaly.py", "alerts.py"]
    
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ {file_name} file exists")
        else:
            print(f"❌ {file_name} file missing")
            return False
    
    print("✅ Directory structure test completed!")
    return True

def main():
    """Run all tests"""
    print("🚀 Telemetry Monitor - Stage 1 Foundation Test")
    print("=" * 50)
    
    try:
        # Test directory structure
        test_directories()
        
        # Test logging
        test_logging()
        
        # Test simulator
        test_simulator()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Stage 1 Foundation is ready!")
        print("\n✨ You can now run the full application with:")
        print("   python main.py")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)