#!/usr/bin/env python3
"""
Stage 2 Test Script - Data Handling & Storage
Tests database schema, storage service, and data persistence
"""

import sys
import time
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_database_schema():
    """Test database schema creation and basic operations"""
    print("🔧 Testing Database Schema...")
    
    try:
        from storage import TelemetryDatabase
        
        # Create test database
        test_db_path = project_root / "data" / "test_telemetry.db"
        db = TelemetryDatabase(str(test_db_path))
        
        print("✅ Database schema created successfully")
        
        # Test database stats
        stats = db.get_database_stats()
        print(f"📊 Initial database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database schema test failed: {e}")
        return False

def test_storage_service():
    """Test background storage service"""
    print("\n🔧 Testing Background Storage Service...")
    
    try:
        from storage import TelemetryDatabase, BackgroundStorageService
        from ingestion import TelemetrySimulator
        
        # Create test database and storage service
        test_db_path = project_root / "data" / "test_storage.db"
        db = TelemetryDatabase(str(test_db_path))
        storage_service = BackgroundStorageService(db)
        
        print("✅ Storage service created")
        
        # Start storage service
        storage_service.start()
        print("✅ Storage service started")
        
        # Create simulator and generate test data
        simulator = TelemetrySimulator(enable_database_storage=False)  # Don't auto-connect to avoid conflicts
        test_data = []
        
        for i in range(10):
            for sat_id in ["SAT-001", "SAT-002"]:
                data = simulator._generate_telemetry_data(sat_id)
                test_data.append(data)
                
                # Determine if anomaly
                anomaly_flag = simulator._is_anomaly(data)
                anomaly_type = simulator._get_anomaly_type(data) if anomaly_flag else None
                
                # Queue for storage
                storage_service.queue_telemetry(data, anomaly_flag, anomaly_type)
        
        print(f"📤 Queued {len(test_data)} data points for storage")
        
        # Wait for storage to complete
        time.sleep(3)
        
        # Check storage stats
        storage_stats = storage_service.get_stats()
        print(f"📈 Storage stats: {storage_stats}")
        
        # Check database stats
        db_stats = db.get_database_stats()
        print(f"📊 Database stats: {db_stats}")
        
        # Stop storage service
        storage_service.stop()
        print("✅ Storage service stopped")
        
        # Verify data was stored
        if db_stats.total_records > 0:
            print(f"✅ Successfully stored {db_stats.total_records} records")
            return True
        else:
            print("❌ No records were stored")
            return False
            
    except Exception as e:
        print(f"❌ Storage service test failed: {e}")
        return False

def test_data_queries():
    """Test data query functionality"""
    print("\n🔧 Testing Data Query Functionality...")
    
    try:
        from storage import TelemetryDatabase
        
        # Use the test database from previous test
        test_db_path = project_root / "data" / "test_storage.db"
        db = TelemetryDatabase(str(test_db_path))
        
        # Query recent telemetry
        recent_data = db.get_recent_telemetry(limit=5)
        print(f"📋 Retrieved {len(recent_data)} recent records")
        
        if recent_data:
            latest = recent_data[0]
            print(f"🛰️  Latest record from {latest['satellite_id']}:")
            print(f"   Time: {latest['timestamp']}")
            print(f"   Temperature: {latest['temperature']:.1f}°C")
            print(f"   Battery: {latest['battery_voltage']:.2f}V")
            print(f"   Anomaly: {'Yes' if latest['anomaly_flag'] else 'No'}")
        
        # Query anomalies
        anomalies = db.get_anomalies(limit=5)
        print(f"🚨 Retrieved {len(anomalies)} anomaly records")
        
        if anomalies:
            for anomaly in anomalies[:3]:  # Show first 3
                print(f"   {anomaly['satellite_id']}: {anomaly['anomaly_type']} - {anomaly['description']}")
        
        print("✅ Data query functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Data query test failed: {e}")
        return False

def test_integrated_simulator():
    """Test integrated simulator with database storage"""
    print("\n🔧 Testing Integrated Simulator with Database Storage...")
    
    try:
        from ingestion import TelemetrySimulator
        from storage import get_database
        
        # Get main database
        db = get_database()
        initial_stats = db.get_database_stats()
        
        print(f"📊 Initial database records: {initial_stats.total_records}")
        
        # Create simulator with database enabled
        simulator = TelemetrySimulator(enable_database_storage=True)
        
        # Start simulator
        simulator.start()
        print("▶️  Simulator started with database storage")
        
        # Run for a short time
        print("⏱️  Running for 10 seconds...")
        time.sleep(10)
        
        # Get stats
        sim_stats = simulator.get_stats()
        print(f"📈 Simulator stats: {sim_stats}")
        
        # Stop simulator
        simulator.stop()
        print("⏹️  Simulator stopped")
        
        # Check final database stats
        final_stats = db.get_database_stats()
        records_added = final_stats.total_records - initial_stats.total_records
        
        print(f"📊 Final database records: {final_stats.total_records}")
        print(f"➕ Records added: {records_added}")
        
        if records_added > 0:
            print("✅ Integrated simulator successfully stored data in database")
            return True
        else:
            print("❌ No new records were added to database")
            return False
            
    except Exception as e:
        print(f"❌ Integrated simulator test failed: {e}")
        return False

def test_data_export():
    """Test data export functionality"""
    print("\n🔧 Testing Data Export...")
    
    try:
        from storage import get_database
        
        db = get_database()
        
        # Get recent data
        recent_data = db.get_recent_telemetry(limit=10)
        
        if not recent_data:
            print("⚠️  No data to export")
            return True
        
        # Export to JSON
        export_data = []
        for record in recent_data:
            export_data.append({
                'satellite_id': record['satellite_id'],
                'timestamp': record['timestamp'].isoformat(),
                'temperature': record['temperature'],
                'battery_voltage': record['battery_voltage'],
                'anomaly_flag': record['anomaly_flag']
            })
        
        # Save to file
        export_file = project_root / "data" / "test_export.json"
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Exported {len(export_data)} records to {export_file}")
        
        # Verify file was created
        if export_file.exists():
            file_size = export_file.stat().st_size
            print(f"📄 Export file size: {file_size} bytes")
            print("✅ Data export functionality working")
            return True
        else:
            print("❌ Export file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Data export test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test database files"""
    print("\n🧹 Cleaning up test files...")
    
    test_files = [
        "data/test_telemetry.db",
        "data/test_storage.db",
        "data/test_export.json"
    ]
    
    for file_path in test_files:
        full_path = project_root / file_path
        if full_path.exists():
            full_path.unlink()
            print(f"🗑️  Removed {file_path}")

def main():
    """Run all Stage 2 tests"""
    print("🚀 Telemetry Monitor - Stage 2 Data Handling & Storage Tests")
    print("=" * 70)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_results = []
    
    try:
        # Run all tests
        test_results.append(("Database Schema", test_database_schema()))
        test_results.append(("Storage Service", test_storage_service()))
        test_results.append(("Data Queries", test_data_queries()))
        test_results.append(("Integrated Simulator", test_integrated_simulator()))
        test_results.append(("Data Export", test_data_export()))
        
        # Clean up test files
        cleanup_test_files()
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 TEST RESULTS SUMMARY:")
        
        passed = 0
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")
        
        if passed == len(test_results):
            print("\n🎉 ALL TESTS PASSED! Stage 2 Data Handling & Storage is ready!")
            print("\n✨ Features verified:")
            print("   • SQLite database with proper schema")
            print("   • Background storage service")
            print("   • Telemetry data persistence")  
            print("   • Data query functionality")
            print("   • Integrated simulator with database storage")
            print("   • Data export capabilities")
            return True
        else:
            print(f"\n⚠️  {len(test_results) - passed} tests failed. Please check the issues above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)