#!/usr/bin/env python3
"""
Test script for Stage 3 - Smart Anomaly Detection (ML)
Tests ML anomaly detection, alert system, and GUI integration.
"""

import sys
import asyncio
import logging
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telemetry_monitor.ingestion import TelemetrySimulator
from telemetry_monitor.storage import TelemetryDatabase, BackgroundStorageService
from telemetry_monitor.anomaly import SmartAnomalyDetector
from telemetry_monitor.alerts import SmartAlertSystem

class Stage3TestRunner:
    def __init__(self):
        self.logger = self._setup_logging()
        self.project_root = project_root
        self.database = None
        self.storage_service = None
        self.ml_detector = None
        self.alert_system = None
        self.simulator = None
        
    def _setup_logging(self):
        """Setup logging for test runner"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(project_root / "logs" / "test_stage3.log")
            ]
        )
        return logging.getLogger("Stage3Test")
    
    async def setup_components(self):
        """Initialize all Stage 3 components"""
        try:
            self.logger.info("Setting up Stage 3 components...")
            
            # Initialize database
            db_path = self.project_root / "data" / "telemetry.db"
            self.database = TelemetryDatabase(str(db_path))
            
            # Initialize storage service
            self.storage_service = BackgroundStorageService(self.database)
            await self.storage_service.start()
            
            # Initialize ML anomaly detector
            self.ml_detector = SmartAnomalyDetector(
                database=self.database,
                storage_service=self.storage_service
            )
            
            # Initialize alert system
            self.alert_system = SmartAlertSystem(
                database=self.database,
                config_path=self.project_root / "config" / "alert_config.json"
            )
            
            # Initialize simulator with ML integration
            self.simulator = TelemetrySimulator(
                storage_service=self.storage_service,
                ml_detector=self.ml_detector,
                alert_system=self.alert_system
            )
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup components: {e}")
            return False
    
    async def test_data_generation(self):
        """Test data generation and storage"""
        try:
            self.logger.info("Testing data generation...")
            
            # Generate some initial data
            await self.simulator.start()
            await asyncio.sleep(5)  # Let it generate some data
            await self.simulator.stop()
            
            # Check database stats
            stats = self.database.get_database_stats()
            self.logger.info(f"Generated {stats.total_records} telemetry records")
            
            if stats.total_records < 10:
                self.logger.warning("Not enough data generated for proper testing")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data generation test failed: {e}")
            return False
    
    async def test_ml_model_training(self):
        """Test ML model training"""
        try:
            self.logger.info("Testing ML model training...")
            
            # Ensure we have enough data
            stats = self.database.get_database_stats()
            if stats.total_records < 50:
                self.logger.info("Generating more data for ML training...")
                await self.simulator.start()
                await asyncio.sleep(10)  # Generate more data
                await self.simulator.stop()
                
                stats = self.database.get_database_stats()
                self.logger.info(f"Now have {stats.total_records} records")
            
            # Test model training
            start_time = time.time()
            training_results = self.ml_detector.train_models(retrain=True)
            training_time = time.time() - start_time
            
            self.logger.info(f"ML training completed in {training_time:.2f}s")
            
            # Log training results
            for model, results in training_results.items():
                self.logger.info(f"{model.title()} Model:")
                self.logger.info(f"  Training time: {results.get('training_time', 0):.2f}s")
                self.logger.info(f"  Threshold: {results.get('threshold', 0):.4f}")
                self.logger.info(f"  Samples: {results.get('n_samples', results.get('n_sequences', 0))}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"ML model training test failed: {e}")
            return False
    
    async def test_anomaly_detection(self):
        """Test anomaly detection with simulated anomalies"""
        try:
            self.logger.info("Testing anomaly detection...")
            
            # Create some test data with known anomalies
            from telemetry_monitor.models import TelemetryData
            
            # Normal data point
            normal_data = TelemetryData(
                satellite_id="TEST-001",
                timestamp=datetime.now(),
                battery_voltage=12.5,
                solar_current=2.3,
                temperature=20.0,
                memory_usage=45.2,
                cpu_usage=35.8,
                signal_strength=-75.2
            )
            
            # Anomalous data point (extreme values)
            anomaly_data = TelemetryData(
                satellite_id="TEST-001",
                timestamp=datetime.now(),
                battery_voltage=8.0,  # Very low
                solar_current=0.1,    # Very low
                temperature=85.0,     # Very high
                memory_usage=95.0,    # Very high
                cpu_usage=98.0,       # Very high
                signal_strength=-120.0  # Very weak
            )
            
            # Test normal data
            normal_score = self.ml_detector.detect_anomaly(normal_data)
            self.logger.info(f"Normal data anomaly score: {normal_score.combined_score:.3f}")
            
            # Test anomalous data
            anomaly_score = self.ml_detector.detect_anomaly(anomaly_data)
            self.logger.info(f"Anomalous data score: {anomaly_score.combined_score:.3f}")
            
            # Verify detection
            if anomaly_score.combined_score > normal_score.combined_score:
                self.logger.info("✅ Anomaly detection working correctly")
                return True
            else:
                self.logger.warning("❌ Anomaly detection may not be working correctly")
                return False
            
        except Exception as e:
            self.logger.error(f"Anomaly detection test failed: {e}")
            return False
    
    async def test_alert_system(self):
        """Test alert system functionality"""
        try:
            self.logger.info("Testing alert system...")
            
            # Create test anomaly scores
            from telemetry_monitor.models import AnomalyScore, TelemetryData
            
            high_anomaly = AnomalyScore(
                combined_score=0.95,
                isolation_forest_score=0.90,
                autoencoder_score=0.88,
                lstm_score=0.92,
                confidence=0.87,
                anomaly_type="EXTREME_VALUES"
            )
            
            test_data = TelemetryData(
                satellite_id="TEST-ALERT",
                timestamp=datetime.now(),
                battery_voltage=5.0,  # Critical low
                solar_current=0.0,
                temperature=90.0,     # Critical high
                memory_usage=98.0,
                cpu_usage=99.0,
                signal_strength=-130.0
            )
            
            # Test alert creation
            alerts_before = len(self.alert_system.get_alert_history())
            
            await self.alert_system.process_anomaly(test_data, high_anomaly)
            
            alerts_after = len(self.alert_system.get_alert_history())
            
            if alerts_after > alerts_before:
                self.logger.info("✅ Alert system created new alert")
                
                # Get recent alerts
                recent_alerts = self.alert_system.get_alert_history(limit=1)
                if recent_alerts:
                    alert = recent_alerts[0]
                    self.logger.info(f"Alert details: {alert.severity} - {alert.message}")
                
                return True
            else:
                self.logger.warning("❌ Alert system did not create expected alert")
                return False
            
        except Exception as e:
            self.logger.error(f"Alert system test failed: {e}")
            return False
    
    async def test_integrated_simulation(self):
        """Test full integrated simulation with ML and alerts"""
        try:
            self.logger.info("Testing integrated simulation...")
            
            # Enable anomaly injection for testing
            self.simulator.anomaly_probability = 0.3  # 30% chance of anomalies
            
            # Run simulation for a bit
            await self.simulator.start()
            await asyncio.sleep(15)  # Run for 15 seconds
            await self.simulator.stop()
            
            # Check results
            stats = self.database.get_database_stats()
            alerts = self.alert_system.get_alert_history(limit=10)
            
            self.logger.info(f"Simulation generated {stats.total_records} total records")
            self.logger.info(f"Found {len(alerts)} alerts")
            
            # Show some alert details
            if alerts:
                for i, alert in enumerate(alerts[-3:]):  # Last 3 alerts
                    self.logger.info(f"Alert {i+1}: {alert.severity} - {alert.satellite_id} - {alert.anomaly_score.combined_score:.3f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Integrated simulation test failed: {e}")
            return False
    
    async def test_data_export(self):
        """Test data export functionality"""
        try:
            self.logger.info("Testing data export...")
            
            # Export recent data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            data = self.database.get_telemetry_data(
                start_time=start_time,
                end_time=end_time,
                limit=100
            )
            
            if not data:
                self.logger.warning("No data to export")
                return False
            
            # Export to JSON
            export_data = []
            for record in data:
                export_data.append({
                    'satellite_id': record.satellite_id,
                    'timestamp': record.timestamp.isoformat(),
                    'battery_voltage': record.battery_voltage,
                    'solar_current': record.solar_current,
                    'temperature': record.temperature,
                    'memory_usage': record.memory_usage,
                    'cpu_usage': record.cpu_usage,
                    'signal_strength': record.signal_strength
                })
            
            export_file = self.project_root / "data" / f"test_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"✅ Exported {len(export_data)} records to {export_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Data export test failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup test resources"""
        try:
            if self.simulator:
                await self.simulator.stop()
            
            if self.storage_service:
                await self.storage_service.stop()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    async def run_all_tests(self):
        """Run all Stage 3 tests"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING STAGE 3 - SMART ANOMALY DETECTION (ML) TESTS")
        self.logger.info("=" * 60)
        
        test_results = {
            "setup": False,
            "data_generation": False,
            "ml_training": False,
            "anomaly_detection": False,
            "alert_system": False,
            "integrated_simulation": False,
            "data_export": False
        }
        
        try:
            # Setup
            test_results["setup"] = await self.setup_components()
            if not test_results["setup"]:
                self.logger.error("Setup failed - cannot continue tests")
                return test_results
            
            # Data generation
            test_results["data_generation"] = await self.test_data_generation()
            
            # ML model training
            test_results["ml_training"] = await self.test_ml_model_training()
            
            # Anomaly detection
            test_results["anomaly_detection"] = await self.test_anomaly_detection()
            
            # Alert system
            test_results["alert_system"] = await self.test_alert_system()
            
            # Integrated simulation
            test_results["integrated_simulation"] = await self.test_integrated_simulation()
            
            # Data export
            test_results["data_export"] = await self.test_data_export()
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
        
        finally:
            await self.cleanup()
        
        # Print results summary
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3 TEST RESULTS SUMMARY")
        self.logger.info("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        
        self.logger.info("-" * 60)
        self.logger.info(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            self.logger.info("🎉 ALL STAGE 3 TESTS PASSED!")
        elif passed >= total * 0.8:
            self.logger.info("✅ Most tests passed - Stage 3 mostly functional")
        else:
            self.logger.warning("⚠️ Multiple test failures - Stage 3 needs attention")
        
        return test_results

async def main():
    """Main test function"""
    print("Testing Stage 3 - Smart Anomaly Detection (ML)")
    print("This will test ML models, anomaly detection, and alert system")
    print("-" * 60)
    
    test_runner = Stage3TestRunner()
    results = await test_runner.run_all_tests()
    
    # Return appropriate exit code
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print("\n🎉 All tests passed! Stage 3 is ready to use.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)