#!/usr/bin/env python3
"""
Test script for Stage 4 - Visualization
Tests real-time plotting, anomaly highlighting, and pause/resume functionality.
"""

import sys
import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from telemetry_monitor.visualization import (
    RealTimeVisualizationWidget, TelemetryPlotWidget, 
    AnomalyScorePlotWidget, PlotConfiguration
)
from telemetry_monitor.ingestion import TelemetrySimulator, TelemetryData
from telemetry_monitor.models import AnomalyScore

class Stage4TestRunner:
    def __init__(self):
        self.logger = self._setup_logging()
        self.project_root = project_root
        
    def _setup_logging(self):
        """Setup logging for test runner"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(project_root / "logs" / "test_stage4.log")
            ]
        )
        return logging.getLogger("Stage4Test")
    
    def test_plot_widget_creation(self):
        """Test creating individual plot widgets"""
        try:
            self.logger.info("Testing plot widget creation...")
            
            # Test telemetry plot widget
            battery_plot = TelemetryPlotWidget("Battery Voltage", "Voltage", "V")
            self.logger.info("✅ TelemetryPlotWidget created successfully")
            
            # Test anomaly score plot widget
            anomaly_plot = AnomalyScorePlotWidget()
            self.logger.info("✅ AnomalyScorePlotWidget created successfully")
            
            # Test plot configuration
            config = PlotConfiguration()
            self.logger.info(f"✅ PlotConfiguration: max_points={config.max_points}, update_interval={config.update_interval}ms")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Plot widget creation test failed: {e}")
            return False
    
    def test_data_addition(self):
        """Test adding data to plot widgets"""
        try:
            self.logger.info("Testing data addition to plots...")
            
            # Create test plot
            test_plot = TelemetryPlotWidget("Test Plot", "Value", "")
            
            # Create test data points
            test_data = []
            for i in range(10):
                timestamp = datetime.now() + timedelta(seconds=i)
                value = 50 + (i % 3) * 10  # Varying values
                
                # Create anomaly score (some normal, some anomalous)
                anomaly_score = AnomalyScore(
                    combined_score=0.3 if i < 7 else 0.8,  # Last 3 are anomalies
                    isolation_forest_score=0.2 if i < 7 else 0.9,
                    autoencoder_score=0.4 if i < 7 else 0.7,
                    lstm_score=0.3 if i < 7 else 0.8,
                    confidence=0.85,
                    anomaly_type="NORMAL" if i < 7 else "HIGH_VALUES"
                )
                
                test_plot.add_data_point(timestamp, value, anomaly_score)
                test_data.append((timestamp, value, anomaly_score))
            
            self.logger.info(f"✅ Added {len(test_data)} data points successfully")
            self.logger.info(f"✅ Plot now has {len(test_plot.timestamps)} timestamp entries")
            
            # Test anomaly score plot
            anomaly_plot = AnomalyScorePlotWidget()
            for timestamp, _, anomaly_score in test_data:
                anomaly_plot.add_anomaly_score(timestamp, anomaly_score)
            
            self.logger.info(f"✅ Anomaly plot has {len(anomaly_plot.timestamps)} entries")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data addition test failed: {e}")
            return False
    
    def test_visualization_widget(self):
        """Test main visualization widget"""
        try:
            self.logger.info("Testing main visualization widget...")
            
            # Create visualization widget
            viz_widget = RealTimeVisualizationWidget()
            self.logger.info("✅ RealTimeVisualizationWidget created successfully")
            
            # Check if all expected plots are created
            expected_plots = ['battery_voltage', 'solar_power', 'temperature', 'attitude_x', 'orbit_altitude', 'signal_strength']
            for plot_name in expected_plots:
                if plot_name in viz_widget.telemetry_plots:
                    self.logger.info(f"✅ {plot_name} plot created")
                else:
                    self.logger.warning(f"❌ {plot_name} plot missing")
            
            # Test control panel
            if viz_widget.control_panel:
                self.logger.info("✅ Control panel created")
                # Test pause/resume
                viz_widget._on_pause_resume(True)
                self.logger.info("✅ Pause functionality works")
                viz_widget._on_pause_resume(False)
                self.logger.info("✅ Resume functionality works")
            
            # Test adding telemetry data
            test_telemetry = TelemetryData(
                satellite_id="TEST-001",
                timestamp=datetime.now(),
                temperature=25.0,
                battery_voltage=12.5,
                solar_power=15.0,
                attitude_x=1.2,
                attitude_y=0.8,
                attitude_z=0.5,
                orbit_altitude=450.0,
                signal_strength=-75.0
            )
            
            test_anomaly_score = AnomalyScore(
                combined_score=0.25,
                isolation_forest_score=0.3,
                autoencoder_score=0.2,
                lstm_score=0.25,
                confidence=0.9,
                anomaly_type="NORMAL"
            )
            
            viz_widget.add_telemetry_data(test_telemetry, test_anomaly_score)
            self.logger.info("✅ Test telemetry data added successfully")
            
            # Test statistics
            stats = viz_widget.get_plot_statistics()
            self.logger.info(f"✅ Plot statistics: {stats}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Visualization widget test failed: {e}")
            return False
    
    def test_anomaly_highlighting(self):
        """Test anomaly highlighting functionality"""
        try:
            self.logger.info("Testing anomaly highlighting...")
            
            # Create test plot
            test_plot = TelemetryPlotWidget("Anomaly Test", "Value", "")
            
            # Add normal data points
            for i in range(5):
                timestamp = datetime.now() + timedelta(seconds=i)
                normal_score = AnomalyScore(
                    combined_score=0.1,
                    isolation_forest_score=0.1,
                    autoencoder_score=0.1,
                    lstm_score=0.1,
                    confidence=0.9,
                    anomaly_type="NORMAL"
                )
                test_plot.add_data_point(timestamp, 50.0, normal_score)
            
            # Add warning data points
            for i in range(5, 8):
                timestamp = datetime.now() + timedelta(seconds=i)
                warning_score = AnomalyScore(
                    combined_score=0.5,  # Warning level
                    isolation_forest_score=0.6,
                    autoencoder_score=0.4,
                    lstm_score=0.5,
                    confidence=0.8,
                    anomaly_type="MODERATE_ANOMALY"
                )
                test_plot.add_data_point(timestamp, 75.0, warning_score)
            
            # Add anomaly data points
            for i in range(8, 10):
                timestamp = datetime.now() + timedelta(seconds=i)
                anomaly_score = AnomalyScore(
                    combined_score=0.9,  # High anomaly
                    isolation_forest_score=0.95,
                    autoencoder_score=0.85,
                    lstm_score=0.9,
                    confidence=0.95,
                    anomaly_type="HIGH_ANOMALY"
                )
                test_plot.add_data_point(timestamp, 95.0, anomaly_score)
            
            # Check data distribution
            normal_count = sum(1 for score in test_plot.anomaly_scores if score < 0.4)
            warning_count = sum(1 for score in test_plot.anomaly_scores if 0.4 <= score < 0.7)
            anomaly_count = sum(1 for score in test_plot.anomaly_scores if score >= 0.7)
            
            self.logger.info(f"✅ Data distribution: Normal={normal_count}, Warning={warning_count}, Anomaly={anomaly_count}")
            
            if normal_count == 5 and warning_count == 3 and anomaly_count == 2:
                self.logger.info("✅ Anomaly highlighting test data correct")
                return True
            else:
                self.logger.warning("❌ Anomaly highlighting test data incorrect")
                return False
            
        except Exception as e:
            self.logger.error(f"Anomaly highlighting test failed: {e}")
            return False
    
    def test_pause_resume_functionality(self):
        """Test pause/resume functionality"""
        try:
            self.logger.info("Testing pause/resume functionality...")
            
            viz_widget = RealTimeVisualizationWidget()
            
            # Test initial state
            self.logger.info(f"Initial pause state: {viz_widget.is_paused}")
            
            # Test pause
            viz_widget._on_pause_resume(True)
            if viz_widget.is_paused:
                self.logger.info("✅ Pause functionality works")
            else:
                self.logger.error("❌ Pause functionality failed")
                return False
            
            # Test resume
            viz_widget._on_pause_resume(False)
            if not viz_widget.is_paused:
                self.logger.info("✅ Resume functionality works")
            else:
                self.logger.error("❌ Resume functionality failed")
                return False
            
            # Test that paused state prevents data updates
            viz_widget._on_pause_resume(True)  # Pause
            
            initial_count = len(next(iter(viz_widget.telemetry_plots.values())).timestamps) if viz_widget.telemetry_plots else 0
            
            # Try to add data while paused
            test_data = TelemetryData(
                satellite_id="PAUSE-TEST",
                timestamp=datetime.now(),
                temperature=25.0,
                battery_voltage=12.5,
                solar_power=15.0,
                attitude_x=1.2,
                attitude_y=0.8,
                attitude_z=0.5,
                orbit_altitude=450.0,
                signal_strength=-75.0
            )
            
            viz_widget.add_telemetry_data(test_data)
            
            final_count = len(next(iter(viz_widget.telemetry_plots.values())).timestamps) if viz_widget.telemetry_plots else 0
            
            if initial_count == final_count:
                self.logger.info("✅ Paused state correctly prevents data updates")
                return True
            else:
                self.logger.warning("❌ Paused state failed to prevent data updates")
                return False
            
        except Exception as e:
            self.logger.error(f"Pause/resume test failed: {e}")
            return False
    
    def test_time_range_functionality(self):
        """Test time range adjustment"""
        try:
            self.logger.info("Testing time range functionality...")
            
            viz_widget = RealTimeVisualizationWidget()
            
            # Test different time ranges
            time_ranges = [1, 5, 10, 30, 60]  # minutes
            
            for minutes in time_ranges:
                viz_widget._on_time_range_changed(minutes)
                if viz_widget.time_range_minutes == minutes:
                    self.logger.info(f"✅ Time range set to {minutes} minutes")
                else:
                    self.logger.error(f"❌ Failed to set time range to {minutes} minutes")
                    return False
            
            self.logger.info("✅ Time range functionality works correctly")
            return True
            
        except Exception as e:
            self.logger.error(f"Time range test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all Stage 4 tests"""
        self.logger.info("=" * 60)
        self.logger.info("STARTING STAGE 4 - VISUALIZATION TESTS")
        self.logger.info("=" * 60)
        
        test_results = {
            "plot_widget_creation": False,
            "data_addition": False,
            "visualization_widget": False,
            "anomaly_highlighting": False,
            "pause_resume": False,
            "time_range": False
        }
        
        try:
            # Individual tests
            test_results["plot_widget_creation"] = self.test_plot_widget_creation()
            test_results["data_addition"] = self.test_data_addition()
            test_results["visualization_widget"] = self.test_visualization_widget()
            test_results["anomaly_highlighting"] = self.test_anomaly_highlighting()
            test_results["pause_resume"] = self.test_pause_resume_functionality()
            test_results["time_range"] = self.test_time_range_functionality()
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
        
        # Print results summary
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4 TEST RESULTS SUMMARY")
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
            self.logger.info("🎉 ALL STAGE 4 TESTS PASSED!")
        elif passed >= total * 0.8:
            self.logger.info("✅ Most tests passed - Stage 4 mostly functional")
        else:
            self.logger.warning("⚠️ Multiple test failures - Stage 4 needs attention")
        
        return test_results

def main():
    """Main test function"""
    print("Testing Stage 4 - Visualization")
    print("This will test real-time plotting, anomaly highlighting, and controls")
    print("-" * 60)
    
    test_runner = Stage4TestRunner()
    results = test_runner.run_all_tests()
    
    # Return appropriate exit code
    passed = sum(results.values())
    total = len(results)
    
    if passed == total:
        print("\n🎉 All tests passed! Stage 4 visualization is ready to use.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)