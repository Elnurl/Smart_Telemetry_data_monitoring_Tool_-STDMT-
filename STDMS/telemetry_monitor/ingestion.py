#!/usr/bin/env python3
"""
Telemetry Data Ingestion Module
Handles data sources including simulation for testing
"""

import random
import time
import threading
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TelemetryData:
    """Data structure for telemetry readings"""
    satellite_id: str
    timestamp: datetime
    temperature: float
    battery_voltage: float
    solar_power: float
    attitude_x: float
    attitude_y: float
    attitude_z: float
    orbit_altitude: float
    signal_strength: float


class TelemetrySimulator:
    """Simulates telemetry data from multiple satellites"""
    
    def __init__(self, enable_database_storage: bool = True):
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.thread = None
        self.data_storage = []
        self.satellites = ["SAT-001", "SAT-002", "SAT-003", "SAT-004"]
        self.anomalies_count = 0
        self.data_points_count = 0
        self.last_update = None
        self.latest_data = None
        self.enable_database_storage = enable_database_storage
        self.storage_service = None
        
        # Initialize database storage if enabled
        if self.enable_database_storage:
            try:
                from storage import get_storage_service
                self.storage_service = get_storage_service()
                self.logger.info("Database storage enabled for simulator")
            except ImportError:
                self.logger.warning("Database storage not available, continuing with memory-only storage")
                self.enable_database_storage = False
        
        # Initialize ML anomaly detection
        self.enable_ml_detection = True
        self.ml_detector = None
        try:
            from anomaly import get_anomaly_detector
            self.ml_detector = get_anomaly_detector()
            self.logger.info("ML anomaly detection enabled for simulator")
        except ImportError:
            self.logger.warning("ML anomaly detection not available")
            self.enable_ml_detection = False
        
        # Satellite base parameters
        self.sat_params = {
            "SAT-001": {"temp_base": 25, "battery_base": 12.5, "altitude_base": 450},
            "SAT-002": {"temp_base": 22, "battery_base": 12.8, "altitude_base": 470},
            "SAT-003": {"temp_base": 28, "battery_base": 12.3, "altitude_base": 460},
            "SAT-004": {"temp_base": 24, "battery_base": 12.6, "altitude_base": 440}
        }
        
    def start(self):
        """Start the telemetry simulation"""
        if not self.is_running:
            self.is_running = True
            
            # Start database storage service if enabled
            if self.enable_database_storage and self.storage_service:
                self.storage_service.start()
                self.logger.info("Database storage service started")
            
            self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.thread.start()
            self.logger.info("Telemetry simulator started")
            
    def stop(self):
        """Stop the telemetry simulation"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            
        # Stop database storage service if enabled
        if self.enable_database_storage and self.storage_service:
            self.storage_service.stop()
            self.logger.info("Database storage service stopped")
            
        self.logger.info("Telemetry simulator stopped")
        
    def clear_data(self):
        """Clear all stored data"""
        self.data_storage.clear()
        self.anomalies_count = 0
        self.data_points_count = 0
        self.last_update = None
        self.latest_data = None
        self.logger.info("Simulator data cleared")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = {
            "satellites": len(self.satellites),
            "data_points": self.data_points_count,
            "anomalies": self.anomalies_count,
            "last_update": self.last_update.strftime("%H:%M:%S") if self.last_update else "Never"
        }
        
        # Add database storage stats if enabled
        if self.enable_database_storage and self.storage_service:
            storage_stats = self.storage_service.get_stats()
            stats.update({
                "db_stored_count": storage_stats.get('stored_count', 0),
                "db_queue_size": storage_stats.get('queue_size', 0),
                "db_anomalies_stored": storage_stats.get('anomalies_stored', 0)
            })
        
        # Add ML detection stats if enabled
        if self.enable_ml_detection and self.ml_detector:
            try:
                ml_status = self.ml_detector.get_model_status()
                stats.update({
                    "ml_models_trained": sum(ml_status['models_trained'].values()),
                    "ml_ensemble_threshold": ml_status['ensemble_threshold']
                })
            except Exception as e:
                self.logger.error(f"Failed to get ML stats: {e}")
        
        return stats
        
    def get_recent_data(self, limit: int = 100) -> List[TelemetryData]:
        """Get recent telemetry data"""
        return self.data_storage[-limit:] if len(self.data_storage) > limit else self.data_storage
    
    def get_latest_data(self) -> Optional[TelemetryData]:
        """Get the most recent telemetry data point"""
        return self.latest_data
        
    def _simulation_loop(self):
        """Main simulation loop running in separate thread"""
        self.logger.info("Starting telemetry simulation loop")
        
        while self.is_running:
            try:
                # Generate data for each satellite
                for sat_id in self.satellites:
                    data = self._generate_telemetry_data(sat_id)
                    self.data_storage.append(data)
                    self.latest_data = data  # Update latest data for visualization
                    self.data_points_count += 1
                    
                    # Check for anomalies using rule-based detection
                    rule_based_anomaly = self._is_anomaly(data)
                    anomaly_type = None
                    
                    # ML-based anomaly detection
                    ml_anomaly_flag = False
                    ml_anomaly_score = 0.0
                    
                    if self.enable_ml_detection and self.ml_detector:
                        try:
                            # Convert data to dict format for ML detection
                            data_dict = {
                                'satellite_id': data.satellite_id,
                                'timestamp': data.timestamp,
                                'temperature': data.temperature,
                                'battery_voltage': data.battery_voltage,
                                'solar_power': data.solar_power,
                                'attitude_x': data.attitude_x,
                                'attitude_y': data.attitude_y,
                                'attitude_z': data.attitude_z,
                                'orbit_altitude': data.orbit_altitude,
                                'signal_strength': data.signal_strength
                            }
                            
                            # Get ML anomaly scores
                            ml_results = self.ml_detector.predict_anomaly([data_dict])
                            if ml_results:
                                ml_result = ml_results[0]
                                ml_anomaly_flag = ml_result.is_anomaly
                                ml_anomaly_score = ml_result.combined_score
                                
                                if ml_anomaly_flag:
                                    anomaly_type = f"ml_{ml_result.anomaly_type}"
                                    self.logger.info(f"ML Anomaly detected in {sat_id}: {anomaly_type} (score: {ml_anomaly_score:.3f})")
                        
                        except Exception as e:
                            self.logger.error(f"ML anomaly detection failed: {e}")
                    
                    # Combine rule-based and ML-based detection
                    final_anomaly_flag = rule_based_anomaly or ml_anomaly_flag
                    
                    if rule_based_anomaly and not ml_anomaly_flag:
                        anomaly_type = self._get_anomaly_type(data)
                        self.logger.warning(f"Rule-based anomaly in {sat_id}: {anomaly_type}")
                    
                    if final_anomaly_flag:
                        self.anomalies_count += 1
                    
                    # Store in database if enabled
                    if self.enable_database_storage and self.storage_service:
                        self.storage_service.queue_telemetry(data, final_anomaly_flag, anomaly_type)
                        
                self.last_update = datetime.now()
                
                # Keep only last 1000 data points to prevent memory issues
                if len(self.data_storage) > 1000:
                    self.data_storage = self.data_storage[-1000:]
                    
                # Sleep for simulation interval (2 seconds)
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error in simulation loop: {e}")
                time.sleep(1)
                
        self.logger.info("Telemetry simulation loop ended")
        
    def _generate_telemetry_data(self, sat_id: str) -> TelemetryData:
        """Generate realistic telemetry data for a satellite"""
        params = self.sat_params[sat_id]
        now = datetime.now()
        
        # Add some realistic variations and noise
        temp_variation = random.uniform(-5, 15)  # Temperature swings in space
        battery_variation = random.uniform(-0.5, 0.3)  # Battery discharge/charge cycles
        altitude_variation = random.uniform(-10, 10)  # Orbital variations
        
        # Simulate day/night cycles affecting solar power and temperature
        hour_factor = abs(12 - now.hour) / 12.0  # 0 at noon, 1 at midnight
        solar_eclipse = hour_factor > 0.7  # Simulate eclipse
        
        return TelemetryData(
            satellite_id=sat_id,
            timestamp=now,
            temperature=params["temp_base"] + temp_variation + (hour_factor * 10),
            battery_voltage=params["battery_base"] + battery_variation - (0.2 if solar_eclipse else 0),
            solar_power=0.0 if solar_eclipse else random.uniform(8.5, 12.0) * (1 - hour_factor * 0.3),
            attitude_x=random.uniform(-180, 180),
            attitude_y=random.uniform(-90, 90),
            attitude_z=random.uniform(-180, 180),
            orbit_altitude=params["altitude_base"] + altitude_variation,
            signal_strength=random.uniform(65, 95) - (hour_factor * 10)  # Weaker signal when farther
        )
        
    def _is_anomaly(self, data: TelemetryData) -> bool:
        """Simple anomaly detection logic"""
        # Define thresholds
        anomaly_conditions = [
            data.temperature > 50 or data.temperature < -20,  # Extreme temperatures
            data.battery_voltage < 11.0 or data.battery_voltage > 14.0,  # Battery issues
            data.signal_strength < 40,  # Poor signal
            data.orbit_altitude < 400 or data.orbit_altitude > 500  # Orbital anomalies
        ]
        
        return any(anomaly_conditions)
    
    def _get_anomaly_type(self, data: TelemetryData) -> str:
        """Determine the type of anomaly"""
        if data.temperature > 50:
            return "temperature_high"
        elif data.temperature < -20:
            return "temperature_low"
        elif data.battery_voltage < 11.0:
            return "battery_low"
        elif data.battery_voltage > 14.0:
            return "battery_high"
        elif data.signal_strength < 40:
            return "signal_weak"
        elif data.orbit_altitude < 400:
            return "altitude_low"
        elif data.orbit_altitude > 500:
            return "altitude_high"
        else:
            return "unknown_anomaly"
        
    def save_data_to_file(self, filename: str = None):
        """Save current data to JSON file"""
        if not filename:
            filename = f"telemetry_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        filepath = data_dir / filename
        
        # Convert data to JSON serializable format
        json_data = []
        for data in self.data_storage:
            json_data.append({
                "satellite_id": data.satellite_id,
                "timestamp": data.timestamp.isoformat(),
                "temperature": data.temperature,
                "battery_voltage": data.battery_voltage,
                "solar_power": data.solar_power,
                "attitude_x": data.attitude_x,
                "attitude_y": data.attitude_y,
                "attitude_z": data.attitude_z,
                "orbit_altitude": data.orbit_altitude,
                "signal_strength": data.signal_strength
            })
            
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        self.logger.info(f"Data saved to {filepath}")
        return filepath


class MQTTSimulator:
    """Future: MQTT data source simulation"""
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to MQTT broker (placeholder)"""
        self.logger.info(f"MQTT connection to {self.broker_host}:{self.broker_port} - Not implemented yet")
        
    def subscribe(self, topic: str):
        """Subscribe to MQTT topic (placeholder)"""
        self.logger.info(f"MQTT subscription to {topic} - Not implemented yet")


class CSVDataLoader:
    """Future: Load telemetry data from CSV files"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> List[TelemetryData]:
        """Load data from CSV file (placeholder)"""
        self.logger.info(f"Loading CSV data from {self.csv_path} - Not implemented yet")
        return []


if __name__ == "__main__":
    # Test the simulator
    logging.basicConfig(level=logging.INFO)
    
    simulator = TelemetrySimulator()
    print("Starting telemetry simulator test...")
    
    simulator.start()
    time.sleep(10)  # Run for 10 seconds
    simulator.stop()
    
    print(f"Generated {len(simulator.data_storage)} data points")
    print(f"Stats: {simulator.get_stats()}")
    
    # Save test data
    simulator.save_data_to_file("test_data.json")