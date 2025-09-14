#!/usr/bin/env python3
"""
Telemetry Data Storage Module
SQLite database integration with PostgreSQL compatibility
"""

import sqlite3
import threading
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import queue
import time

# Import our telemetry data structure
import sys
project_root = Path(__file__).parent
sys.path.append(str(project_root))
from ingestion import TelemetryData


@dataclass
class DatabaseStats:
    """Database statistics"""
    total_records: int
    satellites_count: int
    anomalies_count: int
    latest_timestamp: Optional[datetime]
    oldest_timestamp: Optional[datetime]
    database_size_mb: float


class TelemetryDatabase:
    """SQLite database manager for telemetry data"""
    
    def __init__(self, db_path: str = None):
        """Initialize database connection and schema"""
        self.logger = logging.getLogger(__name__)
        
        # Set database path
        if db_path is None:
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "telemetry.db"
            
        self.db_path = str(db_path)
        self.connection_lock = threading.Lock()
        
        # Initialize database schema
        self._initialize_database()
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def _initialize_database(self):
        """Create database tables and indexes"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main telemetry data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    satellite_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    temperature REAL NOT NULL,
                    battery_voltage REAL NOT NULL,
                    solar_power REAL NOT NULL,
                    attitude_x REAL NOT NULL,
                    attitude_y REAL NOT NULL,
                    attitude_z REAL NOT NULL,
                    orbit_altitude REAL NOT NULL,
                    signal_strength REAL NOT NULL,
                    anomaly_flag BOOLEAN DEFAULT FALSE,
                    anomaly_type TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Anomalies table for detailed anomaly tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telemetry_id INTEGER,
                    satellite_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT DEFAULT 'MEDIUM',
                    description TEXT,
                    parameter_name TEXT,
                    parameter_value REAL,
                    threshold_value REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (telemetry_id) REFERENCES telemetry_data (id)
                )
            """)
            
            # System logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    level TEXT NOT NULL,
                    module TEXT,
                    message TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_data (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_satellite ON telemetry_data (satellite_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_anomaly ON telemetry_data (anomaly_flag)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_satellite ON anomalies (satellite_id)")
            
            conn.commit()
            self.logger.info("Database schema created successfully")
    
    @contextmanager
    def _get_connection(self):
        """Get thread-safe database connection"""
        with self.connection_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()
    
    def store_telemetry(self, data: TelemetryData, anomaly_flag: bool = False, 
                       anomaly_type: str = None) -> int:
        """Store telemetry data in database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO telemetry_data (
                        satellite_id, timestamp, temperature, battery_voltage, 
                        solar_power, attitude_x, attitude_y, attitude_z,
                        orbit_altitude, signal_strength, anomaly_flag, anomaly_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.satellite_id,
                    data.timestamp,
                    data.temperature,
                    data.battery_voltage,
                    data.solar_power,
                    data.attitude_x,
                    data.attitude_y,
                    data.attitude_z,
                    data.orbit_altitude,
                    data.signal_strength,
                    anomaly_flag,
                    anomaly_type
                ))
                
                telemetry_id = cursor.lastrowid
                conn.commit()
                
                # If anomaly, store detailed anomaly record
                if anomaly_flag and anomaly_type:
                    self._store_anomaly_record(telemetry_id, data, anomaly_type)
                
                return telemetry_id
                
        except Exception as e:
            self.logger.error(f"Failed to store telemetry data: {e}")
            raise
    
    def _store_anomaly_record(self, telemetry_id: int, data: TelemetryData, 
                            anomaly_type: str):
        """Store detailed anomaly record"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Determine severity and description based on anomaly type
                severity, description, param_name, param_value, threshold = self._analyze_anomaly(
                    data, anomaly_type
                )
                
                cursor.execute("""
                    INSERT INTO anomalies (
                        telemetry_id, satellite_id, timestamp, anomaly_type,
                        severity, description, parameter_name, parameter_value,
                        threshold_value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    telemetry_id,
                    data.satellite_id,
                    data.timestamp,
                    anomaly_type,
                    severity,
                    description,
                    param_name,
                    param_value,
                    threshold
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store anomaly record: {e}")
    
    def _analyze_anomaly(self, data: TelemetryData, anomaly_type: str) -> Tuple[str, str, str, float, float]:
        """Analyze anomaly and return details"""
        if "temperature" in anomaly_type.lower():
            if data.temperature > 45:
                return ("HIGH", f"Critical high temperature: {data.temperature:.1f}°C", 
                       "temperature", data.temperature, 45.0)
            elif data.temperature < -15:
                return ("HIGH", f"Critical low temperature: {data.temperature:.1f}°C", 
                       "temperature", data.temperature, -15.0)
        
        elif "battery" in anomaly_type.lower():
            if data.battery_voltage < 11.5:
                return ("CRITICAL", f"Low battery voltage: {data.battery_voltage:.2f}V", 
                       "battery_voltage", data.battery_voltage, 11.5)
            elif data.battery_voltage > 13.5:
                return ("MEDIUM", f"High battery voltage: {data.battery_voltage:.2f}V", 
                       "battery_voltage", data.battery_voltage, 13.5)
        
        elif "signal" in anomaly_type.lower():
            return ("MEDIUM", f"Poor signal strength: {data.signal_strength:.1f}dBm", 
                   "signal_strength", data.signal_strength, 50.0)
        
        elif "altitude" in anomaly_type.lower():
            return ("HIGH", f"Orbital altitude anomaly: {data.orbit_altitude:.1f}km", 
                   "orbit_altitude", data.orbit_altitude, 450.0)
        
        return ("MEDIUM", f"General anomaly detected", "unknown", 0.0, 0.0)
    
    def get_recent_telemetry(self, limit: int = 100, satellite_id: str = None,
                           hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get recent telemetry data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                query = """
                    SELECT * FROM telemetry_data 
                    WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours_back)
                
                params = []
                if satellite_id:
                    query += " AND satellite_id = ?"
                    params.append(satellite_id)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                result = []
                for row in rows:
                    result.append({
                        'id': row['id'],
                        'satellite_id': row['satellite_id'],
                        'timestamp': datetime.fromisoformat(row['timestamp']),
                        'temperature': row['temperature'],
                        'battery_voltage': row['battery_voltage'],
                        'solar_power': row['solar_power'],
                        'attitude_x': row['attitude_x'],
                        'attitude_y': row['attitude_y'],
                        'attitude_z': row['attitude_z'],
                        'orbit_altitude': row['orbit_altitude'],
                        'signal_strength': row['signal_strength'],
                        'anomaly_flag': bool(row['anomaly_flag']),
                        'anomaly_type': row['anomaly_type']
                    })
                
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get recent telemetry: {e}")
            return []
    
    def get_anomalies(self, limit: int = 50, resolved: bool = False) -> List[Dict[str, Any]]:
        """Get anomaly records"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM anomalies 
                    WHERE resolved = ?
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (resolved, limit))
                
                rows = cursor.fetchall()
                
                result = []
                for row in rows:
                    result.append({
                        'id': row['id'],
                        'satellite_id': row['satellite_id'],
                        'timestamp': datetime.fromisoformat(row['timestamp']),
                        'anomaly_type': row['anomaly_type'],
                        'severity': row['severity'],
                        'description': row['description'],
                        'parameter_name': row['parameter_name'],
                        'parameter_value': row['parameter_value'],
                        'threshold_value': row['threshold_value'],
                        'resolved': bool(row['resolved'])
                    })
                
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get anomalies: {e}")
            return []
    
    def get_database_stats(self) -> DatabaseStats:
        """Get database statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute("SELECT COUNT(*) FROM telemetry_data")
                total_records = cursor.fetchone()[0]
                
                # Unique satellites
                cursor.execute("SELECT COUNT(DISTINCT satellite_id) FROM telemetry_data")
                satellites_count = cursor.fetchone()[0]
                
                # Anomalies count
                cursor.execute("SELECT COUNT(*) FROM telemetry_data WHERE anomaly_flag = 1")
                anomalies_count = cursor.fetchone()[0]
                
                # Latest timestamp
                cursor.execute("SELECT MAX(timestamp) FROM telemetry_data")
                latest_timestamp_str = cursor.fetchone()[0]
                latest_timestamp = datetime.fromisoformat(latest_timestamp_str) if latest_timestamp_str else None
                
                # Oldest timestamp
                cursor.execute("SELECT MIN(timestamp) FROM telemetry_data")
                oldest_timestamp_str = cursor.fetchone()[0]
                oldest_timestamp = datetime.fromisoformat(oldest_timestamp_str) if oldest_timestamp_str else None
                
                # Database file size
                db_size_mb = Path(self.db_path).stat().st_size / (1024 * 1024) if Path(self.db_path).exists() else 0
                
                return DatabaseStats(
                    total_records=total_records,
                    satellites_count=satellites_count,
                    anomalies_count=anomalies_count,
                    latest_timestamp=latest_timestamp,
                    oldest_timestamp=oldest_timestamp,
                    database_size_mb=db_size_mb
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return DatabaseStats(0, 0, 0, None, None, 0.0)
    
    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old telemetry data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Delete old anomalies first (foreign key constraint)
                cursor.execute("DELETE FROM anomalies WHERE timestamp < ?", (cutoff_date,))
                anomalies_deleted = cursor.rowcount
                
                # Delete old telemetry data
                cursor.execute("DELETE FROM telemetry_data WHERE timestamp < ?", (cutoff_date,))
                telemetry_deleted = cursor.rowcount
                
                conn.commit()
                
                self.logger.info(f"Cleaned database: {telemetry_deleted} telemetry records, {anomalies_deleted} anomaly records deleted")
                
        except Exception as e:
            self.logger.error(f"Failed to clear old data: {e}")
    
    def close(self):
        """Close database connections"""
        self.logger.info("Database connections closed")


class BackgroundStorageService:
    """Background service to store telemetry data"""
    
    def __init__(self, database: TelemetryDatabase):
        self.database = database
        self.logger = logging.getLogger(__name__)
        self.storage_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.stats = {
            'stored_count': 0,
            'anomalies_stored': 0,
            'errors': 0,
            'last_storage': None
        }
    
    def start(self):
        """Start the background storage service"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._storage_loop, daemon=True)
            self.thread.start()
            self.logger.info("Background storage service started")
    
    def stop(self):
        """Stop the background storage service"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.logger.info("Background storage service stopped")
    
    def queue_telemetry(self, data: TelemetryData, anomaly_flag: bool = False, 
                       anomaly_type: str = None):
        """Queue telemetry data for storage"""
        try:
            self.storage_queue.put({
                'data': data,
                'anomaly_flag': anomaly_flag,
                'anomaly_type': anomaly_type,
                'queued_at': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Failed to queue telemetry data: {e}")
            self.stats['errors'] += 1
    
    def _storage_loop(self):
        """Main storage loop running in background thread"""
        self.logger.info("Background storage loop started")
        
        while self.is_running:
            try:
                # Get data from queue with timeout
                try:
                    item = self.storage_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Store data in database
                self.database.store_telemetry(
                    item['data'],
                    item['anomaly_flag'],
                    item['anomaly_type']
                )
                
                # Update stats
                self.stats['stored_count'] += 1
                if item['anomaly_flag']:
                    self.stats['anomalies_stored'] += 1
                self.stats['last_storage'] = datetime.now()
                
                # Mark task as done
                self.storage_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in storage loop: {e}")
                self.stats['errors'] += 1
                time.sleep(0.1)  # Brief pause on error
        
        self.logger.info("Background storage loop ended")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage service statistics"""
        queue_size = self.storage_queue.qsize()
        return {
            **self.stats,
            'queue_size': queue_size,
            'is_running': self.is_running
        }


# Global database instance
_db_instance = None
_storage_service = None

def get_database() -> TelemetryDatabase:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = TelemetryDatabase()
    return _db_instance

def get_storage_service() -> BackgroundStorageService:
    """Get global storage service instance"""
    global _storage_service
    if _storage_service is None:
        _storage_service = BackgroundStorageService(get_database())
    return _storage_service


if __name__ == "__main__":
    # Test the database system
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    from ingestion import TelemetrySimulator
    
    db = TelemetryDatabase()
    storage_service = BackgroundStorageService(db)
    
    print("Testing database and storage service...")
    
    # Start storage service
    storage_service.start()
    
    # Generate and store test data
    simulator = TelemetrySimulator()
    simulator.start()
    
    time.sleep(5)  # Generate some data
    
    # Queue data for storage
    test_data = simulator.get_recent_data(10)
    for data in test_data:
        anomaly_flag = simulator._is_anomaly(data)
        anomaly_type = "temperature_anomaly" if anomaly_flag else None
        storage_service.queue_telemetry(data, anomaly_flag, anomaly_type)
    
    time.sleep(2)  # Let storage complete
    
    # Get stats
    db_stats = db.get_database_stats()
    storage_stats = storage_service.get_stats()
    
    print(f"Database stats: {db_stats}")
    print(f"Storage stats: {storage_stats}")
    
    # Clean up
    simulator.stop()
    storage_service.stop()
    
    print("Database test completed!")