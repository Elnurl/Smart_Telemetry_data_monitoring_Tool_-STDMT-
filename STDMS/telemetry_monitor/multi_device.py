#!/usr/bin/env python3
"""
Multi-Device Support System for Stage 6.3
Handles multiple telemetry sources with device management, unified dashboard, and device-specific configurations.
"""

import logging
import json
import sqlite3
import threading
import queue
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import hashlib
import time

# PySide6 imports for device management UI
try:
    from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                                  QPushButton, QTableWidget, QTableWidgetItem, 
                                  QComboBox, QLineEdit, QTextEdit, QGroupBox, 
                                  QTabWidget, QProgressBar, QCheckBox, QSpinBox,
                                  QDialog, QDialogButtonBox, QFormLayout, QMessageBox)
    from PySide6.QtCore import QTimer, Signal, QObject, QThread, QMutex
    from PySide6.QtGui import QFont, QColor
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False
    logging.warning("PySide6 not available - GUI components disabled")

# Import our existing modules
from config_manager import get_config_manager
from storage import get_database
from anomaly import get_anomaly_detector

class DeviceStatus(Enum):
    """Device connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

class DeviceType(Enum):
    """Supported device types"""
    SATELLITE = "satellite"
    GROUND_STATION = "ground_station"
    WEATHER_STATION = "weather_station"
    SENSOR_ARRAY = "sensor_array"
    CUSTOM = "custom"

@dataclass
class DeviceConfiguration:
    """Device-specific configuration"""
    device_id: str
    device_name: str
    device_type: DeviceType
    connection_params: Dict[str, Any]
    data_mapping: Dict[str, str]  # Maps device fields to standard fields
    polling_interval: int  # seconds
    enabled: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low
    timeout: int = 30  # seconds
    retry_count: int = 3
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeviceInfo:
    """Runtime device information"""
    device_config: DeviceConfiguration
    status: DeviceStatus
    last_seen: Optional[datetime] = None
    last_data: Optional[datetime] = None
    error_count: int = 0
    total_records: int = 0
    uptime_percentage: float = 0.0
    current_values: Dict[str, Any] = field(default_factory=dict)
    health_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class TelemetryMessage:
    """Standardized telemetry message"""
    device_id: str
    timestamp: datetime
    source_data: Dict[str, Any]
    mapped_data: Dict[str, Any]
    message_id: str = ""
    quality_score: float = 1.0
    processing_flags: List[str] = field(default_factory=list)

class DeviceRegistry:
    """Manages device configurations and metadata"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = config_path or Path("telemetry_monitor/devices.json")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.devices: Dict[str, DeviceConfiguration] = {}
        self.device_info: Dict[str, DeviceInfo] = {}
        self.mutex = threading.Lock()
        
        self._load_devices()
        self.logger.info(f"Device registry initialized with {len(self.devices)} devices")
    
    def _load_devices(self):
        """Load device configurations from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                for device_data in data.get('devices', []):
                    # Convert device_type string back to enum
                    device_data['device_type'] = DeviceType(device_data['device_type'])
                    config = DeviceConfiguration(**device_data)
                    self.devices[config.device_id] = config
                    
                    # Initialize device info
                    self.device_info[config.device_id] = DeviceInfo(
                        device_config=config,
                        status=DeviceStatus.UNKNOWN
                    )
                    
                self.logger.info(f"Loaded {len(self.devices)} device configurations")
            else:
                # Create default configuration
                self._create_default_devices()
                
        except Exception as e:
            self.logger.error(f"Error loading device configurations: {e}")
            self._create_default_devices()
    
    def _create_default_devices(self):
        """Create default device configurations"""
        try:
            # Primary satellite
            primary_satellite = DeviceConfiguration(
                device_id="sat_001",
                device_name="Primary Satellite",
                device_type=DeviceType.SATELLITE,
                connection_params={
                    "host": "localhost",
                    "port": 8080,
                    "protocol": "tcp",
                    "auth_key": "default"
                },
                data_mapping={
                    "solar_panel_voltage": "solar_power",
                    "attitude_roll": "attitude_x", 
                    "attitude_pitch": "attitude_y",
                    "attitude_yaw": "attitude_z",
                    "altitude_km": "orbit_altitude",
                    "internal_temp": "temperature"
                },
                polling_interval=60,
                priority=1
            )
            
            # Ground station
            ground_station = DeviceConfiguration(
                device_id="gs_001",
                device_name="Primary Ground Station",
                device_type=DeviceType.GROUND_STATION,
                connection_params={
                    "host": "127.0.0.1",
                    "port": 8081,
                    "protocol": "udp"
                },
                data_mapping={
                    "signal_strength": "signal_quality",
                    "ambient_temp": "temperature",
                    "humidity": "humidity",
                    "wind_speed": "wind_speed"
                },
                polling_interval=120,
                priority=2
            )
            
            self.devices["sat_001"] = primary_satellite
            self.devices["gs_001"] = ground_station
            
            # Initialize device info
            for device_id, config in self.devices.items():
                self.device_info[device_id] = DeviceInfo(
                    device_config=config,
                    status=DeviceStatus.UNKNOWN
                )
            
            self._save_devices()
            self.logger.info("Created default device configurations")
            
        except Exception as e:
            self.logger.error(f"Error creating default devices: {e}")
    
    def _save_devices(self):
        """Save device configurations to file"""
        try:
            with self.mutex:
                data = {
                    'devices': [],
                    'last_updated': datetime.now().isoformat()
                }
                
                for config in self.devices.values():
                    device_dict = asdict(config)
                    # Convert enum to string for JSON serialization
                    device_dict['device_type'] = config.device_type.value
                    data['devices'].append(device_dict)
                
                with open(self.config_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
        except Exception as e:
            self.logger.error(f"Error saving device configurations: {e}")
    
    def add_device(self, config: DeviceConfiguration) -> bool:
        """Add a new device configuration"""
        try:
            with self.mutex:
                if config.device_id in self.devices:
                    self.logger.warning(f"Device {config.device_id} already exists")
                    return False
                
                self.devices[config.device_id] = config
                self.device_info[config.device_id] = DeviceInfo(
                    device_config=config,
                    status=DeviceStatus.UNKNOWN
                )
                
                self._save_devices()
                self.logger.info(f"Added device: {config.device_name} ({config.device_id})")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding device: {e}")
            return False
    
    def update_device(self, device_id: str, config: DeviceConfiguration) -> bool:
        """Update device configuration"""
        try:
            with self.mutex:
                if device_id not in self.devices:
                    self.logger.warning(f"Device {device_id} not found")
                    return False
                
                # Update configuration
                self.devices[device_id] = config
                
                # Update device info configuration reference
                if device_id in self.device_info:
                    self.device_info[device_id].device_config = config
                
                self._save_devices()
                self.logger.info(f"Updated device: {config.device_name} ({device_id})")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating device: {e}")
            return False
    
    def remove_device(self, device_id: str) -> bool:
        """Remove a device configuration"""
        try:
            with self.mutex:
                if device_id not in self.devices:
                    self.logger.warning(f"Device {device_id} not found")
                    return False
                
                device_name = self.devices[device_id].device_name
                del self.devices[device_id]
                
                if device_id in self.device_info:
                    del self.device_info[device_id]
                
                self._save_devices()
                self.logger.info(f"Removed device: {device_name} ({device_id})")
                return True
                
        except Exception as e:
            self.logger.error(f"Error removing device: {e}")
            return False
    
    def get_device(self, device_id: str) -> Optional[DeviceConfiguration]:
        """Get device configuration"""
        return self.devices.get(device_id)
    
    def get_all_devices(self) -> Dict[str, DeviceConfiguration]:
        """Get all device configurations"""
        with self.mutex:
            return self.devices.copy()
    
    def get_device_info(self, device_id: str) -> Optional[DeviceInfo]:
        """Get runtime device information"""
        return self.device_info.get(device_id)
    
    def update_device_status(self, device_id: str, status: DeviceStatus, 
                           last_seen: Optional[datetime] = None,
                           error_count: Optional[int] = None):
        """Update device runtime status"""
        try:
            if device_id in self.device_info:
                info = self.device_info[device_id]
                info.status = status
                
                if last_seen:
                    info.last_seen = last_seen
                
                if error_count is not None:
                    info.error_count = error_count
                    
        except Exception as e:
            self.logger.error(f"Error updating device status: {e}")

class DataProcessor:
    """Processes and maps telemetry data from multiple devices"""
    
    def __init__(self, device_registry: DeviceRegistry):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device_registry = device_registry
        self.anomaly_detector = get_anomaly_detector()
        
        # Data processing statistics
        self.processing_stats = {
            'total_messages': 0,
            'successful_mappings': 0,
            'failed_mappings': 0,
            'anomalies_detected': 0,
            'devices_processed': set()
        }
    
    def process_telemetry_message(self, device_id: str, raw_data: Dict[str, Any]) -> Optional[TelemetryMessage]:
        """Process raw telemetry data from a device"""
        try:
            # Get device configuration
            device_config = self.device_registry.get_device(device_id)
            if not device_config:
                self.logger.warning(f"Unknown device: {device_id}")
                return None
            
            # Create telemetry message
            message = TelemetryMessage(
                device_id=device_id,
                timestamp=datetime.now(),
                source_data=raw_data.copy(),
                mapped_data={},
                message_id=self._generate_message_id(device_id, raw_data)
            )
            
            # Apply data mapping
            mapped_data = self._apply_data_mapping(raw_data, device_config.data_mapping)
            message.mapped_data = mapped_data
            
            # Calculate quality score
            message.quality_score = self._calculate_quality_score(mapped_data, device_config)
            
            # Add processing flags
            message.processing_flags = self._analyze_data_quality(mapped_data, device_config)
            
            # Update statistics
            self.processing_stats['total_messages'] += 1
            self.processing_stats['devices_processed'].add(device_id)
            
            if mapped_data:
                self.processing_stats['successful_mappings'] += 1
            else:
                self.processing_stats['failed_mappings'] += 1
                message.processing_flags.append('MAPPING_FAILED')
            
            # Update device info
            device_info = self.device_registry.get_device_info(device_id)
            if device_info:
                device_info.last_data = message.timestamp
                device_info.total_records += 1
                device_info.current_values = mapped_data
            
            self.logger.debug(f"Processed message from {device_id}: quality={message.quality_score:.3f}")
            return message
            
        except Exception as e:
            self.logger.error(f"Error processing telemetry message from {device_id}: {e}")
            return None
    
    def _apply_data_mapping(self, raw_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Apply field mapping from device-specific to standard format"""
        mapped_data = {}
        
        try:
            for source_field, target_field in mapping.items():
                if source_field in raw_data:
                    value = raw_data[source_field]
                    
                    # Apply value transformations if needed
                    transformed_value = self._transform_value(source_field, target_field, value)
                    mapped_data[target_field] = transformed_value
                    
            return mapped_data
            
        except Exception as e:
            self.logger.error(f"Error applying data mapping: {e}")
            return {}
    
    def _transform_value(self, source_field: str, target_field: str, value: Any) -> Any:
        """Transform values during mapping (e.g., unit conversions)"""
        try:
            # Temperature conversions
            if target_field == 'temperature':
                if 'fahrenheit' in source_field.lower() or 'f' in source_field.lower():
                    # Convert Fahrenheit to Celsius
                    return (float(value) - 32) * 5/9
                elif 'kelvin' in source_field.lower() or 'k' in source_field.lower():
                    # Convert Kelvin to Celsius
                    return float(value) - 273.15
            
            # Power conversions
            elif target_field == 'solar_power':
                if 'voltage' in source_field.lower():
                    # Assume standard solar panel current for power calculation
                    return float(value) * 3.5  # Example conversion
                elif 'milliwatt' in source_field.lower() or 'mw' in source_field.lower():
                    return float(value) / 1000  # Convert mW to W
            
            # Altitude conversions
            elif target_field == 'orbit_altitude':
                if 'meter' in source_field.lower() or 'm' in source_field.lower():
                    return float(value) / 1000  # Convert m to km
                elif 'mile' in source_field.lower():
                    return float(value) * 1.60934  # Convert miles to km
            
            # Default: return as-is with type conversion
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            
            return value
            
        except Exception as e:
            self.logger.warning(f"Error transforming value {value} from {source_field} to {target_field}: {e}")
            return value
    
    def _calculate_quality_score(self, mapped_data: Dict[str, Any], config: DeviceConfiguration) -> float:
        """Calculate data quality score"""
        try:
            if not mapped_data:
                return 0.0
            
            quality_factors = []
            
            # Completeness: ratio of mapped fields to expected fields
            expected_fields = set(config.data_mapping.values())
            present_fields = set(mapped_data.keys())
            completeness = len(present_fields & expected_fields) / len(expected_fields)
            quality_factors.append(completeness * 0.4)  # 40% weight
            
            # Value validity: check for reasonable ranges
            validity_score = 0.0
            valid_count = 0
            
            for field, value in mapped_data.items():
                if self._is_valid_value(field, value):
                    validity_score += 1
                valid_count += 1
            
            if valid_count > 0:
                validity_score /= valid_count
                quality_factors.append(validity_score * 0.4)  # 40% weight
            
            # Timeliness: assume current timestamp is good
            quality_factors.append(0.2)  # 20% weight
            
            return sum(quality_factors)
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {e}")
            return 0.5  # Default middle score
    
    def _is_valid_value(self, field: str, value: Any) -> bool:
        """Check if a value is within reasonable ranges"""
        try:
            if value is None:
                return False
            
            numeric_value = float(value)
            
            # Field-specific validation ranges
            ranges = {
                'temperature': (-100, 150),  # Celsius
                'solar_power': (0, 1000),    # Watts
                'orbit_altitude': (200, 2000), # km
                'attitude_x': (-180, 180),   # degrees
                'attitude_y': (-180, 180),   # degrees
                'attitude_z': (-180, 180),   # degrees
                'signal_quality': (0, 100),  # percentage
                'humidity': (0, 100),        # percentage
                'wind_speed': (0, 200)       # km/h
            }
            
            if field in ranges:
                min_val, max_val = ranges[field]
                return min_val <= numeric_value <= max_val
            
            # Default: any finite number is valid
            return not (math.isnan(numeric_value) or math.isinf(numeric_value))
            
        except (ValueError, TypeError):
            # Non-numeric values might be valid depending on field
            return isinstance(value, str) and len(value) > 0
    
    def _analyze_data_quality(self, mapped_data: Dict[str, Any], config: DeviceConfiguration) -> List[str]:
        """Analyze data quality and return flags"""
        flags = []
        
        try:
            # Check for missing critical fields
            critical_fields = ['temperature', 'solar_power']  # Define critical fields
            missing_critical = [field for field in critical_fields if field not in mapped_data]
            if missing_critical:
                flags.append(f'MISSING_CRITICAL_FIELDS:{",".join(missing_critical)}')
            
            # Check for out-of-range values
            for field, value in mapped_data.items():
                if not self._is_valid_value(field, value):
                    flags.append(f'OUT_OF_RANGE:{field}={value}')
            
            # Check for stale data (if timestamp available)
            # This would require timestamp in the data
            
            return flags
            
        except Exception as e:
            self.logger.warning(f"Error analyzing data quality: {e}")
            return ['ANALYSIS_ERROR']
    
    def _generate_message_id(self, device_id: str, raw_data: Dict[str, Any]) -> str:
        """Generate unique message ID"""
        try:
            # Create hash from device_id, timestamp, and data
            data_str = json.dumps(raw_data, sort_keys=True, default=str)
            hash_input = f"{device_id}_{datetime.now().isoformat()}_{data_str}"
            return hashlib.md5(hash_input.encode()).hexdigest()[:16]
        except Exception:
            return str(uuid.uuid4())[:8]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get data processing statistics"""
        stats = self.processing_stats.copy()
        stats['devices_processed'] = len(stats['devices_processed'])
        return stats

class MultiDeviceManager:
    """Main manager for multi-device telemetry system"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = get_config_manager()
        self.database = get_database()
        
        # Initialize components
        self.device_registry = DeviceRegistry()
        self.data_processor = DataProcessor(self.device_registry)
        
        # Threading components
        self.message_queue = queue.Queue(maxsize=10000)
        self.worker_threads: List[threading.Thread] = []
        self.running = False
        self.stats_lock = threading.Lock()
        
        # Device monitoring
        self.device_monitors: Dict[str, 'DeviceMonitor'] = {}
        self.monitoring_timers: Dict[str, QTimer] = {}
        
        # System statistics
        self.system_stats = {
            'start_time': datetime.now(),
            'total_devices': 0,
            'active_devices': 0,
            'total_messages': 0,
            'messages_per_minute': 0.0,
            'last_update': datetime.now()
        }
        
        self.logger.info("Multi-device manager initialized")
    
    def start(self) -> bool:
        """Start the multi-device system"""
        try:
            if self.running:
                self.logger.warning("Multi-device system already running")
                return True
            
            self.running = True
            
            # Start worker threads for message processing
            num_workers = min(4, len(self.device_registry.get_all_devices()) + 1)
            for i in range(num_workers):
                worker = threading.Thread(target=self._message_worker, name=f"MessageWorker-{i}")
                worker.daemon = True
                worker.start()
                self.worker_threads.append(worker)
            
            # Start device monitoring
            self._start_device_monitoring()
            
            # Start statistics timer
            if PYSIDE6_AVAILABLE:
                stats_timer = QTimer()
                stats_timer.timeout.connect(self._update_system_stats)
                stats_timer.start(10000)  # Update every 10 seconds
            
            self.logger.info(f"Multi-device system started with {num_workers} workers")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting multi-device system: {e}")
            return False
    
    def stop(self):
        """Stop the multi-device system"""
        try:
            self.logger.info("Stopping multi-device system...")
            self.running = False
            
            # Stop device monitoring
            self._stop_device_monitoring()
            
            # Stop worker threads
            for _ in self.worker_threads:
                self.message_queue.put(None)  # Poison pill
            
            for worker in self.worker_threads:
                worker.join(timeout=5)
            
            self.worker_threads.clear()
            
            self.logger.info("Multi-device system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping multi-device system: {e}")
    
    def _message_worker(self):
        """Worker thread for processing telemetry messages"""
        while self.running:
            try:
                # Get message from queue
                message = self.message_queue.get(timeout=1)
                
                if message is None:  # Poison pill
                    break
                
                device_id, raw_data = message
                
                # Process the message
                telemetry_message = self.data_processor.process_telemetry_message(device_id, raw_data)
                
                if telemetry_message:
                    # Store in database
                    self._store_telemetry_message(telemetry_message)
                    
                    # Run anomaly detection
                    self._check_for_anomalies(telemetry_message)
                    
                    # Update device status
                    self.device_registry.update_device_status(
                        device_id, 
                        DeviceStatus.ONLINE, 
                        telemetry_message.timestamp
                    )
                
                self.message_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in message worker: {e}")
    
    def _store_telemetry_message(self, message: TelemetryMessage):
        """Store telemetry message in database"""
        try:
            # Use existing database structure but add device tracking
            mapped_data = message.mapped_data
            
            # Add device metadata
            extended_data = mapped_data.copy()
            extended_data['device_id'] = message.device_id
            extended_data['quality_score'] = message.quality_score
            extended_data['message_id'] = message.message_id
            
            # Store using existing database interface
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            # Create extended telemetry table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_data_multi (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    message_id TEXT UNIQUE,
                    timestamp REAL NOT NULL,
                    solar_power REAL,
                    attitude_x REAL,
                    attitude_y REAL,
                    attitude_z REAL,
                    orbit_altitude REAL,
                    temperature REAL,
                    quality_score REAL,
                    anomaly_score REAL DEFAULT 0.0,
                    processing_flags TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Insert data
            cursor.execute("""
                INSERT OR REPLACE INTO telemetry_data_multi 
                (device_id, message_id, timestamp, solar_power, attitude_x, attitude_y, 
                 attitude_z, orbit_altitude, temperature, quality_score, processing_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.device_id,
                message.message_id,
                message.timestamp.timestamp(),
                mapped_data.get('solar_power'),
                mapped_data.get('attitude_x'),
                mapped_data.get('attitude_y'),
                mapped_data.get('attitude_z'),  
                mapped_data.get('orbit_altitude'),
                mapped_data.get('temperature'),
                message.quality_score,
                json.dumps(message.processing_flags)
            ))
            
            conn.commit()
            conn.close()
            
            # Also store in original format for compatibility
            if mapped_data:
                original_data = {
                    'timestamp': message.timestamp.timestamp(),
                    'solar_power': mapped_data.get('solar_power', 0),
                    'attitude_x': mapped_data.get('attitude_x', 0),
                    'attitude_y': mapped_data.get('attitude_y', 0),
                    'attitude_z': mapped_data.get('attitude_z', 0),
                    'orbit_altitude': mapped_data.get('orbit_altitude', 400),
                    'temperature': mapped_data.get('temperature', 20)
                }
                self.database.store_telemetry_data(**original_data)
            
        except Exception as e:
            self.logger.error(f"Error storing telemetry message: {e}")
    
    def _check_for_anomalies(self, message: TelemetryMessage):
        """Run anomaly detection on telemetry message"""
        try:
            mapped_data = message.mapped_data
            
            # Prepare data for anomaly detection
            if all(field in mapped_data for field in ['solar_power', 'attitude_x', 'attitude_y', 'attitude_z', 'orbit_altitude', 'temperature']):
                data_point = [
                    mapped_data['solar_power'],
                    mapped_data['attitude_x'],
                    mapped_data['attitude_y'],
                    mapped_data['attitude_z'],
                    mapped_data['orbit_altitude'],
                    mapped_data['temperature']
                ]
                
                # Run anomaly detection
                anomaly_score = self.anomaly_detector.detect_anomaly(data_point)
                
                # Update database with anomaly score
                if anomaly_score > 0.5:  # Anomaly threshold
                    self.data_processor.processing_stats['anomalies_detected'] += 1
                    message.processing_flags.append(f'ANOMALY_DETECTED:{anomaly_score:.3f}')
                    
                    # Update device info
                    device_info = self.device_registry.get_device_info(message.device_id)
                    if device_info:
                        device_info.health_metrics['last_anomaly_score'] = anomaly_score
                        device_info.health_metrics['last_anomaly_time'] = message.timestamp.timestamp()
                
        except Exception as e:
            self.logger.error(f"Error checking for anomalies: {e}")
    
    def add_telemetry_data(self, device_id: str, raw_data: Dict[str, Any]) -> bool:
        """Add telemetry data from a device"""
        try:
            if not self.running:
                self.logger.warning("Multi-device system not running")
                return False
            
            # Add to processing queue
            self.message_queue.put((device_id, raw_data))
            
            with self.stats_lock:
                self.system_stats['total_messages'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding telemetry data: {e}")
            return False
    
    def _start_device_monitoring(self):
        """Start monitoring all enabled devices"""
        try:
            for device_id, config in self.device_registry.get_all_devices().items():
                if config.enabled:
                    monitor = DeviceMonitor(device_id, config, self)
                    self.device_monitors[device_id] = monitor
                    monitor.start()
            
            self.logger.info(f"Started monitoring {len(self.device_monitors)} devices")
            
        except Exception as e:
            self.logger.error(f"Error starting device monitoring: {e}")
    
    def _stop_device_monitoring(self):
        """Stop monitoring all devices"""
        try:
            for monitor in self.device_monitors.values():
                monitor.stop()
            
            self.device_monitors.clear()
            self.logger.info("Stopped all device monitoring")
            
        except Exception as e:
            self.logger.error(f"Error stopping device monitoring: {e}")
    
    def _update_system_stats(self):
        """Update system statistics"""
        try:
            with self.stats_lock:
                # Count active devices
                active_devices = sum(1 for info in self.device_registry.device_info.values() 
                                   if info.status == DeviceStatus.ONLINE)
                
                self.system_stats.update({
                    'total_devices': len(self.device_registry.devices),
                    'active_devices': active_devices,
                    'last_update': datetime.now()
                })
                
                # Calculate messages per minute
                time_diff = (datetime.now() - self.system_stats['start_time']).total_seconds() / 60
                if time_diff > 0:
                    self.system_stats['messages_per_minute'] = self.system_stats['total_messages'] / time_diff
            
        except Exception as e:
            self.logger.error(f"Error updating system stats: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            with self.stats_lock:
                status = {
                    'system_stats': self.system_stats.copy(),
                    'processing_stats': self.data_processor.get_processing_statistics(),
                    'device_status': {},
                    'queue_size': self.message_queue.qsize(),
                    'running': self.running
                }
                
                # Add device status
                for device_id, info in self.device_registry.device_info.items():
                    status['device_status'][device_id] = {
                        'name': info.device_config.device_name,
                        'type': info.device_config.device_type.value,
                        'status': info.status.value,
                        'last_seen': info.last_seen.isoformat() if info.last_seen else None,
                        'last_data': info.last_data.isoformat() if info.last_data else None,
                        'total_records': info.total_records,
                        'error_count': info.error_count,
                        'uptime_percentage': info.uptime_percentage,
                        'health_metrics': info.health_metrics
                    }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def get_device_registry(self) -> DeviceRegistry:
        """Get the device registry"""
        return self.device_registry
    
    def restart_device_monitoring(self, device_id: str) -> bool:
        """Restart monitoring for a specific device"""
        try:
            # Stop existing monitor
            if device_id in self.device_monitors:
                self.device_monitors[device_id].stop()
                del self.device_monitors[device_id]
            
            # Start new monitor
            config = self.device_registry.get_device(device_id)
            if config and config.enabled:
                monitor = DeviceMonitor(device_id, config, self)
                self.device_monitors[device_id] = monitor
                monitor.start()
                
                self.logger.info(f"Restarted monitoring for device: {device_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error restarting device monitoring: {e}")
            return False

class DeviceMonitor:
    """Monitors individual device connections and data flow"""
    
    def __init__(self, device_id: str, config: DeviceConfiguration, manager: MultiDeviceManager):
        self.device_id = device_id
        self.config = config
        self.manager = manager
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{device_id}]")
        
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_heartbeat = datetime.now()
        self.consecutive_errors = 0
        
        # Simulation data for demo purposes
        self.simulation_data = self._initialize_simulation_data()
    
    def _initialize_simulation_data(self) -> Dict[str, Any]:
        """Initialize simulation data based on device type"""
        base_data = {
            'timestamp': datetime.now(),
            'device_status': 'operational'
        }
        
        if self.config.device_type == DeviceType.SATELLITE:
            base_data.update({
                'solar_panel_voltage': 28.5,
                'attitude_roll': 0.0,
                'attitude_pitch': 0.0,
                'attitude_yaw': 0.0,
                'altitude_km': 408.0,
                'internal_temp': 22.0,
                'battery_level': 85.0,
                'signal_strength': 92.0
            })
        elif self.config.device_type == DeviceType.GROUND_STATION:
            base_data.update({
                'signal_strength': 88.0,
                'ambient_temp': 15.0,
                'humidity': 45.0,
                'wind_speed': 12.0,
                'atmospheric_pressure': 1013.25,
                'precipitation': 0.0
            })
        elif self.config.device_type == DeviceType.WEATHER_STATION:
            base_data.update({
                'ambient_temp': 18.0,
                'humidity': 65.0,
                'wind_speed': 8.0,
                'wind_direction': 180.0,
                'atmospheric_pressure': 1015.0,
                'precipitation': 0.0,
                'solar_radiation': 450.0
            })
        
        return base_data
    
    def start(self):
        """Start device monitoring"""
        try:
            if self.running:
                return
            
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, name=f"DeviceMonitor-{self.device_id}")
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.logger.info(f"Started monitoring device: {self.config.device_name}")
            
        except Exception as e:
            self.logger.error(f"Error starting device monitor: {e}")
    
    def stop(self):
        """Stop device monitoring"""
        try:
            self.running = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            self.logger.info(f"Stopped monitoring device: {self.config.device_name}")
            
        except Exception as e:
            self.logger.error(f"Error stopping device monitor: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop for the device"""
        while self.running:
            try:
                # Simulate data collection based on device type
                raw_data = self._collect_device_data()
                
                if raw_data:
                    # Send data to manager
                    success = self.manager.add_telemetry_data(self.device_id, raw_data)
                    
                    if success:
                        self.last_heartbeat = datetime.now()
                        self.consecutive_errors = 0
                        
                        # Update device status
                        self.manager.device_registry.update_device_status(
                            self.device_id, 
                            DeviceStatus.ONLINE,
                            self.last_heartbeat
                        )
                    else:
                        self._handle_error("Failed to process telemetry data")
                else:
                    self._handle_error("No data received from device")
                
                # Wait for next polling interval
                time.sleep(self.config.polling_interval)
                
            except Exception as e:
                self._handle_error(f"Monitor loop error: {e}")
                time.sleep(min(self.config.polling_interval, 10))
    
    def _collect_device_data(self) -> Optional[Dict[str, Any]]:
        """Collect data from the device (simulated for demo)"""
        try:
            # In a real implementation, this would:
            # 1. Connect to the actual device using connection_params
            # 2. Request/receive telemetry data
            # 3. Handle device-specific protocols
            
            # For demonstration, generate realistic simulation data
            current_data = self.simulation_data.copy()
            current_data['timestamp'] = datetime.now()
            
            # Add some realistic variations
            if self.config.device_type == DeviceType.SATELLITE:
                # Simulate orbital variations
                import random
                orbital_phase = (time.time() % 5400) / 5400 * 2 * math.pi  # 90-minute orbit
                
                current_data.update({
                    'solar_panel_voltage': 28.5 + 2.0 * math.sin(orbital_phase) + random.uniform(-0.5, 0.5),
                    'attitude_roll': random.uniform(-2.0, 2.0),
                    'attitude_pitch': random.uniform(-2.0, 2.0), 
                    'attitude_yaw': random.uniform(-2.0, 2.0),
                    'altitude_km': 408.0 + random.uniform(-2.0, 2.0),
                    'internal_temp': 22.0 + 10.0 * math.sin(orbital_phase) + random.uniform(-1.0, 1.0),
                    'battery_level': max(75, 85 + 5 * math.sin(orbital_phase)),
                    'signal_strength': 92.0 + random.uniform(-5.0, 5.0)
                })
                
            elif self.config.device_type == DeviceType.GROUND_STATION:
                # Simulate weather variations
                import random
                time_of_day = (time.time() % 86400) / 86400  # Daily cycle
                
                current_data.update({
                    'signal_strength': 88.0 + random.uniform(-10.0, 10.0),
                    'ambient_temp': 15.0 + 8.0 * math.sin(time_of_day * 2 * math.pi - math.pi/2) + random.uniform(-2.0, 2.0),
                    'humidity': 45.0 + random.uniform(-10.0, 15.0),
                    'wind_speed': max(0, 12.0 + random.uniform(-8.0, 8.0)),
                    'atmospheric_pressure': 1013.25 + random.uniform(-5.0, 5.0)
                })
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error collecting device data: {e}")
            return None
    
    def _handle_error(self, error_message: str):
        """Handle device monitoring errors"""
        try:
            self.consecutive_errors += 1
            self.logger.warning(f"Device error (#{self.consecutive_errors}): {error_message}")
            
            # Update device status based on error count
            if self.consecutive_errors >= self.config.retry_count:
                status = DeviceStatus.ERROR
            else:
                status = DeviceStatus.OFFLINE
            
            self.manager.device_registry.update_device_status(
                self.device_id,
                status,
                error_count=self.consecutive_errors
            )
            
            # If too many consecutive errors, pause monitoring temporarily
            if self.consecutive_errors >= self.config.retry_count * 2:
                self.logger.error(f"Too many errors for device {self.device_id}, pausing monitoring")
                time.sleep(60)  # Pause for 1 minute
                
        except Exception as e:
            self.logger.error(f"Error handling device error: {e}")

# GUI Components for Multi-Device Management
if PYSIDE6_AVAILABLE:
    
    class DeviceManagerWidget(QWidget):
        """GUI widget for managing multiple devices"""
        
        def __init__(self, manager: MultiDeviceManager):
            super().__init__()
            self.manager = manager
            self.setup_ui()
            self.refresh_timer = QTimer()
            self.refresh_timer.timeout.connect(self.refresh_device_list)
            self.refresh_timer.start(5000)  # Refresh every 5 seconds
        
        def setup_ui(self):
            """Setup the user interface"""
            layout = QVBoxLayout()
            
            # Title
            title = QLabel("Multi-Device Management")
            title.setFont(QFont("Arial", 16, QFont.Bold))
            layout.addWidget(title)
            
            # Control buttons
            button_layout = QHBoxLayout()
            
            self.add_device_btn = QPushButton("Add Device")
            self.add_device_btn.clicked.connect(self.add_device)
            button_layout.addWidget(self.add_device_btn)
            
            self.edit_device_btn = QPushButton("Edit Device")
            self.edit_device_btn.clicked.connect(self.edit_device)
            button_layout.addWidget(self.edit_device_btn)
            
            self.remove_device_btn = QPushButton("Remove Device")
            self.remove_device_btn.clicked.connect(self.remove_device)
            button_layout.addWidget(self.remove_device_btn)
            
            self.restart_monitor_btn = QPushButton("Restart Monitor")
            self.restart_monitor_btn.clicked.connect(self.restart_monitor)
            button_layout.addWidget(self.restart_monitor_btn)
            
            button_layout.addStretch()
            layout.addLayout(button_layout)
            
            # Device list table
            self.device_table = QTableWidget()
            self.device_table.setColumnCount(8)
            self.device_table.setHorizontalHeaderLabels([
                "Device ID", "Name", "Type", "Status", "Last Seen", 
                "Records", "Errors", "Health"
            ])
            layout.addWidget(self.device_table)
            
            # System status
            status_group = QGroupBox("System Status")
            status_layout = QVBoxLayout()
            
            self.status_label = QLabel("System Status: Initializing...")
            status_layout.addWidget(self.status_label)
            
            status_group.setLayout(status_layout)
            layout.addWidget(status_group)
            
            self.setLayout(layout)
            self.refresh_device_list()
        
        def refresh_device_list(self):
            """Refresh the device list table"""
            try:
                status = self.manager.get_system_status()
                device_status = status.get('device_status', {})
                
                self.device_table.setRowCount(len(device_status))
                
                for row, (device_id, info) in enumerate(device_status.items()):
                    self.device_table.setItem(row, 0, QTableWidgetItem(device_id))
                    self.device_table.setItem(row, 1, QTableWidgetItem(info['name']))
                    self.device_table.setItem(row, 2, QTableWidgetItem(info['type']))
                    
                    # Status with color coding
                    status_item = QTableWidgetItem(info['status'])
                    if info['status'] == 'online':
                        status_item.setBackground(QColor(144, 238, 144))  # Light green
                    elif info['status'] == 'offline':
                        status_item.setBackground(QColor(255, 182, 193))  # Light pink
                    elif info['status'] == 'error':
                        status_item.setBackground(QColor(255, 99, 71))   # Tomato
                    self.device_table.setItem(row, 3, status_item)
                    
                    # Last seen
                    last_seen = info['last_seen']
                    if last_seen:
                        last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                        last_seen_str = last_seen_dt.strftime('%H:%M:%S')
                    else:
                        last_seen_str = "Never"
                    self.device_table.setItem(row, 4, QTableWidgetItem(last_seen_str))
                    
                    self.device_table.setItem(row, 5, QTableWidgetItem(str(info['total_records'])))
                    self.device_table.setItem(row, 6, QTableWidgetItem(str(info['error_count'])))
                    
                    # Health score
                    health_metrics = info.get('health_metrics', {})
                    last_anomaly_score = health_metrics.get('last_anomaly_score', 0.0)
                    health_score = max(0, 100 - (last_anomaly_score * 100 + info['error_count'] * 10))
                    health_item = QTableWidgetItem(f"{health_score:.1f}%")
                    
                    if health_score >= 80:
                        health_item.setBackground(QColor(144, 238, 144))  # Light green
                    elif health_score >= 60:
                        health_item.setBackground(QColor(255, 255, 224))  # Light yellow
                    else:
                        health_item.setBackground(QColor(255, 182, 193))  # Light pink
                    
                    self.device_table.setItem(row, 7, health_item)
                
                # Update system status
                system_stats = status.get('system_stats', {})
                status_text = f"Running: {status.get('running', False)} | "
                status_text += f"Devices: {system_stats.get('active_devices', 0)}/{system_stats.get('total_devices', 0)} | "
                status_text += f"Messages/min: {system_stats.get('messages_per_minute', 0):.1f} | "
                status_text += f"Queue: {status.get('queue_size', 0)}"
                
                self.status_label.setText(status_text)
                
            except Exception as e:
                self.status_label.setText(f"Error refreshing: {e}")
        
        def add_device(self):
            """Add a new device"""
            dialog = DeviceConfigDialog(self)
            if dialog.exec() == QDialog.Accepted:
                config = dialog.get_configuration()
                if self.manager.device_registry.add_device(config):
                    QMessageBox.information(self, "Success", f"Device {config.device_name} added successfully")
                    if self.manager.running:
                        self.manager.restart_device_monitoring(config.device_id)
                    self.refresh_device_list()
                else:
                    QMessageBox.warning(self, "Error", "Failed to add device")
        
        def edit_device(self):
            """Edit selected device"""
            current_row = self.device_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(self, "Warning", "Please select a device to edit")
                return
            
            device_id = self.device_table.item(current_row, 0).text()
            config = self.manager.device_registry.get_device(device_id)
            
            if config:
                dialog = DeviceConfigDialog(self, config)
                if dialog.exec() == QDialog.Accepted:
                    new_config = dialog.get_configuration()
                    if self.manager.device_registry.update_device(device_id, new_config):
                        QMessageBox.information(self, "Success", f"Device {new_config.device_name} updated successfully")
                        if self.manager.running:
                            self.manager.restart_device_monitoring(device_id)
                        self.refresh_device_list()
                    else:
                        QMessageBox.warning(self, "Error", "Failed to update device")
        
        def remove_device(self):
            """Remove selected device"""
            current_row = self.device_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(self, "Warning", "Please select a device to remove")
                return
            
            device_id = self.device_table.item(current_row, 0).text()
            device_name = self.device_table.item(current_row, 1).text()
            
            reply = QMessageBox.question(self, "Confirm Removal", 
                                       f"Are you sure you want to remove device '{device_name}'?",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                if self.manager.device_registry.remove_device(device_id):
                    QMessageBox.information(self, "Success", f"Device {device_name} removed successfully")
                    self.refresh_device_list()
                else:
                    QMessageBox.warning(self, "Error", "Failed to remove device")
        
        def restart_monitor(self):
            """Restart monitoring for selected device"""
            current_row = self.device_table.currentRow()
            if current_row < 0:
                QMessageBox.warning(self, "Warning", "Please select a device to restart monitoring")
                return
            
            device_id = self.device_table.item(current_row, 0).text()
            
            if self.manager.restart_device_monitoring(device_id):
                QMessageBox.information(self, "Success", f"Monitoring restarted for device {device_id}")
            else:
                QMessageBox.warning(self, "Error", "Failed to restart monitoring")
    
    class DeviceConfigDialog(QDialog):
        """Dialog for configuring devices"""
        
        def __init__(self, parent=None, config: Optional[DeviceConfiguration] = None):
            super().__init__(parent)
            self.config = config  # Existing config for editing
            self.setup_ui()
            
            if config:
                self.load_configuration(config)
        
        def setup_ui(self):
            """Setup the dialog UI"""
            self.setWindowTitle("Device Configuration")
            self.setModal(True)
            self.resize(500, 600)
            
            layout = QVBoxLayout()
            
            # Form layout
            form_layout = QFormLayout()
            
            # Basic information
            self.device_id_edit = QLineEdit()
            self.device_name_edit = QLineEdit()
            self.device_type_combo = QComboBox()
            
            for device_type in DeviceType:
                self.device_type_combo.addItem(device_type.value.replace('_', ' ').title(), device_type)
            
            form_layout.addRow("Device ID:", self.device_id_edit)
            form_layout.addRow("Device Name:", self.device_name_edit)
            form_layout.addRow("Device Type:", self.device_type_combo)
            
            # Connection parameters
            self.host_edit = QLineEdit()
            self.port_edit = QSpinBox()
            self.port_edit.setRange(1, 65535)
            self.port_edit.setValue(8080)
            
            self.protocol_combo = QComboBox()
            self.protocol_combo.addItems(["tcp", "udp", "http", "serial"])
            
            form_layout.addRow("Host:", self.host_edit)
            form_layout.addRow("Port:", self.port_edit)
            form_layout.addRow("Protocol:", self.protocol_combo)
            
            # Settings
            self.polling_interval_spin = QSpinBox()
            self.polling_interval_spin.setRange(10, 3600)
            self.polling_interval_spin.setValue(60)
            self.polling_interval_spin.setSuffix(" seconds")
            
            self.priority_spin = QSpinBox()
            self.priority_spin.setRange(1, 3)
            self.priority_spin.setValue(1)
            
            self.timeout_spin = QSpinBox()
            self.timeout_spin.setRange(5, 300)
            self.timeout_spin.setValue(30)
            self.timeout_spin.setSuffix(" seconds")
            
            self.enabled_check = QCheckBox()
            self.enabled_check.setChecked(True)
            
            form_layout.addRow("Polling Interval:", self.polling_interval_spin)
            form_layout.addRow("Priority (1=high):", self.priority_spin)
            form_layout.addRow("Timeout:", self.timeout_spin)
            form_layout.addRow("Enabled:", self.enabled_check)
            
            layout.addLayout(form_layout)
            
            # Data mapping
            mapping_group = QGroupBox("Data Mapping")
            mapping_layout = QVBoxLayout()
            
            self.mapping_text = QTextEdit()
            self.mapping_text.setPlaceholderText(
                "Enter data mapping in JSON format, e.g.:\n"
                "{\n"
                '  "solar_panel_voltage": "solar_power",\n'
                '  "attitude_roll": "attitude_x",\n'
                '  "internal_temp": "temperature"\n'
                "}"
            )
            mapping_layout.addWidget(self.mapping_text)
            
            mapping_group.setLayout(mapping_layout)
            layout.addWidget(mapping_group)
            
            # Buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(self.accept)
            button_box.rejected.connect(self.reject)
            layout.addWidget(button_box)
            
            self.setLayout(layout)
        
        def load_configuration(self, config: DeviceConfiguration):
            """Load existing configuration into the form"""
            self.device_id_edit.setText(config.device_id)
            self.device_name_edit.setText(config.device_name)
            
            # Set device type
            for i in range(self.device_type_combo.count()):
                if self.device_type_combo.itemData(i) == config.device_type:
                    self.device_type_combo.setCurrentIndex(i)
                    break
            
            # Connection parameters
            conn_params = config.connection_params
            self.host_edit.setText(conn_params.get('host', ''))
            self.port_edit.setValue(conn_params.get('port', 8080))
            
            protocol = conn_params.get('protocol', 'tcp')
            index = self.protocol_combo.findText(protocol)
            if index >= 0:
                self.protocol_combo.setCurrentIndex(index)
            
            # Settings
            self.polling_interval_spin.setValue(config.polling_interval)
            self.priority_spin.setValue(config.priority)
            self.timeout_spin.setValue(config.timeout)
            self.enabled_check.setChecked(config.enabled)
            
            # Data mapping
            mapping_json = json.dumps(config.data_mapping, indent=2)
            self.mapping_text.setPlainText(mapping_json)
        
        def get_configuration(self) -> DeviceConfiguration:
            """Get configuration from the form"""
            # Parse data mapping
            try:
                mapping_text = self.mapping_text.toPlainText().strip()
                if mapping_text:
                    data_mapping = json.loads(mapping_text)
                else:
                    data_mapping = {}
            except json.JSONDecodeError:
                data_mapping = {}
            
            return DeviceConfiguration(
                device_id=self.device_id_edit.text(),
                device_name=self.device_name_edit.text(),
                device_type=self.device_type_combo.currentData(),
                connection_params={
                    'host': self.host_edit.text(),
                    'port': self.port_edit.value(),
                    'protocol': self.protocol_combo.currentText()
                },
                data_mapping=data_mapping,
                polling_interval=self.polling_interval_spin.value(),
                enabled=self.enabled_check.isChecked(),
                priority=self.priority_spin.value(),
                timeout=self.timeout_spin.value()
            )

# Global instance
_multi_device_manager = None

def get_multi_device_manager() -> MultiDeviceManager:
    """Get the global multi-device manager instance"""
    global _multi_device_manager
    if _multi_device_manager is None:
        _multi_device_manager = MultiDeviceManager()
    return _multi_device_manager