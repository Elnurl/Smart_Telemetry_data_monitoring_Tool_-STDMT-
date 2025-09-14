#!/usr/bin/env python3
"""
Cloud Database Integration System for Stage 6.4
Adds PostgreSQL and InfluxDB support with connection management, data migration, and hybrid storage options.
"""

import logging
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import queue
import hashlib
from contextlib import asynccontextmanager

# PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from psycopg2.pool import ThreadedConnectionPool
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    logging.warning("psycopg2 not available - PostgreSQL support disabled")

# InfluxDB support
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False
    logging.warning("influxdb-client not available - InfluxDB support disabled")

# Async PostgreSQL support
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logging.warning("asyncpg not available - async PostgreSQL support disabled")

from config_manager import get_config_manager
from storage import get_database

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"

class ConnectionStatus(Enum):
    """Database connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"

@dataclass
class DatabaseConfiguration:
    """Database connection configuration"""
    db_id: str
    db_name: str
    db_type: DatabaseType
    connection_params: Dict[str, Any]
    enabled: bool = True
    priority: int = 1  # 1=primary, 2=secondary, 3=backup
    sync_interval: int = 300  # seconds
    retention_days: int = 90
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatabaseInfo:
    """Runtime database information"""
    config: DatabaseConfiguration
    status: ConnectionStatus
    last_connected: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    error_count: int = 0
    total_records: int = 0
    connection_pool_size: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class PostgreSQLManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, config: DatabaseConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{config.db_id}]")
        self.connection_pool: Optional[ThreadedConnectionPool] = None
        self.connected = False
        
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL support not available - install psycopg2")
    
    def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            params = self.config.connection_params
            
            # Create connection pool
            self.connection_pool = ThreadedConnectionPool(
                minconn=params.get('min_connections', 2),
                maxconn=params.get('max_connections', 10),
                host=params['host'],
                port=params.get('port', 5432),
                database=params['database'],
                user=params['user'],
                password=params['password'],
                sslmode=params.get('sslmode', 'prefer')
            )
            
            # Test connection
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            self.connection_pool.putconn(conn)
            
            self.connected = True
            self.logger.info(f"Connected to PostgreSQL: {version}")
            
            # Initialize database schema
            self._initialize_schema()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from PostgreSQL database"""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
                self.connection_pool = None
            
            self.connected = False
            self.logger.info("Disconnected from PostgreSQL")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from PostgreSQL: {e}")
    
    def _initialize_schema(self):
        """Initialize PostgreSQL database schema"""
        try:
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Create telemetry data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_data (
                    id BIGSERIAL PRIMARY KEY,
                    device_id VARCHAR(100) NOT NULL,
                    message_id VARCHAR(32) UNIQUE,
                    timestamp TIMESTAMPTZ NOT NULL,
                    solar_power REAL,
                    attitude_x REAL,
                    attitude_y REAL,
                    attitude_z REAL,
                    orbit_altitude REAL,
                    temperature REAL,
                    quality_score REAL,
                    anomaly_score REAL DEFAULT 0.0,
                    processing_flags JSONB,
                    raw_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_data(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_device ON telemetry_data(device_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_anomaly ON telemetry_data(anomaly_score) WHERE anomaly_score > 0.5;")
            
            # Create model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    model_type VARCHAR(50) NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    training_samples INTEGER,
                    anomaly_rate REAL,
                    model_version VARCHAR(20),
                    performance_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create system logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    level VARCHAR(20) NOT NULL,
                    component VARCHAR(100),
                    message TEXT,
                    context JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            self.logger.info("PostgreSQL schema initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing PostgreSQL schema: {e}")
            raise
    
    def store_telemetry_data(self, device_id: str, timestamp: datetime, data: Dict[str, Any], 
                           quality_score: float = 1.0, anomaly_score: float = 0.0,
                           processing_flags: List[str] = None, raw_data: Dict[str, Any] = None,
                           message_id: str = None) -> bool:
        """Store telemetry data in PostgreSQL"""
        try:
            if not self.connected:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO telemetry_data 
                (device_id, message_id, timestamp, solar_power, attitude_x, attitude_y, 
                 attitude_z, orbit_altitude, temperature, quality_score, anomaly_score, 
                 processing_flags, raw_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (message_id) DO UPDATE SET
                    solar_power = EXCLUDED.solar_power,
                    attitude_x = EXCLUDED.attitude_x,
                    attitude_y = EXCLUDED.attitude_y,
                    attitude_z = EXCLUDED.attitude_z,
                    orbit_altitude = EXCLUDED.orbit_altitude,
                    temperature = EXCLUDED.temperature,
                    quality_score = EXCLUDED.quality_score,
                    anomaly_score = EXCLUDED.anomaly_score,
                    processing_flags = EXCLUDED.processing_flags,
                    raw_data = EXCLUDED.raw_data
            """, (
                device_id,
                message_id,
                timestamp,
                data.get('solar_power'),
                data.get('attitude_x'),
                data.get('attitude_y'),
                data.get('attitude_z'),
                data.get('orbit_altitude'),
                data.get('temperature'),
                quality_score,
                anomaly_score,
                Json(processing_flags or []),
                Json(raw_data or {})
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing telemetry data in PostgreSQL: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                self.connection_pool.putconn(conn)
            return False
    
    def query_telemetry_data(self, start_time: datetime, end_time: datetime, 
                           device_id: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Query telemetry data from PostgreSQL"""
        try:
            if not self.connected:
                return []
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT * FROM telemetry_data 
                WHERE timestamp BETWEEN %s AND %s
            """
            params = [start_time, end_time]
            
            if device_id:
                query += " AND device_id = %s"
                params.append(device_id)
            
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return [dict(row) for row in results]
            
        except Exception as e:
            self.logger.error(f"Error querying telemetry data from PostgreSQL: {e}")
            if conn:
                cursor.close()
                self.connection_pool.putconn(conn)
            return []
    
    def store_model_performance(self, timestamp: datetime, model_type: str, 
                              metrics: Dict[str, float], model_version: str = None) -> bool:
        """Store model performance data in PostgreSQL"""
        try:
            if not self.connected:
                return False
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_performance 
                (timestamp, model_type, accuracy, precision_score, recall_score, 
                 f1_score, training_samples, anomaly_rate, model_version, performance_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                timestamp,
                model_type,
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1_score'),
                metrics.get('training_samples'),
                metrics.get('anomaly_rate'),
                model_version,
                Json(metrics)
            ))
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing model performance in PostgreSQL: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                self.connection_pool.putconn(conn)
            return False
    
    def cleanup_old_data(self, retention_days: int) -> int:
        """Clean up old data from PostgreSQL"""
        try:
            if not self.connected:
                return 0
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor()
            
            # Delete old telemetry data
            cursor.execute("DELETE FROM telemetry_data WHERE timestamp < %s", (cutoff_date,))
            telemetry_deleted = cursor.rowcount
            
            # Delete old model performance data
            cursor.execute("DELETE FROM model_performance WHERE timestamp < %s", (cutoff_date,))
            performance_deleted = cursor.rowcount
            
            # Delete old system logs
            cursor.execute("DELETE FROM system_logs WHERE timestamp < %s", (cutoff_date,))
            logs_deleted = cursor.rowcount
            
            conn.commit()
            cursor.close()
            self.connection_pool.putconn(conn)
            
            total_deleted = telemetry_deleted + performance_deleted + logs_deleted
            self.logger.info(f"Cleaned up {total_deleted} old records from PostgreSQL")
            
            return total_deleted
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data from PostgreSQL: {e}")
            if conn:
                conn.rollback()
                cursor.close()
                self.connection_pool.putconn(conn)
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            if not self.connected:
                return {}
            
            conn = self.connection_pool.getconn()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            stats = {}
            
            # Table sizes
            cursor.execute("SELECT COUNT(*) as count FROM telemetry_data")
            stats['telemetry_records'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM model_performance")
            stats['model_performance_records'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM system_logs")
            stats['system_log_records'] = cursor.fetchone()['count']
            
            # Database size
            cursor.execute("""
                SELECT pg_size_pretty(pg_database_size(%s)) as size
            """, (self.config.connection_params['database'],))
            stats['database_size'] = cursor.fetchone()['size']
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) as count FROM telemetry_data 
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            stats['recent_records'] = cursor.fetchone()['count']
            
            cursor.close()
            self.connection_pool.putconn(conn)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting PostgreSQL statistics: {e}")
            if conn:
                cursor.close()
                self.connection_pool.putconn(conn)
            return {}

class InfluxDBManager:
    """Manages InfluxDB connections and operations for time-series data"""
    
    def __init__(self, config: DatabaseConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{config.db_id}]")
        self.client: Optional[InfluxDBClient] = None
        self.write_api = None
        self.query_api = None
        self.connected = False
        
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB support not available - install influxdb-client")
    
    def connect(self) -> bool:
        """Connect to InfluxDB"""
        try:
            params = self.config.connection_params
            
            self.client = InfluxDBClient(
                url=params['url'],
                token=params['token'],
                org=params['org'],
                timeout=params.get('timeout', 30000)
            )
            
            # Test connection
            health = self.client.health()
            if health.status != "pass":
                raise Exception(f"InfluxDB health check failed: {health.message}")
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            self.connected = True
            self.logger.info(f"Connected to InfluxDB: {params['url']}")
            
            # Initialize database/bucket
            self._initialize_bucket()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to InfluxDB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from InfluxDB"""
        try:
            if self.client:
                self.client.close()
                self.client = None
                self.write_api = None
                self.query_api = None
            
            self.connected = False
            self.logger.info("Disconnected from InfluxDB")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from InfluxDB: {e}")
    
    def _initialize_bucket(self):
        """Initialize InfluxDB bucket"""
        try:
            params = self.config.connection_params
            bucket_name = params.get('bucket', 'telemetry')
            
            # Check if bucket exists
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets()
            
            bucket_exists = any(bucket.name == bucket_name for bucket in buckets.buckets)
            
            if not bucket_exists:
                # Create bucket with retention policy
                retention_rules = []
                if self.config.retention_days > 0:
                    from influxdb_client.domain.retention_rule import RetentionRule
                    retention_rules.append(RetentionRule(
                        type="expire",
                        every_seconds=self.config.retention_days * 24 * 3600
                    ))
                
                buckets_api.create_bucket(
                    bucket_name=bucket_name,
                    org=params['org'],
                    retention_rules=retention_rules
                )
                
                self.logger.info(f"Created InfluxDB bucket: {bucket_name}")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize InfluxDB bucket: {e}")
    
    def store_telemetry_data(self, device_id: str, timestamp: datetime, data: Dict[str, Any], 
                           quality_score: float = 1.0, anomaly_score: float = 0.0,
                           processing_flags: List[str] = None, raw_data: Dict[str, Any] = None,
                           message_id: str = None) -> bool:
        """Store telemetry data in InfluxDB"""
        try:
            if not self.connected:
                return False
            
            params = self.config.connection_params
            bucket = params.get('bucket', 'telemetry')
            
            # Create point
            point = Point("telemetry") \
                .tag("device_id", device_id) \
                .tag("message_id", message_id or "") \
                .time(timestamp, WritePrecision.S)
            
            # Add fields
            if data.get('solar_power') is not None:
                point = point.field("solar_power", float(data['solar_power']))
            if data.get('attitude_x') is not None:
                point = point.field("attitude_x", float(data['attitude_x']))
            if data.get('attitude_y') is not None:
                point = point.field("attitude_y", float(data['attitude_y']))
            if data.get('attitude_z') is not None:
                point = point.field("attitude_z", float(data['attitude_z']))
            if data.get('orbit_altitude') is not None:
                point = point.field("orbit_altitude", float(data['orbit_altitude']))
            if data.get('temperature') is not None:
                point = point.field("temperature", float(data['temperature']))
            
            point = point.field("quality_score", quality_score)
            point = point.field("anomaly_score", anomaly_score)
            
            # Add processing flags as tags
            if processing_flags:
                for flag in processing_flags[:5]:  # Limit tags
                    point = point.tag(f"flag_{hash(flag) % 1000}", "true")
            
            # Write point
            self.write_api.write(bucket=bucket, record=point)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing telemetry data in InfluxDB: {e}")
            return False
    
    def query_telemetry_data(self, start_time: datetime, end_time: datetime, 
                           device_id: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Query telemetry data from InfluxDB"""
        try:
            if not self.connected:
                return []
            
            params = self.config.connection_params
            bucket = params.get('bucket', 'telemetry')
            
            # Build Flux query
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r._measurement == "telemetry")
            '''
            
            if device_id:
                query += f'|> filter(fn: (r) => r.device_id == "{device_id}")'
            
            query += f'''
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> limit(n: {limit})
                |> sort(columns: ["_time"], desc: true)
            '''
            
            # Execute query
            tables = self.query_api.query(query)
            
            results = []
            for table in tables:
                for record in table.records:
                    result = {
                        'timestamp': record.get_time(),
                        'device_id': record.values.get('device_id'),
                        'solar_power': record.values.get('solar_power'),
                        'attitude_x': record.values.get('attitude_x'),
                        'attitude_y': record.values.get('attitude_y'),
                        'attitude_z': record.values.get('attitude_z'),
                        'orbit_altitude': record.values.get('orbit_altitude'),
                        'temperature': record.values.get('temperature'),
                        'quality_score': record.values.get('quality_score'),
                        'anomaly_score': record.values.get('anomaly_score')
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying telemetry data from InfluxDB: {e}")
            return []
    
    def store_model_performance(self, timestamp: datetime, model_type: str, 
                              metrics: Dict[str, float], model_version: str = None) -> bool:
        """Store model performance data in InfluxDB"""
        try:
            if not self.connected:
                return False
            
            params = self.config.connection_params
            bucket = params.get('bucket', 'telemetry')
            
            # Create point
            point = Point("model_performance") \
                .tag("model_type", model_type) \
                .tag("model_version", model_version or "unknown") \
                .time(timestamp, WritePrecision.S)
            
            # Add metric fields
            for metric_name, value in metrics.items():
                if value is not None:
                    point = point.field(metric_name, float(value))
            
            # Write point
            self.write_api.write(bucket=bucket, record=point)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing model performance in InfluxDB: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            if not self.connected:
                return {}
            
            params = self.config.connection_params
            bucket = params.get('bucket', 'telemetry')
            
            stats = {}
            
            # Count telemetry records
            query = f'''
                from(bucket: "{bucket}")
                |> range(start: 0)
                |> filter(fn: (r) => r._measurement == "telemetry")
                |> count()
            '''
            
            tables = self.query_api.query(query)
            total_count = 0
            for table in tables:
                for record in table.records:
                    total_count += record.get_value()
            
            stats['telemetry_records'] = total_count
            
            # Recent activity (last hour)
            recent_query = f'''
                from(bucket: "{bucket}")
                |> range(start: -1h)
                |> filter(fn: (r) => r._measurement == "telemetry")
                |> count()
            '''
            
            tables = self.query_api.query(recent_query)
            recent_count = 0
            for table in tables:
                for record in table.records:
                    recent_count += record.get_value()
            
            stats['recent_records'] = recent_count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting InfluxDB statistics: {e}")
            return {}

class DataMigrationManager:
    """Handles data migration between different database systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.migration_stats = {
            'total_migrated': 0,
            'failed_migrations': 0,
            'last_migration': None,
            'migration_history': []
        }
    
    def migrate_sqlite_to_postgresql(self, sqlite_db_path: str, postgres_config: DatabaseConfiguration,
                                   batch_size: int = 1000, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Migrate data from SQLite to PostgreSQL"""
        try:
            self.logger.info("Starting migration from SQLite to PostgreSQL")
            
            # Initialize PostgreSQL manager
            postgres_manager = PostgreSQLManager(postgres_config)
            if not postgres_manager.connect():
                return {'success': False, 'error': 'Failed to connect to PostgreSQL'}
            
            # Connect to SQLite
            sqlite_conn = sqlite3.connect(sqlite_db_path)
            sqlite_cursor = sqlite_conn.cursor()
            
            # Get total record count
            sqlite_cursor.execute("SELECT COUNT(*) FROM telemetry_data")
            total_records = sqlite_cursor.fetchone()[0]
            
            migrated_count = 0
            failed_count = 0
            
            # Migrate telemetry data in batches
            sqlite_cursor.execute("""
                SELECT timestamp, solar_power, attitude_x, attitude_y, attitude_z, 
                       orbit_altitude, temperature, anomaly_score 
                FROM telemetry_data 
                ORDER BY timestamp
            """)
            
            while True:
                rows = sqlite_cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                for row in rows:
                    try:
                        timestamp = datetime.fromtimestamp(row[0])
                        data = {
                            'solar_power': row[1],
                            'attitude_x': row[2],
                            'attitude_y': row[3],
                            'attitude_z': row[4],
                            'orbit_altitude': row[5],
                            'temperature': row[6]
                        }
                        
                        success = postgres_manager.store_telemetry_data(
                            device_id="migrated_sqlite",
                            timestamp=timestamp,
                            data=data,
                            anomaly_score=row[7] or 0.0,
                            message_id=hashlib.md5(f"{timestamp}_{row[1]}".encode()).hexdigest()[:16]
                        )
                        
                        if success:
                            migrated_count += 1
                        else:
                            failed_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate record: {e}")
                        failed_count += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(migrated_count, total_records, failed_count)
                
                self.logger.info(f"Migrated {migrated_count}/{total_records} records")
            
            sqlite_conn.close()
            postgres_manager.disconnect()
            
            # Update migration stats
            self.migration_stats.update({
                'total_migrated': self.migration_stats['total_migrated'] + migrated_count,
                'failed_migrations': self.migration_stats['failed_migrations'] + failed_count,
                'last_migration': datetime.now()
            })
            
            migration_result = {
                'success': True,
                'migrated_records': migrated_count,
                'failed_records': failed_count,
                'total_records': total_records,
                'success_rate': (migrated_count / total_records) * 100 if total_records > 0 else 0
            }
            
            self.migration_stats['migration_history'].append(migration_result)
            
            self.logger.info(f"Migration completed: {migrated_count} successful, {failed_count} failed")
            return migration_result
            
        except Exception as e:
            self.logger.error(f"Error during SQLite to PostgreSQL migration: {e}")
            return {'success': False, 'error': str(e)}
    
    def migrate_sqlite_to_influxdb(self, sqlite_db_path: str, influxdb_config: DatabaseConfiguration,
                                 batch_size: int = 1000, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Migrate data from SQLite to InfluxDB"""
        try:
            self.logger.info("Starting migration from SQLite to InfluxDB")
            
            # Initialize InfluxDB manager
            influxdb_manager = InfluxDBManager(influxdb_config)
            if not influxdb_manager.connect():
                return {'success': False, 'error': 'Failed to connect to InfluxDB'}
            
            # Connect to SQLite
            sqlite_conn = sqlite3.connect(sqlite_db_path)
            sqlite_cursor = sqlite_conn.cursor()
            
            # Get total record count
            sqlite_cursor.execute("SELECT COUNT(*) FROM telemetry_data")
            total_records = sqlite_cursor.fetchone()[0]
            
            migrated_count = 0
            failed_count = 0
            
            # Migrate telemetry data in batches
            sqlite_cursor.execute("""
                SELECT timestamp, solar_power, attitude_x, attitude_y, attitude_z, 
                       orbit_altitude, temperature, anomaly_score 
                FROM telemetry_data 
                ORDER BY timestamp
            """)
            
            while True:
                rows = sqlite_cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                for row in rows:
                    try:
                        timestamp = datetime.fromtimestamp(row[0])
                        data = {
                            'solar_power': row[1],
                            'attitude_x': row[2],
                            'attitude_y': row[3],
                            'attitude_z': row[4],
                            'orbit_altitude': row[5],
                            'temperature': row[6]
                        }
                        
                        success = influxdb_manager.store_telemetry_data(
                            device_id="migrated_sqlite",
                            timestamp=timestamp,
                            data=data,
                            anomaly_score=row[7] or 0.0,
                            message_id=hashlib.md5(f"{timestamp}_{row[1]}".encode()).hexdigest()[:16]
                        )
                        
                        if success:
                            migrated_count += 1
                        else:
                            failed_count += 1
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate record: {e}")
                        failed_count += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(migrated_count, total_records, failed_count)
            
            sqlite_conn.close()
            influxdb_manager.disconnect()
            
            # Update migration stats
            self.migration_stats.update({
                'total_migrated': self.migration_stats['total_migrated'] + migrated_count,
                'failed_migrations': self.migration_stats['failed_migrations'] + failed_count,
                'last_migration': datetime.now()
            })
            
            migration_result = {
                'success': True,
                'migrated_records': migrated_count,
                'failed_records': failed_count,
                'total_records': total_records,
                'success_rate': (migrated_count / total_records) * 100 if total_records > 0 else 0
            }
            
            self.migration_stats['migration_history'].append(migration_result)
            
            self.logger.info(f"Migration completed: {migrated_count} successful, {failed_count} failed")
            return migration_result
            
        except Exception as e:
            self.logger.error(f"Error during SQLite to InfluxDB migration: {e}")
            return {'success': False, 'error': str(e)}
    
    def sync_databases(self, source_config: DatabaseConfiguration, target_config: DatabaseConfiguration,
                      sync_period_hours: int = 1) -> Dict[str, Any]:
        """Synchronize data between two databases"""
        try:
            self.logger.info(f"Starting database sync from {source_config.db_type} to {target_config.db_type}")
            
            # Initialize source and target managers
            source_manager = self._get_database_manager(source_config)
            target_manager = self._get_database_manager(target_config)
            
            if not source_manager.connect() or not target_manager.connect():
                return {'success': False, 'error': 'Failed to connect to databases'}
            
            # Get sync period
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=sync_period_hours)
            
            # Query source data
            source_data = source_manager.query_telemetry_data(start_time, end_time, limit=10000)
            
            synced_count = 0
            failed_count = 0
            
            # Sync data to target
            for record in source_data:
                try:
                    timestamp = record.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    
                    data = {
                        'solar_power': record.get('solar_power'),
                        'attitude_x': record.get('attitude_x'),
                        'attitude_y': record.get('attitude_y'),
                        'attitude_z': record.get('attitude_z'),
                        'orbit_altitude': record.get('orbit_altitude'),
                        'temperature': record.get('temperature')
                    }
                    
                    success = target_manager.store_telemetry_data(
                        device_id=record.get('device_id', 'synced'),
                        timestamp=timestamp,
                        data=data,
                        quality_score=record.get('quality_score', 1.0),
                        anomaly_score=record.get('anomaly_score', 0.0),
                        message_id=record.get('message_id')
                    )
                    
                    if success:
                        synced_count += 1
                    else:
                        failed_count += 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to sync record: {e}")
                    failed_count += 1
            
            source_manager.disconnect()
            target_manager.disconnect()
            
            sync_result = {
                'success': True,
                'synced_records': synced_count,
                'failed_records': failed_count,
                'total_records': len(source_data),
                'sync_period_hours': sync_period_hours
            }
            
            self.logger.info(f"Database sync completed: {synced_count} successful, {failed_count} failed")
            return sync_result
            
        except Exception as e:
            self.logger.error(f"Error during database sync: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_database_manager(self, config: DatabaseConfiguration):
        """Get appropriate database manager for configuration"""
        if config.db_type == DatabaseType.POSTGRESQL:
            return PostgreSQLManager(config)
        elif config.db_type == DatabaseType.INFLUXDB:
            return InfluxDBManager(config)
        else:
            raise ValueError(f"Unsupported database type: {config.db_type}")

class CloudDatabaseManager:
    """Main manager for cloud database integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_manager = get_config_manager()
        self.local_database = get_database()
        
        # Database configurations
        self.database_configs: Dict[str, DatabaseConfiguration] = {}
        self.database_info: Dict[str, DatabaseInfo] = {}
        self.database_managers: Dict[str, Union[PostgreSQLManager, InfluxDBManager]] = {}
        
        # Migration manager
        self.migration_manager = DataMigrationManager()
        
        # Sync settings
        self.auto_sync_enabled = False
        self.sync_interval = 300  # seconds
        self.sync_thread: Optional[threading.Thread] = None
        self.sync_queue = queue.Queue()
        
        # Load configurations
        self._load_database_configs()
        
        self.logger.info("Cloud database manager initialized")
    
    def _load_database_configs(self):
        """Load database configurations from file"""
        try:
            config_path = Path("telemetry_monitor/cloud_databases.json")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                for db_data in data.get('databases', []):
                    # Convert db_type string back to enum
                    db_data['db_type'] = DatabaseType(db_data['db_type'])
                    config = DatabaseConfiguration(**db_data)
                    
                    self.database_configs[config.db_id] = config
                    self.database_info[config.db_id] = DatabaseInfo(
                        config=config,
                        status=ConnectionStatus.DISCONNECTED
                    )
                
                self.logger.info(f"Loaded {len(self.database_configs)} database configurations")
            else:
                # Create default configurations
                self._create_default_configs()
                
        except Exception as e:
            self.logger.error(f"Error loading database configurations: {e}")
            self._create_default_configs()
    
    def _create_default_configs(self):
        """Create default database configurations"""
        try:
            # PostgreSQL configuration
            if POSTGRESQL_AVAILABLE:
                postgres_config = DatabaseConfiguration(
                    db_id="postgres_main",
                    db_name="Main PostgreSQL Database",
                    db_type=DatabaseType.POSTGRESQL,
                    connection_params={
                        'host': 'localhost',
                        'port': 5432,
                        'database': 'telemetry',
                        'user': 'postgres',
                        'password': 'password',
                        'sslmode': 'prefer',
                        'min_connections': 2,
                        'max_connections': 10
                    },
                    enabled=False,  # Disabled by default
                    priority=1,
                    sync_interval=300,
                    retention_days=365
                )
                
                self.database_configs["postgres_main"] = postgres_config
                self.database_info["postgres_main"] = DatabaseInfo(
                    config=postgres_config,
                    status=ConnectionStatus.DISCONNECTED
                )
            
            # InfluxDB configuration
            if INFLUXDB_AVAILABLE:
                influxdb_config = DatabaseConfiguration(
                    db_id="influxdb_main",
                    db_name="Main InfluxDB Database",
                    db_type=DatabaseType.INFLUXDB,
                    connection_params={
                        'url': 'http://localhost:8086',
                        'token': 'your-token-here',
                        'org': 'your-org',
                        'bucket': 'telemetry',
                        'timeout': 30000
                    },
                    enabled=False,  # Disabled by default
                    priority=2,
                    sync_interval=300,
                    retention_days=90
                )
                
                self.database_configs["influxdb_main"] = influxdb_config
                self.database_info["influxdb_main"] = DatabaseInfo(
                    config=influxdb_config,
                    status=ConnectionStatus.DISCONNECTED
                )
            
            self._save_database_configs()
            
        except Exception as e:
            self.logger.error(f"Error creating default database configurations: {e}")
    
    def _save_database_configs(self):
        """Save database configurations to file"""
        try:
            config_path = Path("telemetry_monitor/cloud_databases.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'databases': [],
                'last_updated': datetime.now().isoformat()
            }
            
            for config in self.database_configs.values():
                config_dict = asdict(config)
                # Convert enum to string for JSON serialization
                config_dict['db_type'] = config.db_type.value
                data['databases'].append(config_dict)
            
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving database configurations: {e}")
    
    def add_database_config(self, config: DatabaseConfiguration) -> bool:
        """Add a new database configuration"""
        try:
            if config.db_id in self.database_configs:
                self.logger.warning(f"Database configuration {config.db_id} already exists")
                return False
            
            self.database_configs[config.db_id] = config
            self.database_info[config.db_id] = DatabaseInfo(
                config=config,
                status=ConnectionStatus.DISCONNECTED
            )
            
            self._save_database_configs()
            self.logger.info(f"Added database configuration: {config.db_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding database configuration: {e}")
            return False
    
    def connect_database(self, db_id: str) -> bool:
        """Connect to a specific database"""
        try:
            config = self.database_configs.get(db_id)
            if not config:
                self.logger.error(f"Database configuration {db_id} not found")
                return False
            
            # Update status to connecting
            self.database_info[db_id].status = ConnectionStatus.CONNECTING
            
            # Create and connect manager
            if config.db_type == DatabaseType.POSTGRESQL:
                if not POSTGRESQL_AVAILABLE:
                    self.logger.error("PostgreSQL support not available")
                    self.database_info[db_id].status = ConnectionStatus.ERROR
                    return False
                
                manager = PostgreSQLManager(config)
                
            elif config.db_type == DatabaseType.INFLUXDB:
                if not INFLUXDB_AVAILABLE:
                    self.logger.error("InfluxDB support not available")
                    self.database_info[db_id].status = ConnectionStatus.ERROR
                    return False
                
                manager = InfluxDBManager(config)
                
            else:
                self.logger.error(f"Unsupported database type: {config.db_type}")
                self.database_info[db_id].status = ConnectionStatus.ERROR
                return False
            
            # Attempt connection
            if manager.connect():
                self.database_managers[db_id] = manager
                self.database_info[db_id].status = ConnectionStatus.CONNECTED
                self.database_info[db_id].last_connected = datetime.now()
                self.database_info[db_id].error_count = 0
                
                self.logger.info(f"Connected to database: {config.db_name}")
                return True
            else:
                self.database_info[db_id].status = ConnectionStatus.ERROR
                self.database_info[db_id].error_count += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error connecting to database {db_id}: {e}")
            if db_id in self.database_info:
                self.database_info[db_id].status = ConnectionStatus.ERROR
                self.database_info[db_id].error_count += 1
            return False
    
    def disconnect_database(self, db_id: str) -> bool:
        """Disconnect from a specific database"""
        try:
            if db_id in self.database_managers:
                self.database_managers[db_id].disconnect()
                del self.database_managers[db_id]
            
            if db_id in self.database_info:
                self.database_info[db_id].status = ConnectionStatus.DISCONNECTED
            
            self.logger.info(f"Disconnected from database: {db_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from database {db_id}: {e}")
            return False
    
    def connect_all_enabled_databases(self) -> Dict[str, bool]:
        """Connect to all enabled databases"""
        results = {}
        
        for db_id, config in self.database_configs.items():
            if config.enabled:
                results[db_id] = self.connect_database(db_id)
        
        connected_count = sum(results.values())
        self.logger.info(f"Connected to {connected_count}/{len(results)} enabled databases")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive cloud database system status"""
        try:
            status = {
                'total_databases': len(self.database_configs),
                'connected_databases': sum(1 for info in self.database_info.values() 
                                         if info.status == ConnectionStatus.CONNECTED),
                'auto_sync_enabled': self.auto_sync_enabled,
                'sync_interval': self.sync_interval,
                'database_status': {},
                'migration_stats': self.migration_manager.migration_stats,
                'last_updated': datetime.now().isoformat()
            }
            
            # Add individual database status
            for db_id, info in self.database_info.items():
                config = info.config
                
                # Get statistics if connected
                stats = {}
                if db_id in self.database_managers:
                    try:
                        stats = self.database_managers[db_id].get_statistics()
                    except Exception as e:
                        self.logger.warning(f"Could not get statistics for {db_id}: {e}")
                
                status['database_status'][db_id] = {
                    'name': config.db_name,
                    'type': config.db_type.value,
                    'enabled': config.enabled,
                    'status': info.status.value,
                    'priority': config.priority,
                    'last_connected': info.last_connected.isoformat() if info.last_connected else None,
                    'last_sync': info.last_sync.isoformat() if info.last_sync else None,
                    'error_count': info.error_count,
                    'total_records': info.total_records,
                    'performance_metrics': info.performance_metrics,
                    'statistics': stats
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def store_telemetry_data_multi(self, device_id: str, timestamp: datetime, data: Dict[str, Any], 
                                 quality_score: float = 1.0, anomaly_score: float = 0.0,
                                 processing_flags: List[str] = None, raw_data: Dict[str, Any] = None,
                                 message_id: str = None) -> Dict[str, bool]:
        """Store telemetry data in all connected databases"""
        results = {}
        
        try:
            # Store in local SQLite first
            try:
                self.local_database.store_telemetry_data(
                    timestamp=timestamp.timestamp(),
                    solar_power=data.get('solar_power', 0),
                    attitude_x=data.get('attitude_x', 0),
                    attitude_y=data.get('attitude_y', 0),
                    attitude_z=data.get('attitude_z', 0),
                    orbit_altitude=data.get('orbit_altitude', 400),
                    temperature=data.get('temperature', 20),
                    anomaly_score=anomaly_score
                )
                results['local_sqlite'] = True
            except Exception as e:
                self.logger.error(f"Error storing in local SQLite: {e}")
                results['local_sqlite'] = False
            
            # Store in connected cloud databases
            for db_id, manager in self.database_managers.items():
                try:
                    success = manager.store_telemetry_data(
                        device_id=device_id,
                        timestamp=timestamp,
                        data=data,
                        quality_score=quality_score,
                        anomaly_score=anomaly_score,
                        processing_flags=processing_flags,
                        raw_data=raw_data,
                        message_id=message_id
                    )
                    results[db_id] = success
                    
                    if success:
                        self.database_info[db_id].total_records += 1
                    else:
                        self.database_info[db_id].error_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error storing in {db_id}: {e}")
                    results[db_id] = False
                    self.database_info[db_id].error_count += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in multi-database storage: {e}")
            return {'error': str(e)}

# Global instance
_cloud_database_manager = None

def get_cloud_database_manager() -> CloudDatabaseManager:
    """Get the global cloud database manager instance"""
    global _cloud_database_manager
    if _cloud_database_manager is None:
        _cloud_database_manager = CloudDatabaseManager()
    return _cloud_database_manager