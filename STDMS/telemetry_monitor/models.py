#!/usr/bin/env python3
"""
Data models for telemetry monitoring system
Shared data structures used across all modules
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class AnomalyScore:
    """ML-based anomaly score from multiple models"""
    combined_score: float
    isolation_forest_score: float
    autoencoder_score: float
    lstm_score: float
    confidence: float
    anomaly_type: str
    
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