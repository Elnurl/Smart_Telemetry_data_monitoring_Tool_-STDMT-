# Stage 3 - Smart Anomaly Detection (ML)

This document describes the implementation of Stage 3 of the Satellite Telemetry Data Management System (STDMS), which adds Machine Learning-based anomaly detection and smart alert systems.

## Overview

Stage 3 implements sophisticated anomaly detection using multiple machine learning algorithms and provides an intelligent alert system with multiple notification channels. The system combines rule-based and ML-based detection to provide comprehensive monitoring of satellite telemetry data.

## Architecture

### Components

1. **Smart Anomaly Detector** (`anomaly.py`)
   - Isolation Forest for outlier detection
   - Autoencoder neural network for pattern anomalies
   - LSTM for time-series anomalies
   - Ensemble scoring system

2. **Smart Alert System** (`alerts.py`)
   - Multiple notification channels (Email, Slack, Popup)
   - Configurable alert rules and thresholds
   - Alert management and history tracking

3. **Enhanced GUI** (`main.py`)
   - ML model status monitoring
   - Alert dashboard and management
   - Model training controls

4. **Integrated Simulation** (`ingestion.py`)
   - Combined rule-based and ML anomaly detection
   - Real-time processing pipeline

## Machine Learning Models

### 1. Isolation Forest
- **Purpose**: Detect outliers in multi-dimensional feature space
- **Use Case**: Identify data points that deviate significantly from normal patterns
- **Features**: Battery voltage, solar current, temperature, memory/CPU usage, signal strength
- **Output**: Anomaly score (0-1, higher = more anomalous)

### 2. Autoencoder Neural Network
- **Purpose**: Learn normal data patterns and detect reconstruction errors
- **Architecture**: 6-4-2-4-6 dense layers with ReLU activation
- **Use Case**: Detect subtle pattern deviations and correlation anomalies
- **Training**: Learns to reconstruct normal telemetry patterns
- **Output**: Reconstruction error as anomaly score

### 3. LSTM (Long Short-Term Memory)
- **Purpose**: Detect time-series anomalies and temporal patterns
- **Architecture**: 50-unit LSTM layer with dense output
- **Use Case**: Identify temporal anomalies and sequence deviations
- **Features**: Time-windowed sequences of telemetry data
- **Output**: Sequence prediction error as anomaly score

### Ensemble Scoring
The system combines all three models using weighted averaging:
```
Combined Score = (0.4 × Isolation + 0.3 × Autoencoder + 0.3 × LSTM)
```

## Alert System

### Alert Severity Levels
- **LOW**: Minor deviations, informational
- **MEDIUM**: Notable anomalies requiring attention
- **HIGH**: Significant problems requiring action
- **CRITICAL**: Emergency situations requiring immediate response

### Notification Channels

#### 1. Email Notifications
- SMTP-based email alerts
- HTML formatted with telemetry details
- Configurable recipients and SMTP settings

#### 2. Slack Notifications
- Webhook-based Slack integration
- Channel-specific routing
- Rich message formatting with anomaly details

#### 3. Popup Notifications
- Desktop popup alerts for immediate attention
- System tray integration
- Clickable notifications for quick access

### Alert Rules
Configurable rules in `config/alert_config.json`:
```json
{
  "rules": [
    {
      "name": "Critical Battery",
      "conditions": {
        "battery_voltage": {"min": 10.0}
      },
      "severity": "CRITICAL",
      "notifications": ["email", "slack", "popup"]
    }
  ]
}
```

## Features

### 1. Real-time ML Anomaly Detection
- Continuous monitoring of telemetry streams
- Sub-second anomaly detection
- Confidence scoring and uncertainty quantification

### 2. Adaptive Thresholds
- Dynamic threshold adjustment based on historical data
- Percentile-based threshold calculation
- Model-specific threshold optimization

### 3. Feature Engineering
- Automated feature extraction and scaling
- Time-based feature windows for LSTM
- Statistical feature normalization

### 4. Model Management
- Automatic model training and retraining
- Model persistence with joblib/pickle
- Training history and performance tracking

### 5. Alert Management
- Alert acknowledgment and resolution tracking
- Alert history and statistics
- Configurable alert suppression and escalation

## Usage

### Training ML Models
```python
# Train all models
training_results = ml_detector.train_models(retrain=True)

# Train specific model
ml_detector.train_isolation_forest()
ml_detector.train_autoencoder()
ml_detector.train_lstm()
```

### Detecting Anomalies
```python
# Detect anomaly in telemetry data
anomaly_score = ml_detector.detect_anomaly(telemetry_data)

print(f"Combined Score: {anomaly_score.combined_score}")
print(f"Anomaly Type: {anomaly_score.anomaly_type}")
print(f"Confidence: {anomaly_score.confidence}")
```

### Managing Alerts
```python
# Process anomaly and create alerts
await alert_system.process_anomaly(telemetry_data, anomaly_score)

# Get alert history
alerts = alert_system.get_alert_history(limit=10)

# Acknowledge alert
alert_system.acknowledge_alert(alert_id, "Operator reviewed")
```

### GUI Controls
- **Train ML Models**: Retrain all ML models with current data
- **View Alerts**: Display recent alerts and anomalies
- **Dashboard**: Real-time ML model status and alert statistics

## Configuration

### ML Model Configuration
```python
# Isolation Forest parameters
isolation_params = {
    'n_estimators': 100,
    'contamination': 0.1,
    'random_state': 42
}

# Autoencoder parameters
autoencoder_params = {
    'hidden_layers': [4, 2, 4],
    'epochs': 100,
    'batch_size': 32
}

# LSTM parameters
lstm_params = {
    'sequence_length': 10,
    'lstm_units': 50,
    'epochs': 50
}
```

### Alert Configuration
Create/modify `config/alert_config.json`:
```json
{
  "smtp": {
    "server": "smtp.gmail.com",
    "port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password"
  },
  "slack": {
    "webhook_url": "https://hooks.slack.com/services/..."
  },
  "rules": [
    {
      "name": "ML Anomaly",
      "conditions": {
        "ml_score": {"min": 0.7}
      },
      "severity": "HIGH",
      "notifications": ["popup", "slack"]
    }
  ]
}
```

## Testing

Run the comprehensive test suite:
```bash
python test_stage3.py
```

Tests include:
- ML model training and validation
- Anomaly detection accuracy
- Alert system functionality
- Integrated simulation
- Data export capabilities

## Performance

### Typical Performance Metrics
- **Model Training**: 10-30 seconds for full retraining
- **Anomaly Detection**: <50ms per telemetry point
- **Alert Processing**: <100ms end-to-end
- **Memory Usage**: ~200MB with trained models
- **CPU Usage**: 5-15% during normal operation

### Scalability
- Handles 1000+ telemetry points per minute
- Supports multiple satellite monitoring
- Automatic model retraining on schedule
- Efficient in-memory caching

## Troubleshooting

### Common Issues

#### 1. Model Training Fails
- **Cause**: Insufficient training data (need 50+ samples)
- **Solution**: Run simulator longer to generate more data

#### 2. High False Positive Rate
- **Cause**: Threshold too low or insufficient training
- **Solution**: Increase thresholds or retrain with more diverse data

#### 3. Alerts Not Sending
- **Cause**: Misconfigured notification settings
- **Solution**: Check `alert_config.json` and network connectivity

#### 4. Poor LSTM Performance
- **Cause**: Insufficient sequence data
- **Solution**: Ensure continuous data flow for time-series patterns

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger('SmartAnomalyDetector').setLevel(logging.DEBUG)
logging.getLogger('SmartAlertSystem').setLevel(logging.DEBUG)
```

Check logs in:
- `logs/telemetry_monitor.log`
- `logs/test_stage3.log`

## Future Enhancements

1. **Advanced ML Models**
   - Transformer-based anomaly detection
   - Graph neural networks for satellite constellation monitoring
   - Federated learning across multiple ground stations

2. **Enhanced Alert Intelligence**
   - ML-based alert correlation and root cause analysis
   - Predictive alerting based on trend analysis
   - Automated incident response workflows

3. **Performance Optimization**
   - GPU acceleration for neural networks
   - Distributed model training
   - Real-time streaming analytics

4. **Integration Features**
   - REST API for external systems
   - Kafka/RabbitMQ message queue integration
   - Time-series database support (InfluxDB, TimescaleDB)

## Dependencies

### Required Packages
```
scikit-learn>=1.3.0
tensorflow>=2.13.0
keras>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Optional Packages
```
slack-sdk>=3.21.0
smtplib (built-in)
plyer>=2.1.0  # For desktop notifications
```

## License

This project is part of the STDMS system and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Run the test suite to identify specific problems
4. Consult the main STDMS documentation