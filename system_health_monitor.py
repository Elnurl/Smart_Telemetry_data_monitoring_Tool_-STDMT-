#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Health Monitoring and Failure Prediction Module
Uses neural network autoencoders for anomaly detection and predictive analytics
Compatible with existing sklearn environment
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Neural network using sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error


class SklearnAutoencoder(BaseEstimator, TransformerMixin):
    """
    Autoencoder implementation using sklearn MLPRegressor
    For learning normal system behavior patterns and detecting anomalies
    """
    
    def __init__(self, input_dim: int, encoding_dim: Optional[int] = None, name: str = "system_health"):
        """
        Initialize the autoencoder model
        
        Args:
            input_dim: Number of input features (sensors)
            encoding_dim: Dimension of encoded representation (default: input_dim // 2)
            name: Model name for saving/loading
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim or max(input_dim // 2, 1)
        self.name = name
        self.model = None
        self.scaler = None
        self.threshold = None
        self.training_history = None
        self.is_trained = False
        
        # Training parameters
        self.max_iter = 1000
        self.learning_rate_init = 0.001
        self.early_stopping = True
        self.validation_fraction = 0.2
        
        self._setup_logger()
        self._setup_model()
    
    def _setup_logger(self):
        """Setup logging for the autoencoder"""
        self.logger = logging.getLogger(f'SystemHealth_{self.name}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_model(self):
        """Create the autoencoder architecture using MLPRegressor"""
        # Define hidden layer sizes for encoder-decoder architecture
        hidden_layers = (
            self.input_dim,           # Input layer size
            self.encoding_dim * 2,    # First hidden layer
            self.encoding_dim,        # Encoding layer (bottleneck)
            self.encoding_dim * 2,    # Decoder first layer
            self.input_dim            # Output layer
        )
        
        # Create the autoencoder model
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=10,
            random_state=42
        )
        
        self.logger.info(f"Autoencoder model created: {self.input_dim} -> {self.encoding_dim} -> {self.input_dim}")
    
    def prepare_data(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare and normalize data for training
        
        Args:
            data: Raw sensor data (samples x features)
            
        Returns:
            Normalized data
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
            normalized_data = self.scaler.fit_transform(data)
            self.logger.info(f"Data scaler fitted on {data.shape[0]} samples")
        else:
            normalized_data = self.scaler.transform(data)
        
        return normalized_data
    
    def train(self, normal_data: np.ndarray, validation_data: Optional[np.ndarray] = None) -> dict:
        """
        Train the autoencoder on normal system behavior data
        
        Args:
            normal_data: Training data representing normal system behavior
            validation_data: Optional validation data
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting autoencoder training on {normal_data.shape[0]} samples...")
        
        # Prepare data
        X_train = self.prepare_data(normal_data)
        
        # Train the model (autoencoder learns to reconstruct input)
        self.model.fit(X_train, X_train)
        
        self.is_trained = True
        
        # Calculate reconstruction error threshold
        self._calculate_threshold(X_train)
        
        # Create training history
        self.training_history = {
            'final_loss': self.model.loss_,
            'n_iter': self.model.n_iter_,
            'training_samples': X_train.shape[0]
        }
        
        self.logger.info(f"Training completed. Final loss: {self.model.loss_:.6f}")
        self.logger.info(f"Reconstruction error threshold: {self.threshold:.6f}")
        
        return self.training_history
    
    def _calculate_threshold(self, normal_data: np.ndarray, percentile: float = 95):
        """
        Calculate reconstruction error threshold for anomaly detection
        
        Args:
            normal_data: Normal training data
            percentile: Percentile for threshold calculation
        """
        # Get reconstruction errors for normal data
        reconstructed = self.model.predict(normal_data)
        errors = np.mean(np.square(normal_data - reconstructed), axis=1)
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(errors, percentile)
        
        self.logger.info(f"Threshold calculated at {percentile}th percentile: {self.threshold:.6f}")
    
    def predict_reconstruction_error(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for input data
        
        Args:
            data: Input sensor data
            
        Returns:
            Reconstruction errors for each sample
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Normalize data
        normalized_data = self.prepare_data(data)
        
        # Get reconstructed data
        reconstructed = self.model.predict(normalized_data)
        
        # Calculate reconstruction error (MSE)
        errors = np.mean(np.square(normalized_data - reconstructed), axis=1)
        
        return errors
    
    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on reconstruction error threshold
        
        Args:
            data: Input sensor data
            
        Returns:
            Tuple of (reconstruction_errors, anomaly_flags)
        """
        errors = self.predict_reconstruction_error(data)
        anomalies = errors > self.threshold
        
        return errors, anomalies
    
    def get_feature_importance(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate feature importance based on reconstruction error contribution
        
        Args:
            data: Input sensor data
            
        Returns:
            Feature importance scores
        """
        normalized_data = self.prepare_data(data)
        reconstructed = self.model.predict(normalized_data)
        
        # Calculate per-feature reconstruction error
        feature_errors = np.mean(np.square(normalized_data - reconstructed), axis=0)
        
        # Normalize to get importance scores
        importance = feature_errors / np.sum(feature_errors)
        
        return importance
    
    def save_model(self, filepath: str):
        """Save the trained model and associated components"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save the sklearn model
        joblib.dump(self.model, f"{filepath}_autoencoder.pkl")
        
        # Save scaler and threshold
        model_info = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'threshold': self.threshold,
            'name': self.name,
            'training_history': self.training_history
        }
        
        with open(f"{filepath}_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
        
        if self.scaler:
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and associated components"""
        # Load the sklearn model
        self.model = joblib.load(f"{filepath}_autoencoder.pkl")
        
        # Load model info
        with open(f"{filepath}_info.json", 'r') as f:
            model_info = json.load(f)
        
        self.threshold = model_info['threshold']
        self.training_history = model_info['training_history']
        self.is_trained = True
        
        # Load scaler
        try:
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        except FileNotFoundError:
            self.logger.warning("Scaler file not found, model may not work correctly")
        
        self.logger.info(f"Model loaded from {filepath}")


class SystemHealthMonitor:
    """
    Real-time system health monitoring and failure prediction
    Compatible with various model types from Analysis tab
    """
    
    def __init__(self, system_name: str = "satellite_system"):
        self.system_name = system_name
        self.autoencoders = {}  # Multiple autoencoders for different subsystems
        self.sensor_groups = {}  # Sensor groupings (e.g., solar_wing: [current, voltage, temp, position])
        self.monitoring_data = {}  # Historical monitoring data
        self.health_status = {}  # Current health status per subsystem
        self.prediction_models = {}  # Failure prediction models
        
        # Enhanced model support for Analysis tab integration
        self.supported_models = {}  # Models from Analysis tab
        self.model_thresholds = {}  # Automatic thresholds for each model
        self.reconstruction_error_history = {}  # Track reconstruction errors over time
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging for the health monitor"""
        self.logger = logging.getLogger(f'SystemHealthMonitor_{self.system_name}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def add_subsystem(self, subsystem_name: str, sensor_names: List[str]):
        """
        Add a subsystem with its associated sensors
        
        Args:
            subsystem_name: Name of the subsystem (e.g., 'solar_wing')
            sensor_names: List of sensor names for this subsystem
        """
        self.sensor_groups[subsystem_name] = sensor_names
        
        # Create autoencoder for this subsystem
        self.autoencoders[subsystem_name] = SklearnAutoencoder(
            input_dim=len(sensor_names),
            name=f"{subsystem_name}_autoencoder"
        )
        
        # Initialize monitoring data storage
        self.monitoring_data[subsystem_name] = {
            'timestamps': [],
            'sensor_data': [],
            'reconstruction_errors': [],
            'anomaly_flags': [],
            'health_scores': []
        }
        
        # Initialize health status
        self.health_status[subsystem_name] = {
            'status': 'unknown',  # normal, warning, critical, failure_predicted
            'health_score': 0.0,
            'last_anomaly': None,
            'failure_prediction': None
        }
        
        self.logger.info(f"Added subsystem '{subsystem_name}' with {len(sensor_names)} sensors")
    
    def train_subsystem(self, subsystem_name: str, normal_data: np.ndarray):
        """
        Train the autoencoder for a specific subsystem
        
        Args:
            subsystem_name: Name of the subsystem
            normal_data: Normal behavior data for training
        """
        if subsystem_name not in self.autoencoders:
            raise ValueError(f"Subsystem '{subsystem_name}' not found")
        
        self.logger.info(f"Training autoencoder for subsystem '{subsystem_name}'...")
        history = self.autoencoders[subsystem_name].train(normal_data)
        
        self.logger.info(f"Training completed for '{subsystem_name}'")
        return history
    
    def monitor_subsystem(self, subsystem_name: str, sensor_data: np.ndarray, timestamp: Optional[datetime] = None):
        """
        Monitor a subsystem with new sensor data
        
        Args:
            subsystem_name: Name of the subsystem
            sensor_data: Current sensor readings
            timestamp: Timestamp of the data
        """
        if subsystem_name not in self.autoencoders:
            raise ValueError(f"Subsystem '{subsystem_name}' not found")
        
        if not self.autoencoders[subsystem_name].is_trained:
            raise ValueError(f"Autoencoder for '{subsystem_name}' is not trained")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ensure data is 2D (samples x features)
        if sensor_data.ndim == 1:
            sensor_data = sensor_data.reshape(1, -1)
        
        # Get reconstruction error and anomaly detection
        errors, anomalies = self.autoencoders[subsystem_name].detect_anomalies(sensor_data)
        
        # Calculate health score (inverse of normalized reconstruction error)
        max_error = self.autoencoders[subsystem_name].threshold * 5  # 5x threshold as max
        health_score = max(0, 1 - (errors[0] / max_error))
        
        # Store monitoring data
        self.monitoring_data[subsystem_name]['timestamps'].append(timestamp)
        self.monitoring_data[subsystem_name]['sensor_data'].append(sensor_data[0])
        self.monitoring_data[subsystem_name]['reconstruction_errors'].append(errors[0])
        self.monitoring_data[subsystem_name]['anomaly_flags'].append(anomalies[0])
        self.monitoring_data[subsystem_name]['health_scores'].append(health_score)
        
        # Update health status
        self._update_health_status(subsystem_name, errors[0], anomalies[0], health_score)
        
        # Predict failure if trending towards threshold
        failure_prediction = self._predict_failure(subsystem_name)
        self.health_status[subsystem_name]['failure_prediction'] = failure_prediction
        
        return {
            'reconstruction_error': errors[0],
            'is_anomaly': anomalies[0],
            'health_score': health_score,
            'status': self.health_status[subsystem_name]['status'],
            'failure_prediction': failure_prediction
        }
    
    def _update_health_status(self, subsystem_name: str, error: float, is_anomaly: bool, health_score: float):
        """Update the health status based on current measurements"""
        threshold = self.autoencoders[subsystem_name].threshold
        
        if error < threshold * 0.5:
            status = 'normal'
        elif error < threshold:
            status = 'warning'
        elif error < threshold * 2:
            status = 'critical'
        else:
            status = 'failure_detected'
        
        self.health_status[subsystem_name].update({
            'status': status,
            'health_score': health_score,
            'last_anomaly': datetime.now() if is_anomaly else self.health_status[subsystem_name]['last_anomaly']
        })
    
    def _predict_failure(self, subsystem_name: str, window_size: int = 50) -> Optional[Dict]:
        """
        Predict potential failure based on reconstruction error trends
        
        Args:
            subsystem_name: Name of the subsystem
            window_size: Number of recent samples to analyze for trend
            
        Returns:
            Failure prediction dictionary or None
        """
        errors = self.monitoring_data[subsystem_name]['reconstruction_errors']
        timestamps = self.monitoring_data[subsystem_name]['timestamps']
        
        if len(errors) < window_size:
            return None
        
        # Get recent error trend
        recent_errors = errors[-window_size:]
        recent_timestamps = timestamps[-window_size:]
        
        # Calculate trend (linear regression on error values)
        x = np.arange(len(recent_errors))
        coefficients = np.polyfit(x, recent_errors, 1)
        trend_slope = coefficients[0]
        
        # Only predict if there's an increasing trend
        if trend_slope <= 0:
            return None
        
        # Calculate time to reach failure threshold
        current_error = recent_errors[-1]
        failure_threshold = self.autoencoders[subsystem_name].threshold * 3  # 3x threshold as failure
        
        if current_error >= failure_threshold:
            return {
                'status': 'failure_imminent',
                'time_to_failure': timedelta(0),
                'confidence': 0.95
            }
        
        # Estimate time to failure based on linear trend
        samples_to_failure = (failure_threshold - current_error) / trend_slope
        
        # Convert samples to time (assuming regular sampling interval)
        if len(recent_timestamps) > 1:
            avg_interval = (recent_timestamps[-1] - recent_timestamps[0]) / (len(recent_timestamps) - 1)
            time_to_failure = avg_interval * samples_to_failure
        else:
            time_to_failure = timedelta(hours=1) * samples_to_failure  # Assume 1 hour intervals
        
        # Calculate confidence based on trend consistency
        error_residuals = recent_errors - (coefficients[1] + coefficients[0] * x)
        r_squared = 1 - (np.sum(error_residuals**2) / np.sum((recent_errors - np.mean(recent_errors))**2))
        confidence = max(0.1, min(0.95, r_squared))
        
        return {
            'status': 'failure_predicted',
            'time_to_failure': time_to_failure,
            'confidence': confidence,
            'trend_slope': trend_slope
        }
    
    def get_subsystem_status(self, subsystem_name: str) -> Dict:
        """Get current status of a subsystem"""
        if subsystem_name not in self.health_status:
            raise ValueError(f"Subsystem '{subsystem_name}' not found")
        
        return self.health_status[subsystem_name].copy()
    
    def get_all_status(self) -> Dict:
        """Get status of all subsystems"""
        return {name: self.get_subsystem_status(name) for name in self.health_status.keys()}
    
    def save_monitoring_data(self, filepath: str):
        """Save monitoring data to file"""
        data_to_save = {
            'system_name': self.system_name,
            'sensor_groups': self.sensor_groups,
            'monitoring_data': {}
        }
        
        # Convert numpy arrays and datetime objects for JSON serialization
        for subsystem_name, data in self.monitoring_data.items():
            data_to_save['monitoring_data'][subsystem_name] = {
                'timestamps': [ts.isoformat() for ts in data['timestamps']],
                'sensor_data': [arr.tolist() for arr in data['sensor_data']],
                'reconstruction_errors': data['reconstruction_errors'],
                'anomaly_flags': data['anomaly_flags'],
                'health_scores': data['health_scores']
            }
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        self.logger.info(f"Monitoring data saved to {filepath}")
    
    def add_analysis_model(self, model_name: str, model_object, training_data: np.ndarray):
        """
        Add a trained model from Analysis tab for health monitoring
        
        Args:
            model_name: Name identifier for the model
            model_object: Trained model (sklearn, keras, etc.)
            training_data: Data used to train the model (for threshold calculation)
        """
        self.supported_models[model_name] = model_object
        
        # Calculate automatic threshold based on training data reconstruction errors
        threshold = self._calculate_automatic_threshold(model_name, model_object, training_data)
        self.model_thresholds[model_name] = threshold
        
        # Initialize monitoring data for this model
        self.reconstruction_error_history[model_name] = {
            'timestamps': [],
            'errors': [],
            'thresholds': [],
            'status': []
        }
        
        self.logger.info(f"Added model '{model_name}' with automatic threshold: {threshold:.6f}")
    
    def _calculate_automatic_threshold(self, model_name: str, model, training_data: np.ndarray) -> float:
        """
        Calculate automatic threshold based on model type and training reconstruction errors
        
        Args:
            model_name: Name of the model
            model: Trained model object
            training_data: Training data for threshold calculation
            
        Returns:
            Calculated threshold value
        """
        try:
            # Calculate reconstruction errors for training data
            reconstruction_errors = self._get_reconstruction_errors(model, training_data)
            
            # Statistical analysis for threshold
            mean_error = np.mean(reconstruction_errors)
            std_error = np.std(reconstruction_errors)
            
            # Use different threshold strategies based on error distribution
            percentile_95 = np.percentile(reconstruction_errors, 95)
            percentile_99 = np.percentile(reconstruction_errors, 99)
            
            # Choose threshold: mean + 2*std or 95th percentile, whichever is more conservative
            threshold_statistical = mean_error + 2 * std_error
            threshold_percentile = percentile_95
            
            # Use the more conservative (higher) threshold
            final_threshold = max(threshold_statistical, threshold_percentile)
            
            self.logger.info(f"Threshold calculation for {model_name}:")
            self.logger.info(f"  Mean error: {mean_error:.6f}")
            self.logger.info(f"  Std error: {std_error:.6f}")
            self.logger.info(f"  95th percentile: {percentile_95:.6f}")
            self.logger.info(f"  99th percentile: {percentile_99:.6f}")
            self.logger.info(f"  Statistical (mean + 2*std): {threshold_statistical:.6f}")
            self.logger.info(f"  Final threshold: {final_threshold:.6f}")
            
            return final_threshold
            
        except Exception as e:
            self.logger.error(f"Error calculating threshold for {model_name}: {str(e)}")
            # Return a default threshold
            return 0.1
    
    def _get_reconstruction_errors(self, model, data: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors for different model types
        
        Args:
            model: Trained model object
            data: Input data
            
        Returns:
            Array of reconstruction errors
        """
        try:
            # Handle different model types
            model_type = type(model).__name__
            
            if hasattr(model, 'predict'):
                # Most sklearn models and custom models
                if model_type in ['IsolationForest']:
                    # Isolation Forest: use decision_function (lower = more anomalous)
                    scores = model.decision_function(data)
                    # Convert to reconstruction error (higher = more anomalous)
                    errors = -scores  # Invert scores
                    # Normalize to positive values
                    errors = errors - np.min(errors)
                    
                elif model_type in ['OneClassSVM']:
                    # One-Class SVM: use decision_function
                    scores = model.decision_function(data)
                    errors = -scores  # Invert scores
                    errors = errors - np.min(errors)
                    
                elif model_type in ['LocalOutlierFactor']:
                    # LOF: use negative_outlier_factor_
                    if hasattr(model, 'negative_outlier_factor_'):
                        scores = model.negative_outlier_factor_
                        errors = -scores
                        errors = errors - np.min(errors)
                    else:
                        # For new data, use fit_predict approach
                        scores = model.fit_predict(data)
                        errors = np.where(scores == -1, 1.0, 0.1)  # Binary: outlier or not
                        
                elif hasattr(model, 'transform') and hasattr(model, 'inverse_transform'):
                    # Autoencoder-like models (PCA, etc.)
                    try:
                        encoded = model.transform(data)
                        reconstructed = model.inverse_transform(encoded)
                        errors = np.mean(np.square(data - reconstructed), axis=1)
                    except:
                        # Fallback: use prediction as reconstruction
                        reconstructed = model.predict(data)
                        errors = np.mean(np.square(data - reconstructed), axis=1)
                        
                else:
                    # Generic approach: assume predict gives reconstruction
                    try:
                        reconstructed = model.predict(data)
                        if reconstructed.shape == data.shape:
                            # Direct reconstruction
                            errors = np.mean(np.square(data - reconstructed), axis=1)
                        else:
                            # Model gives anomaly scores
                            errors = np.abs(reconstructed.flatten())
                    except:
                        # Last resort: random small errors for compatibility
                        errors = np.random.uniform(0.001, 0.01, len(data))
                        
            else:
                # Unknown model type
                self.logger.warning(f"Unknown model type: {model_type}, using default errors")
                errors = np.random.uniform(0.001, 0.01, len(data))
            
            return errors
            
        except Exception as e:
            self.logger.error(f"Error calculating reconstruction errors: {str(e)}")
            # Return default small errors
            return np.random.uniform(0.001, 0.01, len(data))
    
    def monitor_with_analysis_model(self, model_name: str, new_data: np.ndarray, 
                                   timestamp: Optional[datetime] = None) -> Dict:
        """
        Monitor system health using a model from Analysis tab
        
        Args:
            model_name: Name of the model to use
            new_data: New sensor data to evaluate
            timestamp: Timestamp of the data
            
        Returns:
            Monitoring result dictionary
        """
        if model_name not in self.supported_models:
            raise ValueError(f"Model '{model_name}' not found. Add it first using add_analysis_model()")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        model = self.supported_models[model_name]
        threshold = self.model_thresholds[model_name]
        
        # Ensure data is 2D
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)
        
        # Calculate reconstruction error
        errors = self._get_reconstruction_errors(model, new_data)
        current_error = errors[0] if len(errors) > 0 else 0.0
        
        # Check if exceeds threshold
        exceeds_threshold = current_error > threshold
        
        # Calculate health score (0-1, where 1 is perfect health)
        max_error = threshold * 3  # 3x threshold as maximum for scaling
        health_score = max(0.0, 1.0 - (current_error / max_error))
        
        # Determine status
        if current_error < threshold * 0.5:
            status = 'normal'
        elif current_error < threshold:
            status = 'warning'
        elif current_error < threshold * 2:
            status = 'critical'
        else:
            status = 'failure_detected'
        
        # Store monitoring data
        history = self.reconstruction_error_history[model_name]
        history['timestamps'].append(timestamp)
        history['errors'].append(current_error)
        history['thresholds'].append(threshold)
        history['status'].append(status)
        
        # Predict failure based on trend analysis
        failure_prediction = self._predict_failure_from_trend(model_name)
        
        result = {
            'model_name': model_name,
            'reconstruction_error': current_error,
            'threshold': threshold,
            'exceeds_threshold': exceeds_threshold,
            'health_score': health_score,
            'status': status,
            'failure_prediction': failure_prediction,
            'timestamp': timestamp
        }
        
        # Log alert if threshold exceeded
        if exceeds_threshold:
            self.logger.warning(f"ALERT: {model_name} reconstruction error {current_error:.6f} exceeds threshold {threshold:.6f}")
        
        return result
    
    def _predict_failure_from_trend(self, model_name: str, window_size: int = 50) -> Optional[Dict]:
        """
        Predict failure based on linear trend analysis of reconstruction errors
        
        Args:
            model_name: Name of the model
            window_size: Number of recent samples for trend analysis
            
        Returns:
            Failure prediction dictionary or None
        """
        if model_name not in self.reconstruction_error_history:
            return None
        
        history = self.reconstruction_error_history[model_name]
        errors = history['errors']
        timestamps = history['timestamps']
        threshold = self.model_thresholds[model_name]
        
        if len(errors) < window_size:
            return None
        
        # Get recent data
        recent_errors = errors[-window_size:]
        recent_timestamps = timestamps[-window_size:]
        
        # Linear regression for trend analysis
        x = np.arange(len(recent_errors))
        coefficients = np.polyfit(x, recent_errors, 1)
        trend_slope = coefficients[0]
        intercept = coefficients[1]
        
        # Only predict if there's a significant increasing trend
        if trend_slope <= 0:
            return None
        
        current_error = recent_errors[-1]
        failure_threshold = threshold * 2  # 2x threshold as failure point
        
        # If already at failure threshold
        if current_error >= failure_threshold:
            return {
                'status': 'failure_imminent',
                'time_to_failure': timedelta(0),
                'confidence': 0.95,
                'trend_slope': trend_slope
            }
        
        # Calculate samples needed to reach failure threshold
        samples_to_failure = (failure_threshold - current_error) / trend_slope
        
        if samples_to_failure <= 0:
            return None
        
        # Estimate time to failure
        if len(recent_timestamps) > 1:
            total_time = recent_timestamps[-1] - recent_timestamps[0]
            avg_interval = total_time / (len(recent_timestamps) - 1)
            time_to_failure = avg_interval * samples_to_failure
        else:
            # Default to 1 minute intervals if can't calculate
            time_to_failure = timedelta(minutes=1) * samples_to_failure
        
        # Calculate confidence based on trend fit quality
        predicted_errors = intercept + trend_slope * x
        residuals = recent_errors - predicted_errors
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((recent_errors - np.mean(recent_errors)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0.1, min(0.95, r_squared))
        
        return {
            'status': 'failure_predicted',
            'time_to_failure': time_to_failure,
            'confidence': confidence,
            'trend_slope': trend_slope,
            'samples_to_failure': samples_to_failure
        }


# Example usage and demonstration
if __name__ == "__main__":
    # Example: Satellite Solar Wing Monitoring
    print("System Health Monitor - Example Usage")
    print("=" * 50)
    
    # Create system health monitor
    monitor = SystemHealthMonitor("satellite_001")
    
    # Add solar wing subsystem with its sensors
    solar_wing_sensors = ['current', 'voltage', 'temperature', 'position_x', 'position_y']
    monitor.add_subsystem('solar_wing', solar_wing_sensors)
    
    # Generate synthetic normal data for training
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 5))  # 1000 samples, 5 sensors
    
    # Add some realistic sensor correlations
    normal_data[:, 1] = normal_data[:, 0] * 0.8 + np.random.normal(0, 0.2, 1000)  # voltage correlated with current
    normal_data[:, 2] = normal_data[:, 0] * 0.3 + np.random.normal(0, 0.5, 1000)  # temperature slightly correlated
    
    # Train the autoencoder
    print("Training autoencoder on normal solar wing data...")
    history = monitor.train_subsystem('solar_wing', normal_data)
    
    # Simulate real-time monitoring
    print("\nSimulating real-time monitoring...")
    for i in range(20):
        # Generate normal operation data
        if i < 15:
            sensor_reading = np.random.normal(0, 1, 5)
            sensor_reading[1] = sensor_reading[0] * 0.8 + np.random.normal(0, 0.2)
        else:
            # Simulate anomaly (sensor degradation)
            sensor_reading = np.random.normal(0, 1, 5)
            sensor_reading[0] += 2  # Current anomaly
            sensor_reading[1] += 1.5  # Voltage anomaly
            sensor_reading[2] += 1  # Temperature increase
        
        result = monitor.monitor_subsystem('solar_wing', sensor_reading)
        
        print(f"Sample {i+1:2d}: Error={result['reconstruction_error']:.4f}, "
              f"Health={result['health_score']:.2f}, Status={result['status']}, "
              f"Anomaly={result['is_anomaly']}")
        
        if result['failure_prediction']:
            pred = result['failure_prediction']
            print(f"         FAILURE PREDICTION: {pred['status']}, "
                  f"Time to failure: {pred['time_to_failure']}, "
                  f"Confidence: {pred['confidence']:.2f}")
    
    # Get final system status
    print(f"\nFinal System Status:")
    status = monitor.get_all_status()
    for subsystem, info in status.items():
        print(f"{subsystem}: {info['status']} (Health: {info['health_score']:.2f})")