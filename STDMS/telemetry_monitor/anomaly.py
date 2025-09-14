#!/usr/bin/env python3
"""
Smart Anomaly Detection Module
ML-based anomaly detection using multiple algorithms:
- Isolation Forest for outlier detection
- Autoencoder for multi-sensor anomaly detection  
- LSTM for time-series prediction and temporal anomalies
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import time

# ML Libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Deep Learning Libraries  
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import our data structures
import sys
project_root = Path(__file__).parent
sys.path.append(str(project_root))
from ingestion import TelemetryData
from storage import get_database


@dataclass
class AnomalyScore:
    """Anomaly score result from ML models"""
    timestamp: datetime
    satellite_id: str
    isolation_forest_score: float
    autoencoder_score: float
    lstm_score: float
    combined_score: float
    is_anomaly: bool
    anomaly_type: str
    confidence: float


@dataclass
class ModelMetrics:
    """ML model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    last_trained: datetime


class TelemetryFeatureExtractor:
    """Extract and prepare features from telemetry data for ML models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_columns = [
            'temperature', 'battery_voltage', 'solar_power',
            'attitude_x', 'attitude_y', 'attitude_z',
            'orbit_altitude', 'signal_strength'
        ]
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_features(self, telemetry_data: List[Dict]) -> np.ndarray:
        """Extract numerical features from telemetry data"""
        try:
            df = pd.DataFrame(telemetry_data)
            
            # Basic features
            features = df[self.feature_columns].values
            
            # Add derived features
            df['battery_temp_ratio'] = df['battery_voltage'] / (df['temperature'] + 273.15)  # Kelvin
            df['power_efficiency'] = df['solar_power'] / (df['battery_voltage'] + 0.001)
            df['attitude_magnitude'] = np.sqrt(df['attitude_x']**2 + df['attitude_y']**2 + df['attitude_z']**2)
            df['signal_altitude_ratio'] = df['signal_strength'] / df['orbit_altitude']
            
            # Time-based features (if timestamp available)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                
                # Add temporal features
                extended_features = np.column_stack([
                    features,
                    df[['battery_temp_ratio', 'power_efficiency', 'attitude_magnitude', 
                        'signal_altitude_ratio', 'hour_sin', 'hour_cos']].values
                ])
            else:
                extended_features = np.column_stack([
                    features,
                    df[['battery_temp_ratio', 'power_efficiency', 'attitude_magnitude', 
                        'signal_altitude_ratio']].values
                ])
            
            return extended_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.array([])
    
    def fit_scaler(self, features: np.ndarray):
        """Fit the feature scaler"""
        self.scaler.fit(features)
        self.is_fitted = True
        self.logger.info("Feature scaler fitted")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler"""
        if not self.is_fitted:
            self.logger.warning("Scaler not fitted, fitting on current data")
            self.fit_scaler(features)
        
        return self.scaler.transform(features)
    
    def save_scaler(self, filepath: str):
        """Save the fitted scaler"""
        joblib.dump(self.scaler, filepath)
        self.logger.info(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load a fitted scaler"""
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        self.logger.info(f"Scaler loaded from {filepath}")


class IsolationForestDetector:
    """Isolation Forest anomaly detector for outlier detection"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.threshold = 0.0
        
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the Isolation Forest model"""
        start_time = time.time()
        
        try:
            self.logger.info("Training Isolation Forest model...")
            
            # Fit the model
            self.model.fit(X)
            
            # Calculate threshold based on training data
            scores = self.model.decision_function(X)
            self.threshold = np.percentile(scores, 10)  # Bottom 10% as anomalies
            
            training_time = time.time() - start_time
            self.is_trained = True
            
            self.logger.info(f"Isolation Forest training completed in {training_time:.2f}s")
            
            return {
                'training_time': training_time,
                'threshold': self.threshold,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Isolation Forest training failed: {e}")
            raise
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies and return scores"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Get anomaly scores (higher = more normal, lower = more anomalous)
            scores = self.model.decision_function(X)
            
            # Predict anomalies (-1 = anomaly, 1 = normal)
            predictions = self.model.predict(X)
            
            # Convert to boolean (True = anomaly)
            anomalies = predictions == -1
            
            # Normalize scores to 0-1 range (1 = most anomalous)
            normalized_scores = 1 - ((scores - scores.min()) / (scores.max() - scores.min() + 1e-8))
            
            return anomalies, normalized_scores
            
        except Exception as e:
            self.logger.error(f"Isolation Forest prediction failed: {e}")
            return np.array([]), np.array([])
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        self.logger.info(f"Isolation Forest model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.is_trained = model_data['is_trained']
        self.logger.info(f"Isolation Forest model loaded from {filepath}")


class AutoencoderDetector:
    """Autoencoder neural network for multi-sensor anomaly detection"""
    
    def __init__(self, encoding_dim: int = 10, learning_rate: float = 0.001):
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.threshold = 0.0
        self.history = None
        
    def build_model(self, input_dim: int):
        """Build the autoencoder architecture"""
        try:
            # Input layer
            input_layer = Input(shape=(input_dim,))
            
            # Encoder
            encoded = Dense(32, activation='relu')(input_layer)
            encoded = Dropout(0.2)(encoded)
            encoded = Dense(16, activation='relu')(encoded)
            encoded = Dense(self.encoding_dim, activation='relu')(encoded)
            
            # Decoder
            decoded = Dense(16, activation='relu')(encoded)
            decoded = Dropout(0.2)(decoded)
            decoded = Dense(32, activation='relu')(decoded)
            decoded = Dense(input_dim, activation='linear')(decoded)
            
            # Create model
            self.model = Model(input_layer, decoded)
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"Autoencoder model built with input_dim={input_dim}, encoding_dim={self.encoding_dim}")
            
        except Exception as e:
            self.logger.error(f"Autoencoder model building failed: {e}")
            raise
    
    def train(self, X: np.ndarray, validation_split: float = 0.2, 
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train the autoencoder model"""
        start_time = time.time()
        
        try:
            if self.model is None:
                self.build_model(X.shape[1])
            
            self.logger.info("Training Autoencoder model...")
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train the model (autoencoder learns to reconstruct input)
            self.history = self.model.fit(
                X, X,  # Input and target are the same
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Calculate reconstruction errors on training data to set threshold
            reconstructions = self.model.predict(X, verbose=0)
            reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
            
            # Set threshold as 95th percentile of reconstruction errors
            self.threshold = np.percentile(reconstruction_errors, 95)
            
            training_time = time.time() - start_time
            self.is_trained = True
            
            self.logger.info(f"Autoencoder training completed in {training_time:.2f}s")
            
            return {
                'training_time': training_time,
                'threshold': self.threshold,
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1],
                'epochs_trained': len(self.history.history['loss'])
            }
            
        except Exception as e:
            self.logger.error(f"Autoencoder training failed: {e}")
            raise
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using reconstruction error"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Get reconstructions
            reconstructions = self.model.predict(X, verbose=0)
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
            
            # Determine anomalies based on threshold
            anomalies = reconstruction_errors > self.threshold
            
            # Normalize scores to 0-1 range
            max_error = max(reconstruction_errors.max(), self.threshold)
            normalized_scores = reconstruction_errors / max_error
            
            return anomalies, normalized_scores
            
        except Exception as e:
            self.logger.error(f"Autoencoder prediction failed: {e}")
            return np.array([]), np.array([])
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is not None:
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model.save(model_path)
            
            # Save additional data
            model_data = {
                'threshold': self.threshold,
                'is_trained': self.is_trained,
                'encoding_dim': self.encoding_dim,
                'learning_rate': self.learning_rate
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Autoencoder model saved to {model_path} and {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model = keras.models.load_model(model_path)
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.threshold = model_data['threshold']
            self.is_trained = model_data['is_trained']
            self.encoding_dim = model_data['encoding_dim']
            self.learning_rate = model_data['learning_rate']
            
            self.logger.info(f"Autoencoder model loaded from {model_path} and {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Autoencoder model: {e}")
            raise


class LSTMDetector:
    """LSTM model for time-series prediction and temporal anomaly detection"""
    
    def __init__(self, sequence_length: int = 10, lstm_units: int = 50, 
                 learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.threshold = 0.0
        self.history = None
        
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_model(self, n_features: int):
        """Build the LSTM architecture"""
        try:
            self.model = Sequential([
                LSTM(self.lstm_units, return_sequences=True, 
                     input_shape=(self.sequence_length, n_features)),
                Dropout(0.2),
                LSTM(self.lstm_units // 2, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(n_features, activation='linear')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"LSTM model built with sequence_length={self.sequence_length}, lstm_units={self.lstm_units}")
            
        except Exception as e:
            self.logger.error(f"LSTM model building failed: {e}")
            raise
    
    def train(self, X: np.ndarray, validation_split: float = 0.2, 
              epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """Train the LSTM model"""
        start_time = time.time()
        
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Prepare sequences
            X_seq, y_seq = self.prepare_sequences(X_scaled)
            
            if len(X_seq) == 0:
                raise ValueError("Not enough data to create sequences")
            
            if self.model is None:
                self.build_model(X.shape[-1])
            
            self.logger.info(f"Training LSTM model with {len(X_seq)} sequences...")
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train the model
            self.history = self.model.fit(
                X_seq, y_seq,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Calculate prediction errors to set threshold
            predictions = self.model.predict(X_seq, verbose=0)
            prediction_errors = np.mean(np.square(y_seq - predictions), axis=1)
            
            # Set threshold as 95th percentile of prediction errors
            self.threshold = np.percentile(prediction_errors, 95)
            
            training_time = time.time() - start_time
            self.is_trained = True
            
            self.logger.info(f"LSTM training completed in {training_time:.2f}s")
            
            return {
                'training_time': training_time,
                'threshold': self.threshold,
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1],
                'epochs_trained': len(self.history.history['loss']),
                'n_sequences': len(X_seq)
            }
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            raise
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using prediction error"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Scale the data
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Prepare sequences
            X_seq, y_seq = self.prepare_sequences(X_scaled)
            
            if len(X_seq) == 0:
                # Not enough data for sequences, return no anomalies
                return np.array([False] * len(X)), np.array([0.0] * len(X))
            
            # Get predictions
            predictions = self.model.predict(X_seq, verbose=0)
            
            # Calculate prediction errors
            prediction_errors = np.mean(np.square(y_seq - predictions), axis=1)
            
            # Determine anomalies based on threshold
            anomalies = prediction_errors > self.threshold
            
            # Normalize scores to 0-1 range
            max_error = max(prediction_errors.max(), self.threshold)
            normalized_scores = prediction_errors / max_error
            
            # Pad results to match input length (since sequences are shorter)
            full_anomalies = np.zeros(len(X), dtype=bool)
            full_scores = np.zeros(len(X))
            
            # Fill in the sequence results (starting from sequence_length index)
            if len(anomalies) > 0:
                full_anomalies[self.sequence_length:self.sequence_length + len(anomalies)] = anomalies
                full_scores[self.sequence_length:self.sequence_length + len(normalized_scores)] = normalized_scores
            
            return full_anomalies, full_scores
            
        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            return np.array([False] * len(X)), np.array([0.0] * len(X))
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is not None:
            model_path = filepath.replace('.pkl', '_lstm_model.h5')
            self.model.save(model_path)
            
            # Save additional data including scaler
            model_data = {
                'threshold': self.threshold,
                'is_trained': self.is_trained,
                'sequence_length': self.sequence_length,
                'lstm_units': self.lstm_units,
                'learning_rate': self.learning_rate,
                'scaler': self.scaler
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"LSTM model saved to {model_path} and {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            model_path = filepath.replace('.pkl', '_lstm_model.h5')
            self.model = keras.models.load_model(model_path)
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.threshold = model_data['threshold']
            self.is_trained = model_data['is_trained']
            self.sequence_length = model_data['sequence_length']
            self.lstm_units = model_data['lstm_units']
            self.learning_rate = model_data['learning_rate']
            self.scaler = model_data['scaler']
            
            self.logger.info(f"LSTM model loaded from {model_path} and {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load LSTM model: {e}")
            raise


class SmartAnomalyDetector:
    """Combined ML-based anomaly detection system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = TelemetryFeatureExtractor()
        
        # ML Models
        self.isolation_forest = IsolationForestDetector()
        self.autoencoder = AutoencoderDetector()
        self.lstm = LSTMDetector()
        
        # Model states
        self.models_trained = {
            'isolation_forest': False,
            'autoencoder': False,
            'lstm': False
        }
        
        # Scoring weights for ensemble
        self.model_weights = {
            'isolation_forest': 0.3,
            'autoencoder': 0.4,
            'lstm': 0.3
        }
        
        # Thresholds
        self.ensemble_threshold = 0.6
        self.confidence_threshold = 0.7
        
        # Model paths
        self.models_dir = project_root / "models"
        self.models_dir.mkdir(exist_ok=True)
        
    def prepare_training_data(self, min_samples: int = 100) -> Tuple[np.ndarray, List[Dict]]:
        """Prepare training data from database"""
        try:
            db = get_database()
            
            # Get historical data (more samples for better training)
            historical_data = db.get_recent_telemetry(limit=min_samples * 10, hours_back=24*7)  # Last week
            
            if len(historical_data) < min_samples:
                self.logger.warning(f"Only {len(historical_data)} samples available, minimum {min_samples} recommended")
            
            # Extract features
            features = self.feature_extractor.extract_features(historical_data)
            
            if len(features) == 0:
                raise ValueError("No features extracted from training data")
            
            # Fit scaler
            self.feature_extractor.fit_scaler(features)
            
            # Transform features
            scaled_features = self.feature_extractor.transform_features(features)
            
            self.logger.info(f"Prepared {len(scaled_features)} training samples with {scaled_features.shape[1]} features")
            
            return scaled_features, historical_data
            
        except Exception as e:
            self.logger.error(f"Failed to prepare training data: {e}")
            raise
    
    def train_models(self, retrain: bool = False) -> Dict[str, Any]:
        """Train all ML models"""
        training_results = {}
        
        try:
            self.logger.info("Starting ML model training...")
            
            # Prepare training data
            X, raw_data = self.prepare_training_data()
            
            if len(X) < 50:
                raise ValueError("Insufficient training data. Need at least 50 samples.")
            
            # Train Isolation Forest
            if not self.models_trained['isolation_forest'] or retrain:
                self.logger.info("Training Isolation Forest...")
                training_results['isolation_forest'] = self.isolation_forest.train(X)
                self.models_trained['isolation_forest'] = True
            
            # Train Autoencoder
            if not self.models_trained['autoencoder'] or retrain:
                self.logger.info("Training Autoencoder...")
                training_results['autoencoder'] = self.autoencoder.train(X)
                self.models_trained['autoencoder'] = True
            
            # Train LSTM (needs time-series preparation)
            if not self.models_trained['lstm'] or retrain:
                self.logger.info("Training LSTM...")
                
                # Group data by satellite for time-series
                satellite_data = {}
                for i, sample in enumerate(raw_data):
                    sat_id = sample['satellite_id']
                    if sat_id not in satellite_data:
                        satellite_data[sat_id] = []
                    satellite_data[sat_id].append(X[i])
                
                # Use the satellite with most data for LSTM training
                best_sat = max(satellite_data.keys(), key=lambda k: len(satellite_data[k]))
                lstm_data = np.array(satellite_data[best_sat])
                
                if len(lstm_data) >= 20:  # Minimum for LSTM sequences
                    training_results['lstm'] = self.lstm.train(lstm_data)
                    self.models_trained['lstm'] = True
                else:
                    self.logger.warning("Insufficient time-series data for LSTM training")
            
            # Save models
            self.save_models()
            
            # Save feature scaler
            scaler_path = self.models_dir / "feature_scaler.pkl"
            self.feature_extractor.save_scaler(str(scaler_path))
            
            self.logger.info("ML model training completed successfully")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
            raise
    
    def predict_anomaly(self, telemetry_data: List[Dict]) -> List[AnomalyScore]:
        """Predict anomalies using ensemble of ML models"""
        try:
            if not any(self.models_trained.values()):
                raise ValueError("No models trained yet")
            
            # Extract and scale features
            features = self.feature_extractor.extract_features(telemetry_data)
            if len(features) == 0:
                return []
            
            scaled_features = self.feature_extractor.transform_features(features)
            
            # Get predictions from each model
            results = []
            
            for i, data in enumerate(telemetry_data):
                # Initialize scores
                if_score = 0.0
                ae_score = 0.0
                lstm_score = 0.0
                
                # Isolation Forest prediction
                if self.models_trained['isolation_forest']:
                    _, if_scores = self.isolation_forest.predict_anomaly(scaled_features[i:i+1])
                    if_score = if_scores[0] if len(if_scores) > 0 else 0.0
                
                # Autoencoder prediction
                if self.models_trained['autoencoder']:
                    _, ae_scores = self.autoencoder.predict_anomaly(scaled_features[i:i+1])
                    ae_score = ae_scores[0] if len(ae_scores) > 0 else 0.0
                
                # LSTM prediction (needs sequence context)
                if self.models_trained['lstm'] and len(scaled_features) >= self.lstm.sequence_length:
                    start_idx = max(0, i - self.lstm.sequence_length + 1)
                    end_idx = i + 1
                    lstm_features = scaled_features[start_idx:end_idx]
                    
                    if len(lstm_features) >= self.lstm.sequence_length:
                        # Pad if necessary
                        if len(lstm_features) < self.lstm.sequence_length:
                            padding = np.zeros((self.lstm.sequence_length - len(lstm_features), lstm_features.shape[1]))
                            lstm_features = np.vstack([padding, lstm_features])
                        
                        lstm_features = lstm_features.reshape(1, self.lstm.sequence_length, -1)
                        _, lstm_scores = self.lstm.predict_anomaly(lstm_features)
                        lstm_score = lstm_scores[-1] if len(lstm_scores) > 0 else 0.0
                
                # Calculate ensemble score
                combined_score = (
                    self.model_weights['isolation_forest'] * if_score +
                    self.model_weights['autoencoder'] * ae_score +
                    self.model_weights['lstm'] * lstm_score
                )
                
                # Determine if anomaly
                is_anomaly = combined_score > self.ensemble_threshold
                
                # Determine anomaly type
                anomaly_type = "normal"
                if is_anomaly:
                    if if_score > 0.7:
                        anomaly_type = "outlier"
                    elif ae_score > 0.7:
                        anomaly_type = "multi_sensor_anomaly"
                    elif lstm_score > 0.7:
                        anomaly_type = "temporal_anomaly"
                    else:
                        anomaly_type = "ensemble_anomaly"
                
                # Calculate confidence
                confidence = min(combined_score, 1.0) if is_anomaly else min(1.0 - combined_score, 1.0)
                
                # Create anomaly score result
                timestamp = data.get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                
                result = AnomalyScore(
                    timestamp=timestamp,
                    satellite_id=data.get('satellite_id', 'unknown'),
                    isolation_forest_score=if_score,
                    autoencoder_score=ae_score,
                    lstm_score=lstm_score,
                    combined_score=combined_score,
                    is_anomaly=is_anomaly,
                    anomaly_type=anomaly_type,
                    confidence=confidence
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Anomaly prediction failed: {e}")
            return []
    
    def save_models(self):
        """Save all trained models"""
        try:
            if self.models_trained['isolation_forest']:
                if_path = self.models_dir / "isolation_forest.pkl"
                self.isolation_forest.save_model(str(if_path))
            
            if self.models_trained['autoencoder']:
                ae_path = self.models_dir / "autoencoder.pkl"
                self.autoencoder.save_model(str(ae_path))
            
            if self.models_trained['lstm']:
                lstm_path = self.models_dir / "lstm.pkl"
                self.lstm.save_model(str(lstm_path))
            
            # Save configuration
            config = {
                'models_trained': self.models_trained,
                'model_weights': self.model_weights,
                'ensemble_threshold': self.ensemble_threshold,
                'confidence_threshold': self.confidence_threshold
            }
            
            config_path = self.models_dir / "anomaly_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load configuration
            config_path = self.models_dir / "anomaly_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.models_trained = config.get('models_trained', {})
                self.model_weights = config.get('model_weights', self.model_weights)
                self.ensemble_threshold = config.get('ensemble_threshold', self.ensemble_threshold)
                self.confidence_threshold = config.get('confidence_threshold', self.confidence_threshold)
            
            # Load feature scaler
            scaler_path = self.models_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                self.feature_extractor.load_scaler(str(scaler_path))
            
            # Load models
            if self.models_trained.get('isolation_forest', False):
                if_path = self.models_dir / "isolation_forest.pkl"
                if if_path.exists():
                    self.isolation_forest.load_model(str(if_path))
            
            if self.models_trained.get('autoencoder', False):
                ae_path = self.models_dir / "autoencoder.pkl"
                if ae_path.exists():
                    self.autoencoder.load_model(str(ae_path))
            
            if self.models_trained.get('lstm', False):
                lstm_path = self.models_dir / "lstm.pkl"
                if lstm_path.exists():
                    self.lstm.load_model(str(lstm_path))
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            'models_trained': self.models_trained,
            'ensemble_threshold': self.ensemble_threshold,
            'model_weights': self.model_weights,
            'feature_scaler_fitted': self.feature_extractor.is_fitted
        }


# Global anomaly detector instance
_anomaly_detector = None

def get_anomaly_detector() -> SmartAnomalyDetector:
    """Get global anomaly detector instance"""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = SmartAnomalyDetector()
        # Try to load existing models
        try:
            _anomaly_detector.load_models()
        except:
            pass  # No models to load yet
    return _anomaly_detector


if __name__ == "__main__":
    # Test the anomaly detection system
    logging.basicConfig(level=logging.INFO)
    
    detector = SmartAnomalyDetector()
    
    print("Testing Smart Anomaly Detection System...")
    
    try:
        # Train models
        results = detector.train_models()
        print(f"Training results: {results}")
        
        # Test prediction
        # (This would normally use real telemetry data)
        print("Smart anomaly detection system test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")