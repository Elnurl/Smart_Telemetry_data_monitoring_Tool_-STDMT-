"""Multi-agent fusion models extracted from main.py (Slice C Wave 1)."""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("SecureAnomalyDetection")

try:
    import psutil  # noqa: F401
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from pmdarima import auto_arima  # noqa: F401
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    auto_arima = None  # type: ignore

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_ARIMA_AVAILABLE = True
except ImportError:
    STATSMODELS_ARIMA_AVAILABLE = False
    ARIMA = None  # type: ignore
    SARIMAX = None  # type: ignore

TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    if tf.__version__:
        from tensorflow.keras.layers import (
            Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape,
        )
        from tensorflow.keras.models import Model, Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        TENSORFLOW_AVAILABLE = True
except (ImportError, AttributeError):
    logger.warning("TensorFlow not available. Deep learning models will be disabled.")

try:
    from system_health_monitor import SystemHealthMonitor, SklearnAutoencoder
except ImportError:
    SystemHealthMonitor = None  # type: ignore
    SklearnAutoencoder = None  # type: ignore


def create_sequences(data, sequence_length):
    """
    Create sequences from time series data for LSTM
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        sequence_length: int, length of each sequence
        
    Returns:
        numpy array of shape (n_samples - sequence_length, sequence_length, n_features)
    """
    if len(data) < sequence_length:
        raise ValueError(f"Data length {len(data)} is less than sequence length {sequence_length}")
    
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
    
    return np.array(sequences)


def create_sequences_3d(data, sequence_length):
    """
    Create 3D sequences from time series data for CNN
    Adds channel dimension for Conv1D
    
    Args:
        data: numpy array of shape (n_samples, n_features)
        sequence_length: int, length of each sequence
        
    Returns:
        numpy array of shape (n_samples - sequence_length, sequence_length, n_features, 1)
    """
    sequences = create_sequences(data, sequence_length)
    # Add channel dimension for CNN
    return np.expand_dims(sequences, axis=-1)





class AutoencoderLSTMAgent:
    """
    Agent 1: Hybrid Autoencoder + LSTM for reconstruction error analysis
    Uses reconstruction error to detect anomalies and LSTM for temporal patterns
    """
    
    def __init__(self, input_dim, sequence_length=10, encoding_dim=8):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.lstm_model = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info(f"Initialized AutoencoderLSTMAgent: input_dim={input_dim}, seq_len={sequence_length}")
    
    def build_autoencoder(self):
        """Build the autoencoder model"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for AutoencoderLSTMAgent")
        
        # Encoder
        encoder_input = Input(shape=(self.input_dim,), name='encoder_input')
        encoded = Dense(64, activation='relu', name='encoder_1')(encoder_input)
        encoded = Dense(32, activation='relu', name='encoder_2')(encoded)
        encoded = Dense(self.encoding_dim, activation='relu', name='encoding')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu', name='decoder_1')(encoded)
        decoded = Dense(64, activation='relu', name='decoder_2')(decoded)
        decoder_output = Dense(self.input_dim, activation='linear', name='decoder_output')(decoded)
        
        # Autoencoder model
        self.autoencoder = Model(encoder_input, decoder_output, name='autoencoder')
        self.autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        logger.info("Autoencoder model built successfully")
    
    def build_lstm(self):
        """Build the LSTM model for temporal patterns"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for AutoencoderLSTMAgent")
        
        lstm_input = Input(shape=(self.sequence_length, self.input_dim), name='lstm_input')
        lstm_out = LSTM(64, return_sequences=True, name='lstm_1')(lstm_input)
        lstm_out = LSTM(32, return_sequences=False, name='lstm_2')(lstm_out)
        lstm_output = Dense(self.input_dim, activation='linear', name='lstm_output')(lstm_out)
        
        self.lstm_model = Model(lstm_input, lstm_output, name='lstm')
        self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        logger.info("LSTM model built successfully")
    
    def train(self, X, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train both autoencoder and LSTM models
        
        Args:
            X: Training data of shape (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        try:
            logger.info(f"Training AutoencoderLSTMAgent on {len(X)} samples...")
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)
            
            # Build and train autoencoder
            self.build_autoencoder()
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            self.autoencoder.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Build and train LSTM if we have enough data
            if len(X_scaled) >= self.sequence_length + 10:
                self.build_lstm()
                X_sequences = create_sequences(X_scaled, self.sequence_length)
                y_targets = X_scaled[self.sequence_length:]
                
                self.lstm_model.fit(
                    X_sequences, y_targets,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=[early_stop],
                    verbose=0
                )
            else:
                logger.warning(f"Not enough data for LSTM training (need at least {self.sequence_length + 10} samples)")
            
            # Calculate threshold from reconstruction errors
            reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
            reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
            self.threshold = np.percentile(reconstruction_errors, 95)
            
            self.is_trained = True
            logger.info(f"AutoencoderLSTMAgent trained successfully. Threshold: {self.threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Error training AutoencoderLSTMAgent: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Predict anomalies using reconstruction error
        
        Args:
            X: Data to predict on, shape (n_samples, n_features)
            
        Returns:
            dict with reconstruction_error, anomaly_score, is_anomaly
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # Get reconstruction error from autoencoder
            reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
            reconstruction_error = np.mean(np.square(X_scaled - reconstructed), axis=1)
            
            # Calculate anomaly score (normalized by threshold)
            anomaly_score = reconstruction_error / (self.threshold + 1e-10)
            
            # LSTM prediction (if available and enough data)
            lstm_contribution = np.zeros_like(anomaly_score)
            if self.lstm_model is not None and len(X_scaled) >= self.sequence_length:
                try:
                    X_sequences = create_sequences(X_scaled, self.sequence_length)
                    lstm_pred = self.lstm_model.predict(X_sequences, verbose=0)
                    lstm_error = np.mean(np.square(X_scaled[self.sequence_length:] - lstm_pred), axis=1)
                    # Pad lstm_contribution to match length
                    lstm_contribution = np.concatenate([
                        np.zeros(self.sequence_length),
                        lstm_error / (self.threshold + 1e-10)
                    ])
                except Exception as e:
                    logger.warning(f"LSTM prediction failed: {str(e)}")
            
            # Combined score (autoencoder 70%, LSTM 30%)
            combined_score = 0.7 * anomaly_score + 0.3 * lstm_contribution
            
            return {
                'reconstruction_error': reconstruction_error,
                'anomaly_score': combined_score,
                'is_anomaly': combined_score > 1.0
            }
            
        except Exception as e:
            logger.error(f"Error in AutoencoderLSTMAgent prediction: {str(e)}")
            raise


class CNNAgent:
    """
    Agent 2: 1D CNN for spatial-temporal pattern detection
    Advanced model for detecting complex anomaly patterns
    """
    
    def __init__(self, input_shape, sequence_length=10):
        self.input_shape = input_shape  # (n_features,)
        self.sequence_length = sequence_length
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info(f"Initialized CNNAgent: input_shape={input_shape}, seq_len={sequence_length}")
    
    def build_model(self):
        """Build 1D CNN model"""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for CNNAgent")
        
        # Input shape: (sequence_length, n_features)
        cnn_input = Input(shape=(self.sequence_length, self.input_shape), name='cnn_input')
        
        # 1D Convolutional layers
        conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1')(cnn_input)
        pool1 = MaxPooling1D(pool_size=2, name='pool1')(conv1)
        
        conv2 = Conv1D(32, kernel_size=3, activation='relu', padding='same', name='conv2')(pool1)
        pool2 = MaxPooling1D(pool_size=2, name='pool2')(conv2)
        
        # Flatten and dense layers
        flatten = Flatten(name='flatten')(pool2)
        dense1 = Dense(64, activation='relu', name='dense1')(flatten)
        dropout = Dropout(0.3, name='dropout')(dense1)
        dense2 = Dense(32, activation='relu', name='dense2')(dropout)
        
        # Output: anomaly score
        output = Dense(1, activation='sigmoid', name='output')(dense2)
        
        self.model = Model(cnn_input, output, name='cnn_agent')
        self.model.compile(optimizer=Adam(learning_rate=0.001), 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])
        
        logger.info("CNN model built successfully")
    
    def _generate_pseudo_labels(self, X):
        """Generate pseudo-labels using IsolationForest"""
        # Flatten sequences for IsolationForest
        X_flat = X.reshape(X.shape[0], -1)
        iso = IsolationForest(contamination=0.1, random_state=42)
        labels = iso.fit_predict(X_flat)
        return (labels == -1).astype(np.float32)  # Convert to 0/1
    
    def train(self, X, y=None, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train CNN model
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Optional labels (if None, will generate pseudo-labels)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        try:
            logger.info(f"Training CNNAgent on {len(X)} samples...")
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)
            
            # Create sequences
            if len(X_scaled) < self.sequence_length + 10:
                raise ValueError(f"Need at least {self.sequence_length + 10} samples for CNN training")
            
            X_sequences = create_sequences(X_scaled, self.sequence_length)
            
            # Generate or use provided labels
            if y is None:
                logger.info("Generating pseudo-labels using IsolationForest...")
                y = self._generate_pseudo_labels(X_sequences)
            else:
                y = y[self.sequence_length:]  # Align with sequences
            
            # Build and train model
            self.build_model()
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            self.model.fit(
                X_sequences, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Set threshold
            predictions = self.model.predict(X_sequences, verbose=0).flatten()
            self.threshold = np.percentile(predictions, 95)
            
            self.is_trained = True
            logger.info(f"CNNAgent trained successfully. Threshold: {self.threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Error training CNNAgent: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Predict anomaly scores using CNN
        
        Args:
            X: Data to predict on, shape (n_samples, n_features)
            
        Returns:
            dict with anomaly_score and is_anomaly
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # Create sequences
            if len(X_scaled) < self.sequence_length:
                # Not enough data, return zeros
                return {
                    'anomaly_score': np.zeros(len(X)),
                    'is_anomaly': np.zeros(len(X), dtype=bool)
                }
            
            X_sequences = create_sequences(X_scaled, self.sequence_length)
            anomaly_scores_seq = self.model.predict(X_sequences, verbose=0).flatten()
            
            # Pad scores to match input length
            anomaly_scores = np.concatenate([
                np.zeros(self.sequence_length),
                anomaly_scores_seq
            ])
            
            return {
                'anomaly_score': anomaly_scores,
                'is_anomaly': anomaly_scores > self.threshold
            }
            
        except Exception as e:
            logger.error(f"Error in CNNAgent prediction: {str(e)}")
            raise


class TrendAnalysisModel:
    """
    Trend Analysis using ARIMA/SARIMA for time series forecasting
    Detects anomalies based on deviation from predicted trends
    """
    
    def __init__(self, model_type='arima'):
        self.model_type = model_type
        self.model = None
        self.history = []
        self.is_trained = False
        
        logger.info(f"Initialized TrendAnalysisModel: model_type={model_type}")
    
    def fit(self, time_series_data, seasonal_period=None):
        """
        Fit trend model on historical data
        
        Args:
            time_series_data: 1D array of time series values
            seasonal_period: Period for seasonal ARIMA (optional)
        """
        try:
            logger.info(f"Training {self.model_type.upper()} model on {len(time_series_data)} samples...")
            
            if not STATSMODELS_ARIMA_AVAILABLE:
                logger.warning("statsmodels not available, using simple moving average")
                self.model = 'simple_ma'
                self.history = list(time_series_data)
                self.is_trained = True
                return
            
            if self.model_type == 'arima':
                if PMDARIMA_AVAILABLE:
                    # Auto-detect best parameters
                    self.model = auto_arima(
                        time_series_data,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore',
                        max_p=3, max_q=3, max_d=2
                    )
                else:
                    # Use manual parameters
                    self.model = ARIMA(time_series_data, order=(1, 1, 1))
                    self.model = self.model.fit()
                    
            elif self.model_type == 'sarima' and seasonal_period:
                self.model = SARIMAX(
                    time_series_data,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, seasonal_period)
                )
                self.model = self.model.fit(disp=False)
            else:
                # Fallback to simple ARIMA
                self.model = ARIMA(time_series_data, order=(1, 1, 1))
                self.model = self.model.fit()
            
            self.history = list(time_series_data)
            self.is_trained = True
            logger.info(f"{self.model_type.upper()} model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training TrendAnalysisModel: {str(e)}")
            # Fallback to simple moving average
            self.model = 'simple_ma'
            self.history = list(time_series_data)
            self.is_trained = True
            logger.warning("Using simple moving average as fallback")
    
    def predict(self, steps_ahead=24):
        """
        Predict future values
        
        Args:
            steps_ahead: Number of steps to forecast
            
        Returns:
            numpy array of forecasted values
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        try:
            if self.model == 'simple_ma':
                # Simple moving average forecast
                window = min(10, len(self.history))
                forecast_value = np.mean(self.history[-window:])
                return np.full(steps_ahead, forecast_value)
            else:
                # ARIMA/SARIMA forecast
                forecast = self.model.forecast(steps=steps_ahead)
                return np.array(forecast)
                
        except Exception as e:
            logger.error(f"Error in TrendAnalysisModel prediction: {str(e)}")
            # Fallback: return last value
            return np.full(steps_ahead, self.history[-1])
    
    def detect_trend_anomalies(self, current_value, predicted_value, confidence_level=0.1):
        """
        Detect if current value deviates from predicted trend
        
        Args:
            current_value: Current observed value
            predicted_value: Predicted value from model
            confidence_level: Acceptable deviation (as fraction)
            
        Returns:
            tuple: (is_anomaly: bool, deviation: float)
        """
        if predicted_value == 0:
            predicted_value = 1e-10  # Avoid division by zero
        
        deviation = abs(current_value - predicted_value) / abs(predicted_value)
        is_anomaly = deviation > confidence_level
        
        return is_anomaly, deviation


class MultiAgentFusionSystem:
    """
    Multi-Agent Fusion System combining:
    - Agent 1: Autoencoder + LSTM (reconstruction error)
    - Agent 2: CNN (spatial-temporal patterns)
    - Trend Model: ARIMA (trend prediction)
    """
    
    def __init__(self, input_dim, sequence_length=10):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Initialize agents
        self.agent1 = None  # Will be created if TensorFlow available
        self.agent2 = None  # Will be created if TensorFlow available
        self.trend_model = TrendAnalysisModel(model_type='arima')
        
        # Fusion weights (can be adjusted)
        self.weights = {
            'agent1': 0.4,  # Autoencoder+LSTM weight
            'agent2': 0.4,  # CNN weight
            'trend': 0.2    # Trend model weight
        }
        
        self.is_trained = False
        
        logger.info(f"Initialized MultiAgentFusionSystem with weights: {self.weights}")
    
    def train(self, X, epochs=50, batch_size=32):
        """
        Train all three components
        
        Args:
            X: Training data of shape (n_samples, n_features)
            epochs: Number of training epochs for deep learning models
            batch_size: Batch size for training
        """
        try:
            logger.info("Starting training for MultiAgentFusionSystem")
            
            # Train Agent 1 (Autoencoder + LSTM)
            if TENSORFLOW_AVAILABLE:
                logger.info("[1/3] Training Agent 1 (Autoencoder+LSTM)...")
                self.agent1 = AutoencoderLSTMAgent(self.input_dim, self.sequence_length)
                self.agent1.train(X, epochs=epochs, batch_size=batch_size)
                logger.info("[OK] Agent 1 trained successfully")
            else:
                logger.warning("[SKIP] Agent 1 skipped (TensorFlow not available)")
            
            # Train Agent 2 (CNN)
            if TENSORFLOW_AVAILABLE and len(X) >= self.sequence_length + 10:
                logger.info("[2/3] Training Agent 2 (CNN)...")
                self.agent2 = CNNAgent(self.input_dim, self.sequence_length)
                self.agent2.train(X, epochs=epochs, batch_size=batch_size)
                logger.info("[OK] Agent 2 trained successfully")
            else:
                if not TENSORFLOW_AVAILABLE:
                    logger.warning("[SKIP] Agent 2 skipped (TensorFlow not available)")
                else:
                    logger.warning(f"[SKIP] Agent 2 skipped (need at least {self.sequence_length + 10} samples)")
            
            # Train Trend Model
            logger.info("[3/3] Training Trend Model (ARIMA)...")
            # Use mean of features for univariate trend analysis
            time_series = np.mean(X, axis=1) if X.ndim > 1 else X
            self.trend_model.fit(time_series)
            logger.info("[OK] Trend Model trained successfully")
            
            self.is_trained = True
            logger.info("MultiAgentFusionSystem training complete")
            
        except Exception as e:
            logger.error(f"Error training MultiAgentFusionSystem: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict(self, X, return_details=False):
        """
        Get ensemble prediction from all agents
        
        Args:
            X: Data to predict on, shape (n_samples, n_features)
            return_details: If True, return detailed results from each agent
            
        Returns:
            If return_details=False: (is_anomaly: bool, final_score: float)
            If return_details=True: dict with detailed results
        """
        if not self.is_trained:
            raise RuntimeError("System must be trained before prediction")
        
        try:
            results = {}
            scores = []
            weights_used = []
            
            # Agent 1 prediction
            if self.agent1 is not None:
                agent1_result = self.agent1.predict(X)
                results['agent1'] = agent1_result
                scores.append(np.mean(agent1_result['anomaly_score']))
                weights_used.append(self.weights['agent1'])
            else:
                results['agent1'] = {'anomaly_score': np.zeros(len(X)), 'is_anomaly': np.zeros(len(X), dtype=bool), 'reconstruction_error': np.zeros(len(X))}
                scores.append(0)
                weights_used.append(0)
            
            # Agent 2 prediction
            if self.agent2 is not None:
                agent2_result = self.agent2.predict(X)
                results['agent2'] = agent2_result
                scores.append(np.mean(agent2_result['anomaly_score']))
                weights_used.append(self.weights['agent2'])
            else:
                results['agent2'] = {'anomaly_score': np.zeros(len(X)), 'is_anomaly': np.zeros(len(X), dtype=bool)}
                scores.append(0)
                weights_used.append(0)
            
            # Trend prediction
            current_mean = np.mean(X, axis=1) if X.ndim > 1 else X
            trend_forecast = self.trend_model.predict(steps_ahead=5)
            trend_anomaly, trend_deviation = self.trend_model.detect_trend_anomalies(
                current_mean[-1], 
                trend_forecast[0],
                confidence_level=0.15
            )
            
            results['trend'] = {
                'forecast': trend_forecast,
                'is_anomaly': trend_anomaly,
                'deviation': trend_deviation,
                'current_value': current_mean[-1]
            }
            scores.append(trend_deviation if trend_anomaly else 0)
            weights_used.append(self.weights['trend'])
            
            # Normalize weights if some agents are missing
            total_weight = sum(weights_used)
            if total_weight > 0:
                weights_used = [w / total_weight for w in weights_used]
            
            # Fusion: Weighted combination
            final_score = sum(s * w for s, w in zip(scores, weights_used))
            
            # Final decision (threshold can be adjusted)
            is_anomaly = final_score > 0.6
            
            results['final_score'] = final_score
            results['is_anomaly'] = is_anomaly
            results['weights_used'] = dict(zip(['agent1', 'agent2', 'trend'], weights_used))
            
            if return_details:
                return results
            else:
                return is_anomaly, final_score
                
        except Exception as e:
            logger.error(f"Error in MultiAgentFusionSystem prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def update_weights(self, agent1_weight=None, agent2_weight=None, trend_weight=None):
        """Update fusion weights"""
        if agent1_weight is not None:
            self.weights['agent1'] = agent1_weight
        if agent2_weight is not None:
            self.weights['agent2'] = agent2_weight
        if trend_weight is not None:
            self.weights['trend'] = trend_weight
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        logger.info(f"Updated fusion weights: {self.weights}")

