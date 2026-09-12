"""Production anomaly detectors extracted from main.py (Slice C Wave 1).

Separate from app.models.anomaly_engine.AnomalyModel (modular/dead MainWindow path).
"""
from __future__ import annotations

import logging
import os
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import SGDOneClassSVM, PassiveAggressiveClassifier

logger = logging.getLogger("SecureAnomalyDetection")

import datetime
import pickle
import time

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from app.models.model_types import (
    MODEL_DISPLAY_TO_INTERNAL,
    MODEL_INTERNAL_FALLBACK,
    TOUCHED_44_MODEL_NAMES,
    get_supported_model_names,
    resolve_runtime_model_type,
    to_internal_model_type,
)
from app.security.pickle_safe import SecurityError, safe_pickle_load

# For deep learning models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model # type: ignore
    from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, RepeatVector, TimeDistributed, GRU # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping # type: ignore
    from tensorflow.keras.optimizers import Adam# type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Deep learning models will be disabled.")

# For Prophet models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Prophet models will be disabled.")

# For online learning and reinforcement learning
try:
    from sklearn.linear_model import SGDOneClassSVM, PassiveAggressiveClassifier
    from sklearn.ensemble import ExtraTreesClassifier  
    from sklearn.tree import ExtraTreeClassifier
    ONLINE_LEARNING_AVAILABLE = True
except ImportError:
    ONLINE_LEARNING_AVAILABLE = False
    warnings.warn("Online learning models not available.")

# For River (online machine learning library)
try:
    from river import anomaly, compose, preprocessing, metrics
    from river.tree import HoeffdingTreeClassifier
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    warnings.warn("River online learning library not available. Install with: pip install river")




_ENHANCED_MODEL_TYPES = frozenset(
    {
        "enhanced_isolation_forest",
        "ensemble_voting",
        "ensemble_stacking",
        "adaptive_threshold",
    }
)


class EnhancedAnomalyDetectionModel:
    """Enhanced anomaly detection model with improved performance and ensemble methods"""
    
    def __init__(self, model_type="enhanced_isolation_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_columns = None
        self.sequence_length = 10
        self.reconstruction_error_threshold = None
        self.history = None
        self.metrics = {}
        self.X_train = None  # Store training data for metrics
        self.ensemble_models = []  # For ensemble methods
        self.preprocessing_pipeline = None
        self.model_params = {}  # Store best parameters

    def _runtime_model_type(self):
        requested = getattr(self, "model_type", "enhanced_isolation_forest")
        return resolve_runtime_model_type(to_internal_model_type(requested))
        
    def create_preprocessing_pipeline(self, preprocessing_type="standard"):
        """Create an advanced preprocessing pipeline"""
        try:
            if preprocessing_type == "standard":
                pipeline_steps = [
                    ('scaler', StandardScaler()),
                    ('feature_selector', VarianceThreshold(threshold=0.0)),
                ]
            elif preprocessing_type == "robust":
                pipeline_steps = [
                    ('power_transformer', PowerTransformer(method='yeo-johnson')),
                    ('robust_scaler', RobustScaler()),
                    ('feature_selector', VarianceThreshold(threshold=0.0)),
                ]
            elif preprocessing_type == "pca":
                pipeline_steps = [
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=0.95))
                ]
            else:
                pipeline_steps = [('scaler', StandardScaler())]
            
            self.preprocessing_pipeline = Pipeline(pipeline_steps)
            return True, "Preprocessing pipeline created successfully"
            
        except Exception as e:
            logger.error(f"Error creating preprocessing pipeline: {str(e)}")
            return False, f"Failed to create pipeline: {str(e)}"
    
    def hyperparameter_tuning(self, X, model_type, cv_folds=3):
        """Perform hyperparameter tuning for the model"""
        try:
            if model_type == "enhanced_isolation_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'contamination': [0.05, 0.1, 0.15, 0.2],
                    'max_features': [0.5, 0.7, 1.0],
                    'bootstrap': [True, False]
                }
                base_model = IsolationForest(random_state=42)
                
            elif model_type == "one_class_svm":
                param_grid = {
                    'nu': [0.01, 0.05, 0.1, 0.2],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
                base_model = OneClassSVM()
                
            elif model_type == "elliptic_envelope":
                param_grid = {
                    'contamination': [0.05, 0.1, 0.15, 0.2],
                    'support_fraction': [None, 0.5, 0.7, 0.9]
                }
                base_model = EllipticEnvelope(random_state=42)
                
            else:
                return None, f"Hyperparameter tuning not implemented for {model_type}"
            
            # Use TimeSeriesSplit for time series data or regular CV for other data
            cv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Custom scoring function for unsupervised anomaly detection
            def anomaly_score(estimator, X):
                try:
                    if hasattr(estimator, 'decision_function'):
                        scores = estimator.decision_function(X)
                        return np.mean(scores)  # Higher is better for normal points
                    elif hasattr(estimator, 'score_samples'):
                        scores = estimator.score_samples(X)
                        return np.mean(scores)  # Higher is better
                    else:
                        return 0
                except:
                    return 0
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring=anomaly_score,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X)
            
            self.model_params = grid_search.best_params_
            return grid_search.best_estimator_, f"Best parameters found: {self.model_params}"
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}")
            return None, f"Tuning failed: {str(e)}"
    
    def create_ensemble_model(self, X, ensemble_type="voting"):
        """Create ensemble of anomaly detection models"""
        try:
            if ensemble_type == "voting":
                # Create multiple diverse models
                models = [
                    ('isolation_forest', IsolationForest(
                        n_estimators=100, contamination=0.1, random_state=42)),
                    ('one_class_svm', OneClassSVM(nu=0.1, kernel='rbf')),
                    ('elliptic_envelope', EllipticEnvelope(contamination=0.1, random_state=42)),
                    ('lof', LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True))
                ]
                
                # Train all models
                trained_models = []
                for name, model in models:
                    try:
                        model.fit(X)
                        trained_models.append((name, model))
                    except Exception as e:
                        logger.warning(f"Failed to train {name}: {str(e)}")
                        continue
                
                self.ensemble_models = trained_models
                return True, f"Ensemble created with {len(trained_models)} models"
                
            elif ensemble_type == "stacking":
                # Implement stacking ensemble
                base_models = [
                    IsolationForest(n_estimators=100, contamination=0.1, random_state=42),
                    OneClassSVM(nu=0.1, kernel='rbf'),
                    EllipticEnvelope(contamination=0.1, random_state=42)
                ]
                
                # Train base models and collect predictions
                base_predictions = []
                for model in base_models:
                    try:
                        model.fit(X)
                        if hasattr(model, 'decision_function'):
                            pred = model.decision_function(X)
                        else:
                            pred = model.predict(X)
                        base_predictions.append(pred)
                    except Exception as e:
                        logger.warning(f"Base model training failed: {str(e)}")
                        continue
                
                if base_predictions:
                    # Stack predictions
                    stacked_features = np.column_stack(base_predictions)
                    
                    # Train meta-learner (using Isolation Forest as meta-learner)
                    meta_learner = IsolationForest(contamination=0.1, random_state=42)
                    meta_learner.fit(stacked_features)
                    
                    self.ensemble_models = base_models
                    self.meta_learner = meta_learner
                    return True, "Stacking ensemble created successfully"
                
                return False, "No base models trained successfully for stacking"
            
            else:
                return False, f"Unsupported ensemble type: {ensemble_type}"
                
        except Exception as e:
            logger.error(f"Ensemble creation failed: {str(e)}")
            return False, f"Ensemble failed: {str(e)}"
    
    def train_enhanced_model(self, data, **kwargs):
        """Train enhanced anomaly detection models with better performance"""
        try:
            # Data validation and preparation
            if not isinstance(data, pd.DataFrame):
                return False, "Training data must be a pandas DataFrame"
                
            numeric_data = data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return False, "No numeric columns found in the training data"
            
            self.feature_columns = list(numeric_data.columns)
            
            # Create and fit preprocessing pipeline
            preprocessing_type = kwargs.get('preprocessing_type', 'standard')
            success, message = self.create_preprocessing_pipeline(preprocessing_type)
            if not success:
                return False, message
            
            # Apply preprocessing
            X = self.preprocessing_pipeline.fit_transform(numeric_data)
            self.X_train = X  # Store for metrics calculation
            
            # Model-specific training
            if self.model_type == "enhanced_isolation_forest":
                # Use hyperparameter tuning
                use_tuning = kwargs.get('use_hyperparameter_tuning', True)
                if use_tuning:
                    tuned_model, tune_message = self.hyperparameter_tuning(X, self.model_type)
                    if tuned_model is not None:
                        self.model = tuned_model
                        return True, f"Enhanced Isolation Forest trained with tuning: {tune_message}"
                
                # Fallback to default enhanced parameters
                self.model = IsolationForest(
                    n_estimators=kwargs.get('n_estimators', 200),
                    contamination=kwargs.get('contamination', 0.1),
                    max_features=kwargs.get('max_features', 0.8),
                    bootstrap=kwargs.get('bootstrap', True),
                    random_state=42
                )
                self.model.fit(X)
                return True, "Enhanced Isolation Forest trained successfully"
                
            elif self.model_type == "ensemble_voting":
                success, message = self.create_ensemble_model(X, "voting")
                return success, message
                
            elif self.model_type == "ensemble_stacking":
                success, message = self.create_ensemble_model(X, "stacking")
                return success, message
                
            elif self.model_type == "adaptive_threshold":
                # Adaptive threshold model using multiple methods
                models = {
                    'isolation_forest': IsolationForest(n_estimators=100, contamination=0.1, random_state=42),
                    'elliptic_envelope': EllipticEnvelope(contamination=0.1, random_state=42),
                    'one_class_svm': OneClassSVM(nu=0.1, kernel='rbf')
                }
                
                trained_models = {}
                thresholds = {}
                
                for name, model in models.items():
                    try:
                        model.fit(X)
                        if hasattr(model, 'decision_function'):
                            scores = model.decision_function(X)
                            # Adaptive threshold based on data distribution
                            threshold = np.percentile(scores, kwargs.get('threshold_percentile', 10))
                            thresholds[name] = threshold
                        trained_models[name] = model
                    except Exception as e:
                        logger.warning(f"Failed to train {name}: {str(e)}")
                
                self.ensemble_models = trained_models
                self.adaptive_thresholds = thresholds
                return True, f"Adaptive threshold model trained with {len(trained_models)} base models"
                
            else:
                return False, f"Unsupported enhanced model type: {self.model_type}"
                
        except Exception as e:
            logger.error(f"Enhanced training error: {str(e)}")
            return False, f"Training failed: {str(e)}"

    def train(self, data, **kwargs):
        """Backward-compatible train entrypoint used by UI/worker code."""
        return self.train_enhanced_model(data, **kwargs)
    
    def predict_enhanced(self, data):
        """Enhanced prediction with ensemble voting and confidence scores"""
        try:
            if not self.preprocessing_pipeline:
                return None, "Model not trained or preprocessing pipeline missing"
            
            # Prepare data using the same preprocessing pipeline
            numeric_data = data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return None, "No numeric columns found in prediction data"
            
            # Apply preprocessing
            X = self.preprocessing_pipeline.transform(numeric_data)
            
            if self.model_type == "enhanced_isolation_forest":
                if self.model is None:
                    return None, "Model not trained"
                
                scores = self.model.decision_function(X)
                predictions = self.model.predict(X)
                anomalies = (predictions == -1)
                
                # Calculate confidence scores
                confidence_scores = np.abs(scores) / (np.abs(scores).max() + 1e-8)
                
                return {
                    'anomalies': anomalies,
                    'scores': scores,
                    'confidence': confidence_scores,
                    'method': 'enhanced_isolation_forest'
                }, "Enhanced Isolation Forest prediction completed"
                
            elif self.model_type in ["ensemble_voting", "ensemble_stacking"]:
                if not self.ensemble_models:
                    return None, "Ensemble models not trained"
                
                all_predictions = []
                all_scores = []
                
                # Get predictions from all ensemble models
                for name, model in self.ensemble_models:
                    try:
                        if hasattr(model, 'decision_function'):
                            scores = model.decision_function(X)
                            predictions = model.predict(X)
                        elif hasattr(model, 'predict'):
                            predictions = model.predict(X)
                            scores = np.ones_like(predictions)  # Default scores
                        else:
                            continue
                        
                        all_predictions.append(predictions == -1)  # Convert to boolean
                        all_scores.append(scores)
                        
                    except Exception as e:
                        logger.warning(f"Prediction failed for {name}: {str(e)}")
                        continue
                
                if not all_predictions:
                    return None, "No ensemble models produced valid predictions"
                
                # Majority voting for final prediction
                prediction_matrix = np.array(all_predictions)
                ensemble_anomalies = np.mean(prediction_matrix, axis=0) > 0.5
                
                # Average confidence scores
                if all_scores:
                    average_scores = np.mean(all_scores, axis=0)
                    confidence_scores = np.abs(average_scores) / (np.abs(average_scores).max() + 1e-8)
                else:
                    confidence_scores = np.ones(len(ensemble_anomalies)) * 0.5
                
                return {
                    'anomalies': ensemble_anomalies,
                    'scores': average_scores if all_scores else np.zeros(len(ensemble_anomalies)),
                    'confidence': confidence_scores,
                    'individual_predictions': all_predictions,
                    'method': self.model_type
                }, f"Ensemble prediction completed with {len(all_predictions)} models"
                
            elif self.model_type == "adaptive_threshold":
                if not self.ensemble_models or not hasattr(self, 'adaptive_thresholds'):
                    return None, "Adaptive threshold models not trained"
                
                adaptive_predictions = []
                adaptive_scores = []
                
                for name, model in self.ensemble_models.items():
                    try:
                        if hasattr(model, 'decision_function'):
                            scores = model.decision_function(X)
                            threshold = self.adaptive_thresholds.get(name, 0)
                            predictions = scores < threshold  # Below threshold = anomaly
                        else:
                            predictions = model.predict(X) == -1
                            scores = np.ones_like(predictions, dtype=float)
                        
                        adaptive_predictions.append(predictions)
                        adaptive_scores.append(scores)
                        
                    except Exception as e:
                        logger.warning(f"Adaptive prediction failed for {name}: {str(e)}")
                        continue
                
                if not adaptive_predictions:
                    return None, "No adaptive models produced valid predictions"
                
                # Weighted voting based on model performance
                prediction_matrix = np.array(adaptive_predictions)
                final_anomalies = np.mean(prediction_matrix, axis=0) > 0.5
                
                # Calculate confidence as consistency across models
                confidence_scores = 1.0 - np.std(prediction_matrix.astype(float), axis=0)
                
                return {
                    'anomalies': final_anomalies,
                    'scores': np.mean(adaptive_scores, axis=0) if adaptive_scores else np.zeros(len(final_anomalies)),
                    'confidence': confidence_scores,
                    'method': 'adaptive_threshold'
                }, "Adaptive threshold prediction completed"
                
            else:
                return None, f"Prediction not implemented for {self.model_type}"
                
        except Exception as e:
            logger.error(f"Enhanced prediction error: {str(e)}")
            return None, f"Prediction failed: {str(e)}"

    def predict(self, data):
        """Return legacy (scores, anomalies) tuple for existing callers."""
        prediction_result, message = self.predict_enhanced(data)
        if prediction_result is None:
            return None, message

        if isinstance(prediction_result, dict):
            scores = prediction_result.get("scores")
            anomalies = prediction_result.get("anomalies")
            if scores is not None and anomalies is not None:
                return scores, anomalies

        return None, "Enhanced prediction output format is invalid"
    
    def save(self, filepath):
        """Save enhanced model to file"""
        if self.model is None and not self.ensemble_models:
            return False, "No model to save"
            
        try:
            active_model_type = getattr(self, "resolved_model_type", self._runtime_model_type())
            # Ensure filepath has .pkl extension
            if not filepath.endswith('.pkl'):
                filepath = filepath + '.pkl'
                
            # Create directory if needed
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Prepare enhanced model data
            model_data = {
                "model_type": self.model_type,
                "model": self.model,
                "preprocessing_pipeline": getattr(self, 'preprocessing_pipeline', None),
                "ensemble_models": getattr(self, 'ensemble_models', []),
                "adaptive_thresholds": getattr(self, 'adaptive_thresholds', {}),
                "model_params": getattr(self, 'model_params', {}),
                "feature_columns": getattr(self, 'feature_columns', None),
                "X_train": getattr(self, 'X_train', None),
                "metrics": getattr(self, 'metrics', {})
            }
            
            # Add meta-learner if it exists (for stacking ensemble)
            if hasattr(self, 'meta_learner'):
                model_data["meta_learner"] = self.meta_learner
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            return True, f"Enhanced model saved to {filepath}"
            
        except Exception as e:
            logger.error(f"Enhanced model save error: {str(e)}")
            return False, f"Error saving enhanced model: {str(e)}"
    
    @staticmethod
    def load(filepath):
        """Load enhanced model from file"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return None, f"Model file not found: {filepath}"
                
            model_data = safe_pickle_load(filepath)
            
            if not isinstance(model_data, dict) or "model_type" not in model_data:
                return None, "Invalid enhanced model file format"

            saved_type = str(model_data.get("model_type") or "").strip().lower()
            if saved_type not in _ENHANCED_MODEL_TYPES:
                return None, f"Not an enhanced model payload: {saved_type or '?'}"
            
            # Create enhanced model instance
            model = EnhancedAnomalyDetectionModel(model_data["model_type"])
            
            # Load all attributes
            model.model = model_data.get("model")
            model.preprocessing_pipeline = model_data.get("preprocessing_pipeline")
            model.ensemble_models = model_data.get("ensemble_models", [])
            model.adaptive_thresholds = model_data.get("adaptive_thresholds", {})
            model.model_params = model_data.get("model_params", {})
            model.feature_columns = model_data.get("feature_columns")
            model.X_train = model_data.get("X_train")
            model.metrics = model_data.get("metrics", {})
            
            # Load meta-learner if it exists
            if "meta_learner" in model_data:
                model.meta_learner = model_data["meta_learner"]
            
            return model, f"Enhanced model loaded successfully from {filepath}"
            
        except Exception as e:
            logger.error(f"Enhanced model load error: {str(e)}")
            return None, f"Error loading enhanced model: {str(e)}"

class OnlineLearningAnomalyDetector:
    """Online Learning Anomaly Detection with Reinforcement Learning capabilities"""
    
    def __init__(self, model_type="incremental_isolation_forest", learning_rate=0.01):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.online_scaler = None
        self.feature_columns = None
        self.window_size = 1000  # Sliding window for online learning
        self.data_buffer = []
        self.anomaly_buffer = []
        self.performance_metrics = {}
        self.adaptation_threshold = 0.1  # Threshold for model adaptation
        self.feedback_history = []  # Store feedback for RL
        self.reward_history = []
        self.state_history = []
        self.action_history = []
        
        # RL components
        self.q_table = {}  # Q-learning table for simple RL
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.9  # Discount factor
        self.alpha = 0.1  # Learning rate for Q-learning
        
        # Performance tracking
        self.model_performance = []
        self.concept_drift_detector = None
        
    def initialize_online_model(self, initial_data=None):
        """Initialize the online learning model"""
        try:
            if self.model_type == "incremental_isolation_forest":
                # Use a combination of incremental models
                self.model = {
                    'sgd_ocsvm': SGDOneClassSVM(learning_rate='constant', eta0=self.learning_rate),
                    'passive_aggressive': PassiveAggressiveClassifier(random_state=42),
                    'extra_tree': ExtraTreeClassifier(random_state=42)
                }
                
            elif self.model_type == "river_anomaly" and RIVER_AVAILABLE:
                # Use River online learning library
                self.model = compose.Pipeline(
                    preprocessing.StandardScaler(),
                    anomaly.HalfSpaceTrees(n_trees=10, height=8)
                )
                
            elif self.model_type == "reinforcement_learning":
                # Custom RL-based anomaly detector
                self.model = None  # Will be initialized dynamically
                self.initialize_rl_components()
                
            else:
                return False, f"Unsupported online model type: {self.model_type}"
            
            # Initialize online scaling
            self.online_scaler = preprocessing.StandardScaler() if RIVER_AVAILABLE else StandardScaler()
            
            # Initialize with initial data if provided
            if initial_data is not None:
                self.warm_start(initial_data)
                
            return True, f"Online learning model '{self.model_type}' initialized successfully"
            
        except Exception as e:
            logger.error(f"Online model initialization failed: {str(e)}")
            return False, f"Initialization failed: {str(e)}"
    
    def initialize_rl_components(self):
        """Initialize reinforcement learning components"""
        # Define states (simplified): normal, suspicious, anomalous
        self.states = ['normal', 'suspicious', 'anomalous']
        
        # Define actions: keep_threshold, increase_sensitivity, decrease_sensitivity, retrain
        self.actions = ['keep', 'increase_sens', 'decrease_sens', 'retrain']
        
        # Initialize Q-table
        for state in self.states:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        
        # Initialize base anomaly detection model
        self.base_model = IsolationForest(contamination=0.1, random_state=42)
        
    def warm_start(self, initial_data):
        """Warm start the model with initial data"""
        try:
            if not isinstance(initial_data, pd.DataFrame):
                return False, "Initial data must be a pandas DataFrame"
            
            numeric_data = initial_data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return False, "No numeric columns found"
                
            self.feature_columns = list(numeric_data.columns)
            
            if self.model_type == "incremental_isolation_forest":
                # Warm start incremental models
                X = StandardScaler().fit_transform(numeric_data)
                
                # Initialize base model for reference
                base_model = IsolationForest(contamination=0.1, random_state=42)
                base_model.fit(X)
                base_predictions = base_model.predict(X)
                
                # Train incremental models
                y = (base_predictions == -1).astype(int)  # Convert to binary labels
                
                if 'sgd_ocsvm' in self.model:
                    # For one-class SVM, we only use normal data
                    normal_data = X[y == 0]
                    if len(normal_data) > 0:
                        self.model['sgd_ocsvm'].fit(normal_data, np.ones(len(normal_data)))
                
                if 'passive_aggressive' in self.model:
                    self.model['passive_aggressive'].fit(X, y)
                
                if 'extra_tree' in self.model:
                    self.model['extra_tree'].fit(X, y)
                    
            elif self.model_type == "river_anomaly" and RIVER_AVAILABLE:
                # Warm start River model
                for _, row in numeric_data.iterrows():
                    x = {str(i): float(val) for i, val in enumerate(row)}
                    self.model.learn_one(x)
                    
            elif self.model_type == "reinforcement_learning":
                # Warm start RL model
                X = StandardScaler().fit_transform(numeric_data)
                self.base_model.fit(X)
                
                # Initialize states based on initial data distribution
                scores = self.base_model.decision_function(X)
                self.normal_threshold = np.percentile(scores, 10)
                self.suspicious_threshold = np.percentile(scores, 5)
                
            return True, "Warm start completed successfully"
            
        except Exception as e:
            logger.error(f"Warm start failed: {str(e)}")
            return False, f"Warm start failed: {str(e)}"
    
    def learn_online(self, new_data, feedback=None):
        """Learn from new data point(s) online"""
        try:
            if not isinstance(new_data, pd.DataFrame):
                if isinstance(new_data, (list, np.ndarray)):
                    new_data = pd.DataFrame([new_data], columns=self.feature_columns)
                else:
                    return False, "Invalid data format for online learning"
            
            numeric_data = new_data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return False, "No numeric columns found"
            
            # Update data buffer
            self.data_buffer.extend(numeric_data.values.tolist())
            if len(self.data_buffer) > self.window_size:
                self.data_buffer = self.data_buffer[-self.window_size:]
            
            # Learn based on model type
            if self.model_type == "incremental_isolation_forest":
                success, message = self._learn_incremental(numeric_data, feedback)
                
            elif self.model_type == "river_anomaly" and RIVER_AVAILABLE:
                success, message = self._learn_river(numeric_data, feedback)
                
            elif self.model_type == "reinforcement_learning":
                success, message = self._learn_reinforcement(numeric_data, feedback)
                
            else:
                return False, f"Online learning not implemented for {self.model_type}"
            
            # Check for concept drift and adapt if necessary
            self._check_concept_drift()
            
            return success, message
            
        except Exception as e:
            logger.error(f"Online learning failed: {str(e)}")
            return False, f"Online learning failed: {str(e)}"
    
    def _learn_incremental(self, new_data, feedback=None):
        """Incremental learning for sklearn-based models"""
        try:
            X = new_data.values
            
            # Make predictions first
            predictions = self.predict_online(new_data)
            if predictions is None:
                return False, "Failed to get predictions for learning"
            
            # Update models based on feedback or self-supervision
            if feedback is not None:
                # Supervised learning with feedback
                y = np.array(feedback).astype(int)
                
                if 'passive_aggressive' in self.model:
                    self.model['passive_aggressive'].partial_fit(X, y)
                    
            else:
                # Self-supervised learning - use ensemble voting
                all_preds = []
                
                # Get predictions from all available models
                for name, model in self.model.items():
                    try:
                        if name == 'sgd_ocsvm':
                            pred = model.predict(X)
                            all_preds.append((pred == 1))  # Convert to boolean
                        elif name == 'passive_aggressive':
                            if hasattr(model, 'predict'):
                                pred = model.predict(X)
                                all_preds.append((pred == 1))
                        elif name == 'extra_tree':
                            if hasattr(model, 'predict'):
                                pred = model.predict(X)
                                all_preds.append((pred == 1))
                    except:
                        continue
                
                # Use majority voting as pseudo-labels
                if all_preds:
                    ensemble_pred = np.mean(all_preds, axis=0) > 0.5
                    y = ensemble_pred.astype(int)
                    
                    # Update models that support partial_fit
                    if 'passive_aggressive' in self.model:
                        try:
                            self.model['passive_aggressive'].partial_fit(X, y)
                        except:
                            pass
            
            return True, "Incremental learning completed"
            
        except Exception as e:
            return False, f"Incremental learning failed: {str(e)}"
    
    def _learn_river(self, new_data, feedback=None):
        """Online learning using River library"""
        try:
            if not RIVER_AVAILABLE:
                return False, "River library not available"
            
            for _, row in new_data.iterrows():
                x = {str(i): float(val) for i, val in enumerate(row)}
                
                # Learn from the new data point
                self.model.learn_one(x)
                
                # If feedback is provided, we can use it for supervised learning
                if feedback is not None:
                    # River models are typically unsupervised, but we can track performance
                    pass
            
            return True, "River online learning completed"
            
        except Exception as e:
            return False, f"River learning failed: {str(e)}"
    
    def _learn_reinforcement(self, new_data, feedback=None):
        """Reinforcement learning-based adaptation"""
        try:
            X = new_data.values
            
            # Get current state
            current_state = self._get_current_state(X)
            
            # Get current performance (reward)
            reward = self._calculate_reward(X, feedback)
            
            # Update Q-learning if we have previous state-action pair
            if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
                self._update_q_table(self.last_state, self.last_action, reward, current_state)
            
            # Choose action using epsilon-greedy strategy
            action = self._choose_action(current_state)
            
            # Execute action
            self._execute_action(action, X)
            
            # Store current state and action for next update
            self.last_state = current_state
            self.last_action = action
            
            # Store history
            self.state_history.append(current_state)
            self.action_history.append(action)
            self.reward_history.append(reward)
            if feedback is not None:
                self.feedback_history.append(feedback)
            
            return True, f"RL learning completed - State: {current_state}, Action: {action}, Reward: {reward:.3f}"
            
        except Exception as e:
            return False, f"RL learning failed: {str(e)}"
    
    def _get_current_state(self, X):
        """Determine current state based on recent performance"""
        try:
            if self.base_model is None:
                return 'normal'
            
            scores = self.base_model.decision_function(X)
            avg_score = np.mean(scores)
            
            if avg_score < self.suspicious_threshold:
                return 'anomalous'
            elif avg_score < self.normal_threshold:
                return 'suspicious'
            else:
                return 'normal'
                
        except:
            return 'normal'
    
    def _calculate_reward(self, X, feedback=None):
        """Calculate reward for RL learning"""
        try:
            if feedback is not None:
                # Use feedback to calculate reward
                predictions = self.predict_online(pd.DataFrame(X, columns=self.feature_columns))
                if predictions is not None:
                    accuracy = np.mean(predictions['anomalies'] == np.array(feedback).astype(bool))
                    return accuracy * 2 - 1  # Scale to [-1, 1]
            
            # Use consistency and confidence as reward
            predictions = self.predict_online(pd.DataFrame(X, columns=self.feature_columns))
            if predictions is not None:
                confidence = np.mean(predictions.get('confidence', [0.5]))
                return confidence * 2 - 1  # Scale to [-1, 1]
            
            return 0.0
            
        except:
            return 0.0
    
    def _update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        try:
            current_q = self.q_table[state][action]
            max_next_q = max(self.q_table[next_state].values())
            
            # Q-learning update
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state][action] = new_q
            
        except Exception as e:
            logger.warning(f"Q-table update failed: {str(e)}")
    
    def _choose_action(self, state):
        """Choose action using epsilon-greedy strategy"""
        try:
            if np.random.random() < self.epsilon:
                # Explore: choose random action
                return np.random.choice(self.actions)
            else:
                # Exploit: choose best action
                return max(self.q_table[state], key=self.q_table[state].get)
                
        except:
            return 'keep'  # Default action
    
    def _execute_action(self, action, X):
        """Execute the chosen action"""
        try:
            if action == 'increase_sens':
                # Increase sensitivity by lowering thresholds
                self.suspicious_threshold *= 1.1
                self.normal_threshold *= 1.1
                
            elif action == 'decrease_sens':
                # Decrease sensitivity by raising thresholds
                self.suspicious_threshold *= 0.9
                self.normal_threshold *= 0.9
                
            elif action == 'retrain':
                # Retrain base model with recent data
                if len(self.data_buffer) > 10:
                    recent_data = np.array(self.data_buffer[-100:])  # Last 100 samples
                    self.base_model.fit(recent_data)
                    
                    # Update thresholds
                    scores = self.base_model.decision_function(recent_data)
                    self.normal_threshold = np.percentile(scores, 10)
                    self.suspicious_threshold = np.percentile(scores, 5)
            
            # 'keep' action does nothing
            
        except Exception as e:
            logger.warning(f"Action execution failed: {str(e)}")
    
    def predict_online(self, data):
        """Make predictions with online model"""
        try:
            if self.model is None:
                return None
            
            numeric_data = data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return None
            
            X = numeric_data.values
            
            if self.model_type == "incremental_isolation_forest":
                all_predictions = []
                all_scores = []
                
                for name, model in self.model.items():
                    try:
                        if name == 'sgd_ocsvm':
                            pred = model.predict(X)
                            scores = model.decision_function(X) if hasattr(model, 'decision_function') else pred
                            all_predictions.append(pred == -1)  # Convert to anomaly boolean
                            all_scores.append(scores)
                            
                        elif hasattr(model, 'predict'):
                            pred = model.predict(X)
                            all_predictions.append(pred == 1)  # Assuming 1 is anomaly
                            all_scores.append(pred.astype(float))
                            
                    except Exception as e:
                        logger.warning(f"Prediction failed for {name}: {str(e)}")
                        continue
                
                if all_predictions:
                    # Ensemble prediction
                    ensemble_pred = np.mean(all_predictions, axis=0) > 0.5
                    ensemble_scores = np.mean(all_scores, axis=0) if all_scores else np.ones(len(ensemble_pred)) * 0.5
                    confidence = 1.0 - np.std(all_predictions, axis=0)
                    
                    return {
                        'anomalies': ensemble_pred,
                        'scores': ensemble_scores,
                        'confidence': confidence,
                        'method': 'incremental_ensemble'
                    }
                    
            elif self.model_type == "river_anomaly" and RIVER_AVAILABLE:
                anomalies = []
                scores = []
                
                # Use adaptive threshold based on score distribution
                # River scores are typically negative for normal, positive for anomalies
                # But threshold should be based on percentiles, not fixed value
                all_scores_temp = []
                for _, row in numeric_data.iterrows():
                    x = {str(i): float(val) for i, val in enumerate(row)}
                    score = self.model.score_one(x)
                    all_scores_temp.append(score)
                
                # Calculate threshold based on 95th percentile (more conservative)
                if len(all_scores_temp) > 0:
                    score_array = np.array(all_scores_temp)
                    threshold = np.percentile(score_array, 95)  # Top 5% are anomalies
                    
                    # Re-score with proper threshold
                    for score in all_scores_temp:
                        anomalies.append(score > threshold)
                        scores.append(score)
                else:
                    # Fallback to fixed threshold only if no scores
                    threshold = 0.5
                    for score in all_scores_temp:
                        anomalies.append(score > threshold)
                        scores.append(score)
                
                return {
                    'anomalies': np.array(anomalies),
                    'scores': np.array(scores),
                    'confidence': np.abs(np.array(scores) - threshold) / (np.abs(threshold) + 1e-8),  # Distance from adaptive threshold
                    'method': 'river_online',
                    'threshold_used': threshold
                }
                
            elif self.model_type == "reinforcement_learning":
                if self.base_model is None:
                    return None
                
                scores = self.base_model.decision_function(X)
                # For Isolation Forest, negative scores indicate anomalies
                # Use percentile-based threshold instead of fixed threshold to avoid too many false positives
                if not hasattr(self, 'normal_threshold') or self.normal_threshold is None:
                    # Calculate threshold from current scores if not set
                    threshold = np.percentile(scores, 5)  # Bottom 5% are anomalies
                else:
                    threshold = self.normal_threshold
                
                predictions = scores < threshold  # Below threshold = anomaly
                confidence = np.abs(scores - threshold) / (np.abs(threshold) + 1e-8)
                
                return {
                    'anomalies': predictions,
                    'scores': scores,
                    'confidence': confidence,
                    'method': 'reinforcement_learning',
                    'current_thresholds': {
                        'normal': self.normal_threshold,
                        'suspicious': self.suspicious_threshold
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Online prediction failed: {str(e)}")
            return None
    
    def _check_concept_drift(self):
        """Check for concept drift and adapt accordingly"""
        try:
            if len(self.data_buffer) < 100:  # Need enough data
                return
            
            # Simple concept drift detection based on performance degradation
            recent_data = np.array(self.data_buffer[-50:])  # Last 50 samples
            older_data = np.array(self.data_buffer[-100:-50])  # Previous 50 samples
            
            if self.model_type == "reinforcement_learning" and self.base_model is not None:
                # Compare score distributions
                recent_scores = self.base_model.decision_function(recent_data)
                older_scores = self.base_model.decision_function(older_data)
                
                # Statistical test for distribution difference (simplified)
                from scipy.stats import ks_2samp
                statistic, p_value = ks_2samp(recent_scores, older_scores)
                
                if p_value < 0.01:  # Significant difference detected
                    logger.info(f"Concept drift detected! P-value: {p_value:.6f}")
                    
                    # Trigger adaptation
                    if self.model_type == "reinforcement_learning":
                        # Increase exploration rate temporarily
                        self.epsilon = min(0.3, self.epsilon * 1.5)
                        
                        # Retrain base model
                        self.base_model.fit(recent_data)
                        
                        # Update thresholds
                        scores = self.base_model.decision_function(recent_data)
                        self.normal_threshold = np.percentile(scores, 10)
                        self.suspicious_threshold = np.percentile(scores, 5)
            
        except Exception as e:
            logger.warning(f"Concept drift detection failed: {str(e)}")
    
    def get_learning_statistics(self):
        """Get statistics about the online learning process"""
        try:
            stats = {
                'model_type': self.model_type,
                'data_buffer_size': len(self.data_buffer),
                'feedback_history_size': len(self.feedback_history),
                'learning_rate': self.learning_rate,
                'window_size': self.window_size
            }
            
            if self.model_type == "reinforcement_learning":
                stats.update({
                    'epsilon': self.epsilon,
                    'q_table_size': len(self.q_table),
                    'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
                    'recent_states': self.state_history[-10:] if self.state_history else [],
                    'recent_actions': self.action_history[-10:] if self.action_history else [],
                    'current_thresholds': {
                        'normal': getattr(self, 'normal_threshold', None),
                        'suspicious': getattr(self, 'suspicious_threshold', None)
                    }
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get learning statistics: {str(e)}")
            return {'error': str(e)}
    
    def save(self, filepath):
        """Save online learning model to file"""
        try:
            # Ensure filepath has .pkl extension
            if not filepath.endswith('.pkl'):
                filepath = filepath + '.pkl'
                
            # Create directory if needed
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Prepare online model data
            model_data = {
                "model_type": self.model_type,
                "learning_rate": self.learning_rate,
                "model": self.model,
                "scaler": getattr(self, 'scaler', None),
                "online_scaler": getattr(self, 'online_scaler', None),
                "feature_columns": getattr(self, 'feature_columns', None),
                "window_size": self.window_size,
                "data_buffer": self.data_buffer,
                "anomaly_buffer": self.anomaly_buffer,
                "performance_metrics": self.performance_metrics,
                "adaptation_threshold": self.adaptation_threshold,
                "feedback_history": self.feedback_history,
                "reward_history": self.reward_history,
                "state_history": self.state_history,
                "action_history": self.action_history,
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "model_performance": self.model_performance
            }
            
            # Add RL-specific attributes if they exist
            if hasattr(self, 'base_model'):
                model_data["base_model"] = self.base_model
            if hasattr(self, 'normal_threshold'):
                model_data["normal_threshold"] = self.normal_threshold
            if hasattr(self, 'suspicious_threshold'):
                model_data["suspicious_threshold"] = self.suspicious_threshold
            if hasattr(self, 'states'):
                model_data["states"] = self.states
            if hasattr(self, 'actions'):
                model_data["actions"] = self.actions
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            return True, f"Online learning model saved to {filepath}"
            
        except Exception as e:
            logger.error(f"Online model save error: {str(e)}")
            return False, f"Error saving online model: {str(e)}"
    
    @staticmethod
    def load(filepath):
        """Load online learning model from file"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return None, f"Model file not found: {filepath}"
                
            model_data = safe_pickle_load(filepath)
            
            if not isinstance(model_data, dict) or "model_type" not in model_data:
                return None, "Invalid online model file format"
            
            # Create online model instance
            online_model = OnlineLearningAnomalyDetector(
                model_type=model_data["model_type"],
                learning_rate=model_data.get("learning_rate", 0.01)
            )
            
            # Load all attributes
            online_model.model = model_data.get("model")
            online_model.scaler = model_data.get("scaler")
            online_model.online_scaler = model_data.get("online_scaler")
            online_model.feature_columns = model_data.get("feature_columns")
            online_model.window_size = model_data.get("window_size", 1000)
            online_model.data_buffer = model_data.get("data_buffer", [])
            online_model.anomaly_buffer = model_data.get("anomaly_buffer", [])
            online_model.performance_metrics = model_data.get("performance_metrics", {})
            online_model.adaptation_threshold = model_data.get("adaptation_threshold", 0.1)
            online_model.feedback_history = model_data.get("feedback_history", [])
            online_model.reward_history = model_data.get("reward_history", [])
            online_model.state_history = model_data.get("state_history", [])
            online_model.action_history = model_data.get("action_history", [])
            online_model.q_table = model_data.get("q_table", {})
            online_model.epsilon = model_data.get("epsilon", 0.1)
            online_model.gamma = model_data.get("gamma", 0.9)
            online_model.alpha = model_data.get("alpha", 0.1)
            online_model.model_performance = model_data.get("model_performance", [])
            
            # Load RL-specific attributes if they exist
            if "base_model" in model_data:
                online_model.base_model = model_data["base_model"]
            if "normal_threshold" in model_data:
                online_model.normal_threshold = model_data["normal_threshold"]
            if "suspicious_threshold" in model_data:
                online_model.suspicious_threshold = model_data["suspicious_threshold"]
            if "states" in model_data:
                online_model.states = model_data["states"]
            if "actions" in model_data:
                online_model.actions = model_data["actions"]
            
            return online_model, f"Online learning model loaded successfully from {filepath}"
            
        except Exception as e:
            logger.error(f"Online model load error: {str(e)}")
            return None, f"Error loading online model: {str(e)}"

class AnomalyDetectionModel:
    def __init__(self, model_type="isolation_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.sequence_length = 10   # For sequence-based models like LSTM
        self.reconstruction_error_threshold = None  # For autoencoders
        self.history = None  # To store training history for deep learning models
        self.metrics = {}  # Store model evaluation metrics
        self.X_train = None  # Store training data for metrics calculation
        
    def evaluate_model(self, true_labels=None, predictions=None):
        """Calculate model evaluation metrics"""
        try:
            if true_labels is None or predictions is None:
                # For unsupervised models, use internal metrics
                if hasattr(self.model, 'score_samples'):
                    scores = self.model.score_samples(self.X_train)
                    self.metrics['avg_anomaly_score'] = float(np.mean(scores))
                    self.metrics['std_anomaly_score'] = float(np.std(scores))
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(self.X_train)
                    self.metrics['avg_decision_score'] = float(np.mean(scores))
                    self.metrics['decision_threshold'] = float(np.percentile(scores, 90))
            else:
                # Calculate supervised metrics if we have true labels
                from sklearn.metrics import precision_score, recall_score, f1_score
                self.metrics['precision'] = float(precision_score(true_labels, predictions))
                self.metrics['recall'] = float(recall_score(true_labels, predictions))
                self.metrics['f1_score'] = float(f1_score(true_labels, predictions))
            
            # Add timestamp
            self.metrics['evaluation_time'] = datetime.datetime.now().isoformat()
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating model metrics: {str(e)}")
            return {}

    def _build_feature_frame(self, data, fit=False):
        """Build a numeric feature frame from mixed-type telemetry data."""
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame()

        feature_data = data.copy()

        # Convert explicit datetime-like columns into numeric time features.
        datetime_name_hints = {"time", "timestamp", "datetime", "date", "ds"}
        datetime_cols = [
            col for col in feature_data.columns
            if pd.api.types.is_datetime64_any_dtype(feature_data[col]) or str(col).strip().lower() in datetime_name_hints
        ]
        for col in datetime_cols:
            parsed = pd.to_datetime(feature_data[col], errors="coerce")
            if parsed.notna().any():
                feature_data[f"{col}__hour"] = parsed.dt.hour.fillna(0).astype(float)
                feature_data[f"{col}__dayofweek"] = parsed.dt.dayofweek.fillna(0).astype(float)
                feature_data[f"{col}__day"] = parsed.dt.day.fillna(0).astype(float)
                feature_data[f"{col}__month"] = parsed.dt.month.fillna(0).astype(float)
                min_ts = parsed.min()
                if pd.notna(min_ts):
                    feature_data[f"{col}__elapsed_sec"] = ((parsed - min_ts).dt.total_seconds()).fillna(0.0)
            feature_data = feature_data.drop(columns=[col], errors="ignore")

        # Handle booleans and categoricals.
        bool_cols = feature_data.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            feature_data[col] = feature_data[col].astype(int)

        categorical_cols = feature_data.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            feature_data = pd.get_dummies(
                feature_data,
                columns=list(categorical_cols),
                dummy_na=True,
                dtype=float,
            )

        feature_data = feature_data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if fit:
            self.feature_columns = list(feature_data.columns)
            return feature_data

        if self.feature_columns:
            for missing_col in self.feature_columns:
                if missing_col not in feature_data.columns:
                    feature_data[missing_col] = 0.0
            feature_data = feature_data.drop(columns=[c for c in feature_data.columns if c not in self.feature_columns], errors="ignore")
            feature_data = feature_data[self.feature_columns]

        return feature_data

    def _prepare_prediction_features(self, data):
        """Prepare prediction features aligned to the training schema."""
        if not isinstance(data, pd.DataFrame):
            return None, "Prediction data must be a pandas DataFrame"

        if self.feature_columns is not None:
            pred_data = self._build_feature_frame(data, fit=False)
            if pred_data.empty:
                return None, "None of the required feature columns found in data"
            return pred_data, None

        pred_data = data.select_dtypes(include=[np.number])
        if pred_data.empty:
            return None, "No numeric columns found in prediction data"
        return pred_data, None

    _TIMESTAMP_HINTS = ("date", "time", "timestamp", "datetime", "ds")

    def _runtime_model_type(self):
        """Return executable runtime model type for the current model key."""
        requested = getattr(self, "model_type", "isolation_forest")
        resolved = resolve_runtime_model_type(to_internal_model_type(requested))
        # If Prophet backend is unavailable, use robust statistical fallback.
        if resolved == "prophet" and not PROPHET_AVAILABLE:
            return "z-score"
        return resolved

    def _resolve_timestamp_column(self, data):
        stored = getattr(self, "timestamp_column", None)
        if stored and stored != "index" and stored in data.columns:
            return stored
        for col in data.columns:
            if str(col).strip().lower() in self._TIMESTAMP_HINTS:
                return col
        datetime_cols = [c for c in data.columns if pd.api.types.is_datetime64_any_dtype(data[c])]
        if datetime_cols:
            return datetime_cols[0]
        if stored == "index" or isinstance(data.index, pd.DatetimeIndex):
            return "index"
        return None

    def _resolve_target_column(self, data):
        stored = getattr(self, "target_column", None)
        if stored and stored in data.columns:
            return stored
        numeric = [c for c in data.select_dtypes(include=["number"]).columns]
        for name in ("value", "y"):
            if name in numeric:
                return name
        return numeric[0] if numeric else None

    def _supervised_anomaly_xy(self, X, contamination=0.1, rng_seed=42):
        """Augment clean telemetry with synthetic extremes so RF/XGB learn outliers."""
        rng = np.random.default_rng(rng_seed)
        n_synth = max(8, int(len(X) * max(float(contamination), 0.05)))
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_synth, X.shape[1]))
        magnitudes = 8.0 + rng.random((n_synth, X.shape[1])) * 4.0
        synth = mean + signs * magnitudes * std
        X_aug = np.vstack([X, synth])
        y = np.concatenate([np.zeros(len(X), dtype=int), np.ones(n_synth, dtype=int)])
        return X_aug, y

    def _calibrate_score_threshold(self, scores, contamination=0.1):
        scores = np.asarray(scores, dtype=float)
        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            self.score_threshold = 0.5
            return self.score_threshold
        percentile_cut = float(np.percentile(finite, (1.0 - float(contamination)) * 100.0))
        self.score_threshold = float(max(0.5, percentile_cut))
        return self.score_threshold

    def _supervised_combined_scores(self, X):
        """Blend classifier probability with scaled-feature distance.

        Tree models do not extrapolate to unseen extremes; the distance term
        keeps large telemetry jumps visible even when leaf probabilities stay low.
        """
        proba = self.model.predict_proba(X)
        clf = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        distance = np.max(np.abs(np.asarray(X, dtype=float)), axis=1)
        combined = np.maximum(clf, np.clip(distance / 6.0, 0.0, 1.0))
        return combined, distance

    def _common_state_dict(self):
        resolved = getattr(self, "resolved_model_type", self._runtime_model_type())
        return {
            "model_type": self.model_type,
            "resolved_model_type": resolved,
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "sequence_length": getattr(self, "sequence_length", 10),
            "reconstruction_error_threshold": getattr(self, "reconstruction_error_threshold", None),
            "metrics": getattr(self, "metrics", {}),
            "X_train": getattr(self, "X_train", None),
            "z_stats": getattr(self, "z_stats", None),
            "z_threshold": getattr(self, "z_threshold", None),
            "iqr_bounds": getattr(self, "iqr_bounds", None),
            "iqr_factor": getattr(self, "iqr_factor", None),
            "contamination": getattr(self, "contamination", None),
            "score_threshold": getattr(self, "score_threshold", None),
            "target_column": getattr(self, "target_column", None),
            "timestamp_column": getattr(self, "timestamp_column", None),
            "prophet_data": getattr(self, "prophet_data", None),
            "prophet_model": self.model if resolved == "prophet" else None,
        }

    def _apply_common_state(self, model_data):
        self.model = model_data.get("model")
        if self.model is None and model_data.get("prophet_model") is not None:
            self.model = model_data.get("prophet_model")
        self.scaler = model_data.get("scaler")
        self.feature_columns = model_data.get("feature_columns")
        self.sequence_length = model_data.get("sequence_length", 10)
        self.reconstruction_error_threshold = model_data.get("reconstruction_error_threshold")
        self.metrics = model_data.get("metrics", {})
        self.X_train = model_data.get("X_train")
        if model_data.get("z_stats") is not None:
            self.z_stats = model_data.get("z_stats")
        if model_data.get("z_threshold") is not None:
            self.z_threshold = model_data.get("z_threshold")
        if model_data.get("iqr_bounds") is not None:
            self.iqr_bounds = model_data.get("iqr_bounds")
        if model_data.get("iqr_factor") is not None:
            self.iqr_factor = model_data.get("iqr_factor")
        if model_data.get("contamination") is not None:
            self.contamination = model_data.get("contamination")
        if model_data.get("score_threshold") is not None:
            self.score_threshold = model_data.get("score_threshold")
        if model_data.get("target_column") is not None:
            self.target_column = model_data.get("target_column")
        if model_data.get("timestamp_column") is not None:
            self.timestamp_column = model_data.get("timestamp_column")
        if model_data.get("prophet_data") is not None:
            self.prophet_data = model_data.get("prophet_data")
        requested = model_data.get("model_type", self.model_type)
        self.resolved_model_type = model_data.get("resolved_model_type") or resolve_runtime_model_type(
            to_internal_model_type(requested)
        )

    def train(self, data, **kwargs):
        """Train the anomaly detection model"""
        try:
            # Ensure we're using numeric data for training
            if not isinstance(data, pd.DataFrame):
                return False, "Training data must be a pandas DataFrame"

            active_model_type = self._runtime_model_type()
            self.resolved_model_type = active_model_type
            if active_model_type != self.model_type:
                logger.info(
                    f"Model alias resolution: requested '{self.model_type}' -> runtime backend '{active_model_type}'"
                )

            numeric_data = data.select_dtypes(include=['number'])
            engineered_model_types = {
                "isolation_forest", "lof", "ocsvm", "random_forest",
                "autoencoder", "lstm", "gru", "xgboost"
            }
            if active_model_type in engineered_model_types:
                training_features = self._build_feature_frame(data, fit=True)
                if training_features.empty:
                    return False, "No valid features found in the training data"
                if len(numeric_data.columns) < len(data.columns):
                    engineered_from = sorted(set(data.columns) - set(numeric_data.columns))
                    logger.info(f"Applied feature engineering for non-numeric columns: {engineered_from}")
            else:
                training_features = numeric_data
                if training_features.empty:
                    return False, "No numeric columns found in the training data"
                self.feature_columns = list(training_features.columns)
                if len(numeric_data.columns) < len(data.columns):
                    excluded_cols = set(data.columns) - set(numeric_data.columns)
                    logger.warning(f"Non-numeric columns excluded from model training: {excluded_cols}")

            feature_count = len(training_features.columns)
            n_rows = len(training_features)
            logger.info(
                f"Training pipeline: model={active_model_type}, shape={n_rows:,} rows × {feature_count} features "
                f"(large datasets may take minutes without further messages until the next step completes)"
            )

            # Data preparation
            t_scale = time.perf_counter()
            logger.info(f"StandardScaler fit_transform starting ({n_rows:,} × {feature_count})…")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(training_features)
            logger.info(
                f"StandardScaler finished in {time.perf_counter() - t_scale:.2f}s (output shape {X.shape})"
            )
            
            # Traditional ML models
            if active_model_type == "isolation_forest":
                n_estimators = kwargs.get('n_estimators', 100)
                contamination = kwargs.get('contamination', 0.1)
                self.model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42
                )
                logger.info(
                    f"Fitting IsolationForest (n_estimators={n_estimators}, rows={n_rows:,})…"
                )
                t_fit = time.perf_counter()
                self.model.fit(X)
                logger.info(f"IsolationForest fit finished in {time.perf_counter() - t_fit:.2f}s")
                return True, f"Isolation Forest model trained successfully with {feature_count} features"
                
            elif active_model_type == "lof":
                n_neighbors = kwargs.get('n_neighbors', 20)
                contamination = kwargs.get('contamination', 0.1)
                self.model = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination,
                    novelty=True
                )
                logger.info(
                    f"Fitting LocalOutlierFactor (n_neighbors={n_neighbors}, rows={n_rows:,})…"
                )
                t_fit = time.perf_counter()
                self.model.fit(X)
                logger.info(f"LocalOutlierFactor fit finished in {time.perf_counter() - t_fit:.2f}s")
                return True, f"LOF model trained successfully with {feature_count} features"
            
            elif active_model_type == "ocsvm":
                nu = kwargs.get('nu', 0.05)
                kernel = kwargs.get('kernel', 'rbf')
                gamma = kwargs.get('gamma', 'scale')
                self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
                logger.info(
                    f"Fitting OneClassSVM (nu={nu}, kernel={kernel}, rows={n_rows:,}) — often slow on large n…"
                )
                t_fit = time.perf_counter()
                self.model.fit(X)
                logger.info(f"OneClassSVM fit finished in {time.perf_counter() - t_fit:.2f}s")
                return True, f"One-Class SVM trained successfully with {feature_count} features"

            elif active_model_type == "random_forest":
                n_estimators = kwargs.get('n_estimators', 200)
                max_depth = kwargs.get('max_depth', None)
                contamination = kwargs.get('contamination', 0.1)

                X_aug, y_train = self._supervised_anomaly_xy(X, contamination=contamination)
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    class_weight='balanced_subsample'
                )
                logger.info(
                    f"Fitting RandomForestClassifier (n_estimators={n_estimators}, rows={len(X_aug):,})…"
                )
                t_fit = time.perf_counter()
                self.model.fit(X_aug, y_train)
                logger.info(f"RandomForest fit finished in {time.perf_counter() - t_fit:.2f}s")
                self.contamination = contamination
                train_scores, _ = self._supervised_combined_scores(X)
                self._calibrate_score_threshold(train_scores, contamination)
                return True, f"Random Forest model trained successfully with {feature_count} features"

            # Deep learning models - require TensorFlow
            elif active_model_type in ["autoencoder", "lstm", "gru"] and TENSORFLOW_AVAILABLE:
                # Deep learning params
                epochs = kwargs.get('epochs', 50)
                batch_size = kwargs.get('batch_size', 32)
                validation_split = kwargs.get('validation_split', 0.2)
                patience = kwargs.get('patience', 5)
                learning_rate = kwargs.get('learning_rate', 0.001)
                
                # Set up early stopping
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                )

                if active_model_type == "autoencoder":
                    # Create an autoencoder for anomaly detection
                    input_dim = X.shape[1]
                    encoding_dim = max(1, input_dim // 2)  # Size of the encoded representation
                    
                    # Build encoder
                    input_layer = Input(shape=(input_dim,))
                    encoder = Dense(encoding_dim, activation='relu')(input_layer)
                    
                    # Build decoder
                    decoder = Dense(input_dim, activation='sigmoid')(encoder)
                    
                    # Set up the autoencoder
                    self.model = Model(inputs=input_layer, outputs=decoder)
                    self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
                    
                    # Train the model
                    logger.info(
                        f"Autoencoder: Keras fit starting (samples={X.shape[0]:,}, epochs≤{epochs}, batch={batch_size})…"
                    )
                    self.history = self.model.fit(
                        X, X,  # Autoencoder tries to reconstruct the input
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=[early_stopping],
                        verbose=1
                    )
                    
                    # Calculate reconstruction error threshold
                    logger.info(
                        f"Autoencoder: computing reconstruction MSE on full training set ({X.shape[0]:,} rows) for threshold…"
                    )
                    t_rec = time.perf_counter()
                    reconstructions = self.model.predict(X)
                    logger.info(
                        f"Autoencoder: reconstruction pass done in {time.perf_counter() - t_rec:.2f}s"
                    )
                    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
                    # Set threshold as the 95th percentile of reconstruction errors
                    self.reconstruction_error_threshold = np.percentile(mse, 95)
                    
                    return True, f"Autoencoder model trained successfully with {feature_count} features"
                    
                elif active_model_type in ["lstm", "gru"]:
                    # Check if we have timestamp column for sequence data
                    timestamp_col = kwargs.get('timestamp_column', None)
                    self.sequence_length = kwargs.get('sequence_length', 10)
                    
                    # Create sequences for training
                    sequences = self._create_sequences(X, self.sequence_length)
                    if len(sequences) == 0:
                        return False, f"Not enough data to create sequences with length {self.sequence_length}"
                    model_name = "LSTM" if active_model_type == "lstm" else "GRU"
                    
                    # Define model input shape
                    input_shape = (self.sequence_length, X.shape[1])
                    
                    # Build the model
                    self.model = Sequential()
                    
                    if active_model_type == "lstm":
                        # LSTM Autoencoder
                        self.model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
                        self.model.add(LSTM(units=32, return_sequences=False))
                        self.model.add(RepeatVector(self.sequence_length))
                        self.model.add(LSTM(units=32, return_sequences=True))
                        self.model.add(LSTM(units=64, return_sequences=True))
                        self.model.add(TimeDistributed(Dense(X.shape[1])))
                    else:  # GRU
                        # GRU Autoencoder
                        self.model.add(GRU(units=64, input_shape=input_shape, return_sequences=True))
                        self.model.add(GRU(units=32, return_sequences=False))
                        self.model.add(RepeatVector(self.sequence_length))
                        self.model.add(GRU(units=32, return_sequences=True))
                        self.model.add(GRU(units=64, return_sequences=True))
                        self.model.add(TimeDistributed(Dense(X.shape[1])))
                    
                    # Compile model
                    self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
                    
                    # Train model
                    logger.info(
                        f"{model_name}: Keras fit on sequences "
                        f"(n={len(sequences):,}, seq_len={self.sequence_length})…"
                    )
                    self.history = self.model.fit(
                        sequences, sequences,  # Sequence autoencoder
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=[early_stopping],
                        verbose=1
                    )
                    
                    # Calculate reconstruction error threshold
                    logger.info(f"{model_name}: reconstruction predict for threshold (n={len(sequences):,})…")
                    t_rec = time.perf_counter()
                    reconstructions = self.model.predict(sequences)
                    logger.info(
                        f"{model_name}: reconstruction pass done in {time.perf_counter() - t_rec:.2f}s"
                    )
                    mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
                    # Set threshold as the 95th percentile of reconstruction errors
                    self.reconstruction_error_threshold = np.percentile(mse, 95)
                    
                    return True, f"{model_name} autoencoder model trained successfully"
            
            # XGBoost model 
            elif active_model_type == "xgboost":
                try:
                    import xgboost as xgb
                    
                    n_estimators = kwargs.get('n_estimators', 100)
                    max_depth = kwargs.get('max_depth', 6)
                    learning_rate = kwargs.get('learning_rate', 0.1)
                    contamination = kwargs.get('contamination', 0.1)
                    
                    X_aug, y_train = self._supervised_anomaly_xy(X, contamination=contamination)
                    self.model = xgb.XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42,
                        eval_metric='logloss'
                    )
                    logger.info(
                        f"Fitting XGBClassifier (n_estimators={n_estimators}, rows={len(X_aug):,})…"
                    )
                    t_fit = time.perf_counter()
                    self.model.fit(X_aug, y_train)
                    logger.info(f"XGBoost fit finished in {time.perf_counter() - t_fit:.2f}s")
                    self.contamination = contamination
                    train_scores, _ = self._supervised_combined_scores(X)
                    self._calibrate_score_threshold(train_scores, contamination)
                    return True, f"XGBoost model trained successfully with {feature_count} features"
                    
                except ImportError:
                    return False, "XGBoost library not installed. Please install with: pip install xgboost"
                except Exception as e:
                    return False, f"Error training XGBoost model: {str(e)}"
            
            # IQR (Interquartile Range) method
            elif active_model_type == "iqr_(interquartile_range)":
                iqr_factor = kwargs.get('iqr_factor', 1.5)
                logger.info(
                    f"IQR: computing bounds ({len(numeric_data.columns)} columns, "
                    f"{len(numeric_data):,} rows) — quantiles can be slow on large data…"
                )
                
                # Calculate IQR bounds for each feature
                self.iqr_bounds = {}
                for col in numeric_data.columns:
                    q1 = numeric_data[col].quantile(0.25)
                    q3 = numeric_data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - iqr_factor * iqr
                    upper_bound = q3 + iqr_factor * iqr
                    self.iqr_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
                
                # Store the IQR factor for later use
                self.iqr_factor = iqr_factor
                
                # Set model to indicate training is complete
                self.model = "iqr_trained"
                
                return True, f"IQR model trained successfully with factor {iqr_factor}"
            
            # Z-Score method
            elif active_model_type == "z-score":
                threshold = kwargs.get('threshold', 3.0)
                logger.info(
                    f"Z-Score: computing per-column mean/std ({len(numeric_data.columns)} columns, "
                    f"{len(numeric_data):,} rows)…"
                )
                
                # Calculate mean and std for each feature
                self.z_stats = {}
                for col in numeric_data.columns:
                    self.z_stats[col] = {
                        'mean': numeric_data[col].mean(),
                        'std': numeric_data[col].std()
                    }
                
                # Store the threshold
                self.z_threshold = threshold
                
                # Set model to indicate training is complete
                self.model = "zscore_trained"
                
                return True, f"Z-Score model trained successfully with threshold {threshold}"
            
            # Prophet model for time series
            elif active_model_type == "prophet" and PROPHET_AVAILABLE:
                # Prophet requires a specific data format: ds (dates) and y (values)
                timestamp_col = kwargs.get('timestamp_column', None)
                target_col = kwargs.get('target_column', None)
                
                if timestamp_col is None:
                    timestamp_col = self._resolve_timestamp_column(data)
                    if timestamp_col is None:
                        return False, "Prophet requires a timestamp column. No timestamp column found in data."
                    logger.info(f"Prophet: Auto-detected timestamp column: {timestamp_col}")
                
                if target_col is None:
                    target_col = self._resolve_target_column(data)
                    if target_col is None:
                        return False, "Prophet requires a target column. No numeric columns found in data."
                    logger.info(f"Prophet: Auto-selected target column: {target_col}")
                
                # Validate columns exist
                if timestamp_col != 'index' and timestamp_col not in data.columns:
                    return False, f"Prophet: Timestamp column '{timestamp_col}' not found in data"
                if target_col not in data.columns:
                    return False, f"Prophet: Target column '{target_col}' not found in data"
                
                # Prepare data for Prophet
                if timestamp_col == 'index':
                    prophet_data = pd.DataFrame({
                        'ds': data.index,
                        'y': data[target_col]
                    })
                else:
                    prophet_data = data[[timestamp_col, target_col]].copy()
                    prophet_data.columns = ['ds', 'y']  # Prophet requires these column names
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(prophet_data['ds']):
                    try:
                        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                    except Exception as e:
                        return False, f"Prophet: Could not convert timestamp column to datetime: {str(e)}"
                
                # Drop NaN values
                prophet_data = prophet_data.dropna()
                
                if len(prophet_data) < 2:
                    return False, "Prophet requires at least 2 data points"
                
                # Create and fit Prophet model
                try:
                    self.model = Prophet(
                        yearly_seasonality=kwargs.get('yearly_seasonality', 'auto'),
                        weekly_seasonality=kwargs.get('weekly_seasonality', 'auto'),
                        daily_seasonality=kwargs.get('daily_seasonality', 'auto'),
                        changepoint_prior_scale=kwargs.get('changepoint_prior_scale', 0.05)
                    )
                    
                    logger.info(f"Prophet: fitting on {len(prophet_data):,} rows…")
                    t_pf = time.perf_counter()
                    self.model.fit(prophet_data)
                    logger.info(f"Prophet: fit finished in {time.perf_counter() - t_pf:.2f}s")
                    
                    # Store original data and target column for later use
                    self.prophet_data = prophet_data
                    self.target_column = target_col
                    self.timestamp_column = timestamp_col
                    
                    return True, f"Prophet model trained successfully on {len(prophet_data)} data points"
                except Exception as e:
                    return False, f"Prophet training failed: {str(e)}"
                
            else:
                if active_model_type in ["autoencoder", "lstm", "gru"] and not TENSORFLOW_AVAILABLE:
                    return False, f"TensorFlow is required for {self.model_type} models but is not installed"
                if active_model_type == "prophet" and not PROPHET_AVAILABLE:
                    return False, "Prophet is required but is not installed"
                    
                return False, f"Unsupported model type: {self.model_type}"
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False, f"Error training model: {str(e)}"
    
    def predict(self, data):
        """Predict anomalies in new data"""
        try:
            if self.model is None:
                return None, "Model not trained"
            active_model_type = getattr(self, "resolved_model_type", self._runtime_model_type())
            
            # Traditional ML models
            if active_model_type in ["isolation_forest", "lof", "ocsvm"]:
                pred_data, prep_error = self._prepare_prediction_features(data)
                if prep_error:
                    return None, prep_error
                    
                # Transform using the saved scaler
                X = self.scaler.transform(pred_data)
                
                if active_model_type == "isolation_forest":
                    scores = self.model.decision_function(X)
                    predictions = self.model.predict(X)  # 1: normal, -1: anomaly
                elif active_model_type == "lof":
                    scores = self.model.decision_function(X)
                    predictions = self.model.predict(X)
                else:  # ocsvm
                    scores = self.model.decision_function(X)
                    predictions = self.model.predict(X)
                    
                # Convert to consistent format: True for anomaly, False for normal
                anomalies = (predictions == -1)
                return scores, anomalies
                
            # Deep Learning Models
            elif active_model_type == "autoencoder" and TENSORFLOW_AVAILABLE:
                pred_data, prep_error = self._prepare_prediction_features(data)
                if prep_error:
                    return None, prep_error
                
                # Scale the data
                X = self.scaler.transform(pred_data)
                
                # Get reconstructions
                reconstructions = self.model.predict(X)
                
                # Calculate MSE (anomaly score)
                mse = np.mean(np.power(X - reconstructions, 2), axis=1)
                
                # Determine anomalies using the training-calibrated threshold only.
                threshold = self.reconstruction_error_threshold
                if threshold is None:
                    return None, "Model threshold not calibrated. Retrain the model before prediction."
                anomalies = mse > threshold
                
                return mse, anomalies
                
            # Sequence models (LSTM, GRU)
            elif active_model_type in ["lstm", "gru"] and TENSORFLOW_AVAILABLE:
                pred_data, prep_error = self._prepare_prediction_features(data)
                if prep_error:
                    return None, prep_error
                
                # Scale the data
                X = self.scaler.transform(pred_data)
                
                # Create sequences
                sequences = self._create_sequences(X, self.sequence_length)
                if len(sequences) == 0:
                    return None, f"Not enough data to create sequences with length {self.sequence_length}"
                
                # Get reconstructions
                reconstructions = self.model.predict(sequences)
                
                # Calculate MSE (anomaly score)
                mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
                
                # Pad the scores to match original data length
                pad_size = len(X) - len(mse)
                scores = np.pad(mse, (pad_size, 0), 'constant', constant_values=np.nan)
                
                # Determine anomalies using the training-calibrated threshold only.
                threshold = self.reconstruction_error_threshold
                if threshold is None:
                    return None, "Model threshold not calibrated. Retrain the model before prediction."
                sequence_anomalies = mse > threshold
                
                # Pad the anomalies to match original data length
                anomalies = np.pad(sequence_anomalies, (pad_size, 0), 'constant', constant_values=False)
                
                return scores, anomalies
                
            # XGBoost model
            elif active_model_type in ["xgboost", "random_forest"]:
                pred_data, prep_error = self._prepare_prediction_features(data)
                if prep_error:
                    return None, prep_error
                
                # Scale the data
                X = self.scaler.transform(pred_data)
                
                scores, distance = self._supervised_combined_scores(X)
                contamination = float(getattr(self, "contamination", 0.1))
                threshold = getattr(self, "score_threshold", None)
                if threshold is None:
                    threshold = np.percentile(scores, (1 - contamination) * 100)
                anomalies = (scores > threshold) | (distance > 3.5)
                
                return scores, anomalies
            
            # IQR (Interquartile Range) method
            elif active_model_type == "iqr_(interquartile_range)":
                # Check if model is trained
                if not hasattr(self, 'iqr_bounds') or not self.iqr_bounds:
                    logger.error("IQR model not trained - iqr_bounds not found")
                    return None, None
                
                # Prepare data
                if self.feature_columns is not None:
                    available_cols = [col for col in self.feature_columns if col in data.columns]
                    if not available_cols:
                        logger.error("None of the required feature columns found in data")
                        return None, None
                    pred_data = data[available_cols]
                else:
                    pred_data = data.select_dtypes(include=['number'])
                
                if pred_data.empty or len(pred_data.columns) == 0:
                    logger.error("No numeric columns found for IQR prediction")
                    return None, None
                
                # Calculate anomaly scores and detect anomalies
                anomaly_count = np.zeros(len(pred_data))
                scores = np.zeros(len(pred_data))
                
                for col in pred_data.columns:
                    if col in self.iqr_bounds:
                        bounds = self.iqr_bounds[col]
                        # Count violations for each data point
                        violations = ((pred_data[col] < bounds['lower']) | 
                                    (pred_data[col] > bounds['upper']))
                        anomaly_count += violations.astype(int)
                        
                        # Calculate distance from bounds as score
                        lower_dist = np.maximum(0, bounds['lower'] - pred_data[col])
                        upper_dist = np.maximum(0, pred_data[col] - bounds['upper'])
                        col_score = lower_dist + upper_dist
                        scores += col_score
                
                # Normalize scores by number of features
                if len(pred_data.columns) > 0:
                    scores = scores / len(pred_data.columns)
                
                # Mark as anomaly if more than 25% of features are outside bounds
                # This prevents flagging everything when you have many features
                anomaly_threshold = max(1, len(pred_data.columns) * 0.25)
                anomalies = anomaly_count >= anomaly_threshold
                
                logger.info(f"IQR: {len(pred_data.columns)} features, threshold={anomaly_threshold:.1f}, anomalies={np.sum(anomalies)}/{len(anomalies)}")
                
                return scores, anomalies
            
            # Z-Score method
            elif active_model_type == "z-score":
                # Check if model is trained
                if not hasattr(self, 'z_stats') or not self.z_stats:
                    logger.error("Z-Score model not trained - z_stats not found")
                    return None, None
                
                if not hasattr(self, 'z_threshold'):
                    logger.error("Z-Score model not trained - z_threshold not found")
                    return None, None
                
                # Prepare data
                if self.feature_columns is not None:
                    available_cols = [col for col in self.feature_columns if col in data.columns]
                    if not available_cols:
                        logger.error("None of the required feature columns found in data")
                        return None, None
                    pred_data = data[available_cols]
                else:
                    pred_data = data.select_dtypes(include=['number'])
                
                if pred_data.empty or len(pred_data.columns) == 0:
                    logger.error("No numeric columns found for Z-Score prediction")
                    return None, None
                
                # Calculate Z-scores
                z_scores = np.zeros((len(pred_data), len(pred_data.columns)))
                anomaly_feature_count = np.zeros(len(pred_data))
                
                for i, col in enumerate(pred_data.columns):
                    if col in self.z_stats:
                        stats = self.z_stats[col]
                        if stats['std'] > 0:  # Avoid division by zero
                            z_scores[:, i] = np.abs((pred_data[col] - stats['mean']) / stats['std'])
                            # Count features that exceed threshold
                            anomaly_feature_count += (z_scores[:, i] > self.z_threshold).astype(int)
                        else:
                            z_scores[:, i] = 0
                
                # Use maximum Z-score across features as the anomaly score
                scores = np.max(z_scores, axis=1)
                
                # Mark as anomaly if more than 25% of features exceed threshold
                # This prevents flagging everything when you have many features
                anomaly_threshold = max(1, len(pred_data.columns) * 0.25)
                anomalies = anomaly_feature_count >= anomaly_threshold
                
                logger.info(f"Z-Score: {len(pred_data.columns)} features, threshold={anomaly_threshold:.1f}, anomalies={np.sum(anomalies)}/{len(anomalies)}")
                
                return scores, anomalies
                
            # Prophet model
            elif active_model_type == "prophet" and PROPHET_AVAILABLE:
                timestamp_col = self._resolve_timestamp_column(data)
                if timestamp_col is None:
                    return None, "No timestamp column found for Prophet prediction"
                target_col = self._resolve_target_column(data)
                if target_col is None:
                    return None, "No numeric target column found for Prophet prediction"
                
                if timestamp_col == "index":
                    prophet_data = pd.DataFrame({"ds": data.index, "y": data[target_col]})
                else:
                    prophet_data = data[[timestamp_col, target_col]].copy()
                    prophet_data.columns = ['ds', 'y']
                
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_dtype(prophet_data['ds']):
                    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
                
                # Make prediction using Prophet model
                forecast = self.model.predict(prophet_data)
                
                # Calculate residuals (prediction error)
                residuals = prophet_data['y'].values - forecast['yhat'].values[:len(prophet_data)]
                
                # Use median absolute deviation to identify outliers (anomalies)
                mad = np.median(np.abs(residuals - np.median(residuals)))
                threshold = 3 * 1.4826 * mad  # 3 sigma equivalent for MAD
                
                # Identify anomalies
                scores = np.abs(residuals)
                anomalies = scores > threshold
                
                return scores, anomalies
                
            else:
                return None, f"Prediction not implemented for model type: {self.model_type}"
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return None, f"Error during prediction: {str(e)}"
    
    def save(self, filepath):
        """Save model to file"""
        if self.model is None:
            return False, "No model to save"
            
        try:
            active_model_type = getattr(self, "resolved_model_type", self._runtime_model_type())
            # Ensure filepath has .pkl extension
            if not filepath.endswith('.pkl'):
                filepath = filepath + '.pkl'
                
            # Create directory if needed
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Special handling for TensorFlow models
            if active_model_type in ["autoencoder", "lstm", "gru"] and TENSORFLOW_AVAILABLE and hasattr(self.model, 'save'):
                try:
                    # Save the model architecture and weights separately
                    model_dir = filepath.replace('.pkl', '')
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Save Keras model with proper file extension
                    keras_model_path = os.path.join(model_dir, 'keras_model.keras')
                    self.model.save(keras_model_path)
                    
                    attributes = self._common_state_dict()
                    attributes.pop("model", None)
                    attributes.pop("prophet_model", None)
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(attributes, f)
                        
                    return True, f"TensorFlow model saved to {filepath} and {keras_model_path}"
                    
                except Exception as e:
                    logger.error(f"TensorFlow model save error: {str(e)}")
                    return False, f"Error saving TensorFlow model: {str(e)}"
                
            # Special handling for Prophet models
            elif active_model_type == "prophet" and PROPHET_AVAILABLE:
                try:
                    model_data = self._common_state_dict()
                    with open(filepath, 'wb') as f:
                        pickle.dump(model_data, f)
                    return True, f"Prophet model saved to {filepath}"
                    
                except Exception as e:
                    logger.error(f"Prophet model save error: {str(e)}")
                    return False, f"Error saving Prophet model: {str(e)}"
                
            # Default handling for all other models (including enhanced models)
            else:
                try:
                    model_data = self._common_state_dict()
                    
                    # Add enhanced model specific attributes if they exist
                    if hasattr(self, 'preprocessing_pipeline'):
                        model_data["preprocessing_pipeline"] = self.preprocessing_pipeline
                    if hasattr(self, 'ensemble_models'):
                        model_data["ensemble_models"] = self.ensemble_models
                    if hasattr(self, 'adaptive_thresholds'):
                        model_data["adaptive_thresholds"] = self.adaptive_thresholds
                    if hasattr(self, 'model_params'):
                        model_data["model_params"] = self.model_params
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(model_data, f)
                    return True, f"Model saved to {filepath}"
                    
                except Exception as e:
                    logger.error(f"Model save error: {str(e)}")
                    return False, f"Error saving model: {str(e)}"
                
        except Exception as e:
            logger.error(f"Model save error: {str(e)}")
            return False, f"Error saving model: {str(e)}"
    
    @staticmethod
    def load(filepath):
        """Load model from file"""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                return None, f"Model file not found: {filepath}"
                
            # Check if it's a TensorFlow model by looking for keras_model.keras in the same directory
            model_dir = filepath.replace('.pkl', '')
            keras_model_path = os.path.join(model_dir, 'keras_model.keras')
            is_tensorflow_model = os.path.exists(keras_model_path) and TENSORFLOW_AVAILABLE
            
            # Standard pickle loading first
            try:
                with open(filepath, 'rb') as f:
                    model_data = safe_pickle_load(filepath)
            except Exception as e:
                return None, f"Error reading model file: {str(e)}"
            
            # Handle TensorFlow models
            if is_tensorflow_model and isinstance(model_data, dict) and "model_type" in model_data:
                try:
                    import tensorflow as tf
                    
                    model = AnomalyDetectionModel(model_data["model_type"])
                    model._apply_common_state(model_data)
                    model.model = tf.keras.models.load_model(keras_model_path)
                    
                    return model, "TensorFlow model loaded successfully"
                except Exception as e:
                    logger.error(f"TensorFlow model loading failed: {str(e)}")
                    return None, f"Error loading TensorFlow model: {str(e)}"
            
            prophet_payload = False
            if isinstance(model_data, dict):
                requested_type = str(model_data.get("model_type") or "")
                resolved_saved = model_data.get("resolved_model_type") or resolve_runtime_model_type(
                    to_internal_model_type(requested_type) if requested_type else ""
                )
                prophet_payload = (
                    model_data.get("prophet_model") is not None
                    or resolved_saved == "prophet"
                    or requested_type in {"prophet", "Prophet"}
                )
            if prophet_payload:
                if not PROPHET_AVAILABLE:
                    return None, "Prophet library not available for loading Prophet models"
                try:
                    model = AnomalyDetectionModel(model_data.get("model_type", "prophet"))
                    model._apply_common_state(model_data)
                    if model.model is None:
                        return None, "Prophet payload is missing the fitted model"
                    model.resolved_model_type = "prophet"
                    return model, "Prophet model loaded successfully"
                except Exception as e:
                    return None, f"Error loading Prophet model: {str(e)}"
            
            # Handle traditional ML models and enhanced models
            if isinstance(model_data, dict) and "model_type" in model_data:
                try:
                    # Create model instance
                    model_type = model_data["model_type"]
                    
                    # Check if it's an enhanced model
                    if model_type in ["enhanced_isolation_forest", "ensemble_voting", "ensemble_stacking", "adaptive_threshold"]:
                        try:
                            # Try to load as enhanced model
                            model = EnhancedAnomalyDetectionModel(model_type)
                            
                            # Load enhanced model attributes
                            if "preprocessing_pipeline" in model_data:
                                model.preprocessing_pipeline = model_data["preprocessing_pipeline"]
                            if "ensemble_models" in model_data:
                                model.ensemble_models = model_data["ensemble_models"]
                            if "adaptive_thresholds" in model_data:
                                model.adaptive_thresholds = model_data["adaptive_thresholds"]
                            if "model_params" in model_data:
                                model.model_params = model_data["model_params"]
                                
                        except NameError:
                            # If EnhancedAnomalyDetectionModel is not available, fall back to regular model
                            model = AnomalyDetectionModel(model_type)
                    else:
                        model = AnomalyDetectionModel(model_type)
                    if isinstance(model, AnomalyDetectionModel):
                        model._apply_common_state(model_data)
                    else:
                        model.model = model_data.get("model")
                        model.scaler = model_data.get("scaler")
                        model.feature_columns = model_data.get("feature_columns")
                        if hasattr(model, "X_train"):
                            model.X_train = model_data.get("X_train")
                    
                    return model, f"{model_type} model loaded successfully"
                    
                except Exception as e:
                    logger.error(f"Error loading model attributes: {str(e)}")
                    return None, f"Error loading model: {str(e)}"
            
            # Handle legacy format or corrupted files
            else:
                return None, f"Invalid model file format or corrupted file: {filepath}"
                
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            return None, f"Error loading model: {str(e)}"
    
    def _create_sequences(self, data, seq_length):
        """Create sequences for LSTM/GRU training"""
        sequences = []
        # Check if we have enough data points
        if len(data) <= seq_length:
            return np.array(sequences)
            
        # Create sequences
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
            
        return np.array(sequences)

# Visualization components
