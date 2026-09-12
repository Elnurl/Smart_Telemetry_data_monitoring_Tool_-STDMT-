#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

if sys.version_info >= (3, 14):
    print(
        "STDMS does not support Python 3.14 yet (pandas/PyQt wheels missing).\n"
        "Use Python 3.10 instead:\n"
        "  run.bat\n"
        "  py -3.10 main.py\n"
        r"  %LocalAppData%\Programs\Python\Python310\python.exe main.py"
    )
    raise SystemExit(1)
import json
import hashlib
import inspect
import base64
import logging
import datetime
import time
import socket
import getpass
import threading
import multiprocessing
import urllib.parse
import re
import html
from pathlib import Path
import traceback  # Add this import
import numpy as np
import pandas as pd
import pickle
from scipy import stats  # For statistical tests
from statsmodels.tsa.stattools import adfuller  # For stationarity tests
# Removed Crypto imports
import secrets  # For secure random bytes generation
try:
    from cryptography.fernet import Fernet, InvalidToken
    FERNET_AVAILABLE = True
except ImportError:
    Fernet = None
    InvalidToken = Exception
    FERNET_AVAILABLE = False
import sqlite3  # For telemetry data storage
import smtplib
import uuid  # For generating unique IDs
import types
from collections import defaultdict, deque
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urllib_request
from urllib import error as urllib_error
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
try:
    import serial  # type: ignore
except Exception:
    serial = None

# UI imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTextEdit, QTextBrowser, QFileDialog, QMessageBox, QComboBox,
                             QTabWidget, QGroupBox, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QCheckBox, QRadioButton, QStackedWidget,
                             QDialog, QProgressBar, QScrollArea, QSplitter,
                             QFrame, QToolBar, QStatusBar, QAction, QMenu, QToolButton, 
                             QSizePolicy, QGridLayout, QProgressDialog, QListWidget,
                             QListWidgetItem, QAbstractItemView, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QSize, QEvent
from PyQt5.QtGui import QIcon, QFont, QPixmap, QPalette, QColor, QKeyEvent

# Visualization
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting support

# Machine learning imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import LocalOutlierFactor  # Added missing import
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
#After model implementing need to measure or monitor result metrics to evaluate the model

# Configure logging FIRST (before any code that uses logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("security.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SecureAnomalyDetection")

def to_json_compatible(value):
    """Convert numpy/pandas/datetime objects to JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_json_compatible(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return value


# Slice C Wave 1: model catalog
from app.models.model_types import (
    MODEL_DISPLAY_TO_INTERNAL,
    MODEL_INTERNAL_FALLBACK,
    TOUCHED_44_MODEL_NAMES,
    catalog_display_name,
    format_model_catalog_label,
    get_supported_model_names,
    resolve_runtime_model_type,
    to_internal_model_type,
)

# Incremental modular refactor imports (backward-compatible fallback below).
SettingsManager = None
AppServiceLayer = None
ServiceLayerContext = None
modular_load_alert_routing_config = None
modular_write_audit_event = None
modular_get_runtime_source_context = None
ModularLocalAuthProvider = None
ModularOIDCAuthProvider = None
ModularDataReader = None
ModularPasswordPolicy = None
modular_compute_health_score = None
modular_build_tab_visibility_policy = None
ModularModelRegistry = None
modular_evaluate_obs_limits = None
modular_load_rule_config = None
format_registry_error = None
export_fleet_mission_report = None
try:
    from app.config.settings import SettingsManager
    from app.services.service_layer import AppServiceLayer, ServiceLayerContext
    from app.alerts.policy import load_alert_routing_config as modular_load_alert_routing_config
    from app.storage.audit import (
        write_audit_event as modular_write_audit_event,
        get_runtime_source_context as modular_get_runtime_source_context,
    )
    from app.auth.providers import LocalAuthProvider as ModularLocalAuthProvider, OIDCAuthProvider as ModularOIDCAuthProvider
    from app.ingestion.data_reader import DataReader as ModularDataReader
    from app.security.password_policy import PasswordPolicy as ModularPasswordPolicy
    from app.models.health_scoring import compute_health_score as modular_compute_health_score
    from app.ui.authorization import build_tab_visibility_policy as modular_build_tab_visibility_policy
    from app.models.registry import ModelRegistry as ModularModelRegistry, format_registry_error
    from app.monitoring.obs_limits import evaluate_obs_limits as modular_evaluate_obs_limits
    from app.monitoring.obs_limits import load_rule_config as modular_load_rule_config
    from app.reports.mission_report import export_fleet_mission_report
except Exception as modular_import_error:
    logger.info(f"Modular package not fully available yet: {modular_import_error}")

if ModularModelRegistry is not None:
    ModelRegistry = ModularModelRegistry


# Slice C Wave 1: observability
from app.monitoring.observability import ObservabilityManager, OBSERVABILITY

# Multi-Agent System imports
from system_health_monitor import SystemHealthMonitor, SklearnAutoencoder

try:
    import psutil  # For system metrics collection
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. System health monitoring will be limited.")

try:
    from pmdarima import auto_arima  # For auto ARIMA parameter selection
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logger.warning("pmdarima not available. ARIMA will use manual parameters.")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_ARIMA_AVAILABLE = True
except ImportError:
    STATSMODELS_ARIMA_AVAILABLE = False
    logger.warning("statsmodels ARIMA not available. Trend analysis will be limited.")

# TensorFlow/Keras imports for deep learning models
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    if tf.__version__:
        from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout, Reshape
        from tensorflow.keras.models import Model, Sequential
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        TENSORFLOW_AVAILABLE = True
except (ImportError, AttributeError):
    logger.warning("TensorFlow not available. Deep learning models will be disabled.")

# Theme System
class ThemeManager:
    """Manages light and dark themes for the application"""
    
    def __init__(self):
        self.current_theme = "light"  # Default theme
        
    def get_light_theme(self):
        """Returns light theme stylesheet"""
        return {
            "main_window": """
                QMainWindow {
                    background-color: #ffffff;
                    color: #333333;
                }
                QTabWidget::pane {
                    border: 1px solid #cccccc;
                    background-color: #ffffff;
                }
                QTabBar::tab {
                    background-color: #f0f0f0;
                    color: #333333;
                    padding: 8px 12px;
                    border: 1px solid #cccccc;
                    border-bottom: none;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #ffffff;
                    border-bottom: 1px solid #ffffff;
                }
                QTabBar::tab:hover {
                    background-color: #e0e0e0;
                }
            """,
            
            "groupbox": """
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #cccccc;
                    border-radius: 8px;
                    margin-top: 1ex;
                    background-color: #fafafa;
                    color: #333333;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #2196F3;
                }
            """,
            
            "button_primary": """
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
                QPushButton:pressed {
                    background-color: #1565C0;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """,
            
            "button_secondary": """
                QPushButton {
                    background-color: #f5f5f5;
                    color: #333333;
                    border: 1px solid #cccccc;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                    border-color: #999999;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
            """,
            
            "button_danger": """
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
                QPushButton:pressed {
                    background-color: #c62828;
                }
            """,
            
            "input": """
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 8px;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    background-color: #ffffff;
                    color: #333333;
                    selection-background-color: #2196F3;
                }
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                    border-color: #2196F3;
                }
            """,
            
            "table": """
                QTableWidget {
                    background-color: #ffffff;
                    alternate-background-color: #f5f5f5;
                    color: #333333;
                    gridline-color: #e0e0e0;
                    border: 1px solid #cccccc;
                }
                QTableWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #e0e0e0;
                }
                QTableWidget::item:selected {
                    background-color: #2196F3;
                    color: white;
                }
                QHeaderView::section {
                    background-color: #f0f0f0;
                    color: #333333;
                    padding: 8px;
                    border: 1px solid #cccccc;
                    font-weight: bold;
                }
            """,
            
            "statusbar": """
                QStatusBar {
                    background-color: #f0f0f0;
                    color: #333333;
                    border-top: 1px solid #cccccc;
                }
            """,
            
            "toolbar": """
                QToolBar {
                    spacing: 10px;
                    padding: 5px;
                    background-color: #f8f9fa;
                    border-bottom: 1px solid #dee2e6;
                }
                QToolButton {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                    padding: 8px 15px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QToolButton:hover {
                    background-color: #e0e0e0;
                    border-color: #999999;
                }
                QToolButton:pressed {
                    background-color: #d0d0d0;
                }
            """
        }
    
    def get_dark_theme(self):
        """Returns dark theme stylesheet"""
        return {
            "main_window": """
                QMainWindow {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #2b2b2b;
                }
                QTabBar::tab {
                    background-color: #404040;
                    color: #ffffff;
                    padding: 8px 12px;
                    border: 1px solid #555555;
                    border-bottom: none;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #2b2b2b;
                    border-bottom: 1px solid #2b2b2b;
                }
                QTabBar::tab:hover {
                    background-color: #505050;
                }
            """,
            
            "groupbox": """
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 8px;
                    margin-top: 1ex;
                    background-color: #353535;
                    color: #ffffff;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #64B5F6;
                }
            """,
            
            "button_primary": """
                QPushButton {
                    background-color: #1976D2;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #1565C0;
                }
                QPushButton:pressed {
                    background-color: #0D47A1;
                }
                QPushButton:disabled {
                    background-color: #555555;
                    color: #888888;
                }
            """,
            
            "button_secondary": """
                QPushButton {
                    background-color: #404040;
                    color: #ffffff;
                    border: 1px solid #666666;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #505050;
                    border-color: #888888;
                }
                QPushButton:pressed {
                    background-color: #606060;
                }
            """,
            
            "button_danger": """
                QPushButton {
                    background-color: #d32f2f;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #c62828;
                }
                QPushButton:pressed {
                    background-color: #b71c1c;
                }
            """,
            
            "input": """
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    padding: 8px;
                    border: 1px solid #666666;
                    border-radius: 4px;
                    background-color: #404040;
                    color: #ffffff;
                    selection-background-color: #1976D2;
                }
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                    border-color: #64B5F6;
                }
                QComboBox::drop-down {
                    background-color: #404040;
                    border: none;
                }
                QComboBox::down-arrow {
                    color: #ffffff;
                }
            """,
            
            "table": """
                QTableWidget {
                    background-color: #2b2b2b;
                    alternate-background-color: #353535;
                    color: #ffffff;
                    gridline-color: #555555;
                    border: 1px solid #666666;
                }
                QTableWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #555555;
                }
                QTableWidget::item:selected {
                    background-color: #1976D2;
                    color: white;
                }
                QHeaderView::section {
                    background-color: #404040;
                    color: #ffffff;
                    padding: 8px;
                    border: 1px solid #666666;
                    font-weight: bold;
                }
            """,
            
            "statusbar": """
                QStatusBar {
                    background-color: #404040;
                    color: #ffffff;
                    border-top: 1px solid #666666;
                }
            """,
            
            "toolbar": """
                QToolBar {
                    spacing: 10px;
                    padding: 5px;
                    background-color: #353535;
                    border-bottom: 1px solid #666666;
                }
                QToolButton {
                    background-color: #404040;
                    color: #ffffff;
                    border: 1px solid #666666;
                    padding: 8px 15px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QToolButton:hover {
                    background-color: #505050;
                    border-color: #888888;
                }
                QToolButton:pressed {
                    background-color: #606060;
                }
            """
        }
    
    def get_current_theme(self):
        """Returns the current theme styles"""
        if self.current_theme == "dark":
            return self.get_dark_theme()
        else:
            return self.get_light_theme()
    
    def switch_theme(self):
        """Switch between light and dark themes"""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        return self.current_theme



# Slice C Wave 1: multi-agent models
from app.models.multi_agent import (
    AutoencoderLSTMAgent,
    CNNAgent,
    MultiAgentFusionSystem,
    PMDARIMA_AVAILABLE,
    PSUTIL_AVAILABLE,
    STATSMODELS_ARIMA_AVAILABLE,
    TrendAnalysisModel,
    create_sequences,
    create_sequences_3d,
)

_PLACEHOLDER_EMAIL_DOMAINS = frozenset({"example.com", "example.org", "example.net"})


def _is_deliverable_alert_email(email) -> bool:
    """True when *email* is non-empty and not a reserved placeholder domain."""
    address = str(email or "").strip()
    if not address or "@" not in address:
        return False
    domain = address.split("@", 1)[-1].strip().lower()
    return domain not in _PLACEHOLDER_EMAIL_DOMAINS

# Load typed modular settings when available.
if SettingsManager is not None:
    try:
        _APP_SETTINGS = SettingsManager(os.getcwd()).load()
        ENCRYPTION_KEY_FILE = _APP_SETTINGS.paths.encryption_key_file
        USER_DB_FILE = _APP_SETTINGS.paths.user_db_file
        CONFIG_FILE = _APP_SETTINGS.paths.config_file
        AUTH_CONFIG_FILE = _APP_SETTINGS.paths.auth_config_file
        MODELS_DIR = _APP_SETTINGS.paths.models_dir
        DATA_DIR = _APP_SETTINGS.paths.data_dir
        REPORTS_DIR = _APP_SETTINGS.paths.reports_dir
        TELEMETRY_DB_FILE = _APP_SETTINGS.paths.telemetry_db_file
        RULE_CONFIG_FILE = _APP_SETTINGS.paths.rule_config_file
        ALERT_ROUTING_CONFIG_FILE = _APP_SETTINGS.paths.alert_routing_config_file
        ACTIVITY_LOG_FILE = _APP_SETTINGS.paths.activity_log_file
        AUDIT_LOG_FILE = _APP_SETTINGS.paths.audit_log_file
        AUTO_REPORTS_DIR = _APP_SETTINGS.paths.auto_reports_dir
    except Exception as settings_error:
        logger.warning(f"Failed to load modular settings, using legacy constants: {settings_error}")
        _APP_SETTINGS = None
else:
    _APP_SETTINGS = None

# Observability settings
if _APP_SETTINGS is not None:
    OBSERVABILITY_ENABLED = bool(_APP_SETTINGS.observability.enabled)
    OBSERVABILITY_HOST = str(_APP_SETTINGS.observability.host)
    OBSERVABILITY_PORT = int(_APP_SETTINGS.observability.port)
else:
    OBSERVABILITY_ENABLED = os.getenv("MONITOR_OBS_ENABLED", "1").strip().lower() not in ("0", "false", "no")
    OBSERVABILITY_HOST = os.getenv("MONITOR_OBS_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        OBSERVABILITY_PORT = int(os.getenv("MONITOR_OBS_PORT", "9108"))
    except (TypeError, ValueError):
        OBSERVABILITY_PORT = 9108

# Ensure directories exist
for directory in [os.path.dirname(ENCRYPTION_KEY_FILE), 
                 os.path.dirname(USER_DB_FILE),
                 os.path.dirname(CONFIG_FILE),
                 MODELS_DIR, DATA_DIR, REPORTS_DIR,
                 os.path.dirname(ACTIVITY_LOG_FILE),
                 os.path.dirname(AUDIT_LOG_FILE),
                 AUTO_REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

TIMESTAMP_COLUMN_NAMES = frozenset({
    "time", "timestamp", "datetime", "date", "ts", "ds", "utc", "epoch",
})
_FERNET_PREFIX = b"F1:"


# Slice C Wave 1: trusted pickle helpers
from app.security.pickle_safe import (
    SecurityError,
    configure_trusted_pickle_roots,
    safe_pickle_load,
)

configure_trusted_pickle_roots(DATA_DIR, MODELS_DIR, REPORTS_DIR)


def _trusted_pickle_roots():
    from app.security.pickle_safe import trusted_pickle_roots

    return trusted_pickle_roots()


def _detect_timestamp_column(columns):
    for col in columns:
        if str(col).strip().lower() in TIMESTAMP_COLUMN_NAMES:
            return col
    return None


def _file_telemetry_sort_key(filepath):
    """Prefer latest in-file telemetry timestamp over filesystem mtime."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".csv":
            header = pd.read_csv(filepath, nrows=0)
            ts_col = _detect_timestamp_column(header.columns)
            if ts_col is not None:
                series = pd.read_csv(filepath, usecols=[ts_col])[ts_col]
                parsed = pd.to_datetime(series, errors="coerce", utc=True)
                if parsed.notna().any():
                    return parsed.max().timestamp()
        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list) and payload:
                records = payload
            elif isinstance(payload, dict):
                records = payload.get("data") or payload.get("records") or [payload]
            else:
                records = []
            if records and isinstance(records[0], dict):
                ts_col = _detect_timestamp_column(records[0].keys())
                if ts_col:
                    parsed = pd.to_datetime(
                        [row.get(ts_col) for row in records if isinstance(row, dict)],
                        errors="coerce",
                        utc=True,
                    )
                    if parsed.notna().any():
                        return parsed.max().timestamp()
    except Exception as exc:
        logger.debug("Telemetry sort fallback for %s: %s", filepath, exc)
    return os.path.getmtime(filepath)

# Initialize observability manager (global singleton for MVP instrumentation)
if OBSERVABILITY_ENABLED:
    try:
        OBSERVABILITY.start_http_server(host=OBSERVABILITY_HOST, port=OBSERVABILITY_PORT)
    except Exception as obs_start_error:
        logger.error(f"Failed to start observability server: {obs_start_error}")

# Security utility functions
class SecurityUtils:
    @staticmethod
    def hash_password(password, salt=None):
        if salt is None:
            salt = os.urandom(32)  # 32 bytes salt
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt + key
    
    @staticmethod
    def verify_password(stored_password, provided_password):
        salt = stored_password[:32]  # First 32 bytes is salt
        key = stored_password[32:]
        new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return secrets.compare_digest(new_key, key)
    
    @staticmethod
    def load_encryption_key():
        """Load or create encryption key"""
        try:
            if os.path.exists(ENCRYPTION_KEY_FILE):
                with open(ENCRYPTION_KEY_FILE, 'rb') as f:
                    key = f.read()
            else:
                key = secrets.token_bytes(32)  # 32 bytes key
                with open(ENCRYPTION_KEY_FILE, 'wb') as f:
                    f.write(key)
            return key
        except Exception as e:
            logger.error(f"Error loading encryption key: {e}")
            return None
    
    @staticmethod
    def _fernet_from_key(key):
        if not FERNET_AVAILABLE:
            return None
        return Fernet(base64.urlsafe_b64encode(key[:32]))

    @staticmethod
    def encrypt_data(data, key=None):
        """Encrypt data with Fernet (AES-128-CBC + HMAC); legacy XOR only as fallback."""
        if key is None:
            key = SecurityUtils.load_encryption_key()
        if isinstance(data, str):
            data = data.encode("utf-8")
        fernet = SecurityUtils._fernet_from_key(key)
        if fernet is not None:
            return base64.b64encode(_FERNET_PREFIX + fernet.encrypt(data))
        logger.warning("cryptography package unavailable; using legacy XOR encryption")
        return SecurityUtils._encrypt_data_xor(data, key)

    @staticmethod
    def _encrypt_data_xor(data, key):
        """Legacy XOR encryption kept only for backward compatibility."""
        iv = secrets.token_bytes(16)
        stretched_key = b""
        for i in range(0, len(data), len(key)):
            stretched_key += hashlib.sha256(key + str(i).encode()).digest()
        stretched_key = stretched_key[:len(data)]
        encrypted_data = bytearray(len(data))
        for i in range(len(data)):
            encrypted_data[i] = data[i] ^ stretched_key[i]
        return base64.b64encode(iv + bytes(encrypted_data))

    @staticmethod
    def decrypt_data(encrypted_data, key=None):
        """Decrypt Fernet payloads; fall back to legacy XOR format when needed."""
        if key is None:
            key = SecurityUtils.load_encryption_key()
        raw = base64.b64decode(encrypted_data)
        if raw.startswith(_FERNET_PREFIX):
            fernet = SecurityUtils._fernet_from_key(key)
            if fernet is None:
                raise SecurityError("Fernet payload requires the cryptography package")
            try:
                return fernet.decrypt(raw[len(_FERNET_PREFIX):])
            except InvalidToken as exc:
                raise SecurityError("Failed to decrypt Fernet payload") from exc
        return SecurityUtils._decrypt_data_xor(raw, key)

    @staticmethod
    def _decrypt_data_xor(data, key):
        """Legacy XOR decryption for older encrypted user databases."""
        iv = data[:16]
        payload = data[16:]
        stretched_key = b""
        for i in range(0, len(payload), len(key)):
            stretched_key += hashlib.sha256(key + str(i).encode()).digest()
        stretched_key = stretched_key[:len(payload)]
        decrypted_data = bytearray(len(payload))
        for i in range(len(payload)):
            decrypted_data[i] = payload[i] ^ stretched_key[i]
        return bytes(decrypted_data)
    
    @staticmethod
    def secure_log(action, message, level="info"):
        """Log security events with user, timestamp"""
        if level == "info":
            logger.info(f"{action}: {message}")
        elif level == "warning":
            logger.warning(f"{action}: {message}")
        elif level == "error":
            logger.error(f"{action}: {message}")

# Database initialization
def initialize_telemetry_database():
    """Initialize the SQLite database for telemetry data storage"""
    conn = sqlite3.connect(TELEMETRY_DB_FILE)
    cursor = conn.cursor()
    
    # Table for raw telemetry data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS telemetry_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        source TEXT,
        sensor TEXT,
        value REAL,
        metadata TEXT
    )
    ''')
    
    # Table for detected anomalies
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        detection_time DATETIME,
        source TEXT,
        sensor TEXT, 
        value REAL,
        anomaly_type TEXT,
        detection_method TEXT,
        severity INTEGER,
        anomaly_score REAL,
        description TEXT
    )
    ''')
    
    # Table for model evaluations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        model_type TEXT,
        model_name TEXT,
        parameters TEXT,
        metrics TEXT,
        training_time REAL
    )
    ''')
    
    # Table for alert logs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alert_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        anomaly_id INTEGER,
        alert_type TEXT,
        recipients TEXT,
        status TEXT,
        FOREIGN KEY (anomaly_id) REFERENCES anomalies(id)
    )
    ''')
    
    # Table for data sources configuration
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        type TEXT,
        connection_string TEXT,
        polling_interval INTEGER,
        is_active BOOLEAN,
        last_poll DATETIME,
        config TEXT
    )
    ''')

    # Table for alert incidents (deduplicated, lifecycle-managed alerts)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alert_incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fingerprint TEXT,
        status TEXT,
        severity TEXT,
        message TEXT,
        model_name TEXT,
        first_seen DATETIME,
        last_seen DATETIME,
        occurrence_count INTEGER DEFAULT 1,
        acknowledged_by TEXT,
        acknowledged_at DATETIME,
        escalated INTEGER DEFAULT 0,
        next_escalation_at DATETIME,
        last_payload TEXT
    )
    ''')

    # Table for alert lifecycle events (audit trail)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alert_lifecycle_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER,
        event_time DATETIME,
        event_type TEXT,
        severity TEXT,
        channel TEXT,
        status TEXT,
        details TEXT,
        FOREIGN KEY (incident_id) REFERENCES alert_incidents(id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize the database when module is loaded
initialize_telemetry_database()

# Create or load rule-based detection configuration
def load_rule_config(force_reload=False):
    if modular_load_rule_config is not None:
        return modular_load_rule_config(RULE_CONFIG_FILE, force_reload=force_reload)

    default_config = {
        "rules": [
            {"name": "High CPU Temperature", "parameter": "cpu_temp", "condition": "value > 80", "severity": 3},
            {"name": "Low Disk Space", "parameter": "disk_space", "condition": "value < 10", "severity": 2},
        ],
        "global_settings": {"check_interval_seconds": 60, "min_alert_severity": 2},
    }
    if not os.path.exists(RULE_CONFIG_FILE):
        with open(RULE_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    try:
        with open(RULE_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load rule config: {e}")
        return default_config


def evaluate_obs_limits(data, rules=None, threshold_scale=1.0, mission_mode=None):
    """Evaluate operational bounds (OBS) rules against the latest telemetry row."""
    if modular_evaluate_obs_limits is not None:
        return modular_evaluate_obs_limits(
            data,
            rules,
            rule_config_file=RULE_CONFIG_FILE,
            threshold_scale=threshold_scale,
            mission_mode=mission_mode,
        )

    try:
        from app.monitoring.obs_limits import evaluate_obs_condition
    except Exception:
        logger.warning("OBS modular evaluator unavailable; skipping rule checks")
        return {"ok": True, "violations": []}

    if data is None or len(data) == 0:
        return {"ok": True, "violations": []}
    if rules is None:
        rules = load_rule_config().get("rules", [])
    violations = []
    latest = data.iloc[-1]
    for rule in rules or []:
        param = rule.get("parameter")
        condition = str(rule.get("condition", "")).strip()
        if not param or param not in data.columns or not condition:
            continue
        try:
            value = float(latest[param])
            if evaluate_obs_condition(value, condition):
                violations.append({
                    "name": rule.get("name", param),
                    "parameter": param,
                    "value": value,
                    "condition": condition,
                    "severity": rule.get("severity", 1),
                })
        except Exception:
            continue
    return {"ok": len(violations) == 0, "violations": violations}


def load_alert_routing_config():
    """Load alert routing/escalation configuration."""
    if modular_load_alert_routing_config is not None:
        try:
            return modular_load_alert_routing_config(ALERT_ROUTING_CONFIG_FILE, logger)
        except Exception as e:
            logger.error(f"Modular alert routing config load failed, using local loader: {e}")
    
    default_config = {
        "cooldown_seconds": 60,
        "dedup_window_seconds": 300,
        "escalation_unacked_minutes": 15,
        "routes": {
            "info": ["email"],
            "warning": ["email", "webhook"],
            "critical": ["email", "webhook", "pagerduty"]
        },
        "channels": {
            "email": {"enabled": True},
            "slack": {"enabled": False, "webhook_url": ""},
            "webhook": {"enabled": False, "url": ""},
            "pagerduty": {
                "enabled": False,
                "events_api_url": "https://events.pagerduty.com/v2/enqueue",
                "routing_key": ""
            }
        }
    }
    
    if not os.path.exists(ALERT_ROUTING_CONFIG_FILE):
        try:
            with open(ALERT_ROUTING_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create alert routing config: {e}")
        return default_config
    
    try:
        with open(ALERT_ROUTING_CONFIG_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            merged = default_config.copy()
            merged.update(loaded)
            merged["routes"] = {**default_config["routes"], **loaded.get("routes", {})}
            merged["channels"] = {**default_config["channels"], **loaded.get("channels", {})}
            return merged
    except Exception as e:
        logger.error(f"Failed to load alert routing config: {e}")
    
    return default_config


def get_runtime_source_context():
    """Collect source context for audit logs."""
    if modular_get_runtime_source_context is not None:
        try:
            return modular_get_runtime_source_context()
        except Exception:
            pass
    
    hostname = socket.gethostname()
    local_ip = "unknown"
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        pass
    
    return {
        "hostname": hostname,
        "local_ip": local_ip,
        "os_user": getpass.getuser(),
        "process_id": os.getpid()
    }


def write_audit_event(actor, action, resource="*", outcome="success", details=None, auth_provider="local", source_context=None):
    """Write audit-grade activity event as JSONL."""
    if modular_write_audit_event is not None:
        try:
            modular_write_audit_event(
                audit_log_file=AUDIT_LOG_FILE,
                actor=actor,
                action=action,
                resource=resource,
                outcome=outcome,
                details=details,
                auth_provider=auth_provider,
                source_context=source_context
            )
            return
        except Exception:
            pass
    
    event = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "actor": str(actor) if actor is not None else "unknown",
        "action": str(action),
        "resource": str(resource),
        "outcome": str(outcome),
        "auth_provider": str(auth_provider),
        "source": source_context or get_runtime_source_context(),
        "details": details or {}
    }
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write audit event: {e}")


def load_auth_config():
    """Load pluggable auth provider configuration."""
    default_config = {
        "provider": "local",
        "oidc": {
            "enabled": False,
            "issuer": "",
            "token_endpoint": "",
            "userinfo_endpoint": "",
            "client_id": "",
            "client_secret": "",
            "scope": "openid profile email",
            "username_claim": "preferred_username",
            "email_claim": "email",
            "role_claim": "roles",
            "role_mapping": {
                "admin": "admin",
                "analyst": "analyst",
                "viewer": "viewer"
            },
            "default_role": "viewer"
        }
    }
    
    if not os.path.exists(AUTH_CONFIG_FILE):
        try:
            with open(AUTH_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to create auth config: {e}")
        return default_config
    
    try:
        with open(AUTH_CONFIG_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            merged = default_config.copy()
            merged.update(loaded)
            merged["oidc"] = {**default_config["oidc"], **loaded.get("oidc", {})}
            return merged
    except Exception as e:
        logger.error(f"Failed to load auth config: {e}")
    
    return default_config


if AppServiceLayer is None or ServiceLayerContext is None:
    class ServiceLayerContext:  # type: ignore[no-redef]
        def __init__(self, username, auth_provider="local", session_id=None):
            self.username = username
            self.auth_provider = auth_provider
            self.session_id = session_id

    class AppServiceLayer:  # type: ignore[no-redef]
        def __init__(self, user_manager, alert_policy_manager=None, audit_writer=None):
            self.user_manager = user_manager
            self.alert_policy_manager = alert_policy_manager
            self.audit_writer = audit_writer

        def authorize(self, username, permission, resource="*"):
            return bool(self.user_manager.check_permission(username, permission, resource=resource))

        def process_alert(self, severity, message, payload=None):
            if self.alert_policy_manager is None:
                return {"routed": False}
            return self.alert_policy_manager.process_alert(severity, message, payload or {})

        def acknowledge_alert(self, incident_id, context):
            if self.alert_policy_manager is None:
                return False
            return bool(self.alert_policy_manager.acknowledge(incident_id, context.username))

        def audit(self, context, action, resource="*", outcome="success", details=None):
            if self.audit_writer is None:
                return
            self.audit_writer(
                actor=context.username,
                action=action,
                resource=resource,
                outcome=outcome,
                details=details or {},
                auth_provider=context.auth_provider
            )


class BaseAuthProvider:
    """Pluggable auth provider interface — prefer app.auth.providers when available."""
    provider_name = "base"

    def authenticate(self, username, password, user_manager):
        raise NotImplementedError


# Canonical providers live in app.auth.providers; keep names for UserManager wiring.
if ModularLocalAuthProvider is not None:
    LocalAuthProvider = ModularLocalAuthProvider
else:
    class LocalAuthProvider(BaseAuthProvider):
        provider_name = "local"

        def authenticate(self, username, password, user_manager):
            return user_manager._authenticate_local(username, password)

if ModularOIDCAuthProvider is not None:
    OIDCAuthProvider = ModularOIDCAuthProvider
else:
    class OIDCAuthProvider(BaseAuthProvider):
        provider_name = "oidc"

        def _map_role(self, raw_roles, mapping, default_role="viewer"):
            if raw_roles is None:
                return default_role
            if isinstance(raw_roles, str):
                raw_roles = [raw_roles]
            for role in raw_roles:
                mapped = mapping.get(str(role).lower())
                if mapped:
                    return mapped
            return default_role

        def authenticate(self, username, password, user_manager):
            cfg = (user_manager.auth_config or {}).get("oidc", {})
            if not cfg.get("enabled"):
                return False, None, "OIDC provider is disabled.", False, {"auth_provider": "oidc"}
            return False, None, "OIDC provider module unavailable.", False, {"auth_provider": "oidc"}


# Slice C Wave 1: alerts — single source app.alerts.policy
from app.alerts.policy import AlertPolicyManager

# Maximum number of worker processes to use for data processing
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# User management system
# Canonical auth matrix/providers live in app.auth; this class remains the monolith
# shell used by run.bat (bootstrap, GUI lockout, email hooks). ToolHost login uses
# this instance via LocalAuthProvider → _authenticate_local.
class UserManager:
    ROLES = {
        "admin": ["view_data", "import_data", "process_data", "manage_users", "configure_system", "train_models"],
        "analyst": ["view_data", "import_data", "process_data", "train_models"],
        "viewer": ["view_data", "import_data", "process_data"]
    }
    PERMISSION_MATRIX = {
        "view_data": {"*": ["admin", "analyst", "viewer"]},
        "import_data": {"*": ["admin", "analyst", "viewer"], "data_source:*": ["admin", "analyst"]},
        "process_data": {"*": ["admin", "analyst"]},
        "train_models": {"*": ["admin", "analyst"], "model:production": ["admin"]},
        "manage_users": {"*": ["admin"]},
        "configure_system": {"*": ["admin"]},
        "ack_alert": {"*": ["admin", "analyst"], "alert:*": ["admin", "analyst"]},
        "create_tab": {"*": ["admin", "analyst"], "tab:*": ["admin", "analyst"]},
        "delete_tab": {"tab:*": ["admin"], "*": ["admin"]}
    }
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_MINUTES = 15
    MIN_PASSWORD_LENGTH = 12
    
    def __init__(self):
        self.auth_config = load_auth_config()
        local_provider_cls = LocalAuthProvider
        oidc_provider_cls = OIDCAuthProvider
        self.auth_providers = {
            "local": local_provider_cls(),
            "oidc": oidc_provider_cls()
            # SAML provider can be added later under same interface
        }
        self.load_users()
    
    def _password_meets_policy(self, password):
        """Basic password policy for new/updated passwords."""
        if ModularPasswordPolicy is not None:
            policy = ModularPasswordPolicy(min_length=self.MIN_PASSWORD_LENGTH)
            return policy.validate(password)
        
        if not password or len(password) < self.MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {self.MIN_PASSWORD_LENGTH} characters."
        if not any(ch.isupper() for ch in password):
            return False, "Password must include at least one uppercase letter."
        if not any(ch.islower() for ch in password):
            return False, "Password must include at least one lowercase letter."
        if not any(ch.isdigit() for ch in password):
            return False, "Password must include at least one digit."
        if not any(not ch.isalnum() for ch in password):
            return False, "Password must include at least one special character."
        return True, "Password policy satisfied."
    
    def _generate_bootstrap_password(self):
        """Generate a strong temporary password for initial admin setup."""
        return f"Tmp!{secrets.token_urlsafe(10)}A9"
    
    def _create_bootstrap_admin(self, reason="initial setup"):
        """Create a secure bootstrap admin account requiring password change."""
        bootstrap_password = self._generate_bootstrap_password()
        self.users = {
            "admin": {
                "password": SecurityUtils.hash_password(bootstrap_password).hex(),
                "role": "admin",
                "email": "",
                "auth_provider": "local",
                "password_change_required": True,
                "failed_attempts": 0,
                "lockout_until": None,
                "last_password_change": None
            }
        }
        self.save_users()
        
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            bootstrap_file = os.path.join(DATA_DIR, "bootstrap_admin_password.txt")
            with open(bootstrap_file, "w", encoding="utf-8") as f:
                f.write(
                    "SDA v4.0 - Initial Admin Credentials\n"
                    "------------------------------------------------------------\n"
                    f"Reason: {reason}\n"
                    "Username: admin\n"
                    f"Temporary Password: {bootstrap_password}\n\n"
                    "IMPORTANT:\n"
                    "- This password is temporary and must be changed at first login.\n"
                    "- Delete this file after successful password change.\n"
                )
            logger.warning(f"Bootstrap admin credentials written to: {bootstrap_file}")
        except Exception as e:
            logger.error(f"Failed to write bootstrap admin password file: {e}")
    
    def _ensure_user_defaults(self):
        """Migrate user records to include security fields."""
        changed = False
        for username, user in self.users.items():
            if "failed_attempts" not in user:
                user["failed_attempts"] = 0
                changed = True
            if "lockout_until" not in user:
                user["lockout_until"] = None
                changed = True
            if "password_change_required" not in user:
                user["password_change_required"] = False
                changed = True
            if "last_password_change" not in user:
                user["last_password_change"] = None
                changed = True
            if "auth_provider" not in user:
                user["auth_provider"] = "local"
                changed = True

            email = str(user.get("email", "")).strip()
            if email and not _is_deliverable_alert_email(email):
                user["email"] = ""
                changed = True
            
            # Harden legacy weak bootstrap account if still present
            if username == "admin" and user.get("password"):
                try:
                    stored_pwd = bytes.fromhex(user["password"])
                    if SecurityUtils.verify_password(stored_pwd, "admin"):
                        user["password_change_required"] = True
                        changed = True
                        logger.warning("Detected weak legacy admin password; forced password change is enabled.")
                except Exception:
                    pass
        
        if changed:
            self.save_users()
    
    def _get_lockout_remaining_seconds(self, user):
        lockout_until = user.get("lockout_until")
        if not lockout_until:
            return 0
        
        try:
            unlock_at = datetime.datetime.fromisoformat(lockout_until)
            remaining = (unlock_at - datetime.datetime.utcnow()).total_seconds()
            if remaining <= 0:
                user["lockout_until"] = None
                user["failed_attempts"] = 0
                self.save_users()
                return 0
            return int(remaining)
        except Exception:
            user["lockout_until"] = None
            user["failed_attempts"] = 0
            self.save_users()
            return 0
    
    def load_users(self):
        """Load users from encrypted JSON file"""
        if os.path.exists(USER_DB_FILE):
            try:
                with open(USER_DB_FILE, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_data = SecurityUtils.decrypt_data(encrypted_data)
                self.users = json.loads(decrypted_data)
                if not isinstance(self.users, dict):
                    raise ValueError("User database format is invalid")
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
                self._create_bootstrap_admin(reason="recovery from user database load failure")
                return
            try:
                self._ensure_user_defaults()
            except Exception as e:
                # Defaults must never wipe a successfully decrypted user DB.
                logger.error(f"Failed to normalize user defaults (users kept): {e}")
        else:
            # Create secure bootstrap admin user
            self._create_bootstrap_admin(reason="first run")

    def add_user(self, username, password, role, email):
        """Add new user with email"""
        if username in self.users:
            return False
        
        if role not in self.ROLES:
            return False
        
        meets_policy, _ = self._password_meets_policy(password)
        if not meets_policy:
            return False
        
        self.users[username] = {
            "password": SecurityUtils.hash_password(password).hex(),
            "role": role,
            "email": email,
            "auth_provider": "local",
            "password_change_required": False,
            "failed_attempts": 0,
            "lockout_until": None,
            "last_password_change": datetime.datetime.utcnow().isoformat()
        }
        self.save_users()
        SecurityUtils.secure_log("USER_ADDED", f"New user {username} with role {role} added")
        write_audit_event(
            actor="system",
            action="user_create",
            resource=f"user/{username}",
            outcome="success",
            details={"role": role, "email": email},
            auth_provider="local"
        )
        return True
    
    def save_users(self):
        """Save users to encrypted JSON file"""
        try:
            user_data = json.dumps(self.users).encode('utf-8')
            encrypted_data = SecurityUtils.encrypt_data(user_data)
            with open(USER_DB_FILE, 'wb') as f:
                f.write(encrypted_data)
            logger.info("User database saved successfully")
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
    
    def _authenticate_local(self, username, password):
        """Local credential authentication provider logic."""
        if username not in self.users:
            SecurityUtils.secure_log("LOGIN_FAILED", f"Failed login attempt for username: {username}", "warning")
            return False, None, "Invalid username or password.", False, {"auth_provider": "local"}
        
        user = self.users[username]
        remaining_lockout = self._get_lockout_remaining_seconds(user)
        if remaining_lockout > 0:
            minutes_left = max(1, int(np.ceil(remaining_lockout / 60)))
            return False, None, f"Account locked. Try again in {minutes_left} minute(s).", False, {"auth_provider": "local"}
        
        try:
            stored_pwd_hex = user.get("password")
            if not stored_pwd_hex:
                return False, None, "This account does not support local password login.", False, {"auth_provider": "local"}
            
            stored_pwd = bytes.fromhex(stored_pwd_hex)
            if SecurityUtils.verify_password(stored_pwd, password):
                user["failed_attempts"] = 0
                user["lockout_until"] = None
                user["last_login"] = datetime.datetime.utcnow().isoformat()
                user["auth_provider"] = "local"
                self.save_users()
                
                require_change = bool(user.get("password_change_required", False))
                SecurityUtils.secure_log("LOGIN", f"User {username} logged in successfully")
                return True, user["role"], "Login successful.", require_change, {"auth_provider": "local"}
        except Exception as e:
            logger.error(f"Authentication error for {username}: {e}")
        
        attempts = int(user.get("failed_attempts", 0)) + 1
        user["failed_attempts"] = attempts
        message = "Invalid username or password."
        
        if attempts >= self.MAX_FAILED_ATTEMPTS:
            unlock_time = datetime.datetime.utcnow() + datetime.timedelta(minutes=self.LOCKOUT_MINUTES)
            user["lockout_until"] = unlock_time.isoformat()
            user["failed_attempts"] = 0
            message = f"Too many failed attempts. Account locked for {self.LOCKOUT_MINUTES} minutes."
        
        self.save_users()
        SecurityUtils.secure_log("LOGIN_FAILED", f"Failed login attempt for username: {username}", "warning")
        return False, None, message, False, {"auth_provider": "local"}
    
    def ensure_identity_record(self, username, role, email="", auth_provider="local"):
        """Create/update identity record for external auth providers."""
        username = str(username).strip()
        if not username:
            return
        
        user = self.users.get(username, {})
        user["role"] = role if role in self.ROLES else "viewer"
        user["email"] = email or user.get("email", "")
        user["auth_provider"] = auth_provider
        user.setdefault("password_change_required", False)
        user.setdefault("failed_attempts", 0)
        user.setdefault("lockout_until", None)
        user.setdefault("last_password_change", None)
        user["last_login"] = datetime.datetime.utcnow().isoformat()
        
        self.users[username] = user
        self.save_users()
    
    def _resolve_auth_provider(self):
        provider_name = str((self.auth_config or {}).get("provider", "local")).strip().lower()
        provider = self.auth_providers.get(provider_name)
        if provider is None:
            provider = self.auth_providers.get("local")
            provider_name = "local"
        return provider_name, provider
    
    def authenticate_user(self, username, password):
        """Authenticate via pluggable auth provider interface."""
        auth_span = OBSERVABILITY.start_span("auth.login", {"username": username}) if OBSERVABILITY else None
        if OBSERVABILITY:
            OBSERVABILITY.inc_counter("auth_attempts_total", labels={"result": "attempt"})
        
        provider_name, provider = self._resolve_auth_provider()
        source_context = get_runtime_source_context()
        
        try:
            authenticated, role, message, require_password_change, auth_context = provider.authenticate(username, password, self)
            auth_context = auth_context or {}
            resolved_provider = auth_context.get("auth_provider", provider_name)
            
            if authenticated:
                write_audit_event(
                    actor=username,
                    action="login",
                    resource="auth/session",
                    outcome="success",
                    details={"role": role, "password_change_required": bool(require_password_change)},
                    auth_provider=resolved_provider,
                    source_context=source_context
                )
                if OBSERVABILITY:
                    OBSERVABILITY.inc_counter("auth_attempts_total", labels={"result": "success"})
                    OBSERVABILITY.end_span(auth_span, status="ok", attributes={"auth_provider": resolved_provider})
                return True, role, message, require_password_change, auth_context
            
            write_audit_event(
                actor=username,
                action="login",
                resource="auth/session",
                outcome="failed",
                details={"message": message},
                auth_provider=resolved_provider,
                source_context=source_context
            )
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("auth_attempts_total", labels={"result": "failed"})
                OBSERVABILITY.end_span(auth_span, status="error", attributes={"auth_provider": resolved_provider})
            return False, None, message, False, auth_context
        except Exception as e:
            write_audit_event(
                actor=username,
                action="login",
                resource="auth/session",
                outcome="error",
                details={"error": str(e)},
                auth_provider=provider_name,
                source_context=source_context
            )
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("auth_attempts_total", labels={"result": "error"})
                OBSERVABILITY.end_span(auth_span, status="error", error=e, attributes={"auth_provider": provider_name})
            return False, None, "Authentication provider error.", False, {"auth_provider": provider_name}
    
    def update_password(self, username, new_password):
        """Update password and clear forced-change flag."""
        if username not in self.users:
            return False, "User not found."
        
        meets_policy, policy_message = self._password_meets_policy(new_password)
        if not meets_policy:
            return False, policy_message
        
        self.users[username]["password"] = SecurityUtils.hash_password(new_password).hex()
        self.users[username]["password_change_required"] = False
        self.users[username]["failed_attempts"] = 0
        self.users[username]["lockout_until"] = None
        self.users[username]["last_password_change"] = datetime.datetime.utcnow().isoformat()
        self.save_users()
        
        SecurityUtils.secure_log("PASSWORD_CHANGED", f"User {username} changed password")
        write_audit_event(
            actor=username,
            action="password_change",
            resource=f"user/{username}",
            outcome="success",
            details={},
            auth_provider=self.users[username].get("auth_provider", "local")
        )
        return True, "Password updated successfully."
    
    def check_permission(self, username, permission, resource="*"):
        """Check permission using matrix + resource-level checks."""
        if username not in self.users:
            write_audit_event(
                actor=username,
                action="authorize",
                resource=resource,
                outcome="denied",
                details={"permission": permission, "reason": "unknown_user"},
                auth_provider="unknown"
            )
            return False
        
        user_role = self.users[username]["role"]
        if user_role not in self.ROLES:
            write_audit_event(
                actor=username,
                action="authorize",
                resource=resource,
                outcome="denied",
                details={"permission": permission, "reason": "invalid_role", "role": user_role},
                auth_provider=self.users[username].get("auth_provider", "local")
            )
            return False
        
        resource = str(resource or "*")
        permission = str(permission)
        rules = self.PERMISSION_MATRIX.get(permission)
        
        allowed = False
        if rules:
            # Exact resource match, wildcard selectors, and global fallback
            for selector, roles in rules.items():
                selector = str(selector)
                if selector == "*" and user_role in roles:
                    allowed = True
                elif selector.endswith("*") and resource.startswith(selector[:-1]) and user_role in roles:
                    allowed = True
                elif selector == resource and user_role in roles:
                    allowed = True
                if allowed:
                    break
        else:
            # Backward compatibility fallback to legacy role-permission list
            allowed = permission in self.ROLES[user_role]
        
        if not allowed:
            write_audit_event(
                actor=username,
                action="authorize",
                resource=resource,
                outcome="denied",
                details={"permission": permission, "role": user_role},
                auth_provider=self.users[username].get("auth_provider", "local")
            )
        
        return allowed

# Data handling classes
# Slice C Wave 1: DataProcessor
from app.ingestion.data_processor import DataProcessor

# Slice C Wave 1: detectors
from app.models.detectors import (
    AnomalyDetectionModel,
    EnhancedAnomalyDetectionModel,
    ONLINE_LEARNING_AVAILABLE,
    OnlineLearningAnomalyDetector,
    PROPHET_AVAILABLE,
    RIVER_AVAILABLE,
    TENSORFLOW_AVAILABLE,
)

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Ensure minimum size to prevent negative dimensions
        width = max(width, 1.0)
        height = max(height, 1.0)
        dpi = max(dpi, 50)
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
    
    def resizeEvent(self, event):
        """Override resize event to prevent negative figure sizes"""
        try:
            if event is not None:
                # Get widget size
                size = event.size()
                width = max(size.width(), 100)  # Minimum 100 pixels
                height = max(size.height(), 100)  # Minimum 100 pixels
                
                # Convert to inches (assuming 100 DPI)
                width_inches = max(width / 100.0, 1.0)
                height_inches = max(height / 100.0, 1.0)
                
                # Safely resize the figure
                self.figure.set_size_inches(width_inches, height_inches, forward=False)
                self.draw_idle()
            
            # Call parent resize event
            super().resizeEvent(event)
            
        except Exception as e:
            # Log the error but don't crash the application
            logger.warning(f"Error in canvas resize event: {str(e)}")
            # Ensure minimum figure size
            try:
                self.figure.set_size_inches(5.0, 4.0, forward=False)
                self.draw_idle()
            except:
                pass  # If this fails too, just continue

class SafeFigureCanvas(FigureCanvas):
    """A safer version of FigureCanvas that prevents resize crashes"""
    
    def __init__(self, figure):
        super().__init__(figure)
        self.setMinimumSize(200, 150)  # Set minimum widget size
    
    def resizeEvent(self, event):
        """Override resize event to prevent negative figure sizes"""
        try:
            if event is not None:
                # Get widget size with safety bounds
                size = event.size()
                width = max(size.width(), 200)  # Minimum 200 pixels
                height = max(size.height(), 150)  # Minimum 150 pixels
                
                # Convert to inches with safety bounds
                dpi = self.figure.dpi
                width_inches = max(width / dpi, 2.0)  # Minimum 2 inches
                height_inches = max(height / dpi, 1.5)  # Minimum 1.5 inches
                
                # Safely resize the figure
                self.figure.set_size_inches(width_inches, height_inches, forward=False)
                self.draw_idle()
            
            # Call parent resize event
            super().resizeEvent(event)
            
        except Exception as e:
            # Log the error but don't crash the application
            logger.warning(f"Error in safe canvas resize event: {str(e)}")
            # Ensure safe minimum figure size
            try:
                self.figure.set_size_inches(8.0, 6.0, forward=False)
                self.draw_idle()
            except:
                pass  # If this fails too, just continue

# Worker thread for background tasks
class Worker(QThread):
    finished = pyqtSignal(bool, str, object)
    progress = pyqtSignal(int)
    log_chunk = pyqtSignal(str)
    
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self._console_bridge = None
        if task == "train_model":
            from app.ui.training_console import TrainingConsoleBridge

            self._console_bridge = TrainingConsoleBridge()
            self._console_bridge.chunk.connect(self.log_chunk)
        
    def run(self):
        try:
            if self.task == "train_model":
                model = self.args[0]
                data = self.args[1]
                from app.ui.training_console import capture_stdout

                if self._console_bridge is not None:
                    with capture_stdout(self._console_bridge):
                        result, message = model.train(data, **self.kwargs)
                else:
                    result, message = model.train(data, **self.kwargs)
                self.finished.emit(result, message, model)
                
            elif self.task == "predict":
                model = self.args[0]
                data = self.args[1]
                scores, anomalies = model.predict(data)
                self.finished.emit(True, "Prediction completed", (scores, anomalies))
                
            elif self.task == "preprocess":
                processor = self.args[0]
                result, message = processor.preprocess_data(**self.kwargs)
                self.finished.emit(result, message, processor.preprocessed_data)
                
            elif self.task == "advanced_preprocess":
                processor = self.args[0]
                result, message = processor.advanced_preprocess_data(**self.kwargs)
                self.finished.emit(result, message, processor.preprocessed_data)
                
            elif self.task == "save_model":
                model = self.args[0]
                filepath = self.args[1]
                metadata = self.kwargs.get("metadata", {})
                result, message = model.save(filepath)
                
                # Save metadata if successful
                if result and "model_manager" in self.kwargs:
                    model_manager = self.kwargs["model_manager"]
                    model_manager.save_model_metadata(
                        os.path.basename(filepath),
                        model.model_type,
                        metadata.get("parameters", {}),
                        metadata.get("metrics", {}),
                        filepath
                    )
                    
                self.finished.emit(result, message, filepath)
                
            elif self.task == "load_model":
                filepath = self.args[0]
                result, message, model = self._load_model_blocking(filepath)
                self.finished.emit(result, message, model)
                
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            self.finished.emit(False, f"Error in worker thread: {str(e)}", None)
    
    def _load_model_blocking(self, file_name):
        """Load model in worker thread without GUI updates"""
        try:
            if not os.path.exists(file_name):
                return False, f"Model file not found: {file_name}", None
            
            # Check file extension to determine loading method
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # Initialize model load status
            load_success = False
            error_message = ""
            model = None
            
            # Try the best loading approach for the file type
            if file_ext == '.pkl':
                # Standard pickle format - our primary approach
                try:
                    model, message = AnomalyDetectionModel.load(file_name)
                    if model:
                        load_success = True
                    else:
                        error_message = message
                except Exception as e:
                    logger.error(f"Primary load method failed: {str(e)}")
                    error_message = f"Error loading model: {str(e)}"
                    
                    # Fallback: try alternative pickle loading
                    try:
                        raw_model = safe_pickle_load(file_name)
                            
                        # Check if it's a dict with our expected structure
                        if isinstance(raw_model, dict) and "model_type" in raw_model and "model" in raw_model:
                            model = AnomalyDetectionModel(raw_model["model_type"])
                            model.model = raw_model["model"]
                            model.scaler = raw_model.get("scaler", None)
                            model.feature_columns = raw_model.get("feature_columns", None)
                            
                            load_success = True
                            logger.info("Model loaded using fallback method")
                        elif hasattr(raw_model, "predict"):
                            # Direct model object (like scikit-learn model)
                            model = AnomalyDetectionModel("custom")
                            model.model = raw_model
                            
                            # Try to determine model type from its class
                            if "IsolationForest" in str(raw_model.__class__):
                                model.model_type = "isolation_forest"
                            elif "LocalOutlierFactor" in str(raw_model.__class__):
                                model.model_type = "lof"
                                
                            load_success = True
                    except Exception as fallback_error:
                        logger.error(f"Fallback loading also failed: {str(fallback_error)}")
                        
            elif file_ext == '.h5':
                # TensorFlow/Keras model format
                try:
                    # Check if TensorFlow is available
                    import_error = None
                    try:
                        import tensorflow as tf
                    except ImportError as e:
                        import_error = e
                    
                    if import_error:
                        error_message = "TensorFlow is required to load .h5 models but is not installed."
                        logger.error(error_message)
                    else:
                        # Try to load the model
                        try:
                            # Check if there's a companion .pkl file with metadata
                            meta_file = file_name.replace('.h5', '.pkl')
                            if os.path.exists(meta_file):
                                # Load metadata first
                                with open(meta_file, 'rb') as f:
                                    model_data = safe_pickle_load(filepath)
                                    
                                model = AnomalyDetectionModel(model_data.get("model_type", "deep_learning"))
                                model.scaler = model_data.get("scaler")
                                model.feature_columns = model_data.get("feature_columns")
                                model.reconstruction_error_threshold = model_data.get("reconstruction_error_threshold")
                                
                                # Then load the actual model
                                model.model = tf.keras.models.load_model(file_name)
                            else:
                                # No metadata, load just the model
                                keras_model = tf.keras.models.load_model(file_name)
                                
                                model = AnomalyDetectionModel("deep_learning")
                                model.model = keras_model
                                
                            load_success = True
                            logger.info(f"Deep learning model loaded from {file_name}")
                        except Exception as tf_error:
                            error_message = f"Error loading TensorFlow model: {str(tf_error)}"
                            logger.error(error_message)
                except Exception as e:
                    error_message = f"Error loading .h5 model: {str(e)}"
                    logger.error(error_message)
                    
            elif file_ext == '.joblib':
                # Try to load with joblib
                try:
                    import joblib
                    model_object = joblib.load(file_name)
                    
                    # Handle different types of saved joblib objects
                    if isinstance(model_object, dict) and "model_type" in model_object:
                        # Our custom format
                        model = AnomalyDetectionModel(model_object.get("model_type", "isolation_forest"))
                        model.model = model_object.get("model")
                        model.feature_columns = model_object.get("feature_columns")
                        model.scaler = model_object.get("scaler")
                        
                        load_success = True
                    elif hasattr(model_object, "predict") and hasattr(model_object, "fit"):
                        # Direct scikit-learn model object
                        model = AnomalyDetectionModel()
                        model.model = model_object
                        
                        load_success = True
                except ImportError:
                    error_message = "joblib is required to load .joblib files but is not installed."
                    logger.error(error_message)
                except Exception as e:
                    error_message = f"Error loading joblib model: {str(e)}"
                    logger.error(error_message)
            else:
                error_message = f"Unsupported file format: {file_ext}"
                logger.error(error_message)
            
            if load_success:
                return True, f"Model successfully loaded from {file_name}", model
            else:
                return False, error_message or f"Failed to load model from {file_name}", None
                
        except Exception as e:
            logger.error(f"Unexpected error loading model: {str(e)}")
            return False, f"Unexpected error loading model: {str(e)}", None




class ModelLoadingDialog(QDialog):
    """Progress dialog for model loading operations"""
    
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.worker = None
        self.model = None
        self.success = False
        self.message = ""
        
        self.setWindowTitle("Loading Model")
        self.setModal(True)
        self.setFixedSize(400, 150)
        
        # Set up UI
        layout = QVBoxLayout()
        
        # Info label
        self.info_label = QLabel(f"Loading model from:\n{os.path.basename(filepath)}")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)
        
        # Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_loading)
        layout.addWidget(self.cancel_button)
        
        self.setLayout(layout)
        
        # Start loading immediately
        QTimer.singleShot(100, self.start_loading)
    
    def start_loading(self):
        """Start the model loading in a separate thread"""
        self.status_label.setText("Loading model...")
        
        # Create and start worker thread
        self.worker = Worker("load_model", self.filepath)
        self.worker.finished.connect(self.on_loading_finished)
        self.worker.start()
    
    def on_loading_finished(self, success, message, model):
        """Handle completion of model loading"""
        self.success = success
        self.message = message
        self.model = model
        
        if success:
            self.status_label.setText("Model loaded successfully!")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            QTimer.singleShot(500, self.accept)  # Close after brief delay
        else:
            self.status_label.setText(f"Failed: {message}")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.cancel_button.setText("Close")
    
    def cancel_loading(self):
        """Cancel the loading operation"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        self.reject()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        super().closeEvent(event)

class ModelParameterDialog(QDialog):
    """Dialog window for configuring parameters for multiple models"""
    def __init__(self, selected_models, parent=None):
        super().__init__(parent)
        self.selected_models = selected_models
        self.model_param_widgets = {}
        
        self.setWindowTitle("Configure Model Parameters")
        self.setMinimumSize(700, 600)
        self.setModal(True)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel(f"Configure Parameters for {len(selected_models)} Selected Model(s)")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Info label
        if len(selected_models) > 0:
            info_label = QLabel(f"Selected: {', '.join(selected_models)}")
            info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
            info_label.setWordWrap(True)
            main_layout.addWidget(info_label)
        
        # Scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_widget = QWidget()
        self.params_layout = QVBoxLayout(scroll_widget)
        self.params_layout.setSpacing(10)
        
        # Create parameter sections for each model
        for model_name in selected_models:
            self._create_model_section(model_name)
        
        self.params_layout.addStretch()
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        # Button box
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Reset to defaults button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_to_defaults)
        reset_button.setMinimumWidth(120)
        button_layout.addWidget(reset_button)
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setMinimumWidth(100)
        button_layout.addWidget(cancel_button)
        
        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setDefault(True)
        ok_button.setMinimumWidth(100)
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
        """)
        button_layout.addWidget(ok_button)
        
        main_layout.addLayout(button_layout)
    
    def _create_model_section(self, model_name):
        """Create parameter section for a model"""
        model_type = to_internal_model_type(model_name)
        
        # Create group box
        group_box = QGroupBox(f"{model_name}")
        group_box.setCheckable(True)
        group_box.setChecked(True)
        group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #ccc;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
            }
        """)
        
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setContentsMargins(15, 20, 15, 15)
        
        # Create parameters
        param_widgets = self._create_model_parameters(model_type, model_name)
        
        for label_text, widget in param_widgets:
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: normal; color: #333;")
            form_layout.addRow(label, widget)
        
        group_box.setLayout(form_layout)
        self.params_layout.addWidget(group_box)
        
        # Store widgets
        self.model_param_widgets[model_name] = param_widgets
    
    def _create_model_parameters(self, model_type, model_name):
        """Create parameter widgets for a model type"""
        param_widgets = []
        
        # Check for dependencies
        if model_type in ["autoencoder", "lstm", "gru"] and not TENSORFLOW_AVAILABLE:
            warning_label = QLabel("⚠️ TensorFlow not installed")
            warning_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
            install_label = QLabel("Install: pip install tensorflow")
            install_label.setStyleSheet("color: #1976d2; font-style: italic;")
            param_widgets.append(("Warning:", warning_label))
            param_widgets.append(("", install_label))
            return param_widgets
        
        if model_type == "prophet" and not PROPHET_AVAILABLE:
            warning_label = QLabel("⚠️ Prophet not installed")
            warning_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
            param_widgets.append(("Warning:", warning_label))
            return param_widgets
        
        # Create parameters based on model type
        if model_type == "isolation_forest":
            n_estimators = QSpinBox()
            n_estimators.setRange(10, 1000)
            n_estimators.setValue(100)
            n_estimators.setMinimumWidth(100)
            param_widgets.append(("Estimators:", n_estimators))
            
            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setValue(0.1)
            contamination.setSingleStep(0.01)
            contamination.setMinimumWidth(100)
            param_widgets.append(("Contamination:", contamination))
            
        elif model_type == "local_outlier_factor":
            n_neighbors = QSpinBox()
            n_neighbors.setRange(1, 100)
            n_neighbors.setValue(20)
            n_neighbors.setMinimumWidth(100)
            param_widgets.append(("Neighbors:", n_neighbors))
            
            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setValue(0.1)
            contamination.setSingleStep(0.01)
            contamination.setMinimumWidth(100)
            param_widgets.append(("Contamination:", contamination))
            
        elif model_type in ["autoencoder", "lstm", "gru"]:
            epochs = QSpinBox()
            epochs.setRange(1, 1000)
            epochs.setValue(50)
            epochs.setMinimumWidth(100)
            param_widgets.append(("Epochs:", epochs))
            
            batch_size = QSpinBox()
            batch_size.setRange(1, 1024)
            batch_size.setValue(32)
            batch_size.setMinimumWidth(100)
            param_widgets.append(("Batch Size:", batch_size))
            
            validation_split = QDoubleSpinBox()
            validation_split.setRange(0.01, 0.99)
            validation_split.setValue(0.2)
            validation_split.setSingleStep(0.01)
            validation_split.setMinimumWidth(100)
            param_widgets.append(("Validation Split:", validation_split))
            
            patience = QSpinBox()
            patience.setRange(1, 100)
            patience.setValue(5)
            patience.setMinimumWidth(100)
            param_widgets.append(("Patience:", patience))
            
            learning_rate = QDoubleSpinBox()
            learning_rate.setRange(0.0001, 1.0)
            learning_rate.setValue(0.001)
            learning_rate.setSingleStep(0.0001)
            learning_rate.setDecimals(4)
            learning_rate.setMinimumWidth(100)
            param_widgets.append(("Learning Rate:", learning_rate))
            
            if model_type in ["lstm", "gru"]:
                sequence_length = QSpinBox()
                sequence_length.setRange(1, 100)
                sequence_length.setValue(10)
                sequence_length.setMinimumWidth(100)
                param_widgets.append(("Sequence Length:", sequence_length))
                
        elif model_type == "xgboost":
            n_estimators = QSpinBox()
            n_estimators.setRange(10, 1000)
            n_estimators.setValue(100)
            n_estimators.setMinimumWidth(100)
            param_widgets.append(("Estimators:", n_estimators))
            
            max_depth = QSpinBox()
            max_depth.setRange(1, 20)
            max_depth.setValue(6)
            max_depth.setMinimumWidth(100)
            param_widgets.append(("Max Depth:", max_depth))
            
            learning_rate = QDoubleSpinBox()
            learning_rate.setRange(0.01, 1.0)
            learning_rate.setValue(0.1)
            learning_rate.setSingleStep(0.01)
            learning_rate.setMinimumWidth(100)
            param_widgets.append(("Learning Rate:", learning_rate))
            
            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setValue(0.1)
            contamination.setSingleStep(0.01)
            contamination.setMinimumWidth(100)
            param_widgets.append(("Contamination:", contamination))
            
        elif model_type == "iqr_(interquartile_range)":
            iqr_factor = QDoubleSpinBox()
            iqr_factor.setRange(0.5, 5.0)
            iqr_factor.setValue(1.5)
            iqr_factor.setSingleStep(0.1)
            iqr_factor.setMinimumWidth(100)
            param_widgets.append(("IQR Factor:", iqr_factor))
            
        elif model_type == "z-score":
            threshold = QDoubleSpinBox()
            threshold.setRange(1.0, 5.0)
            threshold.setValue(3.0)
            threshold.setSingleStep(0.1)
            threshold.setMinimumWidth(100)
            param_widgets.append(("Threshold:", threshold))
            
        elif model_type == "prophet":
            yearly = QCheckBox()
            yearly.setChecked(True)
            param_widgets.append(("Yearly Seasonality:", yearly))
            
            weekly = QCheckBox()
            weekly.setChecked(True)
            param_widgets.append(("Weekly Seasonality:", weekly))
            
            daily = QCheckBox()
            daily.setChecked(True)
            param_widgets.append(("Daily Seasonality:", daily))
            
            changepoint = QDoubleSpinBox()
            changepoint.setRange(0.001, 1.0)
            changepoint.setValue(0.05)
            changepoint.setSingleStep(0.001)
            changepoint.setDecimals(3)
            changepoint.setMinimumWidth(100)
            param_widgets.append(("Changepoint Scale:", changepoint))
        
        # Default message for models without specific parameters
        if not param_widgets:
            default_label = QLabel("✓ Using default parameters")
            default_label.setStyleSheet("color: #4caf50; font-style: italic;")
            param_widgets.append(("Configuration:", default_label))
        
        return param_widgets
    
    def reset_to_defaults(self):
        """Reset all parameters to default values"""
        # Recreate all sections
        for i in reversed(range(self.params_layout.count())):
            widget = self.params_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        self.model_param_widgets = {}
        
        for model_name in self.selected_models:
            self._create_model_section(model_name)
        
        self.params_layout.addStretch()
    
    def get_parameter_values(self):
        """Extract parameter values from widgets and return as dictionary"""
        param_values = {}
        
        for model_name, widgets in self.model_param_widgets.items():
            model_type = to_internal_model_type(model_name)
            model_params = {}
            
            # Extract values based on widget type
            for label_text, widget in widgets:
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    # Extract numeric values
                    if "Estimators" in label_text:
                        model_params['n_estimators'] = widget.value()
                    elif "Neighbors" in label_text:
                        model_params['n_neighbors'] = widget.value()
                    elif "Contamination" in label_text:
                        model_params['contamination'] = widget.value()
                    elif "Epochs" in label_text:
                        model_params['epochs'] = widget.value()
                    elif "Batch Size" in label_text:
                        model_params['batch_size'] = widget.value()
                    elif "Validation" in label_text:
                        model_params['validation_split'] = widget.value()
                    elif "Patience" in label_text:
                        model_params['patience'] = widget.value()
                    elif "Learning Rate" in label_text:
                        model_params['learning_rate'] = widget.value()
                    elif "Sequence" in label_text:
                        model_params['sequence_length'] = widget.value()
                    elif "Max Depth" in label_text:
                        model_params['max_depth'] = widget.value()
                    elif "IQR Factor" in label_text:
                        model_params['iqr_factor'] = widget.value()
                    elif "Threshold" in label_text:
                        model_params['threshold'] = widget.value()
                    elif "Changepoint" in label_text:
                        model_params['changepoint_prior_scale'] = widget.value()
                elif isinstance(widget, QCheckBox):
                    # Extract boolean values
                    if "Yearly" in label_text:
                        model_params['yearly_seasonality'] = widget.isChecked()
                    elif "Weekly" in label_text:
                        model_params['weekly_seasonality'] = widget.isChecked()
                    elif "Daily" in label_text:
                        model_params['daily_seasonality'] = widget.isChecked()
            
            param_values[model_name] = model_params
        
        return param_values
    
    def accept(self):
        """Override accept to extract values before dialog closes"""
        # Extract values before widgets are destroyed
        self.parameter_values = self.get_parameter_values()
        super().accept()

    @staticmethod
    def extract_parameters_from_widgets(model_name, widgets):
        """Extract parameter dict from a list of (label, widget) pairs."""
        helper = ModelParameterDialog([model_name])
        helper.model_param_widgets = {model_name: widgets}
        return helper.get_parameter_values().get(model_name, {})


class AddModelDialog(QDialog):
    """Add a model with live configuration for the selected model type."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Model")
        self.setMinimumSize(540, 520)
        self.model_param_widgets = []
        self.selected_model_name = None
        self._param_builder = ModelParameterDialog([], self)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Model Type:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_supported_model_names())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        type_row.addWidget(self.model_combo, 1)
        layout.addLayout(type_row)

        self.config_group = QGroupBox("Model Configuration")
        config_layout = QVBoxLayout(self.config_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.params_container = QWidget()
        self.params_form = QFormLayout(self.params_container)
        self.params_form.setSpacing(10)
        self.params_form.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(self.params_container)
        config_layout.addWidget(scroll)
        layout.addWidget(self.config_group, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("Add Model")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._on_model_changed(self.model_combo.currentText())

    def _clear_params_form(self):
        while self.params_form.rowCount() > 0:
            label_item, field_item = self.params_form.takeRow(0)
            for layout_item in (label_item, field_item):
                if layout_item is None:
                    continue
                widget = layout_item.widget()
                if widget is not None:
                    widget.deleteLater()

    def _on_model_changed(self, model_name):
        self.selected_model_name = model_name
        self._clear_params_form()
        model_type = to_internal_model_type(model_name)
        self.model_param_widgets = self._param_builder._create_model_parameters(model_type, model_name)
        for label_text, widget in self.model_param_widgets:
            label = QLabel(label_text) if label_text else None
            if label is not None:
                label.setStyleSheet("font-weight: normal; color: #333;")
                self.params_form.addRow(label, widget)
            else:
                self.params_form.addRow(widget)
        self.config_group.setTitle(f"Model Configuration — {model_name}")

    def get_model_parameters(self):
        if not self.selected_model_name:
            return {}
        return ModelParameterDialog.extract_parameters_from_widgets(
            self.selected_model_name, self.model_param_widgets
        )

class LoginDialog(QDialog):
    def __init__(self, user_manager, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.authenticated = False
        self.username = None
        self.role = None
        self.auth_provider = "local"
        
        self.setWindowTitle("SDA v4.0 — Sign In")
        self.setFixedSize(500, 450)
        self.setStyleSheet("""
            QDialog {
                background-color: white;
            }
            QLabel {
                color: #333;
                font-size: 14px;
                font-weight: bold;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                min-height: 30px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(8)  # Reduced spacing
        layout.setContentsMargins(20, 20, 20, 20)  # Reduced margins
        
        # Logo (bundled asset — do not depend on a Desktop path)
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        try:
            from app.ui.login_dialog import load_azercosmos_logo

            logo_pixmap = load_azercosmos_logo(width=400, height=100)
        except Exception:
            logo_path = Path(__file__).resolve().parent / "app" / "ui" / "assets" / "azercosmos-logo.png"
            logo_pixmap = QPixmap(str(logo_path))
            if not logo_pixmap.isNull():
                logo_pixmap = logo_pixmap.scaled(400, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap)
        else:
            logo_label.setText("Azercosmos")
            logo_label.setStyleSheet("font-size: 20px; font-weight: 700; color: #2f6fad;")
        layout.addWidget(logo_label)
        
        # Add some spacing
        layout.addSpacing(10)  # Reduced spacing
        
        # Username field
        username_label = QLabel("Username:")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        layout.addWidget(username_label)
        layout.addWidget(self.username_input)
        
        # Password field
        password_label = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(password_label)
        layout.addWidget(self.password_input)
        
        # Add some spacing
        layout.addSpacing(10)  # Reduced spacing
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)  # Reduced spacing
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.attempt_login)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: red; font-size: 14px; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
    def attempt_login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if not username or not password:
            self.status_label.setText("Username and password required")
            return
        
        authenticated, role, message, require_password_change, auth_context = self.user_manager.authenticate_user(username, password)
        if authenticated:
            if require_password_change:
                changed, change_msg = self._enforce_password_change(username)
                if not changed:
                    self.status_label.setText(change_msg)
                    return
            
            self.authenticated = True
            self.username = (auth_context or {}).get("resolved_username", username)
            self.role = role
            self.auth_provider = (auth_context or {}).get("auth_provider", "local")
            self.accept()
        else:
            self.status_label.setText(message or "Invalid username or password")
            # Slow down brute force attempts
            time.sleep(1)
    
    def _enforce_password_change(self, username):
        """Force user to set a new password before entering the app."""
        QMessageBox.information(
            self,
            "Password Update Required",
            "For security, you must set a new password before continuing."
        )
        
        while True:
            new_password, ok = QInputDialog.getText(
                self,
                "Set New Password",
                "Enter a new strong password:",
                QLineEdit.Password
            )
            if not ok:
                return False, "Password change is required to continue."
            
            confirm_password, ok_confirm = QInputDialog.getText(
                self,
                "Confirm Password",
                "Re-enter your new password:",
                QLineEdit.Password
            )
            if not ok_confirm:
                return False, "Password change is required to continue."
            
            if new_password != confirm_password:
                QMessageBox.warning(self, "Mismatch", "Passwords do not match. Please try again.")
                continue
            
            success, update_msg = self.user_manager.update_password(username, new_password)
            if success:
                QMessageBox.information(self, "Password Updated", "Password changed successfully.")
                return True, update_msg
            
            QMessageBox.warning(self, "Invalid Password", update_msg)
    
    # Add keyPressEvent to handle Enter key
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.attempt_login()
        else:
            super().keyPressEvent(event)

# Main application window
class SecureAnomalyDetectionTool(QMainWindow):
    def debug_status(self, context=""):
        """Debug helper to log current status"""
        if not hasattr(self, 'debug_mode') or not self.debug_mode:
            return
            
        logger.debug(f"\nDEBUG STATUS - {context}")
        logger.debug("-" * 40)
        
        # Check critical attributes
        critical_attrs = [
            'health_monitoring_active',
            'health_monitoring_timer',
            'monitoring_folder_path',
            'agent1_status_indicator',
            'agent2_status_indicator',
            'agent3_status_indicator',
            'start_monitoring_btn',
            'stop_monitoring_btn'
        ]
        
        for attr in critical_attrs:
            exists = hasattr(self, attr)
            value = getattr(self, attr, "NOT FOUND")
            logger.debug(f"{attr}: {'EXISTS' if exists else 'MISSING'} = {value}")
        
        logger.debug("-" * 40)
        
    def __init__(self):
        super().__init__()
        
        logger.info("Initializing SecureAnomalyDetectionTool")
        app_boot_span = OBSERVABILITY.start_span("app.bootstrap") if OBSERVABILITY else None
        if OBSERVABILITY:
            OBSERVABILITY.inc_counter("app_starts_total", labels={"status": "attempt"})
        
        # Initialize debugging flags
        self.debug_mode = True
        self.last_error = None
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        logger.info("Theme manager initialized")
        
        # Initialize managers
        self.user_manager = UserManager()
        self.data_processor = DataProcessor()
        self.model = None  # Keep for backward compatibility
        self.trained_models = {}  # Dictionary to store multiple trained models
        self.current_username = None
        self.current_role = None
        self.current_auth_provider = "local"
        self.current_session_id = None
        # Hide legacy top-level Data/Visualization/Health tabs; custom workspaces live under Analysis / ML.
        self.custom_tabs_only_mode = True
        
        # Initialize auto-load tracking
        self.last_processed_file = None
        self.last_processed_time = None
        self.auto_loading_timer = None
        
        # Initialize health monitoring timer
        self.health_monitoring_timer = QTimer()
        self.health_monitoring_timer.timeout.connect(self.collect_health_data)
        
        # Initialize health monitoring agents with improved thresholds
        self.health_agent1 = {
            'data_buffer': [],
            'score_buffer': [],
            'last_update': None,
            'status': 'Initializing',
            'threshold': 0.95,  # More conservative LSTM threshold
            'min_anomaly_spacing': 10,  # Minimum points between anomalies
            'window_size': 50,  # Window size for smoothing
            'weight': 0.4  # LSTM weight in ensemble
        }
        
        self.health_agent2 = {
            'data_buffer': [],
            'score_buffer': [],
            'last_update': None,
            'status': 'Initializing',
            'threshold': 0.92,  # More conservative CNN threshold
            'min_anomaly_spacing': 10,
            'window_size': 50,
            'weight': 0.4  # CNN weight in ensemble
        }
        
        self.health_agent3 = {
            'data_buffer': [],
            'score_buffer': [],
            'last_update': None,
            'status': 'Initializing',
            'threshold': 0.90,  # More conservative ARIMA threshold
            'min_anomaly_spacing': 10,
            'window_size': 50,
            'weight': 0.2  # ARIMA weight in ensemble
        }
        
        # Ensemble parameters
        self.ensemble_threshold = 0.93  # Higher threshold for ensemble decisions
        self.smoothing_window = 5  # Window for smoothing anomaly scores
        self.min_anomaly_duration = 3  # Minimum consecutive points for anomaly
        
        # Initialize health monitoring variables
        self.health_monitoring_active = False
        self.health_data_buffer = []
        self.health_score_buffer = []
        self.health_threshold = 0.85
        self.monitoring_window_size = 100
        
        # Initialize the model registry
        self.model_registry = ModelRegistry(data_dir=DATA_DIR)
        
        # Initialize customizable tabs configuration manager
        self.tab_config_manager = TabConfigurationManager()
        self.custom_tabs = {}  # Dictionary to track custom tab widgets: {tab_id: widget}
        
        # Initialize security settings
        if _APP_SETTINGS is not None:
            self.lock_timeout_ms = int(_APP_SETTINGS.security.lock_timeout_ms)
            self.session_timeout_ms = int(_APP_SETTINGS.security.session_timeout_ms)
        else:
            self.lock_timeout_ms = 900000  # Default 15 minutes (900000 milliseconds)
            self.session_timeout_ms = 28800000  # Absolute session timeout: 8 hours
        
        # Initialize alert policy manager (dedup/cooldown/routing/escalation/audit)
        policy_manager_cls = AlertPolicyManager
        self.alert_policy_manager = policy_manager_cls(
            db_path=TELEMETRY_DB_FILE,
            config_loader=load_alert_routing_config,
            dispatchers={
                "email": self._dispatch_email_route,
                "slack": self._dispatch_slack_route,
                "webhook": self._dispatch_webhook_route,
                "pagerduty": self._dispatch_pagerduty_route
            }
        )
        self.service_layer = AppServiceLayer(
            user_manager=self.user_manager,
            alert_policy_manager=self.alert_policy_manager,
            audit_writer=write_audit_event
        )
        
        # Initialize model parameter attributes
        self.n_estimators_spin = None
        self.contamination_spin = None
        self.n_neighbors_spin = None
        
        # Authenticate user before showing main window
        login_dialog = LoginDialog(self.user_manager)
        result = login_dialog.exec_()
        
        if result == QDialog.Accepted and login_dialog.authenticated:
            self.current_username = login_dialog.username
            self.current_role = login_dialog.role
            self.current_auth_provider = getattr(login_dialog, "auth_provider", "local")
            self.current_session_id = uuid.uuid4().hex
            write_audit_event(
                actor=self.current_username,
                action="session_start",
                resource="auth/session",
                outcome="success",
                details={"role": self.current_role, "session_id": self.current_session_id},
                auth_provider=self.current_auth_provider
            )
            self.initUI()
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("app_starts_total", labels={"status": "success"})
                OBSERVABILITY.end_span(app_boot_span, status="ok", attributes={"user": self.current_username})
        else:
            # Exit if login failed
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("app_starts_total", labels={"status": "cancelled"})
                OBSERVABILITY.end_span(app_boot_span, status="error", attributes={"reason": "login_not_completed"})
            sys.exit()
    
    def initUI(self):
        # Main window settings
        self.setWindowTitle('SDA v4.0')
        self.setGeometry(100, 100, 1200, 800)
        
        # Set minimum size to prevent figure resize issues
        self.setMinimumSize(800, 600)
        
        # Create central widget: Spaceit-style dark nav + STDMS pages
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        from app.ui.sidebar import OpsNavSidebar

        self.ops_nav = OpsNavSidebar()
        self.ops_nav.set_user(self.current_username, self.current_role)
        self.ops_nav.navigate.connect(self._on_ops_nav)
        main_layout.addWidget(self.ops_nav)
        
        # Create tabs (tab bar hidden — navigation is the sidebar)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(False)  # We'll handle closing custom tabs ourselves
        self.tabs.tabBar().hide()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(
            "QTabWidget::pane { border: none; background: #f4f6f9; }"
        )
        
        # Enable context menu for tabs (for delete option)
        self.tabs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabs.customContextMenuRequested.connect(self.on_tab_context_menu)
        
        # Only show tabs based on user permissions
        self.dashboard_tab = QWidget()
        self.tabs.addTab(self.dashboard_tab, "Home")
        tab_policy = None
        if modular_build_tab_visibility_policy is not None:
            try:
                tab_policy = modular_build_tab_visibility_policy(self.service_layer, self.current_username)
            except Exception:
                tab_policy = None
        
        can_import_data = (
            bool(tab_policy.can_import_data)
            if tab_policy is not None
            else self.service_layer.authorize(self.current_username, "import_data", resource="data_source:*")
        )
        if can_import_data and not getattr(self, "custom_tabs_only_mode", False):
            self.data_tab = QWidget()
            self.tabs.addTab(self.data_tab, "Data Import")
            self.setup_data_tab()
        
        can_process_data = (
            bool(tab_policy.can_process_data)
            if tab_policy is not None
            else self.service_layer.authorize(self.current_username, "process_data", resource="analysis")
        )
        if can_process_data and not getattr(self, "custom_tabs_only_mode", False):
            self.analysis_tab = QWidget()
            self.tabs.addTab(self.analysis_tab, "Analysis / ML")
            self.setup_analysis_tab()
        
        if can_process_data and not getattr(self, "custom_tabs_only_mode", False):
            self.visualization_tab = QWidget()
            self.tabs.addTab(self.visualization_tab, "Visualization")
            self.setup_visualization_tab()
        
        if can_process_data and not getattr(self, "custom_tabs_only_mode", False):
            self.health_tab = QWidget()
            self.tabs.addTab(self.health_tab, "System Health")
            self.setup_health_tab()
        
        can_manage_users = (
            bool(tab_policy.can_manage_users)
            if tab_policy is not None
            else self.service_layer.authorize(self.current_username, "manage_users", resource="admin/users")
        )
        if can_manage_users:
            self.admin_tab = QWidget()
            self.tabs.addTab(self.admin_tab, "Administration")
            self.setup_admin_tab()
        
        # Setup the dashboard tab as default
        self.setup_dashboard_tab()
        
        main_layout.addWidget(self.tabs, 1)
        
        # Status bar setup
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(
            f"Logged in as {self.current_username} ({self.current_role}) via {self.current_auth_provider}"
        )
        
        # Security timer - locks app after inactivity
        self.inactivity_timer = QTimer(self)
        self.inactivity_timer.timeout.connect(self.lock_application)
        self.reset_inactivity_timer()
        
        # Escalation loop for unacknowledged incidents
        self.alert_escalation_timer = QTimer(self)
        self.alert_escalation_timer.timeout.connect(self._run_alert_escalation_check)
        escalation_interval_ms = 60000
        if _APP_SETTINGS is not None:
            escalation_interval_ms = int(_APP_SETTINGS.alerting.escalation_check_ms)
        self.alert_escalation_timer.start(escalation_interval_ms)
        
        # Absolute session timeout requires re-authentication even with activity
        self.session_timer = QTimer(self)
        self.session_timer.setSingleShot(True)
        self.session_timer.timeout.connect(self.lock_application)
        self.start_session_timer()
        
        # Track activity globally to enforce inactivity lock reliably
        self.installEventFilter(self)
        
        # Connect events
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # Initialize legacy system health monitor only when that tab is enabled
        if not getattr(self, "custom_tabs_only_mode", False):
            from system_health_monitor import SystemHealthMonitor
            self.health_monitor = SystemHealthMonitor("telemetry_system")
        else:
            self.health_monitor = None
        
        # Sidebar replaces the old top toolbar (functions unchanged)
        self.setup_toolbar()
        for bar in self.findChildren(QToolBar):
            bar.setVisible(False)
        
        # Add "+" button for creating custom tabs (admin only)
        self.setup_custom_tabs_button()
        
        # Load existing custom tabs
        self.load_custom_tabs()
        self._refresh_ops_nav()
        if hasattr(self, "home_subtabs"):
            self.home_subtabs.currentChanged.connect(self._on_home_subtab_changed)

        # Phase 0: local Instrumentation Agent bridge (read-only, localhost:8765)
        self._start_instrumentation_agent()

        self.fleet_refresh_timer = QTimer(self)
        self.fleet_refresh_timer.timeout.connect(self.refresh_fleet_dashboard)
        self.fleet_refresh_timer.start(15000)
        self.refresh_fleet_dashboard()

    def _start_instrumentation_agent(self):
        """Expose tab snapshots to the local agent API and start Phase 1 monitor loop."""
        self.agent_bridge = None
        self.agent_loop = None
        try:
            from app.agent import AgentMonitorLoop, attach_agent_bridge

            self.agent_bridge = attach_agent_bridge(
                self, host="127.0.0.1", port=8765, enabled=True, allow_llm=True
            )
            if self.agent_bridge:
                self.agent_loop = AgentMonitorLoop(self.agent_bridge, parent=self)
                self.agent_loop.signals.refreshed.connect(self._on_agent_view_refreshed)
                self.agent_loop.signals.chat_reply.connect(self._on_agent_chat_reply)
                self.agent_loop.start(interval_seconds=60)
                self._refresh_agent_llm_status()
        except Exception as exc:
            logger.warning("Instrumentation agent not started: %s", exc)

    def _refresh_agent_llm_status(self):
        label = getattr(self, "agent_llm_status_label", None)
        loop = getattr(self, "agent_loop", None)
        if label is None:
            return
        if loop is None:
            self._set_agent_online_pill(False, model_name="")
            return
        try:
            st = loop.llm_status()
            model = str(st.get("model") or "").strip()
            if st.get("ollama_up"):
                self._set_agent_online_pill(True, model_name=model)
            else:
                self._set_agent_online_pill(False, model_name=model)
        except Exception:
            label.setText("status unknown")
            label.setStyleSheet("")

    def _agent_sessions_path(self):
        return os.path.join(DATA_DIR, "agent_chat_sessions.json")

    def _load_agent_sessions(self):
        path = self._agent_sessions_path()
        try:
            from app.agent.chat_render import migrate_transcript_to_messages

            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    cleaned = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        sid = str(item.get("id") or "").strip()
                        if not sid:
                            continue
                        messages = item.get("messages")
                        if not isinstance(messages, list):
                            messages = migrate_transcript_to_messages(
                                str(item.get("transcript") or "")
                            )
                        cleaned.append(
                            {
                                "id": sid,
                                "title": str(item.get("title") or "New chat").strip() or "New chat",
                                "messages": messages,
                                "transcript": str(item.get("transcript") or ""),
                                "updated_at": str(item.get("updated_at") or ""),
                            }
                        )
                    return cleaned
        except Exception as exc:
            logger.warning("Could not load agent chat sessions: %s", exc)
        return []

    def _save_agent_sessions(self):
        path = self._agent_sessions_path()
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._agent_sessions or [], fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("Could not save agent chat sessions: %s", exc)

    def _get_agent_session(self, session_id):
        for s in self._agent_sessions or []:
            if s.get("id") == session_id:
                return s
        return None

    def _init_agent_chat_sessions(self):
        self._agent_sessions = self._load_agent_sessions()
        if not self._agent_sessions:
            self._agent_sessions = [
                {
                    "id": str(uuid.uuid4()),
                    "title": "New chat",
                    "messages": [],
                    "transcript": "",
                    "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
            ]
            self._save_agent_sessions()
        self._refresh_agent_session_list()
        first_id = self._agent_sessions[0]["id"]
        self._select_agent_session(first_id, persist_current=False)

    def _refresh_agent_session_list(self, filter_text=None):
        if not hasattr(self, "agent_chat_session_list"):
            return
        if filter_text is None and hasattr(self, "agent_chat_search_input"):
            filter_text = self.agent_chat_search_input.text()
        needle = (filter_text or "").strip().lower()
        active = getattr(self, "_agent_active_session_id", None)
        self._agent_session_switching = True
        self.agent_chat_session_list.clear()
        select_row = -1
        for idx, session in enumerate(self._agent_sessions or []):
            title = str(session.get("title") or "New chat")
            if needle and needle not in title.lower():
                continue
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, session.get("id"))
            self.agent_chat_session_list.addItem(item)
            if session.get("id") == active:
                select_row = self.agent_chat_session_list.count() - 1
        if select_row >= 0:
            self.agent_chat_session_list.setCurrentRow(select_row)
        elif self.agent_chat_session_list.count() > 0 and active is None:
            self.agent_chat_session_list.setCurrentRow(0)
        self._agent_session_switching = False

    def _filter_agent_sessions(self, *_args):
        self._refresh_agent_session_list()

    def _sync_active_session_from_view(self, *, update_title_from_first_you=False):
        """Persist current chat messages into the active session."""
        from app.agent.chat_render import messages_to_plain_transcript

        session = self._get_agent_session(getattr(self, "_agent_active_session_id", None))
        if session is None:
            return
        messages = list(getattr(self, "_agent_chat_messages", None) or [])
        # Drop ephemeral thinking placeholder from persistence
        messages = [m for m in messages if not m.get("thinking")]
        session["messages"] = messages
        session["transcript"] = messages_to_plain_transcript(messages)
        session["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        if update_title_from_first_you or session.get("title") in ("", "New chat"):
            for msg in messages:
                if str(msg.get("role") or "").lower() not in ("you", "user", "operator"):
                    continue
                after = str(msg.get("text") or "").strip()
                if after:
                    session["title"] = after[:48] + ("…" if len(after) > 48 else "")
                    break
        self._save_agent_sessions()
        self._refresh_agent_session_list()

    def _render_agent_chat_view(self):
        """Rebuild QTextBrowser HTML from structured messages."""
        if not hasattr(self, "agent_view_text"):
            return
        from app.agent.chat_render import render_chat_document

        html_doc = render_chat_document(getattr(self, "_agent_chat_messages", None) or [])
        self.agent_view_text.setHtml(html_doc)
        self._scroll_agent_chat_to_end()

    def _select_agent_session(self, session_id, *, persist_current=True):
        from app.agent.chat_render import migrate_transcript_to_messages

        if persist_current and getattr(self, "_agent_active_session_id", None):
            self._sync_active_session_from_view()
        session = self._get_agent_session(session_id)
        if session is None:
            return
        self._agent_session_switching = True
        self._agent_active_session_id = session_id
        self._agent_thinking_line = None
        messages = session.get("messages")
        if not isinstance(messages, list):
            messages = migrate_transcript_to_messages(str(session.get("transcript") or ""))
            session["messages"] = messages
        self._agent_chat_messages = [dict(m) for m in messages]
        self._render_agent_chat_view()
        # Select matching list item
        if hasattr(self, "agent_chat_session_list"):
            for i in range(self.agent_chat_session_list.count()):
                item = self.agent_chat_session_list.item(i)
                if item and item.data(Qt.UserRole) == session_id:
                    self.agent_chat_session_list.setCurrentRow(i)
                    break
        self._agent_session_switching = False

    def _on_agent_session_item_changed(self, current, _previous):
        if getattr(self, "_agent_session_switching", False):
            return
        if current is None:
            return
        sid = current.data(Qt.UserRole)
        if not sid or sid == getattr(self, "_agent_active_session_id", None):
            return
        if getattr(self, "_agent_chat_busy", False):
            # Stay on active session while a reply is in flight
            self._agent_session_switching = True
            self._refresh_agent_session_list()
            self._agent_session_switching = False
            return
        self._select_agent_session(sid, persist_current=True)

    def _new_agent_session(self):
        if getattr(self, "_agent_chat_busy", False):
            return
        self._sync_active_session_from_view()
        session = {
            "id": str(uuid.uuid4()),
            "title": "New chat",
            "messages": [],
            "transcript": "",
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        self._agent_sessions.insert(0, session)
        self._save_agent_sessions()
        self._refresh_agent_session_list()
        self._select_agent_session(session["id"], persist_current=False)

    def _remove_agent_session(self):
        if getattr(self, "_agent_chat_busy", False):
            return
        item = None
        if hasattr(self, "agent_chat_session_list"):
            item = self.agent_chat_session_list.currentItem()
        sid = item.data(Qt.UserRole) if item else getattr(self, "_agent_active_session_id", None)
        if not sid:
            return
        self._agent_sessions = [s for s in (self._agent_sessions or []) if s.get("id") != sid]
        if not self._agent_sessions:
            self._agent_sessions = [
                {
                    "id": str(uuid.uuid4()),
                    "title": "New chat",
                    "transcript": "",
                    "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                }
            ]
        self._save_agent_sessions()
        self._agent_active_session_id = None
        self._refresh_agent_session_list()
        self._select_agent_session(self._agent_sessions[0]["id"], persist_current=False)

    def _clear_all_agent_sessions(self):
        if getattr(self, "_agent_chat_busy", False):
            return
        reply = QMessageBox.question(
            self,
            "Clear All Chats",
            "Remove all saved agent conversations?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._agent_sessions = [
            {
                "id": str(uuid.uuid4()),
                "title": "New chat",
                "transcript": "",
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
        ]
        self._save_agent_sessions()
        self._agent_active_session_id = None
        if hasattr(self, "agent_chat_search_input"):
            self.agent_chat_search_input.clear()
        self._refresh_agent_session_list()
        self._select_agent_session(self._agent_sessions[0]["id"], persist_current=False)

    def _scroll_agent_chat_to_end(self):
        if not hasattr(self, "agent_view_text"):
            return
        from PyQt5.QtGui import QTextCursor

        cursor = self.agent_view_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.agent_view_text.setTextCursor(cursor)
        self.agent_view_text.ensureCursorVisible()
        bar = self.agent_view_text.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def _append_agent_chat(self, role: str, text: str, *, mode=None, tools=None):
        """Append a structured chat turn and re-render HTML."""
        if not hasattr(self, "agent_view_text"):
            return
        if not hasattr(self, "_agent_chat_messages"):
            self._agent_chat_messages = []
        role_l = (role or "system").strip().lower()
        if role_l in ("you", "user", "operator"):
            role_norm = "you"
        elif role_l in ("agent", "assistant"):
            role_norm = "agent"
        else:
            role_norm = "system"
        stamp = datetime.datetime.now().strftime("%H:%M")
        msg = {
            "role": role_norm,
            "text": (text or "").strip(),
            "ts": stamp,
        }
        if mode:
            msg["mode"] = str(mode)
        if tools:
            msg["tools"] = list(tools)
        self._agent_chat_messages.append(msg)
        self._render_agent_chat_view()
        self._sync_active_session_from_view(update_title_from_first_you=(role_norm == "you"))

    def _show_agent_thinking(self):
        """Non-blocking placeholder until the real reply arrives."""
        if not hasattr(self, "_agent_chat_messages"):
            self._agent_chat_messages = []
        stamp = datetime.datetime.now().strftime("%H:%M")
        self._agent_chat_messages.append(
            {
                "role": "agent",
                "text": "",
                "ts": stamp,
                "thinking": True,
            }
        )
        self._agent_thinking_line = "thinking"
        self._render_agent_chat_view()

    def _replace_agent_thinking(self, mode: str, reply: str, *, tools=None):
        """Replace the thinking placeholder with the real agent answer."""
        if not hasattr(self, "_agent_chat_messages"):
            self._agent_chat_messages = []
        stamp = datetime.datetime.now().strftime("%H:%M")
        body = (reply or "").strip() or "(empty reply)"
        tool_list = list(tools or [])
        # Prefer structured tools; also accept leading [tools: …] in reply text
        replacement = {
            "role": "agent",
            "text": body,
            "ts": stamp,
            "mode": mode,
        }
        if tool_list:
            replacement["tools"] = tool_list
        replaced = False
        for i in range(len(self._agent_chat_messages) - 1, -1, -1):
            if self._agent_chat_messages[i].get("thinking"):
                self._agent_chat_messages[i] = replacement
                replaced = True
                break
        if not replaced:
            self._agent_chat_messages.append(replacement)
        self._agent_thinking_line = None
        self._render_agent_chat_view()
        self._sync_active_session_from_view()

    def _set_agent_chat_busy(self, busy: bool):
        self._agent_chat_busy = bool(busy)
        if hasattr(self, "agent_chat_send_btn"):
            self.agent_chat_send_btn.setEnabled(not busy)
        if hasattr(self, "agent_chat_input"):
            self.agent_chat_input.setEnabled(not busy)
        if hasattr(self, "agent_chat_attach_btn"):
            self.agent_chat_attach_btn.setEnabled(not busy)
        if hasattr(self, "agent_chat_attach_clear_btn"):
            self.agent_chat_attach_clear_btn.setEnabled(not busy)
        if hasattr(self, "agent_view_refresh_btn"):
            self.agent_view_refresh_btn.setEnabled(not busy)
        for name in (
            "agent_chat_new_btn",
            "agent_chat_remove_btn",
            "agent_chat_clear_btn",
            "agent_chat_session_list",
            "agent_rebuild_knowledge_btn",
            "agent_view_decisions_btn",
        ):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(not busy)

    def _refresh_agent_attach_label(self):
        paths = list(getattr(self, "_agent_chat_attachments", None) or [])
        label = getattr(self, "agent_chat_attach_label", None)
        clear_btn = getattr(self, "agent_chat_attach_clear_btn", None)
        if label is None:
            return
        if not paths:
            label.setText("")
            if clear_btn is not None:
                clear_btn.setVisible(False)
            return
        names = [os.path.basename(p) for p in paths]
        label.setText("Attached: " + ", ".join(names))
        if clear_btn is not None:
            clear_btn.setVisible(True)

    def _clear_agent_chat_attachments(self):
        self._agent_chat_attachments = []
        self._refresh_agent_attach_label()

    def _attach_agent_chat_files(self):
        """Open file picker and queue attachments for the next Ask."""
        if getattr(self, "_agent_chat_busy", False):
            return
        from app.agent.chat_attachments import FILE_DIALOG_FILTER, MAX_ATTACHMENTS

        existing = list(getattr(self, "_agent_chat_attachments", None) or [])
        remaining = max(0, MAX_ATTACHMENTS - len(existing))
        if remaining <= 0:
            QMessageBox.information(
                self,
                "Attachments",
                f"Maximum {MAX_ATTACHMENTS} files per message.",
            )
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach files for the agent",
            "",
            FILE_DIALOG_FILTER,
        )
        if not paths:
            return
        seen = set(os.path.realpath(p) for p in existing)
        added = 0
        for path in paths:
            if remaining <= 0:
                break
            real = os.path.realpath(path)
            if real in seen:
                continue
            if not os.path.isfile(real):
                continue
            existing.append(real)
            seen.add(real)
            added += 1
            remaining -= 1
        self._agent_chat_attachments = existing
        self._refresh_agent_attach_label()
        if added:
            self.statusBar().showMessage(
                f"Attached {added} file(s) — click Ask to send with your question.", 4000
            )

    def _on_agent_view_refreshed(self, payload: dict):
        """Update status from periodic monitor cycles (chat stays Ask-driven)."""
        if not hasattr(self, "agent_view_text"):
            return
        updated = payload.get("updated_at") or "—"
        self.agent_view_updated_label.setText(f"last update {updated}")
        if hasattr(self, "mllm_summary_label"):
            mllm = (payload.get("mllm_summary") or "").strip()
            if not mllm:
                lines = payload.get("mllm_lines") or []
                mllm = "\n".join(str(x) for x in lines if x)
            if hasattr(self, "_update_mllm_banner"):
                self._update_mllm_banner(mllm)
            else:
                self.mllm_summary_label.setText(mllm if mllm else "")
                self.mllm_summary_label.setVisible(bool(mllm))
        if hasattr(self, "refresh_draft_alerts_panel"):
            self.refresh_draft_alerts_panel()
        if hasattr(self, "refresh_retrain_signals_panel"):
            self.refresh_retrain_signals_panel()
        if hasattr(self, "_refresh_agent_metric_cards"):
            self._refresh_agent_metric_cards()
        if payload.get("llm_used"):
            self._set_agent_online_pill(True, detail="LLM")
        elif payload.get("ollama_up"):
            self._set_agent_online_pill(True)
        else:
            self._set_agent_online_pill(False)

    def _set_agent_online_pill(self, online: bool, detail: str = "", model_name: str = ""):
        """Update header status pill to match mockup (online · model)."""
        label = getattr(self, "agent_llm_status_label", None)
        if label is None:
            return
        model = (model_name or "").strip()
        if not model:
            try:
                from app.agent.ollama_client import resolve_chat_model

                model = resolve_chat_model() or "qwen3.5:9b"
            except Exception:
                model = "qwen3.5:9b"
        if online:
            label.setText(f"  online · {model}  ")
            label.setStyleSheet(
                "QLabel { background: #dcfce7; color: #166534; border-radius: 10px; "
                "padding: 3px 8px; font-size: 11px; font-weight: 600; }"
            )
        else:
            label.setText("  offline · local  ")
            label.setStyleSheet(
                "QLabel { background: #fee2e2; color: #991b1b; border-radius: 10px; "
                "padding: 3px 8px; font-size: 11px; font-weight: 600; }"
            )

    def _update_mllm_banner(self, text: str):
        label = getattr(self, "mllm_summary_label", None)
        if label is None:
            return
        raw = (text or "").strip()
        if raw.startswith("[M-LLM]"):
            raw = raw[7:].strip()
        if (not raw) or raw in ("Log analysis: —", "—"):
            label.clear()
            label.setVisible(False)
            return
        label.setText(raw)
        label.setVisible(True)

    def _on_agent_chat_reply(self, payload: dict):
        reply = payload.get("reply") or ""
        mode = str(payload.get("agent_mode") or payload.get("node") or "").strip()
        mode_l = mode.lower()
        if mode_l in ("ra-llm", "rallm", "tool"):
            mode_label = "RA-LLM"
        elif mode_l in ("chat", "chitchat", "greeting"):
            mode_label = "Chat"
        elif mode_l in ("c-llm", "cllm", "knowledge"):
            mode_label = "C-LLM"
        elif mode_l in ("llm",) or payload.get("llm_used"):
            mode_label = "RA-LLM" if (payload.get("route") or "tool") == "tool" else "C-LLM"
        elif mode_l == "tools":
            mode_label = "RA-LLM"
        elif mode_l == "offline":
            mode_label = "Offline"
        else:
            mode_label = mode or ("RA-LLM" if payload.get("llm_used") else "Offline")
        trace = payload.get("tool_trace") or []
        # If tools ran, never show Offline just because final narration failed
        if mode_label == "Offline" and any(t.get("ok") for t in (trace or [])):
            mode_label = "RA-LLM"
        tools = []
        if trace:
            for t in trace[:12]:
                tools.append(
                    {
                        "tool": t.get("tool") or "?",
                        "ok": bool(t.get("ok", True)),
                    }
                )
        # Refresh M-LLM strip when Ask carried a log summary
        if hasattr(self, "mllm_summary_label") and payload.get("log_summary"):
            self._update_mllm_banner(payload.get("log_summary"))
        self._replace_agent_thinking(mode_label, reply, tools=tools or None)
        self._set_agent_chat_busy(False)
        updated = datetime.datetime.now().strftime("%H:%M:%S")
        if hasattr(self, "agent_view_updated_label"):
            self.agent_view_updated_label.setText(f"last update {updated}")
        if payload.get("llm_used"):
            self._set_agent_online_pill(True, detail=mode_label)
        self._refresh_agent_llm_status()
        if hasattr(self, "_refresh_agent_metric_cards"):
            self._refresh_agent_metric_cards()

    def _ask_agent_async(self, msg: str, *, attachment_paths=None):
        """Append You turn + thinking..., then ask on a background thread."""
        loop = getattr(self, "agent_loop", None)
        if loop is None:
            QMessageBox.information(self, "Agent", "Agent loop is not running.")
            return
        if getattr(self, "_agent_chat_busy", False):
            return

        from app.agent.chat_attachments import build_message_with_attachments

        paths = list(attachment_paths or [])
        built = build_message_with_attachments(msg, paths)
        if not built.get("ok"):
            errs = built.get("errors") or []
            if errs:
                detail = "; ".join(
                    f"{e.get('filename')}: {e.get('error')}" for e in errs[:5]
                )
                QMessageBox.warning(
                    self,
                    "Attachments",
                    f"Could not read attached file(s):\n{detail}",
                )
            return

        ask_msg = built["message"]
        display = built.get("display") or (msg or "").strip() or ask_msg
        if built.get("errors"):
            failed = ", ".join(
                f"{e.get('filename')} ({e.get('error')})" for e in built["errors"][:4]
            )
            self.statusBar().showMessage(f"Some attachments skipped: {failed}", 6000)

        self._append_agent_chat("You", display)
        self._show_agent_thinking()
        self._set_agent_chat_busy(True)
        self._refresh_agent_llm_status()

        def _run():
            try:
                loop.ask(ask_msg)
            except Exception as exc:
                logger.warning("Agent chat failed: %s", exc)
                loop.signals.chat_reply.emit(
                    {
                        "ok": False,
                        "user_message": display,
                        "reply": f"Error: {exc}",
                        "llm_used": False,
                        "error": str(exc),
                    }
                )

        threading.Thread(target=_run, name="stdms-agent-chat", daemon=True).start()

    def send_agent_chat(self):
        """Send operator question (+ optional attached files) to the agent."""
        msg = ""
        if hasattr(self, "agent_chat_input"):
            msg = self.agent_chat_input.text().strip()
            self.agent_chat_input.clear()
        paths = list(getattr(self, "_agent_chat_attachments", None) or [])
        self._clear_agent_chat_attachments()
        if not msg and not paths:
            return
        self._ask_agent_async(msg, attachment_paths=paths)

    def _refresh_agent_metric_cards(self):
        """No-op: metric cards were removed from the chat panel."""
        return

    def _focus_pending_agent_drafts(self):
        """Open Home → AI Assistant and focus Pending Agent Drafts."""
        try:
            if hasattr(self, "tabs") and getattr(self, "dashboard_tab", None) is not None:
                idx = self.tabs.indexOf(self.dashboard_tab)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            if hasattr(self, "home_subtabs"):
                # AI Assistant is sub-tab 1 (Fleet Overview is 0)
                self.home_subtabs.setCurrentIndex(1)
            if hasattr(self, "_sync_ops_nav_active"):
                self._sync_ops_nav_active()
        except Exception:
            pass
        if hasattr(self, "refresh_draft_alerts_panel"):
            self.refresh_draft_alerts_panel()
        table = getattr(self, "draft_alerts_table", None)
        if table is not None:
            table.setFocus()
            table.scrollToTop()
        self.statusBar().showMessage(
            "Pending Agent Drafts — review Approve / Reject on Home → AI Assistant.", 5000
        )

    def refresh_agent_view_now(self):
        """Analyze Fleet — deep health report for all tabs via Ask path."""
        self._ask_agent_async("Give me a full health report for all tabs")

    def rebuild_agent_knowledge_index(self):
        """Background dual ingest: knowledge/space + knowledge/ground (+ FSM export)."""
        if getattr(self, "_agent_knowledge_busy", False):
            return
        self._agent_knowledge_busy = True
        if hasattr(self, "agent_rebuild_knowledge_btn"):
            self.agent_rebuild_knowledge_btn.setEnabled(False)
        self.statusBar().showMessage("Rebuilding dual knowledge indexes (space/ground)…", 0)

        tabs_config = {}
        try:
            mgr = getattr(self, "tab_config_manager", None)
            if mgr is not None and getattr(mgr, "configs", None):
                tabs_config = dict(mgr.configs)
            elif getattr(self, "custom_tabs", None):
                for tid, widget in list(self.custom_tabs.items()):
                    cfg = getattr(widget, "config", None)
                    if isinstance(cfg, dict):
                        tabs_config[tid] = dict(cfg)
        except Exception:
            tabs_config = {}

        def _run():
            try:
                from app.agent.rag.ingest import ingest_dual_knowledge

                result = ingest_dual_knowledge(tabs_config=tabs_config or None)
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            QTimer.singleShot(0, lambda: self._on_knowledge_rebuild_done(result))

        threading.Thread(target=_run, name="stdms-rag-ingest", daemon=True).start()

    def _on_knowledge_rebuild_done(self, result: dict):
        self._agent_knowledge_busy = False
        if hasattr(self, "agent_rebuild_knowledge_btn"):
            self.agent_rebuild_knowledge_btn.setEnabled(True)
        if hasattr(self, "_refresh_agent_metric_cards"):
            self._refresh_agent_metric_cards()
        ok = bool(result.get("ok"))
        docs = result.get("documents", 0)
        chunks = result.get("chunks", 0)
        indexed = result.get("indexed", 0)
        err = result.get("error") or (result.get("errors") or [None])[0]
        space_n = (result.get("space") or {}).get("documents", 0)
        ground_n = (result.get("ground") or {}).get("documents", 0)
        fsm_path = (result.get("fsm_export") or {}).get("path")
        if ok:
            msg = (
                f"Dual knowledge ready: space={space_n} doc(s), ground={ground_n} doc(s), "
                f"{chunks} chunk(s) total ({indexed} updated). "
                f"FSM export: {fsm_path or 'n/a'}"
            )
            self.statusBar().showMessage(msg, 8000)
            if hasattr(self, "_append_agent_chat"):
                self._append_agent_chat("System", msg)
        else:
            detail = err or result.get("note") or "unknown error"
            self.statusBar().showMessage(f"Knowledge rebuild failed: {detail}", 8000)
            QMessageBox.warning(
                self,
                "Rebuild Knowledge",
                f"Could not rebuild dual knowledge indexes.\n\n{detail}\n\n"
                "Ensure Ollama is running and: ollama pull nomic-embed-text\n"
                "Indexes: data/knowledge_index/rag_space.sqlite + rag_ground.sqlite",
            )

    def show_agent_decisions_dialog(self):
        """Show audit log in a scrollable dialog with optional tab filter."""
        audit = getattr(self, "_agent_audit", None)
        if audit is None and getattr(self, "agent_bridge", None) is not None:
            audit = getattr(self.agent_bridge, "audit", None)
        if audit is None:
            QMessageBox.information(self, "Agent Decisions", "Audit log is not available.")
            return
        try:
            decisions = audit.list_decisions(limit=200)
        except Exception as exc:
            QMessageBox.warning(self, "Agent Decisions", f"Read error: {exc}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Agent Decisions")
        dlg.setMinimumSize(820, 480)
        v = QVBoxLayout(dlg)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter by tab:"))
        tab_filter = QComboBox()
        tab_filter.addItem("All tabs", None)
        tab_ids = sorted({str(d.get("tab_id")) for d in decisions if d.get("tab_id")})
        # Also include live custom tab titles when available
        for tab_id, widget in getattr(self, "custom_tabs", {}).items():
            title = None
            try:
                snap = widget.get_snapshot() if hasattr(widget, "get_snapshot") else {}
                title = (snap or {}).get("title")
            except Exception:
                title = None
            label = f"{title} ({tab_id[:8]}…)" if title else tab_id
            if tab_filter.findData(tab_id) < 0:
                tab_filter.addItem(label, tab_id)
            if tab_id in tab_ids:
                tab_ids.remove(tab_id)
        for tab_id in tab_ids:
            tab_filter.addItem(tab_id, tab_id)
        filter_row.addWidget(tab_filter, 1)
        v.addLayout(filter_row)

        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(["ID", "Time", "Tab", "Tool", "Outcome", "Reasoning"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setWordWrap(True)
        v.addWidget(table, 1)

        def _populate(selected_tab_id=None):
            rows = decisions
            if selected_tab_id:
                rows = [d for d in decisions if str(d.get("tab_id") or "") == str(selected_tab_id)]
            table.setRowCount(len(rows))
            for row, item in enumerate(rows):
                cells = [
                    str(item.get("id", "")),
                    str(item.get("timestamp", ""))[:19],
                    str(item.get("tab_id") or "—"),
                    str(item.get("tool_called") or "—"),
                    str(item.get("outcome") or "—"),
                    str(item.get("reasoning") or item.get("context_summary") or "")[:500],
                ]
                for col, text in enumerate(cells):
                    table.setItem(row, col, QTableWidgetItem(text))
            table.resizeRowsToContents()

        def _on_filter(_idx=None):
            _populate(tab_filter.currentData())

        tab_filter.currentIndexChanged.connect(_on_filter)
        _populate(None)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        v.addWidget(close_btn)
        dlg.exec_()

    def closeEvent(self, event):
        """Stop agent monitor loop and bridge on window close."""
        try:
            loop = getattr(self, "agent_loop", None)
            if loop is not None:
                loop.stop()
                self.agent_loop = None
        except Exception as exc:
            logger.warning("Agent loop stop on close failed: %s", exc)
        try:
            from app.agent import stop_agent_bridge

            stop_agent_bridge()
        except Exception:
            pass
        self.agent_bridge = None
        super().closeEvent(event)

    def refresh_fleet_dashboard(self):
        """Refresh fleet overview table on Dashboard."""
        if not hasattr(self, "fleet_tabs_table"):
            return
        self.fleet_tabs_table.setRowCount(len(self.custom_tabs))
        for row, (tab_id, widget) in enumerate(self.custom_tabs.items()):
            snap = widget.get_snapshot() if hasattr(widget, "get_snapshot") else {}
            cells = [
                snap.get("title", tab_id[:8]),
                "Active" if snap.get("monitoring_active") else "Inactive",
                snap.get("health_state", "Idle"),
                snap.get("watch_status", "—"),
                snap.get("last_file", "—"),
                str(snap.get("trained_models", 0)),
                "OK" if snap.get("obs_ok", True) else f"{snap.get('obs_violations', 0)} viol.",
                snap.get("updated_at", "—"),
            ]
            for col, text in enumerate(cells):
                self.fleet_tabs_table.setItem(row, col, QTableWidgetItem(str(text)[:120]))
        self.refresh_retrain_signals_panel()

    def _resolve_tab_display_name(self, tab_id):
        if not tab_id:
            return "—"
        widget = self.custom_tabs.get(tab_id)
        if widget is not None and hasattr(widget, "config"):
            return str(widget.config.get("title", tab_id))
        return str(tab_id)

    def refresh_retrain_signals_panel(self):
        """Show unacknowledged drift retrain signals on the Dashboard."""
        if not hasattr(self, "retrain_signals_table"):
            return

        registry = getattr(self, "model_registry", None)
        if registry is None or not hasattr(registry, "get_pending_retrain_signals"):
            if hasattr(self, "retrain_signals_summary_label"):
                self.retrain_signals_summary_label.setText("Model registry unavailable.")
            return

        try:
            signals = registry.get_pending_retrain_signals()
        except Exception as exc:
            logger.error("Failed to load pending retrain signals: %s", exc)
            if hasattr(self, "retrain_signals_summary_label"):
                err_text = format_registry_error(exc) if format_registry_error else str(exc)
                self.retrain_signals_summary_label.setText(f"Could not load retrain signals: {err_text}")
            return

        count = len(signals)
        if hasattr(self, "retrain_signals_summary_label"):
            if count == 0:
                self.retrain_signals_summary_label.setText("No pending retrain signals.")
            elif count == 1:
                self.retrain_signals_summary_label.setText("1 model pending retrain (concept drift detected).")
            else:
                self.retrain_signals_summary_label.setText(
                    f"{count} models pending retrain (concept drift detected)."
                )

        self.retrain_signals_table.setRowCount(count)
        for row, signal in enumerate(signals):
            tab_label = self._resolve_tab_display_name(signal.get("tab_id"))
            features = signal.get("drifted_features") or []
            features_text = ", ".join(str(f) for f in features) if features else "—"
            created_at = str(signal.get("created_at") or "—")
            if len(created_at) > 19:
                created_at = created_at[:19]

            values = [
                str(signal.get("id", "—")),
                tab_label,
                str(signal.get("model_name") or "—"),
                f"{float(signal.get('drift_score') or 0.0):.3f}",
                features_text[:80],
                str(signal.get("reason") or "—"),
                created_at,
            ]
            for col, text in enumerate(values):
                self.retrain_signals_table.setItem(row, col, QTableWidgetItem(text))

            acknowledge_btn = QPushButton("Acknowledge")
            signal_id = int(signal["id"])
            acknowledge_btn.clicked.connect(
                lambda _checked=False, sid=signal_id: self._acknowledge_retrain_signal(sid)
            )
            self.retrain_signals_table.setCellWidget(row, 7, acknowledge_btn)

    def _acknowledge_retrain_signal(self, signal_id):
        registry = getattr(self, "model_registry", None)
        if registry is None or not hasattr(registry, "acknowledge_retrain_signal"):
            QMessageBox.warning(self, "Retrain Signals", "Model registry is not available.")
            return

        actor = self.current_username or "operator"
        try:
            updated = registry.acknowledge_retrain_signal(signal_id, acknowledged_by=actor)
        except Exception as exc:
            logger.error("Failed to acknowledge retrain signal %s: %s", signal_id, exc)
            err_text = format_registry_error(exc) if format_registry_error else str(exc)
            QMessageBox.warning(
                self,
                "Retrain Signals",
                f"Could not acknowledge signal #{signal_id}.\n{err_text}",
            )
            return

        if not updated:
            QMessageBox.information(
                self,
                "Retrain Signals",
                f"Signal #{signal_id} was not found or is already acknowledged.",
            )
            return

        self.refresh_retrain_signals_panel()
        self.statusBar().showMessage(
            f"Retrain signal #{signal_id} acknowledged by {actor}.", 5000
        )

    def refresh_draft_alerts_panel(self):
        """Show pending agent propose_* drafts (alert/config) for approval."""
        if not hasattr(self, "draft_alerts_table"):
            return
        registry = getattr(self, "model_registry", None)
        if registry is None or not hasattr(registry, "get_pending_draft_alerts"):
            if hasattr(self, "draft_alerts_summary_label"):
                self.draft_alerts_summary_label.setText("Model registry unavailable.")
            return
        try:
            drafts = registry.get_pending_draft_alerts()
        except Exception as exc:
            logger.error("Failed to load pending draft alerts: %s", exc)
            if hasattr(self, "draft_alerts_summary_label"):
                err_text = format_registry_error(exc) if format_registry_error else str(exc)
                self.draft_alerts_summary_label.setText(f"Could not load drafts: {err_text}")
            return

        count = len(drafts)
        critical_n = sum(
            1
            for d in drafts
            if str(d.get("severity") or "").upper() in ("CRITICAL", "ALERT")
        )
        warning_n = sum(
            1 for d in drafts if str(d.get("severity") or "").upper() in ("WARNING",)
        )
        if hasattr(self, "draft_alerts_summary_label"):
            if count == 0:
                self.draft_alerts_summary_label.setText("No pending agent drafts.")
                self.draft_alerts_summary_label.setStyleSheet("")
            else:
                parts = [f"{count} pending agent draft(s)"]
                if critical_n:
                    parts.append(f"{critical_n} CRITICAL")
                if warning_n:
                    parts.append(f"{warning_n} WARNING")
                self.draft_alerts_summary_label.setText(" · ".join(parts) + " awaiting approval.")
                if critical_n:
                    self.draft_alerts_summary_label.setStyleSheet(
                        "color:#c0392b;font-weight:bold;padding:4px;"
                        "background-color:#fdecea;border:1px solid #e74c3c;border-radius:4px;"
                    )
                elif warning_n:
                    self.draft_alerts_summary_label.setStyleSheet(
                        "color:#b7791f;font-weight:bold;padding:4px;"
                        "background-color:#fff8e6;border:1px solid #f0c36d;border-radius:4px;"
                    )
                else:
                    self.draft_alerts_summary_label.setStyleSheet("")
        for btn_name in ("draft_approve_all_btn", "draft_reject_all_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(count > 0)
        if hasattr(self, "_refresh_agent_metric_cards"):
            self._refresh_agent_metric_cards()

        # Prioritize CRITICAL → WARNING → INFO, then newest id
        _sev_rank = {"CRITICAL": 0, "WARNING": 1, "ALERT": 1, "INFO": 2}
        drafts = sorted(
            drafts,
            key=lambda d: (
                _sev_rank.get(str(d.get("severity") or "").upper(), 9),
                -(int(d.get("id") or 0)),
            ),
        )

        # Short pending-actions line (watchlist + counts)
        if hasattr(self, "pending_actions_summary_label"):
            try:
                from app.agent.watchlist import load_watchlist

                wl = load_watchlist()
                wl_n = len(wl.get("tabs") or [])
                crit = sum(
                    1
                    for d in drafts
                    if str(d.get("severity") or "").upper() == "CRITICAL"
                )
                parts = [f"Pending actions: {count} draft(s)"]
                if crit:
                    parts.append(f"{crit} CRITICAL")
                parts.append(f"watchlist={wl_n}")
                self.pending_actions_summary_label.setText(" · ".join(parts))
            except Exception:
                self.pending_actions_summary_label.setText(
                    f"Pending actions: {count} draft(s)"
                )

        self.draft_alerts_table.setRowCount(count)
        for row, draft in enumerate(drafts):
            tab_label = self._resolve_tab_display_name(draft.get("tab_id"))
            created_at = str(draft.get("created_at") or "—")
            if len(created_at) > 19:
                created_at = created_at[:19]
            values = [
                str(draft.get("id", "—")),
                str(draft.get("kind") or "alert"),
                tab_label,
                str(draft.get("severity") or "—"),
                str(draft.get("proposed_message") or "—")[:100],
                str(draft.get("agent_reasoning") or "—")[:80],
                created_at,
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col == 3 and str(draft.get("severity") or "").upper() == "CRITICAL":
                    item.setForeground(QColor("#b00020"))
                self.draft_alerts_table.setItem(row, col, item)

            draft_id = int(draft["id"])
            actions = QWidget()
            actions_layout = QHBoxLayout(actions)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            approve_btn = QPushButton("Approve")
            reject_btn = QPushButton("Reject")
            approve_btn.clicked.connect(
                lambda _c=False, did=draft_id: self._resolve_draft_alert(did, "approved")
            )
            reject_btn.clicked.connect(
                lambda _c=False, did=draft_id: self._resolve_draft_alert(did, "rejected")
            )
            actions_layout.addWidget(approve_btn)
            actions_layout.addWidget(reject_btn)
            self.draft_alerts_table.setCellWidget(row, 7, actions)

    def _resolve_draft_alert(self, draft_id, status):
        registry = getattr(self, "model_registry", None)
        if registry is None or not hasattr(registry, "resolve_draft_alert"):
            QMessageBox.warning(self, "Agent Drafts", "Model registry is not available.")
            return
        actor = self.current_username or "operator"
        draft = None
        if hasattr(registry, "get_draft_alert"):
            try:
                draft = registry.get_draft_alert(int(draft_id))
            except Exception as exc:
                logger.warning("Could not load draft %s before resolve: %s", draft_id, exc)
        try:
            updated = registry.resolve_draft_alert(draft_id, status, actioned_by=actor)
        except Exception as exc:
            logger.error("Failed to resolve draft %s: %s", draft_id, exc)
            err_text = format_registry_error(exc) if format_registry_error else str(exc)
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Could not {status} draft #{draft_id}.\n{err_text}",
            )
            return
        if not updated:
            QMessageBox.information(
                self,
                "Agent Drafts",
                f"Draft #{draft_id} was not found or is already resolved.",
            )
            return
        self.refresh_draft_alerts_panel()
        self.statusBar().showMessage(
            f"Draft #{draft_id} {status} by {actor}.", 5000
        )
        if str(status).strip().lower() == "approved" and draft:
            QTimer.singleShot(
                0, lambda d=draft: self._execute_approved_agent_draft(d)
            )

    def _approve_all_draft_alerts(self):
        self._bulk_resolve_draft_alerts("approved")

    def _reject_all_draft_alerts(self):
        self._bulk_resolve_draft_alerts("rejected")

    def _bulk_resolve_draft_alerts(self, status: str):
        """Approve or reject every pending draft (with confirmation)."""
        registry = getattr(self, "model_registry", None)
        if registry is None or not hasattr(registry, "get_pending_draft_alerts"):
            QMessageBox.warning(self, "Agent Drafts", "Model registry is not available.")
            return
        try:
            drafts = list(registry.get_pending_draft_alerts() or [])
        except Exception as exc:
            err_text = format_registry_error(exc) if format_registry_error else str(exc)
            QMessageBox.warning(self, "Agent Drafts", f"Could not load drafts.\n{err_text}")
            return
        if not drafts:
            QMessageBox.information(self, "Agent Drafts", "No pending drafts to process.")
            return

        status_l = str(status).strip().lower()
        action_word = "Approve" if status_l == "approved" else "Reject"
        n = len(drafts)
        kinds = {}
        for d in drafts:
            k = str(d.get("kind") or "alert")
            kinds[k] = kinds.get(k, 0) + 1
        kind_summary = ", ".join(f"{c}× {k}" for k, c in sorted(kinds.items()))
        reply = QMessageBox.question(
            self,
            f"{action_word} All Drafts",
            f"{action_word} all {n} pending draft(s)?\n\n{kind_summary}\n\n"
            + (
                "Approved drafts will run their actions (create tab / train / start / stop / …)."
                if status_l == "approved"
                else "Rejected drafts are dismissed and will not run."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        actor = self.current_username or "operator"
        ok_n = 0
        fail_n = 0
        approved_to_run = []
        # Process oldest first so create_tab → train chains stay sensible
        ordered = sorted(drafts, key=lambda d: int(d.get("id") or 0))
        for draft in ordered:
            draft_id = draft.get("id")
            if draft_id is None:
                fail_n += 1
                continue
            # Reload full draft before resolve (payload needed for execute)
            full = draft
            if hasattr(registry, "get_draft_alert"):
                try:
                    loaded = registry.get_draft_alert(int(draft_id))
                    if loaded:
                        full = loaded
                except Exception:
                    pass
            try:
                updated = registry.resolve_draft_alert(
                    int(draft_id), status_l, actioned_by=actor
                )
            except Exception as exc:
                logger.error("Bulk %s failed for draft %s: %s", status_l, draft_id, exc)
                fail_n += 1
                continue
            if not updated:
                fail_n += 1
                continue
            ok_n += 1
            if status_l == "approved" and full:
                approved_to_run.append(full)

        self.refresh_draft_alerts_panel()
        self.statusBar().showMessage(
            f"{action_word} All: {ok_n} ok, {fail_n} failed (by {actor}).", 8000
        )

        # Run approved actions sequentially on the GUI thread
        if approved_to_run:
            def _run_next(queue, idx=0):
                if idx >= len(queue):
                    self.refresh_draft_alerts_panel()
                    return
                try:
                    self._execute_approved_agent_draft(queue[idx])
                except Exception as exc:
                    logger.error("Bulk execute draft failed: %s", exc)
                QTimer.singleShot(50, lambda: _run_next(queue, idx + 1))

            QTimer.singleShot(0, lambda: _run_next(list(approved_to_run), 0))

    def _execute_approved_agent_draft(self, draft):
        """Run train / create_tab / start|stop monitoring after human Approve (GUI thread)."""
        if not isinstance(draft, dict):
            return
        kind = str(draft.get("kind") or "").strip().lower()
        try:
            if kind == "train":
                self._execute_approved_train(draft)
            elif kind == "create_tab":
                self._execute_approved_create_tab(draft)
            elif kind == "start_monitoring":
                self._execute_approved_start_monitoring(draft)
            elif kind == "stop_monitoring":
                self._execute_approved_stop_monitoring(draft)
            elif kind == "remove_model":
                self._execute_approved_remove_model(draft)
            elif kind in ("write_document", "update_document"):
                self._execute_approved_document(draft)
            elif kind in ("write_sop", "update_sop"):
                self._execute_approved_sop(draft)
        except Exception as exc:
            logger.error("Approved draft execute failed (%s): %s", kind, exc)
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Draft was approved but the action failed.\n{exc}",
            )

    def _execute_approved_train(self, draft):
        tab_id = draft.get("tab_id")
        payload = draft.get("proposed_payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        model_id = payload.get("model_id") or None
        widget = (getattr(self, "custom_tabs", None) or {}).get(tab_id)
        if widget is None or not hasattr(widget, "train_model"):
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Cannot train: tab '{tab_id}' is not open.",
            )
            return
        title = (getattr(widget, "config", None) or {}).get("title") or tab_id

        # Enqueue start_monitoring only after train actually finishes (not when worker starts)
        def _on_train_success():
            try:
                trained = 0
                if hasattr(widget, "_count_trained_models"):
                    trained = int(widget._count_trained_models() or 0)
                if trained < 1:
                    return
                did = self._enqueue_start_monitoring_draft(tab_id, title=str(title))
                if did and hasattr(self, "refresh_draft_alerts_panel"):
                    self.refresh_draft_alerts_panel()
                if did:
                    msg = (
                        f"Train completed for '{title}'. "
                        f"Pending start_monitoring draft #{did} — Approve to go online."
                    )
                    self.statusBar().showMessage(msg, 10000)
                    if hasattr(self, "_append_agent_chat"):
                        try:
                            self._append_agent_chat("System", msg)
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning("post-train start draft failed: %s", exc)

        try:
            widget._agent_post_train_hook = _on_train_success
        except Exception:
            pass

        ok = widget.train_model(model_id=model_id, silent=False)
        if ok is False:
            try:
                widget._agent_post_train_hook = None
            except Exception:
                pass
            self.statusBar().showMessage(
                f"Approved train for '{title}' did not start (see messages).", 8000
            )
        else:
            self.statusBar().showMessage(
                f"Training started for '{title}' (approved draft).", 8000
            )

    def _resolve_draft_tab_widget(self, draft):
        tab_id = draft.get("tab_id")
        payload = draft.get("proposed_payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        tabs = getattr(self, "custom_tabs", None) or {}
        widget = tabs.get(tab_id) if tab_id else None
        if widget is None:
            want = str(payload.get("title") or "").strip().lower()
            if want:
                for tid, w in tabs.items():
                    title = str((getattr(w, "config", None) or {}).get("title") or "").strip().lower()
                    if title == want or want in title:
                        return tid, w
        return tab_id, widget

    def _execute_approved_start_monitoring(self, draft):
        tab_id, widget = self._resolve_draft_tab_widget(draft)
        if widget is None or not hasattr(widget, "start_monitoring"):
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Cannot start monitoring: tab '{tab_id}' is not open.",
            )
            return
        if getattr(widget, "monitoring_active", False):
            title = (getattr(widget, "config", None) or {}).get("title") or tab_id
            self.statusBar().showMessage(
                f"Tab '{title}' is already monitoring.", 6000
            )
            return
        widget.start_monitoring()
        title = (getattr(widget, "config", None) or {}).get("title") or tab_id
        if hasattr(self, "refresh_fleet_dashboard"):
            self.refresh_fleet_dashboard()
        self.statusBar().showMessage(
            f"Monitoring started for '{title}' (approved draft).", 8000
        )

    def _execute_approved_stop_monitoring(self, draft):
        tab_id, widget = self._resolve_draft_tab_widget(draft)
        if widget is None or not hasattr(widget, "stop_monitoring"):
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Cannot stop monitoring: tab '{tab_id}' is not open.",
            )
            return
        widget.stop_monitoring()
        title = (getattr(widget, "config", None) or {}).get("title") or tab_id
        if hasattr(self, "refresh_fleet_dashboard"):
            self.refresh_fleet_dashboard()
        self.statusBar().showMessage(
            f"Monitoring stopped for '{title}' (approved draft).", 8000
        )

    def _execute_approved_remove_model(self, draft):
        tab_id, widget = self._resolve_draft_tab_widget(draft)
        payload = draft.get("proposed_payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        model_id = str(payload.get("model_id") or "").strip()
        if widget is None or not hasattr(widget, "remove_model"):
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Cannot remove model: tab '{tab_id}' is not open.",
            )
            return
        if not model_id:
            QMessageBox.warning(self, "Agent Drafts", "Draft is missing model_id.")
            return
        ok = widget.remove_model(model_id, silent=True)
        title = (getattr(widget, "config", None) or {}).get("title") or tab_id
        if ok is False:
            self.statusBar().showMessage(
                f"Model remove cancelled or failed on '{title}'.", 6000
            )
            return
        self.statusBar().showMessage(
            f"Removed model {model_id[:8]}… from '{title}' (approved draft).", 8000
        )

    def _execute_approved_document(self, draft):
        """Write/update a knowledge MD/TXT after Approve."""
        from app.agent.docs_io import write_knowledge_document

        payload = draft.get("proposed_payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        filename = str(payload.get("filename") or "").strip()
        content = payload.get("content")
        kind = str(draft.get("kind") or "").strip().lower()
        mode = "update" if kind == "update_document" else "write"
        if not filename:
            QMessageBox.warning(self, "Agent Drafts", "Draft is missing filename.")
            return
        result = write_knowledge_document(filename, content if content is not None else "", mode=mode)
        if not result.get("ok"):
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Document action failed: {result.get('error') or result.get('status')}",
            )
            return
        msg = (
            f"Knowledge doc '{result.get('filename')}' {result.get('status')}. "
            "Click Rebuild Knowledge to refresh the RAG index."
        )
        self.statusBar().showMessage(msg, 12000)
        if hasattr(self, "_append_agent_chat"):
            try:
                self._append_agent_chat("System", msg)
            except Exception:
                pass

    def _execute_approved_sop(self, draft):
        """Create/update SOP Word (.docx) from template after Approve."""
        from app.agent.docs_io import write_sop_document

        payload = draft.get("proposed_payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        filename = str(payload.get("filename") or "").strip()
        fields = payload.get("sop_fields") if isinstance(payload.get("sop_fields"), dict) else {}
        kind = str(draft.get("kind") or "").strip().lower()
        mode = "update" if kind == "update_sop" else "write"
        result = write_sop_document(fields, filename=filename or None, mode=mode)
        if not result.get("ok"):
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"SOP Word action failed: {result.get('error') or result.get('status')}",
            )
            return
        msg = (
            f"SOP Word doc '{result.get('filename')}' {result.get('status')} "
            f"(template {result.get('template_id')}). "
            "Open it from data/knowledge. Rebuild Knowledge if needed."
        )
        self.statusBar().showMessage(msg, 14000)
        if hasattr(self, "_append_agent_chat"):
            try:
                self._append_agent_chat("System", msg)
            except Exception:
                pass

    def _execute_approved_create_tab(self, draft):
        # Same capability as + New Tab menu: admin (manage_users) or create_tab role.
        if hasattr(self, "service_layer") and hasattr(self, "current_username"):
            try:
                can_create = self.service_layer.authorize(
                    self.current_username, "create_tab", resource="*"
                ) or self.service_layer.authorize(
                    self.current_username, "manage_users", resource="admin/users"
                )
                if not can_create:
                    QMessageBox.warning(
                        self,
                        "Permission Denied",
                        "You do not have permission to create tabs.",
                    )
                    return
            except Exception as exc:
                logger.warning("create_tab authorize check failed: %s", exc)
        payload = draft.get("proposed_payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        partial = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        try:
            from app.monitoring.tab_config import default_tab_config
        except Exception:
            default_tab_config = None
        title = str(partial.get("title") or "Agent Tab").strip() or "Agent Tab"
        # Upgrade weak titles from source_file / data path
        try:
            from app.agent.smart_config import _ideal_title_from_hint, is_weak_title

            if is_weak_title(title):
                src = str(partial.get("source_file") or "")
                hint = Path(src).stem if src else Path(str(partial.get("data_folder") or "")).name
                title = _ideal_title_from_hint(hint, fallback="Health Monitoring")
        except Exception:
            pass
        if default_tab_config is not None:
            config = default_tab_config(title=title)
            config.update(partial)
            config["title"] = title
        else:
            config = dict(partial)
            config.setdefault("title", title)
        # Ensure each model entry has a model_id
        models = config.get("models")
        if isinstance(models, list):
            for m in models:
                if isinstance(m, dict) and not m.get("model_id"):
                    m["model_id"] = str(uuid.uuid4())
        config.setdefault("created_at", datetime.datetime.now().isoformat())
        config["updated_at"] = datetime.datetime.now().isoformat()
        # Reject create if data folder is missing (same guard as propose_create_tab)
        data_folder = str(config.get("data_folder") or "").strip()
        if data_folder and not Path(data_folder).is_dir():
            QMessageBox.warning(
                self,
                "Agent Drafts",
                f"Cannot create tab: Data folder does not exist:\n{data_folder}",
            )
            return
        tab_id = self._create_custom_tab_from_config(config)
        if tab_id:
            self.statusBar().showMessage(
                f"Created tab '{config.get('title')}' from approved draft.", 8000
            )
            # Chain: enqueue train drafts for each suggested model (still Approve-gated)
            self._enqueue_post_create_train_drafts(tab_id, config)

    def _enqueue_post_create_train_drafts(self, tab_id, config):
        """After create_tab Approve, propose train for each model (Propose→Approve)."""
        registry = getattr(self, "model_registry", None)
        if registry is None or not hasattr(registry, "create_draft_alert"):
            return
        models = config.get("models") if isinstance(config, dict) else None
        if not isinstance(models, list) or not models:
            return
        title = (config or {}).get("title") or tab_id
        created = []
        for m in models:
            if not isinstance(m, dict):
                continue
            mid = m.get("model_id")
            mtype = m.get("model_type") or "model"
            if not mid:
                continue
            try:
                did = registry.create_draft_alert(
                    tab_id=tab_id,
                    kind="train",
                    agent_reasoning=(
                        f"Auto-follow-up after create_tab Approve: train {mtype} "
                        f"so monitoring can start."
                    ),
                    proposed_message=f"Train {mtype} on '{title}'",
                    proposed_payload={"model_id": str(mid), "model_type": mtype},
                    severity="INFO",
                )
                created.append(did)
            except Exception as exc:
                logger.warning("Could not enqueue train draft for %s: %s", mid, exc)
        if created:
            msg = (
                f"Created tab '{title}'. Pending train draft(s) #{', #'.join(str(x) for x in created)} — "
                "Approve each (or in order) under Pending Agent Drafts, then Start Monitoring."
            )
            self.statusBar().showMessage(msg, 12000)
            if hasattr(self, "refresh_draft_alerts_panel"):
                self.refresh_draft_alerts_panel()
            if hasattr(self, "_append_agent_chat"):
                try:
                    self._append_agent_chat("System", msg)
                except Exception:
                    pass

    def _enqueue_start_monitoring_draft(self, tab_id, title=""):
        registry = getattr(self, "model_registry", None)
        if registry is None or not hasattr(registry, "create_draft_alert"):
            return None
        # Dedupe pending start drafts for this tab
        try:
            pending = registry.get_pending_draft_alerts(tab_id=tab_id, kind="start_monitoring")
            if pending:
                return pending[0].get("id")
        except Exception:
            pass
        try:
            return registry.create_draft_alert(
                tab_id=tab_id,
                kind="start_monitoring",
                agent_reasoning="Auto-follow-up after train Approve: start monitoring when ready.",
                proposed_message=f"Start monitoring on '{title or tab_id}'",
                proposed_payload={"title": title or ""},
                severity="INFO",
            )
        except Exception as exc:
            logger.warning("Could not enqueue start_monitoring draft: %s", exc)
            return None

    def _create_custom_tab_from_config(self, config):
        """Persist and open a custom monitoring tab from a config dict."""
        if not isinstance(config, dict) or not config.get("title"):
            raise ValueError("Invalid tab config: title is required")
        tab_id = str(uuid.uuid4())
        self.tab_config_manager.add_config(tab_id, config)
        custom_tab = CustomMonitoringTab(tab_id, config, self)
        tab_index = self.tabs.addTab(custom_tab, config["title"])
        self.custom_tabs[tab_id] = custom_tab
        self.tabs.setCurrentIndex(tab_index)
        if hasattr(self, "refresh_fleet_dashboard"):
            self.refresh_fleet_dashboard()
        if hasattr(self, "_refresh_ops_nav"):
            self._refresh_ops_nav()
        logger.info(
            "Created custom monitoring tab from approved draft: %s (ID: %s)",
            config.get("title"),
            tab_id,
        )
        return tab_id

    def open_first_custom_tab(self, section_index=0):
        """Open a custom tab and switch to an internal section."""
        if not self.custom_tabs:
            QMessageBox.information(
                self, "No Custom Tabs",
                "No custom monitoring tab available. Use + New Tab first."
            )
            return
        first_widget = next(iter(self.custom_tabs.values()))
        for i in range(self.tabs.count()):
            if self.tabs.widget(i) == first_widget:
                self.tabs.setCurrentIndex(i)
                break
        if hasattr(first_widget, "custom_nav_list"):
            try:
                if 0 <= int(section_index) < first_widget.custom_nav_list.count():
                    first_widget.custom_nav_list.setCurrentRow(int(section_index))
            except Exception:
                pass

    def _qt_slot_adapter(self, host, method_name, invoke_callback):
        """Bind Qt signals; ignore spurious QPushButton 'checked' args when method takes only self."""
        tool_method = getattr(SecureAnomalyDetectionTool, method_name, None)
        if tool_method is None:
            tool_method = getattr(self, method_name)
        arity = len(inspect.signature(tool_method).parameters)

        if host is self:
            bound = getattr(self, method_name)
            if arity <= 1:
                return lambda *_args, **_kwargs: bound()
            return bound

        if arity <= 1:
            return lambda *_args, **_kwargs: invoke_callback(host, method_name)
        return lambda *args, **kwargs: invoke_callback(host, method_name, *args, **kwargs)

    def _analysis_slot(self, host, method_name):
        """Bind Analysis/ML handlers to main window or custom-tab host."""
        return self._qt_slot_adapter(host, method_name, self._invoke_tab_analysis_method)

    # Methods delegated onto CustomMonitoringTab when running Analysis/ML panel UI.
    _TAB_ANALYSIS_HELPER_NAMES = (
        "get_selected_models", "select_all_models", "clear_all_models",
        "update_model_parameters", "open_parameter_dialog", "_extract_model_parameters",
        "_create_model_parameters", "_get_default_parameters", "_train_next_model",
        "on_single_model_trained", "_on_all_models_trained", "toggle_auto_loading",
        "auto_load_and_process", "on_monitoring_mode_changed", "add_monitoring_channel",
        "remove_monitoring_channel", "clear_monitoring_channels", "multi_channel_monitoring_loop",
        "browse_auto_folder", "browse_prediction_file", "predict_anomalies", "test_on_training_data",
        "toggle_auto_processing", "auto_process_model", "show_action_log", "_is_valid_telemetry_file",
        "get_monitoring_interval_seconds", "on_prediction_completed", "auto_load_latest",
        "update_viz_model_info", "on_model_trained",
    )

    def _ensure_tab_analysis_helpers(self, host):
        """Bind main-window Analysis/ML helpers onto a custom tab host."""
        self._ensure_tab_panel_helpers(host)

    def _invoke_tab_analysis_method(self, host, method_name, *args, **kwargs):
        """Run a SecureAnomalyDetectionTool analysis method with custom-tab as self."""
        if host is not self:
            self._ensure_tab_panel_helpers(host)
        else:
            self._ensure_tab_tool_context(host)
        method = getattr(SecureAnomalyDetectionTool, method_name, None)
        if method is None:
            method = getattr(self, method_name)
        return method(host, *args, **kwargs)

    def export_fleet_mission_report(self):
        """Export combined mission report for all custom tabs."""
        if not self.custom_tabs:
            QMessageBox.information(self, "Mission Report", "No custom monitoring tabs to export.")
            return
        if export_fleet_mission_report is None:
            QMessageBox.warning(self, "Mission Report", "Fleet report module is not available.")
            return
        snapshots = {
            tab_id: widget.get_snapshot() if hasattr(widget, "get_snapshot") else {}
            for tab_id, widget in self.custom_tabs.items()
        }
        report_file = export_fleet_mission_report(REPORTS_DIR, snapshots)
        QMessageBox.information(self, "Mission Report", f"Fleet report exported:\n{report_file}")
    
    def setup_toolbar(self):
        """Setup toolbar: Home / Administration / + New Tab (left), styled actions."""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
            spacing: 8px;
            padding: 8px 12px;
            background-color: #0f172a;
            border-bottom: 1px solid #1e293b;
            }
            QToolButton {
            background-color: transparent;
            color: #e2e8f0;
            border: 1px solid #334155;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 600;
            }
            QToolButton:hover {
            background-color: #1e293b;
            border-color: #64748b;
            color: #ffffff;
            }
            QToolButton:pressed {
            background-color: #334155;
            }
        """)

        home_action = QAction("Home", self)
        home_action.setToolTip("Fleet overview (Home)")
        home_action.triggered.connect(self._show_home_tab)
        toolbar.addAction(home_action)

        if hasattr(self, "admin_tab") and self.tabs.indexOf(self.admin_tab) >= 0:
            admin_action = QAction("Administration", self)
            admin_action.setToolTip("User and system administration")
            admin_action.triggered.connect(self._show_administration_tab)
            toolbar.addAction(admin_action)

        toolbar.addSeparator()

        if self.service_layer.authorize(self.current_username, "manage_users", resource="admin/users"):
            add_tab_action = toolbar.addAction("+ New Tab")
            add_tab_action.setToolTip("Add New Monitoring Tab")
            add_tab_action.triggered.connect(self.add_custom_tab)
            add_tab_button = toolbar.widgetForAction(add_tab_action)
            if add_tab_button:
                add_tab_button.setStyleSheet("""
                    QToolButton {
                        background-color: #4CAF50;
                        color: white;
                        border: 1px solid #45a049;
                        padding: 6px 14px;
                        border-radius: 4px;
                        font-size: 13px;
                        font-weight: bold;
                    }
                    QToolButton:hover {
                        background-color: #45a049;
                        border-color: #3d8b40;
                    }
                    QToolButton:pressed {
                        background-color: #3d8b40;
                        border-color: #2d6b30;
                    }
                """)
            toolbar.addSeparator()

        self.addToolBar(toolbar)
        self._hide_primary_tab_bar_entries()

    def _show_home_tab(self):
        idx = self.tabs.indexOf(self.dashboard_tab)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)
        if hasattr(self, "home_subtabs"):
            self.home_subtabs.setCurrentIndex(0)
        self._sync_ops_nav_active()

    def _show_ai_assistant(self):
        self._show_home_tab()
        if hasattr(self, "home_subtabs"):
            self.home_subtabs.setCurrentIndex(1)
        self._sync_ops_nav_active()

    def _show_administration_tab(self):
        if hasattr(self, "admin_tab"):
            idx = self.tabs.indexOf(self.admin_tab)
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
        self._sync_ops_nav_active()

    def _on_ops_nav(self, key: str):
        key = str(key or "")
        if key == "home":
            self._show_home_tab()
        elif key == "admin":
            self._show_administration_tab()
        elif key == "new_tab":
            self.add_custom_tab()
        elif key.startswith("tab:"):
            tab_id = key.split(":", 1)[-1]
            widget = (getattr(self, "custom_tabs", None) or {}).get(tab_id)
            if widget is not None:
                idx = self.tabs.indexOf(widget)
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            self._sync_ops_nav_active()

    def _on_home_subtab_changed(self, index: int):
        self._sync_ops_nav_active()

    def _refresh_ops_nav(self):
        nav = getattr(self, "ops_nav", None)
        if nav is None:
            return
        nav.set_user(self.current_username, self.current_role)
        can_create = self.service_layer.authorize(
            self.current_username, "manage_users", resource="admin/users"
        )
        nav.set_can_create_tab(bool(can_create))
        nav.set_admin_visible(hasattr(self, "admin_tab") and self.tabs.indexOf(self.admin_tab) >= 0)
        rows = []
        for tab_id, widget in (getattr(self, "custom_tabs", None) or {}).items():
            title = (getattr(widget, "config", None) or {}).get("title") or tab_id
            rows.append((tab_id, title))
        nav.set_telemetry_tabs(rows)
        self._sync_ops_nav_active()

    def _sync_ops_nav_active(self):
        nav = getattr(self, "ops_nav", None)
        if nav is None or not hasattr(self, "tabs"):
            return
        current = self.tabs.currentWidget()
        if current is getattr(self, "dashboard_tab", None):
            sub = getattr(self, "home_subtabs", None)
            key = "ai" if sub is not None and sub.currentIndex() == 1 else "home"
        elif current is getattr(self, "admin_tab", None):
            key = "admin"
        else:
            key = "home"
            for tab_id, widget in (getattr(self, "custom_tabs", None) or {}).items():
                if widget is current:
                    key = f"tab:{tab_id}"
                    break
        nav.set_active(key)

    def _hide_primary_tab_bar_entries(self):
        """Tab bar is hidden; keep Home selected after load."""
        if hasattr(self, "tabs") and self.tabs.tabBar() is not None:
            self.tabs.tabBar().hide()
        self._show_home_tab()
        self._refresh_ops_nav()
    
    def setup_custom_tabs_button(self):
        """Placeholder - button is now added to toolbar in setup_toolbar()"""
        # Button moved to toolbar for better clickability
        pass
    
    def add_custom_tab(self):
        """Open dialog to create a new custom monitoring tab"""
        dialog = TabConfigurationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.config
            
            # Generate unique tab ID
            tab_id = str(uuid.uuid4())
            
            # Save configuration
            self.tab_config_manager.add_config(tab_id, config)
            
            custom_tab = CustomMonitoringTab(tab_id, config, self)
            tab_index = self.tabs.addTab(custom_tab, config['title'])
            self.custom_tabs[tab_id] = custom_tab
            self.tabs.setCurrentIndex(tab_index)
            self.refresh_fleet_dashboard()
            self._refresh_ops_nav()
            
            logger.info(f"Created new custom monitoring tab: {config['title']} (ID: {tab_id})")
    
    def load_custom_tabs(self):
        """Load existing custom tabs from saved configurations"""
        config_ids = self.tab_config_manager.list_configs()
        for tab_id in config_ids:
            config = self.tab_config_manager.get_config(tab_id)
            if config:
                try:
                    custom_tab = CustomMonitoringTab(tab_id, config, self)
                    self.tabs.addTab(custom_tab, config['title'])
                    self.custom_tabs[tab_id] = custom_tab
                    logger.info(f"Loaded custom tab: {config['title']} (ID: {tab_id})")
                except Exception as e:
                    logger.error(f"Error loading custom tab {tab_id}: {str(e)}")
        if hasattr(self, "refresh_fleet_dashboard"):
            self.refresh_fleet_dashboard()
        self._hide_primary_tab_bar_entries()

    def on_tab_context_menu(self, position):
        """Show context menu for custom tabs (right-click)."""
        if not self.service_layer.authorize(self.current_username, "manage_users", resource="admin/users"):
            return

        tab_index = self.tabs.tabBar().tabAt(position)
        if tab_index < 0:
            return

        tab_widget = self.tabs.widget(tab_index)
        tab_id = None
        for tid, widget in self.custom_tabs.items():
            if widget == tab_widget:
                tab_id = tid
                break

        if not tab_id:
            return

        menu = QMenu(self)
        delete_action = menu.addAction("Delete Tab")
        delete_action.triggered.connect(lambda: self.remove_custom_tab(tab_id))
        menu.exec_(self.tabs.tabBar().mapToGlobal(position))
    
    def remove_custom_tab(self, tab_id):
        """Remove a custom tab (admin only)"""
        if not self.service_layer.authorize(self.current_username, "delete_tab", resource=f"tab:{tab_id}"):
            QMessageBox.warning(self, "Permission Denied", 
                              "Only administrators can remove custom tabs.")
            return False
        
        if tab_id in self.custom_tabs:
            # Stop monitoring if active
            custom_tab = self.custom_tabs[tab_id]
            if hasattr(custom_tab, 'monitoring_active') and custom_tab.monitoring_active:
                reply = QMessageBox.question(
                    self,
                    "Stop Monitoring?",
                    f"Tab '{custom_tab.config['title']}' is currently monitoring. "
                    "Stop monitoring and remove tab?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
                custom_tab.stop_monitoring()
            
            for i in range(self.tabs.count()):
                if self.tabs.widget(i) == custom_tab:
                    self.tabs.removeTab(i)
                    break
            
            # Remove from tracking
            del self.custom_tabs[tab_id]
            
            # Remove from config
            self.tab_config_manager.remove_config(tab_id)
            
            logger.info(f"Removed custom tab: {tab_id}")
            if hasattr(self, "_refresh_ops_nav"):
                self._refresh_ops_nav()
            return True
        
        return False
        
    
    def logout(self):
        """Handle user logout and return to login screen"""
        try:
            # Confirm logout if monitoring is active
            monitoring_active = (
                hasattr(self, 'auto_load_timer') and self.auto_load_timer.isActive() or
                hasattr(self, 'auto_process_timer') and self.auto_process_timer.isActive()
            )
            
            if monitoring_active:
                reply = QMessageBox.question(
                    self,
                    "Confirm Logout",
                    "Active monitoring will be stopped. Are you sure you want to logout?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            # Stop monitoring timers
            if hasattr(self, 'auto_load_timer') and self.auto_load_timer.isActive():
                self.auto_load_timer.stop()
            if hasattr(self, 'auto_process_timer') and self.auto_process_timer.isActive():
                self.auto_process_timer.stop()
            
            # Log the logout
            logger.info(f"User {self.current_username} logged out")
            write_audit_event(
                actor=self.current_username,
                action="session_end",
                resource="auth/session",
                outcome="success",
                details={"session_id": self.current_session_id},
                auth_provider=self.current_auth_provider
            )
            
            # Hide the main window
            self.hide()
            
            # Show login dialog
            login_dialog = LoginDialog(self.user_manager)
            result = login_dialog.exec_()
            
            if result == QDialog.Accepted and login_dialog.authenticated:
                # Update user information
                self.current_username = login_dialog.username
                self.current_role = login_dialog.role
                self.current_auth_provider = getattr(login_dialog, "auth_provider", "local")
                self.current_session_id = uuid.uuid4().hex
                write_audit_event(
                    actor=self.current_username,
                    action="session_start",
                    resource="auth/session",
                    outcome="success",
                    details={"role": self.current_role, "session_id": self.current_session_id},
                    auth_provider=self.current_auth_provider
                )
                
                # Reset UI for new user
                self.update_ui_permissions()
                self.data_processor = DataProcessor()
                self.model = None
                
                # Clear displays
                self.update_data_preview()
                if hasattr(self, 'selected_model_label'):
                    self.selected_model_label.setText("No model loaded")
                
                # Update status bar
                self.status_bar.showMessage(
                    f"Logged in as {self.current_username} ({self.current_role}) via {self.current_auth_provider}"
                )
                
                # Show main window again
                self.showMaximized()
                self.reset_inactivity_timer()
                self.start_session_timer()
                
                # Log the new login
                logger.info(f"User {self.current_username} logged in")
            else:
                # If login failed or was cancelled, close the application
                self.close()
                QApplication.quit()
                
        except Exception as e:
            logger.error(f"Error during logout: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error during logout: {str(e)}")
            self.close()
            QApplication.quit()

    def setup_dashboard_tab(self):
        # Home = flat underline sub-tabs (Material-style), not folder tabs
        main_layout = QVBoxLayout(self.dashboard_tab)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(0)

        home_card = QFrame()
        home_card.setObjectName("homeCard")
        home_card.setStyleSheet(
            """
            QFrame#homeCard {
                background-color: #ffffff;
                border: 1px solid #e6e6e6;
                border-radius: 8px;
            }
            """
        )
        home_card.setFrameShape(QFrame.NoFrame)
        card_layout = QVBoxLayout(home_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        self.home_subtabs = QTabWidget()
        self.home_subtabs.setObjectName("homeSubTabs")
        self.home_subtabs.setDocumentMode(True)
        self.home_subtabs.tabBar().setExpanding(False)
        self.home_subtabs.tabBar().setDrawBase(False)
        self.home_subtabs.setStyleSheet(
            """
            QTabWidget#homeSubTabs {
                background: transparent;
                border: none;
            }
            QTabWidget#homeSubTabs::pane {
                border: none;
                border-top: 1px solid #eeeeee;
                background: #ffffff;
                top: 0px;
                margin: 0px;
                padding: 8px;
            }
            QTabWidget#homeSubTabs QTabBar {
                background: transparent;
                border: none;
                alignment: left;
            }
            QTabWidget#homeSubTabs QTabBar::tab {
                background: transparent;
                color: #616161;
                padding: 12px 20px 10px 20px;
                margin: 0px 2px 0px 0px;
                border: none;
                border-bottom: 3px solid transparent;
                font-size: 13px;
                font-weight: 400;
                min-width: 0px;
            }
            QTabWidget#homeSubTabs QTabBar::tab:selected {
                color: #212121;
                font-weight: 700;
                border: none;
                border-bottom: 3px solid #212121;
                background: transparent;
            }
            QTabWidget#homeSubTabs QTabBar::tab:hover:!selected {
                color: #424242;
                background: transparent;
                border-bottom: 3px solid transparent;
            }
            """
        )
        card_layout.addWidget(self.home_subtabs, 1)
        main_layout.addWidget(home_card, 1)

        # ----- Sub-tab 1: AI Assistant + Pending Agent Drafts -----
        agent_page = QWidget()
        agent_page_layout = QVBoxLayout(agent_page)
        agent_page_layout.setSpacing(10)
        agent_page_layout.setContentsMargins(10, 10, 10, 10)

        agent_view_group = QGroupBox("SatOps Agent")
        agent_view_outer = QVBoxLayout()
        agent_splitter = QSplitter(Qt.Horizontal)
        agent_splitter.setChildrenCollapsible(False)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        self.agent_chat_new_btn = QPushButton("New")
        self.agent_chat_new_btn.setToolTip("Start a new conversation")
        self.agent_chat_new_btn.clicked.connect(self._new_agent_session)
        left_layout.addWidget(self.agent_chat_new_btn)

        search_row = QHBoxLayout()
        self.agent_chat_search_input = QLineEdit()
        self.agent_chat_search_input.setPlaceholderText("Search conversations…")
        self.agent_chat_search_input.textChanged.connect(self._filter_agent_sessions)
        search_row.addWidget(self.agent_chat_search_input, 1)
        left_layout.addLayout(search_row)

        self.agent_chat_session_list = QListWidget()
        self.agent_chat_session_list.setMinimumWidth(220)
        self.agent_chat_session_list.setMinimumHeight(280)
        self.agent_chat_session_list.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.agent_chat_session_list.setAlternatingRowColors(True)
        self.agent_chat_session_list.currentItemChanged.connect(self._on_agent_session_item_changed)
        left_layout.addWidget(self.agent_chat_session_list, 1)

        left_btn_row = QHBoxLayout()
        self.agent_chat_remove_btn = QPushButton("Remove")
        self.agent_chat_remove_btn.clicked.connect(self._remove_agent_session)
        self.agent_chat_clear_btn = QPushButton("Clear All")
        self.agent_chat_clear_btn.clicked.connect(self._clear_all_agent_sessions)
        left_btn_row.addWidget(self.agent_chat_remove_btn)
        left_btn_row.addWidget(self.agent_chat_clear_btn)
        left_layout.addLayout(left_btn_row)

        right_panel = QWidget()
        agent_view_layout = QVBoxLayout(right_panel)
        agent_view_layout.setContentsMargins(4, 4, 4, 4)
        agent_view_layout.setSpacing(6)
        agent_header = QHBoxLayout()
        self.agent_view_title_label = QLabel("SatOps Agent")
        self.agent_view_title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.agent_llm_status_label = QLabel("checking…")
        self.agent_view_updated_label = QLabel("last update —")
        agent_header.addWidget(self.agent_view_title_label)
        agent_header.addWidget(self.agent_llm_status_label)
        agent_header.addStretch()
        agent_header.addWidget(self.agent_view_updated_label)
        agent_view_layout.addLayout(agent_header)

        self.mllm_summary_label = QLabel("")
        self.mllm_summary_label.setWordWrap(True)
        self.mllm_summary_label.setVisible(False)
        self.mllm_summary_label.setStyleSheet(
            "QLabel { color: #334155; background: #eef2f7; padding: 8px 10px; "
            "border-radius: 6px; font-size: 12px; }"
        )
        agent_view_layout.addWidget(self.mllm_summary_label)

        self.agent_view_text = QTextBrowser()
        self.agent_view_text.setReadOnly(True)
        self.agent_view_text.setOpenExternalLinks(False)
        self.agent_view_text.setMinimumHeight(360)
        self.agent_view_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._agent_chat_messages = []
        self.agent_view_text.setHtml(
            "<html><body style='background:#ffffff;color:#555555;font-style:italic;"
            "padding:28px;text-align:center;font-size:15px;'>"
            "Chat history appears here. Ask a question or click Analyze Fleet."
            "</body></html>"
        )
        agent_view_layout.addWidget(self.agent_view_text, 1)
        self._agent_thinking_line = None
        self._agent_chat_busy = False

        self._agent_chat_attachments = []  # list[str] absolute paths

        attach_bar = QHBoxLayout()
        attach_bar.setSpacing(8)
        self.agent_chat_attach_label = QLabel("")
        self.agent_chat_attach_label.setWordWrap(True)
        self.agent_chat_attach_label.setStyleSheet(
            "QLabel { color: #455a64; font-size: 12px; padding: 2px 0; }"
        )
        self.agent_chat_attach_clear_btn = QPushButton("Clear files")
        self.agent_chat_attach_clear_btn.setVisible(False)
        self.agent_chat_attach_clear_btn.setStyleSheet(
            "QPushButton {"
            "  padding: 2px 8px; font-size: 11px; min-height: 22px; max-height: 24px;"
            "  border: 1px solid #d0d5db; border-radius: 4px; background: #fafafa; color: #455a64;"
            "}"
            "QPushButton:hover { background: #f0f2f5; }"
        )
        self.agent_chat_attach_clear_btn.clicked.connect(self._clear_agent_chat_attachments)
        attach_bar.addWidget(self.agent_chat_attach_label, 1)
        attach_bar.addWidget(self.agent_chat_attach_clear_btn)
        agent_view_layout.addLayout(attach_bar)

        chat_row = QHBoxLayout()
        chat_row.setSpacing(8)
        self.agent_chat_attach_btn = QPushButton("+")
        self.agent_chat_attach_btn.setToolTip(
            "Attach files for the agent to read (txt, md, csv, json, pdf, docx, …)"
        )
        self.agent_chat_attach_btn.setMinimumHeight(48)
        self.agent_chat_attach_btn.setMinimumWidth(48)
        self.agent_chat_attach_btn.setMaximumWidth(52)
        self.agent_chat_attach_btn.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.agent_chat_attach_btn.setStyleSheet(
            "QPushButton {"
            "  padding: 6px;"
            "  font-size: 20px;"
            "  font-weight: 700;"
            "  border: 1px solid #c8cdd3;"
            "  border-radius: 6px;"
            "  background: #f5f7f9;"
            "  color: #1976d2;"
            "}"
            "QPushButton:hover { background: #e8eef4; }"
            "QPushButton:disabled { color: #9e9e9e; }"
        )
        self.agent_chat_attach_btn.clicked.connect(self._attach_agent_chat_files)
        self.agent_chat_input = QLineEdit()
        self.agent_chat_input.setPlaceholderText("Ask about your fleet…")
        self.agent_chat_input.setMinimumHeight(48)
        self.agent_chat_input.setFont(QFont("Segoe UI", 14))
        self.agent_chat_input.setStyleSheet(
            "QLineEdit {"
            "  padding: 10px 14px;"
            "  font-size: 15px;"
            "  border: 1px solid #c8cdd3;"
            "  border-radius: 6px;"
            "  background: #ffffff;"
            "}"
            "QLineEdit:focus { border: 1px solid #1976d2; }"
        )
        self.agent_chat_input.returnPressed.connect(self.send_agent_chat)
        self.agent_chat_send_btn = QPushButton("Ask")
        self.agent_chat_send_btn.setMinimumHeight(48)
        self.agent_chat_send_btn.setMinimumWidth(78)
        self.agent_chat_send_btn.setFont(QFont("Segoe UI", 13))
        self.agent_chat_send_btn.setStyleSheet(
            "QPushButton {"
            "  padding: 10px 18px;"
            "  font-size: 14px;"
            "  font-weight: 600;"
            "  border: 1px solid #c8cdd3;"
            "  border-radius: 6px;"
            "  background: #f5f7f9;"
            "}"
            "QPushButton:hover { background: #e8eef4; }"
        )
        self.agent_chat_send_btn.clicked.connect(self.send_agent_chat)
        chat_row.addWidget(self.agent_chat_attach_btn)
        chat_row.addWidget(self.agent_chat_input, 1)
        chat_row.addWidget(self.agent_chat_send_btn)
        agent_view_layout.addLayout(chat_row)

        agent_btn_row = QHBoxLayout()
        agent_btn_row.setSpacing(6)
        agent_btn_row.setContentsMargins(0, 4, 0, 0)
        _agent_action_btn_style = (
            "QPushButton {"
            "  padding: 3px 10px;"
            "  font-size: 11px;"
            "  min-height: 24px;"
            "  max-height: 26px;"
            "  border: 1px solid #d0d5db;"
            "  border-radius: 4px;"
            "  background: #fafafa;"
            "  color: #455a64;"
            "}"
            "QPushButton:hover { background: #f0f2f5; }"
        )
        self.agent_view_decisions_btn = QPushButton("View Decisions")
        self.agent_view_decisions_btn.setStyleSheet(_agent_action_btn_style)
        self.agent_view_decisions_btn.clicked.connect(self.show_agent_decisions_dialog)
        self.agent_view_refresh_btn = QPushButton("Analyze Fleet")
        self.agent_view_refresh_btn.setStyleSheet(_agent_action_btn_style)
        self.agent_view_refresh_btn.clicked.connect(self.refresh_agent_view_now)
        self.agent_rebuild_knowledge_btn = QPushButton("Rebuild Knowledge")
        self.agent_rebuild_knowledge_btn.setStyleSheet(_agent_action_btn_style)
        self.agent_rebuild_knowledge_btn.setToolTip(
            "Export tab FSM mission modes → data/knowledge/space/, then index "
            "space/ and ground/ into separate RAG stores (requires Ollama nomic-embed-text)."
        )
        self.agent_rebuild_knowledge_btn.clicked.connect(self.rebuild_agent_knowledge_index)
        agent_btn_row.addWidget(self.agent_view_decisions_btn)
        agent_btn_row.addWidget(self.agent_view_refresh_btn)
        agent_btn_row.addWidget(self.agent_rebuild_knowledge_btn)
        agent_btn_row.addStretch()
        agent_view_layout.addLayout(agent_btn_row)

        agent_splitter.addWidget(left_panel)
        agent_splitter.addWidget(right_panel)
        agent_splitter.setStretchFactor(0, 0)
        agent_splitter.setStretchFactor(1, 1)
        agent_splitter.setSizes([280, 720])
        agent_splitter.setMinimumHeight(420)
        agent_view_outer.addWidget(agent_splitter, 1)
        agent_view_group.setLayout(agent_view_outer)
        agent_view_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        agent_page_layout.addWidget(agent_view_group, 4)

        drafts_group = QGroupBox("Pending Agent Drafts")
        drafts_layout = QVBoxLayout()
        self.draft_alerts_summary_label = QLabel("No pending agent drafts.")
        self.draft_alerts_summary_label.setWordWrap(True)
        drafts_layout.addWidget(self.draft_alerts_summary_label)
        self.pending_actions_summary_label = QLabel("Pending actions: 0 draft(s) · watchlist=0")
        self.pending_actions_summary_label.setWordWrap(True)
        self.pending_actions_summary_label.setStyleSheet("color: #555; font-size: 11px;")
        drafts_layout.addWidget(self.pending_actions_summary_label)
        self.draft_alerts_table = QTableWidget(0, 8)
        self.draft_alerts_table.setHorizontalHeaderLabels([
            "ID", "Kind", "Tab", "Severity", "Message", "Reasoning", "Created", "Action",
        ])
        self.draft_alerts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.draft_alerts_table.setAlternatingRowColors(True)
        self.draft_alerts_table.setMinimumHeight(140)
        self.draft_alerts_table.setMaximumHeight(260)
        self.draft_alerts_table.verticalHeader().setVisible(False)
        drafts_layout.addWidget(self.draft_alerts_table)
        drafts_btn_row = QHBoxLayout()
        drafts_refresh_btn = QPushButton("Refresh Drafts")
        drafts_refresh_btn.clicked.connect(self.refresh_draft_alerts_panel)
        self.draft_approve_all_btn = QPushButton("Approve All")
        self.draft_approve_all_btn.setToolTip("Approve every pending agent draft (runs each action)")
        self.draft_approve_all_btn.clicked.connect(self._approve_all_draft_alerts)
        self.draft_reject_all_btn = QPushButton("Reject All")
        self.draft_reject_all_btn.setToolTip("Reject / dismiss every pending agent draft")
        self.draft_reject_all_btn.clicked.connect(self._reject_all_draft_alerts)
        drafts_btn_row.addWidget(drafts_refresh_btn)
        drafts_btn_row.addWidget(self.draft_approve_all_btn)
        drafts_btn_row.addWidget(self.draft_reject_all_btn)
        drafts_btn_row.addStretch()
        drafts_layout.addLayout(drafts_btn_row)
        drafts_group.setLayout(drafts_layout)
        agent_page_layout.addWidget(drafts_group, 0)

        self._agent_sessions = []
        self._agent_active_session_id = None
        self._agent_session_switching = False
        self._init_agent_chat_sessions()
        self._refresh_agent_llm_status()

        # ----- Sub-tab: Fleet Overview (default, left) -----
        overview_page = QWidget()
        overview_scroll = QScrollArea()
        overview_scroll.setWidgetResizable(True)
        overview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        overview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        overview_outer = QVBoxLayout(overview_page)
        overview_outer.setContentsMargins(0, 0, 0, 0)
        overview_outer.addWidget(overview_scroll)

        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        welcome_group = QGroupBox("Welcome")
        welcome_layout = QVBoxLayout()
        welcome_label = QLabel(f"Welcome {self.current_username}. You are logged in as: {self.current_role}")
        welcome_label.setFont(QFont('Arial', 10))
        welcome_label.setWordWrap(True)
        welcome_layout.addWidget(welcome_label)
        welcome_group.setLayout(welcome_layout)
        layout.addWidget(welcome_group)

        status_group = QGroupBox("System Status")
        status_layout = QFormLayout()
        self.system_status_label = QLabel("System operational")
        self.system_status_label.setWordWrap(True)
        last_update_label = QLabel(QDateTime.currentDateTime().toString())
        last_update_label.setWordWrap(True)
        status_layout.addRow("Status:", self.system_status_label)
        status_layout.addRow("Last Update:", last_update_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        fleet_group = QGroupBox("Monitoring Tabs Overview (fleet status & snapshots)")
        fleet_layout = QVBoxLayout()
        self.fleet_tabs_table = QTableWidget(0, 8)
        self.fleet_tabs_table.setHorizontalHeaderLabels([
            "Tab", "Monitoring", "Health", "Watch Status", "Last File",
            "Models", "OBS", "Updated",
        ])
        self.fleet_tabs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.fleet_tabs_table.setAlternatingRowColors(True)
        self.fleet_tabs_table.setMinimumHeight(260)
        self.fleet_tabs_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fleet_tabs_table.verticalHeader().setDefaultSectionSize(34)
        self.fleet_tabs_table.verticalHeader().setVisible(False)
        fleet_layout.addWidget(self.fleet_tabs_table)
        fleet_report_btn = QPushButton("Export Fleet Mission Report")
        fleet_report_btn.clicked.connect(self.export_fleet_mission_report)
        fleet_layout.addWidget(fleet_report_btn)
        fleet_group.setLayout(fleet_layout)
        layout.addWidget(fleet_group, 1)

        retrain_group = QGroupBox("Pending Retrain Signals")
        retrain_layout = QVBoxLayout()
        self.retrain_signals_summary_label = QLabel("No pending retrain signals.")
        self.retrain_signals_summary_label.setWordWrap(True)
        retrain_layout.addWidget(self.retrain_signals_summary_label)
        self.retrain_signals_table = QTableWidget(0, 8)
        self.retrain_signals_table.setHorizontalHeaderLabels([
            "ID", "Tab", "Model", "Drift", "Features", "Reason", "Detected", "Action",
        ])
        self.retrain_signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.retrain_signals_table.setAlternatingRowColors(True)
        self.retrain_signals_table.setMinimumHeight(140)
        self.retrain_signals_table.setMaximumHeight(220)
        self.retrain_signals_table.verticalHeader().setVisible(False)
        retrain_layout.addWidget(self.retrain_signals_table)
        retrain_refresh_btn = QPushButton("Refresh Retrain Signals")
        retrain_refresh_btn.clicked.connect(self.refresh_retrain_signals_panel)
        retrain_layout.addWidget(retrain_refresh_btn)
        retrain_group.setLayout(retrain_layout)
        layout.addWidget(retrain_group, 0)

        activity_group = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout()
        self.activity_table = QTableWidget(0, 3)
        self.activity_table.setHorizontalHeaderLabels(["Time", "Action", "Details"])
        self.activity_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.activity_table.setMinimumHeight(110)
        self.activity_table.setMaximumHeight(190)
        self.activity_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.activity_table.verticalHeader().setDefaultSectionSize(30)
        self.activity_table.verticalHeader().setMinimumSectionSize(26)
        self.activity_table.verticalHeader().setVisible(False)
        self.activity_table.setAlternatingRowColors(True)
        activity_layout.addWidget(self.activity_table)
        activity_group.setLayout(activity_layout)
        layout.addWidget(activity_group, 0)

        self.load_recent_activity()
        self.refresh_retrain_signals_panel()
        self.refresh_draft_alerts_panel()

        self.activity_table.resizeRowsToContents()
        self.activity_table.update()
        self.activity_table.repaint()

        actions_group = QGroupBox("Quick Actions")
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)

        import_button = QPushButton("Open Data Import")
        import_button.clicked.connect(lambda: self.open_first_custom_tab(section_index=3))
        import_button.setMinimumWidth(80)

        analyze_button = QPushButton("Open Analysis / ML")
        analyze_button.clicked.connect(lambda: self.open_first_custom_tab(section_index=2))
        analyze_button.setMinimumWidth(80)

        report_button = QPushButton("Fleet Report")
        report_button.clicked.connect(self.export_fleet_mission_report)
        report_button.setMinimumWidth(100)

        actions_layout.addWidget(import_button)
        actions_layout.addWidget(analyze_button)
        actions_layout.addWidget(report_button)
        actions_layout.addStretch()

        actions_group_layout = QVBoxLayout()
        actions_group_layout.addWidget(actions_widget)
        actions_group.setLayout(actions_group_layout)
        layout.addWidget(actions_group)

        logout_button = QPushButton("Logout")
        logout_button.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                color: #b91c1c;
                padding: 8px 16px;
                border: 1px solid #fecaca;
                border-radius: 6px;
                font-size: 13px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #fef2f2;
            }
        """)
        logout_button.clicked.connect(self.logout)
        layout.addWidget(logout_button)

        layout.addStretch()

        overview_scroll.setWidget(container_widget)
        # Order: Fleet Overview (default) → AI Assistant
        self.home_subtabs.addTab(overview_page, "Fleet Overview")
        self.home_subtabs.addTab(agent_page, "AI Assistant")
        self.home_subtabs.setCurrentIndex(0)
        
        
    def setup_data_tab(self):
        layout = QVBoxLayout(self.data_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.build_data_import_panel(self, layout)

    def _data_import_stop_slot(self, host):
        if host is self:
            return self.stop_monitoring
        return lambda: self._invoke_tab_data_import_method(host, "stop_data_import_monitoring")

    def _data_slot(self, host, method_name):
        """Bind Data Import handlers to main window or custom-tab host."""
        return self._qt_slot_adapter(host, method_name, self._invoke_tab_data_import_method)

    _TAB_DATA_IMPORT_HELPER_NAMES = (
        "browse_data_file", "load_data", "preprocess_data", "clean_data",
        "browse_log_file", "load_log_file",
        "stop_data_import_monitoring", "update_visualization_features", "refresh_column_analysis",
        "update_feature_list", "on_preprocessing_finished",
    )

    _TAB_PANEL_EXTRA_HELPER_NAMES = (
        "auto_optimize_high_dimensional",
    )

    def _ensure_tab_tool_context(self, host):
        """Shared runtime context when legacy panel methods run on a custom tab."""
        if host is self:
            return
        host.status_bar = self.statusBar()
        host.current_username = self.current_username
        host.current_role = self.current_role
        host.service_layer = getattr(self, "service_layer", None)
        host.user_manager = getattr(self, "user_manager", None)
        host.model_registry = getattr(self, "model_registry", None)
        if not host.data_processor:
            host.data_processor = DataProcessor()
        if not hasattr(host, "trained_models"):
            host.trained_models = {}
        if not hasattr(host, "monitoring_channels"):
            host.monitoring_channels = {}
        if not hasattr(host, "channel_processors"):
            host.channel_processors = {}
        if not hasattr(host, "model_param_widgets"):
            host.model_param_widgets = {}
        if hasattr(host, "stop_btn") and not hasattr(host, "stop_monitoring_btn"):
            host.stop_monitoring_btn = host.stop_btn

    def _bind_tab_tool_method(self, host, name):
        if isinstance(host, CustomMonitoringTab) and name in CustomMonitoringTab._TAB_NATIVE_METHODS:
            return False
        tool_method = getattr(SecureAnomalyDetectionTool, name, None)
        if tool_method is None or not callable(tool_method):
            return False
        setattr(host, name, types.MethodType(tool_method, host))
        return True

    def _ensure_tab_panel_helpers(self, host):
        """Bind all legacy panel methods onto a custom tab host."""
        if host is self or getattr(host, "_tab_panel_helpers_bound", False):
            return
        self._ensure_tab_tool_context(host)
        seen = set()
        for names in (
            self._TAB_DATA_IMPORT_HELPER_NAMES,
            self._TAB_ANALYSIS_HELPER_NAMES,
            self._TAB_VISUALIZATION_HELPER_NAMES,
            self._TAB_PANEL_EXTRA_HELPER_NAMES,
        ):
            for name in names:
                if name in seen:
                    continue
                seen.add(name)
                self._bind_tab_tool_method(host, name)
        host._tab_panel_helpers_bound = True

    def _ensure_tab_data_import_helpers(self, host):
        self._ensure_tab_panel_helpers(host)

    def _invoke_tab_data_import_method(self, host, method_name, *args, **kwargs):
        if host is not self:
            self._ensure_tab_panel_helpers(host)
        else:
            self._ensure_tab_tool_context(host)
        method = getattr(SecureAnomalyDetectionTool, method_name, None)
        if method is None:
            method = getattr(self, method_name)
        return method(host, *args, **kwargs)

    def build_data_import_panel(self, host, parent_layout):
        """Build full Data Import UI on host (main window or custom tab)."""
        from app.tabs.custom_tab.panels.data_import import build_data_import_panel as _build
        from app.tabs.custom_tab.panels.slots import assert_panel_slots_complete
        from app.tabs.custom_tab.widget import build_legacy_panel_slots

        _build(host, parent_layout, slots=assert_panel_slots_complete(build_legacy_panel_slots(self)))

    def setup_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.build_analysis_ml_panel(self, layout)

    def build_analysis_ml_panel(self, host, parent_layout):
        """Build full Analysis/ML UI on host (main window or custom tab)."""
        from app.tabs.custom_tab.panels.analysis_ml import build_analysis_ml_panel as _build
        from app.tabs.custom_tab.panels.slots import assert_panel_slots_complete
        from app.tabs.custom_tab.widget import build_legacy_panel_slots

        _build(host, parent_layout, slots=assert_panel_slots_complete(build_legacy_panel_slots(self)))
    def toggle_auto_loading(self, state):
        """Enable/disable automatic data loading"""
        try:
            if state == Qt.Checked:
                # Check if auto load folder is set
                folder_path = self.auto_folder_input.text()
                if not folder_path or not os.path.isdir(folder_path):
                    QMessageBox.warning(self, "Auto Loading", 
                                      "Please select a valid data folder first.")
                    self.auto_load_check.setChecked(False)
                    return
                
                # Get the loading interval in milliseconds from hours, minutes, and seconds
                hours = self.load_hours_spin.value()
                minutes = self.load_minutes_spin.value()
                seconds = self.load_seconds_spin.value()
                
                # Check if at least one unit is set
                if hours == 0 and minutes == 0 and seconds == 0:
                    QMessageBox.warning(self, "Auto Loading", 
                                      "Please set at least one time unit (hours, minutes, or seconds).")
                    self.auto_load_check.setChecked(False)
                    return
                
                # Convert to milliseconds
                interval_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
                
                # Start the timer with the specified interval
                if not hasattr(self, 'auto_load_timer'):
                    self.auto_load_timer = QTimer(self)
                    self.auto_load_timer.timeout.connect(self.auto_load_and_process)
                
                self.auto_load_timer.start(interval_ms)
                
                # Enable stop monitoring button when available
                if hasattr(self, "stop_monitoring_btn"):
                    self.stop_monitoring_btn.setEnabled(True)
                
                # Create human-readable interval string
                time_parts = []
                if hours > 0:
                    time_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if minutes > 0:
                    time_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
                if seconds > 0:
                    time_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
                interval_str = ", ".join(time_parts)
                
                # Update status
                self.analysis_status_label.setText(
                    f"Auto loading enabled - checking for new data every {interval_str}"
                )
                logger.info(f"Auto loading enabled with interval: {interval_str}")
            else:
                # Stop the timer
                if hasattr(self, 'auto_load_timer'):
                    self.auto_load_timer.stop()
                self.analysis_status_label.setText("Auto loading disabled")
                logger.info("Auto loading disabled")
                
                # Disable stop monitoring button if no monitoring is active
                if not (hasattr(self, 'auto_process_timer') and self.auto_process_timer.isActive()):
                    if hasattr(self, "stop_monitoring_btn"):
                        self.stop_monitoring_btn.setEnabled(False)
                
        except Exception as e:
            logger.error(f"Error toggling auto loading: {str(e)}")
            self.analysis_status_label.setText(f"Error: {str(e)}")
            self.auto_load_check.setChecked(False)

    def _is_valid_telemetry_file(self, file_path):
        """Filter out app config/auth files from telemetry auto-loading."""
        if not isinstance(file_path, str):
            return False
        file_lower = file_path.lower()
        if not file_lower.endswith(('.csv', '.json')):
            return False
        blocked_names = {
            "users.json",
            "auth_config.json",
            "email_config.json",
            "alert_routing_config.json",
            "alert_routing.json",
            "app_config.json",
            "tab_configs.json",
            "settings.json",
        }
        base_name = os.path.basename(file_lower)
        if base_name in blocked_names:
            return False
        return True
    
    def auto_load_and_process(self):
        """Automatically load new data and run model predictions"""
        try:
            # Check monitoring mode
            if hasattr(self, 'monitoring_mode_combo'):
                monitoring_mode = self.monitoring_mode_combo.currentText()
                if monitoring_mode == "Multi-Channel":
                    # Handle multi-channel monitoring
                    self.multi_channel_monitoring_loop()
                    return
            
            # Handle single source monitoring (original logic)
            folder_path = self.auto_folder_input.text()
            if not folder_path or not os.path.isdir(folder_path):
                logger.warning("Auto loading skipped - no valid folder selected")
                return
            
            # Get candidate files in folder
            files = []
            for f in os.listdir(folder_path):
                file_path = os.path.join(folder_path, f)
                if os.path.isfile(file_path):
                    if self._is_valid_telemetry_file(file_path):
                        files.append(file_path)
            
            if not files:
                logger.info("No data files found in auto-load folder")
                return
            
            # Sort by modification time (newest first)
            files.sort(key=os.path.getmtime, reverse=True)

            loaded_any = False
            skipped = 0
            for latest_file in files:
                # Skip file if already processed and unchanged
                if (
                    hasattr(self, 'last_processed_file')
                    and self.last_processed_file == latest_file
                    and os.path.getmtime(latest_file) == getattr(self, 'last_processed_time', None)
                ):
                    continue

                file_extension = os.path.splitext(latest_file)[1].lower()
                if file_extension == '.csv':
                    success, message = self.data_processor.load_csv(latest_file)
                else:
                    success, message = self.data_processor.load_json(latest_file)

                if success:
                    logger.info(f"Successfully loaded new data from {latest_file}")
                    self.last_processed_file = latest_file
                    self.last_processed_time = os.path.getmtime(latest_file)
                    self.update_data_preview()
                    if hasattr(self, 'auto_process_check') and self.auto_process_check.isChecked():
                        self.auto_process_model()
                    loaded_any = True
                    break

                skipped += 1
                logger.warning(f"Skipping file during auto-load ({os.path.basename(latest_file)}): {message}")

            if not loaded_any:
                if skipped > 0:
                    logger.warning("No valid telemetry files loaded from auto-load folder")
                else:
                    logger.info("No new data files to process")
                
        except Exception as e:
            error_msg = f"Error in auto loading: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.analysis_status_label.setText(error_msg)
    
    def on_monitoring_mode_changed(self, mode):
        """Handle monitoring mode changes between single and multi-channel"""
        if mode == "Single Source":
            self.single_source_widget.setVisible(True)
            self.multi_channel_widget.setVisible(False)
        elif mode == "Multi-Channel":
            self.single_source_widget.setVisible(False)
            self.multi_channel_widget.setVisible(True)
            
        self.analysis_status_label.setText(f"Monitoring mode changed to: {mode}")
    
    def add_monitoring_channel(self):
        """Add a new data monitoring channel"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QComboBox, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Monitoring Channel")
        dialog.setModal(True)
        dialog.resize(400, 200)
        
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        
        # Channel name
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Sensor1, Database1, LogFiles")
        form_layout.addRow("Channel Name:", name_input)
        
        # Source type
        source_type_combo = QComboBox()
        source_type_combo.addItems(["Folder", "File", "Database", "API Endpoint"])
        form_layout.addRow("Source Type:", source_type_combo)
        
        # Source path
        path_input = QLineEdit()
        path_input.setPlaceholderText("Path to folder, file, or connection string")
        browse_button = QPushButton("Browse...")
        
        def browse_source():
            if source_type_combo.currentText() == "Folder":
                from PyQt5.QtWidgets import QFileDialog
                folder = QFileDialog.getExistingDirectory(dialog, "Select Data Folder")
                if folder:
                    path_input.setText(folder)
            elif source_type_combo.currentText() == "File":
                from PyQt5.QtWidgets import QFileDialog
                file_path, _ = QFileDialog.getOpenFileName(dialog, "Select Data File", "", "Data Files (*.csv *.json)")
                if file_path:
                    path_input.setText(file_path)
        
        browse_button.clicked.connect(browse_source)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(path_input)
        path_layout.addWidget(browse_button)
        form_layout.addRow("Source Path:", path_layout)
        
        # Data format
        format_combo = QComboBox()
        format_combo.addItems(["Auto-detect", "CSV", "JSON", "XML"])
        form_layout.addRow("Data Format:", format_combo)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec_() == QDialog.Accepted:
            channel_name = name_input.text().strip()
            source_type = source_type_combo.currentText()
            source_path = path_input.text().strip()
            data_format = format_combo.currentText()
            
            if not channel_name or not source_path:
                QMessageBox.warning(self, "Invalid Input", "Channel name and source path are required.")
                return
            
            if channel_name in self.monitoring_channels:
                QMessageBox.warning(self, "Duplicate Channel", f"Channel '{channel_name}' already exists.")
                return
            
            # Add channel to table
            row = self.channels_table.rowCount()
            self.channels_table.insertRow(row)
            self.channels_table.setItem(row, 0, QTableWidgetItem(channel_name))
            self.channels_table.setItem(row, 1, QTableWidgetItem(source_path))
            self.channels_table.setItem(row, 2, QTableWidgetItem(f"{source_type} ({data_format})"))
            self.channels_table.setItem(row, 3, QTableWidgetItem("Ready"))
            
            # Store channel info
            self.monitoring_channels[channel_name] = {
                'source_type': source_type,
                'source_path': source_path,
                'data_format': data_format,
                'status': 'Ready',
                'last_update': None,
                'processor': DataProcessor()  # Each channel gets its own processor
            }
            
            self.channel_processors[channel_name] = DataProcessor()
            
            self.analysis_status_label.setText(f"Added monitoring channel: {channel_name}")
            logger.info(f"Added monitoring channel: {channel_name} -> {source_path}")
    
    def remove_monitoring_channel(self):
        """Remove selected monitoring channel"""
        current_row = self.channels_table.currentRow()
        if current_row >= 0:
            channel_item = self.channels_table.item(current_row, 0)
            if not channel_item:
                QMessageBox.warning(self, "Error", "Could not read channel name.")
                return
            channel_name = channel_item.text()
            
            # Confirm removal
            reply = QMessageBox.question(
                self, "Remove Channel",
                f"Are you sure you want to remove channel '{channel_name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove from table
                self.channels_table.removeRow(current_row)
                
                # Remove from storage
                if channel_name in self.monitoring_channels:
                    del self.monitoring_channels[channel_name]
                if channel_name in self.channel_processors:
                    del self.channel_processors[channel_name]
                
                self.analysis_status_label.setText(f"Removed monitoring channel: {channel_name}")
                logger.info(f"Removed monitoring channel: {channel_name}")
        else:
            QMessageBox.information(self, "No Selection", "Please select a channel to remove.")
    
    def clear_monitoring_channels(self):
        """Clear all monitoring channels"""
        if self.channels_table.rowCount() == 0:
            QMessageBox.information(self, "No Channels", "No channels to clear.")
            return
        
        reply = QMessageBox.question(
            self, "Clear All Channels",
            "Are you sure you want to remove all monitoring channels?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.channels_table.setRowCount(0)
            self.monitoring_channels.clear()
            self.channel_processors.clear()
            
            self.analysis_status_label.setText("All monitoring channels cleared")
            logger.info("Cleared all monitoring channels")
    
    def multi_channel_monitoring_loop(self):
        """Process all monitoring channels for new data"""
        if self.monitoring_mode_combo.currentText() != "Multi-Channel":
            return
        
        if not self.monitoring_channels:
            logger.info("No monitoring channels configured")
            return
        
        try:
            updated_channels = []
            
            for channel_name, channel_info in self.monitoring_channels.items():
                try:
                    # Update channel status
                    self.update_channel_status(channel_name, "Checking...")
                    
                    # Process based on source type
                    source_type = channel_info['source_type']
                    source_path = channel_info['source_path']
                    processor = self.channel_processors[channel_name]
                    
                    success = False
                    
                    if source_type == "Folder":
                        success = self.process_folder_channel(channel_name, source_path, processor)
                    elif source_type == "File":
                        success = self.process_file_channel(channel_name, source_path, processor)
                    elif source_type == "Database":
                        success = self.process_database_channel(channel_name, source_path, processor)
                    elif source_type == "API Endpoint":
                        success = self.process_api_channel(channel_name, source_path, processor)
                    
                    if success:
                        updated_channels.append(channel_name)
                        self.update_channel_status(channel_name, "Updated")
                        channel_info['last_update'] = datetime.datetime.now()
                    else:
                        self.update_channel_status(channel_name, "No new data")
                        
                except Exception as e:
                    error_msg = f"Error processing channel {channel_name}: {str(e)}"
                    logger.error(error_msg)
                    self.update_channel_status(channel_name, "Error")
            
            if updated_channels:
                self.analysis_status_label.setText(
                    f"Multi-channel update: {len(updated_channels)} channels updated - {', '.join(updated_channels)}"
                )
                
                # Run anomaly detection on updated channels if auto-processing is enabled
                if hasattr(self, 'auto_process_check') and self.auto_process_check.isChecked():
                    self.process_multi_channel_anomalies(updated_channels)
            else:
                self.analysis_status_label.setText("Multi-channel check: No new data in any channel")
                
        except Exception as e:
            error_msg = f"Error in multi-channel monitoring: {str(e)}"
            logger.error(error_msg)
            self.analysis_status_label.setText(f"{error_msg}")
    
    def update_channel_status(self, channel_name, status):
        """Update the status of a monitoring channel in the table"""
        for row in range(self.channels_table.rowCount()):
            if self.channels_table.item(row, 0).text() == channel_name:
                self.channels_table.setItem(row, 3, QTableWidgetItem(status))
                break
    
    def process_folder_channel(self, channel_name, folder_path, processor):
        """Process a folder-based monitoring channel"""
        if not os.path.isdir(folder_path):
            return False
        
        # Get candidate files in folder
        files = []
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path) and self._is_valid_telemetry_file(file_path):
                files.append(file_path)
        
        if not files:
            return False
        
        # Sort by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)

        # Check if this is new data
        channel_info = self.monitoring_channels[channel_name]
        for latest_file in files:
            if (
                channel_info.get('last_file') == latest_file
                and os.path.getmtime(latest_file) == channel_info.get('last_file_time')
            ):
                continue

            file_extension = os.path.splitext(latest_file)[1].lower()
            if file_extension == '.csv':
                success, message = processor.load_csv(latest_file)
            else:
                success, message = processor.load_json(latest_file)

            if success:
                channel_info['last_file'] = latest_file
                channel_info['last_file_time'] = os.path.getmtime(latest_file)
                logger.info(f"Channel {channel_name}: Loaded new data from {latest_file}")
                return True

            logger.warning(f"Channel {channel_name}: Skipped invalid data file {os.path.basename(latest_file)} ({message})")

        return False
    
    def process_file_channel(self, channel_name, file_path, processor):
        """Process a single file monitoring channel"""
        if not os.path.exists(file_path):
            return False
        
        # Check if file has been modified
        channel_info = self.monitoring_channels[channel_name]
        current_mtime = os.path.getmtime(file_path)
        
        if channel_info.get('last_file_time') == current_mtime:
            return False
        
        # Load the data
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            success, message = processor.load_csv(file_path)
        elif file_extension == '.json':
            success, message = processor.load_json(file_path)
        else:
            return False
        
        if success:
            channel_info['last_file_time'] = current_mtime
            logger.info(f"Channel {channel_name}: Loaded updated data from {file_path}")
            return True
        
        return False
    
    def process_database_channel(self, channel_name, connection_string, processor):
        """Process a database monitoring channel (placeholder for future implementation)"""
        # This would implement database connectivity
        # For now, return False to indicate no implementation
        logger.info(f"Database monitoring for channel {channel_name} not yet implemented")
        return False
    
    def process_api_channel(self, channel_name, api_endpoint, processor):
        """Process an API endpoint monitoring channel (placeholder for future implementation)"""
        # This would implement API data fetching
        # For now, return False to indicate no implementation
        logger.info(f"API monitoring for channel {channel_name} not yet implemented")
        return False
    
    def process_multi_channel_anomalies(self, updated_channels):
        """Run anomaly detection on updated channels"""
        if not self.model:
            logger.warning("No model available for multi-channel anomaly detection")
            return
        
        try:
            total_anomalies = 0
            channel_results = {}
            
            for channel_name in updated_channels:
                processor = self.channel_processors[channel_name]
                if processor.preprocessed_data is not None:
                    # Run anomaly detection on channel data
                    numeric_data = processor.preprocessed_data.select_dtypes(include=['number'])
                    if not numeric_data.empty:
                        scores, anomalies = self.model.predict(numeric_data)
                        
                        # Store results
                        processor.data["Anomaly Score"] = scores
                        processor.data["Anomaly"] = anomalies
                        
                        num_anomalies = sum(anomalies)
                        total_anomalies += num_anomalies
                        channel_results[channel_name] = {
                            'anomalies': num_anomalies,
                            'total_records': len(processor.data),
                            'anomaly_rate': (num_anomalies / len(processor.data)) * 100 if len(processor.data) > 0 else 0
                        }
                        
                        logger.info(f"Channel {channel_name}: Found {num_anomalies} anomalies in {len(processor.data)} records")
            
            if total_anomalies > 0:
                # Create summary report
                summary = f"ALERT: Multi-channel anomaly detection: {total_anomalies} total anomalies across {len(updated_channels)} channels"
                self.analysis_status_label.setText(summary)
                
                # Log detailed results
                for channel_name, results in channel_results.items():
                    logger.info(f"Channel {channel_name}: {results['anomalies']} anomalies ({results['anomaly_rate']:.2f}% rate)")
                
                # Send email alert if configured
                if hasattr(self, 'send_email_alert'):
                    try:
                        self.send_multi_channel_email_alert(channel_results, total_anomalies)
                    except Exception as e:
                        logger.error(f"Error sending multi-channel email alert: {str(e)}")
            else:
                self.analysis_status_label.setText(f"Multi-channel analysis: No anomalies detected in {len(updated_channels)} channels")
                
        except Exception as e:
            error_msg = f"Error in multi-channel anomaly processing: {str(e)}"
            logger.error(error_msg)
            self.analysis_status_label.setText(f"{error_msg}")
    
    def send_multi_channel_email_alert(self, channel_results, total_anomalies):
        """Send email alert for multi-channel anomaly detection results"""
        alert_span = OBSERVABILITY.start_span(
            "alerts.send_multi_channel_email",
            {"channels": len(channel_results), "total_anomalies": int(total_anomalies)}
        ) if OBSERVABILITY else None
        
        try:
            email_config = self._resolve_email_config()
            config_error = self._get_email_config_error(email_config)
            if config_error:
                logger.warning(f"Multi-channel email alert skipped: {config_error}")
                if OBSERVABILITY:
                    OBSERVABILITY.inc_counter("alerts_sent_total", labels={"channel": "email", "status": "skipped"})
                    OBSERVABILITY.end_span(alert_span, status="error", attributes={"reason": "config_missing"})
                return
            
            sent_count = 0
            for username, user_data in self.user_manager.users.items():
                to_email = str(user_data.get("email", "")).strip()
                if not _is_deliverable_alert_email(to_email):
                    continue

                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                # Create detailed report
                body = f"""
Multi-Channel Anomaly Detection Report


Detection Time: {timestamp}
Total Anomalies: {total_anomalies}
Channels Monitored: {len(channel_results)}

Channel Details:
"""
                    
                for channel_name, results in channel_results.items():
                    body += f"""
- {channel_name}:
  * Anomalies: {results['anomalies']}
  * Total Records: {results['total_records']}
  * Anomaly Rate: {results['anomaly_rate']:.2f}%
"""
                    
                body += f"""
This is an automated alert from the Multi-Channel Telemetry Monitoring Tool.
Alert sent to user: {username} ({user_data["role"]})
"""
                self._send_email_via_config(
                    to_email,
                    "Multi-Channel Anomaly Detection Report",
                    body,
                    email_config
                )
                sent_count += 1
                    
                logger.info(f"Multi-channel detection report email sent to {to_email}")
            
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("alerts_sent_total", sent_count, labels={"channel": "email", "status": "success"})
                OBSERVABILITY.end_span(alert_span, status="ok", attributes={"emails_sent": sent_count})
            
        except Exception as e:
            logger.error(f"Error sending multi-channel email alert: {str(e)}")
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("alerts_sent_total", labels={"channel": "email", "status": "error"})
                OBSERVABILITY.end_span(alert_span, status="error", error=e)

    def setup_health_tab(self):
        """Set up the System Health monitoring tab with debug logging"""
        logger.info("Setting up System Health tab")
        self.debug_status("Before Health Tab Setup")
        
        try:
            # Create main layout
            layout = QVBoxLayout(self.health_tab)
            scroll = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout()
            
            # Add each section using helper methods
            scroll_layout.addWidget(self._create_model_config_section())
            scroll_layout.addWidget(self._create_monitoring_controls())
            scroll_layout.addWidget(self._create_visualization_section())
            
            # Setup scroll area
            scroll_widget.setLayout(scroll_layout)
            scroll.setWidget(scroll_widget)
            scroll.setWidgetResizable(True)
            layout.addWidget(scroll)
            
            # Initialize monitoring state
            self.monitoring_active = False
            self.current_monitoring_folder = None
            self.monitoring_thread = None
            
            # Initialize plots (will be initialized when visualization section is created)
            # Health figure and canvas are created in _create_visualization_section()
            
            # Add debugging status
            self.debug_label = QLabel("Debug Status: Initializing...")
            layout.addWidget(self.debug_label)
            
        except Exception as e:
            logger.error(f"Error in setup_health_tab: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if hasattr(self, 'debug_label'):
                self.debug_label.setText(f"Debug Status: Error - {str(e)}")
            QMessageBox.critical(self, "Error", 
                              "Failed to set up System Health tab. Check the logs for details.")
            raise
        
    def _create_model_config_section(self):
        """Create the model configuration section of the health tab"""
        try:
            group = QGroupBox("Model Configuration")
            layout = QVBoxLayout()
            
            # Model selection
            selection = QHBoxLayout()
            selection.addWidget(QLabel("Active Models:"))
            
            # LSTM controls
            lstm_group = QVBoxLayout()
            lstm_controls = QHBoxLayout()
            self.lstm_check = QCheckBox("LSTM")
            self.lstm_check.setChecked(True)
            self.lstm_weight = QSpinBox()
            self.lstm_weight.setValue(40)
            self.lstm_weight.setRange(0, 100)
            lstm_controls.addWidget(self.lstm_check)
            lstm_controls.addWidget(self.lstm_weight)
            lstm_group.addLayout(lstm_controls)
            self.agent1_health_label = QLabel("Health: Initializing...")
            lstm_group.addWidget(self.agent1_health_label)
            selection.addLayout(lstm_group)
            
            # CNN controls
            cnn_group = QVBoxLayout()
            cnn_controls = QHBoxLayout()
            self.cnn_check = QCheckBox("CNN")
            self.cnn_check.setChecked(True)
            self.cnn_weight = QSpinBox()
            self.cnn_weight.setValue(40)
            self.cnn_weight.setRange(0, 100)
            cnn_controls.addWidget(self.cnn_check)
            cnn_controls.addWidget(self.cnn_weight)
            cnn_group.addLayout(cnn_controls)
            self.agent2_health_label = QLabel("Health: Initializing...")
            cnn_group.addWidget(self.agent2_health_label)
            selection.addLayout(cnn_group)
            
            # ARIMA controls
            arima_group = QVBoxLayout()
            arima_controls = QHBoxLayout()
            self.arima_check = QCheckBox("ARIMA")
            self.arima_check.setChecked(True)
            self.arima_weight = QSpinBox()
            self.arima_weight.setValue(20)
            self.arima_weight.setRange(0, 100)
            arima_controls.addWidget(self.arima_check)
            arima_controls.addWidget(self.arima_weight)
            arima_group.addLayout(arima_controls)
            self.agent3_health_label = QLabel("Health: Initializing...")
            arima_group.addWidget(self.agent3_health_label)
            selection.addLayout(arima_group)
            
            layout.addLayout(selection)
            group.setLayout(layout)
            return group
            
        except Exception as e:
            logger.error(f"Error creating model config section: {str(e)}")
            raise
    
    def _create_monitoring_controls(self):
        """Create the monitoring controls section of the health tab"""
        try:
            group = QGroupBox("System Health Monitoring - Synchronized with Analysis Tab")
            layout = QVBoxLayout()
            
            # Sync status with Analysis tab
            sync_status_row = QHBoxLayout()
            self.sync_status_label = QLabel("🔄 Waiting for Analysis tab to process data...")
            self.sync_status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px;")
            sync_status_row.addWidget(self.sync_status_label)
            sync_status_row.addStretch()
            layout.addLayout(sync_status_row)
            
            # Model info display
            model_info_row = QHBoxLayout()
            self.analysis_model_info_label = QLabel("Analysis Model: Not Available")
            self.analysis_model_info_label.setStyleSheet("color: #666; font-size: 10px; padding: 3px;")
            model_info_row.addWidget(self.analysis_model_info_label)
            model_info_row.addStretch()
            layout.addLayout(model_info_row)
            
            # Threshold display with manual adjustment
            threshold_row = QHBoxLayout()
            threshold_row.addWidget(QLabel("Reconstruction Error Threshold:"))
            self.threshold_value_label = QLabel("Auto (95th percentile)")
            self.threshold_value_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            threshold_row.addWidget(self.threshold_value_label)
            
            self.adjust_threshold_btn = QPushButton("Adjust")
            self.adjust_threshold_btn.setMaximumWidth(60)
            self.adjust_threshold_btn.setStyleSheet("font-size: 9px; padding: 2px 8px;")
            self.adjust_threshold_btn.clicked.connect(self.adjust_threshold_dialog)
            threshold_row.addWidget(self.adjust_threshold_btn)
            threshold_row.addStretch()
            layout.addLayout(threshold_row)
            
            # Control buttons
            control_row = QHBoxLayout()
            self.start_monitoring_btn = QPushButton("Start Monitoring")
            self.start_monitoring_btn.clicked.connect(self.start_health_monitoring)
            self.start_monitoring_btn.setEnabled(False)
            self.start_monitoring_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    padding: 8px 15px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
            
            self.stop_monitoring_btn = QPushButton("Stop")
            self.stop_monitoring_btn.clicked.connect(self.stop_health_monitoring)
            self.stop_monitoring_btn.setEnabled(False)
            self.stop_monitoring_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    padding: 8px 15px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                }
            """)
            
            self.monitoring_status_indicator = QLabel("[Inactive]")
            self.monitoring_status_indicator.setStyleSheet("color: #999; font-weight: bold;")
            
            control_row.addWidget(self.start_monitoring_btn)
            control_row.addWidget(self.stop_monitoring_btn)
            control_row.addWidget(self.monitoring_status_indicator)
            control_row.addStretch()
            layout.addLayout(control_row)
            
            # Time To Failure (TTF) display
            ttf_row = QHBoxLayout()
            ttf_row.addWidget(QLabel("⏱️ Time To Failure (TTF):"))
            self.ttf_label = QLabel("N/A - Start monitoring to calculate")
            self.ttf_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; padding: 5px;")
            ttf_row.addWidget(self.ttf_label)
            ttf_row.addStretch()
            layout.addLayout(ttf_row)
            
            # Status display
            self.anomaly_alert_label = QLabel("System Health Monitoring Ready - Waiting for Analysis tab data")
            self.anomaly_alert_label.setStyleSheet("color: #666; padding: 8px; background-color: #f5f5f5; border-radius: 3px;")
            layout.addWidget(self.anomaly_alert_label)
            
            group.setLayout(layout)
            return group
            
        except Exception as e:
            logger.error(f"Error creating monitoring controls: {str(e)}")
            raise
    
    def _create_visualization_section(self):
        """Create the visualization section of the health tab"""
        try:
            group = QGroupBox("System Health Visualization")
            layout = QVBoxLayout()
            
            self.health_fig = Figure(figsize=(8, 6))
            self.health_canvas = FigureCanvas(self.health_fig)
            layout.addWidget(self.health_canvas)
            
            group.setLayout(layout)
            return group
            
        except Exception as e:
            logger.error(f"Error creating visualization section: {str(e)}")
            raise
    
    def setup_visualization_tab(self):
        layout = QVBoxLayout(self.visualization_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        self.build_visualization_panel(self, layout)

    def _viz_slot(self, host, method_name):
        return self._qt_slot_adapter(host, method_name, self._invoke_tab_visualization_method)

    _TAB_VISUALIZATION_HELPER_NAMES = (
        "plot_feature", "clear_plot", "export_plot", "plot_model_comparison", "plot_metrics",
        "update_visualization_features", "update_viz_model_info", "_apply_time_range_filter",
        "_sync_trained_models_for_viz", "_viz_reset_axes", "_viz_anomaly_scores_series",
        "_plot_time_series_with_anomalies", "_plot_anomaly_distribution",
        "_plot_metrics_anomaly_distribution", "_plot_correlation_heatmap",
        "_plot_statistical_summary", "_plot_anomaly_timeline", "_plot_error_metrics",
        "_plot_model_performance", "_plot_3d_surface", "_plot_3d_waterfall",
        "_plot_model_detection_counts", "_plot_model_agreement_matrix", "_plot_detection_overlap",
        "_plot_individual_predictions", "_plot_score_distributions", "_plot_confidence_comparison",
        "_plot_performance_metrics",
    )

    def _ensure_tab_visualization_helpers(self, host):
        self._ensure_tab_panel_helpers(host)

    def _invoke_tab_visualization_method(self, host, method_name, *args, **kwargs):
        if host is not self:
            self._ensure_tab_panel_helpers(host)
        else:
            self._ensure_tab_tool_context(host)
        method = getattr(SecureAnomalyDetectionTool, method_name, None)
        if method is None:
            method = getattr(self, method_name)
        return method(host, *args, **kwargs)

    def build_visualization_panel(self, host, parent_layout):
        from app.tabs.custom_tab.panels.visualization import build_visualization_panel as _build
        from app.tabs.custom_tab.panels.slots import assert_panel_slots_complete
        from app.tabs.custom_tab.widget import build_legacy_panel_slots

        _build(host, parent_layout, slots=assert_panel_slots_complete(build_legacy_panel_slots(self)))
    def _viz_reset_axes(self):
        """Recreate 2D axes after fig.clear() (stale .axes breaks plot_metrics)."""
        self.plot_canvas.fig.clear()
        self.plot_canvas.axes = self.plot_canvas.fig.add_subplot(111)
        return self.plot_canvas.axes

    def _viz_anomaly_scores_series(self):
        """Return Anomaly Score series from data/preprocessed, or None."""
        dp = getattr(self, "data_processor", None)
        if dp is None:
            return None
        for frame in (getattr(dp, "preprocessed_data", None), getattr(dp, "data", None)):
            if isinstance(frame, pd.DataFrame) and "Anomaly Score" in frame.columns:
                s = frame["Anomaly Score"].dropna()
                if len(s) > 0:
                    return s
        return None

    def plot_metrics(self):
        """Plot selected metrics visualization (real model/data only — no fake 3D)."""
        try:
            metric_type = self.metric_type_combo.currentText()
            plot_type = self.plot_type_combo.currentText()

            if plot_type in ("3D Surface", "3D Waterfall"):
                scores = self._viz_anomaly_scores_series()
                if scores is None:
                    self.visualization_status_label.setText(
                        "3D plots need Anomaly Score in data. Run Test/Predict first, or use Generate → Anomaly Score Distribution."
                    )
                    return
                self.plot_canvas.fig.clear()
                if plot_type == "3D Surface":
                    self._plot_3d_surface(metric_type)
                else:
                    self._plot_3d_waterfall(metric_type)
            elif metric_type == "Anomaly Distribution":
                self._viz_reset_axes()
                self._plot_metrics_anomaly_distribution(plot_type)
            else:
                if not self.model or not getattr(self.model, "metrics", None):
                    self.visualization_status_label.setText(
                        "No model metrics available. Train a model first."
                    )
                    return
                self._viz_reset_axes()
                if metric_type == "Error Metrics":
                    self._plot_error_metrics(plot_type)
                elif metric_type == "Model Performance":
                    self._plot_model_performance(plot_type)

            self.plot_canvas.fig.tight_layout()
            self.plot_canvas.draw()

        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")
            self.visualization_status_label.setText(f"Error plotting metrics: {str(e)}")
    
    def _plot_error_metrics(self, plot_type):
        """Plot error metrics visualization"""
        ax = getattr(self.plot_canvas, "axes", None) or self._viz_reset_axes()
        metrics = getattr(self.model, "prediction_metrics", None) or {}
        if not metrics:
            # Fall back to model.metrics numeric fields
            metrics = {
                k: v
                for k, v in (getattr(self.model, "metrics", None) or {}).items()
                if isinstance(v, (int, float))
            }
        if not metrics:
            ax.text(0.5, 0.5, "No error/prediction metrics on this model", ha="center", va="center")
            self.visualization_status_label.setText("No error metrics available")
            return

        if plot_type == "Line Plot":
            if "mean_score" in metrics and "score_std" in metrics:
                ax.errorbar(
                    [0],
                    [metrics["mean_score"]],
                    yerr=[metrics["score_std"]],
                    fmt="o-",
                    capsize=5,
                )
                ax.set_ylabel("Anomaly Score")
                ax.set_title("Error Metrics with Standard Deviation")
            else:
                keys = list(metrics.keys())[:12]
                ax.plot(keys, [metrics[k] for k in keys], "o-")
                ax.set_title("Error Metrics")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        elif plot_type == "Bar Chart":
            error_metrics = {
                k: v
                for k, v in metrics.items()
                if k in ("mean_score", "min_score", "max_score") or isinstance(v, (int, float))
            }
            keys = list(error_metrics.keys())[:12]
            ax.bar(keys, [error_metrics[k] for k in keys])
            ax.set_ylabel("Score Value")
            ax.set_title("Error Metrics Comparison")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            ax.text(
                0.5,
                0.5,
                f"Plot type '{plot_type}' not supported for Error Metrics.\nUse Line Plot or Bar Chart.",
                ha="center",
                va="center",
            )
        self.visualization_status_label.setText(f"Error metrics ({plot_type})")
    
    def _plot_model_performance(self, plot_type):
        """Plot model performance metrics"""
        ax = getattr(self.plot_canvas, "axes", None) or self._viz_reset_axes()
        metrics = getattr(self.model, "metrics", None) or {}
        if not metrics:
            ax.text(0.5, 0.5, "No performance metrics on this model", ha="center", va="center")
            self.visualization_status_label.setText("No performance metrics available")
            return

        perf_metrics = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and k not in ("training_time",)
        }
        if not perf_metrics:
            ax.text(0.5, 0.5, "No numeric performance metrics", ha="center", va="center")
            return

        keys = list(perf_metrics.keys())
        vals = [perf_metrics[k] for k in keys]
        if plot_type == "Line Plot":
            ax.plot(keys, vals, "o-")
            ax.set_ylabel("Value")
            ax.set_title("Model Performance Score")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        elif plot_type == "Bar Chart":
            ax.bar(keys, vals)
            ax.set_ylabel("Value")
            ax.set_title("Model Performance Metrics")
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        elif plot_type == "Histogram":
            ax.hist(vals, bins=min(10, max(3, len(vals))), alpha=0.75)
            ax.set_title("Performance Metrics Distribution")
        else:
            ax.text(
                0.5,
                0.5,
                f"Use Line/Bar/Histogram for Model Performance.\n'{plot_type}' needs Anomaly Score data.",
                ha="center",
                va="center",
            )
        self.visualization_status_label.setText(f"Model performance ({plot_type})")
    
    def _plot_metrics_anomaly_distribution(self, plot_type):
        """Metrics panel: histogram/box of Anomaly Score (distinct from Generate→distribution)."""
        ax = getattr(self.plot_canvas, "axes", None) or self._viz_reset_axes()
        scores = self._viz_anomaly_scores_series()
        if scores is None:
            ax.text(0.5, 0.5, "Anomaly Score not computed yet", ha="center", va="center")
            self.visualization_status_label.setText("Anomaly Score not available — run Test/Predict first")
            return

        if plot_type in ("Histogram", "Line Plot", "Bar Chart"):
            ax.hist(scores, bins=min(50, max(10, len(scores) // 20)), alpha=0.75)
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Anomaly Scores")
        elif plot_type == "Box Plot":
            ax.boxplot(scores)
            ax.set_ylabel("Anomaly Score")
            ax.set_title("Anomaly Score Distribution")
            if hasattr(self, "model") and getattr(self.model, "reconstruction_error_threshold", None):
                ax.axhline(
                    y=self.model.reconstruction_error_threshold,
                    color="r",
                    linestyle="--",
                    label="Anomaly Threshold",
                )
                ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                f"For '{plot_type}' use Plot Metrics with 3D types,\nor Generate → Anomaly Score Distribution.",
                ha="center",
                va="center",
            )
            return
        self.visualization_status_label.setText(
            f"Anomaly score distribution: n={len(scores)}, mean={float(scores.mean()):.3f}"
        )
    
    def _plot_3d_surface(self, metric_type):
        """Plot 3D surface from real Anomaly Score grid (no synthetic fallback)."""
        try:
            scores = self._viz_anomaly_scores_series()
            if scores is None or len(scores) < 25:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Need at least 25 Anomaly Score points for 3D Surface",
                    ha="center",
                    va="center",
                )
                return

            ax = self.plot_canvas.fig.add_subplot(111, projection="3d")
            side = int(min(50, int(np.sqrt(len(scores)))))
            need = side * side
            z_flat = scores.values[:need]
            if len(z_flat) < need:
                ax_flat = self.plot_canvas.fig.add_subplot(111)
                ax_flat.text(0.5, 0.5, "Not enough score points for grid", ha="center", va="center")
                return
            Z = np.asarray(z_flat, dtype=float).reshape(side, side)
            x = np.arange(side)
            y = np.arange(side)
            X, Y = np.meshgrid(x, y)
            surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="none")
            ax.set_xlabel("Index X")
            ax.set_ylabel("Index Y")
            ax.set_zlabel("Anomaly Score")
            ax.set_title(f"3D Surface: {metric_type} (from Anomaly Score)")
            self.plot_canvas.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            self.visualization_status_label.setText(
                f"3D surface from {need} anomaly scores ({side}×{side})"
            )
        except Exception as e:
            logger.error(f"Error plotting 3D surface: {e}")
            ax = self.plot_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")
    
    def _plot_3d_waterfall(self, metric_type):
        """Plot 3D bars from rolling Anomaly Score windows (no synthetic fallback)."""
        try:
            scores = self._viz_anomaly_scores_series()
            if scores is None or len(scores) < 20:
                ax = self.plot_canvas.fig.add_subplot(111)
                ax.text(
                    0.5,
                    0.5,
                    "Need Anomaly Score data for 3D Waterfall.\nRun Test/Predict first.",
                    ha="center",
                    va="center",
                )
                self.visualization_status_label.setText("3D Waterfall blocked — no Anomaly Score")
                return

            ax = self.plot_canvas.fig.add_subplot(111, projection="3d")
            vals = scores.values.astype(float)
            n_bins = 10
            chunk = max(1, len(vals) // n_bins)
            rolled = [
                float(np.mean(vals[i : i + chunk]))
                for i in range(0, min(len(vals), chunk * n_bins), chunk)
            ][:n_bins]
            while len(rolled) < n_bins:
                rolled.append(rolled[-1] if rolled else 0.0)
            Z = np.tile(np.asarray(rolled), (n_bins, 1)).T
            xpos, ypos = np.meshgrid(np.arange(n_bins), np.arange(n_bins))
            zpos = np.zeros_like(Z)
            dz = Z.flatten()
            zmax = float(np.max(dz)) if np.max(dz) > 0 else 1.0
            colors = plt.cm.viridis(dz / zmax)
            ax.bar3d(
                xpos.flatten(),
                ypos.flatten(),
                zpos.flatten(),
                0.8,
                0.8,
                dz,
                color=colors,
                zsort="average",
                alpha=0.85,
            )
            ax.set_xlabel("Window")
            ax.set_ylabel("Series")
            ax.set_zlabel("Mean Score")
            ax.set_title(f"3D Waterfall: {metric_type} (rolling Anomaly Score)")
            self.visualization_status_label.setText(
                f"3D waterfall from {len(vals)} scores → {n_bins} rolling windows"
            )
        except Exception as e:
            logger.error(f"Error plotting 3D waterfall: {e}")
            ax = self.plot_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center")
    
    def setup_admin_tab(self):
        # Main layout with scroll area for responsive design
        main_layout = QVBoxLayout(self.admin_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container widget for scroll area
        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Initialize system health monitor (legacy)
        self.health_monitor = SystemHealthMonitor("telemetry_system")
        
        # Initialize multi-agent system components
        self.multi_agent_system = None
        self.health_data_buffer = []
        self.health_monitoring_active = False
        
        # ====== TITLE ======
        title_layout = QHBoxLayout()
        title_label = QLabel("Multi-Agent System Health Monitoring")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: #2E4BC6;")
        title_label.setWordWrap(True)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        self.monitoring_status_indicator = QLabel("[Inactive]")
        self.monitoring_status_indicator.setFont(QFont("Arial", 12))
        self.monitoring_status_indicator.setStyleSheet("color: #999;")
        self.monitoring_status_indicator.setWordWrap(True)
        title_layout.addWidget(self.monitoring_status_indicator)
        layout.addLayout(title_layout)
        
        # Add description
        desc_label = QLabel("Real-time parallel agent monitoring: LSTM+Autoencoder | CNN | ARIMA Trend Analysis")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-style: italic; padding: 5px; font-size: 11px;")
        layout.addWidget(desc_label)
        
        # ====== CONTROL PANEL ======
        control_group = QGroupBox("Monitoring Controls")
        control_layout = QVBoxLayout()
        
        # Monitoring folder row
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Monitoring Folder:"))
        
        self.monitoring_folder_label = QLabel("Not Selected")
        self.monitoring_folder_label.setStyleSheet("color: #999; font-style: italic; padding: 3px;")
        self.monitoring_folder_label.setWordWrap(True)
        folder_row.addWidget(self.monitoring_folder_label, 1)
        
        select_folder_btn = QPushButton("Select Folder")
        select_folder_btn.clicked.connect(self.select_monitoring_folder)
        select_folder_btn.setMaximumWidth(120)
        folder_row.addWidget(select_folder_btn)
        
        control_layout.addLayout(folder_row)
        
        # First row: Status and basic controls
        first_row = QHBoxLayout()
        
        # Data status indicator
        self.data_status_label = QLabel("Data: Not Loaded")
        self.data_status_label.setStyleSheet("color: #FF6F00; font-weight: bold; padding: 5px; font-size: 10px;")
        first_row.addWidget(self.data_status_label)
        first_row.addSpacing(15)
        
        # Interval control with hours, minutes, seconds
        first_row.addWidget(QLabel("Check Interval:"))
        
        # Hours
        self.monitoring_hours_spin = QSpinBox()
        self.monitoring_hours_spin.setRange(0, 24)
        self.monitoring_hours_spin.setValue(0)
        self.monitoring_hours_spin.setSuffix(" h")
        self.monitoring_hours_spin.setMinimumWidth(60)
        self.monitoring_hours_spin.setMaximumWidth(70)
        first_row.addWidget(self.monitoring_hours_spin)
        
        # Minutes
        self.monitoring_minutes_spin = QSpinBox()
        self.monitoring_minutes_spin.setRange(0, 59)
        self.monitoring_minutes_spin.setValue(0)
        self.monitoring_minutes_spin.setSuffix(" m")
        self.monitoring_minutes_spin.setMinimumWidth(60)
        self.monitoring_minutes_spin.setMaximumWidth(70)
        first_row.addWidget(self.monitoring_minutes_spin)
        
        # Seconds
        self.monitoring_seconds_spin = QSpinBox()
        self.monitoring_seconds_spin.setRange(1, 59)
        self.monitoring_seconds_spin.setValue(5)
        self.monitoring_seconds_spin.setSuffix(" s")
        self.monitoring_seconds_spin.setMinimumWidth(60)
        self.monitoring_seconds_spin.setMaximumWidth(70)
        first_row.addWidget(self.monitoring_seconds_spin)
        
        # Total interval display
        self.interval_total_label = QLabel("(5s)")
        self.interval_total_label.setStyleSheet("color: #666; font-style: italic; font-size: 9px;")
        first_row.addWidget(self.interval_total_label)
        
        # Connect value changes to update total display
        self.monitoring_hours_spin.valueChanged.connect(self.update_interval_display)
        self.monitoring_minutes_spin.valueChanged.connect(self.update_interval_display)
        self.monitoring_seconds_spin.valueChanged.connect(self.update_interval_display)
        
        first_row.addSpacing(15)
        
        # Start/Stop buttons
        self.start_monitoring_btn = QPushButton("Start")
        self.start_monitoring_btn.clicked.connect(self.start_health_monitoring)
        self.start_monitoring_btn.setMinimumWidth(70)
        self.start_monitoring_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        first_row.addWidget(self.start_monitoring_btn)
        
        self.stop_monitoring_btn = QPushButton("Stop")
        self.stop_monitoring_btn.clicked.connect(self.stop_health_monitoring)
        self.stop_monitoring_btn.setEnabled(False)
        self.stop_monitoring_btn.setMinimumWidth(70)
        self.stop_monitoring_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 6px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        first_row.addWidget(self.stop_monitoring_btn)
        
        first_row.addStretch()
        control_layout.addLayout(first_row)
        
        # Second row: Automatic trigger controls
        second_row = QHBoxLayout()
        
        # Auto-start checkbox
        self.auto_start_health_check = QCheckBox("Auto-Start After Analysis Completes")
        self.auto_start_health_check.setToolTip("Automatically start System Health monitoring after Analysis tab completes prediction on new data")
        self.auto_start_health_check.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #2196F3;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)
        self.auto_start_health_check.stateChanged.connect(self.on_auto_start_health_changed)
        second_row.addWidget(self.auto_start_health_check)
        
        # Status indicator for auto-start
        self.auto_start_status_label = QLabel("")
        self.auto_start_status_label.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        second_row.addWidget(self.auto_start_status_label)
        
        second_row.addStretch()
        control_layout.addLayout(second_row)
        
        # Info label
        info_label = QLabel("💡 Enable auto-start to automatically monitor parallel agents after Analysis tab processes new data")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 9px; padding: 5px; background-color: #f5f5f5; border-radius: 3px;")
        control_layout.addWidget(info_label)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # ====== OVERALL SYSTEM STATUS ======
        overall_status_group = QGroupBox("Overall System Status")
        overall_status_layout = QVBoxLayout()
        
        # System status indicator
        status_row = QHBoxLayout()
        self.anomaly_alert_label = QLabel("System Status: NORMAL")
        self.anomaly_alert_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.anomaly_alert_label.setStyleSheet("color: #4CAF50; padding: 8px; background-color: #E8F5E9; border-radius: 5px;")
        self.anomaly_alert_label.setAlignment(Qt.AlignCenter)
        status_row.addWidget(self.anomaly_alert_label)
        overall_status_layout.addLayout(status_row)
        
        # Fusion weights display
        weights_row = QHBoxLayout()
        self.weights_label = QLabel("Fusion Weights: Agent1=40%, Agent2=40%, Trend=20%")
        self.weights_label.setStyleSheet("color: #666; font-size: 10px;")
        weights_row.addWidget(self.weights_label)
        
        config_weights_btn = QPushButton("Configure Weights")
        config_weights_btn.clicked.connect(self.configure_fusion_weights)
        config_weights_btn.setStyleSheet("font-size: 9px; padding: 3px 10px;")
        config_weights_btn.setMaximumWidth(120)
        weights_row.addWidget(config_weights_btn)
        weights_row.addStretch()
        overall_status_layout.addLayout(weights_row)
        
        overall_status_group.setLayout(overall_status_layout)
        layout.addWidget(overall_status_group)
        
        # ====== AGENT 1: LSTM + AUTOENCODER ======
        agent1_group = QGroupBox("Agent 1: LSTM + Autoencoder")
        agent1_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2196F3;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        agent1_layout = QVBoxLayout()
        
        # Agent 1 Status Row
        agent1_status_row = QHBoxLayout()
        self.agent1_status_indicator = QLabel("● Inactive")
        self.agent1_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
        agent1_status_row.addWidget(self.agent1_status_indicator)
        
        self.agent1_health_label = QLabel("Health: N/A")
        self.agent1_health_label.setStyleSheet("color: #666; font-size: 11px;")
        agent1_status_row.addWidget(self.agent1_health_label)
        agent1_status_row.addStretch()
        agent1_layout.addLayout(agent1_status_row)
        
        # Agent 1 Metrics Grid
        agent1_metrics_grid = QGridLayout()
        
        # Performance Metrics
        agent1_metrics_grid.addWidget(QLabel("Reconstruction Error:"), 0, 0)
        self.agent1_recon_error_label = QLabel("N/A")
        self.agent1_recon_error_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        agent1_metrics_grid.addWidget(self.agent1_recon_error_label, 0, 1)
        
        agent1_metrics_grid.addWidget(QLabel("Threshold:"), 0, 2)
        self.agent1_threshold_label = QLabel("N/A")
        agent1_metrics_grid.addWidget(self.agent1_threshold_label, 0, 3)
        
        agent1_metrics_grid.addWidget(QLabel("Processing Time:"), 1, 0)
        self.agent1_process_time_label = QLabel("N/A")
        agent1_metrics_grid.addWidget(self.agent1_process_time_label, 1, 1)
        
        agent1_metrics_grid.addWidget(QLabel("Samples Processed:"), 1, 2)
        self.agent1_samples_label = QLabel("0")
        agent1_metrics_grid.addWidget(self.agent1_samples_label, 1, 3)
        
        # Parameters
        agent1_metrics_grid.addWidget(QLabel("LSTM Units:"), 2, 0)
        self.agent1_lstm_units_label = QLabel("64")
        agent1_metrics_grid.addWidget(self.agent1_lstm_units_label, 2, 1)
        
        agent1_metrics_grid.addWidget(QLabel("Encoding Dim:"), 2, 2)
        self.agent1_encoding_dim_label = QLabel("32")
        agent1_metrics_grid.addWidget(self.agent1_encoding_dim_label, 2, 3)
        
        agent1_layout.addLayout(agent1_metrics_grid)
        
        # Agent 1 Visualization
        self.agent1_canvas = MplCanvas(width=12, height=3)
        agent1_layout.addWidget(self.agent1_canvas)
        
        agent1_group.setLayout(agent1_layout)
        layout.addWidget(agent1_group)
        
        # Initialize status indicators
        self.agent1_status_indicator = QLabel("● Inactive")
        self.agent1_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
        
        self.agent2_status_indicator = QLabel("● Inactive")
        self.agent2_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
        
        self.agent3_status_indicator = QLabel("● Inactive")
        self.agent3_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
        
        # ====== AGENT 2: CNN ======
        agent2_group = QGroupBox("Agent 2: CNN (Convolutional Neural Network)")
        agent2_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #FF9800;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        agent2_layout = QVBoxLayout()
        
        # Agent 2 Status Row
        agent2_status_row = QHBoxLayout()
        self.agent2_status_indicator = QLabel("● Inactive")
        self.agent2_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
        agent2_status_row.addWidget(self.agent2_status_indicator)
        
        self.agent2_health_label = QLabel("Health: N/A")
        self.agent2_health_label.setStyleSheet("color: #666; font-size: 11px;")
        agent2_status_row.addWidget(self.agent2_health_label)
        agent2_status_row.addStretch()
        agent2_layout.addLayout(agent2_status_row)
        
        # Agent 2 Metrics Grid
        agent2_metrics_grid = QGridLayout()
        
        # Performance Metrics
        agent2_metrics_grid.addWidget(QLabel("Detection Score:"), 0, 0)
        self.agent2_detection_score_label = QLabel("N/A")
        self.agent2_detection_score_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        agent2_metrics_grid.addWidget(self.agent2_detection_score_label, 0, 1)
        
        agent2_metrics_grid.addWidget(QLabel("Confidence:"), 0, 2)
        self.agent2_confidence_label = QLabel("N/A")
        agent2_metrics_grid.addWidget(self.agent2_confidence_label, 0, 3)
        
        agent2_metrics_grid.addWidget(QLabel("Processing Time:"), 1, 0)
        self.agent2_process_time_label = QLabel("N/A")
        agent2_metrics_grid.addWidget(self.agent2_process_time_label, 1, 1)
        
        agent2_metrics_grid.addWidget(QLabel("Samples Processed:"), 1, 2)
        self.agent2_samples_label = QLabel("0")
        agent2_metrics_grid.addWidget(self.agent2_samples_label, 1, 3)
        
        # Parameters
        agent2_metrics_grid.addWidget(QLabel("Conv Filters:"), 2, 0)
        self.agent2_filters_label = QLabel("32")
        agent2_metrics_grid.addWidget(self.agent2_filters_label, 2, 1)
        
        agent2_metrics_grid.addWidget(QLabel("Kernel Size:"), 2, 2)
        self.agent2_kernel_label = QLabel("3")
        agent2_metrics_grid.addWidget(self.agent2_kernel_label, 2, 3)
        
        agent2_layout.addLayout(agent2_metrics_grid)
        
        # Agent 2 Visualization
        self.agent2_canvas = MplCanvas(width=12, height=3)
        agent2_layout.addWidget(self.agent2_canvas)
        
        agent2_group.setLayout(agent2_layout)
        layout.addWidget(agent2_group)
        
        # ====== TREND MODEL: ARIMA ======
        trend_group = QGroupBox("Trend Model: ARIMA (AutoRegressive Integrated Moving Average)")
        trend_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        trend_layout = QVBoxLayout()
        
        # Trend Status Row
        trend_status_row = QHBoxLayout()
        self.trend_status_indicator = QLabel("● Inactive")
        self.trend_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
        trend_status_row.addWidget(self.trend_status_indicator)
        
        self.trend_health_label = QLabel("Health: N/A")
        self.trend_health_label.setStyleSheet("color: #666; font-size: 11px;")
        trend_status_row.addWidget(self.trend_health_label)
        trend_status_row.addStretch()
        trend_layout.addLayout(trend_status_row)
        
        # Trend Metrics Grid
        trend_metrics_grid = QGridLayout()
        
        # Performance Metrics
        trend_metrics_grid.addWidget(QLabel("Trend Score:"), 0, 0)
        self.trend_score_label = QLabel("N/A")
        self.trend_score_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        trend_metrics_grid.addWidget(self.trend_score_label, 0, 1)
        
        trend_metrics_grid.addWidget(QLabel("Forecast Accuracy:"), 0, 2)
        self.trend_accuracy_label = QLabel("N/A")
        trend_metrics_grid.addWidget(self.trend_accuracy_label, 0, 3)
        
        trend_metrics_grid.addWidget(QLabel("Processing Time:"), 1, 0)
        self.trend_process_time_label = QLabel("N/A")
        trend_metrics_grid.addWidget(self.trend_process_time_label, 1, 1)
        
        trend_metrics_grid.addWidget(QLabel("Samples Analyzed:"), 1, 2)
        self.trend_samples_label = QLabel("0")
        trend_metrics_grid.addWidget(self.trend_samples_label, 1, 3)
        
        # Parameters
        trend_metrics_grid.addWidget(QLabel("ARIMA Order (p,d,q):"), 2, 0)
        self.trend_order_label = QLabel("(5,1,0)")
        trend_metrics_grid.addWidget(self.trend_order_label, 2, 1)
        
        trend_metrics_grid.addWidget(QLabel("Window Size:"), 2, 2)
        self.trend_window_label = QLabel("50")
        trend_metrics_grid.addWidget(self.trend_window_label, 2, 3)
        
        trend_layout.addLayout(trend_metrics_grid)
        
        # Trend Visualization
        self.trend_canvas = MplCanvas(width=12, height=3)
        trend_layout.addWidget(self.trend_canvas)
        
        trend_group.setLayout(trend_layout)
        layout.addWidget(trend_group)
        
        # ====== FUSION RESULT VISUALIZATION ======
        fusion_group = QGroupBox("Fusion Result (Combined Multi-Agent Decision)")
        fusion_layout = QVBoxLayout()
        
        self.reconstruction_info_label = QLabel("Waiting for monitoring to start...")
        self.reconstruction_info_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        fusion_layout.addWidget(self.reconstruction_info_label)
        
        # Combined visualization
        self.combined_canvas = MplCanvas(width=14, height=4)
        fusion_layout.addWidget(self.combined_canvas)
        
        fusion_group.setLayout(fusion_layout)
        layout.addWidget(fusion_group)
        
        # Add stretch at the end
        layout.addStretch()
        
        # Set container widget and add to scroll area
        scroll_area.setWidget(container_widget)
        main_layout.addWidget(scroll_area)
        
        # Initialize monitoring timer
        self.health_monitoring_timer = QTimer(self)
        self.health_monitoring_timer.timeout.connect(self.collect_health_data)
        
        # Initialize canvas with empty plot
        self._init_empty_plots()
        
        logger.info("Multi-Agent System Health tab initialized successfully")
    
    def _init_empty_plots(self):
        """Initialize empty plot with proper labels"""
        try:
            # Combined reconstruction error plot
            self.combined_canvas.axes.clear()
            self.combined_canvas.axes.set_xlabel('Time')
            self.combined_canvas.axes.set_ylabel('Reconstruction Error')
            self.combined_canvas.axes.set_title('Final Reconstruction Error (Multi-Agent Fusion)')
            self.combined_canvas.axes.axhline(y=0.6, color='red', linestyle='--', label='Threshold', alpha=0.7)
            self.combined_canvas.axes.grid(True, alpha=0.3)
            self.combined_canvas.axes.legend()
            self.combined_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error initializing empty plot: {str(e)}")
    
    def update_telemetry_data_status(self):
        """Update the data status indicator when telemetry data is loaded"""
        try:
            # Check if data is available
            has_data = hasattr(self.data_processor, 'preprocessed_data') and self.data_processor.preprocessed_data is not None
            
            # Check if model is trained
            has_model = hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'model') and self.model.model is not None
            
            if has_data:
                num_samples = len(self.data_processor.preprocessed_data)
                num_features = len(self.data_processor.preprocessed_data.columns)
                
                if has_model:
                    # Data loaded AND model trained - fully ready!
                    self.data_status_label.setText(
                        f"Data Status: Loaded ({num_samples} samples, {num_features} features) | Model: {self.model.model_type} [Trained] - READY"
                    )
                    self.data_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px; background-color: #E8F5E9; border-radius: 3px;")
                    logger.info(f"System Health READY: {num_samples} samples, {num_features} features, model: {self.model.model_type}")
                else:
                    # Data loaded but waiting for model
                    self.data_status_label.setText(
                        f"Data Status: Loaded ({num_samples} samples, {num_features} features) - Waiting for Model Training..."
                    )
                    self.data_status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px; background-color: #FFF3E0; border-radius: 3px;")
                    logger.info(f"Telemetry data loaded: {num_samples} samples, {num_features} features - awaiting model")
            else:
                self.data_status_label.setText("Data Status: Not Loaded - Load data from Data Import tab, then train model in Analysis tab")
                self.data_status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 5px; background-color: #FFEBEE; border-radius: 3px;")
        except Exception as e:
            logger.error(f"Error updating data status: {str(e)}")
            self.data_status_label.setText("Data Status: Error checking data")
            self.data_status_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 5px;")
    
    def _integrate_model_with_health_monitoring(self):
        """Integrate newly trained model with system health monitoring (legacy support)"""
        try:
            if not hasattr(self, 'model') or not self.model or not self.model.model:
                return
            
            if not hasattr(self.data_processor, 'preprocessed_data') or self.data_processor.preprocessed_data is None:
                return
            
            # Get training data for threshold calculation
            numeric_data = self.data_processor.preprocessed_data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return
            
            # Create model name
            model_name = f"{self.model.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add model to system health monitoring
            self.health_monitor.add_analysis_model(
                model_name=model_name,
                model_object=self.model.model,
                training_data=numeric_data.values
            )
            
            logger.info(f"Model {model_name} integrated with legacy health monitoring")
            
        except Exception as e:
            logger.error(f"Error integrating model with health monitoring: {str(e)}")
    
    def _auto_start_system_health_monitoring(self):
        """Automatically start System Health monitoring after model training completes"""
        try:
            if getattr(self, "custom_tabs_only_mode", False):
                logger.info("Custom-tabs-only mode enabled; skipping legacy System Health auto-start.")
                return

            # Check if we have the necessary data
            if not hasattr(self.data_processor, 'preprocessed_data') or self.data_processor.preprocessed_data is None:
                logger.info("No preprocessed data available for System Health auto-start")
                return
            
            # Check if we have enough data
            if len(self.data_processor.preprocessed_data) < 100:
                logger.info(f"Not enough data for System Health auto-start: {len(self.data_processor.preprocessed_data)} samples")
                return
            
            # Check if System Health tab exists and has necessary attributes
            if not hasattr(self, 'health_monitoring_active'):
                logger.warning("System Health tab not properly initialized")
                return
            
            # Check if monitoring is already active
            if self.health_monitoring_active:
                logger.info("System Health monitoring already active")
                return
            
            # Create monitoring data cache
            if hasattr(self.data_processor, 'preprocessed_data') and self.data_processor.preprocessed_data is not None:
                self._monitoring_data_cache = self.data_processor.preprocessed_data.copy()
                logger.info(f"Created monitoring data cache: {len(self._monitoring_data_cache)} samples")
            
            # Update data status in System Health tab
            if hasattr(self, 'update_telemetry_data_status'):
                self.update_telemetry_data_status()
            
            # Log the auto-start
            logger.info("="*60)
            logger.info("AUTO-STARTING SYSTEM HEALTH MONITORING")
            logger.info(f"Data available: {len(self.data_processor.preprocessed_data)} samples")
            logger.info(f"Model trained: {self.model.model_type}")
            logger.info("="*60)
            
            # Auto-start monitoring with configured interval
            interval_sec = self.get_monitoring_interval_seconds()
            
            interval_ms = interval_sec * 1000
            
            # Initialize data index for cycling through telemetry
            self._data_index = 0
            
            # Start monitoring
            self.health_monitoring_active = True
            if hasattr(self, 'health_monitoring_timer'):
                self.health_monitoring_timer.start(interval_ms)
            
            # Update UI
            if hasattr(self, 'start_monitoring_btn'):
                self.start_monitoring_btn.setEnabled(False)
            if hasattr(self, 'stop_monitoring_btn'):
                self.stop_monitoring_btn.setEnabled(True)
            if hasattr(self, 'monitoring_status_indicator'):
                self.monitoring_status_indicator.setText("[Active - Auto-Started]")
                self.monitoring_status_indicator.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Update alert
            if hasattr(self, 'anomaly_alert_label'):
                self.anomaly_alert_label.setText("Auto-Started: Monitoring Analysis Model...")
                self.anomaly_alert_label.setStyleSheet("color: #2196F3; padding: 10px; background-color: #E3F2FD; border-radius: 5px;")
            
            logger.info(f"System Health monitoring auto-started successfully (interval: {interval_sec}s)")
            logger.info(f"Monitoring model: {self.model.model_type}")
            logger.info(f"Auto-start details - Model: {self.model.model_type}, Samples: {len(self.data_processor.preprocessed_data)}, Features: {len(self.data_processor.preprocessed_data.columns)}")
            
        except Exception as e:
            logger.error(f"Error auto-starting System Health monitoring: {str(e)}")
            logger.error(traceback.format_exc())
    
    def add_subsystem(self):
        """Add a new subsystem for health monitoring"""
        name = self.subsystem_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a subsystem name")
            return
        
        if name in self.health_monitor.sensor_groups:
            QMessageBox.warning(self, "Warning", f"Subsystem '{name}' already exists")
            return
        
        # Start with empty sensor list - user will add sensors separately
        try:
            self.health_monitor.add_subsystem(name, [])
            self.current_subsystem_combo.addItem(name)
            self.subsystem_name_input.clear()
            self.update_subsystems_display()
            QMessageBox.information(self, "Success", f"Subsystem '{name}' added successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add subsystem: {str(e)}")
    
    def add_sensor(self):
        """Add a sensor to the current subsystem"""
        current_subsystem = self.current_subsystem_combo.currentText()
        sensor_name = self.sensor_name_input.text().strip()
        
        if not current_subsystem:
            QMessageBox.warning(self, "Warning", "Please select a subsystem first")
            return
        
        if not sensor_name:
            QMessageBox.warning(self, "Warning", "Please enter a sensor name")
            return
        
        if current_subsystem not in self.health_monitor.sensor_groups:
            QMessageBox.warning(self, "Warning", "Selected subsystem not found")
            return
        
        # Check if sensor already exists
        if sensor_name in self.health_monitor.sensor_groups[current_subsystem]:
            QMessageBox.warning(self, "Warning", f"Sensor '{sensor_name}' already exists")
            return
        
        try:
            # Add sensor to the subsystem
            self.health_monitor.sensor_groups[current_subsystem].append(sensor_name)
            
            # Recreate autoencoder with new sensor count
            sensor_count = len(self.health_monitor.sensor_groups[current_subsystem])
            self.health_monitor.autoencoders[current_subsystem] = SklearnAutoencoder(
                input_dim=sensor_count,
                name=f"{current_subsystem}_autoencoder"
            )
            
            self.sensor_name_input.clear()
            self.update_sensors_display()
            self.update_subsystems_display()
            QMessageBox.information(self, "Success", f"Sensor '{sensor_name}' added to '{current_subsystem}'")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add sensor: {str(e)}")
    
    def on_subsystem_selected(self):
        """Update sensor display when subsystem selection changes"""
        self.update_sensors_display()
    
    def update_subsystems_display(self):
        """Update the subsystems table"""
        subsystems = list(self.health_monitor.sensor_groups.keys())
        self.subsystems_table.setRowCount(len(subsystems))
        
        for i, subsystem in enumerate(subsystems):
            self.subsystems_table.setItem(i, 0, QTableWidgetItem(subsystem))
            
            sensor_count = len(self.health_monitor.sensor_groups[subsystem])
            self.subsystems_table.setItem(i, 1, QTableWidgetItem(str(sensor_count)))
            
            # Get status and health score
            if subsystem in self.health_monitor.health_status:
                status_info = self.health_monitor.health_status[subsystem]
                status = status_info['status']
                health_score = status_info['health_score']
                
                status_item = QTableWidgetItem(status.title())
                if status == 'normal':
                    status_item.setBackground(QColor("#4CAF50"))
                elif status == 'warning':
                    status_item.setBackground(QColor("#FF9800"))
                elif status == 'critical':
                    status_item.setBackground(QColor("#f44336"))
                else:
                    status_item.setBackground(QColor("#9E9E9E"))
                
                self.subsystems_table.setItem(i, 2, status_item)
                self.subsystems_table.setItem(i, 3, QTableWidgetItem(f"{health_score:.2f}"))
            else:
                self.subsystems_table.setItem(i, 2, QTableWidgetItem("Not Trained"))
                self.subsystems_table.setItem(i, 3, QTableWidgetItem("N/A"))
    
    def update_sensors_display(self):
        """Update the sensors table for the selected subsystem"""
        current_subsystem = self.current_subsystem_combo.currentText()
        if not current_subsystem or current_subsystem not in self.health_monitor.sensor_groups:
            self.sensors_table.setRowCount(0)
            return
        
        sensors = self.health_monitor.sensor_groups[current_subsystem]
        self.sensors_table.setRowCount(len(sensors))
        
        for i, sensor in enumerate(sensors):
            self.sensors_table.setItem(i, 0, QTableWidgetItem(sensor))
            self.sensors_table.setItem(i, 1, QTableWidgetItem("N/A"))  # Current value
            self.sensors_table.setItem(i, 2, QTableWidgetItem("Unknown"))  # Status
    
    def load_training_data(self):
        """Load training data for autoencoder training"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Normal Operation Training Data", 
            "", 
            "CSV files (*.csv);;Excel files (*.xlsx);;All files (*)"
        )
        
        if file_name:
            try:
                # Load data
                if file_name.endswith('.csv'):
                    data = pd.read_csv(file_name)
                elif file_name.endswith('.xlsx'):
                    data = pd.read_excel(file_name)
                else:
                    QMessageBox.warning(self, "Warning", "Unsupported file format")
                    return
                
                # Store training data
                self.training_data = data
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Loaded training data with {data.shape[0]} samples and {data.shape[1]} features"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load training data: {str(e)}")
    
    def train_health_model(self):
        """Train autoencoder models for all subsystems"""
        if not hasattr(self, 'training_data') or self.training_data is None:
            QMessageBox.warning(self, "Warning", "Please load training data first")
            return
        
        if not self.health_monitor.sensor_groups:
            QMessageBox.warning(self, "Warning", "Please add subsystems and sensors first")
            return
        
        try:
            for subsystem_name, sensors in self.health_monitor.sensor_groups.items():
                if not sensors:  # Skip subsystems with no sensors
                    continue
                
                # Extract relevant columns from training data
                missing_sensors = [s for s in sensors if s not in self.training_data.columns]
                if missing_sensors:
                    QMessageBox.warning(
                        self, 
                        "Warning", 
                        f"Missing sensors in training data for '{subsystem_name}': {missing_sensors}"
                    )
                    continue
                
                # Get training data for this subsystem
                subsystem_data = self.training_data[sensors].dropna()
                
                if subsystem_data.empty:
                    QMessageBox.warning(
                        self, 
                        "Warning", 
                        f"No valid training data for subsystem '{subsystem_name}'"
                    )
                    continue
                
                # Train the autoencoder
                self.health_monitor.train_subsystem(subsystem_name, subsystem_data.values)
            
            self.update_subsystems_display()
            QMessageBox.information(self, "Success", "Autoencoder models trained successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train models: {str(e)}")
    
    def start_health_monitoring(self):
        """Start real-time health monitoring"""
        try:
            # Reset agent data buffers
            self.health_agent1 = {
                'data_buffer': [],
                'score_buffer': [],
                'last_update': None,
                'status': 'Active',
                'threshold': 0.85,
                'window_size': 100
            }
            
            self.health_agent2 = {
                'data_buffer': [],
                'score_buffer': [],
                'last_update': None,
                'status': 'Active',
                'threshold': 0.85,
                'window_size': 100
            }
            
            self.health_agent3 = {
                'data_buffer': [],
                'score_buffer': [],
                'last_update': None,
                'status': 'Active',
                'threshold': 0.85,
                'window_size': 100
            }
            
            # Initialize monitoring
            self.health_monitoring_active = True
            self.health_monitoring_timer.start(1000)  # Update every second
            
            # Update agent status labels
            self.agent1_health_label.setText("Health: Active")
            self.agent2_health_label.setText("Health: Active")
            self.agent3_health_label.setText("Health: Active")
            
            self.start_monitoring_btn.setEnabled(False)
            self.stop_monitoring_btn.setEnabled(True)
            self.health_status_label.setText("Status: Monitoring Active")
            self.health_status_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")
            
            QMessageBox.information(self, "Success", "Health monitoring started!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start monitoring: {str(e)}")
    
    def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.health_monitoring_active = False
        self.health_monitoring_timer.stop()
        
        self.start_monitoring_btn.setEnabled(True)
        self.stop_monitoring_btn.setEnabled(False)
        self.health_status_label.setText("Status: Monitoring Stopped")
        self.health_status_label.setStyleSheet("font-size: 14px; color: #666;")
    
    def update_health_monitoring(self):
        """Update health monitoring display"""
        if not self.health_monitoring_active:
            return
        
        try:
            # Simulate sensor data (in real implementation, this would come from actual sensors)
            current_time = datetime.datetime.now()
            
            for subsystem_name, autoencoder in self.health_monitor.autoencoders.items():
                if not autoencoder.is_trained:
                    continue
                
                sensors = self.health_monitor.sensor_groups[subsystem_name]
                if not sensors:
                    continue
                
                # Generate simulated sensor data (replace with actual sensor readings)
                sensor_data = np.random.normal(0, 1, len(sensors))
                
                # Add some anomalies occasionally for demonstration
                if np.random.random() < 0.1:  # 10% chance of anomaly
                    sensor_data += np.random.normal(2, 0.5, len(sensors))
                
                # Monitor the subsystem
                result = self.health_monitor.monitor_subsystem(subsystem_name, sensor_data, current_time)
                
                # Add alerts if anomaly detected or failure predicted
                if result['is_anomaly'] or result['failure_prediction']:
                    self.add_health_alert(subsystem_name, result, current_time)
            
            # Update displays
            self.update_subsystems_display()
            self.update_health_dashboard()
            self.last_update_label.setText(f"Last Update: {current_time.strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"Error in health monitoring update: {str(e)}")
    
    def add_health_alert(self, subsystem_name, result, timestamp):
        """Add a health alert to the alerts table"""
        row = self.alerts_table.rowCount()
        self.alerts_table.insertRow(row)
        
        time_str = timestamp.strftime('%H:%M:%S')
        self.alerts_table.setItem(row, 0, QTableWidgetItem(time_str))
        self.alerts_table.setItem(row, 1, QTableWidgetItem(subsystem_name))
        
        if result['failure_prediction']:
            pred = result['failure_prediction']
            alert_type = "Failure Prediction"
            message = f"{pred['status']} - Time to failure: {pred['time_to_failure']}"
            confidence = f"{pred['confidence']:.2f}"
        else:
            alert_type = "Anomaly Detected"
            message = f"Reconstruction error: {result['reconstruction_error']:.4f}"
            confidence = "N/A"
        
        type_item = QTableWidgetItem(alert_type)
        if alert_type == "Failure Prediction":
            type_item.setBackground(QColor("#f44336"))
        else:
            type_item.setBackground(QColor("#FF9800"))
        
        self.alerts_table.setItem(row, 2, type_item)
        self.alerts_table.setItem(row, 3, QTableWidgetItem(message))
        self.alerts_table.setItem(row, 4, QTableWidgetItem(confidence))
        
        # Scroll to bottom to show latest alerts
        self.alerts_table.scrollToBottom()
        
        # Update active alerts count
        active_alerts = self.alerts_table.rowCount()
        self.active_alerts_label.setText(f"Active Alerts: {active_alerts}")
    
    def update_health_dashboard(self):
        """Update the health monitoring dashboard charts"""
        if not self.health_monitor.monitoring_data:
            return
        
        self.health_figure.clear()
        
        # Create subplots for different metrics
        gs = self.health_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Overall health scores
        ax1 = self.health_figure.add_subplot(gs[0, :])
        
        for subsystem_name, data in self.health_monitor.monitoring_data.items():
            if data['health_scores']:
                timestamps = data['timestamps'][-50:]  # Last 50 points
                health_scores = data['health_scores'][-50:]
                ax1.plot(timestamps, health_scores, marker='o', label=subsystem_name)
        
        ax1.set_title('System Health Scores Over Time')
        ax1.set_ylabel('Health Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Reconstruction errors
        ax2 = self.health_figure.add_subplot(gs[1, 0])
        
        for subsystem_name, data in self.health_monitor.monitoring_data.items():
            if data['reconstruction_errors']:
                errors = data['reconstruction_errors'][-50:]
                ax2.plot(errors, label=subsystem_name)
                
                # Add threshold line
                if subsystem_name in self.health_monitor.autoencoders:
                    threshold = self.health_monitor.autoencoders[subsystem_name].threshold
                    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
        
        ax2.set_title('Reconstruction Errors')
        ax2.set_ylabel('Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Anomaly detection
        ax3 = self.health_figure.add_subplot(gs[1, 1])
        
        subsystem_names = []
        anomaly_counts = []
        
        for subsystem_name, data in self.health_monitor.monitoring_data.items():
            if data['anomaly_flags']:
                subsystem_names.append(subsystem_name)
                anomaly_counts.append(sum(data['anomaly_flags'][-100:]))  # Last 100 samples
        
        if subsystem_names:
            ax3.bar(subsystem_names, anomaly_counts, color=['#f44336' if c > 5 else '#4CAF50' for c in anomaly_counts])
            ax3.set_title('Anomaly Counts (Last 100 Samples)')
            ax3.set_ylabel('Count')
        
        # Calculate overall health
        if subsystem_names:
            overall_health = np.mean([
                self.health_monitor.health_status[name]['health_score'] 
                for name in subsystem_names 
                if name in self.health_monitor.health_status
            ])
            
            self.overall_health_label.setText(f"Overall Health: {overall_health:.2f}")
            
            if overall_health > 0.8:
                self.overall_health_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4CAF50;")
            elif overall_health > 0.6:
                self.overall_health_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF9800;")
            else:
                self.overall_health_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
        
        self.health_canvas.draw()
    
    # Enhanced System Health Monitoring Methods
    def on_model_selected(self):
        """Handle model selection change in health monitoring"""
        selected_model = self.available_models_combo.currentText()
        if selected_model and selected_model != "No models available":
            self.current_monitoring_model = selected_model
            self.update_models_status_display()
    

    
    def update_interval_display(self):
        """Update the total interval display when spinners change"""
        total_seconds = self.get_monitoring_interval_seconds()
        
        # Format nicely
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0:
            parts.append(f"{seconds}s")
        
        display_str = " ".join(parts) if parts else "0s"
        self.interval_total_label.setText(f"({display_str})")
    
    def on_auto_start_health_changed(self, state):
        """Handle auto-start health monitoring checkbox state change"""
        try:
            if state == Qt.Checked:
                self.auto_start_status_label.setText("✓ Will auto-start after Analysis completes")
                self.auto_start_status_label.setStyleSheet("color: #4CAF50; font-size: 9px; font-style: italic;")
                logger.info("System Health auto-start enabled - will start after Analysis processing")
            else:
                self.auto_start_status_label.setText("Manual start only")
                self.auto_start_status_label.setStyleSheet("color: #999; font-size: 9px; font-style: italic;")
                logger.info("System Health auto-start disabled")
        except Exception as e:
            logger.error(f"Error handling auto-start health change: {str(e)}")
    
    def _trigger_system_health_monitoring(self):
        """Automatically trigger System Health monitoring after Analysis completes"""
        try:
            if getattr(self, "custom_tabs_only_mode", False):
                logger.info("Custom-tabs-only mode enabled; legacy System Health trigger ignored.")
                return

            logger.info("Triggering System Health monitoring automatically...")
            
            # Check if data was pre-cached in on_prediction_completed()
            if not hasattr(self, '_monitoring_data_cache') or self._monitoring_data_cache is None:
                logger.error("Auto-trigger: Cache should have been created in on_prediction_completed() but is missing")
                if hasattr(self, 'auto_start_status_label'):
                    self.auto_start_status_label.setText("✗ Cache missing - timing issue")
                    self.auto_start_status_label.setStyleSheet("color: #F44336; font-size: 9px; font-style: italic;")
                return
            
            logger.info(f"Auto-trigger: Using pre-cached data ({len(self._monitoring_data_cache)} samples)")
            
            # Update data status
            if hasattr(self, 'data_status_label'):
                self.data_status_label.setText("Data: Cached & Ready")
                self.data_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px; font-size: 10px;")
            
            # Switch to System Health tab
            self.tabs.setCurrentIndex(4)  # System Health is tab index 4
            
            # Start monitoring (will use the cached data)
            self.start_health_monitoring()
            
            # Update status
            if hasattr(self, 'auto_start_status_label'):
                self.auto_start_status_label.setText("✓ Auto-started successfully")
                self.auto_start_status_label.setStyleSheet("color: #4CAF50; font-size: 9px; font-style: italic; font-weight: bold;")
            
            logger.info("System Health monitoring auto-started successfully with cached data")
            
        except Exception as e:
            logger.error(f"Error auto-triggering System Health monitoring: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if hasattr(self, 'auto_start_status_label'):
                self.auto_start_status_label.setText(f"✗ Auto-start failed: {str(e)}")
                self.auto_start_status_label.setStyleSheet("color: #f44336; font-size: 9px; font-style: italic;")
    
    def _initialize_health_agents(self):
        """Initialize pre-trained AI agents for System Health Monitoring"""
        try:
            logger.info("Initializing System Health AI agents...")
            
            # Agent 1: LSTM + Autoencoder (pre-configured, ready to use)
            self.health_agent1 = {
                'name': 'LSTM+Autoencoder',
                'type': 'lstm_autoencoder',
                'lstm_units': 64,
                'encoding_dim': 32,
                'threshold': 0.6,
                'weight': 0.4,
                'status': 'ready',
                'samples_processed': 0
            }
            
            # Agent 2: CNN (pre-configured, ready to use)
            self.health_agent2 = {
                'name': 'CNN',
                'type': 'cnn',
                'conv_filters': 32,
                'kernel_size': 3,
                'threshold': 0.7,
                'weight': 0.4,
                'status': 'ready',
                'samples_processed': 0
            }
            
            # Agent 3: ARIMA Trend Model (pre-configured, ready to use)
            self.health_agent3 = {
                'name': 'ARIMA',
                'type': 'arima',
                'order': (5, 1, 0),
                'window_size': 50,
                'threshold': 0.5,
                'weight': 0.2,
                'status': 'ready',
                'samples_processed': 0
            }
            
            # Fusion settings
            self.fusion_weights = {
                'agent1': 0.4,
                'agent2': 0.4,
                'trend': 0.2
            }
            
            # Initialize data buffers for each agent
            self.agent1_data_buffer = []
            self.agent2_data_buffer = []
            self.trend_data_buffer = []
            self.fusion_data_buffer = []
            
            self.health_agents_initialized = True
            
            # Update UI
            self.agent1_health_label.setText("Health: Ready")
            self.agent1_health_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            
            self.agent2_health_label.setText("Health: Ready")
            self.agent2_health_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            
            if hasattr(self, 'trend_health_label'):
                self.trend_health_label.setText("Health: Ready")
                self.trend_health_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            else:
                logger.warning("trend_health_label not found, skipping update")
            
            self.data_status_label.setText("Agents: Initialized & Ready")
            self.data_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px; background-color: #E8F5E9; border-radius: 3px;")
            
            logger.info("System Health AI agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing health agents: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.health_agents_initialized = False
    
    def adjust_threshold_dialog(self):
        """Open dialog to manually adjust reconstruction error threshold"""
        try:
            current_threshold = getattr(self, 'health_reconstruction_threshold', None)
            if current_threshold is None:
                current_threshold = 0.0
            
            from PyQt5.QtWidgets import QInputDialog
            value, ok = QInputDialog.getDouble(
                self,
                "Adjust Threshold",
                f"Reconstruction Error Threshold:\n(Current: {current_threshold:.6f})",
                current_threshold,
                0.0,
                1000000.0,
                6
            )
            
            if ok:
                self.health_reconstruction_threshold = value
                self.health_threshold_mode = "manual"
                self.threshold_value_label.setText(f"Manual: {value:.6f}")
                self.threshold_value_label.setStyleSheet("font-weight: bold; color: #FF9800;")
                logger.info(f"Threshold manually adjusted to: {value:.6f}")
                
        except Exception as e:
            logger.error(f"Error adjusting threshold: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to adjust threshold: {str(e)}")
    
    def select_monitoring_folder(self):
        """Select folder to monitor for incoming data files"""
        try:
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "Select Folder to Monitor",
                "" if not hasattr(self, 'monitoring_folder_path') else self.monitoring_folder_path
            )
            
            if folder_path:
                # Update monitoring folder path
                self.monitoring_folder_path = folder_path
                
                # Update UI elements
                display_path = folder_path
                if len(display_path) > 50:
                    display_path = "..." + display_path[-47:]
                self.folder_path_label.setText(display_path)
                self.folder_path_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                
                # Enable start button
                self.start_monitoring_btn.setEnabled(True)
                
                # Update status
                self.anomaly_alert_label.setText("System Health Monitoring Ready - Folder Selected")
                self.anomaly_alert_label.setStyleSheet("""
                    padding: 10px;
                    background-color: #E8F5E9;
                    border: 2px solid #4CAF50;
                    border-radius: 5px;
                    color: #2E7D32;
                """)
                
                # Initialize monitoring variables
                self.monitored_files = set()
                self.last_file_check = time.time()
                self._monitoring_data_cache = None
                self._data_index = 0
                
                logger.info(f"Monitoring folder set to: {folder_path}")
                
        except Exception as e:
            logger.error(f"Error selecting monitoring folder: {str(e)}")
            QMessageBox.warning(self, "Error", f"Could not set monitoring folder: {str(e)}")
    
    def get_monitoring_interval_seconds(self):
        """Calculate total monitoring interval in seconds from hours, minutes, seconds"""
        hours = self.monitoring_hours_spin.value() if hasattr(self, 'monitoring_hours_spin') else 0
        minutes = self.monitoring_minutes_spin.value() if hasattr(self, 'monitoring_minutes_spin') else 0
        seconds = self.monitoring_seconds_spin.value() if hasattr(self, 'monitoring_seconds_spin') else 5
        
        total_seconds = (hours * 3600) + (minutes * 60) + seconds
        
        # Ensure at least 1 second
        if total_seconds < 1:
            total_seconds = 1
            
        return total_seconds

    def _get_analysis_interval_seconds(self, require_enabled=False):
        """Return the Analysis auto-processing interval if configured"""
        try:
            if require_enabled:
                if not (hasattr(self, 'auto_process_check') and self.auto_process_check.isChecked()):
                    return None
            if not hasattr(self, 'process_hours_spin'):
                return None
            hours = self.process_hours_spin.value()
            minutes = self.process_minutes_spin.value()
            seconds = self.process_seconds_spin.value()
            total_seconds = (hours * 3600) + (minutes * 60) + seconds
            if total_seconds <= 0:
                return None
            return total_seconds
        except Exception as e:
            logger.error(f"Error retrieving Analysis interval: {str(e)}")
            return None
    
    def start_health_monitoring(self):
        """Start the multi-agent health monitoring system with independent pre-trained models"""
        logger.info("Starting health monitoring")
        
        # Debug info
        logger.debug("Current object attributes:")
        for attr in dir(self):
            if not attr.startswith('__'):
                logger.debug(f"Has attribute '{attr}': {hasattr(self, attr)}")
        logger.debug("-"*30)
        
        try:
            # Initialize status indicators if they don't exist
            if not hasattr(self, 'agent1_status_indicator'):
                self.agent1_status_indicator = QLabel("● Inactive")
                self.agent1_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
            
            if not hasattr(self, 'agent2_status_indicator'):
                self.agent2_status_indicator = QLabel("● Inactive")
                self.agent2_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
            
            if not hasattr(self, 'agent3_status_indicator'):
                self.agent3_status_indicator = QLabel("● Inactive")
                self.agent3_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
            
            if not hasattr(self, 'trend_status_indicator'):
                self.trend_status_indicator = QLabel("● Inactive")
                self.trend_status_indicator.setStyleSheet("color: #999; font-size: 12px; font-weight: bold;")
            
            # Check if Analysis tab has processed data and model (synchronized mode)
            if not hasattr(self, 'data_processor') or self.data_processor.data is None or len(self.data_processor.data) == 0:
                QMessageBox.warning(
                    self, "No Data Available",
                    "No data available from Analysis tab.\n\n"
                    "System Health Monitoring is synchronized with Analysis tab.\n"
                    "Please:\n"
                    "1. Load data in Analysis tab\n"
                    "2. Train a model in Analysis tab\n"
                    "3. Run prediction in Analysis tab\n\n"
                    "Then System Health Monitoring will automatically use the same data and model."
                )
                return
            
            if not hasattr(self, 'model') or self.model is None:
                QMessageBox.warning(
                    self, "No Model Available",
                    "No model available from Analysis tab.\n\n"
                    "Please train a model in Analysis tab first.\n"
                    "System Health Monitoring uses the trained model to calculate reconstruction errors."
                )
                return
            
            # Initialize pre-trained agents if not already loaded
            if not hasattr(self, 'health_agents_initialized') or not self.health_agents_initialized:
                self._initialize_health_agents()
            
            # Check if monitoring is already active
            if self.health_monitoring_active:
                QMessageBox.information(self, "Already Running", "Monitoring is already active.")
                return
            
            # Get interval in seconds
            synced_interval = self._get_analysis_interval_seconds(require_enabled=True)
            interval_sec = synced_interval or self.get_monitoring_interval_seconds()
            interval_ms = interval_sec * 1000
            using_synced_interval = synced_interval is not None
            
            # Start file monitoring
            self.health_monitoring_active = True
            self.health_monitoring_timer.start(interval_ms)
            
            # Update UI
            self.start_monitoring_btn.setEnabled(False)
            self.stop_monitoring_btn.setEnabled(True)
            self.monitoring_status_indicator.setText("[Active - Monitoring Folder]")
            self.monitoring_status_indicator.setStyleSheet("color: #4CAF50; font-weight: bold;")
            
            # Update agent status (with safety checks)
            if hasattr(self, 'agent1_status_indicator'):
                self.agent1_status_indicator.setText("● Active")
                self.agent1_status_indicator.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
            
            if hasattr(self, 'agent2_status_indicator'):
                self.agent2_status_indicator.setText("● Active")
                self.agent2_status_indicator.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
            
            if hasattr(self, 'trend_status_indicator'):
                self.trend_status_indicator.setText("● Active")
                self.trend_status_indicator.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
            else:
                # Initialize if missing
                logger.warning("trend_status_indicator not found, initializing...")
                if hasattr(self, 'health_tab'):
                    # Try to find or create it in the health tab layout
                    pass  # Will be handled by UI initialization
            
            # Update sync status
            self.sync_status_label.setText("✅ Synchronized with Analysis tab - Monitoring Active")
            self.sync_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
            
            # Update model info
            model_name = getattr(self.model, 'model_type', 'Unknown')
            self.analysis_model_info_label.setText(f"Analysis Model: {model_name} | Active")
            
            # Update alert
            data_samples = len(self.data_processor.data)
            self.anomaly_alert_label.setText(
                f"✅ Monitoring Active | Data: {data_samples} samples | Model: {model_name}"
            )
            self.anomaly_alert_label.setStyleSheet("color: #4CAF50; padding: 10px; background-color: #E8F5E9; border-radius: 5px;")
            
            # Format interval display (with safety checks)
            if using_synced_interval:
                hours = int(interval_sec // 3600)
                minutes = int((interval_sec % 3600) // 60)
                seconds = int(interval_sec % 60)
            else:
                hours = self.monitoring_hours_spin.value() if hasattr(self, 'monitoring_hours_spin') else 0
                minutes = self.monitoring_minutes_spin.value() if hasattr(self, 'monitoring_minutes_spin') else 0
                seconds = self.monitoring_seconds_spin.value() if hasattr(self, 'monitoring_seconds_spin') else 5
            interval_display = []
            if hours > 0:
                interval_display.append(f"{hours}h")
            if minutes > 0:
                interval_display.append(f"{minutes}m")
            if seconds > 0:
                interval_display.append(f"{seconds}s")
            interval_str = " ".join(interval_display) if interval_display else "5s"
            
            logger.info(f"System Health monitoring started (interval: {interval_str} = {interval_sec}s)")
            logger.info(f"Synchronized with Analysis tab - Using model: {model_name}")
            logger.info(f"Data samples: {data_samples}")
            
            if using_synced_interval:
                sync_note = " (Synced with Analysis auto-processing interval)"
            else:
                sync_note = ""
            QMessageBox.information(
                self, "Monitoring Started",
                f"System Health monitoring is now active.\n\n"
                f"- Model: {model_name}\n"
                f"- Data samples: {data_samples}\n"
                f"- Check interval: {interval_str} ({interval_sec} seconds){sync_note}\n\n"
                f"Monitoring reconstruction errors and predicting Time To Failure (TTF)."
            )
            
        except Exception as e:
            logger.error(f"Error starting health monitoring: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to start monitoring:\n{str(e)}")
    
    def stop_health_monitoring(self):
        """Stop the multi-agent health monitoring system"""
        try:
            if not self.health_monitoring_active:
                return
            
            # Confirm stop
            reply = QMessageBox.question(
                self, "Confirm Stop",
                f"Stop health monitoring?\n\n"
                f"Current buffer: {len(self.health_agent1['data_buffer']) if hasattr(self, 'health_agent1') else 0} samples\n"
                f"Agents: 3 pre-trained models\n\n"
                f"Data will be preserved.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
            
            # Stop monitoring
            self.health_monitoring_timer.stop()
            self.health_monitoring_active = False
            
            # Update UI
            self.start_monitoring_btn.setEnabled(True)
            self.stop_monitoring_btn.setEnabled(False)
            self.monitoring_status_indicator.setText("[Inactive]")
            self.monitoring_status_indicator.setStyleSheet("color: #999;")
            
            self.anomaly_alert_label.setText("Monitoring Stopped")
            self.anomaly_alert_label.setStyleSheet("color: #666; padding: 10px; background-color: #F5F5F5; border-radius: 5px;")
            
            logger.info("Multi-agent health monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping health monitoring: {str(e)}")
    
    def smooth_anomaly_scores(self, scores, window_size=5):
        """Apply smoothing to anomaly scores to reduce noise"""
        return np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    
    def validate_anomaly(self, scores, threshold, min_duration=3):
        """Validate anomalies using minimum duration and threshold"""
        anomalies = scores > threshold
        # Require minimum consecutive points above threshold
        for i in range(len(anomalies)-min_duration+1):
            if all(anomalies[i:i+min_duration]):
                return True
        return False
    
    def collect_health_data(self):
        """Monitor reconstruction errors from Analysis tab and predict Time To Failure (TTF)"""
        logger.debug("\nCollecting health data - Synchronized with Analysis tab")
        self.debug_status("Start of Health Data Collection")
        
        try:
            # Check if Analysis tab has processed data and model
            if not hasattr(self, 'data_processor') or self.data_processor.data is None or len(self.data_processor.data) == 0:
                self.sync_status_label.setText("⏳ Waiting for Analysis tab to process data...")
                self.sync_status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px;")
                self.anomaly_alert_label.setText("No data available from Analysis tab - Please process data in Analysis tab first")
                self.anomaly_alert_label.setStyleSheet(
                    "color: #FF6F00; padding: 10px; background-color: #FFF3E0; "
                    "border: 2px solid #FF9800; border-radius: 5px; font-weight: bold;"
                )
                return
            
            if not hasattr(self, 'model') or self.model is None:
                self.sync_status_label.setText("⏳ Waiting for Analysis tab to train model...")
                self.sync_status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px;")
                self.anomaly_alert_label.setText("No model available from Analysis tab - Please train model in Analysis tab first")
                self.anomaly_alert_label.setStyleSheet(
                    "color: #FF6F00; padding: 10px; background-color: #FFF3E0; "
                    "border: 2px solid #FF9800; border-radius: 5px; font-weight: bold;"
                )
                return
            
            # Update sync status
            self.sync_status_label.setText("✅ Synchronized with Analysis tab")
            self.sync_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
            
            # Update model info
            model_name = getattr(self.model, 'model_type', 'Unknown')
            self.analysis_model_info_label.setText(f"Analysis Model: {model_name}")
            
            # Determine base sampling interval (prefer Analysis interval)
            sample_interval_seconds = self._get_analysis_interval_seconds(require_enabled=False)
            if not sample_interval_seconds:
                sample_interval_seconds = self.get_monitoring_interval_seconds()
            if not sample_interval_seconds or sample_interval_seconds <= 0:
                sample_interval_seconds = 5.0
            
            # Use preprocessed data from Analysis tab if available, otherwise use raw data
            if hasattr(self.data_processor, 'preprocessed_data') and self.data_processor.preprocessed_data is not None:
                telemetry_data = self.data_processor.preprocessed_data.copy()
            else:
                telemetry_data = self.data_processor.data.copy()
            
            # Ensure telemetry_data is a DataFrame (not numpy array)
            if not isinstance(telemetry_data, pd.DataFrame):
                # Convert numpy array to DataFrame
                if isinstance(telemetry_data, np.ndarray):
                    logger.info(f"Converting numpy array to DataFrame (shape: {telemetry_data.shape})")
                    # Create column names if needed
                    if telemetry_data.ndim == 1:
                        telemetry_data = pd.DataFrame(telemetry_data, columns=['value'])
                    else:
                        telemetry_data = pd.DataFrame(
                            telemetry_data, 
                            columns=[f'feature_{i}' for i in range(telemetry_data.shape[1])]
                        )
                else:
                    logger.error(f"Unexpected data type: {type(telemetry_data)}")
                    return
            
            # Estimate sample interval from timestamp-like columns before dropping them
            timestamp_column = None
            for candidate in ['timestamp', 'Timestamp', 'time', 'Time', 'datetime', 'DateTime']:
                if candidate in telemetry_data.columns:
                    timestamp_column = candidate
                    break
            
            if timestamp_column:
                try:
                    ts = pd.to_datetime(telemetry_data[timestamp_column], errors='coerce')
                    diffs = ts.diff().dt.total_seconds().dropna()
                    if len(diffs) > 0:
                        median_diff = np.median(diffs[np.isfinite(diffs)])
                        if median_diff and median_diff > 0:
                            sample_interval_seconds = float(median_diff)
                            logger.debug(f"Sample interval derived from timestamps: {sample_interval_seconds:.3f}s")
                except Exception as e:
                    logger.warning(f"Could not derive sample interval from timestamp column: {str(e)}")
            
            # Get numeric columns only
            try:
                numeric_cols = telemetry_data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) == 0:
                    logger.warning("No numeric columns found in data")
                    return
                
                telemetry_data = telemetry_data[numeric_cols]
            except Exception as e:
                logger.error(f"Error selecting numeric columns: {str(e)}")
                # Fallback: use all columns if select_dtypes fails
                logger.warning("Falling back to using all columns")
                telemetry_data = telemetry_data
            
            # Validate numeric data
            telemetry_data = telemetry_data.replace([np.inf, -np.inf], np.nan)
            invalid_ratio = telemetry_data.isna().sum().sum() / (telemetry_data.size or 1)
            if invalid_ratio > 0:
                logger.warning(f"Found {invalid_ratio:.2%} invalid numeric values - filling with 0.0")
            telemetry_data = telemetry_data.fillna(0.0)
            
            if telemetry_data.empty or telemetry_data.shape[1] == 0:
                logger.warning("Telemetry data is empty after cleaning")
                return
            
            if telemetry_data.shape[0] < 2:
                logger.warning("Not enough rows for reconstruction error calculation")
                return
            
            # Initialize reconstruction error buffer if not exists
            if not hasattr(self, 'reconstruction_error_history'):
                self.reconstruction_error_history = []
            
            # Initialize data index for cycling through data
            if not hasattr(self, '_health_data_index'):
                self._health_data_index = 0
            
            # Cycle through data (replay/cycle through the same dataset)
            if self._health_data_index >= len(telemetry_data):
                self._health_data_index = 0  # Restart from beginning
            
            # Get current batch of data for reconstruction error calculation
            # Use a sliding window approach
            batch_size = getattr(self, 'health_batch_size', 200)
            if batch_size <= 0:
                batch_size = 200
            self.health_batch_size = batch_size
            window_size = min(batch_size, len(telemetry_data))
            start_idx = self._health_data_index
            end_idx = min(start_idx + window_size, len(telemetry_data))
            
            if end_idx - start_idx < 10:  # Need at least 10 samples
                # If near end, wrap around
                batch_data = pd.concat([
                    telemetry_data.iloc[start_idx:],
                    telemetry_data.iloc[:window_size - (len(telemetry_data) - start_idx)]
                ])
            else:
                batch_data = telemetry_data.iloc[start_idx:end_idx]
            
            self._health_data_index = end_idx
            
            # Keep as DataFrame - models may expect DataFrame format
            # Clean the data
            batch_data = batch_data.astype(float)
            batch_data = batch_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            if batch_data.isna().any().any():
                logger.debug("Residual NaNs found after cleaning batch data - filling with zeros")
                batch_data = batch_data.fillna(0.0)
            
            if batch_data.shape[1] == 0:
                logger.warning("Batch data has no numeric features after cleaning")
                return
            
       
            try:
                reconstruction_errors = self._calculate_reconstruction_errors(self.model, batch_data)
                
                if reconstruction_errors is None or len(reconstruction_errors) == 0:
                    logger.warning("Could not calculate reconstruction errors")
                    return
                
                # Add to history (keep last 500 points for trend analysis)
                self.reconstruction_error_history.extend(reconstruction_errors.tolist())
                if len(self.reconstruction_error_history) > 500:
                    self.reconstruction_error_history = self.reconstruction_error_history[-500:]
                
            except Exception as e:
                logger.error(f"Error calculating reconstruction errors: {str(e)}")
                return
            

            threshold_mode = getattr(self, "health_threshold_mode", "auto")
            if threshold_mode != "manual":
                self.health_threshold_mode = "auto"
                # Recompute AUTO threshold continuously from recent history to adapt to drift.
                if len(self.reconstruction_error_history) > 0:
                    recent_window = self.reconstruction_error_history[-min(300, len(self.reconstruction_error_history)):]
                    calculated_threshold = np.percentile(recent_window, 95)
                    # Validate calculated threshold
                    if calculated_threshold is not None and not np.isnan(calculated_threshold) and calculated_threshold > 0:
                        self.health_reconstruction_threshold = float(calculated_threshold)
                        self.threshold_value_label.setText(f"Auto: {self.health_reconstruction_threshold:.6f}")
                        self.threshold_value_label.setStyleSheet("font-weight: bold; color: #2196F3;")
                    else:
                        # Fallback: use mean + 2*std if percentile calculation fails
                        mean_error = np.mean(recent_window)
                        std_error = np.std(recent_window)
                        self.health_reconstruction_threshold = float(mean_error + 2 * std_error)
                        self.threshold_value_label.setText(f"Auto (fallback): {self.health_reconstruction_threshold:.6f}")
                        self.threshold_value_label.setStyleSheet("font-weight: bold; color: #FF9800;")
                        logger.warning(f"Percentile calculation failed, using fallback threshold: {self.health_reconstruction_threshold:.6f}")
                else:
                    # No history yet - use a default threshold
                    self.health_reconstruction_threshold = 0.1
                    self.threshold_value_label.setText(f"Default: {self.health_reconstruction_threshold:.6f}")
                    self.threshold_value_label.setStyleSheet("font-weight: bold; color: #999;")
                    logger.info(f"Using default threshold: {self.health_reconstruction_threshold:.6f} (no history yet)")
            else:
                # Keep manual threshold value stable and explicit in UI.
                self.threshold_value_label.setText(f"Manual: {float(self.health_reconstruction_threshold):.6f}")
                self.threshold_value_label.setStyleSheet("font-weight: bold; color: #FF9800;")
            
            # Final validation - ensure threshold is never None, NaN, or negative
            if self.health_reconstruction_threshold is None or np.isnan(self.health_reconstruction_threshold) or self.health_reconstruction_threshold <= 0:
                logger.error("Invalid threshold detected, resetting to default")
                self.health_reconstruction_threshold = 0.1
                self.threshold_value_label.setText(f"Default: {self.health_reconstruction_threshold:.6f}")
                self.threshold_value_label.setStyleSheet("font-weight: bold; color: #999;")
            
            threshold = float(self.health_reconstruction_threshold)  # Ensure it's a float
            
  
            ttf_prediction = None
            ttf_confidence = None
            
            if len(self.reconstruction_error_history) >= 50:  # Need at least 50 points for ARIMA
                try:
                    # Use ARIMA to predict when reconstruction error will cross threshold
                    ttf_prediction, ttf_confidence = self._predict_time_to_failure(
                        np.array(self.reconstruction_error_history),
                        threshold,
                        sample_interval_seconds
                    )
                except Exception as e:
                    logger.error(f"Error predicting TTF: {str(e)}")
      
            
            # Current reconstruction error (mean) + peak error for spike-aware alerting
            current_error = np.mean(reconstruction_errors) if len(reconstruction_errors) > 0 else 0.0
            peak_error = np.max(reconstruction_errors) if len(reconstruction_errors) > 0 else current_error
            current_time = datetime.datetime.now()
            
            # Check if error is increasing
            if len(self.reconstruction_error_history) >= 10:
                recent_errors = self.reconstruction_error_history[-10:]
                error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]  # Slope
                is_increasing = error_trend > 0
            else:
                is_increasing = False
            
            # Update TTF display
            if ttf_prediction is not None:
                if ttf_prediction > 0:
                    hours = int(ttf_prediction)
                    minutes = int((ttf_prediction - hours) * 60)
                    ttf_display = f"{hours}h {minutes}m"
                    if ttf_confidence:
                        ttf_display += f" (Confidence: {ttf_confidence:.1f}%)"
                    
                    if is_increasing and current_error < threshold:
                        self.ttf_label.setText(f"⚠️ {ttf_display} - Error increasing")
                        self.ttf_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800; padding: 5px;")
                    elif current_error >= threshold:
                        self.ttf_label.setText("🔴 CRITICAL - Threshold exceeded!")
                        self.ttf_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #f44336; padding: 5px;")
                    else:
                        self.ttf_label.setText(f"✅ {ttf_display}")
                        self.ttf_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50; padding: 5px;")
                else:
                    self.ttf_label.setText("⚠️ TTF prediction unavailable")
                    self.ttf_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF9800; padding: 5px;")
            else:
                samples_needed = max(0, 50 - len(self.reconstruction_error_history))
                self.ttf_label.setText(f"⏳ Collecting data... ({samples_needed} more samples needed)")
                self.ttf_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #666; padding: 5px;")
            
            # Update status
            status_text = (
                f"Monitoring: {len(self.reconstruction_error_history)} samples | "
                f"Current Error: {current_error:.6f} | "
                f"Peak Error: {peak_error:.6f} | "
                f"Threshold: {threshold:.6f}"
            )
            if ttf_prediction:
                status_text += f" | TTF: {ttf_prediction:.1f} hours"
            
            self.anomaly_alert_label.setText(status_text)

            # Update metric cards in the modern Health tab.
            if hasattr(self, "current_error_label"):
                self.current_error_label.setText(f"{current_error:.6f}")
            if hasattr(self, "threshold_status_label"):
                self.threshold_status_label.setText(f"{threshold:.6f}")
            if hasattr(self, "prediction_status_label"):
                if current_error >= threshold:
                    self.prediction_status_label.setText("Critical")
                elif ttf_prediction is not None and ttf_prediction <= 4:
                    self.prediction_status_label.setText("Warning")
                elif ttf_prediction is not None:
                    self.prediction_status_label.setText("Forecasting")
                else:
                    self.prediction_status_label.setText("Monitoring")
            if hasattr(self, "last_update_label"):
                self.last_update_label.setText(current_time.strftime("%H:%M:%S"))
            
            if current_error >= threshold:
                self.anomaly_alert_label.setStyleSheet(
                    "color: #B71C1C; padding: 10px; background-color: #FFEBEE; "
                    "border: 2px solid #f44336; border-radius: 5px; font-weight: bold;"
                )
            elif is_increasing:
                self.anomaly_alert_label.setStyleSheet(
                    "color: #FF6F00; padding: 10px; background-color: #FFF3E0; "
                    "border: 2px solid #FF9800; border-radius: 5px; font-weight: bold;"
                )
            else:
                self.anomaly_alert_label.setStyleSheet(
                    "color: #4CAF50; padding: 10px; background-color: #E8F5E9; border-radius: 5px;"
                )
            
        
            alert_payload = {
                'model_name': getattr(self.model, 'model_type', 'Unknown'),
                'reconstruction_error': float(current_error),
                'peak_error': float(peak_error),
                'threshold': float(threshold),
                'exceeds_threshold': bool(current_error >= threshold or peak_error >= threshold),
                'health_score': (
                    modular_compute_health_score(float(current_error), float(threshold))
                    if modular_compute_health_score is not None
                    else max(0.0, 1.0 - (current_error / (threshold + 1e-6)))
                )
            }
            
            alert_severity = None
            alert_message = None
            if current_error >= threshold or peak_error >= threshold:
                trigger_error = max(current_error, peak_error)
                alert_severity = 'critical'
                alert_message = f"Reconstruction error {trigger_error:.4f} exceeded threshold {threshold:.4f}"
                self._trigger_health_alert(
                    alert_severity,
                    alert_message,
                    alert_payload
                )
            elif ttf_prediction is not None and ttf_prediction > 0:
                alert_payload['failure_prediction'] = {
                    'status': 'ttf_warning' if ttf_prediction <= 2 else 'ttf_info',
                    'time_to_failure': f"{ttf_prediction:.2f}h",
                    'confidence': ttf_confidence
                }
                if ttf_prediction <= 1:
                    alert_severity = 'critical'
                    alert_message = f"Predicted failure in {ttf_prediction:.2f} hours"
                    self._trigger_health_alert(
                        alert_severity,
                        alert_message,
                        alert_payload
                    )
                elif ttf_prediction <= 4:
                    alert_severity = 'warning'
                    alert_message = f"Potential failure within {ttf_prediction:.2f} hours"
                    self._trigger_health_alert(
                        alert_severity,
                        alert_message,
                        alert_payload
                    )
                else:
                    alert_severity = 'info'
                    alert_message = f"Monitoring anomaly trend (TTF ~{ttf_prediction:.2f}h)"
                    self._trigger_health_alert(
                        alert_severity,
                        alert_message,
                        alert_payload
                    )

            # Keep Active Alerts table populated with meaningful warning/critical events.
            if alert_severity in {"warning", "critical"}:
                if not hasattr(self, "_last_health_table_alert_time"):
                    self._last_health_table_alert_time = 0.0
                if not hasattr(self, "_last_health_table_alert_signature"):
                    self._last_health_table_alert_signature = ""
                signature = f"{alert_severity}:{round(float(max(current_error, peak_error)), 5)}:{round(float(threshold), 5)}"
                now_ts = time.time()
                table_cooldown = max(15, int(getattr(self, "health_alert_cooldown", 60)))
                if (now_ts - self._last_health_table_alert_time) >= table_cooldown or signature != self._last_health_table_alert_signature:
                    table_result = {
                        "timestamp": current_time,
                        "model_name": alert_payload.get("model_name", "Unknown"),
                        "reconstruction_error": float(max(current_error, peak_error)),
                        "threshold": float(threshold),
                        "exceeds_threshold": bool(alert_payload.get("exceeds_threshold")),
                        "status": str(alert_severity),
                        "failure_prediction": alert_payload.get("failure_prediction"),
                    }
                    self.add_alert_to_table(table_result)
                    self._last_health_table_alert_time = now_ts
                    self._last_health_table_alert_signature = signature

            # Refresh model overview table on every cycle
            self.update_models_status_display()
    
            if len(self.reconstruction_error_history) >= 10:
                self._update_reconstruction_error_plot(
                    self.reconstruction_error_history,
                    threshold,
                    ttf_prediction
                )
                
        except Exception as e:
            logger.error(f"Error collecting health data: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _calculate_reconstruction_errors(self, model, X):
        """Calculate reconstruction errors from Analysis model
        
        Args:
            model: Trained model object
            X: Input data (DataFrame or numpy array)
        """
        try:
            # Ensure X is DataFrame (models typically expect DataFrame)
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            elif isinstance(X, pd.DataFrame) or (hasattr(X, 'columns') and hasattr(X, 'copy')):
                X_df = X.copy()
            else:
                try:
                    X_df = pd.DataFrame(X)
                except Exception:
                    logger.error(f"Unexpected data type for reconstruction errors: {type(X)}")
                    return None
            
            # Convert to numpy for calculations
            X_np = X_df.values.astype(float)
            if X_np.size == 0:
                logger.warning("No data available for reconstruction error calculation")
                return None
            
            if not np.isfinite(X_np).all():
                logger.debug("Non-finite values detected in X_np - applying nan_to_num")
                X_np = np.nan_to_num(X_np, nan=0.0, posinf=1.0, neginf=-1.0)
            errors = None
            
            # Handle different model types
            model_type = getattr(model, 'model_type', 'unknown')
            
            if model_type in ['autoencoder', 'lstm_autoencoder', 'cnn_autoencoder']:
                # Autoencoder models - predict reconstruction
                if hasattr(model, 'predict'):
                    try:
                        # Try with DataFrame first (most models expect DataFrame)
                        reconstructed = model.predict(X_df)
                        # Convert to numpy array
                        if isinstance(reconstructed, pd.DataFrame):
                            reconstructed = reconstructed.values
                        elif hasattr(reconstructed, 'values'):
                            reconstructed = reconstructed.values
                        elif hasattr(reconstructed, 'to_numpy'):
                            reconstructed = reconstructed.to_numpy()
                        reconstructed = np.asarray(reconstructed)
                        
                        # Ensure shapes match
                        if reconstructed.ndim == 1:
                            reconstructed = reconstructed.reshape(-1, 1)
                        if X_np.ndim == 1:
                            X_np = X_np.reshape(-1, 1)
                        
                        # Match dimensions for error calculation
                        min_samples = min(X_np.shape[0], reconstructed.shape[0])
                        min_features = min(X_np.shape[1] if X_np.ndim > 1 else 1, 
                                         reconstructed.shape[1] if reconstructed.ndim > 1 else 1)
                        
                        X_trimmed = X_np[:min_samples]
                        if X_trimmed.ndim == 1:
                            X_trimmed = X_trimmed.reshape(-1, 1)
                        
                        recon_trimmed = reconstructed[:min_samples]
                        if recon_trimmed.ndim == 1:
                            recon_trimmed = recon_trimmed.reshape(-1, 1)
                        
                        # Calculate reconstruction errors
                        if X_trimmed.shape == recon_trimmed.shape:
                            errors = np.mean(np.square(X_trimmed - recon_trimmed), axis=1)
                        else:
                            # Flatten and calculate
                            errors = np.mean(np.square(X_trimmed.flatten()[:len(recon_trimmed.flatten())] - recon_trimmed.flatten()))
                            errors = np.array([errors] * min_samples)
                    except Exception as e:
                        logger.error(f"Error in model prediction: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        errors = None
                else:
                    errors = None
                    
            elif model_type in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
                # These models return anomaly scores, convert to reconstruction-like errors
                try:
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(X_np)  # sklearn models expect numpy
                        # Convert scores to errors (lower scores = higher errors)
                        errors = -scores
                        errors = errors - np.min(errors)  # Normalize to start from 0
                    elif hasattr(model, 'score_samples'):
                        scores = model.score_samples(X_np)  # sklearn models expect numpy
                        errors = -scores
                        errors = errors - np.min(errors)
                    else:
                        errors = None
                except Exception as e:
                    logger.error(f"Error with sklearn model prediction: {str(e)}")
                    errors = None
                    
            elif hasattr(model, 'transform') and hasattr(model, 'inverse_transform'):
                # PCA-like models
                try:
                    encoded = model.transform(X_np)  # sklearn models expect numpy
                    reconstructed = model.inverse_transform(encoded)
                    errors = np.mean(np.square(X_np - reconstructed), axis=1)
                except Exception as e:
                    logger.error(f"Error with transform/inverse_transform: {str(e)}")
                    errors = None
                    
            elif hasattr(model, 'predict'):
                # Generic model - try to use predict as reconstruction
                try:
                    # Try DataFrame first (most custom models expect DataFrame)
                    reconstructed = model.predict(X_df)
                    # Convert to numpy
                    if isinstance(reconstructed, pd.DataFrame):
                        reconstructed = reconstructed.values
                    elif hasattr(reconstructed, 'values'):
                        reconstructed = reconstructed.values
                    elif hasattr(reconstructed, 'to_numpy'):
                        reconstructed = reconstructed.to_numpy()
                    reconstructed = np.asarray(reconstructed)
                    
                    # Calculate errors
                    if X_np.shape == reconstructed.shape:
                        errors = np.mean(np.square(X_np - reconstructed), axis=1)
                    else:
                        # If prediction is different shape, use it as error directly
                        errors = np.abs(reconstructed.flatten())
                        if len(errors) != len(X_np):
                            # Repeat or interpolate to match length
                            errors = np.interp(np.linspace(0, len(errors)-1, len(X_np)), 
                                             np.arange(len(errors)), errors)
                except Exception as e:
                    logger.error(f"Error in generic model prediction: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    errors = None
            else:
                logger.warning(f"Unknown model type for reconstruction error calculation: {model_type}")
                errors = None
            
            if errors is None or len(errors) == 0:
                errors = self._fallback_reconstruction_errors(model, X_df, X_np)
            
            if errors is None:
                return None
            
            errors = np.nan_to_num(np.asarray(errors).flatten(), nan=0.0, posinf=1.0, neginf=0.0)
            return errors
            
        except Exception as e:
            logger.error(f"Error calculating reconstruction errors: {str(e)}")
            return None
    
    def _fallback_reconstruction_errors(self, model, X_df, X_np):
        """Fallback strategy when model type is unknown or primary method fails"""
        n_samples = len(X_df)
        try:
            # Try decision_function / score_samples if available
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X_np)
                errors = -np.asarray(scores).flatten()
                errors = errors - np.min(errors)
                return errors
            
            if hasattr(model, 'score_samples'):
                scores = model.score_samples(X_np)
                errors = -np.asarray(scores).flatten()
                errors = errors - np.min(errors)
                return errors
            
            # Try predict_proba / predict
            if hasattr(model, 'predict_proba'):
                scores = model.predict_proba(X_df)
                scores = np.asarray(scores)
                if scores.ndim > 1:
                    scores = scores.max(axis=1)
                errors = 1.0 - scores
                return errors
            
            if hasattr(model, 'predict'):
                predictions = model.predict(X_df)
                predictions = np.asarray(predictions).flatten()
                if len(predictions) != n_samples:
                    predictions = np.interp(
                        np.linspace(0, len(predictions) - 1, n_samples),
                        np.arange(len(predictions)),
                        predictions
                    )
                # Treat deviations from median as reconstruction error proxy
                median_val = np.median(predictions)
                errors = np.abs(predictions - median_val)
                return errors
            
            logger.warning("Fallback reconstruction error: using constant placeholder values")
        except Exception as e:
            logger.error(f"Fallback reconstruction error failed: {str(e)}")
        
        # Final fallback: small constant errors to keep pipeline alive
        return np.full(n_samples, 0.1)
    
    def _predict_time_to_failure(self, error_history, threshold, sample_interval_seconds=5.0):
        """Use ARIMA to predict Time To Failure (TTF) based on reconstruction error trend"""
        try:
            if len(error_history) < 50:
                return None, None
            
            if sample_interval_seconds is None or sample_interval_seconds <= 0:
                sample_interval_seconds = 5.0
            step_seconds = float(sample_interval_seconds)
            
            # Use recent history for prediction
            recent_errors = error_history[-100:] if len(error_history) > 100 else error_history
            error_series = np.array(recent_errors)
            error_series = np.nan_to_num(error_series, nan=0.0, posinf=np.max(error_series[np.isfinite(error_series)] + 1e-3 if np.isfinite(error_series).any() else 1.0), neginf=0.0)
            
            if np.allclose(error_series, error_series[0]):
                logger.info("Reconstruction error series is constant - no TTF prediction")
                return None, None
            
            # Check if error is increasing
            if len(error_series) < 10:
                return None, None
            
            # Calculate trend
            x = np.arange(len(error_series))
            trend_slope = np.polyfit(x, error_series, 1)[0]
            
            # If error is decreasing or stable, TTF is not applicable
            if trend_slope <= 0:
                return None, None
            
            # If current error already exceeds threshold, TTF is 0
            current_error = error_series[-1]
            if current_error >= threshold:
                return 0.0, 100.0  # Already failed
            
            # Prepare ARIMA cache for performance
            if not hasattr(self, '_arima_cache'):
                self._arima_cache = {'model': None, 'history_len': 0, 'order': None, 'fitted_at': 0}
            
            cache = self._arima_cache
            fitted_model = None
            
            # Reuse cached model if history hasn't changed much
            if cache['model'] is not None and cache['history_len'] >= len(error_series) - 5:
                fitted_model = cache['model']
                logger.debug("Reusing cached ARIMA model")
            
            # Fit ARIMA model
            try:
                from statsmodels.tsa.arima.model import ARIMA
                
                # Make series stationary if needed (differencing)
                diff_errors = np.diff(error_series)
                
                if fitted_model is None:
                    arima_orders = [(1, 1, 1), (2, 1, 2), (1, 2, 1)]
                    for order in arima_orders:
                        try:
                            arima_model = ARIMA(error_series, order=order)
                            fitted_model = arima_model.fit()
                            cache.update({'model': fitted_model, 'history_len': len(error_series), 'order': order, 'fitted_at': time.time()})
                            logger.debug(f"ARIMA model fitted with order {order}")
                            break
                        except Exception as fit_err:
                            logger.warning(f"ARIMA order {order} failed: {str(fit_err)}")
                
                if fitted_model is not None:
                    # Forecast ahead until threshold is crossed
                    max_forecast_steps = 1000  # Maximum steps to forecast
                    forecast_steps = 0
                    predicted_error = current_error
                    
                    try:
                        while predicted_error < threshold and forecast_steps < max_forecast_steps:
                            forecast_steps += 1
                            forecast = fitted_model.forecast(steps=forecast_steps)
                            predicted_error = forecast[-1]
                            
                            # Safety check: if forecast is decreasing, break
                            if forecast_steps > 1 and forecast[-1] < forecast[-2]:
                                break
                    except Exception as forecast_err:
                        logger.warning(f"ARIMA forecast failed: {str(forecast_err)}")
                        forecast_steps = 0
                    
                    if forecast_steps > 0:
                        # Convert steps to time using actual sampling interval
                        ttf_seconds = forecast_steps * step_seconds
                        ttf_hours = ttf_seconds / 3600.0
                        
                        # Calculate confidence based on model fit quality
                        # Simple heuristic: confidence decreases with forecast horizon
                        confidence = max(50.0, 100.0 - (forecast_steps * 0.1))
                        
                        return ttf_hours, confidence
                # Fallback: linear trend extrapolation
                logger.warning("ARIMA fitting failed for all orders, using linear extrapolation")
                if trend_slope > 0:
                    steps_to_threshold = (threshold - current_error) / trend_slope
                    ttf_seconds = max(0, steps_to_threshold) * step_seconds
                    ttf_hours = ttf_seconds / 3600.0
                    confidence = 70.0  # Lower confidence for linear extrapolation
                    return ttf_hours, confidence
                else:
                    return None, None
                        
            except ImportError:
                logger.warning("statsmodels not available, using linear extrapolation")
                # Fallback: simple linear extrapolation
                if trend_slope > 0:
                    steps_to_threshold = (threshold - current_error) / trend_slope
                    ttf_seconds = max(0, steps_to_threshold) * step_seconds
                    ttf_hours = ttf_seconds / 3600.0
                    confidence = 60.0
                    return ttf_hours, confidence
                else:
                    return None, None
                
        except Exception as e:
            logger.error(f"Error predicting TTF: {str(e)}")
            return None, None
    
    def _update_system_health_sync(self):
        """Update System Health tab sync status when Analysis tab processes new data"""
        try:
            if getattr(self, "custom_tabs_only_mode", False):
                return

            # Update sync status label
            if hasattr(self, 'sync_status_label'):
                if hasattr(self, 'data_processor') and self.data_processor.data is not None and len(self.data_processor.data) > 0:
                    if hasattr(self, 'model') and self.model is not None:
                        self.sync_status_label.setText("✅ Synchronized - New data available from Analysis tab")
                        self.sync_status_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
                        
                        # Enable Start button if monitoring is not active
                        if hasattr(self, 'start_monitoring_btn') and not self.health_monitoring_active:
                            self.start_monitoring_btn.setEnabled(True)
                        
                        # Update model info
                        if hasattr(self, 'analysis_model_info_label'):
                            model_name = getattr(self.model, 'model_type', 'Unknown')
                            self.analysis_model_info_label.setText(f"Analysis Model: {model_name} | Ready")
                    else:
                        self.sync_status_label.setText("⏳ Waiting for model training in Analysis tab...")
                        self.sync_status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px;")
                else:
                    self.sync_status_label.setText("⏳ Waiting for data processing in Analysis tab...")
                    self.sync_status_label.setStyleSheet("color: #FF9800; font-weight: bold; padding: 5px;")
            
            # Reset reconstruction error history when new data is processed
            if hasattr(self, 'reconstruction_error_history'):
                # Keep some history but mark that new data is available
                logger.info("System Health tab notified of new data from Analysis tab")
                
        except Exception as e:
            logger.error(f"Error updating System Health sync: {str(e)}")
    
    def _update_reconstruction_error_plot(self, error_history, threshold, ttf_prediction=None):
        """Update visualization with reconstruction error trend"""
        try:
            if not hasattr(self, 'health_fig') or self.health_fig is None:
                return
            
            self.health_fig.clear()
            ax = self.health_fig.add_subplot(111)
            
            # Plot reconstruction error history
            x_data = list(range(len(error_history)))
            ax.plot(x_data, error_history, 'b-', linewidth=2, label='Reconstruction Error', alpha=0.7)
            
            # Plot threshold line
            ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                      label=f'Threshold ({threshold:.6f})')
            
            # Highlight current error
            if len(error_history) > 0:
                current_error = error_history[-1]
                ax.scatter([len(error_history)-1], [current_error], 
                          color='red' if current_error >= threshold else 'green',
                          s=100, zorder=5, label='Current Error')
            
            # If TTF prediction available, show forecast
            if ttf_prediction and ttf_prediction > 0 and len(error_history) >= 50:
                try:
                    # Simple linear forecast visualization
                    recent_errors = error_history[-20:]
                    x_recent = list(range(len(error_history)-20, len(error_history)))
                    trend_slope = np.polyfit(x_recent, recent_errors, 1)[0]
                    
                    if trend_slope > 0:
                        # Extend trend line to show when threshold will be crossed
                        forecast_x = list(range(len(error_history), len(error_history) + int(ttf_prediction)))
                        forecast_y = [error_history[-1] + trend_slope * (i+1) for i in range(len(forecast_x))]
                        forecast_y = [min(y, threshold * 1.2) for y in forecast_y]  # Cap at 1.2x threshold
                        
                        ax.plot(forecast_x, forecast_y, 'orange', linestyle=':', 
                               linewidth=2, alpha=0.6, label=f'TTF Forecast ({ttf_prediction:.1f}h)')
                except:
                    pass
            else:
                # Make prediction state explicit when no reliable TTF forecast is available.
                ax.text(
                    0.99,
                    0.95,
                    "No imminent failure prediction",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#607D8B",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECEFF1", edgecolor="#CFD8DC", alpha=0.9),
                )
            
            ax.set_xlabel('Sample Index', fontsize=10)
            ax.set_ylabel('Reconstruction Error', fontsize=10)
            ax.set_title('Reconstruction Error Trend & Time To Failure Prediction', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            self.health_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating reconstruction error plot: {str(e)}")
    
    def _run_agent1_detection(self, sample_array):
        """Agent 1: LSTM+Autoencoder - Reconstruction error based detection"""
        try:
            # Simulate reconstruction error (0.0-1.0)
            # In real implementation, this would use trained LSTM autoencoder
            mean_val = np.mean(sample_array)
            std_val = np.std(sample_array)
            
            # Simple anomaly score: higher std or extreme mean = higher score
            score = min(1.0, abs(std_val / (mean_val + 1e-6)) * 0.2 + np.random.uniform(0, 0.3))
            
            return score
        except Exception as e:
            logger.error(f"Agent 1 detection error: {str(e)}")
            return 0.5
    
    def _run_agent2_detection(self, sample_array):
        """Agent 2: CNN - Pattern based detection"""
        try:
            # Simulate CNN detection score (0.0-1.0)
            # In real implementation, this would use trained CNN
            max_val = np.max(sample_array)
            min_val = np.min(sample_array)
            
            # Simple anomaly score: larger range = higher score
            score = min(1.0, abs(max_val - min_val) / (abs(max_val) + 1e-6) * 0.3 + np.random.uniform(0, 0.25))
            
            return score
        except Exception as e:
            logger.error(f"Agent 2 detection error: {str(e)}")
            return 0.5
    
    def _run_agent3_detection(self, sample_array):
        """Agent 3: ARIMA - Trend based detection"""
        try:
            # Simulate ARIMA trend anomaly score (0.0-1.0)
            # In real implementation, this would use ARIMA forecasting
            
            # Use buffer for trend analysis
            if len(self.health_agent3['data_buffer']) > 5:
                recent_scores = self.health_agent3['data_buffer'][-5:]
                trend = np.mean(recent_scores)
            else:
                trend = 0.3
            
            # Random variation around trend
            score = min(1.0, max(0.0, trend + np.random.uniform(-0.15, 0.15)))
            
            return score
        except Exception as e:
            logger.error(f"Agent 3 detection error: {str(e)}")
            return 0.5
    
    def _update_agent1_plot(self, data_buffer):
        """Update Agent 1 (LSTM+Autoencoder) reconstruction error plot"""
        try:
            ax = self.agent1_canvas.figure.clear()
            ax = self.agent1_canvas.figure.add_subplot(111)
            
            x_data = list(range(len(data_buffer)))
            
            # Plot reconstruction error over time
            ax.plot(x_data, data_buffer, 'b-', linewidth=2, label='Recon Error')
            ax.axhline(y=self.health_agent1['threshold'], color='r', linestyle='--', 
                      linewidth=1.5, label=f"Threshold ({self.health_agent1['threshold']:.2f})")
            
            ax.set_title('Agent 1: LSTM+Autoencoder Reconstruction Error', fontsize=10, fontweight='bold')
            ax.set_xlabel('Sample', fontsize=9)
            ax.set_ylabel('Error', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            self.agent1_canvas.draw()
        except Exception as e:
            logger.error(f"Error updating Agent 1 plot: {str(e)}")
    
    def _update_agent2_plot(self, data_buffer):
        """Update Agent 2 (CNN) detection score plot"""
        try:
            ax = self.agent2_canvas.figure.clear()
            ax = self.agent2_canvas.figure.add_subplot(111)
            
            x_data = list(range(len(data_buffer)))
            
            # Plot CNN detection scores
            ax.plot(x_data, data_buffer, 'g-', linewidth=2, label='Detection Score')
            ax.axhline(y=self.health_agent2['threshold'], color='r', linestyle='--',
                      linewidth=1.5, label=f"Threshold ({self.health_agent2['threshold']:.2f})")
            
            ax.set_title('Agent 2: CNN Pattern Detection', fontsize=10, fontweight='bold')
            ax.set_xlabel('Sample', fontsize=9)
            ax.set_ylabel('Score', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            self.agent2_canvas.draw()
        except Exception as e:
            logger.error(f"Error updating Agent 2 plot: {str(e)}")
    
    def _update_agent3_plot(self, data_buffer):
        """Update Agent 3 (ARIMA) trend analysis plot"""
        try:
            ax = self.trend_canvas.figure.clear()
            ax = self.trend_canvas.figure.add_subplot(111)
            
            x_data = list(range(len(data_buffer)))
            
            # Plot trend scores
            ax.plot(x_data, data_buffer, 'm-', linewidth=2, label='Trend Score')
            ax.axhline(y=self.health_agent3['threshold'], color='r', linestyle='--',
                      linewidth=1.5, label=f"Threshold ({self.health_agent3['threshold']:.2f})")
            
            ax.set_title('Agent 3: ARIMA Trend Analysis', fontsize=10, fontweight='bold')
            ax.set_xlabel('Sample', fontsize=9)
            ax.set_ylabel('Score', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            self.trend_canvas.draw()
        except Exception as e:
            logger.error(f"Error updating Agent 3 plot: {str(e)}")
    
    def _update_fusion_plot(self, fusion_buffer, is_anomaly, fusion_score):
        """Update fusion (combined) result plot"""
        try:
            ax = self.combined_canvas.figure.clear()
            ax = self.combined_canvas.figure.add_subplot(111)
            
            x_data = list(range(len(fusion_buffer)))
            
            # Color code based on anomaly detection
            colors = ['red' if val > 0.65 else 'green' for val in fusion_buffer]
            
            # Plot fusion scores
            ax.scatter(x_data, fusion_buffer, c=colors, s=30, alpha=0.6)
            ax.plot(x_data, fusion_buffer, 'k-', linewidth=1.5, alpha=0.5, label='Fusion Score')
            ax.axhline(y=0.65, color='r', linestyle='--', linewidth=1.5, label='Threshold (0.65)')
            
            ax.set_title('Multi-Agent Fusion Result', fontsize=10, fontweight='bold')
            ax.set_xlabel('Sample', fontsize=9)
            ax.set_ylabel('Fusion Score', fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            self.combined_canvas.draw()
        except Exception as e:
            logger.error(f"Error updating fusion plot: {str(e)}")
    
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(traceback.format_exc())
    
    def train_multi_agent_system(self):
        """Train the multi-agent system on collected data"""
        try:
            logger.info("="*60)
            logger.info("INITIATING MULTI-AGENT SYSTEM TRAINING")
            logger.info("="*60)
            
            if len(self.health_data_buffer) < 100:
                QMessageBox.warning(
                    self, "Insufficient Data",
                    f"Need at least 100 samples to train.\nCurrently have: {len(self.health_data_buffer)}"
                )
                return
            
            # Prepare training data
            X = np.array(self.health_data_buffer)
            
            # Show progress dialog
            progress = QProgressDialog("Training Multi-Agent System...", None, 0, 100, self)
            progress.setWindowTitle("Training")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()
            
            try:
                # Initialize system
                progress.setLabelText("Initializing agents...")
                progress.setValue(10)
                QApplication.processEvents()
                
                input_dim = X.shape[1]
                sequence_length = min(10, len(X) // 10)  # Adaptive sequence length
                self.multi_agent_system = MultiAgentFusionSystem(input_dim, sequence_length)
                
                # Train system
                progress.setLabelText("Training Agent 1 (Autoencoder+LSTM)...")
                progress.setValue(30)
                QApplication.processEvents()
                
                self.multi_agent_system.train(X, epochs=30, batch_size=16)
                
                progress.setValue(90)
                QApplication.processEvents()
                
                # Update status labels
                if self.multi_agent_system.agent1:
                    self.agent1_detail_label.setText("[Trained]\nReconstruction Error Analysis")
                    self.agent1_detail_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
                
                if self.multi_agent_system.agent2:
                    self.agent2_detail_label.setText("[Trained]\nSpatial-Temporal Patterns")
                    self.agent2_detail_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
                
                self.trend_detail_label.setText("[Trained]\nTime Series Forecasting")
                self.trend_detail_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10px;")
                
                progress.setValue(100)
                
                logger.info("="*60)
                logger.info("MULTI-AGENT SYSTEM TRAINING COMPLETE")
                logger.info("="*60)
                
                QMessageBox.information(
                    self, "Training Complete",
                    "Multi-agent system trained successfully!\n\n"
                    "All agents are now operational:\n"
                    "- Agent 1: Autoencoder+LSTM [OK]\n"
                    "- Agent 2: CNN [OK]\n"
                    "- Trend Model: ARIMA [OK]\n\n"
                    "Real-time anomaly detection is now active."
                )
                
            finally:
                progress.close()
                
        except Exception as e:
            logger.error(f"Error training multi-agent system: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(
                self, "Training Error",
                f"Failed to train multi-agent system:\n\n{str(e)}\n\n"
                f"Check if TensorFlow is installed for deep learning models."
            )
    
    def update_agent1_plot(self, agent1_results, data):
        """Update Agent 1 reconstruction error plot"""
        try:
            self.agent1_canvas.axes.clear()
            
            if 'reconstruction_error' in agent1_results:
                errors = agent1_results['reconstruction_error']
                time_points = np.arange(len(errors))
                
                # Plot reconstruction error
                self.agent1_canvas.axes.plot(time_points, errors, color='#2196F3', linewidth=2, label='Reconstruction Error')
                
                # Plot threshold line
                if self.multi_agent_system and self.multi_agent_system.agent1:
                    threshold = self.multi_agent_system.agent1.threshold
                    self.agent1_canvas.axes.axhline(y=threshold, color='#f44336', linestyle='--', 
                                                    linewidth=2, label=f'Threshold ({threshold:.4f})', alpha=0.7)
                
                # Highlight anomalies
                if 'is_anomaly' in agent1_results:
                    anomalies = agent1_results['is_anomaly']
                    anomaly_indices = np.where(anomalies)[0]
                    if len(anomaly_indices) > 0:
                        self.agent1_canvas.axes.scatter(anomaly_indices, errors[anomaly_indices], 
                                                        color='red', s=100, marker='X', 
                                                        label='Anomalies', zorder=5)
                
                self.agent1_canvas.axes.set_xlabel('Time')
                self.agent1_canvas.axes.set_ylabel('Error')
                self.agent1_canvas.axes.set_title('Agent 1: Autoencoder Reconstruction Error')
                self.agent1_canvas.axes.legend(loc='upper right', fontsize=8)
                self.agent1_canvas.axes.grid(True, alpha=0.3)
                
                # Update info label
                latest_error = errors[-1]
                status = "ANOMALY" if agent1_results['is_anomaly'][-1] else "Normal"
                self.agent1_info_label.setText(f"Latest Error: {latest_error:.6f} | Status: {status}")
                
            self.agent1_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating agent1 plot: {str(e)}")
    
    def update_agent2_plot(self, agent2_results, data):
        """Update Agent 2 CNN anomaly score plot"""
        try:
            self.agent2_canvas.axes.clear()
            
            if 'anomaly_score' in agent2_results:
                scores = agent2_results['anomaly_score']
                time_points = np.arange(len(scores))
                
                # Plot anomaly scores
                self.agent2_canvas.axes.plot(time_points, scores, color='#FF9800', linewidth=2, label='Anomaly Score')
                
                # Plot threshold line
                if self.multi_agent_system and self.multi_agent_system.agent2:
                    threshold = self.multi_agent_system.agent2.threshold
                    self.agent2_canvas.axes.axhline(y=threshold, color='#f44336', linestyle='--', 
                                                    linewidth=2, label=f'Threshold ({threshold:.4f})', alpha=0.7)
                
                # Highlight anomalies
                if 'is_anomaly' in agent2_results:
                    anomalies = agent2_results['is_anomaly']
                    anomaly_indices = np.where(anomalies)[0]
                    if len(anomaly_indices) > 0:
                        self.agent2_canvas.axes.scatter(anomaly_indices, scores[anomaly_indices], 
                                                        color='red', s=100, marker='X', 
                                                        label='Anomalies', zorder=5)
                
                self.agent2_canvas.axes.set_xlabel('Time')
                self.agent2_canvas.axes.set_ylabel('Score')
                self.agent2_canvas.axes.set_title('Agent 2: CNN Anomaly Detection')
                self.agent2_canvas.axes.legend(loc='upper right', fontsize=8)
                self.agent2_canvas.axes.grid(True, alpha=0.3)
                
                # Update info label
                latest_score = scores[-1]
                status = "ANOMALY" if agent2_results['is_anomaly'][-1] else "Normal"
                self.agent2_info_label.setText(f"Latest Score: {latest_score:.6f} | Status: {status}")
                
            self.agent2_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating agent2 plot: {str(e)}")
    
    def update_trend_plot(self, trend_results, data):
        """Update trend analysis and forecast plot"""
        try:
            self.trend_canvas.axes.clear()
            
            # Calculate historical mean
            historical_mean = np.mean(data, axis=1) if data.ndim > 1 else data
            time_points = np.arange(len(historical_mean))
            
            # Plot historical data
            self.trend_canvas.axes.plot(time_points, historical_mean, color='#2196F3', 
                                        linewidth=2, label='Historical', marker='o', markersize=3)
            
            # Plot forecast
            if 'forecast' in trend_results:
                forecast = trend_results['forecast']
                forecast_points = np.arange(len(historical_mean), len(historical_mean) + len(forecast))
                self.trend_canvas.axes.plot(forecast_points, forecast, color='#4CAF50', 
                                            linestyle='--', linewidth=2, label='Forecast', marker='s', markersize=4)
                
                # Confidence interval (simplified)
                lower_bound = forecast * 0.9
                upper_bound = forecast * 1.1
                self.trend_canvas.axes.fill_between(forecast_points, lower_bound, upper_bound, 
                                                    alpha=0.2, color='#4CAF50', label='Confidence')
            
            # Mark trend anomaly
            if trend_results.get('is_anomaly', False):
                current_value = trend_results.get('current_value', historical_mean[-1])
                self.trend_canvas.axes.scatter(len(historical_mean)-1, current_value, 
                                               color='red', s=200, marker='X', 
                                               label='Trend Anomaly', zorder=5)
            
            self.trend_canvas.axes.set_xlabel('Time')
            self.trend_canvas.axes.set_ylabel('System Health Metric')
            self.trend_canvas.axes.set_title('Trend Analysis & Future Forecast')
            self.trend_canvas.axes.legend(loc='upper left', fontsize=8)
            self.trend_canvas.axes.grid(True, alpha=0.3)
            
            # Update info label
            if 'forecast' in trend_results:
                next_pred = trend_results['forecast'][0]
                deviation = trend_results.get('deviation', 0)
                self.trend_info_label.setText(
                    f"Next prediction: {next_pred:.4f} | Deviation: {deviation*100:.2f}% | "
                    f"Status: {'ANOMALY' if trend_results.get('is_anomaly') else 'Normal'}"
                )
            
            self.trend_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating trend plot: {str(e)}")
    
    def update_combined_plot(self, results, data):
        """Update combined reconstruction error plot (final fusion result)"""
        try:
            self.combined_canvas.axes.clear()
            
            # Get the fused reconstruction error
            final_score = results.get('fused_score', 0)
            weights = results.get('weights_used', {})
            is_anomaly = results.get('is_anomaly', False)
            
            # Maintain history of reconstruction errors
            if not hasattr(self, 'reconstruction_error_history'):
                self.reconstruction_error_history = []
            
            self.reconstruction_error_history.append(final_score)
            if len(self.reconstruction_error_history) > 100:
                self.reconstruction_error_history.pop(0)
            
            time_points = np.arange(len(self.reconstruction_error_history))
            
            # Plot reconstruction errors
            self.combined_canvas.axes.plot(time_points, self.reconstruction_error_history, 
                                           color='#2196F3', linewidth=2.5, label='Reconstruction Error', marker='o', markersize=3)
            
            # Threshold line (95th percentile or 0.6 as default)
            threshold = 0.6
            self.combined_canvas.axes.axhline(y=threshold, color='#f44336', linestyle='--', 
                                             linewidth=2, label=f'Threshold ({threshold})', alpha=0.8)
            
            # Highlight anomalies with red markers
            anomaly_indices = [i for i, score in enumerate(self.reconstruction_error_history) if score > threshold]
            if anomaly_indices:
                anomaly_scores = [self.reconstruction_error_history[i] for i in anomaly_indices]
                self.combined_canvas.axes.scatter(anomaly_indices, anomaly_scores, 
                                                 color='#FF1744', s=150, marker='X', 
                                                 label='Detected Anomalies', zorder=5, edgecolors='black', linewidths=1)
            
            # Add shaded region for normal zone
            self.combined_canvas.axes.axhspan(0, threshold, alpha=0.1, color='green', label='Normal Zone')
            
            # Labels and styling
            self.combined_canvas.axes.set_xlabel('Time Step', fontsize=11, fontweight='bold')
            self.combined_canvas.axes.set_ylabel('Reconstruction Error', fontsize=11, fontweight='bold')
            self.combined_canvas.axes.set_title(
                'Final Reconstruction Error (Multi-Agent Fusion: Autoencoder+LSTM, CNN, ARIMA)', 
                fontsize=12, fontweight='bold', pad=10
            )
            self.combined_canvas.axes.legend(loc='upper right', fontsize=9, framealpha=0.9)
            self.combined_canvas.axes.grid(True, alpha=0.3, linestyle='--')
            
            # Set y-axis limits for better visualization
            if len(self.reconstruction_error_history) > 0:
                max_error = max(self.reconstruction_error_history)
                self.combined_canvas.axes.set_ylim(0, max(max_error * 1.2, threshold * 1.5))
            
            # Update weights label display
            if weights:
                self.weights_label.setText(
                    f"Fusion Weights: Agent1={weights.get('agent1', 0)*100:.0f}%, "
                    f"Agent2={weights.get('agent2', 0)*100:.0f}%, "
                    f"Trend={weights.get('trend', 0)*100:.0f}%"
                )
            
            self.combined_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating combined reconstruction error plot: {str(e)}")
            logger.error(traceback.format_exc())
    
    def update_realtime_plot(self, buffer_data, is_anomaly, anomaly_score):
        """Update real-time monitoring plot using Analysis tab model predictions"""
        try:
            self.combined_canvas.axes.clear()
            
            # Maintain history of anomaly scores
            if not hasattr(self, 'anomaly_score_history'):
                self.anomaly_score_history = []
            
            self.anomaly_score_history.append(anomaly_score)
            if len(self.anomaly_score_history) > 100:
                self.anomaly_score_history.pop(0)
            
            time_points = np.arange(len(self.anomaly_score_history))
            
            # Plot anomaly scores
            self.combined_canvas.axes.plot(time_points, self.anomaly_score_history, 
                                           color='#2196F3', linewidth=2.5, label='Anomaly Score', marker='o', markersize=3)
            
            # Threshold line
            threshold = getattr(self.model, 'threshold', 0.5)
            self.combined_canvas.axes.axhline(y=threshold, color='#f44336', linestyle='--', 
                                             linewidth=2, label=f'Threshold ({threshold:.3f})', alpha=0.8)
            
            # Highlight current anomaly if detected
            if is_anomaly:
                self.combined_canvas.axes.scatter([len(self.anomaly_score_history)-1], [anomaly_score], 
                                                 color='#FF1744', s=200, marker='X', 
                                                 label='ANOMALY!', zorder=5, edgecolors='black', linewidths=2)
            
            # Add shaded region for normal zone
            self.combined_canvas.axes.axhspan(0, threshold, alpha=0.1, color='green', label='Normal Zone')
            
            # Labels and styling
            self.combined_canvas.axes.set_xlabel('Time Step', fontsize=11, fontweight='bold')
            self.combined_canvas.axes.set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
            self.combined_canvas.axes.set_title(
                f'Real-Time Telemetry Monitoring - Model: {self.model.model_type.upper()}', 
                fontsize=12, fontweight='bold', pad=10
            )
            self.combined_canvas.axes.legend(loc='upper right', fontsize=9, framealpha=0.9)
            self.combined_canvas.axes.grid(True, alpha=0.3, linestyle='--')
            
            # Set y-axis limits for better visualization
            if len(self.anomaly_score_history) > 0:
                max_score = max(self.anomaly_score_history)
                self.combined_canvas.axes.set_ylim(0, max(max_score * 1.2, threshold * 2))
            
            # Update weights label with model info
            if hasattr(self.model, 'metrics') and self.model.metrics:
                metrics = self.model.metrics
                self.weights_label.setText(
                    f"Model: {self.model.model_type} | "
                    f"Training time: {metrics.get('training_time', 'N/A')} | "
                    f"Features: {metrics.get('features_used', 'N/A')}"
                )
            
            self.combined_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating real-time plot: {str(e)}")
            logger.error(traceback.format_exc())
    
    def configure_fusion_weights(self):
        """Open dialog to configure fusion weights"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle("Configure Fusion Weights")
            dialog.setModal(True)
            layout = QVBoxLayout(dialog)
            
            # Instructions
            label = QLabel("Adjust the weights for each agent in the fusion system:")
            layout.addWidget(label)
            
            # Current weights
            current_weights = self.multi_agent_system.weights if self.multi_agent_system else {'agent1': 0.4, 'agent2': 0.4, 'trend': 0.2}
            
            # Agent 1 weight
            agent1_layout = QHBoxLayout()
            agent1_layout.addWidget(QLabel("Agent 1 (Autoencoder+LSTM):"))
            agent1_spin = QDoubleSpinBox()
            agent1_spin.setRange(0.0, 1.0)
            agent1_spin.setValue(current_weights['agent1'])
            agent1_spin.setSingleStep(0.05)
            agent1_layout.addWidget(agent1_spin)
            layout.addLayout(agent1_layout)
            
            # Agent 2 weight
            agent2_layout = QHBoxLayout()
            agent2_layout.addWidget(QLabel("Agent 2 (CNN):"))
            agent2_spin = QDoubleSpinBox()
            agent2_spin.setRange(0.0, 1.0)
            agent2_spin.setValue(current_weights['agent2'])
            agent2_spin.setSingleStep(0.05)
            agent2_layout.addWidget(agent2_spin)
            layout.addLayout(agent2_layout)
            
            # Trend weight
            trend_layout = QHBoxLayout()
            trend_layout.addWidget(QLabel("Trend Model (ARIMA):"))
            trend_spin = QDoubleSpinBox()
            trend_spin.setRange(0.0, 1.0)
            trend_spin.setValue(current_weights['trend'])
            trend_spin.setSingleStep(0.05)
            trend_layout.addWidget(trend_spin)
            layout.addLayout(trend_layout)
            
            # Note
            note_label = QLabel("Note: Weights will be automatically normalized to sum to 1.0")
            note_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
            layout.addWidget(note_label)
            
            # Buttons
            button_layout = QHBoxLayout()
            save_btn = QPushButton("Save")
            save_btn.clicked.connect(dialog.accept)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(dialog.reject)
            button_layout.addWidget(save_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            if dialog.exec_() == QDialog.Accepted and self.multi_agent_system:
                self.multi_agent_system.update_weights(
                    agent1_spin.value(),
                    agent2_spin.value(),
                    trend_spin.value()
                )
                QMessageBox.information(self, "Success", "Fusion weights updated successfully!")
                
        except Exception as e:
            logger.error(f"Error configuring weights: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to configure weights:\n{str(e)}")
    
    def on_sync_toggled(self, checked):
        """Handle sync with analysis tab toggle"""
        if checked:
            # Try to get folder from analysis tab
            if hasattr(self, 'folder_path') and self.folder_path:
                self.monitoring_folder_label.setText(self.folder_path)
                self.monitoring_folder_label.setStyleSheet("color: #4CAF50;")
            else:
                self.monitoring_folder_label.setText("Set folder in Analysis tab first")
                self.monitoring_folder_label.setStyleSheet("color: #FF9800;")
        else:
            self.monitoring_folder_label.setText("Manual monitoring (not implemented)")
            self.monitoring_folder_label.setStyleSheet("color: #666;")
    
    def start_health_monitoring_new(self):
        """Start the enhanced health monitoring with failure prediction"""
        if not self.health_monitor.supported_models:
            QMessageBox.warning(self, "Warning", "No models available for monitoring. Train a model in Analysis tab first.")
            return
        
        if not self.current_monitoring_model or self.current_monitoring_model not in self.health_monitor.supported_models:
            QMessageBox.warning(self, "Warning", "Please select a valid model for monitoring.")
            return
        
        try:
            # Start monitoring
            self.failure_prediction_active = True
            interval_ms = self.monitoring_interval_spin.value() * 1000
            self.failure_prediction_timer.start(interval_ms)
            
            # Update UI
            self.start_health_monitoring_btn.setEnabled(False)
            self.stop_health_monitoring_btn.setEnabled(True)
            self.health_status_label.setText("Status: Failure Prediction Active")
            self.health_status_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")
            
            QMessageBox.information(self, "Success", f"Failure prediction monitoring started for model: {self.current_monitoring_model}")
            logger.info(f"Started failure prediction monitoring for model: {self.current_monitoring_model}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start monitoring: {str(e)}")
            logger.error(f"Error starting health monitoring: {str(e)}")
    
    def stop_health_monitoring_new(self):
        """Stop the enhanced health monitoring"""
        self.failure_prediction_active = False
        self.failure_prediction_timer.stop()
        
        # Update UI
        self.start_health_monitoring_btn.setEnabled(True)
        self.stop_health_monitoring_btn.setEnabled(False)
        self.health_status_label.setText("Status: Monitoring Stopped")
        self.health_status_label.setStyleSheet("font-size: 14px; color: #666;")
        
        logger.info("Stopped failure prediction monitoring")
    
    def test_with_current_data(self):
        """Test failure prediction with current analysis data"""
        if not self.current_monitoring_model:
            QMessageBox.warning(self, "Warning", "Please select a model first.")
            return
        
        if not hasattr(self.data_processor, 'preprocessed_data') or self.data_processor.preprocessed_data is None:
            QMessageBox.warning(self, "Warning", "No data available. Load data in Analysis tab first.")
            return
        
        try:
            # Get current data
            numeric_data = self.data_processor.preprocessed_data.select_dtypes(include=['number'])
            if numeric_data.empty:
                QMessageBox.warning(self, "Warning", "No numeric data found.")
                return
            
            # Take last row as current data
            current_data = numeric_data.iloc[-1:].values
            
            # Run monitoring
            result = self.health_monitor.monitor_with_analysis_model(
                self.current_monitoring_model, 
                current_data
            )
            
            # Update UI with results
            self.update_dashboard_with_result(result)
            self.add_alert_to_table(result)
            
            # Show result to user
            status_msg = f"Test completed:\nError: {result['reconstruction_error']:.6f}\nThreshold: {result['threshold']:.6f}\nStatus: {result['status']}"
            if result['failure_prediction']:
                pred = result['failure_prediction']
                status_msg += f"\nFailure Prediction: {pred['status']}\nTime to failure: {pred['time_to_failure']}"
            
            QMessageBox.information(self, "Test Results", status_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Test failed: {str(e)}")
            logger.error(f"Error during test: {str(e)}")
    
    def update_failure_prediction_monitoring(self):
        """Update failure prediction monitoring - check for new data"""
        if not self.failure_prediction_active or not self.current_monitoring_model:
            return
        
        try:
            # Check if sync with analysis tab is enabled
            if not self.sync_with_analysis_check.isChecked():
                return
            
            # Check if there's new data to process
            if hasattr(self, 'folder_path') and self.folder_path and hasattr(self, 'file_watcher'):
                # This would be enhanced to actually check for new files
                # For now, simulate with current data if available
                if hasattr(self.data_processor, 'preprocessed_data') and self.data_processor.preprocessed_data is not None:
                    self._process_monitoring_data()
            
        except Exception as e:
            logger.error(f"Error in failure prediction monitoring: {str(e)}")
    
    def _process_monitoring_data(self):
        """Process current data for failure prediction monitoring"""
        try:
            numeric_data = self.data_processor.preprocessed_data.select_dtypes(include=['number'])
            if numeric_data.empty:
                return
            
            # Take last few rows to simulate new incoming data
            n_samples = min(5, len(numeric_data))
            recent_data = numeric_data.iloc[-n_samples:].values
            
            for i, data_row in enumerate(recent_data):
                result = self.health_monitor.monitor_with_analysis_model(
                    self.current_monitoring_model,
                    data_row.reshape(1, -1)
                )
                
                # Update dashboard with latest result
                if i == len(recent_data) - 1:  # Only update dashboard with the latest
                    self.update_dashboard_with_result(result)
                
                # Add alert if threshold exceeded or failure predicted
                if result['exceeds_threshold'] or result['failure_prediction']:
                    self.add_alert_to_table(result)
                    self._send_alert_notification(result)
            
        except Exception as e:
            logger.error(f"Error processing monitoring data: {str(e)}")
    
    def update_dashboard_with_result(self, result):
        """Update the dashboard with monitoring result"""
        try:
            # Update error and threshold display
            self.current_error_label.setText(f"Current Error: {result['reconstruction_error']:.6f}")
            self.threshold_status_label.setText(f"Threshold: {result['threshold']:.6f}")
            
            # Update status with color coding
            status = result['status']
            if status == 'normal':
                self.prediction_status_label.setText("Status: Normal")
                self.prediction_status_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")
            elif status == 'warning':
                self.prediction_status_label.setText("Status: Warning")
                self.prediction_status_label.setStyleSheet("font-size: 14px; color: #FF9800; font-weight: bold;")
            elif status == 'critical':
                self.prediction_status_label.setText("Status: Critical")
                self.prediction_status_label.setStyleSheet("font-size: 14px; color: #f44336; font-weight: bold;")
            else:
                self.prediction_status_label.setText("Status: Failure Detected")
                self.prediction_status_label.setStyleSheet("font-size: 14px; color: #B71C1C; font-weight: bold;")
            
            # Update failure prediction
            if result['failure_prediction']:
                pred = result['failure_prediction']
                self.time_to_failure_label.setText(f"Time to Failure: {pred['time_to_failure']}")
                self.time_to_failure_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #f44336;")
            else:
                self.time_to_failure_label.setText("Time to Failure: N/A")
                self.time_to_failure_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2E4BC6;")
            
            # Update last update time
            self.last_update_label.setText(f"Last Update: {result['timestamp'].strftime('%H:%M:%S')}")
            
            # Update models status table
            self.update_models_status_display()
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
    
    def add_alert_to_table(self, result):
        """Add alert to the alerts table"""
        try:
            row = self.health_alerts_table.rowCount()
            self.health_alerts_table.insertRow(row)
            
            # Time
            time_str = result['timestamp'].strftime('%H:%M:%S')
            self.health_alerts_table.setItem(row, 0, QTableWidgetItem(time_str))
            
            # Model
            self.health_alerts_table.setItem(row, 1, QTableWidgetItem(result['model_name']))
            
            # Error
            error_item = QTableWidgetItem(f"{result['reconstruction_error']:.6f}")
            if result['exceeds_threshold']:
                error_item.setBackground(QColor("#f44336"))
            self.health_alerts_table.setItem(row, 2, error_item)
            
            # Threshold
            self.health_alerts_table.setItem(row, 3, QTableWidgetItem(f"{result['threshold']:.6f}"))
            
            # Status
            status_item = QTableWidgetItem(result['status'].title())
            if result['status'] == 'normal':
                status_item.setBackground(QColor("#4CAF50"))
            elif result['status'] == 'warning':
                status_item.setBackground(QColor("#FF9800"))
            elif result['status'] == 'critical':
                status_item.setBackground(QColor("#f44336"))
            else:
                status_item.setBackground(QColor("#B71C1C"))
            self.health_alerts_table.setItem(row, 4, status_item)
            
            # Prediction
            if result['failure_prediction']:
                pred = result['failure_prediction']
                pred_text = f"{pred['status']} ({pred['time_to_failure']})"
                pred_item = QTableWidgetItem(pred_text)
                pred_item.setBackground(QColor("#f44336"))
            else:
                pred_item = QTableWidgetItem("Normal")
                pred_item.setBackground(QColor("#4CAF50"))
            self.health_alerts_table.setItem(row, 5, pred_item)
            
            # Scroll to bottom
            self.health_alerts_table.scrollToBottom()
            
            # Keep only last 100 alerts to avoid performance issues
            if self.health_alerts_table.rowCount() > 100:
                self.health_alerts_table.removeRow(0)
            
        except Exception as e:
            logger.error(f"Error adding alert to table: {str(e)}")
    
    def update_models_status_display(self):
        """Update the models status table"""
        try:
            models = list(self.health_monitor.supported_models.keys())
            self.models_status_table.setRowCount(len(models))
            
            for i, model_name in enumerate(models):
                # Model Name
                self.models_status_table.setItem(i, 0, QTableWidgetItem(model_name))
                
                # Model Type
                model_obj = self.health_monitor.supported_models[model_name]
                model_type = type(model_obj).__name__
                self.models_status_table.setItem(i, 1, QTableWidgetItem(model_type))
                
                # Threshold
                threshold = self.health_monitor.model_thresholds.get(model_name, 0.0)
                self.models_status_table.setItem(i, 2, QTableWidgetItem(f"{threshold:.6f}"))
                
                # Status and Health Score
                if model_name in self.health_monitor.reconstruction_error_history:
                    history = self.health_monitor.reconstruction_error_history[model_name]
                    if history['errors']:
                        last_error = history['errors'][-1]
                        last_status = history['status'][-1]
                        
                        # Calculate health score
                        health_score = max(0.0, 1.0 - (last_error / (threshold * 3)))
                        
                        status_item = QTableWidgetItem(last_status.title())
                        if last_status == 'normal':
                            status_item.setBackground(QColor("#4CAF50"))
                        elif last_status == 'warning':
                            status_item.setBackground(QColor("#FF9800"))
                        elif last_status == 'critical':
                            status_item.setBackground(QColor("#f44336"))
                        else:
                            status_item.setBackground(QColor("#B71C1C"))
                        
                        self.models_status_table.setItem(i, 3, status_item)
                        self.models_status_table.setItem(i, 4, QTableWidgetItem(f"{health_score:.2f}"))
                    else:
                        self.models_status_table.setItem(i, 3, QTableWidgetItem("No Data"))
                        self.models_status_table.setItem(i, 4, QTableWidgetItem("N/A"))
                else:
                    self.models_status_table.setItem(i, 3, QTableWidgetItem("Not Monitored"))
                    self.models_status_table.setItem(i, 4, QTableWidgetItem("N/A"))
            
        except Exception as e:
            logger.error(f"Error updating models status display: {str(e)}")
    
    def _build_alert_notification_message(self, severity, message, payload=None, incident_id=None, escalated=False):
        """Create a normalized alert message for all channels."""
        payload = payload or {}
        model_name = payload.get("model_name", "Unknown")
        reconstruction_error = payload.get("reconstruction_error")
        threshold = payload.get("threshold")
        health_score = payload.get("health_score")
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        lines = [
            f"Incident ID: {incident_id}" if incident_id is not None else "Incident ID: N/A",
            f"Severity: {str(severity).upper()}",
            f"Status: {'ESCALATED' if escalated else 'TRIGGERED'}",
            f"Time: {timestamp}",
            f"Model: {model_name}",
            f"Message: {message}"
        ]
        if reconstruction_error is not None:
            lines.append(f"Reconstruction Error: {float(reconstruction_error):.6f}")
        if threshold is not None:
            lines.append(f"Threshold: {float(threshold):.6f}")
        if health_score is not None:
            lines.append(f"Health Score: {float(health_score):.4f}")
        
        if payload.get("failure_prediction"):
            pred = payload["failure_prediction"]
            lines.append(
                "Failure Prediction: "
                f"{pred.get('status')} | TTF={pred.get('time_to_failure')} | Confidence={pred.get('confidence')}"
            )
        
        return "\n".join(lines)
    
    def _dispatch_email_route(self, incident_id, severity, message, payload, channel_config, escalated=False):
        """Route alert via configured email channel."""
        payload = (payload or {}).copy()
        email_config = self._resolve_email_config()
        config_error = self._get_email_config_error(email_config)
        if config_error:
            raise RuntimeError(config_error)
        
        subject_prefix = "[ESCALATED]" if escalated else "[ALERT]"
        subject = f"{subject_prefix} {str(severity).upper()} incident #{incident_id}"
        body = self._build_alert_notification_message(
            severity=severity,
            message=message,
            payload=payload,
            incident_id=incident_id,
            escalated=escalated
        )
        
        sent = 0
        for username, user_data in self.user_manager.users.items():
            to_email = str(user_data.get("email", "")).strip()
            if not _is_deliverable_alert_email(to_email):
                if to_email:
                    logger.info(f"Skipping placeholder alert email for user '{username}': {to_email}")
                continue
            self._send_email_via_config(to_email, subject, body, email_config)
            sent += 1
        
        if sent == 0:
            raise RuntimeError("No user emails configured for alert email route.")
    
    def _http_post_json(self, url, payload, headers=None, timeout=10):
        safe_payload = to_json_compatible(payload)
        data = json.dumps(safe_payload, ensure_ascii=False).encode("utf-8")
        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)
        req = urllib_request.Request(url=url, data=data, headers=req_headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                status_code = int(getattr(resp, "status", 200))
        except urllib_error.HTTPError as e:
            raise RuntimeError(f"HTTPError {e.code} for {url}") from e
        except urllib_error.URLError as e:
            raise RuntimeError(f"URLError for {url}: {e.reason}") from e
        if status_code >= 400:
            raise RuntimeError(f"HTTP {status_code} from {url}")
    
    def _dispatch_slack_route(self, incident_id, severity, message, payload, channel_config, escalated=False):
        webhook_url = str(channel_config.get("webhook_url", "")).strip()
        if not webhook_url:
            raise RuntimeError("Slack webhook_url is not configured.")
        text = self._build_alert_notification_message(
            severity=severity,
            message=message,
            payload=payload,
            incident_id=incident_id,
            escalated=escalated
        )
        self._http_post_json(webhook_url, {"text": text})
    
    def _dispatch_webhook_route(self, incident_id, severity, message, payload, channel_config, escalated=False):
        url = str(channel_config.get("url", "")).strip()
        if not url:
            raise RuntimeError("Webhook url is not configured.")
        envelope = {
            "incident_id": int(incident_id),
            "severity": str(severity).lower(),
            "message": message,
            "escalated": bool(escalated),
            "payload": payload or {},
            "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"
        }
        self._http_post_json(url, envelope)
    
    def _dispatch_pagerduty_route(self, incident_id, severity, message, payload, channel_config, escalated=False):
        events_url = str(channel_config.get("events_api_url", "https://events.pagerduty.com/v2/enqueue")).strip()
        routing_key = str(channel_config.get("routing_key", "")).strip()
        if not routing_key:
            raise RuntimeError("PagerDuty routing_key is not configured.")
        
        payload_text = self._build_alert_notification_message(
            severity=severity,
            message=message,
            payload=payload,
            incident_id=incident_id,
            escalated=escalated
        )
        pd_event = {
            "routing_key": routing_key,
            "event_action": "trigger",
            "dedup_key": f"telemetry-incident-{incident_id}",
            "payload": {
                "summary": f"{'[ESCALATED] ' if escalated else ''}{str(severity).upper()} incident #{incident_id}: {message}",
                "severity": "critical" if str(severity).lower() == "critical" else "warning",
                "source": "satellite-telemetry-monitor",
                "custom_details": {
                    "incident_id": incident_id,
                    "details": payload_text
                }
            }
        }
        self._http_post_json(events_url, pd_event)
    
    def _has_any_active_monitoring(self):
        """Return True when any monitoring workload is actively running."""
        try:
            # Legacy/analysis timers
            if hasattr(self, "auto_load_timer") and self.auto_load_timer is not None and self.auto_load_timer.isActive():
                return True
            if hasattr(self, "auto_process_timer") and self.auto_process_timer is not None and self.auto_process_timer.isActive():
                return True

            # Legacy health monitoring flow
            if bool(getattr(self, "health_monitoring_active", False)):
                return True

            # Custom tabs
            if hasattr(self, "custom_tabs"):
                for _tab_id, tab in (self.custom_tabs or {}).items():
                    if bool(getattr(tab, "monitoring_active", False)):
                        return True
        except Exception:
            return False
        return False

    def _run_alert_escalation_check(self):
        """Periodic escalation worker for unacknowledged incidents."""
        try:
            if not hasattr(self, "alert_policy_manager") or self.alert_policy_manager is None:
                return
            if not self._has_any_active_monitoring():
                # Do not escalate/send while monitoring is idle.
                return
            escalated_ids = self.alert_policy_manager.run_escalation_cycle()
            if escalated_ids:
                logger.warning(f"Escalated {len(escalated_ids)} incident(s): {escalated_ids}")
        except Exception as e:
            logger.error(f"Alert escalation cycle failed: {str(e)}")
    
    def acknowledge_alert_incident(self, incident_id, acknowledged_by=None):
        """Public helper to acknowledge an incident from UI workflows or scripts."""
        if not hasattr(self, "service_layer") or self.service_layer is None:
            return False
        context = ServiceLayerContext(
            username=acknowledged_by or getattr(self, "current_username", "system"),
            auth_provider=getattr(self, "current_auth_provider", "local"),
            session_id=getattr(self, "current_session_id", None)
        )
        return self.service_layer.acknowledge_alert(incident_id, context)

    def _send_alert_notification(self, result):
        """Send local alert notifications (logging + audit file)."""
        try:
            # Log alert
            model_name = result.get('model_name', 'Unknown')
            reconstruction_error = result.get('reconstruction_error', 0.0)
            threshold = result.get('threshold', 0.0)
            status = result.get('status', 'info')
            alert_msg = f"HEALTH ALERT - Model: {model_name}, Error: {reconstruction_error:.6f}, Threshold: {threshold:.6f}, Status: {status}"
            if result.get('message'):
                alert_msg += f", Message: {result['message']}"
            if result.get('failure_prediction'):
                pred = result['failure_prediction']
                alert_msg += f", Failure Prediction: {pred.get('status')}, Time: {pred.get('time_to_failure')}, Confidence: {pred.get('confidence')}"
            
            logger.warning(alert_msg)
            
            # Write to health monitoring log file
            self._write_health_log(result)
                
        except Exception as e:
            logger.error(f"Error sending alert notification: {str(e)}")

    def _trigger_health_alert(self, severity, message, payload=None):
        """Centralized helper to dispatch health alerts with cooldown"""
        try:
            if not hasattr(self, 'health_alert_cooldown'):
                self.health_alert_cooldown = 60  # seconds
            if not hasattr(self, '_last_health_alert_time'):
                self._last_health_alert_time = 0
            
            now = time.time()
            if severity != 'critical' and (now - self._last_health_alert_time) < self.health_alert_cooldown:
                return
            
            self._last_health_alert_time = now
            
            payload = (payload or {}).copy()
            payload.setdefault('model_name', getattr(self.model, 'model_type', 'Unknown'))
            payload.setdefault('reconstruction_error', 0.0)
            payload.setdefault('threshold', 0.0)
            payload.setdefault('exceeds_threshold', payload.get('reconstruction_error', 0.0) >= payload.get('threshold', 0.0))
            payload.setdefault('health_score', 0.0)
            payload.setdefault('failure_prediction', None)
            payload['status'] = severity
            payload['message'] = message
            payload['timestamp'] = datetime.datetime.now()
            
            logger.warning(f"System Health Alert [{severity.upper()}]: {message}")
            self._send_alert_notification(payload)
            
            # Policy pipeline: dedup + cooldown + routing + persistence
            if hasattr(self, "service_layer") and self.service_layer is not None:
                policy_payload = to_json_compatible(payload)
                policy_result = self.service_layer.process_alert(severity, message, policy_payload)
                payload["incident_id"] = policy_result.get("incident_id")
                if OBSERVABILITY:
                    routed_status = "routed" if policy_result.get("routed") else "suppressed"
                    OBSERVABILITY.inc_counter(
                        "alerts_policy_events_total",
                        labels={"severity": str(severity).lower(), "status": routed_status}
                    )
            
        except Exception as e:
            logger.error(f"Failed to trigger health alert: {str(e)}")
    
    def _write_health_log(self, result):
        """Write health monitoring result to log file"""
        try:
            log_entry = {
                'timestamp': result['timestamp'].isoformat(),
                'model_name': str(result['model_name']),
                'reconstruction_error': float(result['reconstruction_error']),
                'threshold': float(result['threshold']),
                'exceeds_threshold': bool(result['exceeds_threshold']),
                'health_score': float(result['health_score']),
                'status': str(result['status']),
                'failure_prediction': None
            }
            
            if result.get('failure_prediction'):
                fp = result['failure_prediction']
                log_entry['failure_prediction'] = {
                    'status': str(fp.get('status')),
                    'time_to_failure': fp.get('time_to_failure'),
                    'confidence': float(fp.get('confidence')) if fp.get('confidence') is not None else None
                }
            
            # Append to health monitoring log
            health_log_file = os.path.join("logs", "health_monitoring.log")
            os.makedirs("logs", exist_ok=True)
            
            with open(health_log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Error writing health log: {str(e)}")
    
    def _send_email_alert(self, result):
        """Send email alert for health monitoring (placeholder)"""
        # This would integrate with the existing email system
        # For now, just log that an email would be sent
        logger.info(f"Email alert would be sent for model {result['model_name']} with status {result['status']}")
    
    def resizeEvent(self, event):
        """Override resize event to prevent figure size issues"""
        try:
            # Ensure minimum window size
            if event is not None:
                size = event.size()
                width = max(size.width(), 800)
                height = max(size.height(), 600)
                
                # If the size is too small, set it to minimum
                if size.width() < 800 or size.height() < 600:
                    self.resize(width, height)
                    return
            
            # Call parent resize event
            super().resizeEvent(event)
            
        except Exception as e:
            # Log the error but don't crash the application
            logger.warning(f"Error in main window resize event: {str(e)}")

    def setup_admin_tab(self):
        # Main layout with scroll area for responsive design
        main_layout = QVBoxLayout(self.admin_tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Container widget for scroll area
        container_widget = QWidget()
        layout = QVBoxLayout(container_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # User management section
        user_group = QGroupBox("User Management")
        user_layout = QVBoxLayout()
        
        # User list
        self.user_table = QTableWidget(0, 4)
        self.user_table.setHorizontalHeaderLabels(["Username", "Role", "Email", "Actions"])
        self.user_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.user_table.setMinimumHeight(400)
        self.user_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        user_layout.addWidget(self.user_table)
        
        # Add user form
        add_user_form = QFormLayout()
        self.new_username_input = QLineEdit()
        self.new_username_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.new_password_input = QLineEdit()
        self.new_password_input.setEchoMode(QLineEdit.Password)
        self.new_password_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.new_email_input = QLineEdit()
        self.new_email_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.new_role_combo = QComboBox()
        self.new_role_combo.addItems(UserManager.ROLES.keys())
        self.new_role_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        add_user_form.addRow("Username:", self.new_username_input)
        add_user_form.addRow("Password:", self.new_password_input)
        add_user_form.addRow("Email:", self.new_email_input)
        add_user_form.addRow("Role:", self.new_role_combo)
        
        add_user_button = QPushButton("Add User")
        add_user_button.clicked.connect(self.add_user)
        add_user_button.setMinimumWidth(100)
        
        user_layout.addLayout(add_user_form)
        user_layout.addWidget(add_user_button)
        
        user_group.setLayout(user_layout)
        layout.addWidget(user_group)
        
        # Security settings section
        security_group = QGroupBox("Security Settings")
        security_layout = QVBoxLayout()
        
        # Lock timer settings
        lock_timer_layout = QFormLayout()
        
        # Current lock timer display
        current_timeout = getattr(self, 'lock_timeout_ms', 900000)
        current_minutes = current_timeout // 60000
        
        self.current_lock_time_label = QLabel(f"{current_minutes} minutes")
        self.current_lock_time_label.setWordWrap(True)
        lock_timer_layout.addRow("Current Auto-Lock Time:", self.current_lock_time_label)
        
        # Lock timer adjustment - responsive
        lock_timer_options = QHBoxLayout()
        
        # Predefined time options
        self.lock_time_combo = QComboBox()
        self.lock_time_combo.addItems([
            "5 minutes",
            "10 minutes", 
            "15 minutes",
            "30 minutes",
            "60 minutes",
            "Custom"
        ])
        self.lock_time_combo.setCurrentText("15 minutes")
        self.lock_time_combo.currentTextChanged.connect(self.on_lock_time_changed)
        self.lock_time_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.lock_time_combo.setMinimumWidth(100)
        
        lock_timer_options.addWidget(QLabel("Set Auto-Lock:"))
        lock_timer_options.addWidget(self.lock_time_combo)
        
        # Custom time input (initially hidden)
        self.custom_lock_time_spin = QSpinBox()
        self.custom_lock_time_spin.setRange(1, 180)
        self.custom_lock_time_spin.setValue(15)
        self.custom_lock_time_spin.setSuffix(" minutes")
        self.custom_lock_time_spin.setVisible(False)
        self.custom_lock_time_spin.valueChanged.connect(self.on_custom_lock_time_changed)
        self.custom_lock_time_spin.setMinimumWidth(80)
        
        lock_timer_options.addWidget(self.custom_lock_time_spin)
        lock_timer_options.addStretch()
        
        security_layout.addLayout(lock_timer_options)
        
        # Apply button
        apply_lock_time_button = QPushButton("Apply Lock Timer Settings")
        apply_lock_time_button.clicked.connect(self.apply_lock_timer_settings)
        apply_lock_time_button.setMinimumWidth(150)
        apply_lock_time_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        security_layout.addWidget(apply_lock_time_button)
        
        # Lock timer info
        lock_info_label = QLabel("Note: The auto-lock timer protects your data by requiring re-authentication after inactivity.")
        lock_info_label.setWordWrap(True)
        lock_info_label.setStyleSheet("color: #6c757d; font-style: italic; padding: 5px;")
        security_layout.addWidget(lock_info_label)
        
        security_group.setLayout(security_layout)
        layout.addWidget(security_group)
        
        # Email Configuration section
        email_group = QGroupBox("Email Configuration (for Alert Notifications)")
        email_layout = QVBoxLayout()
        
        # Load existing email settings
        self.email_config_file = os.path.join(DATA_DIR, "email_config.json")
        self.email_config = self.load_email_config()
        
        email_form = QFormLayout()
        
        self.smtp_server_input = QLineEdit()
        self.smtp_server_input.setText(self.email_config.get('smtp_server', 'smtp.gmail.com'))
        self.smtp_server_input.setPlaceholderText("smtp.gmail.com")
        
        self.smtp_port_input = QSpinBox()
        self.smtp_port_input.setRange(1, 65535)
        self.smtp_port_input.setValue(self.email_config.get('smtp_port', 587))
        
        self.smtp_username_input = QLineEdit()
        self.smtp_username_input.setText(self.email_config.get('smtp_username', ''))
        self.smtp_username_input.setPlaceholderText("your-email@gmail.com")
        
        self.smtp_password_input = QLineEdit()
        self.smtp_password_input.setText(self.email_config.get('smtp_password', ''))
        self.smtp_password_input.setEchoMode(QLineEdit.Password)
        self.smtp_password_input.setPlaceholderText("App password or account password")
        
        self.from_email_input = QLineEdit()
        self.from_email_input.setText(self.email_config.get('from_email', ''))
        self.from_email_input.setPlaceholderText("sender@example.com")
        
        email_form.addRow("SMTP Server:", self.smtp_server_input)
        email_form.addRow("SMTP Port:", self.smtp_port_input)
        email_form.addRow("Username:", self.smtp_username_input)
        email_form.addRow("Password:", self.smtp_password_input)
        email_form.addRow("From Email:", self.from_email_input)
        
        email_layout.addLayout(email_form)
        
        # Test email button
        test_email_btn = QPushButton("Test Email Configuration")
        test_email_btn.clicked.connect(self.test_email_configuration)
        test_email_btn.setMinimumWidth(150)
        
        # Save email config button
        save_email_btn = QPushButton("Save Email Configuration")
        save_email_btn.clicked.connect(self.save_email_configuration)
        save_email_btn.setMinimumWidth(150)
        save_email_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        
        email_buttons_layout = QHBoxLayout()
        email_buttons_layout.addWidget(test_email_btn)
        email_buttons_layout.addWidget(save_email_btn)
        email_buttons_layout.addStretch()
        email_layout.addLayout(email_buttons_layout)
        
        # Email status label
        self.email_status_label = QLabel()
        self.email_status_label.setWordWrap(True)
        email_layout.addWidget(self.email_status_label)
        
        email_info_label = QLabel("Note: Email alerts will be sent from custom monitoring tabs when enabled. Configure SMTP settings here.")
        email_info_label.setWordWrap(True)
        email_info_label.setStyleSheet("color: #6c757d; font-style: italic; padding: 5px;")
        email_layout.addWidget(email_info_label)
        
        email_group.setLayout(email_layout)
        layout.addWidget(email_group)
        
        # Status message
        self.admin_status_label = QLabel()
        self.admin_status_label.setWordWrap(True)
        layout.addWidget(self.admin_status_label)
        
        # Add stretch at the end
        layout.addStretch()
        
        # Set container widget and add to scroll area
        scroll_area.setWidget(container_widget)
        main_layout.addWidget(scroll_area)
        
        # Load existing users
        self.load_users()
    
    def load_email_config(self):
        """Load email configuration from file"""
        try:
            if os.path.exists(self.email_config_file):
                with open(self.email_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading email config: {str(e)}")
        return {}
    
    def _resolve_email_config(self):
        """Resolve email config, with environment variables overriding file values."""
        config = {}
        if hasattr(self, 'email_config') and isinstance(self.email_config, dict):
            config.update(self.email_config)
        
        if hasattr(self, 'email_config_file') and os.path.exists(self.email_config_file):
            try:
                with open(self.email_config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                if isinstance(file_config, dict):
                    config.update(file_config)
            except Exception as e:
                logger.error(f"Error reading email config file: {str(e)}")
        
        env_map = {
            'smtp_server': os.getenv('MONITOR_SMTP_SERVER'),
            'smtp_port': os.getenv('MONITOR_SMTP_PORT'),
            'smtp_username': os.getenv('MONITOR_SMTP_USERNAME'),
            'smtp_password': os.getenv('MONITOR_SMTP_PASSWORD'),
            'from_email': os.getenv('MONITOR_FROM_EMAIL')
        }
        for key, value in env_map.items():
            if value is not None and str(value).strip():
                config[key] = str(value).strip()
        
        try:
            config['smtp_port'] = int(config.get('smtp_port', 587))
        except (TypeError, ValueError):
            config['smtp_port'] = 587
        
        return config
    
    def _get_email_config_error(self, config):
        """Return missing-field message or None."""
        required = ['smtp_server', 'smtp_username', 'smtp_password', 'from_email']
        missing = [field for field in required if not config.get(field)]
        if missing:
            return (
                "Missing email settings: "
                + ", ".join(missing)
                + ". Configure Administration > Email or set MONITOR_SMTP_* environment variables."
            )
        return None
    
    def _send_email_via_config(self, to_email, subject, body, config):
        """Send a plaintext email using the provided SMTP config."""
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(config['smtp_server'], config.get('smtp_port', 587))
        server.starttls()
        server.login(config['smtp_username'], config['smtp_password'])
        server.send_message(msg)
        server.quit()
    
    def save_email_configuration(self):
        """Save email configuration to file"""
        try:
            config = {
                'smtp_server': self.smtp_server_input.text().strip(),
                'smtp_port': self.smtp_port_input.value(),
                'smtp_username': self.smtp_username_input.text().strip(),
                'smtp_password': self.smtp_password_input.text().strip(),
                'from_email': self.from_email_input.text().strip()
            }
            
            # Validate
            if not all([config['smtp_server'], config['smtp_username'], config['from_email']]):
                self.email_status_label.setText("Error: SMTP Server, Username, and From Email are required")
                self.email_status_label.setStyleSheet("color: red;")
                return
            
            # Save to file
            os.makedirs(os.path.dirname(self.email_config_file), exist_ok=True)
            with open(self.email_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            self.email_config = config
            self.email_status_label.setText("Email configuration saved successfully!")
            self.email_status_label.setStyleSheet("color: green;")
            logger.info("Email configuration saved")
            
        except Exception as e:
            logger.error(f"Error saving email config: {str(e)}")
            self.email_status_label.setText(f"Error saving configuration: {str(e)}")
            self.email_status_label.setStyleSheet("color: red;")
    
    def test_email_configuration(self):
        """Test email configuration by sending a test email"""
        try:
            config = self._resolve_email_config()
            config.update({
                'smtp_server': self.smtp_server_input.text().strip() or config.get('smtp_server'),
                'smtp_port': self.smtp_port_input.value() or config.get('smtp_port', 587),
                'smtp_username': self.smtp_username_input.text().strip() or config.get('smtp_username'),
                'smtp_password': self.smtp_password_input.text().strip() or config.get('smtp_password'),
                'from_email': self.from_email_input.text().strip() or config.get('from_email')
            })
            
            config_error = self._get_email_config_error(config)
            if config_error:
                self.email_status_label.setText(f"Error: {config_error}")
                self.email_status_label.setStyleSheet("color: red;")
                return
            
            test_recipient = config['smtp_username']  # Send to self
            subject = "Test Email from Monitoring System"
            body = "This is a test email from SDA v4.0.\n\nIf you received this, your email configuration is working correctly."
            self._send_email_via_config(test_recipient, subject, body, config)
            
            self.email_status_label.setText(f"Test email sent successfully to {test_recipient}!")
            self.email_status_label.setStyleSheet("color: green;")
            
        except Exception as e:
            logger.error(f"Error testing email: {str(e)}")
            self.email_status_label.setText(f"Error sending test email: {str(e)}")
            self.email_status_label.setStyleSheet("color: red;")

    def add_user(self):
        if not self.service_layer.authorize(self.current_username, "manage_users", resource="admin/users"):
            self.admin_status_label.setText("Permission denied: cannot add users")
            return
        
        username = self.new_username_input.text()
        password = self.new_password_input.text()
        email = self.new_email_input.text()
        role = self.new_role_combo.currentText()
        
        if not username or not password or not email:
            self.admin_status_label.setText("Username, password, and email are required")
            return
        
        # Basic email validation
        if '@' not in email or '.' not in email:
            self.admin_status_label.setText("Please enter a valid email address")
            return
        
        meets_policy, policy_message = self.user_manager._password_meets_policy(password)
        if not meets_policy:
            self.admin_status_label.setText(policy_message)
            return
        
        success = self.user_manager.add_user(username, password, role, email)
        if success:
            self.admin_status_label.setText(f"User {username} added successfully")
            write_audit_event(
                actor=self.current_username,
                action="user_create",
                resource=f"user/{username}",
                outcome="success",
                details={"role": role, "email": email},
                auth_provider=self.current_auth_provider
            )
            self.load_users()
            # Clear input fields after successful addition
            self.new_username_input.clear()
            self.new_password_input.clear()
            self.new_email_input.clear()
        else:
            self.admin_status_label.setText(f"Failed to add user {username}")
            write_audit_event(
                actor=self.current_username,
                action="user_create",
                resource=f"user/{username}",
                outcome="failed",
                details={"role": role},
                auth_provider=self.current_auth_provider
            )

    def load_users(self):
        self.user_table.setRowCount(0)
        
        for username, user_data in self.user_manager.users.items():
            row = self.user_table.rowCount()
            self.user_table.insertRow(row)
            self.user_table.setItem(row, 0, QTableWidgetItem(username))
            self.user_table.setItem(row, 1, QTableWidgetItem(user_data["role"]))
            self.user_table.setItem(row, 2, QTableWidgetItem(user_data.get("email", "")))
            
            # Create delete button
            delete_btn = QPushButton("Delete User")
            delete_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f8f9fa;
                    color: black;
                    border: none;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #c82333;
                    color: white;
                    
                    
                }
                QPushButton:pressed {
                    background-color: #bd2130;
                }
            """)
        
            delete_btn.clicked.connect(lambda checked, u=username: self.delete_user(u))
            self.user_table.setCellWidget(row, 3, delete_btn)

    def delete_user(self, username):
        """Delete a user from the system"""
        try:
            if not self.service_layer.authorize(self.current_username, "manage_users", resource=f"user/{username}"):
                QMessageBox.warning(self, "Delete User", "Permission denied.")
                return
            
            # Don't allow deleting the admin user
            if username == "admin":
                QMessageBox.warning(self, "Delete User", 
                                  "The admin user cannot be deleted.")
                return

            # Don't allow users to delete themselves
            if username == self.current_username:
                QMessageBox.warning(self, "Delete User", 
                                  "You cannot delete your own account.")
                return

            # Confirm deletion
            reply = QMessageBox.question(self, "Delete User",
                                       f"Are you sure you want to delete user {username}?",
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)

            if reply == QMessageBox.Yes:
                # Delete user from the user database
                if username in self.user_manager.users:
                    del self.user_manager.users[username]
                    self.user_manager.save_users()
                    
                    # Log the deletion
                    logger.info(f"User {username} deleted by {self.current_username}")
                    write_audit_event(
                        actor=self.current_username,
                        action="user_delete",
                        resource=f"user/{username}",
                        outcome="success",
                        details={},
                        auth_provider=self.current_auth_provider
                    )
                    
                    # Update the user table
                    self.load_users()
                    
                    # Show success message
                    self.admin_status_label.setText(f"User {username} deleted successfully")
                else:
                    self.admin_status_label.setText(f"User {username} not found")
                    write_audit_event(
                        actor=self.current_username,
                        action="user_delete",
                        resource=f"user/{username}",
                        outcome="failed",
                        details={"reason": "not_found"},
                        auth_provider=self.current_auth_provider
                    )

        except Exception as e:
            error_msg = f"Error deleting user: {str(e)}"
            logger.error(error_msg)
            self.admin_status_label.setText(error_msg)
            QMessageBox.critical(self, "Error", f"Failed to delete user: {str(e)}")

    def send_email_alert(self, num_anomalies, anomaly_data):
        """Send email alert to all users with detection report"""
        alert_span = OBSERVABILITY.start_span(
            "alerts.send_detection_email",
            {"num_anomalies": int(num_anomalies)}
        ) if OBSERVABILITY else None
        
        try:
            email_config = self._resolve_email_config()
            config_error = self._get_email_config_error(email_config)
            if config_error:
                logger.warning(f"Detection report email skipped: {config_error}")
                self.analysis_status_label.setText(f"Warning: {config_error}")
                if OBSERVABILITY:
                    OBSERVABILITY.inc_counter("alerts_sent_total", labels={"channel": "email", "status": "skipped"})
                    OBSERVABILITY.end_span(alert_span, status="error", attributes={"reason": "config_missing"})
                return
            
            sent_count = 0
            for username, user_data in self.user_manager.users.items():
                to_email = str(user_data.get("email", "")).strip()
                if not _is_deliverable_alert_email(to_email):
                    continue

                # Calculate anomaly rate
                total_records = len(self.data_processor.data)
                anomaly_rate = (num_anomalies / total_records) * 100 if total_records > 0 else 0
                    
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                body = (
                    "Telemetry Monitoring Alert\n"
                        
                    f"Detection Time : {timestamp}\n"
                    f"Total Records  : {total_records}\n"
                    f"Anomalies Found: {num_anomalies}\n"
                    f"Anomaly Rate   : {anomaly_rate:.2f}%\n\n"
                    "Summary:\n"
                    "- The anomaly detection pipeline identified deviations that exceed the configured thresholds.\n"
                    "- Please review the latest telemetry batch and investigate the affected subsystems.\n"
                    "- Recommended next step: validate the data source, confirm sensor health, and cross-check with system logs.\n\n"
                    "This notification was generated automatically by the Telemetry Monitoring Tool.\n"
                    f"Recipient: {username} ({user_data['role']})\n"
                    "For additional details, consult the System Health tab or the attached reports.\n"
                )
                self._send_email_via_config(
                    to_email,
                    "Anomaly Detection Report",
                    body,
                    email_config
                )
                sent_count += 1
                    
                logger.info(f"Detection report email sent to {to_email}")
            
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("alerts_sent_total", sent_count, labels={"channel": "email", "status": "success"})
                OBSERVABILITY.end_span(alert_span, status="ok", attributes={"emails_sent": sent_count})
            
        except Exception as e:
            error_msg = f"Error sending detection report email: {str(e)}"
            logger.error(error_msg)
            self.analysis_status_label.setText(f"Warning: Failed to send email alert: {str(e)}")
            if OBSERVABILITY:
                OBSERVABILITY.inc_counter("alerts_sent_total", labels={"channel": "email", "status": "error"})
                OBSERVABILITY.end_span(alert_span, status="error", error=e)

    def on_lock_time_changed(self, text):
        """Handle lock time combo box changes"""
        if text == "Custom":
            self.custom_lock_time_spin.setVisible(True)
        else:
            self.custom_lock_time_spin.setVisible(False)
    
    def on_custom_lock_time_changed(self, value):
        """Handle custom lock time spinner changes"""
        # Update the current display when custom value changes
        pass
    
    def apply_lock_timer_settings(self):
        """Apply the selected lock timer settings"""
        try:
            # Get the selected time
            selected_time = self.lock_time_combo.currentText()
            
            if selected_time == "Custom":
                minutes = self.custom_lock_time_spin.value()
            else:
                # Parse minutes from the selection
                minutes = int(selected_time.split()[0])
            
            # Convert to milliseconds
            timeout_ms = minutes * 60 * 1000
            
            # Store the new timeout value
            self.lock_timeout_ms = timeout_ms
            
            # Update the timer
            if hasattr(self, 'inactivity_timer'):
                self.inactivity_timer.stop()
                self.inactivity_timer.start(timeout_ms)
                
                # Update the display
                self.current_lock_time_label.setText(f"{minutes} minutes")
                
                # Update status
                self.admin_status_label.setText(f"Auto-lock timer updated to {minutes} minutes")
                
                # Log the change
                logger.info(f"Auto-lock timer changed to {minutes} minutes by {self.current_username}")
                
                # Show confirmation message
                QMessageBox.information(
                    self, 
                    "Lock Timer Updated", 
                    f"Auto-lock timer has been set to {minutes} minutes.\n\n"
                    f"The tool will now automatically lock after {minutes} minutes of inactivity."
                )
            else:
                self.admin_status_label.setText("Error: Timer not initialized")
                
        except Exception as e:
            error_msg = f"Error updating lock timer: {str(e)}"
            logger.error(error_msg)
            self.admin_status_label.setText(f"Error: {error_msg}")
            QMessageBox.critical(self, "Error", f"Failed to update lock timer: {str(e)}")
    
    def get_current_lock_time_minutes(self):
        """Get the current lock timer in minutes"""
        if hasattr(self, 'inactivity_timer'):
            # Get remaining time or interval if available
            return 15  # Default fallback
        return 15

    def browse_data_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*)")
        if file_name:
            self.file_path_input.setText(file_name)
            self.data_status_label.setText(f"Selected file: {file_name}")

    def browse_log_file(self):
        """Select a mission/event log file for M-LLM ingest (Data Import)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Log File",
            "",
            "Log files (*.log *.txt);;All files (*.*)",
        )
        if not path:
            return
        if hasattr(self, "log_path_input"):
            self.log_path_input.setText(path)
        if hasattr(self, "config"):
            self.config["log_file"] = path
            parent = getattr(self, "parent_window", None)
            tab_id = getattr(self, "tab_id", None)
            if parent and tab_id and hasattr(parent, "tab_config_manager"):
                parent.tab_config_manager.add_config(tab_id, self.config)

    def load_log_file(self):
        """Ingest .log/.txt into LogMonitor when host is a custom tab."""
        # Prefer CustomMonitoringTab native implementation when bound that way
        from app.tabs.custom_tab.widget import CustomMonitoringTab

        if isinstance(self, CustomMonitoringTab):
            return CustomMonitoringTab.load_log_file(self)
        path = ""
        if hasattr(self, "log_path_input"):
            path = self.log_path_input.text().strip()
        if not path:
            QMessageBox.warning(self, "Load Logs", "Please select a log file first.")
            return
        tab_id = getattr(self, "tab_id", None)
        if not tab_id:
            QMessageBox.information(
                self,
                "Load Logs",
                "Log ingest is available on custom monitoring tabs (Data Import page).",
            )
            return
        return CustomMonitoringTab.load_log_file(self)
    
    def browse_auto_folder(self):
        """Browse for folder containing data files to auto-load from"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.auto_folder_input.setText(folder)
            if hasattr(self, "config") and hasattr(self, "tab_id"):
                self.config["data_folder"] = folder
                parent = getattr(self, "parent_window", None)
                if parent and hasattr(parent, "tab_config_manager"):
                    parent.tab_config_manager.add_config(self.tab_id, self.config)
            
            # Check if the folder has valid data files
            has_valid_files = False
            for file in os.listdir(folder):
                if file.lower().endswith(('.csv', '.json')):
                    has_valid_files = True
                    break
                    
            if hasattr(self, "data_status_label"):
                if has_valid_files:
                    self.data_status_label.setText(f"Selected auto-load folder: {folder}")
                else:
                    self.data_status_label.setText(f"Warning: No CSV or JSON files found in {folder}")
    
    def auto_load_latest(self):
        """Auto load the most recently modified file from the selected folder"""
        folder_path = self.auto_folder_input.text()
        if not folder_path or not os.path.isdir(folder_path):
            self.data_status_label.setText("Please select a valid folder first")
            return
            
        try:
            # Get all files in the folder
            files = []
            for f in os.listdir(folder_path):
                file_path = os.path.join(folder_path, f)
                if os.path.isfile(file_path):
                    # Only include CSV and JSON files
                    if file_path.lower().endswith(('.csv', '.json')):
                        files.append(file_path)
            
            if not files:
                self.data_status_label.setText("No CSV or JSON files found in the selected folder")
                return
                
            # Sort by latest telemetry timestamp when available, else filesystem mtime.
            files.sort(key=_file_telemetry_sort_key, reverse=True)
            
            # Get the most recently modified file
            latest_file = files[0]
            file_extension = os.path.splitext(latest_file)[1].lower()
            
            self.data_status_label.setText(f"Found latest file: {os.path.basename(latest_file)}")
            QApplication.processEvents()  # Update UI immediately
            
            # Set the file path and update the UI
            self.file_path_input.setText(latest_file)
            
            # Determine file type
            if file_extension == '.csv':
                self.file_type_combo.setCurrentText("CSV")
                self.data_status_label.setText(f"Loading CSV file: {os.path.basename(latest_file)}")
                QApplication.processEvents()
                success, message = self.data_processor.load_csv(latest_file)
            elif file_extension == '.json':
                self.file_type_combo.setCurrentText("JSON")
                self.data_status_label.setText(f"Loading JSON file: {os.path.basename(latest_file)}")
                QApplication.processEvents()
                success, message = self.data_processor.load_json(latest_file)
            else:
                self.data_status_label.setText(f"Unsupported file type: {file_extension}")
                return
            
            # Process load result
            if success:
                self.data_status_label.setText(f"Successfully loaded: {os.path.basename(latest_file)}")
                self.update_data_preview()
                
                # Update feature dropdown in visualization tab
                if hasattr(self, 'feature_combo'):
                    self.feature_combo.clear()
                    if self.data_processor.data is not None:
                        self.feature_combo.addItems(self.data_processor.data.columns)
                        
                        # Auto-select 'value' if it exists
                        index = self.feature_combo.findText('value')
                        if (index >= 0):
                            self.feature_combo.setCurrentIndex(index)
            else:
                self.data_status_label.setText(f"Error loading file: {message}")
        
        except Exception as e:
            self.data_status_label.setText(f"Error in auto-loading: {str(e)}")
            logger.error(f"Auto-load error: {str(e)}")
    
    def load_data(self):
        file_path = self.file_path_input.text()
        file_type = self.file_type_combo.currentText().lower()
        
        if not file_path:
            self.data_status_label.setText("Please select a data file")
            return
        
        if file_type == "csv":
            success, message = self.data_processor.load_csv(file_path)
        elif file_type == "json":
            success, message = self.data_processor.load_json(file_path)
        else:
            self.data_status_label.setText("Unsupported file type")
            return
        
        if success:
            self.data_status_label.setText(message)
            self.update_data_preview()
            
            # AUTO-SET MONITORING FOLDER for System Health tab
            import os
            folder_path = os.path.dirname(file_path)
            if folder_path:
                self.monitoring_folder_path = folder_path
                if hasattr(self, 'monitoring_folder_label'):
                    self.monitoring_folder_label.setText(folder_path)
                    self.monitoring_folder_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 3px;")
                    logger.info(f"System Health monitoring folder auto-set to: {folder_path}")
            
            # Trigger multi-column analysis when data is loaded (only if exists)
            if hasattr(self, 'refresh_column_analysis') and hasattr(self, 'timestamp_combo'):
                self.refresh_column_analysis()
            
            # Update feature dropdown in visualization tab
            if hasattr(self, 'feature_combo'):
                self.feature_combo.clear()
                if self.data_processor.data is not None:
                    self.feature_combo.addItems(self.data_processor.data.columns)
                    
            # If we have 'time' and 'value' columns, auto-select them for visualization
            if (self.data_processor.data is not None and 
                'time' in self.data_processor.data.columns and 
                'value' in self.data_processor.data.columns and
                hasattr(self, 'timestamp_combo')):
                # Set timestamp column in combo box
                index = self.timestamp_combo.findText('time')
                if index >= 0:
                    self.timestamp_combo.setCurrentIndex(index)
                
                # Select 'value' in feature combo box if it exists
                if hasattr(self, 'feature_combo'):
                    index = self.feature_combo.findText('value')
                    if index >= 0:
                        self.feature_combo.setCurrentIndex(index)
        else:
            self.data_status_label.setText(message)
    
    def update_data_preview(self):
        data = self.data_processor.data
        if data is not None:
            self.data_table.setRowCount(min(data.shape[0], 100))  # Limit to 100 rows for performance
            self.data_table.setColumnCount(data.shape[1])
            self.data_table.setHorizontalHeaderLabels(data.columns)
            
            for i in range(min(data.shape[0], 100)):  # Limit to 100 rows
                for j in range(data.shape[1]):
                    self.data_table.setItem(i, j, QTableWidgetItem(str(data.iat[i, j])))
            
            self.timestamp_combo.clear()
            self.timestamp_combo.addItems(data.columns)
            
            # Auto-select 'time' as timestamp column if it exists
            index = self.timestamp_combo.findText('time')
            if index >= 0:
                self.timestamp_combo.setCurrentIndex(index)
            
            self.feature_list.setRowCount(data.shape[1])
            for i, col in enumerate(data.columns):
                self.feature_list.setItem(i, 0, QTableWidgetItem(col))
                # Mark 'value' columns as "Yes" for features, others as "No"
                if col == 'value' or col.startswith('value'):
                    self.feature_list.setItem(i, 1, QTableWidgetItem("Yes"))
                else:
                    self.feature_list.setItem(i, 1, QTableWidgetItem("No"))
            
            # Update visualization feature combo box
            self.update_visualization_features()
    
    def preprocess_data(self):
        if not hasattr(self.data_processor, 'data') or self.data_processor.data is None:
            self.data_status_label.setText("No data loaded. Please load data first.")
            return
        
        timestamp_col = None
        if hasattr(self, 'timestamp_combo') and self.timestamp_combo.count() > 0:
            timestamp_col = self.timestamp_combo.currentText()
        
        # Collect selected feature columns from the new multi-column interface
        feature_cols = []
        
        # First try to get features from the new multi-column interface
        if hasattr(self, 'selected_features_list') and self.selected_features_list.rowCount() > 0:
            for i in range(self.selected_features_list.rowCount()):
                col_item = self.selected_features_list.item(i, 0)
                if col_item:
                    feature_cols.append(col_item.text())
        else:
            # Fall back to legacy feature list for backward compatibility
            for i in range(self.feature_list.rowCount()):
                col_item = self.feature_list.item(i, 0)
                use_item = self.feature_list.item(i, 1)
                if col_item and use_item and use_item.text().lower() == "yes":
                    feature_cols.append(col_item.text())
        
        normalize = self.normalize_check.isChecked()
        remove_outliers = self.remove_outliers_check.isChecked()
        
        # Get high-dimensional optimization settings
        preprocessing_steps = {}
        if hasattr(self, 'dimred_combo'):
            dimred_method = self.dimred_combo.currentText()
            if "PCA (95% variance)" in dimred_method:
                preprocessing_steps['dimension_reduction'] = 'pca'
                preprocessing_steps['pca_variance'] = 0.95
            elif "PCA (90% variance)" in dimred_method:
                preprocessing_steps['dimension_reduction'] = 'pca'
                preprocessing_steps['pca_variance'] = 0.90
            elif "PCA (Custom components)" in dimred_method:
                preprocessing_steps['dimension_reduction'] = 'pca'
                preprocessing_steps['pca_components'] = self.dimred_param_input.value()
            elif "Feature Selection (Top K)" in dimred_method:
                preprocessing_steps['feature_selection'] = 'top_k'
                preprocessing_steps['k_features'] = self.dimred_param_input.value()
            elif "Variance Threshold" in dimred_method:
                preprocessing_steps['variance_threshold'] = True
            elif "Correlation Filter" in dimred_method:  
                preprocessing_steps['correlation_filter'] = True
                preprocessing_steps['correlation_threshold'] = 0.95
        
        # Check if we have anomaly data that needs to be preserved
        has_anomaly_data = "Anomaly" in self.data_processor.data.columns
        anomaly_data = None
        anomaly_score = None
        
        if has_anomaly_data:
            # Save anomaly data before preprocessing
            anomaly_data = self.data_processor.data["Anomaly"].copy()
            if "Anomaly Score" in self.data_processor.data.columns:
                anomaly_score = self.data_processor.data["Anomaly Score"].copy()
            logger.info("Preserving anomaly data during preprocessing")
        
        # Create worker to run advanced preprocessing with high-dimensional support
        if preprocessing_steps:
            # Use advanced preprocessing for high-dimensional data
            self.worker = Worker("advanced_preprocess", self.data_processor, 
                               timestamp_col=timestamp_col, feature_cols=feature_cols, 
                               preprocessing_steps=preprocessing_steps)
        else:
            # Use standard preprocessing  
            self.worker = Worker("preprocess", self.data_processor, timestamp_col=timestamp_col, 
                               feature_cols=feature_cols, normalize=normalize, 
                               remove_outliers=remove_outliers)
        self.worker.finished.connect(lambda success, message, result: 
                                   self.on_preprocessing_finished(success, message, result, 
                                                               has_anomaly_data, anomaly_data, anomaly_score))
        self.worker.start()
        
        self.data_status_label.setText("Preprocessing data...")
    
    def on_preprocessing_finished(self, success, message, result, has_anomaly_data=False, 
                                 anomaly_data=None, anomaly_score=None):
        """Handle completion of preprocessing with anomaly data preservation"""
        self.data_status_label.setText(message)
        
        # Restore anomaly data if available
        if success and has_anomaly_data and self.data_processor.preprocessed_data is not None:
            try:
                # The length might have changed if outliers were removed
                if len(self.data_processor.preprocessed_data) == len(anomaly_data):
                    self.data_processor.preprocessed_data["Anomaly"] = anomaly_data
                    if anomaly_score is not None:
                        self.data_processor.preprocessed_data["Anomaly Score"] = anomaly_score
                    logger.info("Successfully restored anomaly data after preprocessing")
                else:
                    logger.warning("Could not restore anomaly data - row count mismatch after preprocessing")
            except Exception as e:
                logger.error(f"Error restoring anomaly data: {str(e)}")
        
        # Update the visualization tab feature combo box
        if success and hasattr(self, 'feature_combo'):
            self.feature_combo.clear()
            if self.data_processor.preprocessed_data is not None:
                # Add all columns including 'Anomaly' if it's there
                all_columns = list(self.data_processor.preprocessed_data.columns)
                self.feature_combo.addItems(all_columns)
                
                # Auto-select 'value' if it exists
                value_index = self.feature_combo.findText('value')
                if value_index >= 0:
                    self.feature_combo.setCurrentIndex(value_index)
        
        self.update_data_preview()
    
    def on_dataset_type_changed(self, dataset_type):
        """Handle dataset type changes and auto-configure columns"""
        if not hasattr(self.data_processor, 'data') or self.data_processor.data is None:
            return
        
        self.column_stats_label.setText(f"Dataset Type: {dataset_type}")
        
        # Auto-configure based on dataset type
        if dataset_type == "Time Series (timestamp + values)":
            self.auto_detect_timestamp.setChecked(True)
            self.auto_configure_timeseries()
        elif dataset_type == "Multi-Feature (multiple sensors/metrics)":
            self.auto_configure_multifeature()
        elif dataset_type == "IoT Telemetry (device_id + multiple readings)":
            self.auto_configure_iot_telemetry()
        elif dataset_type == "Log Data (timestamp + multiple fields)":
            self.auto_configure_log_data()
        else:
            self.refresh_column_analysis()
    
    def auto_configure_timeseries(self):
        """Auto-configure for time series data"""
        if self.data_processor.data is None:
            return
        
        columns = self.data_processor.data.columns.tolist()
        
        # Try to identify timestamp column
        timestamp_candidates = []
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'date', 'timestamp', 'ts', 'datetime']):
                timestamp_candidates.append(col)
        
        if timestamp_candidates:
            self.timestamp_combo.setCurrentText(timestamp_candidates[0])
        
        # Set remaining columns as features
        feature_cols = [col for col in columns if col not in timestamp_candidates]
        self.populate_feature_tables(columns, timestamp_candidates[0] if timestamp_candidates else None, feature_cols)
    
    def auto_configure_multifeature(self):
        """Auto-configure for multi-feature sensor data"""
        if self.data_processor.data is None:
            return
        
        columns = self.data_processor.data.columns.tolist()
        numeric_cols = self.data_processor.data.select_dtypes(include=['number']).columns.tolist()
        
        # All numeric columns become features
        timestamp_col = None
        feature_cols = numeric_cols
        
        self.populate_feature_tables(columns, timestamp_col, feature_cols)
    
    def auto_configure_iot_telemetry(self):
        """Auto-configure for IoT telemetry data"""
        if self.data_processor.data is None:
            return
        
        columns = self.data_processor.data.columns.tolist()
        
        # Look for device ID, timestamp, and sensor readings
        device_id_candidates = []
        timestamp_candidates = []
        feature_candidates = []
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['device', 'sensor', 'id', 'node']):
                device_id_candidates.append(col)
            elif any(keyword in col_lower for keyword in ['time', 'date', 'timestamp']):
                timestamp_candidates.append(col)
            elif self.data_processor.data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                feature_candidates.append(col)
        
        timestamp_col = timestamp_candidates[0] if timestamp_candidates else None
        if timestamp_col:
            self.timestamp_combo.setCurrentText(timestamp_col)
        
        self.populate_feature_tables(columns, timestamp_col, feature_candidates)
    
    def auto_configure_log_data(self):
        """Auto-configure for log data with timestamp and multiple fields"""
        if self.data_processor.data is None:
            return
        
        columns = self.data_processor.data.columns.tolist()
        
        # Identify timestamp and exclude non-numeric fields that are likely identifiers
        timestamp_candidates = []
        feature_candidates = []
        exclude_keywords = ['id', 'name', 'user', 'ip', 'host', 'source', 'level']
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'date', 'timestamp']):
                timestamp_candidates.append(col)
            elif not any(keyword in col_lower for keyword in exclude_keywords):
                if self.data_processor.data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    feature_candidates.append(col)
        
        timestamp_col = timestamp_candidates[0] if timestamp_candidates else None
        if timestamp_col:
            self.timestamp_combo.setCurrentText(timestamp_col)
        
        self.populate_feature_tables(columns, timestamp_col, feature_candidates)
    
    def populate_feature_tables(self, all_columns, timestamp_col, feature_cols):
        """Populate the available and selected feature tables"""
        # Clear existing tables
        self.available_columns_list.setRowCount(0)
        self.selected_features_list.setRowCount(0)
        
        # Populate available columns
        for col in all_columns:
            if col != timestamp_col:  # Don't include timestamp in available features
                row = self.available_columns_list.rowCount()
                self.available_columns_list.insertRow(row)
                
                # Column name
                self.available_columns_list.setItem(row, 0, QTableWidgetItem(col))
                
                # Data type
                dtype = str(self.data_processor.data[col].dtype)
                self.available_columns_list.setItem(row, 1, QTableWidgetItem(dtype))
                
                # Sample values (first few non-null values)
                sample_values = self.data_processor.data[col].dropna().head(3).astype(str).tolist()
                sample_text = ", ".join(sample_values[:3]) + ("..." if len(sample_values) > 3 else "")
                self.available_columns_list.setItem(row, 2, QTableWidgetItem(sample_text))
        
        # Populate selected features
        for col in feature_cols:
            if col in all_columns:
                row = self.selected_features_list.rowCount()
                self.selected_features_list.insertRow(row)
                
                self.selected_features_list.setItem(row, 0, QTableWidgetItem(col))
                dtype = str(self.data_processor.data[col].dtype)
                self.selected_features_list.setItem(row, 1, QTableWidgetItem(dtype))
                self.selected_features_list.setItem(row, 2, QTableWidgetItem("Feature"))
                self.selected_features_list.setItem(row, 3, QTableWidgetItem("None"))
    
    def assign_timestamp_column(self):
        """Assign selected column as timestamp"""
        current_row = self.available_columns_list.currentRow()
        if current_row >= 0:
            col_item = self.available_columns_list.item(current_row, 0)
            if col_item:
                col_name = col_item.text()
                
                # Add to timestamp combo
                if self.timestamp_combo.findText(col_name) == -1:
                    self.timestamp_combo.addItem(col_name)
                self.timestamp_combo.setCurrentText(col_name)
                
                self.column_stats_label.setText(f"Timestamp column set to: {col_name}")
            else:
                QMessageBox.warning(self, "Error", "Could not read column name.")
        else:
            QMessageBox.information(self, "No Selection", "Please select a column to assign as timestamp.")
    
    def assign_feature_column(self):
        """Add selected column to features"""
        current_row = self.available_columns_list.currentRow()
        if current_row >= 0:
            col_item = self.available_columns_list.item(current_row, 0)
            if not col_item:
                QMessageBox.warning(self, "Error", "Could not read column name.")
                return
            
            col_name = col_item.text()
            
            # Check if already in features
            for row in range(self.selected_features_list.rowCount()):
                feature_item = self.selected_features_list.item(row, 0)
                if feature_item and feature_item.text() == col_name:
                    QMessageBox.information(self, "Already Added", f"Column '{col_name}' is already in features.")
                    return
            
            # Add to features table
            row = self.selected_features_list.rowCount()
            self.selected_features_list.insertRow(row)
            
            dtype_item = self.available_columns_list.item(current_row, 1)
            dtype = dtype_item.text() if dtype_item else "Unknown"
            
            self.selected_features_list.setItem(row, 0, QTableWidgetItem(col_name))
            self.selected_features_list.setItem(row, 1, QTableWidgetItem(dtype))
            self.selected_features_list.setItem(row, 2, QTableWidgetItem("Feature"))
            self.selected_features_list.setItem(row, 3, QTableWidgetItem("None"))
            
            self.column_stats_label.setText(f"➕ Added feature: {col_name}")
        else:
            QMessageBox.information(self, "No Selection", "Please select a column to add as feature.")
    
    def remove_feature_column(self):
        """Remove selected column from features"""
        current_row = self.selected_features_list.currentRow()
        if current_row >= 0:
            col_item = self.selected_features_list.item(current_row, 0)
            col_name = col_item.text() if col_item else "Unknown"
            self.selected_features_list.removeRow(current_row)
            self.column_stats_label.setText(f"➖ Removed feature: {col_name}")
        else:
            QMessageBox.information(self, "No Selection", "Please select a feature to remove.")
    
    def select_all_features(self):
        """Select all numeric columns as features"""
        if self.data_processor.data is None:
            return
        
        # Clear current features
        self.selected_features_list.setRowCount(0)
        
        # Get all numeric columns except timestamp
        timestamp_col = self.timestamp_combo.currentText()
        numeric_cols = self.data_processor.data.select_dtypes(include=['number']).columns.tolist()
        
        if timestamp_col in numeric_cols:
            numeric_cols.remove(timestamp_col)
        
        # Add all numeric columns as features
        for col in numeric_cols:
            row = self.selected_features_list.rowCount()
            self.selected_features_list.insertRow(row)
            
            dtype = str(self.data_processor.data[col].dtype)
            self.selected_features_list.setItem(row, 0, QTableWidgetItem(col))
            self.selected_features_list.setItem(row, 1, QTableWidgetItem(dtype))
            self.selected_features_list.setItem(row, 2, QTableWidgetItem("Feature"))
            self.selected_features_list.setItem(row, 3, QTableWidgetItem("None"))
        
        self.column_stats_label.setText(f"Selected {len(numeric_cols)} numeric columns as features")
    
    def apply_feature_transform(self):
        """Apply transformation to selected feature"""
        current_row = self.selected_features_list.currentRow()
        transform = self.transform_combo.currentText()
        
        if current_row >= 0:
            col_item = self.selected_features_list.item(current_row, 0)
            if col_item:
                col_name = col_item.text()
                self.selected_features_list.setItem(current_row, 3, QTableWidgetItem(transform))
                self.column_stats_label.setText(f"Applied {transform} to {col_name}")
            else:
                QMessageBox.warning(self, "Error", "Could not read feature name.")
        else:
            QMessageBox.information(self, "No Selection", "Please select a feature to transform.")
    
    def refresh_column_analysis(self):
        """Refresh column analysis and statistics"""
        if self.data_processor.data is None:
            if hasattr(self, 'column_stats_label'):
                self.column_stats_label.setText("Dataset Info: No data loaded")
            return
        
        df = self.data_processor.data
        
        # Basic dataset info
        total_cols = len(df.columns)
        numeric_cols = len(df.select_dtypes(include=['number']).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        datetime_cols = len(df.select_dtypes(include=['datetime']).columns)
        
        # Update timestamp combo with potential timestamp columns (only if it exists)
        if hasattr(self, 'timestamp_combo'):
            self.timestamp_combo.clear()
            self.timestamp_combo.addItem("")  # Empty option
            
            for col in df.columns:
                # Check if column looks like a timestamp
                col_lower = col.lower()
                if (any(keyword in col_lower for keyword in ['time', 'date', 'timestamp', 'ts', 'datetime']) or
                    df[col].dtype == 'datetime64[ns]'):
                    self.timestamp_combo.addItem(col)
            
            # Add all columns to timestamp combo as options
            for col in df.columns:
                if self.timestamp_combo.findText(col) == -1:
                    self.timestamp_combo.addItem(col)
        
        # Auto-populate available columns table (only if it exists)
        if hasattr(self, 'available_columns_list'):
            self.available_columns_list.setRowCount(0)
            for i, col in enumerate(df.columns):
                self.available_columns_list.insertRow(i)
                self.available_columns_list.setItem(i, 0, QTableWidgetItem(col))
                self.available_columns_list.setItem(i, 1, QTableWidgetItem(str(df[col].dtype)))
                
                # Sample values
                sample_values = df[col].dropna().head(3).astype(str).tolist()
                sample_text = ", ".join(sample_values[:3]) + ("..." if len(sample_values) > 3 else "")
                self.available_columns_list.setItem(i, 2, QTableWidgetItem(sample_text))
        
        # Update stats label (only if it exists)
        stats_text = (f"Dataset: {len(df)} rows × {total_cols} columns | "
                     f"Numeric: {numeric_cols} | Categorical: {categorical_cols} | "
                     f"DateTime: {datetime_cols}")
        
        if hasattr(self, 'column_stats_label'):
            if hasattr(self.data_processor, 'feature_columns') and self.data_processor.feature_columns:
                stats_text += f" | Selected Features: {len(self.data_processor.feature_columns)}"
            self.column_stats_label.setText(stats_text)
        
        # Also update legacy feature list for compatibility
        if hasattr(self, 'feature_list'):
            self.update_feature_list()
    
    def auto_optimize_high_dimensional(self):
        """Automatically optimize settings for high-dimensional datasets"""
        if self.data_processor.data is None:
            QMessageBox.information(self, "No Data", "Please load data first.")
            return
        
        df = self.data_processor.data
        n_samples, n_features = df.shape
        numeric_features = len(df.select_dtypes(include=['number']).columns)
        
        # Update dimension info
        self.dimension_info_label.setText(
            f"Dataset: {n_samples} samples × {n_features} features ({numeric_features} numeric)"
        )
        
        # Determine if this is high-dimensional
        is_high_dimensional = numeric_features > 20
        is_very_high_dimensional = numeric_features > 100
        is_ultra_high_dimensional = numeric_features > 1000
        
        # Auto-configure based on dimensionality
        if is_ultra_high_dimensional:
            # Ultra high-dimensional (>1000 features)
            self.dimred_combo.setCurrentText("PCA (90% variance)")
            self.enable_incremental_check.setChecked(True)
            self.batch_processing_check.setChecked(True)
            self.parallel_processing_check.setChecked(True)
            recommended_models = "Incremental IsoForest, SGD One-Class SVM"
            optimization_msg = "Ultra high-dimensional dataset detected! Enabled aggressive dimensionality reduction and incremental learning."
            
        elif is_very_high_dimensional:
            # Very high-dimensional (100-1000 features)
            self.dimred_combo.setCurrentText("PCA (95% variance)")
            self.enable_incremental_check.setChecked(False)
            self.batch_processing_check.setChecked(True)
            self.parallel_processing_check.setChecked(True)
            recommended_models = "Enhanced IsoForest, Random Forest"
            optimization_msg = "High-dimensional dataset detected! Enabled PCA and batch processing."
            
        elif is_high_dimensional:
            # Moderately high-dimensional (20-100 features)
            self.dimred_combo.setCurrentText("Feature Selection (Top K)")
            self.dimred_param_input.setValue(min(50, numeric_features // 2))
            self.parallel_processing_check.setChecked(True)
            recommended_models = "Ensemble Models, Enhanced IsoForest"
            optimization_msg = "Multi-dimensional dataset detected! Enabled feature selection and parallel processing."
            
        else:
            # Low-dimensional (<20 features)
            self.dimred_combo.setCurrentText("None (use all features)")
            recommended_models = "All models suitable"
            optimization_msg = "Standard dataset - all features can be used directly."
        
        # Update model recommendations
        self.highdim_model_label.setText(f"Recommended: {recommended_models}")
        
        # Show optimization summary
        QMessageBox.information(
            self, "High-Dimensional Optimization", 
            f"{optimization_msg}\n\n"
            f"Dataset Analysis:\n"
            f"• Total Features: {n_features}\n"
            f"• Numeric Features: {numeric_features}\n" 
            f"• Samples: {n_samples}\n"
            f"• Recommended Models: {recommended_models}\n\n"
            f"Optimization Applied:\n"
            f"• Dimensionality Reduction: {self.dimred_combo.currentText()}\n"
            f"• Incremental Learning: {'Enabled' if self.enable_incremental_check.isChecked() else 'Disabled'}\n"
            f"• Batch Processing: {'Enabled' if self.batch_processing_check.isChecked() else 'Disabled'}\n"
            f"• Parallel Processing: {'Enabled' if self.parallel_processing_check.isChecked() else 'Disabled'}"
        )
        
        # Also update column statistics
        stats_text = (f"High-Dim Dataset: {n_samples} samples × {n_features} features | "
                     f"Numeric: {numeric_features} | Dimensionality: "
                     f"{'Ultra-High' if is_ultra_high_dimensional else 'Very High' if is_very_high_dimensional else 'High' if is_high_dimensional else 'Standard'}")
        
        self.column_stats_label.setText(stats_text)
    
    def update_feature_list(self):
        """Update the legacy feature list for backward compatibility"""
        if self.data_processor.data is None or not hasattr(self, 'feature_list'):
            return
        
        self.feature_list.setRowCount(0)
        for col in self.data_processor.data.columns:
            row = self.feature_list.rowCount()
            self.feature_list.insertRow(row)
            self.feature_list.setItem(row, 0, QTableWidgetItem(col))
            self.feature_list.setItem(row, 1, QTableWidgetItem("Yes"))
    
    def select_all_models(self):
        """Select all models in the list"""
        for i in range(self.model_list_widget.count()):
            item = self.model_list_widget.item(i)
            item.setCheckState(Qt.Checked)
    
    def clear_all_models(self):
        """Clear all model selections"""
        for i in range(self.model_list_widget.count()):
            item = self.model_list_widget.item(i)
            item.setCheckState(Qt.Unchecked)
    
    def get_selected_models(self):
        """Get list of selected model names"""
        selected = []
        for i in range(self.model_list_widget.count()):
            item = self.model_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                stored = item.data(Qt.UserRole)
                selected.append(catalog_display_name(stored or item.text()))
        return selected
    
    def open_parameter_dialog(self):
        """Open dialog to configure model parameters"""
        selected_models = self.get_selected_models()
        
        if not selected_models:
            QMessageBox.information(
                self,
                "No Models Selected",
                "Please select at least one model from the list before configuring parameters."
            )
            return
        
        # Create and show dialog
        dialog = ModelParameterDialog(selected_models, self)
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            # Store the parameter VALUES (not widgets) from dialog
            self.model_param_values = dialog.parameter_values
            
            # Update status label
            if len(selected_models) == 1:
                self.selected_model_label.setText(f"Selected: {selected_models[0]} - Parameters configured ✓")
            else:
                self.selected_model_label.setText(f"Selected: {len(selected_models)} models - Parameters configured ✓")
            
            logger.info(f"Parameters configured for {len(selected_models)} models")
    
    def update_model_parameters(self):
        """Update UI when model selection changes"""
        selected_models = self.get_selected_models()
        
        # Update label to show selected models
        if not selected_models:
            self.selected_model_label.setText("No models selected")
        elif len(selected_models) == 1:
            self.selected_model_label.setText(f"Selected: {selected_models[0]}")
        else:
            self.selected_model_label.setText(f"Selected: {len(selected_models)} models ({', '.join(selected_models[:3])}{'...' if len(selected_models) > 3 else ''})")
        
        # Update combo box for backward compatibility
        if selected_models:
            self.model_type_combo.setCurrentText(selected_models[0])
    
    def _create_model_parameters(self, model_type, model_name):
        """Create parameter widgets for a specific model type"""
        param_widgets = []
        
        # Show warning if TensorFlow is required but not available
        if model_type in ["autoencoder", "lstm", "gru"] and not TENSORFLOW_AVAILABLE:
            warning_label = QLabel("⚠️ TensorFlow not installed")
            warning_label.setStyleSheet("color: red; font-weight: normal;")
            install_label = QLabel("pip install tensorflow")
            install_label.setStyleSheet("color: blue; font-style: italic; font-weight: normal;")
            param_widgets.append(("Warning:", warning_label))
            param_widgets.append(("Install:", install_label))
            return param_widgets
        
        if model_type == "isolation_forest":
            n_estimators = QSpinBox()
            n_estimators.setRange(10, 1000)
            n_estimators.setValue(100)
            param_widgets.append(("Estimators:", n_estimators))
            
            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setValue(0.1)
            contamination.setSingleStep(0.01)
            param_widgets.append(("Contamination:", contamination))
            
        elif model_type == "local_outlier_factor":
            n_neighbors = QSpinBox()
            n_neighbors.setRange(1, 100)
            n_neighbors.setValue(20)
            param_widgets.append(("Neighbors:", n_neighbors))
            
            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setValue(0.1)
            contamination.setSingleStep(0.01)
            param_widgets.append(("Contamination:", contamination))
            
        elif model_type in ["autoencoder", "lstm", "gru"]:
            epochs = QSpinBox()
            epochs.setRange(1, 1000)
            epochs.setValue(50)
            param_widgets.append(("Epochs:", epochs))
            
            batch_size = QSpinBox()
            batch_size.setRange(1, 1024)
            batch_size.setValue(32)
            param_widgets.append(("Batch Size:", batch_size))
            
            validation_split = QDoubleSpinBox()
            validation_split.setRange(0.01, 0.99)
            validation_split.setValue(0.2)
            validation_split.setSingleStep(0.01)
            param_widgets.append(("Validation Split:", validation_split))
            
            patience = QSpinBox()
            patience.setRange(1, 100)
            patience.setValue(5)
            param_widgets.append(("Patience:", patience))
            
            learning_rate = QDoubleSpinBox()
            learning_rate.setRange(0.0001, 1.0)
            learning_rate.setValue(0.001)
            learning_rate.setSingleStep(0.0001)
            learning_rate.setDecimals(4)
            param_widgets.append(("Learning Rate:", learning_rate))
            
            if model_type in ["lstm", "gru"]:
                sequence_length = QSpinBox()
                sequence_length.setRange(1, 100)
                sequence_length.setValue(10)
                param_widgets.append(("Sequence Length:", sequence_length))
                
        elif model_type == "xgboost":
            n_estimators = QSpinBox()
            n_estimators.setRange(10, 1000)
            n_estimators.setValue(100)
            param_widgets.append(("Estimators:", n_estimators))
            
            max_depth = QSpinBox()
            max_depth.setRange(1, 20)
            max_depth.setValue(6)
            param_widgets.append(("Max Depth:", max_depth))
            
            learning_rate = QDoubleSpinBox()
            learning_rate.setRange(0.01, 1.0)
            learning_rate.setValue(0.1)
            learning_rate.setSingleStep(0.01)
            param_widgets.append(("Learning Rate:", learning_rate))
            
            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setValue(0.1)
            contamination.setSingleStep(0.01)
            param_widgets.append(("Contamination:", contamination))
            
        elif model_type == "iqr_(interquartile_range)":
            iqr_factor = QDoubleSpinBox()
            iqr_factor.setRange(0.5, 5.0)
            iqr_factor.setValue(1.5)
            iqr_factor.setSingleStep(0.1)
            param_widgets.append(("IQR Factor:", iqr_factor))
            
        elif model_type == "z-score":
            threshold = QDoubleSpinBox()
            threshold.setRange(1.0, 5.0)
            threshold.setValue(3.0)
            threshold.setSingleStep(0.1)
            param_widgets.append(("Threshold:", threshold))
            
        elif model_type == "prophet":
            if not PROPHET_AVAILABLE:
                warning_label = QLabel("⚠️ Prophet not installed")
                warning_label.setStyleSheet("color: red; font-weight: normal;")
                param_widgets.append(("Warning:", warning_label))
                return param_widgets
            
            yearly = QCheckBox()
            yearly.setChecked(True)
            param_widgets.append(("Yearly Seasonality:", yearly))
            
            weekly = QCheckBox()
            weekly.setChecked(True)
            param_widgets.append(("Weekly Seasonality:", weekly))
            
            daily = QCheckBox()
            daily.setChecked(True)
            param_widgets.append(("Daily Seasonality:", daily))
            
            changepoint = QDoubleSpinBox()
            changepoint.setRange(0.001, 1.0)
            changepoint.setValue(0.05)
            changepoint.setSingleStep(0.001)
            changepoint.setDecimals(3)
            param_widgets.append(("Changepoint Scale:", changepoint))
        
        # For other models (ensemble, random forest, etc.), add default message
        if not param_widgets:
            default_label = QLabel("✓ Using default parameters")
            default_label.setStyleSheet("color: green; font-weight: normal; font-style: italic;")
            param_widgets.append(("Configuration:", default_label))
        
        return param_widgets
    
    def _extract_model_parameters(self, model_name, model_type):
        """Extract parameter values for a specific model"""
        # Check if we have configured parameter values
        if hasattr(self, 'model_param_values') and model_name in self.model_param_values:
            return self.model_param_values[model_name]
        
        # Use defaults if no configuration found
        return self._get_default_parameters(model_type)
    
    def _get_default_parameters(self, model_type):
        """Get default parameters for a model type"""
        defaults = {
            'isolation_forest': {'n_estimators': 100, 'contamination': 0.1},
            'local_outlier_factor': {'n_neighbors': 20, 'contamination': 0.1},
            'autoencoder': {'epochs': 50, 'batch_size': 32, 'validation_split': 0.2, 'learning_rate': 0.001},
            'lstm': {'epochs': 50, 'batch_size': 32, 'validation_split': 0.2, 'learning_rate': 0.001, 'sequence_length': 10},
            'gru': {'epochs': 50, 'batch_size': 32, 'validation_split': 0.2, 'learning_rate': 0.001, 'sequence_length': 10},
            'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'contamination': 0.1},
            'iqr_(interquartile_range)': {'iqr_factor': 1.5},
            'z-score': {'threshold': 3.0},
            'prophet': {'yearly_seasonality': True, 'weekly_seasonality': True, 'daily_seasonality': True, 'changepoint_prior_scale': 0.05}
        }
        return defaults.get(model_type, {})
    
    def train_model(self):
        if not hasattr(self.data_processor, 'preprocessed_data') or self.data_processor.preprocessed_data is None:
            self.analysis_status_label.setText("No preprocessed data available. Please import and preprocess data first.")
            return
        
        # Get selected models
        selected_models = self.get_selected_models()
        if not selected_models:
            self.analysis_status_label.setText("Please select at least one model to train.")
            return
        
        try:
            # Get only numeric columns for training
            preprocessed_data = self.data_processor.preprocessed_data
            numeric_data = preprocessed_data.select_dtypes(include=['number'])
            if numeric_data.empty:
                self.analysis_status_label.setText("No numeric features found in the preprocessed data.")
                return
                
            # Display progress in status bar
            self.status_bar.showMessage(f"Training {len(selected_models)} model(s)...")
            
            # Clear previous trained models
            self.trained_models = {}
            
            # For multi-model, we'll train them sequentially
            # Store all models and their parameters
            self.models_to_train = []
            
            for model_name in selected_models:
                model_type = to_internal_model_type(model_name)
                model_params = {}
                
                # Create new model instance
                model = AnomalyDetectionModel(model_type)
                
                # Extract parameters from the stored widgets for this model
                model_params = self._extract_model_parameters(model_name, model_type)
                
                # Skip if dependencies are missing
                runtime_type = resolve_runtime_model_type(model_type)
                if runtime_type in ["autoencoder", "lstm", "gru"] and not TENSORFLOW_AVAILABLE:
                    self.analysis_status_label.setText(f"TensorFlow required for {model_type}. Skipping...")
                    continue
                
                if runtime_type == "prophet" and not PROPHET_AVAILABLE:
                    self.analysis_status_label.setText(f"Prophet required for Prophet model. Skipping...")
                    continue
                
                # Store model and params for training
                self.models_to_train.append((model_name, model, model_params))
            
            if not self.models_to_train:
                self.analysis_status_label.setText("No valid models to train.")
                return
            
            # Start training first model
            self.current_training_index = 0
            self.numeric_data_for_training = numeric_data
            self.full_data_for_training = preprocessed_data  # Keep full data for Prophet
            self._train_next_model()
            
        except Exception as e:
            error_msg = f"Error setting up model training: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.analysis_status_label.setText(error_msg)
    
    def _on_train_log_chunk(self, text: str) -> None:
        """Mirror Keras stdout into the Analysis Training Console."""
        from app.ui.training_console import append_console_chunk

        console = getattr(self, "training_console", None)
        if console is not None:
            append_console_chunk(console, text)

    def _train_next_model(self):
        """Train the next model in the queue"""
        if self.current_training_index >= len(self.models_to_train):
            # All models trained
            self._on_all_models_trained()
            return
        
        model_name, model, model_params = self.models_to_train[self.current_training_index]
        
        # Update status
        progress_text = f"Training model {self.current_training_index + 1}/{len(self.models_to_train)}: {model_name}"
        self.analysis_status_label.setText(progress_text)
        self.status_bar.showMessage(progress_text)
        
        # Use full data for Prophet (needs timestamp), numeric data for others
        training_data = self.full_data_for_training if model.model_type == "prophet" else self.numeric_data_for_training
        
        # Create worker for async training
        self.worker = Worker(
            "train_model",
            model,
            training_data,
            **model_params
        )
        
        # Connect signals
        self.worker.finished.connect(self.on_single_model_trained)
        self.worker.progress.connect(lambda p: self.status_bar.showMessage(f"{progress_text} - {p}%"))
        self.worker.log_chunk.connect(self._on_train_log_chunk)
        from app.ui.training_console import append_training_header, clear_training_console

        if self.current_training_index == 0:
            clear_training_console(self)
        append_training_header(self, progress_text)
        
        # Disable UI elements during training
        self.model_list_widget.setEnabled(False)
        
        # Start training
        self.worker.start()
    
    def on_single_model_trained(self, success, message, model):
        """Callback when a single model finishes training"""
        model_name, _, _ = self.models_to_train[self.current_training_index]
        
        if success and model is not None:
            # Store the trained model
            self.trained_models[model_name] = model
            logger.info(f"Successfully trained {model_name}")
        else:
            logger.error(f"Failed to train {model_name}: {message}")
            self.trained_models[model_name] = None  # Mark as failed
        
        # Move to next model
        self.current_training_index += 1
        self._train_next_model()
    
    def _on_all_models_trained(self):
        """Called when all models have been trained"""
        # Re-enable UI elements
        self.model_list_widget.setEnabled(True)
        
        # Reset status bar
        self.status_bar.showMessage(f"Logged in as {self.current_username} ({self.current_role})")
        
        # Count successful models
        successful_models = [name for name, model in self.trained_models.items() if model is not None]
        failed_models = [name for name, model in self.trained_models.items() if model is None]
        
        if successful_models:
            # Set the first successful model as the primary model for backward compatibility
            self.model = self.trained_models[successful_models[0]]
            
            # Update UI
            status_text = f"✅ Trained {len(successful_models)}/{len(self.trained_models)} models successfully"
            if failed_models:
                status_text += f" ({len(failed_models)} failed: {', '.join(failed_models)})"
            
            self.analysis_status_label.setText(status_text + " - Ready for predictions")
            self.selected_model_label.setText(f"Trained: {', '.join(successful_models[:3])}{'...' if len(successful_models) > 3 else ''}")
            
            # Update visualization tab model info
            if hasattr(self, 'viz_model_info_label'):
                viz_text = f"✅ {len(successful_models)} trained models: {', '.join(successful_models)}"
                if failed_models:
                    viz_text += f"\n❌ Failed: {', '.join(failed_models)}"
                self.viz_model_info_label.setText(viz_text)
                self.viz_model_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; padding: 5px;")
            
            # Update detailed model info in visualization tab
            self.update_viz_model_info()
            
            # Enable prediction-related UI elements
            if hasattr(self, 'predict_button'):
                self.predict_button.setEnabled(True)
            
            logger.info(f"Multi-model training complete. Successful: {successful_models}")
        else:
            self.analysis_status_label.setText("❌ All models failed to train")
            logger.error("All models failed to train")
            
    def on_model_trained(self, success, message, model):
        # Re-enable UI elements
        self.model_type_combo.setEnabled(True)
        
        # Reset status bar
        self.status_bar.showMessage(f"Logged in as {self.current_username} ({self.current_role})")
        
        if success and model is not None:
            self.model = model
            
            try:
                # Update metrics safely
                metrics: dict[str, object] = {
                    'model_type': str(self.model.model_type),
                    'training_time': datetime.datetime.now().isoformat(),
                }
                
                # Add data info safely
                try:
                    if hasattr(self.data_processor, 'data') and self.data_processor.data is not None:
                        metrics['data_points'] = len(self.data_processor.data)
                    if hasattr(self.model, 'feature_columns') and self.model.feature_columns:
                        metrics['features_used'] = len(self.model.feature_columns)
                except Exception as data_error:
                    logger.warning(f"Could not get data info for metrics: {str(data_error)}")
                
                # Add model-specific metrics safely
                if hasattr(self.model, 'score_samples') and hasattr(self.model, 'offset_'):
                    try:
                        # Use only numeric data for scoring
                        numeric_data = self.data_processor.preprocessed_data.select_dtypes(include=['number'])
                        if not numeric_data.empty and len(numeric_data) > 0:
                            scores = self.model.score_samples(numeric_data)
                            if scores is not None and len(scores) > 0:
                                metrics.update({
                                    'mean_score': float(np.mean(scores)),
                                    'std_score': float(np.std(scores)),
                                    'min_score': float(np.min(scores)),
                                    'max_score': float(np.max(scores))
                                })
                    except Exception as score_error:
                        logger.warning(f"Could not calculate model scores: {str(score_error)}")
                        # Continue without scores - not critical
                
                # Store metrics safely
                if not hasattr(self.model, 'metrics'):
                    self.model.metrics = {}
                self.model.metrics.update(metrics)
                
                # Update UI
                self.analysis_status_label.setText("Model trained successfully - Ready for predictions")
                if hasattr(self, 'selected_model_label'):
                    self.selected_model_label.setText(f"Current model: {self.model.model_type}")
                
                # Enable prediction-related UI elements
                if hasattr(self, 'predict_button'):
                    self.predict_button.setEnabled(True)
                
                logger.info(f"Successfully trained {self.model.model_type} model with metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error updating model metrics: {str(e)}")
                logger.error(f"Metrics error traceback: {traceback.format_exc()}")
                # Still show success since the model was trained
                self.analysis_status_label.setText("Model trained successfully (metrics update failed)")
                if hasattr(self, 'selected_model_label'):
                    self.selected_model_label.setText(f"Current model: {self.model.model_type}")
        else:
            error_msg = message if message else "Unknown error during training"
            self.analysis_status_label.setText(f"Training failed: {error_msg}")
            logger.error(f"Model training failed: {error_msg}")

    def save_model(self):
        """Save the trained model to a file"""
        if not self.model or not self.model.model:
            QMessageBox.warning(self, "Save Model", "No trained model available to save")
            return
        
        # Create models directory if it doesn't exist
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            
        # Generate default filename with timestamp
        default_name = f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        default_path = os.path.join(MODELS_DIR, default_name)
        
        # Open save dialog
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Model", default_path, "Model Files (*.pkl);;All Files (*)"
        )
        
        if file_name:
            success, message = self.model.save(file_name)
            if success:
                self.analysis_status_label.setText(f"Model saved successfully to: {file_name}")
                logger.info(f"Model saved to: {file_name}")
            else:
                QMessageBox.critical(self, "Save Error", message)
                self.analysis_status_label.setText("Failed to save model")
    
    def load_model(self):
        """Load a model from file with enhanced support for multiple formats using threaded loading"""
        # Show a file dialog with support for multiple model formats
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Model", 
            MODELS_DIR, 
            "All Model Files (*.pkl *.h5 *.joblib *.sav);;Pickle Files (*.pkl);;HDF5 Files (*.h5);;Joblib Files (*.joblib);;Scikit-learn Files (*.sav);;All Files (*)"
        )
        
        if not file_name:
            return False
        
        # Show progress dialog and load model in thread
        progress_dialog = ModelLoadingDialog(file_name, self)
        result = progress_dialog.exec_()
        
        if result == QDialog.Accepted and progress_dialog.success:
            # Model loaded successfully
            self.model = progress_dialog.model
            
            # Update UI to reflect loaded model
            self._update_ui_for_loaded_model(file_name)
            
            # Update status
            self.analysis_status_label.setText(f"Model loaded: {os.path.basename(file_name)}")
            self.status_bar.showMessage(f"Model loaded successfully from {os.path.basename(file_name)}")
            
            return True
        else:
            # Loading failed or was cancelled
            error_message = progress_dialog.message if not progress_dialog.success else "Loading was cancelled"
            QMessageBox.warning(self, "Model Loading Failed", f"Failed to load model:\n{error_message}")
            return False
    
    def _update_ui_for_loaded_model(self, file_path=""):
        """Update UI components to reflect the loaded model"""
        if not self.model:
            return
            
        # Update model type selector to match loaded model
        model_type = self.model.model_type
        
        if model_type == "isolation_forest":
            self.model_type_combo.setCurrentText("Isolation Forest")
        elif model_type == "lof":
            self.model_type_combo.setCurrentText("Local Outlier Factor")
        elif model_type == "autoencoder":
            self.model_type_combo.setCurrentText("Autoencoder")
        elif model_type == "lstm":
            self.model_type_combo.setCurrentText("LSTM")
        elif model_type == "gru":
            self.model_type_combo.setCurrentText("GRU")
        elif model_type == "prophet":
            self.model_type_combo.setCurrentText("Prophet")
        else:
            self.model_type_combo.setCurrentText("Custom")
            
        # Show selected file name
        if hasattr(self, 'selected_model_label') and file_path:
            self.selected_model_label.setText(f"Selected model: {os.path.basename(file_path)}")
    
    def show_recent_models(self):
        """Show a dialog with recently used models that can be loaded"""
        recent_models = self.model_registry.get_recent_models(10)
        
        if not recent_models:
            QMessageBox.information(self, "Recent Models", "No recently used models found.")
            return
            
        # Create a dialog to show recent models
        dialog = QDialog(self)
        dialog.setWindowTitle("Recent Models")
        dialog.setMinimumSize(500, 300)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select a model to load:"))
        
        # Create a table to display models
        model_table = QTableWidget(len(recent_models), 5)
        model_table.setHorizontalHeaderLabels(["Name", "Type", "Last Used", "Usage Count", "Path"])
        model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        model_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        
        # Fill the table
        for row, (id, name, model_type, last_used, use_count, filepath) in enumerate(recent_models):
            model_table.setItem(row, 0, QTableWidgetItem(name))
            model_table.setItem(row, 1, QTableWidgetItem(model_type))
            model_table.setItem(row, 2, QTableWidgetItem(last_used))
            model_table.setItem(row, 3, QTableWidgetItem(str(use_count)))
            model_table.setItem(row, 4, QTableWidgetItem(filepath))
            # Store the model ID in the first column for reference
            model_table.item(row, 0).setData(Qt.UserRole, id)
        
        layout.addWidget(model_table)
        
        # Add buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load Selected Model")
        cancel_button = QPushButton("Cancel")
        
        button_layout.addWidget(load_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        load_button.clicked.connect(lambda: self._load_selected_model(model_table) and dialog.accept())
        cancel_button.clicked.connect(dialog.reject)
        
        # Show the dialog
        dialog.exec_()
    
    def _load_selected_model(self, table):
        """Load the selected model from the recent models dialog using threaded loading"""
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a model to load.")
            return False
            
        # Get the selected model information
        row = selected_rows[0].row()
        model_id = table.item(row, 0).data(Qt.UserRole)
        filepath = table.item(row, 4).text()
        
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "File Not Found", f"The model file was not found at:\n{filepath}")
            return False
        
        # Show progress dialog and load model in thread
        progress_dialog = ModelLoadingDialog(filepath, self)
        result = progress_dialog.exec_()
        
        if result == QDialog.Accepted and progress_dialog.success:
            # Model loaded successfully
            self.model = progress_dialog.model
            
            # Update UI to reflect loaded model
            self._update_ui_for_loaded_model(filepath)
            
            # Update status
            self.analysis_status_label.setText(f"Model loaded: {os.path.basename(filepath)}")
            self.status_bar.showMessage(f"Model loaded successfully from {os.path.basename(filepath)}")
            
            # Record usage in model registry
            try:
                self.model_registry.record_model_usage(model_id)
            except Exception as e:
                logger.warning(f"Could not record model usage: {str(e)}")
            
            return True
        else:
            # Loading failed or was cancelled
            error_message = progress_dialog.message if not progress_dialog.success else "Loading was cancelled"
            QMessageBox.warning(self, "Model Loading Failed", f"Failed to load model:\n{error_message}")
            return False
    
    def test_on_training_data(self):
        """Test trained models on the training data to verify they work correctly"""
        # Check if we have any trained models
        active_models = {k: v for k, v in self.trained_models.items() if v is not None} if self.trained_models else {}
        
        if not active_models and self.model is None:
            QMessageBox.warning(self, "No Models", "No trained model available. Please train a model first.")
            return
        
        # Check if we have training data
        if not hasattr(self.data_processor, 'preprocessed_data') or self.data_processor.preprocessed_data is None:
            QMessageBox.warning(self, "No Training Data", "No training data available. Please load and train on data first.")
            return
        
        # If we have no models in trained_models but have self.model, add it
        if not active_models and self.model is not None:
            active_models = {'primary': self.model}
        
        logger.info("Testing models on training data...")
        self.analysis_status_label.setText("Testing on training data...")
        
        # Use full training dataframe so model-specific feature engineering can run
        training_data = self.data_processor.preprocessed_data
        numeric_data = training_data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            QMessageBox.warning(self, "No Data", "No numeric features found in training data")
            return
        
        logger.info(f"Training data - Shape: {numeric_data.shape}")
        logger.info(f"Training data sample statistics:")
        for col in list(numeric_data.columns)[:5]:  # Log first 5 columns
            logger.info(f"  {col}: min={numeric_data[col].min():.4f}, max={numeric_data[col].max():.4f}, mean={numeric_data[col].mean():.4f}")
        
        # Run ensemble prediction
        try:
            all_predictions = {}
            failed_models = []
            
            for model_name, model_instance in active_models.items():
                try:
                    logger.info(f"Testing {model_name} model on training data...")
                    
                    # Let each model prepare/align its own features.
                    # This avoids KeyError when engineered columns are expected.
                    result = model_instance.predict(training_data)
                    
                    # Handle different return formats from predict()
                    # Some models return (scores, anomalies) tuple
                    # Some models (like IQR) return predictions array in -1/1 format
                    if result is None:
                        logger.warning(f"{model_name} returned None predictions")
                        failed_models.append(model_name)
                        continue
                    
                    # Check if result is a tuple (scores, anomalies)
                    if isinstance(result, tuple) and len(result) == 2:
                        scores, anomalies = result
                        # Convert boolean anomalies to -1/1 format for consistency
                        if isinstance(anomalies, np.ndarray) and anomalies.dtype == bool:
                            predictions = np.where(anomalies, -1, 1)
                        elif isinstance(anomalies, (list, np.ndarray)):
                            # Already in -1/1 format or numeric
                            predictions = np.array(anomalies)
                            # Convert boolean-like to -1/1 if needed
                            if predictions.dtype == bool:
                                predictions = np.where(predictions, -1, 1)
                        else:
                            logger.warning(f"{model_name} returned unexpected anomaly format: {type(anomalies)}")
                            failed_models.append(model_name)
                            continue
                    else:
                        # Single array return (e.g., IQR model)
                        predictions = np.asarray(result)
                        # Ensure it's in -1/1 format
                        if predictions.dtype == bool:
                            predictions = np.where(predictions, -1, 1)
                        elif not np.all(np.isin(predictions, [-1, 1])):
                            # Convert from other formats (e.g., 0/1, True/False)
                            # Assume > 0 or True means anomaly
                            predictions = np.where(predictions > 0, -1, 1)
                    
                    if len(predictions) > 0:
                        all_predictions[model_name] = predictions
                        anomaly_count = np.sum(predictions == -1)
                        anomaly_pct = (anomaly_count / len(predictions)) * 100 if len(predictions) > 0 else 0
                        logger.info(f"{model_name}: {anomaly_count}/{len(predictions)} anomalies ({anomaly_pct:.2f}%)")
                    else:
                        logger.warning(f"{model_name} returned empty predictions")
                        failed_models.append(model_name)
                        
                except Exception as e:
                    logger.error(f"Error testing {model_name}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    failed_models.append(model_name)
            
            if not all_predictions:
                error_msg = f"All {len(active_models)} models failed to make predictions on training data."
                if failed_models:
                    error_msg += f"\nFailed models: {', '.join(failed_models)}"
                self.analysis_status_label.setText(error_msg)
                QMessageBox.warning(self, "Prediction Failed", error_msg)
                return
            
            # Ensemble voting
            if len(all_predictions) > 1:
                logger.info("Performing ensemble voting on training data predictions...")
                predictions_array = np.array(list(all_predictions.values()))
                ensemble_predictions = np.apply_along_axis(
                    lambda x: -1 if np.sum(x == -1) > len(x) / 2 else 1,
                    axis=0,
                    arr=predictions_array
                )
            else:
                ensemble_predictions = list(all_predictions.values())[0]
            
            # Count anomalies
            total_points = len(ensemble_predictions)
            anomaly_count = np.sum(ensemble_predictions == -1)
            anomaly_percentage = (anomaly_count / total_points) * 100
            
            logger.info(f"Training data test complete: {anomaly_count}/{total_points} anomalies ({anomaly_percentage:.2f}%)")
            
            # **CRITICAL: Store the predictions in the preprocessed_data for visualization**
            # Convert -1/1 format to boolean (True = anomaly)
            anomaly_boolean = (ensemble_predictions == -1)
            self.data_processor.preprocessed_data["Anomaly"] = anomaly_boolean
            
            # Also compute anomaly scores if we have them from individual models
            if all_predictions:
                # For statistical models, use the ensemble decision as score
                # Higher positive values = more likely to be anomaly
                anomaly_scores = np.zeros(total_points)
                for model_name, predictions in all_predictions.items():
                    # Count how many models vote for anomaly at each point
                    anomaly_scores += (predictions == -1).astype(float)
                # Normalize to 0-1 range
                anomaly_scores = anomaly_scores / len(all_predictions)
                self.data_processor.preprocessed_data["Anomaly Score"] = anomaly_scores
                logger.info(f"Stored anomaly results in preprocessed_data: {anomaly_count} anomalies marked")
            
            status_msg = f"Training Data Test: {anomaly_count}/{total_points} anomalies ({anomaly_percentage:.1f}%)"
            if failed_models:
                status_msg += f" | {len(failed_models)} models failed"
            
            self.analysis_status_label.setText(status_msg)
            
            # Show results dialog
            QMessageBox.information(
                self,
                "Training Data Test Results",
                f"Models tested on {total_points:,} training data points\n\n"
                f"Anomalies detected: {anomaly_count:,} ({anomaly_percentage:.2f}%)\n"
                f"Normal points: {total_points - anomaly_count:,} ({100 - anomaly_percentage:.2f}%)\n\n"
                f"Models used: {len(all_predictions)}/{len(active_models)}\n"
                + (f"Failed models: {', '.join(failed_models)}" if failed_models else "All models succeeded")
            )
            
        except Exception as e:
            error_msg = f"Error during training data test: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.analysis_status_label.setText("Training data test failed")
            QMessageBox.critical(self, "Test Error", error_msg)
    
    def predict_anomalies(self):
        file_path = self.prediction_file_input.text()
        if not file_path:
            self.analysis_status_label.setText("Please select a prediction data file")
            return
        
        # Check if we have any trained models
        # First check trained_models dict, then fall back to self.model
        active_models = {k: v for k, v in self.trained_models.items() if v is not None} if self.trained_models else {}
        
        if not active_models and self.model is None:
            self.analysis_status_label.setText("No trained model available. Please train a model first.")
            return
        
        # If we have no models in trained_models but have self.model, add it
        if not active_models and self.model is not None:
            model_name = getattr(self.model, 'model_type', 'Model')
            active_models = {model_name: self.model}
        
        try:
            # Try to load with the current file type selected
            file_type = self.file_type_combo.currentText().lower()
            success = False
            
            if file_type == "csv":
                success, message = self.data_processor.load_csv(file_path)
            elif file_type == "json":
                success, message = self.data_processor.load_json(file_path)
            
            if not success:
                if file_type == "csv":
                    success, message = self.data_processor.load_json(file_path)
                else:
                    success, message = self.data_processor.load_csv(file_path)
            
            if success:
                # Clear preprocessed_data when loading new test data to avoid visualization issues
                # This ensures the visualization uses the new test data, not stale training data
                self.data_processor.preprocessed_data = None
                logger.info("Cleared preprocessed_data to ensure visualization uses new test data")
                
                # Keep full dataframe for model-specific preprocessing/feature engineering
                prediction_data = self.data_processor.data
                numeric_data = prediction_data.select_dtypes(include=['number'])
                
                if numeric_data.empty:
                    self.analysis_status_label.setText("No numeric features found in the data for prediction")
                    return
                
                # Get preprocessing parameters from the first model
                first_model = next(iter(active_models.values()))
                
                logger.info(f"Loaded prediction data - Shape: {numeric_data.shape}")
                logger.info(f"Prediction data sample statistics:")
                for col in list(numeric_data.columns)[:5]:  # Log first 5 columns
                    logger.info(f"  {col}: min={numeric_data[col].min():.4f}, max={numeric_data[col].max():.4f}, mean={numeric_data[col].mean():.4f}")
                
                if numeric_data.empty:
                    self.analysis_status_label.setText("No numeric features found in the data for prediction")
                    return
                
                # Log dataset info
                logger.info(f"Prediction data shape: {numeric_data.shape}")
                logger.info(f"Number of models available: {len(active_models)}")
                
                # Check if we have multiple trained models for ensemble prediction
                if len(active_models) > 1:
                    self.status_bar.showMessage(f"Running ensemble prediction with {len(active_models)} models...")
                    self.analysis_status_label.setText(f"Ensemble prediction with {len(active_models)} models on {len(prediction_data)} rows...")
                    
                    # Run ensemble prediction with active models
                    self._run_ensemble_prediction(prediction_data, active_models)
                else:
                    # Single model prediction
                    single_model = next(iter(active_models.values()))
                    self.status_bar.showMessage("Running prediction...")
                    self.analysis_status_label.setText(f"Predicting anomalies in {len(prediction_data)} rows...")
                    
                    self.worker = Worker("predict", single_model, prediction_data)
                    self.worker.finished.connect(self.on_prediction_completed)
                    self.worker.start()
            else:
                self.analysis_status_label.setText(f"Failed to load data: {message}")
                
        except Exception as e:
            error_msg = f"Error during prediction setup: {str(e)}"
            self.analysis_status_label.setText(error_msg)
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _run_ensemble_prediction(self, numeric_data, active_models=None):
        """Run prediction using multiple models and combine results via ensemble voting"""
        try:
            # Use provided models or fall back to self.trained_models
            models_to_use = active_models if active_models is not None else self.trained_models
            
            if not models_to_use:
                self.analysis_status_label.setText("No models available for ensemble prediction")
                return
            
            all_predictions = []
            all_scores = []
            successful_models = []
            failed_models = []
            model_predictions_dict = {}  # Store per-model results for visualization
            
            # Get predictions from each trained model
            for model_name, model in models_to_use.items():
                if model is None:
                    logger.warning(f"Skipping {model_name}: model is None")
                    failed_models.append(model_name)
                    continue
                
                try:
                    # Update status for each model
                    self.status_bar.showMessage(f"Predicting with {model_name}...")
                    
                    # Get prediction from this model
                    scores, anomalies = model.predict(numeric_data)
                    
                    # Validate results
                    if scores is None or anomalies is None:
                        logger.error(f"{model_name} returned None results")
                        failed_models.append(model_name)
                        continue
                    
                    # Convert to numpy arrays and ensure 1D
                    scores = np.asarray(scores).flatten()
                    anomalies = np.asarray(anomalies).flatten()
                    
                    # Check for empty results
                    if len(scores) == 0 or len(anomalies) == 0:
                        logger.error(f"{model_name} returned empty results")
                        failed_models.append(model_name)
                        continue
                    
                    # Ensure boolean type for anomalies
                    if anomalies.dtype != bool:
                        anomalies = anomalies.astype(bool)
                    
                    # Check if lengths match the data
                    expected_len = len(numeric_data)
                    if len(anomalies) != expected_len:
                        logger.warning(f"{model_name} returned {len(anomalies)} predictions, expected {expected_len}")
                        # Adjust to match expected length
                        if len(anomalies) > expected_len:
                            anomalies = anomalies[:expected_len]
                            scores = scores[:expected_len]
                        else:
                            # Pad with False/0 if too short
                            anomalies = np.pad(anomalies, (0, expected_len - len(anomalies)), constant_values=False)
                            scores = np.pad(scores, (0, expected_len - len(scores)), constant_values=0)
                    
                    all_predictions.append(anomalies)
                    all_scores.append(scores)
                    successful_models.append(model_name)
                    
                    # Store per-model results for detailed visualization
                    model_predictions_dict[model_name] = {
                        'scores': scores.copy(),
                        'anomalies': anomalies.copy(),
                        'anomaly_count': int(np.sum(anomalies)),
                        'mean_score': float(np.mean(scores)),
                        'max_score': float(np.max(scores)),
                        'min_score': float(np.min(scores))
                    }
                    
                    logger.info(f"[OK] {model_name}: {np.sum(anomalies)}/{len(anomalies)} anomalies detected")
                    
                except Exception as e:
                    logger.error(f"[X] Error predicting with {model_name}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    failed_models.append(model_name)
                    continue
            
            # Store per-model predictions for visualization access
            self.model_predictions = model_predictions_dict
            
            if not all_predictions:
                error_msg = f"All {len(models_to_use)} models failed to make predictions"
                if failed_models:
                    error_msg += f": {', '.join(failed_models)}"
                self.analysis_status_label.setText(error_msg)
                logger.error(error_msg)
                return
            
            # Check that all predictions have the same length
            prediction_lengths = [len(pred) for pred in all_predictions]
            if len(set(prediction_lengths)) > 1:
                logger.warning(f"Predictions have different lengths: {prediction_lengths}")
                # Use the minimum length to ensure compatibility
                min_length = min(prediction_lengths)
                all_predictions = [pred[:min_length] for pred in all_predictions]
                all_scores = [score[:min_length] for score in all_scores]
                logger.info(f"Truncated all predictions to length {min_length}")
            
            # Ensemble voting: majority vote
            # Stack arrays properly
            predictions_array = np.vstack(all_predictions)  # Shape: (num_models, num_samples)
            scores_array = np.vstack(all_scores)  # Shape: (num_models, num_samples)
            
            # Count votes for each sample
            votes = np.sum(predictions_array.astype(int), axis=0)
            
            # Majority voting: anomaly if more than half of models agree
            ensemble_anomalies = votes > (len(predictions_array) / 2)
            
            # Average scores across all models
            ensemble_scores = np.mean(scores_array, axis=0)
            
            # Log ensemble results
            num_anomalies = int(np.sum(ensemble_anomalies))
            logger.info(f"=" * 60)
            logger.info(f"ENSEMBLE PREDICTION COMPLETE")
            logger.info(f"=" * 60)
            logger.info(f"Successful models: {len(successful_models)}/{len(models_to_use)}")
            logger.info(f"Models used: {', '.join(successful_models)}")
            if failed_models:
                logger.info(f"Failed models: {', '.join(failed_models)}")
            logger.info(f"Ensemble result: {num_anomalies}/{len(ensemble_anomalies)} anomalies detected")
            logger.info(f"Final shapes - scores: {ensemble_scores.shape}, anomalies: {ensemble_anomalies.shape}")
            logger.info(f"=" * 60)
            
            # Create result tuple
            result = (ensemble_scores, ensemble_anomalies)
            
            # Create detailed message
            success_msg = f"Ensemble prediction with {len(successful_models)}/{len(models_to_use)} models"
            if failed_models:
                success_msg += f" ({len(failed_models)} failed)"
            
            # Call the completion handler with ensemble results
            self.on_prediction_completed(True, success_msg, result)
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.analysis_status_label.setText(f"Ensemble prediction error: {str(e)}")
    
    def on_prediction_completed(self, success, message, result):
        # Reset status bar
        if hasattr(self, 'status_bar') and self.status_bar is not None:
            self.status_bar.showMessage(f"Logged in as {self.current_username} ({self.current_role})")
        
        if success and result is not None:
            try:
                # Safely unpack results
                if isinstance(result, tuple) and len(result) == 2:
                    scores, anomalies = result
                else:
                    logger.error(f"Invalid result format: {type(result)}")
                    self.analysis_status_label.setText("Error: Invalid prediction result format")
                    return

                # Check for empty results early
                if scores is None or anomalies is None:
                    error_msg = "Prediction returned empty results"
                    logger.error(error_msg)
                    self.analysis_status_label.setText(error_msg)
                    QMessageBox.warning(self, "Prediction Warning", 
                                      "No anomalies could be detected in the data. Please check your model configuration.")
                    return

                # Convert scores and anomalies to numpy arrays if they aren't already
                scores = np.asarray(scores) if scores is not None else np.array([])
                anomalies = np.asarray(anomalies) if anomalies is not None else np.array([])

                # Verify data shapes and handle empty arrays
                if len(scores) == 0 or len(anomalies) == 0:
                    error_msg = "Empty prediction results - no data to process"
                    logger.error(error_msg)
                    self.analysis_status_label.setText(error_msg)
                    QMessageBox.warning(self, "Prediction Warning",
                                      "The prediction process returned no results. Please check your input data.")
                    return

                # Convert boolean masks if needed
                if anomalies.dtype != bool:
                    anomalies = anomalies.astype(bool)
                    logger.info("Converted anomalies to boolean type")

                # Rest of the function remains the same
                # Add results to data - ensure lengths align with the DataFrame
                target_len = len(self.data_processor.data.index)

                def _resize_numeric(arr: np.ndarray, tgt: int) -> np.ndarray:
                    n = len(arr)
                    if n == tgt:
                        return arr
                    if n == 0:
                        return np.full(tgt, np.nan, dtype=float)
                    try:
                        # Interpolate/decimate uniformly to match length
                        x_old = np.linspace(0.0, 1.0, n)
                        x_new = np.linspace(0.0, 1.0, tgt)
                        return np.interp(x_new, x_old, arr.astype(float))
                    except Exception:
                        # Fallback to index sampling
                        idx = (np.linspace(0, max(n - 1, 0), tgt)).astype(int)
                        return arr[idx].astype(float)

                def _resize_bool(arr: np.ndarray, tgt: int) -> np.ndarray:
                    n = len(arr)
                    if n == tgt:
                        return arr.astype(bool)
                    if n == 0:
                        return np.zeros(tgt, dtype=bool)
                    if n > tgt:
                        # Compress: any anomaly in the chunk marks the bucket as True
                        chunks = np.array_split(arr.astype(bool), tgt)
                        return np.array([np.any(c) for c in chunks], dtype=bool)
                    else:
                        # Upsample: sample with replacement along the array
                        idx = (np.linspace(0, max(n - 1, 0), tgt)).astype(int)
                        return arr.astype(bool)[idx]

                if len(scores) != target_len or len(anomalies) != target_len:
                    logger.warning(
                        f"Prediction/result length mismatch. scores={len(scores)}, anomalies={len(anomalies)}, index={target_len}. Auto-aligning.")
                scores_aligned = _resize_numeric(scores, target_len)
                anomalies_aligned = _resize_bool(anomalies, target_len)

                self.data_processor.data["Anomaly Score"] = pd.Series(scores_aligned, index=self.data_processor.data.index)
                self.data_processor.data["Anomaly"] = pd.Series(anomalies_aligned, index=self.data_processor.data.index)
                
                # Add per-model predictions if available (from ensemble)
                if hasattr(self, 'model_predictions') and self.model_predictions:
                    logger.info(f"Adding per-model predictions for {len(self.model_predictions)} models to dataframe")
                    for model_name, model_data in self.model_predictions.items():
                        # Align per-model data to match target length
                        model_scores = _resize_numeric(model_data['scores'], target_len)
                        model_anomalies = _resize_bool(model_data['anomalies'], target_len)
                        
                        # Add columns for each model's predictions
                        safe_model_name = model_name.replace(' ', '_').replace('-', '_')
                        self.data_processor.data[f"{safe_model_name}_Score"] = pd.Series(model_scores, index=self.data_processor.data.index)
                        self.data_processor.data[f"{safe_model_name}_Anomaly"] = pd.Series(model_anomalies, index=self.data_processor.data.index)
                    
                    logger.info(f"Successfully added {len(self.model_predictions)} model-specific prediction columns")
                
                # Make sure that the anomaly column is also available in preprocessed_data if it exists
                if self.data_processor.preprocessed_data is not None:
                    # Check if lengths match
                    if len(self.data_processor.preprocessed_data) != len(self.data_processor.data):
                        logger.warning(f"Preprocessed data has different length: {len(self.data_processor.preprocessed_data)} vs {len(self.data_processor.data)}")
                        
                        # Create a mapping between the two datasets to align anomalies
                        # We use the indices of the preprocessed data to align with the raw data
                        try:
                            # Add the anomaly detection results to the preprocessed data, matching by index
                            pre_idx = self.data_processor.preprocessed_data.index
                            raw_idx = self.data_processor.data.index
                            score_series = pd.Series(scores, index=raw_idx)
                            anom_series = pd.Series(anomalies, index=raw_idx)
                            # If indexes are comparable, reindex; otherwise resize to length
                            try:
                                self.data_processor.preprocessed_data["Anomaly Score"] = score_series.reindex(pre_idx)
                                self.data_processor.preprocessed_data["Anomaly"] = anom_series.reindex(pre_idx)
                            except Exception:
                                preproc_len = len(self.data_processor.preprocessed_data)
                                # Fallback: resize to preprocessed length
                                resized_scores = _resize_numeric(np.asarray(scores), preproc_len)
                                resized_anoms = _resize_bool(np.asarray(anomalies), preproc_len)
                                self.data_processor.preprocessed_data["Anomaly Score"] = resized_scores
                                self.data_processor.preprocessed_data["Anomaly"] = resized_anoms
                        except Exception as mapping_error:
                            logger.error(f"Error aligning anomaly results with preprocessed data: {str(mapping_error)}")
                            # Fallback approach - directly assign truncated/padded arrays
                            preproc_len = len(self.data_processor.preprocessed_data)
                            
                            if len(scores) > preproc_len:
                                self.data_processor.preprocessed_data["Anomaly Score"] = scores[:preproc_len]
                            else:
                                self.data_processor.preprocessed_data["Anomaly Score"] = np.pad(
                                    scores, (0, preproc_len - len(scores)), 'constant', constant_values=np.nan)
                                
                            if len(anomalies) > preproc_len:
                                self.data_processor.preprocessed_data["Anomaly"] = anomalies[:preproc_len]
                            else:
                                self.data_processor.preprocessed_data["Anomaly"] = np.pad(
                                    anomalies, (0, preproc_len - len(anomalies)), 'constant', constant_values=False)
                    else:
                        # If lengths match, direct assignment is fine
                        self.data_processor.preprocessed_data["Anomaly Score"] = scores
                        self.data_processor.preprocessed_data["Anomaly"] = anomalies
                    
                    logger.info(f"Added anomaly detection results to preprocessed data")
                    
                # Calculate summary stats - ensure we're working with numeric types
                # Use "pure Python" sum rather than numpy.sum() to avoid type issues
                anomaly_count = sum(1 for val in anomalies if val)
                total_items = len(anomalies)
                
                # Calculate percentage safely
                if total_items > 0:
                    anomaly_percent = (anomaly_count / total_items) * 100
                else:
                    anomaly_percent = 0
                
                # Update UI
                self.analysis_status_label.setText(
                    f"Prediction completed: Found {anomaly_count} anomalies ({anomaly_percent:.2f}%)"
                )
                
                # Update visualization tab model info with prediction results
                self.update_viz_model_info()
                
                # Update data preview
                self.update_data_preview()
                
                # Also update visualization feature list if needed
                if hasattr(self, 'feature_combo'):
                    self.feature_combo.clear()
                    if self.data_processor.data is not None:
                        all_columns = list(self.data_processor.data.columns)
                        self.feature_combo.addItems(all_columns)
                        
                        # Automatically switch to 'value' column for visualization if it exists
                        value_index = self.feature_combo.findText('value')
                        if value_index >= 0:
                            self.feature_combo.setCurrentIndex(value_index)
                
                # Send email alert if anomalies are found
                if anomaly_count > 0:
                    anomaly_data = self.data_processor.data[self.data_processor.data["Anomaly"] == True].copy()
                    self.send_email_alert(anomaly_count, anomaly_data)
                
                # Update System Health tab - notify that new data is available
                self._update_system_health_sync()
                
                # Calculate prediction metrics
                prediction_metrics = {
                    'prediction_time': datetime.datetime.now().isoformat(),
                    'total_samples': len(anomalies),
                    'anomaly_count': int(sum(anomalies)),
                    'anomaly_rate': float(sum(anomalies)) / len(anomalies) * 100
                }
                
                # Add score statistics if available
                if scores is not None:
                    prediction_metrics.update({
                        'mean_score': float(np.mean(scores)),
                        'min_score': float(np.min(scores)),
                        'max_score': float(np.max(scores)),
                        'score_std': float(np.std(scores))
                    })
                
                # Store prediction metrics
                if not hasattr(self.model, 'prediction_metrics'):
                    self.model.prediction_metrics = {}
                self.model.prediction_metrics.update(prediction_metrics)
                
                # Update metrics display if dialog is open
                if hasattr(self, '_metrics_dialog') and self._metrics_dialog.isVisible():
                    self.show_model_metrics()
                
            except Exception as e:
                logger.error(f"Error processing prediction results: {str(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                
                # Add more detailed error logging
                if result is not None:
                    try:
                        scores, anomalies = result
                        logger.error(f"Scores info - Type: {type(scores)}, Length: {len(scores) if hasattr(scores, '__len__') else 'N/A'}")
                        logger.error(f"Anomalies info - Type: {type(anomalies)}, Length: {len(anomalies) if hasattr(anomalies, '__len__') else 'N/A'}")
                        
                        if hasattr(anomalies, '__len__') and len(anomalies) > 0:
                            logger.error(f"First few anomaly items: {anomalies[:5].tolist()}")
                    except Exception as debug_error:
                        logger.error(f"Error during debug logging: {str(debug_error)}")
                
                self.analysis_status_label.setText(f"Error processing prediction results: {str(e)}")
        else:
            error_message = message if isinstance(message, str) else "Unknown error during prediction"
            self.analysis_status_label.setText(f"Prediction failed: {error_message}")
            QMessageBox.warning(self, "Prediction Error", error_message)

    def show_model_metrics(self):
        """Enhanced display of model metrics in a new window"""
        if not hasattr(self, 'model') or not self.model:
            QMessageBox.warning(self, "Model Metrics", "No model available")
            return
            
        # Create or update metrics dialog
        self._metrics_dialog = QDialog(self)
        self._metrics_dialog.setWindowTitle("Model Evaluation Metrics")
        self._metrics_dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(self._metrics_dialog)
        
        # Model info section
        info_group = QGroupBox("Model Information")
        info_layout = QFormLayout()
        info_layout.addRow("Model Type:", QLabel(str(self.model.model_type)))
        if hasattr(self.model, 'feature_columns'):
            info_layout.addRow("Features:", QLabel(str(len(self.model.feature_columns))))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Metrics table
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        
        metrics_table = QTableWidget()
        metrics_table.setColumnCount(2)
        metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Collect all available metrics
        all_metrics = {}
        
        # Add basic metrics
        if hasattr(self.model, 'metrics'):
            all_metrics.update(self.model.metrics)
        
        # Add prediction metrics if available
        if hasattr(self.model, 'prediction_metrics'):
            all_metrics.update(self.model.prediction_metrics)
        
        # Populate table
        metrics_table.setRowCount(len(all_metrics))
        for i, (metric, value) in enumerate(all_metrics.items()):
            metrics_table.setItem(i, 0, QTableWidgetItem(str(metric)))
            
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            metrics_table.setItem(i, 1, QTableWidgetItem(formatted_value))
        
        metrics_layout.addWidget(metrics_table)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda: self.show_model_metrics())
        
        export_btn = QPushButton("Export Metrics")
        export_btn.clicked.connect(lambda: self.export_metrics(all_metrics))
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._metrics_dialog.accept)
        
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(export_btn)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self._metrics_dialog.exec_()

    def export_metrics(self, metrics):
        """Export metrics to a CSV file"""
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Export Metrics", 
                os.path.join(REPORTS_DIR, "model_metrics.csv"),
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_name:
                df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                df.to_csv(file_name, index=False)
                QMessageBox.information(self, "Export Success", 
                                      f"Metrics exported successfully to:\n{file_name}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting metrics: {str(e)}")

    def auto_process_model(self):
        """Automatically run model predictions on loaded data"""
        try:
            if not self.model:
                logger.warning("Auto processing skipped - no model loaded")
                return
                
            if not hasattr(self.data_processor, 'data') or self.data_processor.data is None:
                logger.warning("Auto processing skipped - no data loaded")
                return
            
            # Get the data ready for prediction
            data = self.data_processor.preprocessed_data if self.data_processor.preprocessed_data is not None else self.data_processor.data
            
            # Get only numeric columns for prediction
            numeric_data = data.select_dtypes(include=['number'])
            if numeric_data.empty:
                logger.warning("No numeric features found for auto processing")
                return
            
            logger.info("Starting auto processing with model...")
            self.analysis_status_label.setText("Running auto processing...")
            
            # Create worker for prediction
            self.worker = Worker("predict", self.model, numeric_data)
            # Change to use on_prediction_completed instead of the non-existent method
            self.worker.finished.connect(self.on_prediction_completed)
            self.worker.start()
            
        except Exception as e:
            error_msg = f"Error in auto processing: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.analysis_status_label.setText(error_msg)

    def toggle_auto_processing(self, state):
        """Enable/disable automatic model processing"""
        try:
            if state == Qt.Checked:
                # Check if we have a model loaded
                if self.model is None:
                    QMessageBox.warning(self, "Auto Processing", 
                                      "Please load or train a model first.")
                    self.auto_process_check.setChecked(False)
                    return
                
                # Get the processing interval in milliseconds from hours, minutes, and seconds
                hours = self.process_hours_spin.value()
                minutes = self.process_minutes_spin.value()
                seconds = self.process_seconds_spin.value()
                
                # Check if at least one unit is set
                if hours == 0 and minutes == 0 and seconds == 0:
                    QMessageBox.warning(self, "Auto Processing", 
                                      "Please set at least one time unit (hours, minutes, or seconds).")
                    self.auto_process_check.setChecked(False)
                    return
                
                # Convert to milliseconds
                interval_ms = (hours * 3600 + minutes * 60 + seconds) * 1000
                
                # Start the timer with the specified interval
                if not hasattr(self, 'auto_process_timer'):
                    self.auto_process_timer = QTimer(self)
                    self.auto_process_timer.timeout.connect(self.auto_process_model)
                
                self.auto_process_timer.start(interval_ms)
                
                # Create human-readable interval string
                time_parts = []
                if hours > 0:
                    time_parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if minutes > 0:
                    time_parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
                if seconds > 0:
                    time_parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
                interval_str = ", ".join(time_parts)
                
                # Update status
                self.analysis_status_label.setText(
                    f"Auto processing enabled - processing every {interval_str}"
                )
                logger.info(f"Auto processing enabled with interval: {interval_str}")
            else:
                # Stop the timer
                if hasattr(self, 'auto_process_timer'):
                    self.auto_process_timer.stop()
                self.analysis_status_label.setText("Auto processing disabled")
                logger.info("Auto processing disabled")
                
        except Exception as e:
            logger.error(f"Error toggling auto processing: {str(e)}")
            self.analysis_status_label.setText(f"Error: {str(e)}")
            self.auto_process_check.setChecked(False)

    def browse_prediction_file(self):
        """Browse for file to use for anomaly prediction"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Prediction Data File", "", 
            "Data Files (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_name:
            self.prediction_file_input.setText(file_name)
            self.analysis_status_label.setText(f"Selected prediction file: {file_name}")
            
            # Try to auto-select file format based on extension
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension == '.csv':
                self.file_type_combo.setCurrentText("CSV")
            elif file_extension == '.json':
                self.file_type_combo.setCurrentText("JSON")

    def stop_data_import_monitoring(self):
        """Stop data-import auto-load/process and tab monitoring without exiting the app."""
        try:
            if hasattr(self, "auto_load_timer") and self.auto_load_timer.isActive():
                self.auto_load_timer.stop()
                if hasattr(self, "auto_load_check"):
                    self.auto_load_check.setChecked(False)
            if hasattr(self, "auto_process_timer") and self.auto_process_timer.isActive():
                self.auto_process_timer.stop()
                if hasattr(self, "auto_process_check"):
                    self.auto_process_check.setChecked(False)
            if hasattr(self, "worker") and self.worker.isRunning():
                try:
                    self.worker.terminate()
                    self.worker.wait(3000)
                except Exception:
                    pass
            if getattr(self, "monitoring_active", False) and hasattr(self, "tab_id"):
                CustomMonitoringTab.stop_monitoring(self)
            if hasattr(self, "stop_monitoring_btn"):
                self.stop_monitoring_btn.setEnabled(False)
            if hasattr(self, "data_status_label"):
                self.data_status_label.setText("Monitoring stopped")
            if hasattr(self, "analysis_status_label"):
                self.analysis_status_label.setText("Monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping data import monitoring: {e}")

    def stop_monitoring(self):
        """Stop all monitoring and auto-processing activities, then exit the application"""
        try:
            # First verify if any monitoring is actually active
            monitoring_active = False
            
            if hasattr(self, 'auto_load_timer') and self.auto_load_timer.isActive():
                monitoring_active = True
            if hasattr(self, 'auto_process_timer') and self.auto_process_timer.isActive():
                monitoring_active = True
            if hasattr(self, 'worker') and self.worker.isRunning():
                monitoring_active = True
            
            message = "Stop monitoring and exit the application?"
            if monitoring_active:
                message = "Active monitoring will be stopped.\n\nAre you sure you want to stop monitoring and exit the application?"
            
            reply = QMessageBox.question(
                self, 
                'Stop Monitoring and Exit', 
                message,
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # Stop auto-loading timer if running
                    if hasattr(self, 'auto_load_timer') and self.auto_load_timer.isActive():
                        self.auto_load_timer.stop()
                        self.auto_load_check.setChecked(False)
                        logger.info("Auto-loading stopped successfully")
                    
                    # Stop auto-processing timer if running
                    if hasattr(self, 'auto_process_timer') and self.auto_process_timer.isActive():
                        self.auto_process_timer.stop()
                        self.auto_process_check.setChecked(False)
                        logger.info("Auto-processing stopped successfully")
                    
                    # Cancel any running worker threads
                    if hasattr(self, 'worker') and self.worker.isRunning():
                        try:
                            self.worker.terminate()
                            success = self.worker.wait(3000)  # Wait up to 3 seconds
                            if not success:
                                logger.warning("Worker thread did not terminate gracefully")
                            else:
                                logger.info("Worker thread terminated successfully")
                        except Exception as worker_error:
                            logger.error(f"Error terminating worker thread: {str(worker_error)}")
                    
                    # Save any pending changes or state
                    try:
                        if hasattr(self, 'data_processor') and self.data_processor.data is not None:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_file = os.path.join(DATA_DIR, f"backup_before_exit_{timestamp}.csv")
                            self.data_processor.data.to_csv(backup_file, index=False)
                            logger.info(f"Data backup saved to {backup_file}")
                    except Exception as backup_error:
                        logger.error(f"Error saving data backup: {str(backup_error)}")
                    
                    # Update UI
                    self.stop_monitoring_btn.setEnabled(False)
                    self.analysis_status_label.setText("Monitoring stopped, application closing...")
                    self.data_status_label.setText("Monitoring stopped, application closing...")
                    
                    # Process any pending events before closing
                    QApplication.processEvents()
                    
                    # Log the exit
                    logger.info("Application exit initiated after stopping monitoring")

                    try:
                        loop = getattr(self, "agent_loop", None)
                        if loop is not None:
                            loop.stop()
                            self.agent_loop = None
                    except Exception:
                        pass
                    try:
                        from app.agent import stop_agent_bridge

                        stop_agent_bridge()
                    except Exception:
                        pass
                    self.agent_bridge = None
                    
                    # Close the application
                    QApplication.quit()
                    
                except Exception as shutdown_error:
                    error_msg = f"Error during shutdown process: {str(shutdown_error)}"
                    logger.error(error_msg)
                    QMessageBox.critical(self, "Error", 
                                      f"Error during shutdown:\n{error_msg}\n\nPlease try closing the application manually.")
            
        except Exception as e:
            error_msg = f"Critical error stopping monitoring: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Critical Error", 
                               f"A critical error occurred:\n{error_msg}\n\nPlease try closing the application manually.")

    def clean_data(self):
        """Clean up loaded data and reset monitoring state"""
        try:
            reply = QMessageBox.question(self, 'Clean Data', 
                                       'Are you sure you want to clean all loaded data?\n'
                                       'This will reset all data and monitoring states.',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                if getattr(self, "tab_id", None):
                    self.stop_data_import_monitoring()
                else:
                    self.stop_monitoring()
                
                # Clear data
                if hasattr(self, 'data_processor'):
                    self.data_processor.data = None
                    self.data_processor.preprocessed_data = None
                    self.data_processor.feature_columns = []
                    self.data_processor.timestamp_column = None
                
                # Reset file inputs
                self.file_path_input.clear()
                self.auto_folder_input.clear()
                self.prediction_file_input.clear()
                
                # Clear data preview table
                self.data_table.setRowCount(0)
                self.data_table.setColumnCount(0)
                
                # Clear feature selections
                self.timestamp_combo.clear()
                self.feature_list.setRowCount(0)
                
                # Clear visualization
                if hasattr(self, 'plot_canvas'):
                    self.plot_canvas.axes.clear()
                    self.plot_canvas.draw()
                
                # Reset status labels
                self.data_status_label.setText("All data cleared")
                self.analysis_status_label.setText("Ready")
                self.visualization_status_label.setText("")
                
                # Clear last processed file tracking
                self.last_processed_file = None
                self.last_processed_time = None
                
                logger.info("Data and monitoring state cleaned")
                QMessageBox.information(self, "Clean Complete", 
                                      "All data has been cleared and monitoring state reset.")
                
        except Exception as e:
            error_msg = f"Error cleaning data: {str(e)}"
            logger.error(error_msg)
            QMessageBox.warning(self, "Error", error_msg)

    def show_action_log(self):
        """Show recent actions in a separate window"""
        log_dialog = QDialog(self)
        log_dialog.setWindowTitle("Action Log")
        log_dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(log_dialog)
        
        # Create log viewer
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        
        # Read from log file
        try:
            with open("security.log", "r") as f:
                log_content = f.read()
            log_text.setText(log_content)
        except Exception as e:
            log_text.setText(f"Error reading log file: {str(e)}")
        
        layout.addWidget(log_text)
        
        # Add refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(lambda: self.refresh_log_view(log_text))
        layout.addWidget(refresh_btn)
        
        log_dialog.exec_()
    
    def refresh_log_view(self, log_text):
        """Refresh the log viewer content"""
        try:
            with open("security.log", "r") as f:
                log_content = f.read()
            log_text.setText(log_content)
        except Exception as e:
            log_text.setText(f"Error reading log file: {str(e)}")

    def plot_feature(self):
        """Enhanced plot method with multiple visualization types"""
        try:
            # Get current selections
            feature = self.feature_combo.currentText()
            viz_type = self.viz_type_combo.currentText()
            time_range = self.time_range_combo.currentText()
            
            if not feature and viz_type == "Time Series with Anomalies":
                self.visualization_status_label.setText("Please select a feature to plot")
                return
            
            # Determine which dataset to use
            # Check both data and preprocessed_data for anomaly results
            data = None
            data_source = "none"
            
            # Priority 1: preprocessed_data with Anomaly column (training data test results)
            if (self.data_processor.preprocessed_data is not None and 
                "Anomaly" in self.data_processor.preprocessed_data.columns):
                anomaly_count_prep = self.data_processor.preprocessed_data["Anomaly"].sum()
                if anomaly_count_prep > 0:  # Has actual anomalies marked
                    data = self.data_processor.preprocessed_data
                    data_source = "preprocessed_data with anomalies"
            
            # Priority 2: regular data with Anomaly column (new data test results)
            if (data is None and 
                self.data_processor.data is not None and 
                "Anomaly" in self.data_processor.data.columns):
                anomaly_count_data = self.data_processor.data["Anomaly"].sum()
                if anomaly_count_data > 0:  # Has actual anomalies marked
                    data = self.data_processor.data
                    data_source = "data with anomalies"
            
            # Priority 3: preprocessed_data without anomalies
            if data is None and self.data_processor.preprocessed_data is not None:
                data = self.data_processor.preprocessed_data
                data_source = "preprocessed_data"
            
            # Priority 4: raw data
            if data is None:
                data = self.data_processor.data
                data_source = "raw data"
            
            logger.info(f"Visualization using: {data_source}")
            
            if data is None:
                self.visualization_status_label.setText("No data available for plotting")
                return
            
            # Log anomaly information if available
            if "Anomaly" in data.columns:
                # Handle different formats
                anomaly_col = data["Anomaly"]
                if anomaly_col.dtype == bool:
                    anomaly_count = anomaly_col.sum()
                elif anomaly_col.dtype in [np.int32, np.int64, int]:
                    anomaly_count = (anomaly_col == -1).sum()
                else:
                    anomaly_count = anomaly_col.astype(bool).sum()
                logger.info(f"Dataset has {anomaly_count} anomalies out of {len(data)} points")
            
            # Clear previous plot and set up subplots if needed
            self.plot_canvas.fig.clear()
            
            # Apply time range filtering
            filtered_data = self._apply_time_range_filter(data, time_range)
            
            # Generate the appropriate visualization
            if viz_type == "Time Series with Anomalies":
                self._plot_time_series_with_anomalies(filtered_data, feature)
            elif viz_type == "Anomaly Score Distribution":
                self._plot_anomaly_distribution(filtered_data)
            elif viz_type == "Feature Correlation Heatmap":
                self._plot_correlation_heatmap(filtered_data)
            elif viz_type == "Statistical Summary":
                self._plot_statistical_summary(filtered_data, feature)
            elif viz_type == "Anomaly Timeline":
                self._plot_anomaly_timeline(filtered_data)
            
            # Apply final formatting
            self.plot_canvas.fig.tight_layout()
            self.plot_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error in plot_feature: {str(e)}")
            self.visualization_status_label.setText(f"Error generating plot: {str(e)}")
    
    def _apply_time_range_filter(self, data, time_range):
        """Apply time range filtering to data"""
        if time_range == "All Data":
            return data
        if time_range == "Last 100 Points":
            return data.tail(100)
        if time_range == "Last 500 Points":
            return data.tail(500)
        if time_range == "Last 1000 Points":
            return data.tail(1000)
        if time_range == "Custom Range":
            # No date picker wired yet — keep full series and note in status
            if hasattr(self, "visualization_status_label"):
                self.visualization_status_label.setText(
                    "Custom Range not configured — showing All Data. Use Last N Points for now."
                )
            return data
        return data

    def _sync_trained_models_for_viz(self):
        """Map custom-tab ``self.models`` into ``trained_models`` for comparison plots."""
        tab_models = getattr(self, "models", None)
        if not isinstance(tab_models, dict) or not tab_models:
            return
        synced = {}
        for mid, model in tab_models.items():
            if model is None:
                continue
            base = str(getattr(model, "model_type", None) or mid)[:48]
            name = base
            n = 2
            while name in synced:
                name = f"{base} ({n})"
                n += 1
            synced[name] = model
        if synced:
            self.trained_models = synced
            if self.model is None:
                self.model = next(iter(synced.values()))
    
    def _plot_time_series_with_anomalies(self, data, feature):
        """Plot clean time series with anomalies"""
        if feature not in data.columns:
            self.visualization_status_label.setText(f"Feature '{feature}' not found in data")
            return
        
        ax = self.plot_canvas.fig.add_subplot(111)
        
        # Get x-axis data
        if self.data_processor.timestamp_column and self.data_processor.timestamp_column in data.columns:
            x = pd.to_datetime(data[self.data_processor.timestamp_column])
            x_label = "Time"
        else:
            x = np.arange(len(data))
            x_label = "Data Points"
        
        y = pd.to_numeric(data[feature], errors="coerce")
        smooth = bool(
            hasattr(self, "smooth_lines_check") and self.smooth_lines_check.isChecked()
        )
        if smooth and len(y) >= 5:
            window = max(3, min(21, len(y) // 25 * 2 + 1))
            y_plot = y.rolling(window=window, center=True, min_periods=1).mean()
            label = f"{feature} (smoothed)"
            linewidth = 1.8
            alpha = 0.9
        else:
            y_plot = y
            label = feature
            linewidth = 1.0
            alpha = 0.75

        ax.plot(
            x,
            y_plot,
            label=label,
            color="#1f77b4",
            linewidth=linewidth,
            alpha=alpha,
            linestyle="-",
        )
        
        # Add anomalies if requested and available
        if self.highlight_anomalies_check.isChecked() and "Anomaly" in data.columns:
            # Handle both boolean and -1/1 format
            anomaly_col = data["Anomaly"]
            
            # Convert to boolean mask - handle multiple formats
            if anomaly_col.dtype == bool:
                anomaly_mask = anomaly_col  # Already boolean
            elif anomaly_col.dtype in [np.int32, np.int64, int]:
                # -1 = anomaly, 1 = normal (sklearn convention)
                anomaly_mask = (anomaly_col == -1)
            else:
                # Try to convert to boolean
                anomaly_mask = anomaly_col.astype(bool)
            
            anomaly_count = anomaly_mask.sum()
            logger.info(f"Plotting {anomaly_count} anomalies out of {len(data)} points")
            
            if anomaly_count > 0:
                logger.info(f"Visualization: Adding {anomaly_count} red dots for anomalies")
                # Ensure proper alignment between x-axis and anomaly data
                # Convert to numpy arrays to avoid index alignment issues
                if isinstance(x, pd.Series):
                    x_values = x.values
                else:
                    x_values = np.asarray(x)
                
                if isinstance(anomaly_mask, pd.Series):
                    mask_array = anomaly_mask.values
                else:
                    mask_array = np.asarray(anomaly_mask)
                # Scatter on raw feature values (not smoothed) so anomalies stay accurate
                feature_values = y.values if hasattr(y, "values") else np.asarray(y)
                anomaly_x = x_values[mask_array]
                anomaly_y = feature_values[mask_array]
                
                logger.info(f"Anomaly positions: x range [{anomaly_x.min()} to {anomaly_x.max()}], y range [{anomaly_y.min():.2f} to {anomaly_y.max():.2f}]")
                
                ax.scatter(anomaly_x, 
                          anomaly_y,
                          color='red',
                          marker='o',
                          s=60,
                          label=f'Anomalies ({anomaly_count})',
                          zorder=5,
                          edgecolors='darkred',
                          linewidth=1)
            else:
                logger.warning("No anomalies to plot - anomaly_mask has no True values")
        
        # Customize plot
        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{feature} Value', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature} - Time Series with Anomalies', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9, fancybox=True, shadow=True)
        
        # Format x-axis for datetime
        if pd.api.types.is_datetime64_any_dtype(x):
            self.plot_canvas.fig.autofmt_xdate()
        
        # Update status
        anomaly_count = data["Anomaly"].sum() if "Anomaly" in data.columns else 0
        status_msg = f"Time series plot: {len(data)} points, {anomaly_count} anomalies highlighted"
        self.visualization_status_label.setText(status_msg)
    
    def _plot_anomaly_distribution(self, data):
        """Plot distribution of anomaly scores"""
        if "Anomaly Score" not in data.columns:
            self.visualization_status_label.setText("No anomaly scores available for distribution plot")
            return
        
        scores = data["Anomaly Score"].dropna()
        if len(scores) == 0:
            self.visualization_status_label.setText("No valid anomaly scores found")
            return
        
        ax = self.plot_canvas.fig.add_subplot(111)
        
        # Create histogram
        n_bins = min(50, len(scores) // 10) if len(scores) > 100 else 20
        counts, bins, patches = ax.hist(scores, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_score = scores.mean()
        std_score = scores.std()
        median_score = scores.median()
        
        ax.axvline(mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.3f}')
        ax.axvline(median_score, color='green', linestyle='--', label=f'Median: {median_score:.3f}')
        
        # Add threshold if available
        if hasattr(self.model, 'reconstruction_error_threshold') and self.model.reconstruction_error_threshold:
            ax.axvline(self.model.reconstruction_error_threshold, 
                      color='orange', 
                      linestyle='-', 
                      linewidth=2,
                      label=f'Threshold: {self.model.reconstruction_error_threshold:.3f}')
        
        ax.set_xlabel('Anomaly Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Distribution of Anomaly Scores', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        self.visualization_status_label.setText(
            f"Score distribution: μ={mean_score:.3f}, σ={std_score:.3f}, n={len(scores)}"
        )
    
    def _plot_correlation_heatmap(self, data):
        """Plot correlation heatmap of numeric features"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            self.visualization_status_label.setText("Need at least 2 numeric features for correlation plot")
            return
        
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        ax = self.plot_canvas.fig.add_subplot(111)
        
        # Create heatmap
        im = ax.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(correlation_matrix.columns)))
        ax.set_yticks(np.arange(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Add colorbar
        cbar = self.plot_canvas.fig.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=10)
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                value = correlation_matrix.iloc[i, j]
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color='white' if abs(value) > 0.5 else 'black', fontsize=8)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
        
        self.visualization_status_label.setText(
            f"Correlation heatmap: {len(correlation_matrix.columns)} features analyzed"
        )
    
    def _plot_statistical_summary(self, data, feature):
        """Plot statistical summary of selected feature"""
        if feature not in data.columns:
            self.visualization_status_label.setText(f"Feature '{feature}' not found")
            return
        
        feature_data = data[feature].dropna()
        if len(feature_data) == 0:
            self.visualization_status_label.setText(f"No valid data for feature '{feature}'")
            return
        
        # Create subplots
        fig = self.plot_canvas.fig
        ax1 = fig.add_subplot(221)  # Histogram
        ax2 = fig.add_subplot(222)  # Box plot
        ax3 = fig.add_subplot(223)  # Q-Q plot
        ax4 = fig.add_subplot(224)  # Time series
        
        # Histogram
        ax1.hist(feature_data, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_title(f'{feature} - Distribution')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(feature_data)
        ax2.set_title(f'{feature} - Box Plot')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot (approximate)
        from scipy import stats
        stats.probplot(feature_data, dist="norm", plot=ax3)
        ax3.set_title(f'{feature} - Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # Time series
        x = np.arange(len(feature_data))
        ax4.plot(x, feature_data, alpha=0.7)
        ax4.set_title(f'{feature} - Time Series')
        ax4.set_xlabel('Index')
        ax4.set_ylabel('Value')
        ax4.grid(True, alpha=0.3)
        
        # Calculate statistics
        mean_val = feature_data.mean()
        std_val = feature_data.std()
        median_val = feature_data.median()
        
        self.visualization_status_label.setText(
            f"Statistics: μ={mean_val:.3f}, σ={std_val:.3f}, median={median_val:.3f}"
        )
    
    def _plot_anomaly_timeline(self, data):
        """Plot timeline of anomaly occurrences"""
        if "Anomaly" not in data.columns:
            self.visualization_status_label.setText("No anomaly data available for timeline")
            return
        
        anomalies = data[data["Anomaly"] == True]
        if len(anomalies) == 0:
            self.visualization_status_label.setText("No anomalies found in data")
            return
        
        ax = self.plot_canvas.fig.add_subplot(111)
        
        # Get time data
        if self.data_processor.timestamp_column and self.data_processor.timestamp_column in anomalies.columns:
            x = pd.to_datetime(anomalies[self.data_processor.timestamp_column])
            x_label = "Time"
        else:
            x = anomalies.index
            x_label = "Index"
        
        # Create timeline plot
        y = np.ones(len(anomalies))  # All at same height
        colors = ['red' if 'Anomaly Score' not in anomalies.columns 
                 else plt.cm.Reds(score/anomalies['Anomaly Score'].max()) 
                 for score in (anomalies['Anomaly Score'] if 'Anomaly Score' in anomalies.columns 
                              else [1]*len(anomalies))]
        
        ax.scatter(x, y, c=colors, s=60, alpha=0.7, edgecolors='darkred')
        
        # Add severity levels if score available
        if "Anomaly Score" in anomalies.columns:
            scores = anomalies["Anomaly Score"]
            ax2 = ax.twinx()
            ax2.plot(x, scores, color='orange', alpha=0.5, marker='o', linestyle='-', markersize=4)
            ax2.set_ylabel('Anomaly Score', color='orange')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Anomaly Occurrences')
        ax.set_title(f'Anomaly Timeline - {len(anomalies)} Anomalies Detected')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for datetime
        if pd.api.types.is_datetime64_any_dtype(x):
            self.plot_canvas.fig.autofmt_xdate()
        
        self.visualization_status_label.setText(
            f"Timeline: {len(anomalies)} anomalies over {len(data)} data points"
        )
    
    def plot_model_comparison(self):
        """Plot comparison of multiple trained models"""
        try:
            self._sync_trained_models_for_viz()
            self.update_viz_model_info()

            if not getattr(self, "trained_models", None):
                self.visualization_status_label.setText(
                    "No trained models available for comparison. Train models on this tab first."
                )
                return

            valid_models = {
                name: model
                for name, model in self.trained_models.items()
                if model is not None
            }
            if not valid_models:
                self.visualization_status_label.setText("No valid trained models found")
                return

            data = (
                self.data_processor.preprocessed_data
                if self.data_processor.preprocessed_data is not None
                else self.data_processor.data
            )
            if data is None:
                self.visualization_status_label.setText("No data loaded for comparison")
                return
            if "Anomaly" not in data.columns and "Anomaly Score" not in data.columns:
                self.visualization_status_label.setText(
                    "No anomaly results yet. Use Test Models / Predict before Compare Models."
                )
                return
            # Ensure Anomaly column exists for count/agreement helpers
            if "Anomaly" not in data.columns and "Anomaly Score" in data.columns:
                data = data.copy()
                data["Anomaly"] = (pd.to_numeric(data["Anomaly Score"], errors="coerce") > 0.5).astype(int)

            comparison_type = self.viz_comparison_type.currentText()
            self.plot_canvas.fig.clear()

            if comparison_type == "Model Detection Counts":
                self._plot_model_detection_counts(valid_models, data)
            elif comparison_type == "Model Agreement Matrix":
                self._plot_model_agreement_matrix(valid_models, data)
            elif comparison_type == "Detection Overlap Venn":
                self._plot_detection_overlap(valid_models, data)
            elif comparison_type == "Individual Model Predictions":
                self._plot_individual_predictions(valid_models, data)
            elif comparison_type == "Score Distributions":
                self._plot_score_distributions(valid_models, data)
            elif comparison_type == "Confidence Comparison":
                self._plot_confidence_comparison(valid_models, data)
            elif comparison_type == "Performance Metrics":
                self._plot_performance_metrics(valid_models, data)

            self.plot_canvas.fig.tight_layout()
            self.plot_canvas.draw()

        except Exception as e:
            logger.error(f"Error in plot_model_comparison: {str(e)}")
            self.visualization_status_label.setText(f"Error comparing models: {str(e)}")
    
    def _plot_model_detection_counts(self, models, data):
        """Bar chart comparing anomaly detection counts across models - ENHANCED"""
        ax = self.plot_canvas.fig.add_subplot(111)
        
        model_names = list(models.keys())
        detection_counts = []
        
        # Get real detection counts from per-model predictions if available
        if hasattr(self, 'model_predictions') and self.model_predictions:
            for model_name in model_names:
                if model_name in self.model_predictions:
                    detection_counts.append(self.model_predictions[model_name]['anomaly_count'])
                else:
                    # Fallback: check dataframe columns
                    safe_name = model_name.replace(' ', '_').replace('-', '_')
                    anomaly_col = f"{safe_name}_Anomaly"
                    if anomaly_col in data.columns:
                        detection_counts.append(int(data[anomaly_col].sum()))
                    else:
                        detection_counts.append(0)
        else:
            # Fallback to checking dataframe columns
            for model_name in model_names:
                safe_name = model_name.replace(' ', '_').replace('-', '_')
                anomaly_col = f"{safe_name}_Anomaly"
                if anomaly_col in data.columns:
                    detection_counts.append(int(data[anomaly_col].sum()))
                else:
                    detection_counts.append(0)
        
        # If no per-model data found, show ensemble result
        if not any(detection_counts):
            ensemble_anomalies = data["Anomaly"].sum() if "Anomaly" in data.columns else 0
            detection_counts = [ensemble_anomalies] * len(model_names)
        
        # Create bar chart with distinct colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        bar_colors = [colors[i % len(colors)] for i in range(len(model_names))]
        
        bars = ax.bar(range(len(model_names)), detection_counts, 
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Customize plot
        ax.set_xlabel('Models', fontsize=11, fontweight='bold')
        ax.set_ylabel('Anomalies Detected', fontsize=11, fontweight='bold')
        ax.set_title(f'Anomaly Detection Comparison - {len(model_names)} Models', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars with percentage
        total_points = len(data)
        for i, (bar, count) in enumerate(zip(bars, detection_counts)):
            height = bar.get_height()
            percentage = (count / total_points * 100) if total_points > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add ensemble comparison line if available
        if "Anomaly" in data.columns:
            ensemble_count = int(data["Anomaly"].sum())
            ax.axhline(y=ensemble_count, color='red', linestyle='--', linewidth=2, 
                      label=f'Ensemble: {ensemble_count}', alpha=0.7)
            ax.legend(loc='upper right', fontsize=9)
        
        self.visualization_status_label.setText(
            f"Comparison: {len(model_names)} models analyzed, Total data points: {total_points}"
        )
    
    def _plot_model_agreement_matrix(self, models, data):
        """Heatmap showing agreement between models - ENHANCED"""
        ax = self.plot_canvas.fig.add_subplot(111)
        
        model_names = list(models.keys())
        n_models = len(model_names)
        
        # Create agreement matrix based on actual predictions
        agreement_matrix = np.zeros((n_models, n_models))
        
        # Get per-model anomaly predictions
        model_anomalies = []
        for model_name in model_names:
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            anomaly_col = f"{safe_name}_Anomaly"
            
            if anomaly_col in data.columns:
                model_anomalies.append(data[anomaly_col].values.astype(bool))
            elif hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                model_anomalies.append(self.model_predictions[model_name]['anomalies'])
            else:
                # No data available for this model
                model_anomalies.append(np.zeros(len(data), dtype=bool))
        
        # Calculate pairwise agreement (Jaccard similarity)
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    agreement_matrix[i, j] = 1.0  # Perfect agreement with self
                else:
                    # Jaccard similarity: intersection / union
                    intersection = np.sum(model_anomalies[i] & model_anomalies[j])
                    union = np.sum(model_anomalies[i] | model_anomalies[j])
                    agreement_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        # Plot heatmap
        im = ax.imshow(agreement_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(model_names, fontsize=9)
        ax.set_title('Model Agreement Matrix (Jaccard Similarity)', fontsize=13, fontweight='bold')
        
        # Add text annotations with agreement values
        for i in range(n_models):
            for j in range(n_models):
                text_color = 'white' if agreement_matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                       ha='center', va='center', color=text_color, fontweight='bold', fontsize=9)
        
        # Add colorbar with label
        cbar = self.plot_canvas.fig.colorbar(im, ax=ax)
        cbar.set_label('Agreement Level (0=No Agreement, 1=Perfect Agreement)', 
                      rotation=270, labelpad=20, fontsize=10)
        
        # Calculate average agreement
        avg_agreement = np.mean(agreement_matrix[np.triu_indices(n_models, k=1)])
        
        self.visualization_status_label.setText(
            f"Agreement matrix: {n_models} models, Average agreement: {avg_agreement:.2%}"
        )
    
    def _plot_detection_overlap(self, models, data):
        """Show overlap in anomaly detection between models - ENHANCED"""
        model_names = list(models.keys())
        n_models = len(model_names)
        
        # Get per-model anomaly counts
        model_counts = []
        model_anomalies = []
        
        for model_name in model_names:
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            anomaly_col = f"{safe_name}_Anomaly"
            
            if anomaly_col in data.columns:
                anomalies = data[anomaly_col].values.astype(bool)
                model_anomalies.append(anomalies)
                model_counts.append(int(np.sum(anomalies)))
            elif hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                anomalies = self.model_predictions[model_name]['anomalies']
                model_anomalies.append(anomalies)
                model_counts.append(self.model_predictions[model_name]['anomaly_count'])
            else:
                model_anomalies.append(np.zeros(len(data), dtype=bool))
                model_counts.append(0)
        
        ensemble_anomalies = data["Anomaly"].sum() if "Anomaly" in data.columns else 0
        
        # Create subplots for better visualization
        if n_models <= 3:
            # Venn diagram style for up to 3 models
            ax = self.plot_canvas.fig.add_subplot(111)
            
            # Calculate overlaps
            if n_models >= 2:
                # All models agree
                all_agree = np.all(model_anomalies, axis=0).sum()
                # Any model detects
                any_detect = np.any(model_anomalies, axis=0).sum()
                # Only one model
                only_one = sum([np.sum(model_anomalies[i] & ~np.any([model_anomalies[j] for j in range(n_models) if j != i], axis=0)) 
                               for i in range(n_models)])
                
                categories = ['All Models\nAgree', 'Any Model\nDetects', 'Single Model\nOnly', 'Ensemble\nResult']
                counts = [all_agree, any_detect, only_one, ensemble_anomalies]
                colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
                
                bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    percentage = (count / len(data) * 100) if len(data) > 0 else 0
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(count)}\n({percentage:.1f}%)',
                           ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                ax.set_ylabel('Number of Anomalies', fontsize=11, fontweight='bold')
                ax.set_title(f'Detection Overlap Analysis - {n_models} Models', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                
                self.visualization_status_label.setText(
                    f"Overlap: All agree={all_agree}, Any detect={any_detect}, Total points={len(data)}"
                )
            else:
                # Single model - show pie chart
                sizes = [model_counts[0], len(data) - model_counts[0]]
                labels = [f'Anomalies\n({model_counts[0]})', f'Normal\n({len(data) - model_counts[0]})']
                colors = ['#ff6b6b', '#4ecdc4']
                explode = (0.1, 0)
                
                ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                       shadow=True, startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
                ax.set_title(f'Detection Results - {model_names[0]}', fontsize=13, fontweight='bold')
                
                self.visualization_status_label.setText(
                    f"Single model detection: {model_counts[0]}/{len(data)} anomalies"
                )
        else:
            # For more than 3 models, show grouped bar chart
            ax = self.plot_canvas.fig.add_subplot(111)
            
            x = np.arange(len(model_names))
            width = 0.35
            
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            bar_colors = [colors_list[i % len(colors_list)] for i in range(len(model_names))]
            
            bars = ax.bar(x, model_counts, width, label='Individual Models', 
                         color=bar_colors, alpha=0.8, edgecolor='black')
            
            # Add ensemble line
            ax.axhline(y=ensemble_anomalies, color='red', linestyle='--', linewidth=2,
                      label=f'Ensemble: {ensemble_anomalies}', alpha=0.7)
            
            ax.set_xlabel('Models', fontsize=11, fontweight='bold')
            ax.set_ylabel('Anomalies Detected', fontsize=11, fontweight='bold')
            ax.set_title(f'Model Detection Overlap - {n_models} Models', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add value labels
            for bar, count in zip(bars, model_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            self.visualization_status_label.setText(
                f"Detection overlap: {n_models} models analyzed, Ensemble={ensemble_anomalies}"
            )
    
    def _plot_individual_predictions(self, models, data):
        """Plot individual model predictions over time - ENHANCED"""
        ax = self.plot_canvas.fig.add_subplot(111)
        
        model_names = list(models.keys())
        
        # Get x-axis
        if self.data_processor.timestamp_column and self.data_processor.timestamp_column in data.columns:
            x = pd.to_datetime(data[self.data_processor.timestamp_column])
            x_label = "Time"
        else:
            x = np.arange(len(data))
            x_label = "Data Points"
        
        # Define colors for each model
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        # Plot each model's predictions
        plotted_models = 0
        for idx, model_name in enumerate(model_names):
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            anomaly_col = f"{safe_name}_Anomaly"
            
            anomaly_mask = None
            if anomaly_col in data.columns:
                anomaly_mask = data[anomaly_col].values.astype(bool)
            elif hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                anomaly_mask = self.model_predictions[model_name]['anomalies']
            
            if anomaly_mask is not None and np.any(anomaly_mask):
                # Plot with different y-offsets for visibility
                y_offset = idx * 0.15
                color = colors_list[idx % len(colors_list)]
                marker = markers[idx % len(markers)]
                
                ax.scatter(x[anomaly_mask], 
                          np.ones(np.sum(anomaly_mask)) + y_offset,
                          label=f'{model_name} ({np.sum(anomaly_mask)})',
                          color=color, 
                          marker=marker, 
                          s=40, 
                          alpha=0.7,
                          edgecolors='black',
                          linewidth=0.5)
                plotted_models += 1
        
        # Plot ensemble result at the top
        if "Anomaly" in data.columns:
            anomaly_mask = data["Anomaly"] == True
            if np.any(anomaly_mask):
                ax.scatter(x[anomaly_mask], 
                          np.ones(np.sum(anomaly_mask)) + (len(model_names) * 0.15),
                          label=f'Ensemble ({np.sum(anomaly_mask)})', 
                          color='red', 
                          marker='*', 
                          s=100, 
                          alpha=0.8,
                          edgecolors='darkred',
                          linewidth=1,
                          zorder=10)
        
        # Customize plot
        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Model Predictions (offset for visibility)', fontsize=11, fontweight='bold')
        ax.set_title(f'Individual Model Predictions - {len(model_names)} Models', 
                    fontsize=13, fontweight='bold')
        
        # Set y-axis limits with some padding
        ax.set_ylim(0.5, (len(model_names) + 1) * 0.15 + 1.5)
        
        # Remove y-ticks as they're just for offset visualization
        ax.set_yticks([])
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, 
                 framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Format x-axis for datetime
        if pd.api.types.is_datetime64_any_dtype(x):
            self.plot_canvas.fig.autofmt_xdate()
        
        self.visualization_status_label.setText(
            f"Individual predictions from {plotted_models} models displayed with vertical offset for clarity"
        )
    
    def _plot_score_distributions(self, models, data):
        """Plot distribution of anomaly scores for each model - NEW"""
        model_names = list(models.keys())
        n_models = len(model_names)

        # Reserve one extra panel for ensemble when available.
        include_ensemble_panel = "Anomaly Score" in data.columns and n_models < 6
        total_panels = n_models + (1 if include_ensemble_panel else 0)

        # Create subplot grid sized for all panels.
        n_cols = min(3, max(1, total_panels))
        n_rows = (total_panels + n_cols - 1) // n_cols
        
        fig = self.plot_canvas.fig
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, model_name in enumerate(model_names):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            
            # Get model scores
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            score_col = f"{safe_name}_Score"
            
            scores = None
            if score_col in data.columns:
                scores = data[score_col].dropna().values
            elif hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                scores = self.model_predictions[model_name]['scores']
            
            if scores is not None and len(scores) > 0:
                color = colors_list[idx % len(colors_list)]
                
                # Plot histogram
                n_bins = min(50, len(scores) // 10) if len(scores) > 100 else 20
                ax.hist(scores, bins=n_bins, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
                
                # Add statistics lines
                mean_score = np.mean(scores)
                median_score = np.median(scores)
                
                ax.axvline(mean_score, color='red', linestyle='--', linewidth=1.5, 
                          label=f'μ={mean_score:.3f}', alpha=0.8)
                ax.axvline(median_score, color='green', linestyle='--', linewidth=1.5,
                          label=f'Med={median_score:.3f}', alpha=0.8)
                
                ax.set_title(model_name, fontsize=9, fontweight='bold')
                ax.set_xlabel('Score', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                ax.legend(fontsize=7, loc='upper right')
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=7)
        
        # Add ensemble comparison if available
        if include_ensemble_panel:
            ax = fig.add_subplot(n_rows, n_cols, n_models + 1)
            ensemble_scores = data["Anomaly Score"].dropna().values
            
            if len(ensemble_scores) > 0:
                n_bins = min(50, len(ensemble_scores) // 10) if len(ensemble_scores) > 100 else 20
                ax.hist(ensemble_scores, bins=n_bins, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
                
                mean_score = np.mean(ensemble_scores)
                ax.axvline(mean_score, color='darkred', linestyle='--', linewidth=1.5,
                          label=f'μ={mean_score:.3f}', alpha=0.8)
                
                ax.set_title('Ensemble', fontsize=9, fontweight='bold')
                ax.set_xlabel('Score', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.2)
                ax.tick_params(labelsize=7)
        
        self.visualization_status_label.setText(
            f"Score distributions for {n_models} models with statistical measures"
        )
    
    def _plot_confidence_comparison(self, models, data):
        """Compare confidence levels across models - NEW"""
        ax = self.plot_canvas.fig.add_subplot(111)
        
        model_names = list(models.keys())
        
        # Collect score statistics for each model
        model_stats = []
        for model_name in model_names:
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            score_col = f"{safe_name}_Score"
            
            scores = None
            if score_col in data.columns:
                scores = data[score_col].dropna().values
            elif hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                scores = self.model_predictions[model_name]['scores']
            
            if scores is not None and len(scores) > 0:
                model_stats.append({
                    'name': model_name,
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'q25': np.percentile(scores, 25),
                    'median': np.median(scores),
                    'q75': np.percentile(scores, 75),
                    'min': np.min(scores),
                    'max': np.max(scores)
                })
        
        if not model_stats:
            ax.text(0.5, 0.5, 'No score data available', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            return
        
        # Create box plot
        positions = np.arange(len(model_stats))
        box_data = []
        labels = []
        
        for stat in model_stats:
            safe_name = stat['name'].replace(' ', '_').replace('-', '_')
            score_col = f"{safe_name}_Score"
            
            if score_col in data.columns:
                scores = data[score_col].dropna().values
                box_data.append(scores)
                labels.append(f"{stat['name']}\n(n={len(scores)})")
        
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
            
            # Color the boxes
            colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for patch, color in zip(bp['boxes'], [colors_list[i % len(colors_list)] for i in range(len(box_data))]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Anomaly Score', fontsize=11, fontweight='bold')
            ax.set_title('Model Confidence Comparison (Box Plot)', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            self.visualization_status_label.setText(
                f"Confidence comparison: {len(model_stats)} models, Box plot shows quartiles and outliers"
            )
    
    def _plot_performance_metrics(self, models, data):
        """Compare performance metrics across models - NEW"""
        model_names = list(models.keys())
        
        # Collect metrics
        metrics_data = {
            'Model': [],
            'Anomalies Detected': [],
            'Detection Rate (%)': [],
            'Mean Score': [],
            'Max Score': [],
            'Score Variance': []
        }
        
        for model_name in model_names:
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            score_col = f"{safe_name}_Score"
            anomaly_col = f"{safe_name}_Anomaly"
            
            scores = None
            anomalies = None
            
            if score_col in data.columns:
                scores = data[score_col].dropna().values
            elif hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                scores = self.model_predictions[model_name]['scores']
            
            if anomaly_col in data.columns:
                anomalies = data[anomaly_col].values.astype(bool)
            elif hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                anomalies = self.model_predictions[model_name]['anomalies']
            
            if scores is not None and anomalies is not None:
                metrics_data['Model'].append(model_name)
                metrics_data['Anomalies Detected'].append(int(np.sum(anomalies)))
                metrics_data['Detection Rate (%)'].append(np.sum(anomalies) / len(anomalies) * 100)
                metrics_data['Mean Score'].append(np.mean(scores))
                metrics_data['Max Score'].append(np.max(scores))
                metrics_data['Score Variance'].append(np.var(scores))
        
        if not metrics_data['Model']:
            ax = self.plot_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No performance metrics available',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            return
        
        # Create multi-panel visualization
        fig = self.plot_canvas.fig
        
        # Panel 1: Detection counts
        ax1 = fig.add_subplot(2, 2, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_data['Model'])))
        bars = ax1.bar(range(len(metrics_data['Model'])), metrics_data['Anomalies Detected'],
                      color=colors, edgecolor='black', alpha=0.8)
        ax1.set_xticks(range(len(metrics_data['Model'])))
        ax1.set_xticklabels(metrics_data['Model'], rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Count', fontsize=9, fontweight='bold')
        ax1.set_title('Anomalies Detected', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Detection rate
        ax2 = fig.add_subplot(2, 2, 2)
        bars = ax2.bar(range(len(metrics_data['Model'])), metrics_data['Detection Rate (%)'],
                      color=colors, edgecolor='black', alpha=0.8)
        ax2.set_xticks(range(len(metrics_data['Model'])))
        ax2.set_xticklabels(metrics_data['Model'], rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Percentage', fontsize=9, fontweight='bold')
        ax2.set_title('Detection Rate', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Mean score
        ax3 = fig.add_subplot(2, 2, 3)
        bars = ax3.bar(range(len(metrics_data['Model'])), metrics_data['Mean Score'],
                      color=colors, edgecolor='black', alpha=0.8)
        ax3.set_xticks(range(len(metrics_data['Model'])))
        ax3.set_xticklabels(metrics_data['Model'], rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Score', fontsize=9, fontweight='bold')
        ax3.set_title('Mean Anomaly Score', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Panel 4: Score variance
        ax4 = fig.add_subplot(2, 2, 4)
        bars = ax4.bar(range(len(metrics_data['Model'])), metrics_data['Score Variance'],
                      color=colors, edgecolor='black', alpha=0.8)
        ax4.set_xticks(range(len(metrics_data['Model'])))
        ax4.set_xticklabels(metrics_data['Model'], rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Variance', fontsize=9, fontweight='bold')
        ax4.set_title('Score Variance', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        self.visualization_status_label.setText(
            f"Performance metrics for {len(metrics_data['Model'])} models across 4 dimensions"
        )
    
    def clear_plot(self):
        """Clear the current plot"""
        self.plot_canvas.fig.clear()
        self.plot_canvas.draw()
        self.visualization_status_label.setText("Plot cleared")
    
    def export_plot(self):
        """Export current plot to file"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Plot",
                f"anomaly_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg)"
            )
            
            if file_path:
                self.plot_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.visualization_status_label.setText(f"Plot exported to {file_path}")
                logger.info(f"Plot exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting plot: {str(e)}")
            self.visualization_status_label.setText(f"Error exporting plot: {str(e)}")
    
    def update_visualization_features(self):
        """Update the feature combo box in visualization tab when data changes"""
        try:
            if hasattr(self, 'feature_combo'):
                self.feature_combo.clear()
                
                data = self.data_processor.preprocessed_data if self.data_processor.preprocessed_data is not None else self.data_processor.data
                if data is not None:
                    # Add numeric columns only
                    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Remove timestamp column if it exists
                    if self.data_processor.timestamp_column and self.data_processor.timestamp_column in numeric_columns:
                        numeric_columns.remove(self.data_processor.timestamp_column)
                    
                    self.feature_combo.addItems(numeric_columns)
                    
                    # Auto-select 'value' if it exists
                    if 'value' in numeric_columns:
                        index = self.feature_combo.findText('value')
                        if index >= 0:
                            self.feature_combo.setCurrentIndex(index)
                    
        except Exception as e:
            logger.error(f"Error updating visualization features: {str(e)}")
    
    def update_viz_model_info(self):
        """Update the model info label in visualization tab - NEW"""
        try:
            if not hasattr(self, 'viz_model_info_label'):
                return

            self._sync_trained_models_for_viz()
            
            if hasattr(self, 'trained_models') and self.trained_models:
                valid_models = {name: model for name, model in self.trained_models.items() if model is not None}
                
                if valid_models:
                    info_lines = [f"<b>✓ {len(valid_models)} Models Trained:</b>"]
                    
                    for model_name, model in valid_models.items():
                        model_type = getattr(model, 'model_type', 'Unknown')
                        info_lines.append(f"  • <b>{model_name}</b> ({model_type})")
                        
                        # Add prediction stats if available
                        if hasattr(self, 'model_predictions') and model_name in self.model_predictions:
                            pred_data = self.model_predictions[model_name]
                            info_lines.append(f"    → {pred_data['anomaly_count']} anomalies, "
                                           f"avg score: {pred_data['mean_score']:.3f}")
                    
                    if hasattr(self, 'model_predictions') and self.model_predictions:
                        info_lines.append(f"<br><i>Per-model predictions available for detailed comparison</i>")
                    
                    self.viz_model_info_label.setText("<br>".join(info_lines))
                    self.viz_model_info_label.setStyleSheet("color: #2c3e50; font-size: 10px; padding: 8px; "
                                                            "background-color: #ecf0f1; border-radius: 4px;")
                else:
                    self.viz_model_info_label.setText("No valid models available")
                    self.viz_model_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
            else:
                self.viz_model_info_label.setText("No models trained yet")
                self.viz_model_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
                
        except Exception as e:
            logger.error(f"Error updating viz model info: {str(e)}")

    def load_recent_activity(self):
        """Load recent activity from the log file into the activity table"""
        try:
            if not hasattr(self, 'activity_table'):
                logger.warning("Activity table not initialized")
                return
                
            self.activity_table.setRowCount(0)  # Clear existing rows
            
            # Read last 100 lines from log file
            try:
                with open("security.log", "r") as f:
                    # Read all lines and get last 100
                    lines = f.readlines()[-100:]
                    
                    # Process each line
                    for line in lines:
                        try:
                            # Parse log line (format: timestamp - logger - level - message)
                            parts = line.strip().split(" - ", 3)
                            if len(parts) >= 4:
                                timestamp, logger_name, level, message = parts
                                
                                # Add new row
                                row = self.activity_table.rowCount()
                                self.activity_table.insertRow(row)
                                # Set row height immediately after insertion
                                self.activity_table.setRowHeight(row, 80)  # Extra tall for better visibility
                                
                                # Set time
                                self.activity_table.setItem(row, 0, QTableWidgetItem(timestamp))
                                
                                # Set action (extracted from message)
                                action = message.split(":")[0] if ":" in message else "Event"
                                self.activity_table.setItem(row, 1, QTableWidgetItem(action))
                                
                                # Set details
                                details = message.split(":", 1)[1].strip() if ":" in message else message
                                self.activity_table.setItem(row, 2, QTableWidgetItem(details))
                                
                                # Color-code based on log level
                                color = QColor("black")
                                if level.lower() == "warning":
                                    color = QColor("orange")
                                elif level.lower() == "error":
                                    color = QColor("red")
                                elif level.lower() == "critical":
                                    color = QColor("dark red")
                                    
                                # Apply color to row
                                for col in range(3):
                                    item = self.activity_table.item(row, col)
                                    if item:
                                        item.setForeground(color)
                        except Exception as parse_error:
                            logger.warning(f"Error parsing log line: {str(parse_error)}")
                            continue
                    
            except FileNotFoundError:
                logger.warning("Security log file not found")
                self.activity_table.setRowCount(1)
                self.activity_table.setItem(0, 0, QTableWidgetItem(datetime.datetime.now().isoformat()))
                self.activity_table.setItem(0, 1, QTableWidgetItem("System"))
                self.activity_table.setItem(0, 2, QTableWidgetItem("Activity logging started"))
                # Set row height for this case too
                self.activity_table.setRowHeight(0, 80)
                
            # Resize columns to content
            self.activity_table.resizeColumnsToContents()
            
            # Set row height to double the default (make rows taller)
            for row in range(self.activity_table.rowCount()):
                self.activity_table.setRowHeight(row, 80)  # Extra tall height for visibility
            
            # Sort by time descending
            self.activity_table.sortItems(0, Qt.DescendingOrder)
            
        except Exception as e:
            logger.error(f"Error loading recent activity: {str(e)}")
            self.activity_table.setRowCount(1)
            self.activity_table.setItem(0, 0, QTableWidgetItem(datetime.datetime.now().isoformat()))
            self.activity_table.setItem(0, 1, QTableWidgetItem("Error"))
            self.activity_table.setItem(0, 2, QTableWidgetItem(f"Failed to load activity: {str(e)}"))
            # Set row height even for error case
            self.activity_table.setRowHeight(0, 40)

    def generate_report(self):
        """Generate a comprehensive report of anomaly detection results"""
        try:
            if not hasattr(self.data_processor, 'data') or self.data_processor.data is None:
                QMessageBox.warning(self, "Report Generation", 
                                  "No data available to generate report")
                return
                
            # Create reports directory if it doesn't exist
            if not os.path.exists(REPORTS_DIR):
                os.makedirs(REPORTS_DIR)
                
            # Generate timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(REPORTS_DIR, f"anomaly_report_{timestamp}.html")
            
            # Start building HTML report
            html_content = f"""
            <html>
            <head>
                <title>Anomaly Detection Report - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2196F3; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f5f5f5; }}
                    .highlight {{ background-color: #ffeb3b; }}
                    .anomaly {{ color: red; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Anomaly Detection Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            """
            
            # Add model information if available
            if self.model:
                html_content += f"""
                <h2>Model Information</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Model Type</td><td>{self.model.model_type}</td></tr>
                """
                
                if hasattr(self.model, 'metrics'):
                    for metric, value in self.model.metrics.items():
                        html_content += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                        
                html_content += "</table>"
            
            # Add data summary
            data = self.data_processor.data
            html_content += f"""
            <h2>Data Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Records</td><td>{len(data)}</td></tr>
            """
            
            if "Anomaly" in data.columns:
                anomaly_count = len(data[data["Anomaly"] == True])
                anomaly_rate = (anomaly_count / len(data)) * 100
                html_content += f"""
                <tr><td>Anomalies Found</td><td>{anomaly_count}</td></tr>
                <tr><td>Anomaly Rate</td><td>{anomaly_rate:.2f}%</td></tr>
                """
            
            html_content += "</table>"
            
            # Add feature statistics
            if self.data_processor.feature_columns:
                html_content += """
                <h2>Feature Statistics</h2>
                <table>
                    <tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
                """
                
                stats = self.data_processor.get_feature_stats()
                for feature, feat_stats in stats.items():
                    html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{feat_stats['mean']:.2f}</td>
                        <td>{feat_stats['std']:.2f}</td>
                        <td>{feat_stats['min']:.2f}</td>
                        <td>{feat_stats['max']:.2f}</td>
                    </tr>
                    """
                
                html_content += "</table>"
            
            # Add anomaly details if available
            if "Anomaly" in data.columns:
                html_content += """
                <h2>Detected Anomalies</h2>
                <table>
                    <tr><th>Timestamp</th><th>Feature</th><th>Value</th><th>Anomaly Score</th></tr>
                """
                
                anomalies = data[data["Anomaly"] == True].copy()
                for _, row in anomalies.iterrows():
                    timestamp = row[self.data_processor.timestamp_column] if self.data_processor.timestamp_column else "N/A"
                    feature = self.feature_combo.currentText()
                    value = row[feature] if feature in row else "N/A"
                    score = row["Anomaly Score"] if "Anomaly Score" in row else "N/A"
                    
                    html_content += f"""
                    <tr class="anomaly">
                        <td>{timestamp}</td>
                        <td>{feature}</td>
                        <td>{value}</td>
                        <td>{score}</td>
                    </tr>
                    """
                
                html_content += "</table>"
            
            # Close HTML document
            html_content += """
            </body>
            </html>
            """
            
            # Save report
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            # Show success message with option to open report
            reply = QMessageBox.question(
                self, 
                "Report Generated",
                f"Report saved to:\n{report_file}\n\nWould you like to open it now?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Open report in default web browser
                import webbrowser
                webbrowser.open(f'file://{os.path.abspath(report_file)}')
                
            logger.info(f"Generated anomaly detection report: {report_file}")
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Report Generation Error", error_msg)

    def lock_application(self):
        """Lock the application after inactivity period"""
        try:
            # Check if timers exist and are active before stopping them
            if hasattr(self, 'auto_load_timer'):
                if self.auto_load_timer and self.auto_load_timer.isActive():
                    self.auto_load_timer.stop()
            
            if hasattr(self, 'auto_process_timer'):
                if self.auto_process_timer and self.auto_process_timer.isActive():
                    self.auto_process_timer.stop()
            
            # Store monitoring state
            was_auto_loading = hasattr(self, 'auto_load_check') and self.auto_load_check and self.auto_load_check.isChecked()
            was_auto_processing = hasattr(self, 'auto_process_check') and self.auto_process_check and self.auto_process_check.isChecked()
            
            # Create and show login dialog
            login_dialog = LoginDialog(self.user_manager, self)
            result = login_dialog.exec_()
            
            if result == QDialog.Accepted and login_dialog.authenticated:
                # Restore previous username and role
                if login_dialog.username == self.current_username:
                    # Resume monitoring if it was active
                    if was_auto_loading:
                        self.toggle_auto_loading(Qt.Checked)
                    if was_auto_processing:
                        self.toggle_auto_processing(Qt.Checked)
                        
                    # Reset inactivity timer
                    self.reset_inactivity_timer()
                    self.start_session_timer()
                    
                    # Log successful unlock
                    logger.info(f"Application unlocked by user {self.current_username}")
                    write_audit_event(
                        actor=self.current_username,
                        action="session_unlock",
                        resource="auth/session",
                        outcome="success",
                        details={"session_id": self.current_session_id},
                        auth_provider=self.current_auth_provider
                    )
                else:
                    # Different user logged in - reset UI
                    self.current_username = login_dialog.username
                    self.current_role = login_dialog.role
                    self.current_auth_provider = getattr(login_dialog, "auth_provider", "local")
                    self.current_session_id = uuid.uuid4().hex
                    self.status_bar.showMessage(
                        f"Logged in as {self.current_username} ({self.current_role}) via {self.current_auth_provider}"
                    )
                    write_audit_event(
                        actor=self.current_username,
                        action="session_switch",
                        resource="auth/session",
                        outcome="success",
                        details={"session_id": self.current_session_id, "reason": "unlock_as_different_user"},
                        auth_provider=self.current_auth_provider
                    )
                    
                    # Update UI based on new user permissions
                    self.update_ui_permissions()
                    self.reset_inactivity_timer()
                    self.start_session_timer()
                    
                    logger.info(f"Application relocked and accessed by different user {self.current_username}")
            else:
                # Login failed - close application
                logger.warning("Failed to unlock application - closing for security")
                self.close()
                
        except Exception as e:
            logger.error(f"Error during application lock: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.close()

    def reset_inactivity_timer(self):
        """Reset the inactivity timer"""
        if hasattr(self, 'inactivity_timer'):
            self.inactivity_timer.stop()
            # Use configurable timeout
            timeout_ms = getattr(self, 'lock_timeout_ms', 900000)  # Default 15 minutes
            self.inactivity_timer.start(timeout_ms)
    
    def start_session_timer(self):
        """Start/reset absolute session timer."""
        if hasattr(self, 'session_timer'):
            self.session_timer.stop()
            timeout_ms = getattr(self, 'session_timeout_ms', 28800000)  # Default 8 hours
            self.session_timer.start(timeout_ms)
    
    def eventFilter(self, obj, event):
        """Reset inactivity timer on user activity events."""
        try:
            if hasattr(self, 'inactivity_timer'):
                activity_events = {
                    QEvent.MouseButtonPress,
                    QEvent.MouseButtonRelease,
                    QEvent.MouseMove,
                    QEvent.Wheel,
                    QEvent.KeyPress,
                    QEvent.TouchBegin
                }
                if event.type() in activity_events:
                    self.reset_inactivity_timer()
        except Exception:
            pass
        
        return super().eventFilter(obj, event)

    def update_ui_permissions(self):
        """Update UI elements based on user permissions"""
        try:
            # Remove all tabs first
            while self.tabs.count() > 0:
                self.tabs.removeTab(0)
                
            # Always show dashboard
            self.tabs.addTab(self.dashboard_tab, "Home")
            tab_policy = None
            if modular_build_tab_visibility_policy is not None:
                try:
                    tab_policy = modular_build_tab_visibility_policy(self.service_layer, self.current_username)
                except Exception:
                    tab_policy = None
            
            # Add other tabs based on permissions
            can_import_data = (
                bool(tab_policy.can_import_data)
                if tab_policy is not None
                else self.service_layer.authorize(self.current_username, "import_data", resource="data_source:*")
            )
            if can_import_data and not getattr(self, "custom_tabs_only_mode", False):
                self.tabs.addTab(self.data_tab, "Data Import")
                
            can_process_data = (
                bool(tab_policy.can_process_data)
                if tab_policy is not None
                else self.service_layer.authorize(self.current_username, "process_data", resource="analysis")
            )
            if can_process_data and not getattr(self, "custom_tabs_only_mode", False):
                self.tabs.addTab(self.analysis_tab, "Analysis / ML")
            if can_process_data and not getattr(self, "custom_tabs_only_mode", False):
                self.tabs.addTab(self.visualization_tab, "Visualization")
                
            can_manage_users = (
                bool(tab_policy.can_manage_users)
                if tab_policy is not None
                else self.service_layer.authorize(self.current_username, "manage_users", resource="admin/users")
            )
            if can_manage_users:
                self.tabs.addTab(self.admin_tab, "Administration")

            # Re-add custom monitoring tabs after permission rebuild
            for tab_id, custom_tab in list(getattr(self, "custom_tabs", {}).items()):
                title = getattr(custom_tab, "config", {}).get("title") or tab_id
                if self.tabs.indexOf(custom_tab) < 0:
                    self.tabs.addTab(custom_tab, title)

            self._hide_primary_tab_bar_entries()
                
        except Exception as e:
            logger.error(f"Error updating UI permissions: {str(e)}")

    def on_tab_changed(self, index):
        """Handle tab changes and reset inactivity timer"""
        try:
            # Reset inactivity timer
            self.reset_inactivity_timer()
            
            # Log tab change
            tab_name = self.tabs.tabText(index)
            logger.info(f"User {self.current_username} switched to {tab_name} tab")
            if hasattr(self, "_sync_ops_nav_active"):
                self._sync_ops_nav_active()
            
            # Update status bar
            self.status_bar.showMessage(
                f"Logged in as {self.current_username} ({self.current_role}) via {self.current_auth_provider} - {tab_name}"
            )
            
            # Refresh tab-specific content
            if tab_name == "Home":
                self.load_recent_activity()
            elif tab_name == "Data Import":
                self.update_data_preview()
            elif tab_name in ("Analysis", "Analysis / ML"):
                if hasattr(self, 'selected_model_label'):
                    if self.model:
                        self.selected_model_label.setText(f"Current model: {self.model.model_type}")
                    else:
                        self.selected_model_label.setText("No model loaded")
            elif tab_name == "Administration":
                self.load_users()
                
        except Exception as e:
            logger.error(f"Error handling tab change: {str(e)}")
            self.status_bar.showMessage(f"Error switching tabs: {str(e)}")

    # ---- Health Tab UI Overrides (single-path, modernized) ----
    def _health_tab_stylesheet(self):
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #d8dee9;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
                color: #2e3a59;
            }
            QFrame#HealthCard {
                border: 1px solid #e5e9f2;
                border-radius: 10px;
                background: #f9fbff;
                padding: 8px;
            }
            QLabel#HealthMetricTitle {
                color: #5c6b84;
                font-size: 10px;
                font-weight: 600;
            }
            QLabel#HealthMetricValue {
                color: #1f2d3d;
                font-size: 14px;
                font-weight: 700;
            }
            QTableWidget {
                border: 1px solid #dfe6f1;
                border-radius: 6px;
                gridline-color: #eef2f8;
                selection-background-color: #d7e8ff;
                alternate-background-color: #f8fbff;
            }
            QHeaderView::section {
                background: #eef3fb;
                color: #2e3a59;
                font-weight: 700;
                border: none;
                padding: 6px;
            }
        """

    def _create_health_metric_card(self, title, value):
        card = QFrame()
        card.setObjectName("HealthCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("HealthMetricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("HealthMetricValue")
        value_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card, value_label

    def setup_health_tab(self):
        """Modern single-path System Health dashboard UI."""
        logger.info("Setting up modernized System Health tab")
        self.debug_status("Health Tab UI Override")

        # Clear existing layout safely
        existing_layout = self.health_tab.layout()
        if existing_layout is not None:
            while existing_layout.count():
                item = existing_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        self.health_tab.setStyleSheet(self._health_tab_stylesheet())
        root_layout = QVBoxLayout(self.health_tab)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Controls and sync state
        controls_group = QGroupBox("Monitoring Controls")
        controls_layout = QVBoxLayout()

        status_row = QHBoxLayout()
        self.sync_status_label = QLabel("Waiting for Analysis tab data/model...")
        self.sync_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        self.analysis_model_info_label = QLabel("Analysis Model: Not Available")
        self.analysis_model_info_label.setStyleSheet("color: #5c6b84;")
        status_row.addWidget(self.sync_status_label, 1)
        status_row.addWidget(self.analysis_model_info_label, 1)
        controls_layout.addLayout(status_row)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold:"))
        self.threshold_value_label = QLabel("Auto (95th percentile)")
        self.threshold_value_label.setStyleSheet("font-weight: bold; color: #2E4BC6;")
        self.adjust_threshold_btn = QPushButton("Adjust")
        self.adjust_threshold_btn.setMaximumWidth(80)
        self.adjust_threshold_btn.clicked.connect(self.adjust_threshold_dialog)
        threshold_row.addWidget(self.threshold_value_label)
        threshold_row.addWidget(self.adjust_threshold_btn)
        threshold_row.addStretch()
        controls_layout.addLayout(threshold_row)

        action_row = QHBoxLayout()
        self.start_monitoring_btn = QPushButton("Start Monitoring")
        self.start_monitoring_btn.clicked.connect(self.start_health_monitoring)
        self.start_monitoring_btn.setEnabled(False)
        self.stop_monitoring_btn = QPushButton("Stop Monitoring")
        self.stop_monitoring_btn.clicked.connect(self.stop_health_monitoring)
        self.stop_monitoring_btn.setEnabled(False)
        self.monitoring_status_indicator = QLabel("[Inactive]")
        self.monitoring_status_indicator.setStyleSheet("color: #999; font-weight: bold;")
        action_row.addWidget(self.start_monitoring_btn)
        action_row.addWidget(self.stop_monitoring_btn)
        action_row.addWidget(self.monitoring_status_indicator)
        action_row.addStretch()
        controls_layout.addLayout(action_row)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Top status cards
        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)

        status_card, self.anomaly_alert_label = self._create_health_metric_card("System Status", "Waiting for monitoring start")
        ttf_card, self.ttf_label = self._create_health_metric_card("Time To Failure (TTF)", "N/A")
        err_card, self.current_error_label = self._create_health_metric_card("Current Error", "N/A")
        th_card, self.threshold_status_label = self._create_health_metric_card("Threshold", "N/A")
        pred_card, self.prediction_status_label = self._create_health_metric_card("Prediction", "N/A")
        upd_card, self.last_update_label = self._create_health_metric_card("Last Update", "N/A")

        self.anomaly_alert_label.setObjectName("HealthMetricValue")
        self.ttf_label.setObjectName("HealthMetricValue")

        cards_row.addWidget(status_card, 2)
        cards_row.addWidget(ttf_card, 2)
        cards_row.addWidget(err_card, 1)
        cards_row.addWidget(th_card, 1)
        cards_row.addWidget(pred_card, 1)
        cards_row.addWidget(upd_card, 1)
        layout.addLayout(cards_row)

        # Main visualization
        viz_group = QGroupBox("Health Trend Visualization")
        viz_layout = QVBoxLayout()
        self.combined_canvas = MplCanvas(width=14, height=4)
        viz_layout.addWidget(self.combined_canvas)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Backward compatibility aliases for existing plot helpers
        self.health_canvas = self.combined_canvas
        self.health_fig = self.combined_canvas.figure

        # Alerts table
        alerts_group = QGroupBox("Active Alerts")
        alerts_layout = QVBoxLayout()
        self.health_alerts_table = QTableWidget(0, 6)
        self.health_alerts_table.setHorizontalHeaderLabels(["Time", "Model", "Error", "Threshold", "Status", "Prediction"])
        self.health_alerts_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.health_alerts_table.setAlternatingRowColors(True)
        self.health_alerts_table.setMinimumHeight(200)
        alerts_layout.addWidget(self.health_alerts_table)
        self.active_alerts_label = QLabel("Active Alerts: 0")
        self.active_alerts_label.setStyleSheet("color: #5c6b84; font-weight: 600;")
        alerts_layout.addWidget(self.active_alerts_label)
        alerts_group.setLayout(alerts_layout)
        layout.addWidget(alerts_group)

        # Legacy compatibility alias (used by add_health_alert)
        self.alerts_table = self.health_alerts_table

        # Model status table
        model_group = QGroupBox("Model Status Overview")
        model_layout = QVBoxLayout()
        self.models_status_table = QTableWidget(0, 5)
        self.models_status_table.setHorizontalHeaderLabels(["Model", "Type", "Threshold", "Status", "Health"])
        self.models_status_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.models_status_table.setAlternatingRowColors(True)
        self.models_status_table.setMinimumHeight(150)
        model_layout.addWidget(self.models_status_table)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Compatibility aliases used by legacy failure-prediction controls
        self.start_health_monitoring_btn = self.start_monitoring_btn
        self.stop_health_monitoring_btn = self.stop_monitoring_btn
        if not hasattr(self, "health_monitoring_timer"):
            self.health_monitoring_timer = QTimer(self)
            self.health_monitoring_timer.timeout.connect(self.collect_health_data)

        # Final assembly
        scroll.setWidget(container)
        root_layout.addWidget(scroll)
        self.health_monitoring_active = False
        if not hasattr(self, "health_threshold_mode"):
            self.health_threshold_mode = "auto"

    def add_alert_to_table(self, result):
        """Override: keep active-alert count synchronized with the single Health table."""
        try:
            row = self.health_alerts_table.rowCount()
            self.health_alerts_table.insertRow(row)

            time_str = result['timestamp'].strftime('%H:%M:%S')
            self.health_alerts_table.setItem(row, 0, QTableWidgetItem(time_str))
            self.health_alerts_table.setItem(row, 1, QTableWidgetItem(result['model_name']))

            error_item = QTableWidgetItem(f"{result['reconstruction_error']:.6f}")
            if result['exceeds_threshold']:
                error_item.setBackground(QColor("#f44336"))
            self.health_alerts_table.setItem(row, 2, error_item)

            self.health_alerts_table.setItem(row, 3, QTableWidgetItem(f"{result['threshold']:.6f}"))

            status_item = QTableWidgetItem(result['status'].title())
            if result['status'] == 'normal':
                status_item.setBackground(QColor("#4CAF50"))
            elif result['status'] == 'warning':
                status_item.setBackground(QColor("#FF9800"))
            elif result['status'] == 'critical':
                status_item.setBackground(QColor("#f44336"))
            else:
                status_item.setBackground(QColor("#B71C1C"))
            self.health_alerts_table.setItem(row, 4, status_item)

            if result['failure_prediction']:
                pred = result['failure_prediction']
                pred_text = f"{pred['status']} ({pred['time_to_failure']})"
                pred_item = QTableWidgetItem(pred_text)
                pred_item.setBackground(QColor("#f44336"))
            else:
                pred_item = QTableWidgetItem("Normal")
                pred_item.setBackground(QColor("#4CAF50"))
            self.health_alerts_table.setItem(row, 5, pred_item)

            self.health_alerts_table.scrollToBottom()
            if self.health_alerts_table.rowCount() > 100:
                self.health_alerts_table.removeRow(0)

            if hasattr(self, "active_alerts_label"):
                self.active_alerts_label.setText(f"Active Alerts: {self.health_alerts_table.rowCount()}")
        except Exception as e:
            logger.error(f"Error adding alert to table: {str(e)}")

# CUSTOMIZABLE TABS SYSTEM


class TabConfigurationManager:
    """Manages saving and loading of customizable tab configurations"""
    
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(DATA_DIR, "custom_tabs_config.json")
        self.config_file = config_file
        self.configs = {}
        self.load_configs()
    
    def load_configs(self):
        """Load tab configurations from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.configs = json.load(f)
                logger.info(f"Loaded {len(self.configs)} custom tab configurations")
            else:
                self.configs = {}
                logger.info("No existing tab configurations found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading tab configurations: {str(e)}")
            self.configs = {}
    
    def save_configs(self):
        """Save tab configurations to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.configs, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.configs)} custom tab configurations")
            return True
        except Exception as e:
            logger.error(f"Error saving tab configurations: {str(e)}")
            return False
    
    def add_config(self, tab_id, config):
        """Add or update a tab configuration"""
        self.configs[tab_id] = config
        return self.save_configs()
    
    def get_config(self, tab_id):
        """Get configuration for a tab"""
        return self.configs.get(tab_id, None)
    
    def remove_config(self, tab_id):
        """Remove a tab configuration"""
        if tab_id in self.configs:
            del self.configs[tab_id]
            return self.save_configs()
        return False
    
    def list_configs(self):
        """List all tab configurations"""
        return list(self.configs.keys())

class TabConfigurationDialog(QDialog):
    """Dialog for configuring a new custom monitoring tab"""
    
    def __init__(self, parent=None, existing_config=None):
        super().__init__(parent)
        self.existing_config = existing_config
        self.config = None
        self.initUI()
    
    def initUI(self):
        """Initialize the configuration dialog UI"""
        self.setWindowTitle("Configure Monitoring Tab" if not self.existing_config else "Edit Monitoring Tab")
        self.setMinimumSize(560, 520)
        self.resize(620, 760)

        dialog_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        # Tab Title
        title_group = QGroupBox("Tab Information")
        title_layout = QVBoxLayout()
        title_input_layout = QHBoxLayout()
        title_input_layout.addWidget(QLabel("Tab Title:"))
        self.title_input = QLineEdit()
        if self.existing_config:
            self.title_input.setText(self.existing_config.get('title', ''))
        else:
            self.title_input.setPlaceholderText("e.g., EPS Data Monitoring and Health")
        title_input_layout.addWidget(self.title_input)
        title_layout.addLayout(title_input_layout)
        title_group.setLayout(title_layout)
        layout.addWidget(title_group)

        # Data Source
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout()
        source_path_layout = QHBoxLayout()
        source_path_layout.addWidget(QLabel("Data Folder:"))
        self.data_folder_input = QLineEdit()
        self.data_folder_input.setReadOnly(True)
        if self.existing_config:
            self.data_folder_input.setText(self.existing_config.get('data_folder', ''))
        browse_folder_btn = QPushButton("Browse...")
        browse_folder_btn.clicked.connect(self.browse_data_folder)
        source_path_layout.addWidget(self.data_folder_input)
        self.browse_folder_btn = browse_folder_btn
        source_path_layout.addWidget(self.browse_folder_btn)
        source_layout.addLayout(source_path_layout)

        file_type_row = QHBoxLayout()
        file_type_row.addWidget(QLabel("File Type:"))
        self.file_type_combo = QComboBox()
        self.file_type_combo.addItems(["CSV", "JSON"])
        if self.existing_config:
            ft = self.existing_config.get("data_file_type", "CSV")
            idx = self.file_type_combo.findText(ft)
            if idx >= 0:
                self.file_type_combo.setCurrentIndex(idx)
        file_type_row.addWidget(self.file_type_combo)
        file_type_row.addStretch()
        source_layout.addLayout(file_type_row)
        self.file_type_combo.currentTextChanged.connect(lambda _t: self._refresh_feature_checklist())

        dataset_mode_row = QHBoxLayout()
        dataset_mode_row.addWidget(QLabel("Dataset Window:"))
        self.dataset_mode_combo = QComboBox()
        self.dataset_mode_combo.addItems([
            "Full Latest File",
            "Daily",
            "Weekly",
            "Monthly",
        ])
        if self.existing_config:
            dm = self.existing_config.get("dataset_mode", "Full Latest File")
            idx = self.dataset_mode_combo.findText(dm)
            if idx >= 0:
                self.dataset_mode_combo.setCurrentIndex(idx)
        dataset_mode_row.addWidget(self.dataset_mode_combo)
        dataset_mode_row.addStretch()
        source_layout.addLayout(dataset_mode_row)

        source_layout.addWidget(QLabel("Telemetry Features (optional — unchecked = use all numeric):"))
        self.features_list = QListWidget()
        self.features_list.setMaximumHeight(120)
        self.features_list.setSelectionMode(QAbstractItemView.NoSelection)
        source_layout.addWidget(self.features_list)
        if self.existing_config and self.existing_config.get("data_folder"):
            self._refresh_feature_checklist()

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Input Mode / Stream Connector
        input_mode_group = QGroupBox("Input Mode")
        input_mode_layout = QVBoxLayout()

        self.active_device_monitoring_checkbox = QCheckBox(
            "Active Device Monitoring (Serial ingestion)"
        )
        if self.existing_config:
            self.active_device_monitoring_checkbox.setChecked(
                bool(self.existing_config.get("active_device_monitoring_enabled", False))
            )
        input_mode_layout.addWidget(self.active_device_monitoring_checkbox)

        input_mode_row = QHBoxLayout()
        input_mode_row.addWidget(QLabel("Source Type:"))
        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItems(["CSV Polling", "MQTT Stream", "OPC-UA Stream"])
        if self.existing_config:
            configured_mode = self.existing_config.get("input_mode", "CSV Polling")
            mode_idx = self.input_mode_combo.findText(configured_mode)
            if mode_idx >= 0:
                self.input_mode_combo.setCurrentIndex(mode_idx)
        input_mode_row.addWidget(self.input_mode_combo)
        input_mode_row.addStretch()
        input_mode_layout.addLayout(input_mode_row)

        self.stream_settings_widget = QWidget()
        stream_form = QFormLayout(self.stream_settings_widget)

        self.mqtt_broker_input = QLineEdit()
        self.mqtt_broker_input.setPlaceholderText("e.g., localhost")
        self.mqtt_port_input = QSpinBox()
        self.mqtt_port_input.setRange(1, 65535)
        self.mqtt_port_input.setValue(1883)
        self.mqtt_topic_input = QLineEdit()
        self.mqtt_topic_input.setPlaceholderText("e.g., telemetry/satellite/eps")

        self.opcua_endpoint_input = QLineEdit()
        self.opcua_endpoint_input.setPlaceholderText("e.g., opc.tcp://127.0.0.1:4840")
        self.opcua_nodes_input = QLineEdit()
        self.opcua_nodes_input.setPlaceholderText("e.g., ns=2;i=2,ns=2;i=3")

        if self.existing_config:
            self.mqtt_broker_input.setText(self.existing_config.get("mqtt_broker", ""))
            self.mqtt_port_input.setValue(int(self.existing_config.get("mqtt_port", 1883)))
            self.mqtt_topic_input.setText(self.existing_config.get("mqtt_topic", ""))
            self.opcua_endpoint_input.setText(self.existing_config.get("opcua_endpoint", ""))
            self.opcua_nodes_input.setText(self.existing_config.get("opcua_nodes", ""))

        stream_form.addRow("MQTT Broker:", self.mqtt_broker_input)
        stream_form.addRow("MQTT Port:", self.mqtt_port_input)
        stream_form.addRow("MQTT Topic:", self.mqtt_topic_input)
        stream_form.addRow("OPC-UA Endpoint:", self.opcua_endpoint_input)
        stream_form.addRow("OPC-UA Nodes:", self.opcua_nodes_input)
        input_mode_layout.addWidget(self.stream_settings_widget)

        self.serial_settings_widget = QWidget()
        serial_form = QFormLayout(self.serial_settings_widget)
        self.serial_port_input = QLineEdit()
        self.serial_port_input.setPlaceholderText("e.g., COM3 or /dev/ttyUSB0")
        self.serial_baud_spin = QSpinBox()
        self.serial_baud_spin.setRange(1200, 2000000)
        self.serial_baud_spin.setValue(115200)
        self.serial_delimiter_input = QLineEdit(",")
        self.serial_delimiter_input.setMaxLength(2)
        self.serial_fields_input = QLineEdit()
        self.serial_fields_input.setPlaceholderText("e.g., ch1,ch2,temp,pressure")
        if self.existing_config:
            self.serial_port_input.setText(self.existing_config.get("serial_port", ""))
            self.serial_baud_spin.setValue(int(self.existing_config.get("serial_baudrate", 115200)))
            self.serial_delimiter_input.setText(self.existing_config.get("serial_delimiter", ","))
            self.serial_fields_input.setText(",".join(self.existing_config.get("serial_fields", [])))
        serial_form.addRow("Serial Port:", self.serial_port_input)
        serial_form.addRow("Baudrate:", self.serial_baud_spin)
        serial_form.addRow("Delimiter:", self.serial_delimiter_input)
        serial_form.addRow("Field names:", self.serial_fields_input)
        input_mode_layout.addWidget(self.serial_settings_widget)

        input_mode_group.setLayout(input_mode_layout)
        layout.addWidget(input_mode_group)
        self.input_mode_combo.currentTextChanged.connect(self._on_input_mode_changed)
        self.active_device_monitoring_checkbox.toggled.connect(self._on_device_monitoring_toggled)
        self._on_input_mode_changed(self.input_mode_combo.currentText())
        self._on_device_monitoring_toggled(self.active_device_monitoring_checkbox.isChecked())

        # Subsystem Grouping
        subsystem_group = QGroupBox("Subsystem Grouping (Optional)")
        subsystem_layout = QVBoxLayout()

        subsystem_name_layout = QHBoxLayout()
        subsystem_name_layout.addWidget(QLabel("Subsystem Name:"))
        self.subsystem_name_input = QLineEdit()
        if self.existing_config:
            self.subsystem_name_input.setText(self.existing_config.get('subsystem_name', '') or '')
        self.subsystem_name_input.setPlaceholderText("e.g., EPS, Power System, Thermal Control")
        subsystem_name_layout.addWidget(self.subsystem_name_input)
        subsystem_layout.addLayout(subsystem_name_layout)

        subsystem_group.setLayout(subsystem_layout)
        layout.addWidget(subsystem_group)

        # Model Configuration - Multi-model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()

        selection_controls_layout = QHBoxLayout()
        selection_controls_layout.addWidget(QLabel("Model Selection:"))
        select_all_models_btn = QPushButton("Select All")
        clear_models_btn = QPushButton("Clear")
        select_all_models_btn.clicked.connect(self._select_all_initial_models)
        clear_models_btn.clicked.connect(self._clear_all_initial_models)
        selection_controls_layout.addWidget(select_all_models_btn)
        selection_controls_layout.addWidget(clear_models_btn)
        selection_controls_layout.addStretch()
        model_layout.addLayout(selection_controls_layout)

        self.model_multi_list = QListWidget()
        self.model_multi_list.setMaximumHeight(180)
        self.model_multi_list.setSelectionMode(QAbstractItemView.NoSelection)
        available_models = get_supported_model_names()
        existing_model_names = []
        if self.existing_config:
            existing_model_names = [m.get("model_type") for m in self.existing_config.get("models", []) if m.get("model_type")]

        for model_name in available_models:
            item = QListWidgetItem(format_model_catalog_label(model_name))
            item.setData(Qt.UserRole, model_name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            checked = bool(existing_model_names and model_name in existing_model_names)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.model_multi_list.addItem(item)
        model_layout.addWidget(self.model_multi_list)

        model_selection_layout = QHBoxLayout()
        model_selection_layout.addWidget(QLabel("Primary Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(available_models)
        if self.existing_config:
            models_list = self.existing_config.get('models', [])
            if models_list and len(models_list) > 0:
                model_name = models_list[0].get('model_type', 'Isolation Forest')
            else:
                model_name = self.existing_config.get('model_type', 'Isolation Forest')
            index = self.model_combo.findText(model_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        model_selection_layout.addWidget(self.model_combo)
        model_selection_layout.addStretch()
        model_layout.addLayout(model_selection_layout)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setMaximumHeight(200)
        params_widget = QWidget()
        self.params_layout = QFormLayout(params_widget)
        self.params_layout.setSpacing(5)
        params_scroll.setWidget(params_widget)
        model_layout.addWidget(params_scroll)
        self.params_widgets = {}

        self.model_combo.currentTextChanged.connect(self.update_model_parameters)
        self.model_multi_list.itemChanged.connect(self._on_model_selection_changed)
        self._sync_primary_model_combo_with_selection()
        self.update_model_parameters()

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Monitoring Schedule
        schedule_group = QGroupBox("Monitoring Schedule")
        schedule_layout = QVBoxLayout()

        schedule_type_layout = QHBoxLayout()
        schedule_type_layout.addWidget(QLabel("Schedule Type:"))
        self.schedule_type_combo = QComboBox()
        self.schedule_type_combo.addItems(["Continuous", "On-Demand", "Scheduled"])
        if self.existing_config:
            schedule_type = self.existing_config.get('schedule_type', 'Continuous')
            idx = self.schedule_type_combo.findText(schedule_type)
            if idx >= 0:
                self.schedule_type_combo.setCurrentIndex(idx)
        schedule_type_layout.addWidget(self.schedule_type_combo)
        schedule_type_layout.addStretch()
        schedule_layout.addLayout(schedule_type_layout)

        self.interval_widget = QWidget()
        interval_layout = QHBoxLayout(self.interval_widget)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_layout.addWidget(QLabel("Check Interval:"))

        self.interval_hours = QSpinBox()
        self.interval_hours.setRange(0, 23)
        self.interval_hours.setSuffix(" h")
        if self.existing_config:
            interval_ms = self.existing_config.get('interval_ms', 300000) // 3600000
            self.interval_hours.setValue(interval_ms)

        self.interval_minutes = QSpinBox()
        self.interval_minutes.setRange(0, 59)
        self.interval_minutes.setSuffix(" m")
        if self.existing_config:
            interval_ms = self.existing_config.get('interval_ms', 300000)
            remaining_minutes = (interval_ms // 60000) % 60
            self.interval_minutes.setValue(remaining_minutes)

        self.interval_seconds = QSpinBox()
        self.interval_seconds.setRange(0, 59)
        self.interval_seconds.setSuffix(" s")
        if self.existing_config:
            interval_ms = self.existing_config.get('interval_ms', 300000)
            remaining_seconds = (interval_ms // 1000) % 60
            self.interval_seconds.setValue(remaining_seconds)
        else:
            self.interval_minutes.setValue(5)

        interval_layout.addWidget(self.interval_hours)
        interval_layout.addWidget(self.interval_minutes)
        interval_layout.addWidget(self.interval_seconds)
        interval_layout.addStretch()
        schedule_layout.addWidget(self.interval_widget)

        self.utc_schedule_widget = QWidget()
        utc_layout = QHBoxLayout(self.utc_schedule_widget)
        utc_layout.setContentsMargins(0, 0, 0, 0)
        utc_layout.addWidget(QLabel("Run At (UTC):"))

        self.utc_hour_spin = QSpinBox()
        self.utc_hour_spin.setRange(0, 23)
        self.utc_hour_spin.setSuffix(" h")
        self.utc_minute_spin = QSpinBox()
        self.utc_minute_spin.setRange(0, 59)
        self.utc_minute_spin.setSuffix(" m")

        if self.existing_config:
            self.utc_hour_spin.setValue(int(self.existing_config.get("schedule_utc_hour", 0)))
            self.utc_minute_spin.setValue(int(self.existing_config.get("schedule_utc_minute", 0)))
        else:
            self.utc_hour_spin.setValue(0)
            self.utc_minute_spin.setValue(0)

        utc_layout.addWidget(self.utc_hour_spin)
        utc_layout.addWidget(self.utc_minute_spin)
        utc_layout.addStretch()
        schedule_layout.addWidget(self.utc_schedule_widget)

        self.schedule_type_combo.currentTextChanged.connect(self._on_schedule_type_changed)
        self._on_schedule_type_changed(self.schedule_type_combo.currentText())

        schedule_group.setLayout(schedule_layout)
        layout.addWidget(schedule_group)

        # Operational limits
        ops_group = QGroupBox("Operational Settings")
        ops_layout = QFormLayout()
        self.max_training_rows_spin = QSpinBox()
        self.max_training_rows_spin.setRange(1000, 2000000)
        self.max_training_rows_spin.setSingleStep(10000)
        self.max_training_rows_spin.setValue(
            int(self.existing_config.get("max_training_rows", 100000)) if self.existing_config else 100000
        )
        self.monitoring_window_spin = QSpinBox()
        self.monitoring_window_spin.setRange(50, 500000)
        self.monitoring_window_spin.setSingleStep(100)
        self.monitoring_window_spin.setValue(
            int(self.existing_config.get("monitoring_window_rows", 500)) if self.existing_config else 500
        )
        ops_layout.addRow("Max rows for training:", self.max_training_rows_spin)
        ops_layout.addRow("Monitoring window (latest rows):", self.monitoring_window_spin)
        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)

        # Mission Modes (FSM)
        from app.models.fsm import DEFAULT_MISSION_MODES, ensure_fsm_fields

        mission_group = QGroupBox("Mission Modes")
        mission_layout = QVBoxLayout()
        mission_hint = QLabel(
            "Threshold scale widens (e.g. eclipse 1.5) or tightens (safe_mode 0.5) "
            "OBS / fusion anomaly limits for the active operating regime."
        )
        mission_hint.setWordWrap(True)
        mission_layout.addWidget(mission_hint)

        cfg_for_modes = ensure_fsm_fields(dict(self.existing_config or {}))
        saved_modes = {
            str(m.get("name", "")).lower(): m for m in (cfg_for_modes.get("mission_modes") or [])
        }
        self.mission_mode_scale_spins = {}
        for default_mode in DEFAULT_MISSION_MODES:
            name = default_mode["name"]
            saved = saved_modes.get(name, default_mode)
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name.title()}:"))
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 5.0)
            spin.setSingleStep(0.1)
            spin.setDecimals(2)
            try:
                spin.setValue(float(saved.get("threshold_scale", default_mode["threshold_scale"])))
            except (TypeError, ValueError):
                spin.setValue(float(default_mode["threshold_scale"]))
            spin.setToolTip(str(saved.get("description") or default_mode.get("description") or ""))
            self.mission_mode_scale_spins[name] = spin
            row.addWidget(QLabel("threshold_scale"))
            row.addWidget(spin)
            row.addStretch()
            mission_layout.addLayout(row)

        current_row = QHBoxLayout()
        current_row.addWidget(QLabel("Current mode:"))
        self.current_mission_mode_combo = QComboBox()
        for default_mode in DEFAULT_MISSION_MODES:
            self.current_mission_mode_combo.addItem(default_mode["name"])
        current_name = str(cfg_for_modes.get("current_mission_mode") or "nominal")
        idx = self.current_mission_mode_combo.findText(current_name)
        if idx >= 0:
            self.current_mission_mode_combo.setCurrentIndex(idx)
        current_row.addWidget(self.current_mission_mode_combo)
        current_row.addStretch()
        mission_layout.addLayout(current_row)
        mission_group.setLayout(mission_layout)
        layout.addWidget(mission_group)

        # Email Alert Settings
        email_alert_group = QGroupBox("Email Alert Configuration")
        email_alert_layout = QVBoxLayout()

        self.email_alert_checkbox = QCheckBox("Enable Email Alerts for This Tab")
        if self.existing_config:
            self.email_alert_checkbox.setChecked(self.existing_config.get('email_alerts_enabled', False))
        email_alert_layout.addWidget(self.email_alert_checkbox)

        recipient_layout = QHBoxLayout()
        recipient_layout.addWidget(QLabel("Alert Recipient Emails:"))
        self.alert_email_dropdown_btn = QToolButton()
        self.alert_email_dropdown_btn.setText("Select Recipients")
        self.alert_email_dropdown_btn.setPopupMode(QToolButton.InstantPopup)
        self.alert_email_dropdown_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.alert_email_dropdown_menu = QMenu(self)
        self.alert_email_dropdown_btn.setMenu(self.alert_email_dropdown_menu)
        recipient_layout.addWidget(self.alert_email_dropdown_btn)
        self.alert_email_selected_label = QLabel("No recipient selected")
        self.alert_email_selected_label.setWordWrap(True)
        recipient_layout.addWidget(self.alert_email_selected_label, 1)
        email_alert_layout.addLayout(recipient_layout)

        self.selected_alert_emails = []
        self._refresh_alert_email_dropdown()

        email_alert_group.setLayout(email_alert_layout)
        layout.addWidget(email_alert_group)

        scroll_area.setWidget(content_widget)
        dialog_layout.addWidget(scroll_area, 1)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_config)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        dialog_layout.addLayout(button_layout)
    
    def browse_data_folder(self):
        """Browse for data folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.data_folder_input.setText(folder)
            self._refresh_feature_checklist()

    def _refresh_feature_checklist(self):
        """Populate optional per-tab feature checklist from latest data file."""
        self.features_list.clear()
        folder = self.data_folder_input.text().strip()
        if not folder or not os.path.isdir(folder):
            return
        ext = ".csv" if self.file_type_combo.currentText() == "CSV" else ".json"
        files = list(Path(folder).glob(f"*{ext}"))
        if not files:
            return
        latest = max(files, key=os.path.getmtime)
        try:
            if ext == ".csv":
                sample = pd.read_csv(latest, nrows=200)
            else:
                sample = pd.read_json(latest)
                if isinstance(sample, list):
                    sample = pd.DataFrame(sample)
        except Exception:
            return
        saved = set(self.existing_config.get("selected_features", []) if self.existing_config else [])
        use_saved = bool(saved)
        for col in sample.select_dtypes(include=[np.number]).columns:
            item = QListWidgetItem(str(col))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            checked = (col in saved) if use_saved else True
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.features_list.addItem(item)

    def _on_input_mode_changed(self, mode_text):
        """Toggle stream settings visibility by source mode."""
        if hasattr(self, "active_device_monitoring_checkbox") and self.active_device_monitoring_checkbox.isChecked():
            self.stream_settings_widget.setVisible(False)
            return
        is_csv = mode_text == "CSV Polling"
        self.data_folder_input.setEnabled(is_csv)
        self.stream_settings_widget.setVisible(not is_csv)

    def _on_device_monitoring_toggled(self, enabled):
        """Switch between active serial ingestion and passive file monitoring."""
        self.serial_settings_widget.setVisible(bool(enabled))
        self.input_mode_combo.setEnabled(False)
        self.data_folder_input.setEnabled(not bool(enabled))
        if hasattr(self, "browse_folder_btn"):
            self.browse_folder_btn.setEnabled(not bool(enabled))
        self.file_type_combo.setEnabled(not bool(enabled))
        self.dataset_mode_combo.setEnabled(not bool(enabled))
        self.features_list.setEnabled(not bool(enabled))
        self.input_mode_combo.setCurrentText("CSV Polling")
        self.stream_settings_widget.setVisible(False)

    def _on_schedule_type_changed(self, schedule_type):
        """Adapt schedule controls by mode."""
        self.interval_widget.setVisible(schedule_type == "Continuous")
        self.utc_schedule_widget.setVisible(schedule_type == "Scheduled")

    def _get_registered_user_emails(self):
        """Collect registered user emails from UserManager for dropdown."""
        emails = []
        user_manager = getattr(self.parent(), "user_manager", None)
        if user_manager and hasattr(user_manager, "users"):
            for _username, user_data in (user_manager.users or {}).items():
                email = str(user_data.get("email", "")).strip()
                if _is_deliverable_alert_email(email):
                    emails.append(email)
        # Unique sorted
        return sorted(set(emails), key=lambda e: e.lower())

    def _update_alert_email_selection_label(self):
        if not self.selected_alert_emails:
            self.alert_email_selected_label.setText("No recipient selected")
            return
        preview = ", ".join(self.selected_alert_emails[:3])
        if len(self.selected_alert_emails) > 3:
            preview += f" (+{len(self.selected_alert_emails) - 3} more)"
        self.alert_email_selected_label.setText(preview)

    def _on_alert_email_action_toggled(self, checked):
        action = self.sender()
        if action is None:
            return
        email = action.data()
        if not email:
            return
        if checked and email not in self.selected_alert_emails:
            self.selected_alert_emails.append(email)
        elif (not checked) and email in self.selected_alert_emails:
            self.selected_alert_emails.remove(email)
        self._update_alert_email_selection_label()

    def _refresh_alert_email_dropdown(self):
        """Build recipient dropdown with registered emails and multi-selection."""
        self.alert_email_dropdown_menu.clear()
        registered_emails = self._get_registered_user_emails()

        # Preload saved values when editing existing config.
        if self.existing_config and not self.selected_alert_emails:
            saved_list = self.existing_config.get("alert_recipient_emails", [])
            if not saved_list:
                raw = str(self.existing_config.get("alert_recipient_email", "")).strip()
                saved_list = [e.strip() for e in raw.split(",") if e.strip()]
            self.selected_alert_emails = list(saved_list)

        if not registered_emails:
            placeholder = QAction("No registered user emails", self)
            placeholder.setEnabled(False)
            self.alert_email_dropdown_menu.addAction(placeholder)
            self._update_alert_email_selection_label()
            return

        for email in registered_emails:
            action = QAction(email, self)
            action.setCheckable(True)
            action.setData(email)
            action.setChecked(email in self.selected_alert_emails)
            action.toggled.connect(self._on_alert_email_action_toggled)
            self.alert_email_dropdown_menu.addAction(action)

        self._update_alert_email_selection_label()

    def _selected_model_names(self):
        """Return checked model names from the multi-select list."""
        names = []
        for i in range(self.model_multi_list.count()):
            item = self.model_multi_list.item(i)
            if item.checkState() == Qt.Checked:
                stored = item.data(Qt.UserRole)
                names.append(catalog_display_name(stored or item.text()))
        return names

    def _select_all_initial_models(self):
        for i in range(self.model_multi_list.count()):
            self.model_multi_list.item(i).setCheckState(Qt.Checked)

    def _clear_all_initial_models(self):
        for i in range(self.model_multi_list.count()):
            self.model_multi_list.item(i).setCheckState(Qt.Unchecked)

    def _sync_primary_model_combo_with_selection(self):
        """Keep primary model combo aligned with selected models."""
        selected = self._selected_model_names()
        current = self.model_combo.currentText()

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if selected:
            self.model_combo.addItems(selected)
            if current in selected:
                self.model_combo.setCurrentText(current)
            else:
                self.model_combo.setCurrentIndex(0)
        else:
            self.model_combo.addItems(get_supported_model_names())
            if current:
                idx = self.model_combo.findText(current)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _on_model_selection_changed(self, _item):
        self._sync_primary_model_combo_with_selection()
        self.update_model_parameters()

    def update_model_parameters(self):
        """Update model parameter inputs based on selected model"""
        while self.params_layout.rowCount() > 0:
            self.params_layout.removeRow(0)
        self.params_widgets.clear()

        model_type = self.model_combo.currentText()

        existing_params = {}
        if self.existing_config:
            existing_params = self.existing_config.get('model_parameters', {})

        if model_type in ["Isolation Forest", "Enhanced Isolation Forest"]:
            n_estimators = QSpinBox()
            n_estimators.setRange(10, 1000)
            n_estimators.setValue(existing_params.get('n_estimators', 100))
            self.params_layout.addRow("Number of Estimators:", n_estimators)
            self.params_widgets['n_estimators'] = n_estimators

            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setSingleStep(0.01)
            contamination.setValue(existing_params.get('contamination', 0.1))
            contamination.setDecimals(2)
            self.params_layout.addRow("Contamination:", contamination)
            self.params_widgets['contamination'] = contamination

        elif model_type in ["LSTM", "GRU", "Autoencoder"]:
            epochs = QSpinBox()
            epochs.setRange(1, 500)
            epochs.setValue(existing_params.get('epochs', 50))
            self.params_layout.addRow("Epochs:", epochs)
            self.params_widgets['epochs'] = epochs

            batch_size = QSpinBox()
            batch_size.setRange(8, 256)
            batch_size.setValue(existing_params.get('batch_size', 32))
            self.params_layout.addRow("Batch Size:", batch_size)
            self.params_widgets['batch_size'] = batch_size

            if model_type in ["LSTM", "GRU"]:
                lstm_units = QSpinBox()
                lstm_units.setRange(16, 256)
                lstm_units.setValue(existing_params.get('lstm_units', 64))
                self.params_layout.addRow("LSTM/GRU Units:", lstm_units)
                self.params_widgets['lstm_units'] = lstm_units

                seq_length = QSpinBox()
                seq_length.setRange(5, 100)
                seq_length.setValue(existing_params.get('sequence_length', 10))
                self.params_layout.addRow("Sequence Length:", seq_length)
                self.params_widgets['sequence_length'] = seq_length

        elif model_type in ["XGBoost", "XGBoost RUL"]:
            n_estimators = QSpinBox()
            n_estimators.setRange(10, 1000)
            n_estimators.setValue(existing_params.get('n_estimators', 100))
            self.params_layout.addRow("Number of Estimators:", n_estimators)
            self.params_widgets['n_estimators'] = n_estimators

            max_depth = QSpinBox()
            max_depth.setRange(1, 20)
            max_depth.setValue(existing_params.get('max_depth', 6))
            self.params_layout.addRow("Max Depth:", max_depth)
            self.params_widgets['max_depth'] = max_depth

            learning_rate = QDoubleSpinBox()
            learning_rate.setRange(0.01, 1.0)
            learning_rate.setSingleStep(0.01)
            learning_rate.setValue(existing_params.get('learning_rate', 0.1))
            learning_rate.setDecimals(2)
            self.params_layout.addRow("Learning Rate:", learning_rate)
            self.params_widgets['learning_rate'] = learning_rate

        elif model_type in ["Random Forest", "Random Forest RUL"]:
            n_estimators = QSpinBox()
            n_estimators.setRange(10, 1000)
            n_estimators.setValue(existing_params.get('n_estimators', 200))
            self.params_layout.addRow("Number of Trees:", n_estimators)
            self.params_widgets['n_estimators'] = n_estimators

            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setSingleStep(0.01)
            contamination.setValue(existing_params.get('contamination', 0.1))
            contamination.setDecimals(2)
            self.params_layout.addRow("Contamination:", contamination)
            self.params_widgets['contamination'] = contamination

        elif model_type == "One-Class SVM":
            nu = QDoubleSpinBox()
            nu.setRange(0.001, 0.5)
            nu.setSingleStep(0.01)
            nu.setValue(existing_params.get('nu', 0.05))
            nu.setDecimals(3)
            self.params_layout.addRow("Nu:", nu)
            self.params_widgets['nu'] = nu

        elif model_type == "IQR (Interquartile Range)":
            iqr_factor = QDoubleSpinBox()
            iqr_factor.setRange(0.5, 5.0)
            iqr_factor.setSingleStep(0.1)
            iqr_factor.setValue(existing_params.get('iqr_factor', 1.5))
            iqr_factor.setDecimals(1)
            self.params_layout.addRow("IQR Factor:", iqr_factor)
            self.params_widgets['iqr_factor'] = iqr_factor

        elif model_type == "Z-Score":
            z_threshold = QDoubleSpinBox()
            z_threshold.setRange(1.0, 5.0)
            z_threshold.setSingleStep(0.1)
            z_threshold.setValue(existing_params.get('z_threshold', 3.0))
            z_threshold.setDecimals(1)
            self.params_layout.addRow("Z-Score Threshold:", z_threshold)
            self.params_widgets['z_threshold'] = z_threshold

        elif model_type == "Local Outlier Factor":
            n_neighbors = QSpinBox()
            n_neighbors.setRange(5, 100)
            n_neighbors.setValue(existing_params.get('n_neighbors', 20))
            self.params_layout.addRow("N Neighbors:", n_neighbors)
            self.params_widgets['n_neighbors'] = n_neighbors

            contamination = QDoubleSpinBox()
            contamination.setRange(0.01, 0.5)
            contamination.setSingleStep(0.01)
            contamination.setValue(existing_params.get('contamination', 0.1))
            contamination.setDecimals(2)
            self.params_layout.addRow("Contamination:", contamination)
            self.params_widgets['contamination'] = contamination

    def accept_config(self):
        """Accept and save configuration"""
        title = self.title_input.text().strip()
        if not title:
            QMessageBox.warning(self, "Validation Error", "Please enter a tab title.")
            return
        
        data_folder = self.data_folder_input.text().strip()
        input_mode = "CSV Polling"
        active_device_monitoring_enabled = self.active_device_monitoring_checkbox.isChecked()
        if not active_device_monitoring_enabled and input_mode == "CSV Polling":
            if not data_folder:
                QMessageBox.warning(self, "Validation Error", "Please select a data folder.")
                return
            if not os.path.exists(data_folder):
                QMessageBox.warning(self, "Validation Error", "Selected data folder does not exist.")
                return
        elif not active_device_monitoring_enabled:
            # Keep a stable fallback path for stream tabs.
            if not data_folder:
                data_folder = self.existing_config.get("data_folder", "") if self.existing_config else ""
        else:
            if not self.serial_port_input.text().strip():
                QMessageBox.warning(self, "Validation Error", "Please set Serial Port for active device monitoring.")
                return
        
        selected_models = self._selected_model_names()
        if not selected_models:
            QMessageBox.warning(self, "Validation Error", "Please select at least one model.")
            return

        primary_model = self.model_combo.currentText() if self.model_combo.currentText() else selected_models[0]
        if primary_model not in selected_models:
            primary_model = selected_models[0]

        model_params = {}
        for param_name, widget in self.params_widgets.items():
            if isinstance(widget, QSpinBox):
                model_params[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                model_params[param_name] = widget.value()
        
        schedule_type = self.schedule_type_combo.currentText()

        # Calculate interval in milliseconds (used only for continuous mode)
        interval_ms = (self.interval_hours.value() * 3600000 +
                       self.interval_minutes.value() * 60000 +
                       self.interval_seconds.value() * 1000)

        if schedule_type == "Continuous" and interval_ms == 0:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please set a valid interval for Continuous monitoring (greater than 0)."
            )
            return

        # On-Demand and Scheduled modes do not use interval execution.
        if schedule_type in ("On-Demand", "Scheduled"):
            interval_ms = 0
        
        # Validate email alert settings if enabled
        email_alerts_enabled = self.email_alert_checkbox.isChecked()
        alert_recipients = [e.strip() for e in self.selected_alert_emails if str(e).strip()]
        
        if email_alerts_enabled and not alert_recipients:
            QMessageBox.warning(self, "Validation Error", 
                              "Please select at least one alert recipient when email alerts are enabled.")
            return
        
        # Build models list (support multiple initial models)
        subsystem_name = self.subsystem_name_input.text().strip()

        existing_by_type = {}
        if self.existing_config and 'models' in self.existing_config:
            for existing_model in self.existing_config.get('models', []):
                m_type = existing_model.get('model_type')
                if m_type:
                    existing_by_type[m_type] = existing_model

        models_list = []
        for model_name in selected_models:
            existing_model = existing_by_type.get(model_name, {})
            model_entry = {
                'model_type': model_name,
                'model_parameters': model_params.copy() if model_name == primary_model else existing_model.get('model_parameters', {}),
                'model_id': existing_model.get('model_id', str(uuid.uuid4()))
            }
            models_list.append(model_entry)

        selected_features = []
        for i in range(self.features_list.count()):
            item = self.features_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_features.append(item.text())
        
        self.config = {
            'title': title,
            'data_folder': data_folder,
            'data_file_type': self.file_type_combo.currentText(),
            'dataset_mode': self.dataset_mode_combo.currentText(),
            'selected_features': selected_features,
            'input_mode': input_mode,
            'active_device_monitoring_enabled': active_device_monitoring_enabled,
            'serial_port': self.serial_port_input.text().strip(),
            'serial_baudrate': int(self.serial_baud_spin.value()),
            'serial_delimiter': (self.serial_delimiter_input.text() or ",")[:1],
            'serial_fields': [f.strip() for f in self.serial_fields_input.text().split(",") if f.strip()],
            'mqtt_broker': self.mqtt_broker_input.text().strip(),
            'mqtt_port': int(self.mqtt_port_input.value()),
            'mqtt_topic': self.mqtt_topic_input.text().strip(),
            'opcua_endpoint': self.opcua_endpoint_input.text().strip(),
            'opcua_nodes': self.opcua_nodes_input.text().strip(),
            'subsystem_name': subsystem_name if subsystem_name else None,
            'model_type': primary_model,  # Keep for backward compatibility
            'model_parameters': model_params,  # Keep for backward compatibility
            'models': models_list,  # New: list of multiple models
            'schedule_type': schedule_type,
            'interval_ms': interval_ms,
            'schedule_utc_hour': int(self.utc_hour_spin.value()),
            'schedule_utc_minute': int(self.utc_minute_spin.value()),
            'max_training_rows': int(self.max_training_rows_spin.value()),
            'monitoring_window_rows': int(self.monitoring_window_spin.value()),
            'email_alerts_enabled': email_alerts_enabled,
            'alert_recipient_email': ", ".join(alert_recipients) if email_alerts_enabled else '',
            'alert_recipient_emails': alert_recipients if email_alerts_enabled else [],
            'created_at': datetime.datetime.now().isoformat() if not self.existing_config else self.existing_config.get('created_at'),
            'updated_at': datetime.datetime.now().isoformat()
        }

        from app.models.fsm import DEFAULT_MISSION_MODES, ensure_fsm_fields

        mission_modes = []
        for default_mode in DEFAULT_MISSION_MODES:
            name = default_mode["name"]
            spin = self.mission_mode_scale_spins.get(name)
            scale = float(spin.value()) if spin is not None else float(default_mode["threshold_scale"])
            mission_modes.append(
                {
                    "name": name,
                    "threshold_scale": scale,
                    "expected_deviation_multiplier": scale,
                    "description": default_mode.get("description", ""),
                    "expected_patterns": {},
                }
            )
        self.config["mission_modes"] = mission_modes
        self.config["current_mission_mode"] = self.current_mission_mode_combo.currentText()
        ensure_fsm_fields(self.config)

        self.accept()


# CUSTOM MONITORING TAB (app.tabs.custom_tab — Slice B)
from app.tabs.custom_tab import (
    CustomMonitoringTab,
    TabMonitoringWorker,
    TabTrainWorker,
    configure_custom_tab,
)


def _make_legacy_panel_slots(tool):
    from app.tabs.custom_tab.panels.slots import LegacyPanelSlots

    return LegacyPanelSlots(
        tool_window=tool,
        ensure_panel_helpers=tool._ensure_tab_panel_helpers,
        data_slot=tool._data_slot,
        data_import_stop_slot=tool._data_import_stop_slot,
        analysis_slot=tool._analysis_slot,
        viz_slot=tool._viz_slot,
        invoke_analysis_method=tool._invoke_tab_analysis_method,
        invoke_visualization_method=tool._invoke_tab_visualization_method,
        get_supported_model_names=get_supported_model_names,
        mpl_canvas_cls=MplCanvas,
    )


configure_custom_tab(
    DATA_DIR=DATA_DIR,
    REPORTS_DIR=REPORTS_DIR,
    SecureAnomalyDetectionTool=SecureAnomalyDetectionTool,
    DataProcessor=DataProcessor,
    AnomalyDetectionModel=AnomalyDetectionModel,
    EnhancedAnomalyDetectionModel=EnhancedAnomalyDetectionModel,
    MplCanvas=MplCanvas,
    format_registry_error=format_registry_error,
    to_internal_model_type=to_internal_model_type,
    AddModelDialog=AddModelDialog,
    TabConfigurationDialog=TabConfigurationDialog,
    safe_pickle_load=safe_pickle_load,
    build_legacy_panel_slots=_make_legacy_panel_slots,
)


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        
        # Set application name and organization for settings
        app.setApplicationName("SDA v4.0")
        app.setOrganizationName("Azercosmos")
        
        # Initialize main window with exception handling
        try:
            window = SecureAnomalyDetectionTool()
            window.show()
        except Exception as e:
            logger.critical(f"Failed to initialize application: {str(e)}")
            import traceback
            logger.critical(f"Traceback: {traceback.format_exc()}")
            
            # Show error dialog
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText("Application Error")
            error_dialog.setInformativeText(f"Failed to initialize the application: {str(e)}")
            error_dialog.setDetailedText(traceback.format_exc())
            error_dialog.setStandardButtons(QMessageBox.Ok)
            error_dialog.exec_()
            sys.exit(1)
            
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {str(e)}")
