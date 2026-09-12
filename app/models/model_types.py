"""UI model catalog and display/internal mapping (Slice C Wave 1)."""
from __future__ import annotations

import re

# Unified model catalog and mapping layer.
# "Touched 44 models" are exposed in UI; advanced models are resolved to
# implemented runtime backends for stable operation.
TOUCHED_44_MODEL_NAMES = [
    "Isolation Forest",
    "Local Outlier Factor",
    "One-Class SVM",
    "Random Forest",
    "Incremental Isolation Forest",
    "XGBoost",
    "Autoencoder",
    "LSTM",
    "GRU",
    "LSTM-AE",
    "TCN-AE",
    "Transformer-AE",
    "VAE",
    "GDN",
    "MTAD-GAT",
    "Deep SVDD",
    "CNN",
    "ARIMA",
    "SARIMA",
    "Prophet",
    "IQR (Interquartile Range)",
    "Z-Score",
    "Weibull RUL",
    "Cox PH RUL",
    "XGBoost RUL",
    "Random Forest RUL",
    "TFT",
    "N-BEATS",
    "N-HiTS",
    "Bidirectional LSTM + MC Dropout",
    "PINNs",
    "DeepHit",
    "DRSA",
    "Mamba",
    "TS2Vec",
    "SimMTM",
    "TNC",
    "CoST",
    "DevNet",
    "REPEN",
    "TimesFM",
    "Moirai",
    "Lag-Llama",
    "Federated Learning",
]

MODEL_DISPLAY_TO_INTERNAL = {
    "Isolation Forest": "isolation_forest",
    "Enhanced Isolation Forest": "enhanced_isolation_forest",
    "Local Outlier Factor": "lof",
    "One-Class SVM": "ocsvm",
    "Ensemble Voting": "ensemble_voting",
    "Ensemble Stacking": "ensemble_stacking",
    "Adaptive Threshold Ensemble": "adaptive_threshold",
    "Random Forest": "random_forest",
    "Incremental Isolation Forest": "incremental_isolation_forest",
    "River Anomaly": "river_anomaly",
    "Reinforcement Learning Detector": "reinforcement_learning",
    "XGBoost": "xgboost",
    "Autoencoder": "autoencoder",
    "LSTM": "lstm",
    "GRU": "gru",
    "LSTM-AE": "lstm_ae",
    "TCN-AE": "tcn_ae",
    "Transformer-AE": "transformer_ae",
    "VAE": "vae",
    "GDN": "gdn",
    "MTAD-GAT": "mtad_gat",
    "Deep SVDD": "deep_svdd",
    "CNN": "cnn",
    "ARIMA": "arima",
    "SARIMA": "sarima",
    "Prophet": "prophet",
    "IQR (Interquartile Range)": "iqr_(interquartile_range)",
    "Z-Score": "z-score",
    "Weibull RUL": "weibull_rul",
    "Cox PH RUL": "cox_ph_rul",
    "XGBoost RUL": "xgboost_rul",
    "Random Forest RUL": "random_forest_rul",
    "TFT": "tft",
    "N-BEATS": "n_beats",
    "N-HiTS": "n_hits",
    "Bidirectional LSTM + MC Dropout": "bilstm_mc_dropout",
    "PINNs": "pinns",
    "DeepHit": "deephit",
    "DRSA": "drsa",
    "Mamba": "mamba",
    "TS2Vec": "ts2vec",
    "SimMTM": "simmtm",
    "TNC": "tnc",
    "CoST": "cost",
    "DevNet": "devnet",
    "REPEN": "repen",
    "TimesFM": "timesfm",
    "Moirai": "moirai",
    "Lag-Llama": "lag_llama",
    "Federated Learning": "federated_learning",
}

MODEL_INTERNAL_FALLBACK = {
    "incremental_isolation_forest": "isolation_forest",
    "enhanced_isolation_forest": "isolation_forest",
    "river_anomaly": "isolation_forest",
    "reinforcement_learning": "isolation_forest",
    "ensemble_voting": "isolation_forest",
    "ensemble_stacking": "random_forest",
    "adaptive_threshold": "isolation_forest",
    "lstm_ae": "lstm",
    "tcn_ae": "autoencoder",
    "transformer_ae": "autoencoder",
    "vae": "autoencoder",
    "gdn": "autoencoder",
    "mtad_gat": "autoencoder",
    "deep_svdd": "ocsvm",
    "cnn": "autoencoder",
    "arima": "prophet",
    "sarima": "prophet",
    "weibull_rul": "prophet",
    "cox_ph_rul": "xgboost",
    "xgboost_rul": "xgboost",
    "random_forest_rul": "random_forest",
    "tft": "prophet",
    "n_beats": "prophet",
    "n_hits": "prophet",
    "bilstm_mc_dropout": "lstm",
    "pinns": "prophet",
    "deephit": "xgboost",
    "drsa": "xgboost",
    "mamba": "lstm",
    "ts2vec": "isolation_forest",
    "simmtm": "isolation_forest",
    "tnc": "isolation_forest",
    "cost": "isolation_forest",
    "devnet": "isolation_forest",
    "repen": "isolation_forest",
    "timesfm": "prophet",
    "moirai": "prophet",
    "lag_llama": "prophet",
    "federated_learning": "isolation_forest",
}

NATIVE_RUNTIME_BACKENDS = frozenset({
    "isolation_forest",
    "lof",
    "ocsvm",
    "random_forest",
    "xgboost",
    "autoencoder",
    "lstm",
    "gru",
    "iqr_(interquartile_range)",
    "z-score",
    "prophet",
})

_CATALOG_USES_SUFFIX = re.compile(r"\s*\(uses [^)]+\)\s*$", re.IGNORECASE)


def catalog_display_name(model_display_name):
    """Strip UI fallback suffix so 'TimesFM  (uses Prophet)' -> 'TimesFM'."""
    return _CATALOG_USES_SUFFIX.sub("", str(model_display_name or "")).strip()


def runtime_backend_display_name(runtime_internal):
    """Map an executable backend key back to a UI name."""
    for display_name, internal in MODEL_DISPLAY_TO_INTERNAL.items():
        if internal == runtime_internal:
            return display_name
    return str(runtime_internal)


def format_model_catalog_label(display_name):
    """Show the real backend when the catalog name is only an alias."""
    clean = catalog_display_name(display_name)
    internal = to_internal_model_type(clean)
    runtime = resolve_runtime_model_type(internal)
    if runtime == internal:
        return clean
    return f"{clean}  (uses {runtime_backend_display_name(runtime)})"


def get_supported_model_names():
    """Return the UI model list for the 44 touched models."""
    return list(TOUCHED_44_MODEL_NAMES)


def to_internal_model_type(model_display_name):
    """Normalize display model name to internal key."""
    model_display_name = catalog_display_name(model_display_name)
    if model_display_name in MODEL_DISPLAY_TO_INTERNAL:
        return MODEL_DISPLAY_TO_INTERNAL[model_display_name]
    return str(model_display_name).strip().lower().replace(" ", "_")


def resolve_runtime_model_type(internal_model_type):
    """Resolve advanced aliases to executable backend model types."""
    return MODEL_INTERNAL_FALLBACK.get(internal_model_type, internal_model_type)
