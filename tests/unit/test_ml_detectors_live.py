"""Live train/predict checks for production anomaly detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.anomaly_engine import SUPPORTED_BACKENDS, AnomalyModel
from app.models.detectors import AnomalyDetectionModel
from app.models.model_types import resolve_runtime_model_type, to_internal_model_type


def _telemetry(n: int = 240, n_outliers: int = 12, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
            "voltage": rng.normal(28.0, 0.2, n),
            "current": rng.normal(2.0, 0.07, n),
            "temp": rng.normal(25.0, 0.6, n),
        }
    )
    labels = np.zeros(n, dtype=bool)
    idx = np.arange(n - n_outliers, n)
    df.loc[idx, ["voltage", "current", "temp"]] = [70.0, 16.0, 80.0]
    labels[idx] = True
    return df, labels


@pytest.mark.parametrize("backend", sorted(SUPPORTED_BACKENDS))
def test_anomaly_engine_train_predict_flags_outliers(backend):
    df, labels = _telemetry()
    train_df = df.iloc[:180]
    model = AnomalyModel(backend, {"n_estimators": 30, "n_neighbors": 12, "contamination": 0.1})
    ok, msg = model.train(train_df)
    assert ok, msg
    scores, anomalies = model.predict(df)
    assert scores is not None and anomalies is not None
    assert len(scores) == len(df)
    assert int(np.sum(anomalies[labels])) >= 1


@pytest.mark.parametrize(
    "model_type,kwargs",
    [
        ("isolation_forest", {"n_estimators": 30, "contamination": 0.1}),
        ("lof", {"n_neighbors": 12, "contamination": 0.1}),
        ("ocsvm", {"nu": 0.1}),
        ("random_forest", {"n_estimators": 30, "contamination": 0.1}),
        ("xgboost", {"n_estimators": 30, "max_depth": 3, "contamination": 0.1}),
        ("iqr_(interquartile_range)", {"iqr_factor": 1.5}),
        ("z-score", {"threshold": 3.0}),
    ],
)
def test_production_detector_train_predict(model_type, kwargs):
    df, labels = _telemetry()
    model = AnomalyDetectionModel(model_type)
    ok, msg = model.train(df.iloc[:180], **kwargs)
    if model_type == "xgboost" and not ok and "not installed" in str(msg).lower():
        pytest.skip(msg)
    assert ok, msg
    scores, anomalies = model.predict(df)
    assert scores is not None
    assert not isinstance(anomalies, str), anomalies
    assert len(np.asarray(scores)) == len(df)
    assert int(np.sum(np.asarray(anomalies)[labels])) >= 1


def test_alias_resolution_maps_advanced_names_to_real_backends():
    assert resolve_runtime_model_type(to_internal_model_type("LSTM-AE")) == "lstm"
    assert resolve_runtime_model_type(to_internal_model_type("TimesFM")) == "prophet"
    assert resolve_runtime_model_type(to_internal_model_type("Isolation Forest")) == "isolation_forest"
    assert to_internal_model_type("TimesFM  (uses Prophet)") == "timesfm"


def test_enhanced_isolation_forest_trains_without_labels():
    from app.models.detectors import EnhancedAnomalyDetectionModel

    df, labels = _telemetry()
    model = EnhancedAnomalyDetectionModel("enhanced_isolation_forest")
    ok, msg = model.train(df.iloc[:180], use_hyperparameter_tuning=False, n_estimators=40)
    assert ok, msg
    scores, anomalies = model.predict(df)
    assert scores is not None
    assert not isinstance(anomalies, str), anomalies
    assert int(np.sum(np.asarray(anomalies)[labels])) >= 1


@pytest.mark.parametrize("model_type", ["z-score", "iqr_(interquartile_range)"])
def test_statistical_model_save_load_roundtrip(tmp_path, model_type):
    from app.security.pickle_safe import configure_trusted_pickle_roots

    configure_trusted_pickle_roots(str(tmp_path))
    df, labels = _telemetry()
    model = AnomalyDetectionModel(model_type)
    ok, msg = model.train(df.iloc[:180])
    assert ok, msg
    path = tmp_path / f"{model_type}.pkl"
    saved, save_msg = model.save(str(path))
    assert saved, save_msg
    loaded, load_msg = AnomalyDetectionModel.load(str(path))
    assert loaded is not None, load_msg
    scores, anomalies = loaded.predict(df)
    assert scores is not None
    assert not isinstance(anomalies, str), anomalies
    assert int(np.sum(np.asarray(anomalies)[labels])) >= 1


def test_prophet_alias_payload_loads_without_prophet_type_key(tmp_path):
    from app.security.pickle_safe import configure_trusted_pickle_roots

    configure_trusted_pickle_roots(str(tmp_path))
    path = tmp_path / "arima.pkl"
    payload = {
        "model_type": "arima",
        "prophet_model": "fitted-stub",
        "target_column": "voltage",
        "timestamp_column": "timestamp",
    }
    import pickle

    with open(path, "wb") as handle:
        pickle.dump(payload, handle)

    from app.models.detectors import PROPHET_AVAILABLE

    loaded, load_msg = AnomalyDetectionModel.load(str(path))
    if not PROPHET_AVAILABLE:
        assert loaded is None
        assert "Prophet" in str(load_msg)
        return
    assert loaded is not None, load_msg
    assert loaded.model == "fitted-stub"
    assert loaded.resolved_model_type == "prophet"
    assert loaded.target_column == "voltage"


def test_prophet_predict_accepts_voltage_when_trained_on_value():
    from app.models.detectors import PROPHET_AVAILABLE

    if not PROPHET_AVAILABLE:
        pytest.skip("Prophet not installed")
    df, labels = _telemetry()
    train = df.iloc[:180].rename(columns={"voltage": "value"})
    model = AnomalyDetectionModel("prophet")
    ok, msg = model.train(
        train,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    assert ok, msg
    scores, anomalies = model.predict(df)
    assert scores is not None
    assert not isinstance(anomalies, str), anomalies
    assert len(np.asarray(scores)) == len(df)
