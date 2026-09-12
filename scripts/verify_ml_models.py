"""Live check: train + predict for STDMS ML backends and saved custom-tab models."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_telemetry(n: int = 360, n_outliers: int = 18, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "voltage": rng.normal(28.0, 0.25, n),
            "current": rng.normal(2.1, 0.08, n),
            "temp": rng.normal(25.0, 0.8, n),
        }
    )
    labels = np.zeros(n, dtype=bool)
    outlier_idx = np.arange(n - n_outliers, n)
    df.loc[outlier_idx, "voltage"] = 72.0
    df.loc[outlier_idx, "current"] = 18.5
    df.loc[outlier_idx, "temp"] = 88.0
    labels[outlier_idx] = True
    return df, labels


def _score_quality(scores: np.ndarray, anomalies: np.ndarray, labels: np.ndarray) -> dict:
    scores = np.asarray(scores, dtype=float)
    anomalies = np.asarray(anomalies, dtype=bool)
    finite = np.isfinite(scores)
    scores = np.where(finite, scores, np.nanmin(scores[finite]) if finite.any() else 0.0)
    normal = scores[~labels]
    outlier = scores[labels]
    mean_normal = float(np.mean(normal)) if len(normal) else float("nan")
    mean_outlier = float(np.mean(outlier)) if len(outlier) else float("nan")
    recall = float(np.mean(anomalies[labels])) if labels.any() else float("nan")
    flagged = int(np.sum(anomalies))
    # Higher score should mean more anomalous for most backends.
    # IsolationForest decision_function is inverted in AnomalyModel but not in AnomalyDetectionModel.
    separated = bool(np.isfinite(mean_outlier) and np.isfinite(mean_normal) and mean_outlier != mean_normal)
    return {
        "n": int(len(scores)),
        "flagged": flagged,
        "outlier_recall": round(recall, 3),
        "mean_score_normal": round(mean_normal, 4),
        "mean_score_outlier": round(mean_outlier, 4),
        "score_gap": round(mean_outlier - mean_normal, 4),
        "separated": separated,
    }


def _ok_predict(scores, anomalies, n_rows: int) -> tuple[bool, str]:
    if scores is None or anomalies is None:
        return False, f"predict returned empty: scores={scores!r} anomalies={anomalies!r}"
    if isinstance(anomalies, str):
        return False, f"predict error: {anomalies}"
    scores = np.asarray(scores)
    anomalies = np.asarray(anomalies)
    if len(scores) != n_rows or len(anomalies) != n_rows:
        return False, f"length mismatch scores={len(scores)} anomalies={len(anomalies)} expected={n_rows}"
    return True, "ok"


def check_anomaly_engine(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    from app.models.anomaly_engine import AnomalyModel, SUPPORTED_BACKENDS

    train_df = df.iloc[: int(len(df) * 0.75)].copy()
    rows = []
    for backend in sorted(SUPPORTED_BACKENDS):
        started = time.perf_counter()
        model = AnomalyModel(backend, {"n_estimators": 40, "n_neighbors": 15, "contamination": 0.08})
        ok, msg = model.train(train_df)
        if not ok:
            rows.append({"layer": "anomaly_engine", "model": backend, "ok": False, "detail": msg})
            continue
        scores, anomalies = model.predict(df)
        pred_ok, pred_msg = _ok_predict(scores, anomalies, len(df))
        quality = _score_quality(scores, anomalies, labels) if pred_ok else {}
        rows.append(
            {
                "layer": "anomaly_engine",
                "model": backend,
                "ok": pred_ok,
                "detail": pred_msg if not pred_ok else f"trained; {quality}",
                "seconds": round(time.perf_counter() - started, 2),
                **quality,
            }
        )
    return rows


def check_production_detectors(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    from app.models.detectors import (
        PROPHET_AVAILABLE,
        TENSORFLOW_AVAILABLE,
        AnomalyDetectionModel,
        EnhancedAnomalyDetectionModel,
    )

    train_df = df.iloc[: int(len(df) * 0.75)].copy()
    cores = [
        ("isolation_forest", {"n_estimators": 50, "contamination": 0.08}),
        ("lof", {"n_neighbors": 15, "contamination": 0.08}),
        ("ocsvm", {"nu": 0.08}),
        ("random_forest", {"n_estimators": 40, "contamination": 0.08}),
        ("xgboost", {"n_estimators": 40, "max_depth": 3, "contamination": 0.08}),
        ("iqr_(interquartile_range)", {"iqr_factor": 1.5}),
        ("z-score", {"threshold": 3.0}),
    ]
    optional = []
    if PROPHET_AVAILABLE:
        optional.append(("prophet", {"yearly_seasonality": False, "weekly_seasonality": False, "daily_seasonality": False}))
    if TENSORFLOW_AVAILABLE:
        dl_kwargs = {"epochs": 2, "batch_size": 32, "patience": 1, "validation_split": 0.2}
        optional.extend(
            [
                ("autoencoder", dl_kwargs),
                ("lstm", dl_kwargs),
                ("gru", dl_kwargs),
            ]
        )

    rows = []
    for model_type, kwargs in cores + optional:
        started = time.perf_counter()
        try:
            model = AnomalyDetectionModel(model_type)
            ok, msg = model.train(train_df, **kwargs)
            if not ok:
                rows.append(
                    {
                        "layer": "detectors",
                        "model": model_type,
                        "ok": False,
                        "detail": msg,
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                )
                continue
            scores, anomalies = model.predict(df)
            pred_ok, pred_msg = _ok_predict(scores, anomalies, len(df))
            quality = _score_quality(scores, anomalies, labels) if pred_ok else {}
            rows.append(
                {
                    "layer": "detectors",
                    "model": model_type,
                    "ok": pred_ok,
                    "detail": pred_msg if not pred_ok else f"{msg}; {quality}",
                    "seconds": round(time.perf_counter() - started, 2),
                    **quality,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "layer": "detectors",
                    "model": model_type,
                    "ok": False,
                    "detail": f"{exc}\n{traceback.format_exc(limit=2)}",
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )

    started = time.perf_counter()
    try:
        enhanced = EnhancedAnomalyDetectionModel("enhanced_isolation_forest")
        ok, msg = enhanced.train(train_df, use_hyperparameter_tuning=False, n_estimators=50)
        if not ok:
            rows.append(
                {
                    "layer": "detectors",
                    "model": "enhanced_isolation_forest",
                    "ok": False,
                    "detail": msg,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
        else:
            scores, anomalies = enhanced.predict(df)
            pred_ok, pred_msg = _ok_predict(scores, anomalies, len(df))
            quality = _score_quality(scores, anomalies, labels) if pred_ok else {}
            rows.append(
                {
                    "layer": "detectors",
                    "model": "enhanced_isolation_forest",
                    "ok": pred_ok,
                    "detail": pred_msg if not pred_ok else f"{msg}; {quality}",
                    "seconds": round(time.perf_counter() - started, 2),
                    **quality,
                }
            )
    except Exception as exc:
        rows.append(
            {
                "layer": "detectors",
                "model": "enhanced_isolation_forest",
                "ok": False,
                "detail": f"{exc}",
                "seconds": round(time.perf_counter() - started, 2),
            }
        )

    if not PROPHET_AVAILABLE:
        rows.append({"layer": "detectors", "model": "prophet", "ok": False, "detail": "SKIP: Prophet not installed"})
    if not TENSORFLOW_AVAILABLE:
        for name in ("autoencoder", "lstm", "gru"):
            rows.append({"layer": "detectors", "model": name, "ok": False, "detail": "SKIP: TensorFlow not installed"})

    return rows


def check_alias_fallbacks() -> list[dict]:
    from app.models.model_types import (
        TOUCHED_44_MODEL_NAMES,
        resolve_runtime_model_type,
        to_internal_model_type,
    )

    real_backends = {
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
    }
    rows = []
    native = 0
    fallback = 0
    for display in TOUCHED_44_MODEL_NAMES:
        internal = to_internal_model_type(display)
        runtime = resolve_runtime_model_type(internal)
        is_native = internal in real_backends and runtime == internal
        if is_native:
            native += 1
        else:
            fallback += 1
        rows.append(
            {
                "layer": "catalog",
                "model": display,
                "ok": runtime in real_backends,
                "detail": f"{internal} -> {runtime}" + ("" if is_native else " (fallback)"),
            }
        )
    rows.append(
        {
            "layer": "catalog",
            "model": "_summary",
            "ok": True,
            "detail": f"{native} native backends, {fallback} aliases fall back to another backend, {len(TOUCHED_44_MODEL_NAMES)} UI names",
        }
    )
    return rows


def check_saved_custom_tab_models(df: pd.DataFrame) -> list[dict]:
    from app.models.detectors import AnomalyDetectionModel

    rows = []
    manifests = list((ROOT / "data" / "custom_tabs").glob("*/models/manifest.json"))
    if not manifests:
        rows.append({"layer": "saved", "model": "(none)", "ok": False, "detail": "No custom-tab manifests found"})
        return rows

    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"layer": "saved", "model": str(manifest_path), "ok": False, "detail": str(exc)})
            continue
        for model_id, meta in payload.items():
            model_type = meta.get("model_type", model_id)
            rel = meta.get("filepath") or ""
            path = ROOT / rel if rel else manifest_path.parent / f"{model_id}.pkl"
            if not path.is_file():
                # try next to manifest
                alt = manifest_path.parent / f"{model_id}.pkl"
                path = alt if alt.is_file() else path
            if not path.is_file():
                rows.append(
                    {
                        "layer": "saved",
                        "model": f"{model_type} ({model_id[:8]})",
                        "ok": False,
                        "detail": f"file missing: {path}",
                    }
                )
                continue
            try:
                loaded, msg = AnomalyDetectionModel.load(str(path))
                if loaded is None:
                    rows.append({"layer": "saved", "model": model_type, "ok": False, "detail": msg})
                    continue
                scores, anomalies = loaded.predict(df)
                pred_ok, pred_msg = _ok_predict(scores, anomalies, len(df))
                rows.append(
                    {
                        "layer": "saved",
                        "model": f"{model_type} ({path.name})",
                        "ok": pred_ok,
                        "detail": pred_msg if not pred_ok else f"loaded + predict ok; flagged={int(np.sum(anomalies))}",
                    }
                )
            except Exception as exc:
                rows.append({"layer": "saved", "model": model_type, "ok": False, "detail": str(exc)})
    return rows


def check_support_modules() -> list[dict]:
    from app.models.drift import TabConceptDriftChecker, compute_psi
    from app.models.forecast import estimate_ttf_hours, short_forecast
    from app.models.health_scoring import compute_health_score

    rows = []
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 300)
    shifted = rng.normal(2.5, 1.0, 300)
    psi = compute_psi(ref, shifted)
    rows.append({"layer": "support", "model": "drift.compute_psi", "ok": psi > 0.1, "detail": f"psi={psi:.3f}"})

    checker = TabConceptDriftChecker(reference_window_size=80)
    df = pd.DataFrame({"temp": np.linspace(0, 1, 120)})
    first = checker.check(df)
    rows.append(
        {
            "layer": "support",
            "model": "drift.TabConceptDriftChecker",
            "ok": first.get("status") == "reference_initialized",
            "detail": str(first),
        }
    )

    forecast = short_forecast(np.linspace(0.1, 0.4, 30), steps=5)
    ttf = estimate_ttf_hours(np.linspace(0.1, 0.4, 40), 0.6, 0.9)
    rows.append(
        {
            "layer": "support",
            "model": "forecast.short_forecast",
            "ok": forecast is not None,
            "detail": f"forecast={forecast}",
        }
    )
    rows.append(
        {
            "layer": "support",
            "model": "forecast.estimate_ttf_hours",
            "ok": ttf is not None and ttf > 0,
            "detail": f"ttf={ttf}",
        }
    )
    score = compute_health_score(0.5, 1.0)
    rows.append(
        {
            "layer": "support",
            "model": "health_scoring.compute_health_score",
            "ok": 0.49 <= score <= 0.51,
            "detail": f"score={score}",
        }
    )
    return rows


def _print_table(rows: list[dict]) -> None:
    for row in rows:
        mark = "PASS" if row.get("ok") else "FAIL"
        extra = ""
        if "outlier_recall" in row:
            extra = f"  recall={row['outlier_recall']} flagged={row.get('flagged')} gap={row.get('score_gap')}"
        seconds = f"  {row['seconds']}s" if "seconds" in row else ""
        print(f"  [{mark}] {row.get('layer')}/{row.get('model')}{seconds}{extra}")
        detail = str(row.get("detail") or "")
        if detail and (not row.get("ok") or row.get("model") == "_summary"):
            for line in detail.strip().splitlines()[:4]:
                print(f"         {line}")


def main() -> int:
    print("STDMS ML model live check")
    print("=" * 72)
    df, labels = _make_telemetry()
    print(f"Synthetic telemetry: {len(df)} rows, {int(labels.sum())} injected outliers")

    all_rows: list[dict] = []
    sections = [
        ("1) Modular AnomalyModel backends", lambda: check_anomaly_engine(df, labels)),
        ("2) Production AnomalyDetectionModel", lambda: check_production_detectors(df, labels)),
        ("3) UI catalog aliases (44 names)", check_alias_fallbacks),
        ("4) Saved custom-tab .pkl models", lambda: check_saved_custom_tab_models(df)),
        ("5) Drift / forecast / health helpers", check_support_modules),
    ]

    for title, fn in sections:
        print(f"\n{title}")
        print("-" * 72)
        try:
            rows = fn()
        except Exception as exc:
            print(f"  [FAIL] section crashed: {exc}")
            traceback.print_exc()
            rows = [{"layer": "section", "model": title, "ok": False, "detail": str(exc)}]
        _print_table(rows)
        all_rows.extend(rows)

    checked = [r for r in all_rows if r.get("model") != "_summary"]
    passed = sum(1 for r in checked if r.get("ok"))
    failed = [r for r in checked if not r.get("ok")]
    print("\n" + "=" * 72)
    print(f"RESULT: {passed}/{len(checked)} checks passed, {len(failed)} failed/skipped")
    return 0 if not any(r.get("layer") in {"anomaly_engine", "detectors", "support"} and not r.get("ok") and "SKIP" not in str(r.get("detail")) for r in failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
