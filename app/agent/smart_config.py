"""Data-folder inspection and ideal custom-tab config suggestions (Faza A)."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Optional

from app.monitoring.tab_config import DEFAULT_MODELS, default_tab_config

_TIMESTAMP_NAME_RE = re.compile(
    r"(^|_)(time|timestamp|datetime|date|epoch|utc|acquired)(_|$)|sample.?time",
    re.IGNORECASE,
)

_BATTERY_HINTS = (
    "battery",
    "batt",
    "voltage",
    "current",
    "temp",
    "temperature",
    "soc",
    "heater",
    "eclipse",
    "bus",
)

_DATA_FILE_SUFFIXES = {".csv", ".json", ".CSV", ".JSON"}


def _project_root() -> Path:
    """STDMS repo root (parent of ``app/``)."""
    return Path(__file__).resolve().parents[2]


def resolve_data_paths(path: str | Path) -> dict[str, Any]:
    """Resolve operator path into folder + optional pinned CSV source_file.

    Relative paths like ``data/BatteryTemperature.csv`` are tried against
    ``cwd`` then the project root. ``ok`` is True only when the folder exists.
    """
    raw_s = str(path or "").strip()
    if not raw_s:
        return {
            "ok": False,
            "data_folder": "",
            "source_file": "",
            "title_hint": "",
            "error": "empty_path",
        }

    raw = Path(raw_s).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw))
        candidates.append(_project_root() / raw)

    resolved: Optional[Path] = None
    for cand in candidates:
        try:
            p = cand.resolve()
        except OSError:
            continue
        if p.is_file() or p.is_dir():
            resolved = p
            break

    if resolved is None:
        try:
            best = candidates[0].resolve()
        except OSError:
            best = candidates[0] if candidates else raw
        looks_file = best.suffix in _DATA_FILE_SUFFIXES
        folder = best.parent if looks_file else best
        return {
            "ok": False,
            "data_folder": str(folder),
            "source_file": str(best) if looks_file else "",
            "title_hint": best.stem if looks_file else (best.name if best.name != "data" else ""),
            "error": "path_not_found",
        }

    source_file = ""
    folder = resolved
    if resolved.is_file():
        source_file = str(resolved)
        folder = resolved.parent
    title_hint = (
        resolved.stem
        if resolved.is_file()
        else (folder.name if folder.name != "data" else "")
    )
    return {
        "ok": folder.is_dir(),
        "data_folder": str(folder),
        "source_file": source_file,
        "title_hint": title_hint,
    }


def _safe_folder(path: str | Path) -> Path:
    return Path(resolve_data_paths(path)["data_folder"])


def _ideal_title_from_hint(hint: str, fallback: str = "Health Monitoring") -> str:
    import re

    stem = (hint or "").strip()
    if not stem or stem.lower() in ("data", "monitoring", ".", "health monitoring"):
        return fallback
    nice = re.sub(r"(?<!^)(?=[A-Z])", " ", stem).replace("_", " ").replace("-", " ")
    nice = re.sub(r"\s+", " ", nice).strip()
    if "monitor" not in nice.lower():
        nice = f"{nice} Monitoring"
    return nice


_WEAK_TITLES = frozenset(
    {
        "",
        "monitoring",
        "health monitoring",
        "new monitoring tab",
        "agent tab",
        "tab",
        "untitled",
    }
)


def is_weak_title(title: str) -> bool:
    return str(title or "").strip().lower() in _WEAK_TITLES


def _uniq(seq: list[str]) -> list[str]:
    out, seen = [], set()
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _is_timestamp_candidate(col: str, series: Any) -> bool:
    name = str(col)
    if _TIMESTAMP_NAME_RE.search(name):
        return True
    # Never treat pure numeric dtypes as timestamps (to_datetime(float) = epoch ns)
    try:
        import pandas as pd

        if pd.api.types.is_numeric_dtype(series):
            return False
        sample = series.dropna().head(20)
        if sample.empty:
            return False
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or str(series.dtype).startswith("datetime")
        ):
            return False
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        if float(parsed.notna().mean()) >= 0.8:
            return True
    except Exception:
        pass
    return False


def _sample_range(series: Any) -> Optional[dict[str, Any]]:
    try:
        import pandas as pd

        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        return {
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "mean": round(float(s.mean()), 6),
        }
    except Exception:
        return None


def inspect_data_folder(
    data_folder: str,
    *,
    max_files: int = 8,
    sample_rows: int = 5000,
    source_file: str = "",
) -> dict[str, Any]:
    """Profile CSV files in a folder for tab configuration (Faza A.1).

    If ``source_file`` (or a CSV path in ``data_folder``) is given, prefer that
    file so multi-CSV folders do not pollute feature selection.
    """
    try:
        import pandas as pd
    except ImportError:
        return {"ok": False, "error": "pandas_unavailable"}

    resolved = resolve_data_paths(source_file or data_folder)
    folder = Path(resolved["data_folder"])
    pinned = resolved.get("source_file") or ""
    if source_file:
        pinned = str(Path(source_file).expanduser().resolve()) if Path(source_file).expanduser().exists() else pinned

    if not folder.is_dir():
        return {
            "ok": False,
            "error": "folder_not_found",
            "data_folder": str(folder),
            "source_file": pinned,
        }

    csv_files = sorted(folder.glob("*.csv")) + sorted(folder.glob("*.CSV"))
    seen: set[str] = set()
    unique_files: list[Path] = []
    for f in csv_files:
        key = str(f).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_files.append(f)

    # Prefer pinned source file first (and optionally only that file for features)
    if pinned:
        pin_path = Path(pinned)
        if pin_path.is_file():
            unique_files = [pin_path] + [
                f for f in unique_files if f.resolve() != pin_path.resolve()
            ]
            # When operator pointed at one CSV, inspect that file primarily
            unique_files = unique_files[:1]

    if not unique_files:
        return {
            "ok": False,
            "error": "no_csv_files",
            "data_folder": str(folder),
            "source_file": pinned,
            "files": [],
            "columns": [],
            "numeric_cols": [],
            "timestamp_candidates": [],
            "row_count": 0,
            "null_rates": {},
            "sample_ranges": {},
        }

    files_out: list[dict[str, Any]] = []
    all_numeric: list[str] = []
    all_columns: list[str] = []
    all_timestamps: list[str] = []
    merged_null: dict[str, float] = {}
    merged_ranges: dict[str, dict[str, Any]] = {}
    total_rows = 0
    quality_notes: list[str] = []

    # Match GUI DataProcessor / DataReader column contract (time, value, value2, …)
    try:
        from app.ingestion.data_reader import DataReader
    except Exception:
        DataReader = None  # type: ignore

    for path in unique_files[: max(1, int(max_files))]:
        try:
            df_raw = pd.read_csv(path, nrows=int(sample_rows))
        except Exception as exc:
            files_out.append({"filename": path.name, "ok": False, "error": str(exc)})
            continue

        original_columns = [str(c) for c in df_raw.columns]
        rename_map: dict[str, str] = {}
        if DataReader is not None and len(df_raw.columns) >= 2:
            try:
                norm = DataReader._normalize_dataframe(df_raw)
                df = norm.dataframe
                rename_map = dict(norm.rename_map or {})
                if rename_map:
                    quality_notes.append(
                        f"{path.name}: normalized to GUI schema "
                        f"{list(df.columns)} (was {original_columns[:6]})"
                    )
            except Exception:
                df = df_raw
        else:
            df = df_raw

        columns = [str(c) for c in df.columns]
        numeric = [str(c) for c in df.select_dtypes(include="number").columns.tolist()]
        null_rates: dict[str, Optional[float]] = {}
        sample_ranges: dict[str, dict[str, Any]] = {}
        ts_cands: list[str] = []

        for c in df.columns:
            cname = str(c)
            try:
                null_rates[cname] = round(float(df[c].isna().mean()), 4)
            except Exception:
                null_rates[cname] = None
            if _is_timestamp_candidate(cname, df[c]):
                ts_cands.append(cname)
            rng = _sample_range(df[c])
            if rng is not None:
                sample_ranges[cname] = rng
                if cname not in numeric and cname not in ts_cands:
                    numeric.append(cname)

        # Guarantee "time" is treated as timestamp after GUI normalize
        if "time" in columns and "time" not in ts_cands:
            ts_cands.insert(0, "time")

        high_null = [c for c, r in null_rates.items() if r is not None and r > 0.3]
        if high_null:
            quality_notes.append(f"{path.name}: high null rate in {high_null[:8]}")

        rows = int(len(df))
        total_rows += rows
        all_numeric.extend(numeric)
        all_columns.extend(columns)
        all_timestamps.extend(ts_cands)
        for k, v in null_rates.items():
            if v is not None:
                merged_null[k] = max(merged_null.get(k, 0.0), float(v))
        for k, v in sample_ranges.items():
            merged_ranges[k] = v

        files_out.append(
            {
                "filename": path.name,
                "ok": True,
                "rows_sampled": rows,
                "columns": columns,
                "original_columns": original_columns,
                "rename_map": rename_map,
                "numeric_columns": numeric,
                "timestamp_candidates": ts_cands,
                "null_rates": null_rates,
                "sample_ranges": sample_ranges,
                "path": str(path),
            }
        )

    columns = _uniq(all_columns)
    ts_set = set(all_timestamps)
    numeric_cols = _uniq([c for c in all_numeric if c not in ts_set])
    timestamp_candidates = _uniq(all_timestamps)
    if not numeric_cols:
        quality_notes.append("No numeric columns found — tab may not train successfully.")

    return {
        "ok": True,
        "status": "ok",
        "data_folder": str(folder),
        "source_file": pinned,
        "title_hint": resolved.get("title_hint") or "",
        "file_count": len(unique_files),
        "files_inspected": files_out,
        # A.1 contract
        "columns": columns,
        "numeric_cols": numeric_cols,
        "timestamp_candidates": timestamp_candidates,
        "row_count": total_rows,
        "null_rates": merged_null,
        "sample_ranges": merged_ranges,
        # aliases
        "numeric_features": numeric_cols,
        "all_columns": columns,
        "rows_sampled_total": total_rows,
        "quality_notes": quality_notes,
        "data_quality_ok": len(numeric_cols) > 0 and total_rows > 0,
    }


def _pick_models(
    *,
    purpose_l: str,
    n_features: int,
    row_count: int,
) -> list[dict[str, Any]]:
    """Choose 2–3 anomaly models suited to the profile (not all catalog models)."""
    picks: list[dict[str, Any]] = [
        {
            "model_type": "Isolation Forest",
            "model_parameters": {
                "n_estimators": 200 if row_count >= 500 else 100,
                "contamination": 0.05,
            },
            "why": "Robust multivariate anomaly detector for health telemetry",
        },
        {
            "model_type": "Z-Score",
            "model_parameters": {"threshold": 3.0},
            "why": "Fast univariate baseline for voltage/temp/current spikes",
        },
    ]
    if n_features >= 2 and row_count >= 50:
        picks.append(
            {
                "model_type": "Local Outlier Factor",
                "model_parameters": {
                    "n_neighbors": 20 if row_count >= 200 else 10,
                    "contamination": 0.05,
                },
                "why": "Local density anomalies when several correlated features exist",
            }
        )
    elif "eclipse" in purpose_l or "supervised" in purpose_l:
        picks.append(
            {
                "model_type": "IQR (Interquartile Range)",
                "model_parameters": {},
                "why": "Simple robust range check for eclipse/seasonal shifts",
            }
        )

    models = []
    for m in picks[:3]:
        mt = m["model_type"] if m["model_type"] in DEFAULT_MODELS else "Isolation Forest"
        models.append(
            {
                "model_type": mt,
                "model_parameters": dict(m["model_parameters"]),
                "model_id": str(uuid.uuid4()),
                "why": m.get("why"),
            }
        )
    return models


def suggest_tab_config(
    *,
    title: str = "Health Monitoring",
    data_folder: str = "",
    purpose: str = "",
    inspection: Optional[dict[str, Any]] = None,
    sop_hints: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Suggest tab_name/features/models/schedule from inspect + SOP (Faza A.2)."""
    purpose_l = (purpose or title or "").lower()

    insp = inspection if isinstance(inspection, dict) else None
    if insp is None and data_folder:
        insp = inspect_data_folder(data_folder)
    if not insp or not insp.get("ok"):
        insp = insp or {"ok": False, "error": "no_inspection"}

    folder = str(insp.get("data_folder") or data_folder or "")
    source_file = str(insp.get("source_file") or "")
    # Ideal name from pinned CSV / path hint when caller title is weak
    if is_weak_title(title):
        hint = str(insp.get("title_hint") or "")
        if not hint and source_file:
            hint = Path(source_file).stem
        tab_name = _ideal_title_from_hint(hint, fallback="Health Monitoring")
    else:
        tab_name = (title or "Health Monitoring").strip() or "Health Monitoring"

    features = list(insp.get("numeric_cols") or insp.get("numeric_features") or [])
    row_count = int(insp.get("row_count") or insp.get("rows_sampled_total") or 0)
    ts_cands = list(insp.get("timestamp_candidates") or [])

    # Also treat path/purpose battery hints from folder/file names
    path_l = f"{folder} {source_file} {tab_name}".lower()
    if any(h in path_l for h in _BATTERY_HINTS):
        purpose_l = purpose_l + " battery health"

    # Prefer GUI-normalized value* columns (matches DataReader / runtime train)
    gui_feats = [f for f in features if str(f).lower() == "value" or str(f).lower().startswith("value")]
    if gui_feats:
        features = gui_feats + [f for f in features if f not in gui_feats]
    elif any(h in purpose_l for h in _BATTERY_HINTS) and features:
        preferred = [
            f for f in features if any(h in str(f).lower() for h in _BATTERY_HINTS)
        ]
        if preferred:
            features = preferred + [f for f in features if f not in preferred]
    ts_set = set(ts_cands)
    features = [f for f in features if f not in ts_set][:24]

    interval_ms = 300_000
    if "eclipse" in purpose_l:
        interval_ms = 60_000
    elif any(h in purpose_l for h in ("battery", "health", "voltage", "temp")):
        interval_ms = 180_000

    window_size = 500 if len(features) < 10 else 800
    if row_count and row_count < window_size:
        window_size = max(50, min(500, row_count))

    models = _pick_models(
        purpose_l=purpose_l, n_features=len(features), row_count=row_count
    )

    cfg = default_tab_config(title=tab_name)
    cfg.update(
        {
            "title": tab_name,
            "data_folder": folder,
            "source_file": source_file,
            "data_file_type": "CSV",
            "dataset_mode": "Full Latest File",
            "input_mode": "CSV Polling",
            "selected_features": features,
            "subsystem_name": "Power" if any(h in purpose_l for h in _BATTERY_HINTS) else "",
            "models": [
                {
                    "model_type": m["model_type"],
                    "model_parameters": dict(m["model_parameters"]),
                    "model_id": m["model_id"],
                }
                for m in models
            ],
            "model_type": models[0]["model_type"],
            "model_parameters": dict(models[0]["model_parameters"]),
            "schedule_type": "Continuous",
            "interval_ms": interval_ms,
            "max_training_rows": 100_000,
            "monitoring_window_rows": window_size,
        }
    )

    reasoning_parts = [
        f"Tab '{tab_name}' from folder `{folder or data_folder}`.",
        f"Selected {len(features)} numeric feature(s)"
        + (f"; timestamp col(s): {ts_cands[:3]}" if ts_cands else "")
        + ".",
        f"Models ({len(models)}): "
        + ", ".join(
            f"{m['model_type']}" + (f" — {m['why']}" if m.get("why") else "")
            for m in models
        )
        + ".",
        f"Schedule Continuous every {interval_ms // 1000}s; window_size={window_size}.",
    ]
    if insp.get("quality_notes"):
        reasoning_parts.append(
            "Data quality: " + "; ".join(str(x) for x in insp["quality_notes"][:5])
        )
    if sop_hints:
        reasoning_parts.append(
            "SOP/context: " + "; ".join(str(s)[:120] for s in sop_hints[:5])
        )

    reasoning = " ".join(reasoning_parts)

    return {
        "ok": True,
        "status": "ok",
        # A.2 contract
        "tab_name": tab_name,
        "features": features,
        "models": models,
        "schedule_type": "Continuous",
        "interval_ms": interval_ms,
        "window_size": window_size,
        "reasoning": reasoning,
        # GUI-ready config for propose_create_tab
        "config": cfg,
        "inspection_ok": bool(insp.get("ok")),
        "data_quality_ok": bool(insp.get("data_quality_ok")),
        "recommended_next": [
            "propose_create_tab with config from this suggestion",
            "After Approve: propose_train for each model_id",
            "propose_start_monitoring when trained",
        ],
    }


def suggestion_to_tab_config(suggestion: dict[str, Any]) -> dict[str, Any]:
    """Normalize suggest_tab_config (or partial) output into a create_tab config dict."""
    if not isinstance(suggestion, dict):
        return {}
    if isinstance(suggestion.get("config"), dict) and suggestion["config"].get("title"):
        return dict(suggestion["config"])

    tab_name = str(
        suggestion.get("tab_name") or suggestion.get("title") or "Health Monitoring"
    )
    cfg = default_tab_config(title=tab_name)
    features = list(
        suggestion.get("features") or suggestion.get("selected_features") or []
    )
    models_in = suggestion.get("models") or []
    models = []
    for m in models_in:
        if not isinstance(m, dict):
            continue
        models.append(
            {
                "model_type": m.get("model_type") or "Isolation Forest",
                "model_parameters": dict(m.get("model_parameters") or {}),
                "model_id": m.get("model_id") or str(uuid.uuid4()),
            }
        )
    if not models:
        models = [
            {
                "model_type": "Isolation Forest",
                "model_parameters": {},
                "model_id": str(uuid.uuid4()),
            }
        ]
    cfg.update(
        {
            "title": tab_name,
            "selected_features": features,
            "models": models,
            "model_type": models[0]["model_type"],
            "model_parameters": dict(models[0].get("model_parameters") or {}),
            "schedule_type": suggestion.get("schedule_type") or "Continuous",
            "interval_ms": int(
                suggestion.get("interval_ms") or cfg.get("interval_ms") or 300_000
            ),
            "monitoring_window_rows": int(
                suggestion.get("window_size")
                or suggestion.get("monitoring_window_rows")
                or cfg.get("monitoring_window_rows")
                or 500
            ),
        }
    )
    if suggestion.get("data_folder"):
        cfg["data_folder"] = str(suggestion["data_folder"])
    return cfg
