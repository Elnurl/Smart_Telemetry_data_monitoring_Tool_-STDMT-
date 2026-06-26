"""Model train/save/load helpers for custom monitoring tabs (PyQt-free core)."""

from __future__ import annotations

import datetime
import json
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("STDMS.TabModelOps")

ENHANCED_INTERNAL_TYPES = frozenset({
    "enhanced_isolation_forest",
    "ensemble_voting",
    "ensemble_stacking",
    "adaptive_threshold",
})

INTERNAL_TO_DISPLAY_TYPE = {
    "isolation_forest": "Isolation Forest",
    "enhanced_isolation_forest": "Enhanced Isolation Forest",
    "local_outlier_factor": "Local Outlier Factor",
    "lof": "Local Outlier Factor",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "gru": "GRU",
    "autoencoder": "Autoencoder",
    "iqr_(interquartile_range)": "IQR (Interquartile Range)",
    "z-score": "Z-Score",
    "prophet": "Prophet",
}


def internal_to_display_type(
    internal_type: str,
    display_to_internal: Optional[Dict[str, str]] = None,
) -> str:
    if display_to_internal:
        for display_name, mapped_internal in display_to_internal.items():
            if mapped_internal == internal_type:
                return display_name
    return INTERNAL_TO_DISPLAY_TYPE.get(internal_type, internal_type)


def normalize_model_type_key(
    model_type: str,
    to_internal: Optional[Callable[[str], str]] = None,
) -> str:
    model_type = str(model_type or "").strip()
    if not model_type:
        return ""
    if to_internal:
        return to_internal(model_type)
    return model_type.lower().replace(" ", "_")


def model_types_equivalent(
    left: str,
    right: str,
    to_internal: Optional[Callable[[str], str]] = None,
) -> bool:
    if str(left).strip() == str(right).strip():
        return True
    return normalize_model_type_key(left, to_internal) == normalize_model_type_key(right, to_internal)


def resolve_training_target(
    config: dict,
    model_id: Optional[str] = None,
) -> Tuple[str, dict, str, dict]:
    """Resolve model_id and config entry for training."""
    models_list = list(config.get("models", []))
    if model_id is None:
        if models_list:
            model_id = models_list[0].get("model_id")
        else:
            model_id = str(uuid.uuid4())
            models_list = [{
                "model_type": config.get("model_type", "Isolation Forest"),
                "model_parameters": config.get("model_parameters", {}),
                "model_id": model_id,
            }]
            config["models"] = models_list

    model_config = next((m for m in models_list if m.get("model_id") == model_id), None)
    if not model_config:
        raise ValueError("Model configuration not found.")

    model_type = model_config.get("model_type", "Isolation Forest")
    model_params = model_config.get("model_parameters", {}) or {}
    return model_id, model_config, model_type, model_params


def create_model_instance(
    model_type: str,
    model_params: Optional[dict],
    *,
    to_internal_model_type: Callable[[str], str],
    anomaly_detection_model_cls: Any,
    enhanced_anomaly_detection_model_cls: Any,
) -> Tuple[Any, str]:
    internal_model_type = to_internal_model_type(model_type)
    if internal_model_type in ENHANCED_INTERNAL_TYPES:
        model_obj = enhanced_anomaly_detection_model_cls(internal_model_type)
    else:
        model_obj = anomaly_detection_model_cls(internal_model_type)

    for param_name, param_value in (model_params or {}).items():
        if hasattr(model_obj, param_name):
            setattr(model_obj, param_name, param_value)
    return model_obj, internal_model_type


def autosave_model(models_dir: str | Path, model_id: str, model_obj: Any, model_type: str) -> None:
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    filepath = models_path / f"{model_id}.pkl"

    if hasattr(model_obj, "save"):
        model_obj.save(str(filepath))
    else:
        with open(filepath, "wb") as handle:
            pickle.dump(model_obj, handle)

    manifest_path = models_path / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read model manifest: %s", exc)

    manifest[model_id] = {
        "model_type": model_type,
        "filepath": str(filepath),
        "saved_at": datetime.datetime.now().isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_model_to_file(model_obj: Any, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model_obj, "save"):
        model_obj.save(str(path))
        return
    with open(path, "wb") as handle:
        pickle.dump(model_obj, handle)


def load_model_from_file(
    filepath: str | Path,
    *,
    anomaly_detection_model_cls: Any,
    enhanced_anomaly_detection_model_cls: Any,
    safe_pickle_load: Callable[[str], Any],
) -> Any:
    path = str(filepath)

    try:
        loaded_model, _message = enhanced_anomaly_detection_model_cls.load(path)
        if loaded_model is not None:
            return loaded_model
    except Exception:
        pass

    try:
        loaded_model, load_msg = anomaly_detection_model_cls.load(path)
        if loaded_model is not None:
            return loaded_model
        if load_msg:
            logger.debug("AnomalyDetectionModel.load message for %s: %s", path, load_msg)
    except Exception:
        pass

    return safe_pickle_load(path)


def bootstrap_from_manifest(
    models_dir: str | Path,
    models_list: List[dict],
    *,
    anomaly_detection_model_cls: Any,
    enhanced_anomaly_detection_model_cls: Any,
    safe_pickle_load: Callable[[str], Any],
) -> Tuple[List[dict], Dict[str, Any], int]:
    """Restore tab models from manifest; returns (models_list, loaded_models, count)."""
    manifest_path = Path(models_dir) / "manifest.json"
    if not manifest_path.exists():
        return models_list, {}, 0

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read tab model manifest: %s", exc)
        return models_list, {}, 0

    updated_models_list = list(models_list)
    loaded_models: Dict[str, Any] = {}
    loaded_count = 0

    for model_id, entry in manifest.items():
        filepath = entry.get("filepath")
        if not filepath or not os.path.exists(filepath):
            continue
        try:
            model_obj = load_model_from_file(
                filepath,
                anomaly_detection_model_cls=anomaly_detection_model_cls,
                enhanced_anomaly_detection_model_cls=enhanced_anomaly_detection_model_cls,
                safe_pickle_load=safe_pickle_load,
            )
        except Exception as load_err:
            logger.warning("Could not restore model %s: %s", model_id, load_err)
            continue

        loaded_models[model_id] = model_obj
        loaded_count += 1
        if not any(m.get("model_id") == model_id for m in updated_models_list):
            updated_models_list.append({
                "model_type": entry.get("model_type", "Unknown"),
                "model_parameters": {},
                "model_id": model_id,
            })

    return updated_models_list, loaded_models, loaded_count


def build_loaded_model_entry(loaded_model: Any) -> Tuple[str, dict]:
    """Create a new model list entry from a loaded model object."""
    if hasattr(loaded_model, "model_type"):
        display_type = internal_to_display_type(str(loaded_model.model_type))
    else:
        display_type = "Unknown"

    new_model_id = str(uuid.uuid4())
    return new_model_id, {
        "model_type": display_type,
        "model_parameters": {},
        "model_id": new_model_id,
    }
