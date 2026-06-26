import json
import pickle
from pathlib import Path

import pytest

from app.tabs.custom_tab import model_ops


class _FakeStandardModel:
  def __init__(self, model_type):
    self.model_type = model_type
    self.trained = False

  def train(self, data, **kwargs):
    self.trained = True
    return True, "trained"

  @classmethod
  def load(cls, filepath):
    with open(filepath, "rb") as handle:
      payload = pickle.load(handle)
    return cls(payload["model_type"]), "ok"


class _FakeEnhancedModel:
  def __init__(self, model_type):
    self.model_type = model_type

  @classmethod
  def load(cls, filepath):
    with open(filepath, "rb") as handle:
      payload = pickle.load(handle)
    if payload.get("enhanced"):
      return cls(payload["model_type"]), "enhanced ok"
    raise ValueError("not enhanced")


def test_create_model_instance_uses_enhanced_class():
  model_obj, internal = model_ops.create_model_instance(
    "Enhanced Isolation Forest",
    {"n_estimators": 50},
    to_internal_model_type=lambda name: "enhanced_isolation_forest",
    anomaly_detection_model_cls=_FakeStandardModel,
    enhanced_anomaly_detection_model_cls=_FakeEnhancedModel,
  )
  assert isinstance(model_obj, _FakeEnhancedModel)
  assert internal == "enhanced_isolation_forest"


def test_autosave_and_bootstrap_round_trip(tmp_path: Path):
  model_id = "model-1"
  model_obj = _FakeStandardModel("isolation_forest")
  model_ops.autosave_model(tmp_path, model_id, model_obj, "Isolation Forest")

  models_list, loaded_models, count = model_ops.bootstrap_from_manifest(
    tmp_path,
    [],
    anomaly_detection_model_cls=_FakeStandardModel,
    enhanced_anomaly_detection_model_cls=_FakeEnhancedModel,
    safe_pickle_load=lambda path: _FakeStandardModel("fallback"),
  )
  assert count == 1
  assert model_id in loaded_models
  assert models_list[0]["model_type"] == "Isolation Forest"


def test_load_model_from_file_prefers_enhanced_loader(tmp_path: Path):
  enhanced_path = tmp_path / "enhanced.pkl"
  with open(enhanced_path, "wb") as handle:
    pickle.dump({"model_type": "enhanced_isolation_forest", "enhanced": True}, handle)

  loaded = model_ops.load_model_from_file(
    enhanced_path,
    anomaly_detection_model_cls=_FakeStandardModel,
    enhanced_anomaly_detection_model_cls=_FakeEnhancedModel,
    safe_pickle_load=lambda path: pytest.fail("safe_pickle_load should not be used"),
  )
  assert isinstance(loaded, _FakeEnhancedModel)


def test_resolve_training_target_creates_default_config():
  config = {"model_type": "Isolation Forest", "model_parameters": {"n_estimators": 25}}
  model_id, model_config, model_type, model_params = model_ops.resolve_training_target(config)
  assert model_id
  assert model_config["model_id"] == model_id
  assert model_type == "Isolation Forest"
  assert model_params["n_estimators"] == 25
  assert len(config["models"]) == 1
