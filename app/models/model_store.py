"""Per-tab model persistence with a JSON manifest (local offline storage)."""

from __future__ import annotations

import datetime
import json
import logging
import pickle
import uuid
from pathlib import Path

from app.models.anomaly_engine import AnomalyModel

logger = logging.getLogger("STDMS.ModelStore")


class ModelStore:
    def __init__(self, base_dir: Path, tab_id: str):
        self.models_dir = Path(base_dir) / "custom_tabs" / tab_id / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.models_dir / "manifest.json"

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read model manifest: %s", exc)
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def save(self, model: AnomalyModel, model_id: str | None = None) -> str:
        model_id = model_id or str(uuid.uuid4())
        filepath = self.models_dir / f"{model_id}.pkl"
        with open(filepath, "wb") as fh:
            pickle.dump(model.state(), fh)
        manifest = self._load_manifest()
        manifest[model_id] = {
            "model_type": model.model_type,
            "backend": model.backend,
            "filepath": str(filepath),
            "saved_at": datetime.datetime.now().isoformat(),
        }
        self._save_manifest(manifest)
        return model_id

    def load_all(self) -> dict[str, AnomalyModel]:
        manifest = self._load_manifest()
        models: dict[str, AnomalyModel] = {}
        for model_id, info in manifest.items():
            filepath = Path(info.get("filepath", ""))
            if not filepath.exists():
                continue
            try:
                with open(filepath, "rb") as fh:
                    state = pickle.load(fh)
                models[model_id] = AnomalyModel.from_state(state)
            except Exception as exc:
                logger.warning("Failed to load model %s: %s", model_id, exc)
        return models

    def list_models(self) -> dict:
        return self._load_manifest()

    def delete(self, model_id: str) -> bool:
        manifest = self._load_manifest()
        info = manifest.pop(model_id, None)
        if not info:
            return False
        try:
            Path(info.get("filepath", "")).unlink(missing_ok=True)
        except Exception:
            pass
        self._save_manifest(manifest)
        return True
