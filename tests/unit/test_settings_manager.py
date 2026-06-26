import os
from pathlib import Path

from app.config.settings import SettingsManager


def test_settings_manager_env_override(tmp_path: Path, monkeypatch):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MONITOR_OBS_ENABLED", "0")
    monkeypatch.setenv("MONITOR_OBS_HOST", "0.0.0.0")
    monkeypatch.setenv("MONITOR_OBS_PORT", "9200")

    settings = SettingsManager(workspace_root=str(tmp_path)).load()
    assert settings.observability.enabled is False
    assert settings.observability.host == "0.0.0.0"
    assert settings.observability.port == 9200


def test_settings_manager_creates_directories(tmp_path: Path):
    settings = SettingsManager(workspace_root=str(tmp_path)).load()
    assert (tmp_path / Path(settings.paths.data_dir)).exists()
    assert (tmp_path / Path(settings.paths.models_dir)).exists()
    assert (tmp_path / "logs").exists()

