from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class AppPaths:
    encryption_key_file: str = "keys/encryption_key.bin"
    user_db_file: str = "data/users.json"
    config_file: str = "config/app_config.json"
    auth_config_file: str = "config/auth_config.json"
    alert_routing_config_file: str = "config/alert_routing_config.json"
    models_dir: str = "models/"
    data_dir: str = "data/"
    reports_dir: str = "reports/"
    telemetry_db_file: str = "data/telemetry.db"
    rule_config_file: str = "config/rule_config.json"
    activity_log_file: str = "logs/activity.log"
    audit_log_file: str = "logs/audit_activity.log"
    auto_reports_dir: str = "reports/auto"


@dataclass
class ObservabilitySettings:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 9108


@dataclass
class SecuritySettings:
    lock_timeout_ms: int = 900000
    session_timeout_ms: int = 28800000


@dataclass
class AlertingSettings:
    escalation_check_ms: int = 60000


@dataclass
class AppSettings:
    paths: AppPaths = field(default_factory=AppPaths)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    alerting: AlertingSettings = field(default_factory=AlertingSettings)


class SettingsManager:
    """Typed settings loader with env + file overrides."""

    def __init__(self, workspace_root: str | None = None):
        self.workspace_root = Path(workspace_root or os.getcwd())

    def load(self) -> AppSettings:
        settings = AppSettings()
        self._apply_file_overrides(settings)
        self._apply_env_overrides(settings)
        self._ensure_directories(settings)
        return settings

    def _settings_file(self) -> Path:
        return self.workspace_root / "config" / "settings.json"

    def _apply_file_overrides(self, settings: AppSettings) -> None:
        file_path = self._settings_file()
        if not file_path.exists():
            return
        try:
            raw: Dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return

        paths = raw.get("paths", {})
        for key, value in paths.items():
            if hasattr(settings.paths, key):
                setattr(settings.paths, key, str(value))

        obs = raw.get("observability", {})
        if "enabled" in obs:
            settings.observability.enabled = bool(obs["enabled"])
        if "host" in obs:
            settings.observability.host = str(obs["host"])
        if "port" in obs:
            settings.observability.port = int(obs["port"])

        sec = raw.get("security", {})
        if "lock_timeout_ms" in sec:
            settings.security.lock_timeout_ms = int(sec["lock_timeout_ms"])
        if "session_timeout_ms" in sec:
            settings.security.session_timeout_ms = int(sec["session_timeout_ms"])

        alerting = raw.get("alerting", {})
        if "escalation_check_ms" in alerting:
            settings.alerting.escalation_check_ms = int(alerting["escalation_check_ms"])

    def _apply_env_overrides(self, settings: AppSettings) -> None:
        enabled = os.getenv("MONITOR_OBS_ENABLED")
        if enabled is not None:
            settings.observability.enabled = enabled.strip().lower() not in ("0", "false", "no")
        settings.observability.host = os.getenv("MONITOR_OBS_HOST", settings.observability.host).strip() or settings.observability.host
        settings.observability.port = int(os.getenv("MONITOR_OBS_PORT", str(settings.observability.port)))

    def _ensure_directories(self, settings: AppSettings) -> None:
        def _resolve(path_value: str) -> Path:
            candidate = Path(path_value)
            if candidate.is_absolute():
                return candidate
            return self.workspace_root / candidate

        candidates = [
            _resolve(settings.paths.encryption_key_file).parent,
            _resolve(settings.paths.user_db_file).parent,
            _resolve(settings.paths.config_file).parent,
            _resolve(settings.paths.auth_config_file).parent,
            _resolve(settings.paths.alert_routing_config_file).parent,
            _resolve(settings.paths.models_dir),
            _resolve(settings.paths.data_dir),
            _resolve(settings.paths.reports_dir),
            _resolve(settings.paths.activity_log_file).parent,
            _resolve(settings.paths.audit_log_file).parent,
            _resolve(settings.paths.auto_reports_dir),
        ]
        for directory in candidates:
            if str(directory):
                directory.mkdir(parents=True, exist_ok=True)

