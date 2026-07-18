from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("STDMS.TabConfig")

DEFAULT_MODELS = [
    "Isolation Forest",
    "Local Outlier Factor",
    "Z-Score",
    "IQR (Interquartile Range)",
    "Random Forest",
    "Prophet",
    "LSTM",
    "Autoencoder",
]


def default_tab_config(title: str = "New Monitoring Tab") -> dict[str, Any]:
    from app.models.fsm import DEFAULT_MISSION_MODES, ensure_fsm_fields

    cfg = {
        "title": title,
        "data_folder": "",
        "data_file_type": "CSV",
        "dataset_mode": "Full Latest File",
        "input_mode": "CSV Polling",
        "selected_features": [],
        "subsystem_name": "",
        "models": [{"model_type": "Isolation Forest", "model_parameters": {}}],
        "model_type": "Isolation Forest",
        "schedule_type": "Continuous",
        "interval_ms": 300_000,
        "schedule_utc_hour": 0,
        "schedule_utc_minute": 0,
        "max_training_rows": 100_000,
        "monitoring_window_rows": 500,
        "email_alerts_enabled": False,
        "alert_recipients": [],
        "mqtt_broker": "",
        "mqtt_port": 1883,
        "mqtt_topic": "",
        "opcua_endpoint": "",
        "opcua_nodes": "",
        "mission_modes": [dict(m) for m in DEFAULT_MISSION_MODES],
        "current_mission_mode": "nominal",
    }
    return ensure_fsm_fields(cfg)


class TabConfigurationManager:
    """Persist custom monitoring tab configurations (legacy JSON compatible)."""

    def __init__(self, config_file: Path):
        self.config_file = config_file
        self.configs: dict[str, dict[str, Any]] = {}
        self.load_configs()

    def load_configs(self) -> None:
        try:
            if self.config_file.exists():
                self.configs = json.loads(self.config_file.read_text(encoding="utf-8"))
                logger.info("Loaded %d tab configurations", len(self.configs))
            else:
                self.configs = {}
        except Exception as exc:
            logger.error("Error loading tab configurations: %s", exc)
            self.configs = {}

    def save_configs(self) -> bool:
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self.config_file.write_text(
                json.dumps(self.configs, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            logger.error("Error saving tab configurations: %s", exc)
            return False

    def create_tab(self, config: dict[str, Any]) -> str:
        tab_id = str(uuid.uuid4())
        self.configs[tab_id] = config
        self.save_configs()
        return tab_id

    def update_tab(self, tab_id: str, config: dict[str, Any]) -> bool:
        if tab_id not in self.configs:
            return False
        self.configs[tab_id] = config
        return self.save_configs()

    def remove_tab(self, tab_id: str) -> bool:
        if tab_id not in self.configs:
            return False
        del self.configs[tab_id]
        return self.save_configs()

    def get_config(self, tab_id: str) -> dict[str, Any] | None:
        return self.configs.get(tab_id)

    def list_tabs(self) -> list[tuple[str, dict[str, Any]]]:
        return [(tab_id, cfg) for tab_id, cfg in self.configs.items()]
