"""Operational bounds (OBS) rule evaluation — safe condition parsing, no eval()."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("STDMS.OBS")

DEFAULT_RULE_CONFIG_FILE = Path("config/rule_config.json")

_OBS_CONDITION_RE = re.compile(
    r"^\s*value\s*(==|!=|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)

_RULE_CONFIG_CACHE: dict[str, dict[str, Any]] = {}


def evaluate_obs_condition(value: float, condition: str) -> bool:
    """Evaluate a safe OBS rule without arbitrary code execution."""
    match = _OBS_CONDITION_RE.match(str(condition).strip())
    if not match:
        logger.warning("Rejected unsafe OBS condition: %r", condition)
        return False
    operator, threshold_raw = match.group(1), float(match.group(2))
    if operator == ">":
        return value > threshold_raw
    if operator == "<":
        return value < threshold_raw
    if operator == ">=":
        return value >= threshold_raw
    if operator == "<=":
        return value <= threshold_raw
    if operator == "==":
        return value == threshold_raw
    if operator == "!=":
        return value != threshold_raw
    return False


def default_rule_config() -> dict[str, Any]:
    return {
        "rules": [
            {
                "name": "High CPU Temperature",
                "parameter": "cpu_temp",
                "condition": "value > 80",
                "severity": 3,
            },
            {
                "name": "Low Disk Space",
                "parameter": "disk_space",
                "condition": "value < 10",
                "severity": 2,
            },
        ],
        "global_settings": {"check_interval_seconds": 60, "min_alert_severity": 2},
    }


def load_rule_config(
    rule_config_file: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load OBS rules from JSON with mtime-based caching."""
    path = Path(rule_config_file or DEFAULT_RULE_CONFIG_FILE)
    default_config = default_rule_config()
    cache_key = str(path.resolve()) if path.exists() else str(path)

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_config, indent=4), encoding="utf-8")
        _RULE_CONFIG_CACHE[cache_key] = {
            "mtime": path.stat().st_mtime,
            "config": default_config,
        }
        return default_config

    try:
        mtime = path.stat().st_mtime
        cached = _RULE_CONFIG_CACHE.get(cache_key)
        if not force_reload and cached and cached.get("mtime") == mtime:
            return cached["config"]

        config = json.loads(path.read_text(encoding="utf-8"))
        _RULE_CONFIG_CACHE[cache_key] = {"mtime": mtime, "config": config}
        return config
    except Exception as exc:
        logger.error("Failed to load rule config from %s: %s", path, exc)
        return default_config


def evaluate_obs_limits(
    data: pd.DataFrame | None,
    rules: list[dict[str, Any]] | None = None,
    *,
    rule_config_file: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate operational bounds rules against the latest telemetry row."""
    if data is None or len(data) == 0:
        return {"ok": True, "violations": []}

    if rules is None:
        rules = load_rule_config(rule_config_file).get("rules", [])

    violations: list[dict[str, Any]] = []
    latest = data.iloc[-1]
    for rule in rules or []:
        param = rule.get("parameter")
        condition = str(rule.get("condition", "")).strip()
        if not param or param not in data.columns or not condition:
            continue
        try:
            value = float(latest[param])
            if evaluate_obs_condition(value, condition):
                violations.append(
                    {
                        "name": rule.get("name", param),
                        "parameter": param,
                        "value": value,
                        "condition": condition,
                        "severity": rule.get("severity", 1),
                    }
                )
        except Exception:
            continue
    return {"ok": len(violations) == 0, "violations": violations}
