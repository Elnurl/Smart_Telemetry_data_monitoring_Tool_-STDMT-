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


def parse_obs_condition(condition: str) -> tuple[str, float] | None:
    """Return (operator, threshold) for a safe OBS condition, else None."""
    match = _OBS_CONDITION_RE.match(str(condition).strip())
    if not match:
        return None
    return match.group(1), float(match.group(2))


def evaluate_obs_condition(
    value: float,
    condition: str,
    *,
    threshold_scale: float = 1.0,
) -> bool:
    """Evaluate a safe OBS rule without arbitrary code execution.

    ``threshold_scale`` widens (scale>1) or tightens (scale<1) inequality bounds
    so mission modes like eclipse can suppress expected thermal swings.
    """
    parsed = parse_obs_condition(condition)
    if not parsed:
        logger.warning("Rejected unsafe OBS condition: %r", condition)
        return False
    operator, threshold_raw = parsed
    if threshold_scale and threshold_scale != 1.0:
        from app.models.fsm import scale_bound

        threshold_raw = scale_bound(threshold_raw, operator, float(threshold_scale))
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
    threshold_scale: float = 1.0,
    mission_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate operational bounds rules against the latest telemetry row.

    When ``threshold_scale != 1``, values that breach the base limit but stay
    inside the scaled limit are reported under ``mode_normal`` (not anomalies).
    """
    if data is None or len(data) == 0:
        return {
            "ok": True,
            "violations": [],
            "mode_normal": [],
            "threshold_scale": float(threshold_scale or 1.0),
            "mission_mode": mission_mode,
        }

    if rules is None:
        rules = load_rule_config(rule_config_file).get("rules", [])

    try:
        scale = float(threshold_scale) if threshold_scale else 1.0
    except (TypeError, ValueError):
        scale = 1.0
    if scale <= 0:
        scale = 1.0

    violations: list[dict[str, Any]] = []
    mode_normal: list[dict[str, Any]] = []
    latest = data.iloc[-1]
    for rule in rules or []:
        param = rule.get("parameter")
        condition = str(rule.get("condition", "")).strip()
        if not param or param not in data.columns or not condition:
            continue
        try:
            value = float(latest[param])
            base_hit = evaluate_obs_condition(value, condition, threshold_scale=1.0)
            scaled_hit = evaluate_obs_condition(value, condition, threshold_scale=scale)
            entry = {
                "name": rule.get("name", param),
                "parameter": param,
                "value": value,
                "condition": condition,
                "severity": rule.get("severity", 1),
                "threshold_scale": scale,
                "mission_mode": mission_mode,
            }
            if scaled_hit:
                violations.append(entry)
            elif base_hit and scale != 1.0:
                entry = dict(entry)
                entry["reason"] = "mode-normal"
                mode_normal.append(entry)
        except Exception:
            continue
    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "mode_normal": mode_normal,
        "threshold_scale": scale,
        "mission_mode": mission_mode,
    }
