"""Phase 1 FSM — mission mode threshold scaling."""

from __future__ import annotations

import pandas as pd

from app.models.fsm import (
    MissionModeStore,
    apply_threshold_scale,
    ensure_fsm_fields,
    resolve_current_mode,
    scale_bound,
)
from app.monitoring.obs_limits import evaluate_obs_condition, evaluate_obs_limits
from app.monitoring.tab_config import default_tab_config


def test_default_tab_config_includes_mission_modes():
    cfg = default_tab_config("Battery")
    assert cfg["current_mission_mode"] == "nominal"
    names = {m["name"] for m in cfg["mission_modes"]}
    assert {"nominal", "eclipse", "maneuver", "safe_mode"} <= names


def test_resolve_current_mode_eclipse_scale():
    cfg = ensure_fsm_fields(
        {
            "mission_modes": [
                {"name": "nominal", "threshold_scale": 1.0},
                {"name": "eclipse", "threshold_scale": 1.5},
            ],
            "current_mission_mode": "eclipse",
        }
    )
    mode = resolve_current_mode(cfg)
    assert mode.name == "eclipse"
    assert mode.threshold_scale == 1.5


def test_scale_bound_upper_and_lower():
    assert scale_bound(80.0, ">", 1.5) == 120.0
    assert scale_bound(80.0, ">", 0.5) == 40.0
    assert abs(scale_bound(10.0, "<", 1.5) - (10.0 / 1.5)) < 1e-9
    assert scale_bound(10.0, "<", 0.5) == 20.0


def test_obs_eclipse_suppresses_false_positive_vs_nominal():
    """Same thermal reading: anomaly in nominal, mode-normal in eclipse."""
    df = pd.DataFrame({"temp": [70.0, 72.0, 100.0]})
    rules = [
        {
            "name": "High Temperature",
            "parameter": "temp",
            "condition": "value > 80",
            "severity": 3,
        }
    ]

    nominal = evaluate_obs_limits(df, rules=rules, threshold_scale=1.0, mission_mode="nominal")
    assert nominal["ok"] is False
    assert len(nominal["violations"]) == 1
    assert nominal["mode_normal"] == []

    eclipse = evaluate_obs_limits(df, rules=rules, threshold_scale=1.5, mission_mode="eclipse")
    # 100 > 80 but not > 120 → mode-normal, not anomaly
    assert eclipse["ok"] is True
    assert eclipse["violations"] == []
    assert len(eclipse["mode_normal"]) == 1
    assert eclipse["mode_normal"][0]["reason"] == "mode-normal"


def test_obs_safe_mode_more_sensitive():
    df = pd.DataFrame({"temp": [50.0]})
    rules = [{"name": "High", "parameter": "temp", "condition": "value > 80", "severity": 2}]
    # 50 is under both nominal (80) and safe (40)? Wait 50 > 40 with scale 0.5
    # value > 80 * 0.5 = 40 → 50 > 40 → violation in safe mode
    safe = evaluate_obs_limits(df, rules=rules, threshold_scale=0.5, mission_mode="safe_mode")
    assert safe["ok"] is False
    nominal = evaluate_obs_limits(df, rules=rules, threshold_scale=1.0, mission_mode="nominal")
    assert nominal["ok"] is True


def test_evaluate_obs_condition_respects_scale():
    assert evaluate_obs_condition(100.0, "value > 80", threshold_scale=1.0)
    assert not evaluate_obs_condition(100.0, "value > 80", threshold_scale=1.5)
    assert evaluate_obs_condition(100.0, "value > 80", threshold_scale=1.5) is False


def test_apply_threshold_scale_fusion():
    assert abs(apply_threshold_scale(0.6, 1.5, lo=0.35, hi=0.995) - 0.9) < 1e-9
    assert apply_threshold_scale(0.6, 0.5, lo=0.35, hi=0.995) == 0.35  # clamped to min


def test_mission_mode_store_history(tmp_path):
    store = MissionModeStore(db_path=str(tmp_path / "fsm.sqlite"))
    store.sync_tab_modes(
        "tab-1",
        [
            {"name": "nominal", "threshold_scale": 1.0},
            {"name": "eclipse", "threshold_scale": 1.5},
        ],
    )
    t1 = store.set_mode("tab-1", "nominal", trigger_source="manual")
    assert t1.to_mode == "nominal"
    t2 = store.set_mode("tab-1", "eclipse", trigger_source="log_event")
    assert t2.from_mode == "nominal"
    assert t2.to_mode == "eclipse"
    current = store.get_current_mode("tab-1")
    assert current is not None
    assert current.name == "eclipse"
    assert current.threshold_scale == 1.5
    hist = store.history("tab-1")
    assert len(hist) >= 2
    assert hist[0]["mode_name"] == "eclipse"
    assert hist[0]["exited_at"] is None
