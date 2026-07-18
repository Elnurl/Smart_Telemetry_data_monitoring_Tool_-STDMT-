"""Faza B–F — model lab, insights, proactive dedupe, watchlist."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.audit import AgentAuditLog
from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.insights import detect_patterns, explain_anomaly, forecast_risk
from app.agent.model_lab import (
    compare_tab_models,
    propose_model_plan,
    suggest_optimize_from_comparison,
)
from app.agent.proactive import (
    clear_dedupe_cache,
    execute_proactive_actions,
    proactive_actions_from_tabs,
    should_emit,
)
from app.agent.runner import AgentRunner
from app.agent.watchlist import (
    add_to_watchlist,
    load_watchlist,
    pending_actions_summary,
    remove_from_watchlist,
)
from app.models.registry import ModelRegistry


def test_compare_tab_models_ranks_best():
    listing = {
        "ok": True,
        "tab_id": "t1",
        "title": "CH1",
        "models": [
            {
                "model_id": "a",
                "model_type": "Isolation Forest",
                "trained": True,
                "metrics": {"precision": 0.5, "recall": 0.4, "f1": 0.45},
            },
            {
                "model_id": "b",
                "model_type": "Z-Score",
                "trained": True,
                "metrics": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            },
            {"model_id": "c", "model_type": "LOF", "trained": False, "metrics": {}},
        ],
    }
    out = compare_tab_models(listing)
    assert out["ok"] is True
    assert out["best_model_id"] == "b"
    hints = suggest_optimize_from_comparison(out)
    assert hints["ok"] is True
    assert hints["best_model_id"] == "b"
    assert any(h["action"] == "propose_train" for h in hints["optimize_hints"])


def test_propose_model_plan_has_grid():
    plan = propose_model_plan(purpose="battery health")
    assert plan["ok"] is True
    assert len(plan["candidates"]) >= 3
    types = {c["model_type"] for c in plan["candidates"]}
    assert "Isolation Forest" in types


def test_explain_anomaly_and_patterns_forecast():
    analysis = {
        "status": "ok",
        "tab_id": "t1",
        "title": "Eclipse",
        "snapshot": {"health_state": "Warning", "monitoring_active": True},
        "anomaly": {"health_state": "Warning", "fusion_score": 0.62, "obs_ok": True},
        "drift": {"is_drift": True, "drift_score": 0.7, "drifted_features": ["temp"]},
        "recent_events": [{"health_state": "Warning", "anomalies": {"IF": 3}}],
    }
    expl = explain_anomaly(analysis, models_listing={"ok": True, "models": []})
    assert expl["ok"] is True
    assert expl["root_causes"]
    assert "propose_retrain" in expl["recommendation"] or "propose_train" in expl[
        "recommendation"
    ]

    pats = detect_patterns(
        score_history=[0.1, 0.12, 0.15, 0.2, 0.35, 0.5],
        events=[{"health_state": "Warning"}] * 4,
    )
    assert pats["trend"] == "rising"
    assert any(p["type"] == "recurring_warnings" for p in pats["patterns"])
    assert "spike_rate" in pats
    assert pats.get("method_note")

    # Seasonal / eclipse-like mid dip on a longer series
    seasonal_hist = [0.2] * 8 + [0.05] * 8 + [0.25] * 8
    seasonal = detect_patterns(score_history=seasonal_hist)
    assert seasonal["seasonal_hints"] or any(
        "eclipse" in p.get("type", "") or "seasonal" in p.get("type", "")
        for p in seasonal["patterns"]
    )

    risk = forecast_risk(
        forecast_value=0.55,
        patterns=pats,
        anomaly=analysis["anomaly"],
        drift=analysis["drift"],
    )
    assert risk["risk_score"] >= 0.4
    assert risk["risk_level"] in ("MEDIUM", "HIGH")
    assert risk.get("observation")
    assert risk.get("recommendation")


def test_proactive_dedupe(tmp_path: Path):
    clear_dedupe_cache()
    registry = ModelRegistry(db_path=str(tmp_path / "models.sqlite"))

    class Host:
        def list_pending_drafts(self):
            return registry.get_pending_draft_alerts()

        def propose_retrain(self, tab_id, *, reasoning="", drift_score=0.0, drifted_features=None):
            signal_id = registry.mark_retrain_needed(
                drift_score=float(drift_score or 0.0),
                reason=reasoning or "test",
                drifted_features=list(drifted_features or []),
                tab_id=tab_id,
            )
            return {"ok": True, "signal_id": signal_id, "tab_id": tab_id}

        def propose_alert(
            self,
            tab_id,
            *,
            proposed_message="",
            agent_reasoning="",
            severity="WARNING",
            kind="alert",
            proposed_payload=None,
        ):
            draft_id = registry.create_draft_alert(
                tab_id=tab_id,
                kind=kind,
                agent_reasoning=agent_reasoning,
                proposed_message=proposed_message,
                proposed_payload=proposed_payload or {},
                severity=severity,
            )
            return {"ok": True, "draft_id": draft_id, "kind": kind}

    tabs = [
        {
            "tab_id": "t1",
            "snapshot": {
                "title": "Batt",
                "health_state": "Warning",
                "drift": True,
                "obs_ok": True,
                "fusion_score": 0.6,
                "monitoring_active": True,
                "drift_score": 0.8,
            },
        }
    ]
    host = Host()
    actions = proactive_actions_from_tabs(tabs, dedupe_seconds=900, host=host)
    assert any(a["tool"] == "propose_alert" for a in actions)
    assert any(a["tool"] == "propose_retrain" for a in actions)

    results = execute_proactive_actions(host, actions)
    assert any(r["ok"] for r in results)

    actions2 = proactive_actions_from_tabs(tabs, dedupe_seconds=900, host=host)
    assert actions2 == []
    assert should_emit("t1", "alert", dedupe_seconds=900) is False
    clear_dedupe_cache()


def test_stop_monitoring_pending_dedupe_and_severity(tmp_path: Path):
    clear_dedupe_cache()
    registry = ModelRegistry(db_path=str(tmp_path / "models.sqlite"))

    class Tab:
        def __init__(self):
            self.tab_id = "t-crit"
            self.config = {"title": "Monitoring"}
            self.last_snapshot = {
                "title": "Monitoring",
                "health_state": "Critical",
                "monitoring_active": True,
            }

        def get_snapshot(self):
            return dict(self.last_snapshot)

    class Window:
        def __init__(self):
            self.custom_tabs = {"t-crit": Tab()}
            self.model_registry = registry

    from app.agent.bridge import MainWindowToolHost, set_tool_host
    from app.agent.tools import propose_stop_monitoring

    host = MainWindowToolHost(Window())
    set_tool_host(host)
    try:
        first = propose_stop_monitoring(
            "Stop monitoring on Monitoring",
            tab_id="t-crit",
            agent_reasoning="The tab 'Monitoring' is in a critical health state",
            host=host,
        )
        assert first.get("ok")
        assert first.get("deduped") is not True
        drafts = registry.get_pending_draft_alerts(kind="stop_monitoring")
        assert len(drafts) == 1
        assert drafts[0]["severity"] == "CRITICAL"

        second = propose_stop_monitoring(
            "Stop monitoring on Monitoring",
            tab_id="t-crit",
            agent_reasoning="The tab 'Monitoring' is in a critical health state",
            host=host,
        )
        assert second.get("ok")
        assert second.get("deduped") is True
        assert second.get("draft_id") == first.get("draft_id")
        assert len(registry.get_pending_draft_alerts(kind="stop_monitoring")) == 1

        # Time-cache alone would allow another cycle; pending gate still blocks
        clear_dedupe_cache()
        tabs = [
            {
                "tab_id": "t-crit",
                "snapshot": {
                    "title": "Monitoring",
                    "health_state": "critical",
                    "monitoring_active": True,
                    "drift": False,
                    "obs_ok": True,
                },
            }
        ]
        actions = proactive_actions_from_tabs(tabs, dedupe_seconds=60, host=host)
        stop_acts = [a for a in actions if a["tool"] == "propose_stop_monitoring"]
        assert stop_acts == []  # pending exists
    finally:
        set_tool_host(None)
        clear_dedupe_cache()


def test_watchlist_persist(tmp_path: Path):
    path = tmp_path / "wl.json"
    add_to_watchlist("tab-1", title="Battery", note="focus", path=path)
    cur = load_watchlist(path=path)
    assert len(cur["tabs"]) == 1
    assert cur["tabs"][0]["tab_id"] == "tab-1"
    remove_from_watchlist("tab-1", path=path)
    assert load_watchlist(path=path)["tabs"] == []


def test_pending_actions_summary(tmp_path: Path):
    registry = ModelRegistry(db_path=str(tmp_path / "models.sqlite"))
    registry.create_draft_alert(
        tab_id="t1",
        kind="alert",
        proposed_message="test",
        severity="CRITICAL",
    )

    class Host:
        def list_pending_drafts(self):
            return registry.get_pending_draft_alerts()

        def get_pending_retrain_signals(self):
            return []

    out = pending_actions_summary(Host())
    assert out["pending_draft_count"] == 1
    assert "CRITICAL" in out["summary"] or "pending" in out["summary"].lower()


def test_runner_cycle_creates_proactive_drafts(tmp_path: Path):
    clear_dedupe_cache()
    registry = ModelRegistry(db_path=str(tmp_path / "models.sqlite"))
    audit = AgentAuditLog(db_path=str(tmp_path / "audit.sqlite"))

    class Tab:
        def __init__(self):
            self.tab_id = "t1"
            self.config = {"title": "Batt"}
            self.last_snapshot = {
                "tab_id": "t1",
                "title": "Batt",
                "health_state": "Warning",
                "fusion_score": 0.7,
                "drift": True,
                "obs_ok": True,
                "monitoring_active": True,
                "drift_score": 0.9,
                "alert_count": 1,
                "trained_models": 1,
                "updated_at": "now",
            }
            self.last_drift_result = {"is_drift": True, "drift_score": 0.9}
            self.score_history = []
            self.anomaly_events = []

        def get_snapshot(self):
            return dict(self.last_snapshot)

    class Window:
        def __init__(self):
            self.custom_tabs = {"t1": Tab()}
            self.model_registry = registry

    host = MainWindowToolHost(Window())
    set_tool_host(host)
    try:
        runner = AgentRunner(audit, lambda: host, allow_llm=False)
        cycle = runner.run_agent_cycle(task="fleet_monitor")
        assert cycle["ok"] is True
        assert any(r.get("ok") for r in (cycle.get("proactive_results") or []))
        drafts = registry.get_pending_draft_alerts()
        retrains = registry.get_pending_retrain_signals()
        assert drafts or retrains
    finally:
        set_tool_host(None)
        clear_dedupe_cache()
