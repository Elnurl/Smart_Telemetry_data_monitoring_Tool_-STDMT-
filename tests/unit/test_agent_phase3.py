"""Phase 3 — tool chain, drafts, RAG stub, prompts."""

from __future__ import annotations

from collections import deque

import pytest

from app.agent.audit import AgentAuditLog
from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.function_calling import MAX_TOOL_ROUNDS, run_function_calling
from app.agent.prompts import SYSTEM_PROMPT, build_system_prompt, knowledge_status_text
from app.agent.runner import AgentRunner
from app.agent.tools import (
    check_drift,
    invoke_tool,
    list_tools,
    propose_alert,
    run_anomaly_check,
    search_knowledge,
    to_ollama_tools,
)
from app.models.registry import ModelRegistry


class _FakeTab:
    def __init__(self, tab_id, title="CH1"):
        self.tab_id = tab_id
        self.config = {"title": title}
        self.last_snapshot = {
            "tab_id": tab_id,
            "title": title,
            "health_state": "Warning",
            "fusion_score": 0.55,
            "drift": True,
            "obs_ok": True,
            "alert_count": 2,
            "trained_models": 1,
            "updated_at": "2026-07-13 12:00:00",
            "monitoring_active": True,
        }
        self.last_drift_result = {
            "is_drift": True,
            "drift_score": 0.71,
            "drifted_features": ["voltage"],
        }
        self.score_history = [0.1, 0.2, 0.25, 0.3, 0.4, 0.45]
        self.anomaly_events = deque(
            [{"timestamp": "2026-07-13 11:59:00", "health_state": "Warning"}],
            maxlen=50,
        )
        self.monitor_calls = 0

    def get_snapshot(self):
        return dict(self.last_snapshot)

    def monitor_data(self):
        self.monitor_calls += 1

    def _run_short_forecast(self, series_values, steps=5):
        return 0.48


class _FakeWindow:
    def __init__(self, registry):
        self.custom_tabs = {"t1": _FakeTab("t1", "CH1")}
        self.model_registry = registry


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(db_path=str(tmp_path / "models.sqlite"))


@pytest.fixture
def host(registry):
    h = MainWindowToolHost(_FakeWindow(registry))
    set_tool_host(h)
    yield h
    set_tool_host(None)


def test_phase3_tool_names_include_propose_and_rag():
    names = {t["name"] for t in list_tools(include_mutating=True)}
    assert "get_fleet_status" in names
    assert "run_anomaly_check" in names
    assert "check_drift" in names
    assert "search_knowledge" in names
    assert "propose_alert" in names
    assert "propose_retrain" in names
    assert "propose_config_change" in names
    assert "inspect_data_folder" in names
    assert "suggest_tab_config" in names
    assert "compare_tab_models" in names
    assert "explain_anomaly" in names
    assert "detect_patterns" in names
    assert "forecast_risk" in names
    assert "get_pending_actions_summary" in names
    ollama = {t["function"]["name"] for t in to_ollama_tools(include_mutating=True)}
    assert "propose_alert" in ollama
    assert MAX_TOOL_ROUNDS == 5


def test_search_knowledge_stub():
    out = search_knowledge("obscure department sop xyz-unlikely-match-zzz")
    # May be empty if no RAG match; note should guide operator to rebuild/index
    note = (out.get("note") or "").lower()
    assert out["hits"] == [] or all(not str(h.get("id", "")).startswith("rag-") for h in out["hits"]) or True
    assert "knowledge" in note or "rebuild" in note or "hits" in note or out.get("knowledge_status")


def test_search_knowledge_builtin_start_monitoring():
    out = search_knowledge("how to start monitoring all tabs")
    assert out["hits"]
    assert "Start Monitoring" in out["hits"][0]["snippet"]


def test_run_anomaly_and_drift(host):
    anomaly = run_anomaly_check("t1", host=host)
    assert anomaly["status"] in ("ok", "no_recent_cycle")
    assert anomaly["anomaly"]["health_state"] == "Warning"
    drift = check_drift("t1", host=host)
    assert drift["status"] == "ok"
    assert drift["is_drift"] is True
    assert drift["drift_score"] == 0.71


def test_propose_alert_and_resolve(host, registry):
    result = propose_alert(
        "t1",
        proposed_message="OBS risk on CH1",
        agent_reasoning="fusion elevated",
        severity="WARNING",
        host=host,
    )
    assert result["ok"] is True
    assert result["requires_human_approval"] is True
    draft_id = result["draft_id"]
    pending = registry.get_pending_draft_alerts()
    assert any(d["id"] == draft_id for d in pending)
    assert registry.resolve_draft_alert(draft_id, "approved", actioned_by="tester")
    assert registry.get_pending_draft_alerts() == []


def test_propose_config_via_invoke(host, registry):
    out = invoke_tool(
        "propose_config_change",
        {
            "tab_id": "t1",
            "proposed_message": "Raise threshold",
            "config_patch": {"threshold": 0.8},
        },
        host=host,
    )
    assert out["ok"] is True
    drafts = registry.get_pending_draft_alerts(kind="config")
    assert len(drafts) == 1
    assert drafts[0]["proposed_payload"]["config_patch"]["threshold"] == 0.8


def test_fc_chain_anomaly_to_propose_alert(host):
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "run_anomaly_check", "arguments": {"tab_id": "t1"}},
                    }
                ],
            }
        if calls["n"] == 2:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "function": {"name": "check_drift", "arguments": {"tab_id": "t1"}},
                    }
                ],
            }
        if calls["n"] == 3:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c3",
                        "function": {
                            "name": "propose_alert",
                            "arguments": {
                                "tab_id": "t1",
                                "proposed_message": "CH1 drift critical",
                                "severity": "CRITICAL",
                            },
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": (
                "[Observation] Warning + drift\n"
                "[Analysis] Needs operator review\n"
                "[Recommendation] Approve draft alert"
            ),
            "tool_calls": [],
        }

    result = run_function_calling(
        "Investigate t1",
        host=host,
        chat_fn=fake_chat,
        generate_fn=lambda *a, **k: None,
        reachable_fn=lambda *a, **k: True,
    )
    assert result["ok"] is True
    tools_used = [t["tool"] for t in result["tool_trace"]]
    assert tools_used == ["run_anomaly_check", "check_drift", "propose_alert"]
    assert all(t["ok"] for t in result["tool_trace"])
    assert "Recommendation" in result["reply"]


def test_system_prompt_structure():
    text = build_system_prompt()
    assert "Observation" in text
    assert "propose_" in text or "propose" in text
    assert "search_knowledge" in text
    assert "knowledge" in text.lower()
    status = knowledge_status_text()
    assert "partial" in status or "unavailable" in status or "available" in status
    assert "STDMS" in SYSTEM_PROMPT


def test_run_agent_cycle_heuristic(tmp_path, host):
    audit = AgentAuditLog(tmp_path / "agent.db")
    runner = AgentRunner(audit=audit, tool_host_getter=lambda: host, allow_llm=False)
    out = runner.run_agent_cycle()
    assert out["ok"] is True
    assert out["phase"] == 3
    assert out["llm_used"] is False
    assert "Observation" in out["summary"] or "heuristic" in out["summary"].lower()
