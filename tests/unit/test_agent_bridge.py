"""Phase 0 Instrumentation Agent — read-only bridge tests."""

from __future__ import annotations

from collections import deque

import pytest

from app.agent.audit import AgentAuditLog
from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.runner import AgentRunner
from app.agent.tools import (
    check_pending_retrain_signals,
    get_all_snapshots,
    get_tab_history,
    get_tab_snapshot,
    invoke_tool,
    list_tools,
)


class _FakeTab:
    def __init__(self, tab_id, title="CH1"):
        self.tab_id = tab_id
        self.config = {"title": title}
        self.last_snapshot = {
            "tab_id": tab_id,
            "title": title,
            "health_state": "Nominal",
            "drift": False,
            "obs_ok": True,
            "alert_count": 1,
            "trained_models": 2,
            "updated_at": "2026-07-04 12:00:00",
        }
        self.anomaly_events = deque(
            [
                {"timestamp": "2026-07-04 11:59:00", "health_state": "Warning", "fused_score": 0.8},
                {"timestamp": "2026-07-04 11:58:00", "health_state": "Nominal", "fused_score": 0.1},
            ],
            maxlen=500,
        )

    def get_snapshot(self):
        return dict(self.last_snapshot)


class _FakeRegistry:
    def get_pending_retrain_signals(self):
        return [{"id": 1, "tab_id": "t1", "drift_score": 0.5, "acknowledged": 0}]


class _FakeWindow:
    def __init__(self):
        self.custom_tabs = {
            "t1": _FakeTab("t1", "CH1"),
            "t2": _FakeTab("t2", "CH2"),
        }
        self.model_registry = _FakeRegistry()


@pytest.fixture
def host():
    h = MainWindowToolHost(_FakeWindow())
    set_tool_host(h)
    yield h
    set_tool_host(None)


def test_list_tools_are_read_only():
    tools = list_tools()
    assert tools
    assert all(not t.get("mutates") for t in tools)
    names = {t["name"] for t in tools}
    assert "get_all_snapshots" in names
    assert "train_model" not in names
    assert "send_alert" not in names


def test_main_window_host_list_and_snapshot(host):
    tabs = host.list_tabs()
    assert len(tabs) == 2
    assert tabs[0]["snapshot"]["health_state"] == "Nominal"

    snap = host.get_snapshot("t1")
    assert snap["title"] == "CH1"
    assert host.get_snapshot("missing") is None


def test_history_and_retrain_signals(host):
    history = host.get_history("t1", n=1)
    assert len(history) == 1
    assert history[0]["health_state"] == "Warning"

    signals = host.get_pending_retrain_signals()
    assert signals[0]["id"] == 1


def test_tool_registry_dispatch(host):
    tabs = get_all_snapshots()
    assert len(tabs) == 2
    assert get_tab_snapshot("t1")["tab_id"] == "t1"
    assert len(get_tab_history("t1", n=10)) == 2
    assert check_pending_retrain_signals()[0]["drift_score"] == 0.5
    assert invoke_tool("get_tab_snapshot", {"tab_id": "t2"})["title"] == "CH2"


def test_audit_and_runner_observation_only(tmp_path, host, monkeypatch):
    import app.agent.ollama_client as oc

    monkeypatch.setattr(oc, "ollama_reachable", lambda *_a, **_k: False)
    monkeypatch.setattr(oc, "call_ollama", lambda *_a, **_k: None)

    audit = AgentAuditLog(tmp_path / "agent_decisions.db")
    runner = AgentRunner(audit=audit, tool_host_getter=lambda: host)
    result = runner.run(tab_id="t1", task="summarize")
    assert result["ok"] is True
    assert result["outcome"] == "observation_only"
    assert result["air_gap"] is True
    assert result["llm_used"] is False
    assert "CH1" in result["context_summary"]
    assert "No data left this host" in result["reasoning"] or "Ollama" in result["reasoning"]

    decisions = audit.list_decisions(limit=10)
    assert len(decisions) == 1
    assert decisions[0]["tool_called"] == "get_tab_snapshot"
    assert decisions[0]["outcome"] == "observation_only"


def test_air_gap_policy_rejects_non_loopback_bind():
    from app.agent.policy import assert_loopback_bind, is_loopback_url

    assert assert_loopback_bind("127.0.0.1") == "127.0.0.1"
    assert assert_loopback_bind("localhost") == "127.0.0.1"
    with pytest.raises(ValueError):
        assert_loopback_bind("0.0.0.0")
    with pytest.raises(ValueError):
        assert_loopback_bind("192.168.1.10")
    assert is_loopback_url("http://127.0.0.1:11434") is True
    assert is_loopback_url("https://api.openai.com/v1") is False


def test_fastapi_tabs_endpoint(tmp_path, host):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.agent.server import create_app

    audit = AgentAuditLog(tmp_path / "agent_decisions.db")
    runner = AgentRunner(audit=audit, tool_host_getter=lambda: host)
    client = TestClient(create_app(audit=audit, runner=runner))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["tool_host_attached"] is True
    assert health.json()["air_gap"] is True
    assert health.json()["llm_enabled"] is False
    assert health.json()["data_egress"] == "none"
    assert health.json()["phase"] == 4
    assert health.json()["mutating_tools"] is True
    assert "knowledge" in health.json()

    tabs = client.get("/tabs")
    assert tabs.status_code == 200
    body = tabs.json()
    assert len(body["tabs"]) == 2
    assert body["tabs"][0]["snapshot"]["health_state"] == "Nominal"

    snap = client.get("/tabs/t1/snapshot")
    assert snap.status_code == 200
    assert snap.json()["snapshot"]["title"] == "CH1"

    hist = client.get("/tabs/t1/history?n=1")
    assert hist.status_code == 200
    assert len(hist.json()["history"]) == 1

    missing = client.get("/tabs/nope/snapshot")
    assert missing.status_code == 404

    run = client.post("/agent/run", json={"tab_id": "t1", "task": "summarize"})
    assert run.status_code == 200
    assert run.json()["outcome"] == "observation_only"

    decisions = client.get("/agent/decisions")
    assert decisions.status_code == 200
    assert len(decisions.json()["decisions"]) >= 1
