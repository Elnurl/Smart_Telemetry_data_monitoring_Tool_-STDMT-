"""Phase 0 Instrumentation Agent — read-only bridge tests."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

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
        self.config = {
            "title": title,
            "models": [
                {
                    "model_id": f"{tab_id}-model",
                    "model_type": "Isolation Forest",
                    "model_parameters": {},
                }
            ],
        }
        self.models = {}
        self.selected_model_id = None
        self.monitoring_active = False
        self.trained_model_ids = []
        import pandas as pd

        self.data_processor = SimpleNamespace(
            data=pd.DataFrame({"time": ["t1", "t2"], "value": [1.0, 2.0]}),
            preprocessed_data=None,
        )
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

    def train_model(self, model_id=None, silent=False):
        self.trained_model_ids.append(model_id)
        return True

    def stop_monitoring(self):
        self.monitoring_active = False


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
        self.tab_config_manager = SimpleNamespace(
            add_config=lambda *_args, **_kwargs: None,
            remove_config=lambda *_args, **_kwargs: None,
        )

    def _create_custom_tab_from_config(self, config):
        tab_id = f"t{len(self.custom_tabs) + 1}"
        self.custom_tabs[tab_id] = _FakeTab(tab_id, config["title"])
        return tab_id


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


def test_api_token_rejects_short_secret():
    from app.agent.server import resolve_api_token

    with pytest.raises(ValueError):
        resolve_api_token("too-short")


def test_toolhost_mutation_respects_current_user_rbac():
    window = _FakeWindow()
    window.current_username = "viewer"
    window.service_layer = SimpleNamespace(
        authorize=lambda _username, _permission, resource="*": False
    )
    restricted = MainWindowToolHost(window)

    result = restricted.create_tab({"title": "Denied"}, requested_by="test")
    assert result["ok"] is False
    assert result["status"] == "forbidden"


def test_fastapi_tabs_endpoint(tmp_path, host):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.agent.server import create_app

    audit = AgentAuditLog(tmp_path / "agent_decisions.db")
    runner = AgentRunner(audit=audit, tool_host_getter=lambda: host)
    token = "test-token-with-at-least-24-characters"
    client = TestClient(create_app(audit=audit, runner=runner, api_token=token))
    auth = {"Authorization": f"Bearer {token}"}

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["tool_host_attached"] is True
    assert health.json()["air_gap"] is True
    assert health.json()["llm_enabled"] is False
    assert health.json()["data_egress"] == "none"
    assert health.json()["phase"] == 4
    assert health.json()["sprint"] == 3
    assert health.json()["auth"] == "bearer_required"
    assert health.json()["mutating_tools"] is True
    assert "knowledge" in health.json()

    assert client.get("/tabs").status_code == 401
    assert client.get("/tabs", headers={"Authorization": "Bearer wrong"}).status_code == 401

    tabs = client.get("/tabs", headers=auth)
    assert tabs.status_code == 200
    body = tabs.json()
    assert len(body["tabs"]) == 2
    assert body["tabs"][0]["snapshot"]["health_state"] == "Nominal"
    versioned_tabs = client.get("/v1/ops/tabs", headers=auth)
    assert versioned_tabs.status_code == 200
    assert len(versioned_tabs.json()["tabs"]) == 2
    assert client.get("/v1/agent/tools", headers=auth).status_code == 200

    definition = client.get("/tabs/t1", headers=auth)
    assert definition.status_code == 200
    assert definition.json()["config"]["title"] == "CH1"

    snap = client.get("/tabs/t1/snapshot", headers=auth)
    assert snap.status_code == 200
    assert snap.json()["snapshot"]["title"] == "CH1"

    hist = client.get("/tabs/t1/history?n=1", headers=auth)
    assert hist.status_code == 200
    assert len(hist.json()["history"]) == 1

    missing = client.get("/tabs/nope/snapshot", headers=auth)
    assert missing.status_code == 404

    schema = client.get("/tabs/t1/data/schema", headers=auth)
    assert schema.status_code == 200
    assert schema.json()["row_count"] == 2
    assert schema.json()["columns"][1]["name"] == "value"

    preview = client.get("/tabs/t1/data/preview?limit=1", headers=auth)
    assert preview.status_code == 200
    assert preview.json()["returned"] == 1
    assert preview.json()["rows"][0]["value"] == 2.0

    created = client.post(
        "/tabs",
        json={"config": {"title": "API Tab"}, "requested_by": "test"},
        headers=auth,
    )
    assert created.status_code == 202
    created_op = client.get(
        f"/operations/{created.json()['operation_id']}",
        headers=auth,
    ).json()
    assert created_op["status"] == "completed"
    created_tab_id = created_op["result"]["tab_id"]

    updated = client.patch(
        f"/tabs/{created_tab_id}",
        json={"updates": {"title": "API Tab Updated"}, "requested_by": "test"},
        headers=auth,
    )
    assert updated.status_code == 202
    assert client.get(f"/tabs/{created_tab_id}", headers=auth).json()["config"]["title"] == (
        "API Tab Updated"
    )

    deleted = client.delete(f"/tabs/{created_tab_id}?requested_by=test", headers=auth)
    assert deleted.status_code == 202
    assert client.get(f"/tabs/{created_tab_id}", headers=auth).status_code == 404

    train = client.post(
        "/tabs/t1/train",
        json={"model_id": "t1-model", "requested_by": "test"},
        headers=auth,
    )
    assert train.status_code == 202
    operation_id = train.json()["operation_id"]
    operation = client.get(f"/operations/{operation_id}", headers=auth)
    assert operation.status_code == 200
    assert operation.json()["status"] == "completed"

    run = client.post(
        "/agent/run",
        json={"tab_id": "t1", "task": "summarize"},
        headers=auth,
    )
    assert run.status_code == 200
    assert run.json()["outcome"] == "observation_only"

    decisions = client.get("/agent/decisions", headers=auth)
    assert decisions.status_code == 200
    assert len(decisions.json()["decisions"]) >= 1
