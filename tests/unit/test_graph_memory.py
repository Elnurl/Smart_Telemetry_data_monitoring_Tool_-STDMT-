"""Sprint 2 — GraphMemoryStore (NetworkX + SQLite) unit tests."""

from __future__ import annotations

import pytest

from app.agent.memory.factory import get_graph_memory, reset_graph_memory_for_tests
from app.agent.memory.networkx_store import NetworkXGraphStore
from app.agent.memory.sync import record_decision_memory, sync_fleet_into


@pytest.fixture(autouse=True)
def _reset():
    reset_graph_memory_for_tests()
    yield
    reset_graph_memory_for_tests()


def test_upsert_and_recall(tmp_path):
    store = NetworkXGraphStore(db_path=str(tmp_path / "g.sqlite"))
    store.upsert_node(
        "Tab",
        "t1",
        {"title": "Battery", "health": "Warning", "mission_mode": "eclipse"},
    )
    store.upsert_node("Mode", "eclipse", {"threshold_scale": 1.5})
    store.upsert_edge("Tab", "t1", "IN_MODE", "Mode", "eclipse")
    hits = store.recall("battery eclipse warning", tab_id="t1", limit=10)
    assert hits
    assert any("Battery" in (h.get("title") or "") or h.get("id") == "Tab:t1" for h in hits)
    st = store.stats()
    assert st["nodes"] >= 2
    assert st["edges"] >= 1
    assert st["backend"] == "networkx"


def test_persistence_reload(tmp_path):
    db = str(tmp_path / "persist.sqlite")
    a = NetworkXGraphStore(db_path=db)
    a.upsert_node("Tab", "persist-me", {"title": "EPS", "health": "OK"})
    a.upsert_edge("Tab", "persist-me", "IN_MODE", "Mode", "nominal")
    del a
    b = NetworkXGraphStore(db_path=db)
    node = b.get_node("Tab", "persist-me")
    assert node is not None
    assert node.get("title") == "EPS"
    assert b.stats()["nodes"] >= 2


def test_sync_from_fake_host(tmp_path):
    store = NetworkXGraphStore(db_path=str(tmp_path / "sync.sqlite"))

    class Host:
        def list_tabs(self):
            return [
                {
                    "tab_id": "tab-a",
                    "title": "Eclipse",
                    "snapshot": {
                        "title": "Eclipse",
                        "health_state": "Warning",
                        "mission_mode": "eclipse",
                        "threshold_scale": 1.5,
                        "monitoring_active": True,
                        "trained_models": 2,
                        "alert_count": 1,
                    },
                }
            ]

        def list_pending_drafts(self):
            return [
                {
                    "draft_id": "d1",
                    "tab_id": "tab-a",
                    "kind": "alert",
                    "severity": "WARNING",
                    "status": "pending",
                    "proposed_message": "Check thermal",
                }
            ]

        def get_pending_retrain_signals(self):
            return [
                {
                    "id": "r1",
                    "tab_id": "tab-a",
                    "drift_score": 0.8,
                    "status": "pending",
                    "drifted_features": ["temp"],
                }
            ]

    out = sync_fleet_into(store, Host())
    assert out["ok"] is True
    assert out["tabs"] == 1
    assert out["drafts"] == 1
    assert out["retrain"] == 1
    assert store.get_node("Draft", "d1") is not None
    neigh = store.neighbors("Tab", "tab-a", rel="HAS_DRAFT")
    assert any(n.get("id") == "Draft:d1" for n in neigh)


def test_record_decision_memory(tmp_path):
    store = NetworkXGraphStore(db_path=str(tmp_path / "dec.sqlite"))
    store.upsert_node("Tab", "t9", {"title": "Bus"})
    record_decision_memory(
        store,
        decision_id=42,
        tab_id="t9",
        outcome="function_calling",
        tool_called="get_fleet_status",
        reasoning="Fleet quiet",
        tool_trace=["get_fleet_status", "search_agent_memory"],
    )
    assert store.get_node("Decision", "42") is not None
    assert store.get_node("Tool", "get_fleet_status") is not None
    links = store.neighbors("Tab", "t9", rel="HAD_DECISION")
    assert any(n.get("id") == "Decision:42" for n in links)


def test_factory_returns_networkx(tmp_path, monkeypatch):
    monkeypatch.delenv("STDMS_GRAPH_BACKEND", raising=False)
    reset_graph_memory_for_tests()
    store = get_graph_memory(db_path=str(tmp_path / "fac.sqlite"), _fresh=True)
    assert isinstance(store, NetworkXGraphStore)


def test_neo4j_stub_raises():
    from app.agent.memory.neo4j_store import Neo4jGraphStore

    with pytest.raises(NotImplementedError, match="Neo4j"):
        Neo4jGraphStore()


def test_search_agent_memory_tool(tmp_path, monkeypatch):
    from app.agent.tools import search_agent_memory

    reset_graph_memory_for_tests()
    store = get_graph_memory(db_path=str(tmp_path / "tool.sqlite"), _fresh=True)
    store.upsert_node("Tab", "tx", {"title": "Thermal", "health": "Warning"})

    class Host:
        def list_tabs(self):
            return []

        def list_pending_drafts(self):
            return []

        def get_pending_retrain_signals(self):
            return []

    # Bypass throttle by calling store.recall path via tool after sync noop
    out = search_agent_memory("thermal warning", tab_id="tx", host=Host())
    assert out["ok"] is True
    assert out["count"] >= 1
