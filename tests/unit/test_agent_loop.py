"""Phase 1 — AgentMonitorLoop unit tests."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from app.agent.audit import AgentAuditLog
from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.loop import AgentMonitorLoop
from app.agent.ollama_client import DEFAULT_OLLAMA_URL, call_ollama, ollama_reachable


class _FakeTab:
    def __init__(self, tab_id, title="CH1", *, drift=False, fusion=0.12, health="Nominal", alerts=0):
        self.tab_id = tab_id
        self.config = {"title": title}
        self.last_snapshot = {
            "tab_id": tab_id,
            "title": title,
            "health_state": health,
            "drift": drift,
            "fusion_score": fusion,
            "obs_ok": True,
            "alert_count": alerts,
            "trained_models": 2,
            "last_file": "telem.csv",
            "updated_at": "2026-07-12 12:00:00",
            "monitoring_active": True,
        }
        self.anomaly_events = deque(maxlen=10)

    def get_snapshot(self):
        return dict(self.last_snapshot)


class _FakeRegistry:
    def get_pending_retrain_signals(self):
        return []


class _FakeWindow:
    def __init__(self):
        self.custom_tabs = {
            "t1": _FakeTab("t1", "CH1", drift=False, fusion=0.12),
            "t2": _FakeTab("t2", "CH2", drift=True, fusion=0.41, health="Warning", alerts=2),
        }
        self.model_registry = _FakeRegistry()


@dataclass
class _Bridge:
    tool_host: Any
    audit: Any
    runner: Any = None
    server: Any = None
    allow_llm: bool = False


@pytest.fixture
def bridge(tmp_path):
    window = _FakeWindow()
    host = MainWindowToolHost(window)
    set_tool_host(host)
    audit = AgentAuditLog(tmp_path / "agent_decisions.db")
    ctx = _Bridge(tool_host=host, audit=audit, allow_llm=False)
    yield ctx
    set_tool_host(None)


@pytest.fixture
def force_ollama_offline(monkeypatch):
    """Keep unit tests deterministic even when a local Ollama is running."""
    monkeypatch.setattr("app.agent.loop.ollama_reachable", lambda *_a, **_k: False)
    monkeypatch.setattr("app.agent.loop.call_ollama", lambda *_a, **_k: None)
    monkeypatch.setattr("app.agent.ollama_client.ollama_reachable", lambda *_a, **_k: False)
    monkeypatch.setattr("app.agent.ollama_client.call_ollama", lambda *_a, **_k: None)
    monkeypatch.setattr("app.agent.function_calling.ollama_reachable", lambda *_a, **_k: False)
    monkeypatch.setattr("app.agent.function_calling.chat_ollama", lambda *_a, **_k: None)
    monkeypatch.setattr("app.agent.function_calling.call_ollama", lambda *_a, **_k: None)


def test_rule_based_summary_per_tab(bridge):
    loop = AgentMonitorLoop(bridge)
    snapshots = []
    for entry in bridge.tool_host.list_tabs():
        snap = dict(entry.get("snapshot") or {})
        snap.setdefault("tab_id", entry.get("tab_id"))
        snap.setdefault("title", entry.get("title"))
        snapshots.append(snap)

    summary, lines = loop._rule_based_summary(snapshots)
    assert len(lines) >= 2
    assert any("CH1" in ln for ln in lines)
    assert any("CH2" in ln for ln in lines)
    assert "Priority" in summary or "drift" in summary.lower() or "Retrain" in summary or "attention" in summary.lower()
    joined = "\n".join(lines)
    assert "Heuristic" in joined or "alert" in joined.lower() or "drift" in joined.lower()


def test_monitor_loop_ask_without_ollama(bridge, force_ollama_offline):
    loop = AgentMonitorLoop(bridge, allow_llm=False)
    result = loop.ask("Which tab needs attention?")
    assert result["ok"] is True
    assert result["llm_used"] is False
    assert "Ollama offline" in result["reply"] or "Heuristic" in result["reply"] or "CH" in result["reply"]
    decisions = bridge.audit.list_decisions(limit=5)
    assert any(d.get("tool_called") == "agent_chat" for d in decisions)


def test_monitor_loop_run_once_audits_and_signals(bridge, force_ollama_offline):
    pytest.importorskip("PyQt5")
    from PyQt5.QtCore import QCoreApplication
    import sys

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    loop = AgentMonitorLoop(bridge, allow_llm=False)
    received = []

    def _on(payload):
        received.append(payload)

    loop.signals.refreshed.connect(_on)
    result = loop.run_once()

    assert result["ok"] is True
    assert result["llm_used"] is False
    assert result["decision_id"] is not None
    assert "CH1" in result["summary"] or any("CH1" in ln for ln in result["lines"])

    app.processEvents()
    assert received, "Dashboard signal should fire"
    assert received[0]["decision_id"] == result["decision_id"]

    decisions = bridge.audit.list_decisions(limit=5)
    assert decisions
    assert decisions[0]["tool_called"] == "agent_monitor_cycle"
    assert decisions[0]["outcome"] == "observation_only"


def test_monitor_loop_start_stop(bridge, force_ollama_offline):
    loop = AgentMonitorLoop(bridge, allow_llm=False)
    loop.start(interval_seconds=5)
    assert loop.is_running
    # Wait until first cycle finishes (may include a short Ollama probe timeout)
    deadline = time.time() + 5.0
    while loop.last_result is None and time.time() < deadline:
        time.sleep(0.1)
    loop.stop(timeout=2.0)
    assert not loop.is_running
    assert loop.last_result is not None
    assert loop.last_result.get("ok") is True


def test_build_context_includes_tabs(bridge):
    loop = AgentMonitorLoop(bridge)
    snapshots = []
    for entry in bridge.tool_host.list_tabs():
        snap = dict(entry.get("snapshot") or {})
        snap.setdefault("title", entry.get("title"))
        snapshots.append(snap)
    ctx = loop._build_context(snapshots, pending_retrain=[])
    assert "CH1" in ctx
    assert "CH2" in ctx
    assert "health=" in ctx
