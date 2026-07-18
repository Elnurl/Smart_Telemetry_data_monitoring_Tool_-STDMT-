"""Phase 2 — function calling unit tests (mocked Ollama)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import pytest

from app.agent.audit import AgentAuditLog
from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.function_calling import run_function_calling
from app.agent.runner import AgentRunner
from app.agent.tools import invoke_tool, list_tools, to_ollama_tools


class _FakeTab:
    def __init__(self, tab_id, title="CH1"):
        self.tab_id = tab_id
        self.config = {"title": title}
        self.last_snapshot = {
            "tab_id": tab_id,
            "title": title,
            "health_state": "Warning",
            "drift": True,
            "obs_ok": True,
            "alert_count": 2,
            "trained_models": 1,
            "updated_at": "2026-07-13 12:00:00",
            "monitoring_active": True,
        }
        self.anomaly_events = deque(
            [{"timestamp": "2026-07-13 11:59:00", "health_state": "Warning"}],
            maxlen=50,
        )

    def get_snapshot(self):
        return dict(self.last_snapshot)


class _FakeRegistry:
    def get_pending_retrain_signals(self):
        return [{"id": 9, "tab_id": "t1", "drift_score": 0.6, "acknowledged": 0}]


class _FakeWindow:
    def __init__(self):
        self.custom_tabs = {"t1": _FakeTab("t1", "CH1")}
        self.model_registry = _FakeRegistry()


@pytest.fixture
def host():
    h = MainWindowToolHost(_FakeWindow())
    set_tool_host(h)
    yield h
    set_tool_host(None)


def test_to_ollama_tools_schema():
    tools = to_ollama_tools()
    assert tools
    names = {t["function"]["name"] for t in tools}
    assert "get_all_snapshots" in names
    assert "get_tab_snapshot" in names
    assert "train_model" not in names
    snap = next(t for t in tools if t["function"]["name"] == "get_tab_snapshot")
    assert "tab_id" in snap["function"]["parameters"]["properties"]


def test_invoke_rejects_unknown_and_passes_host(host):
    with pytest.raises(KeyError):
        invoke_tool("train_model", {})
    snap = invoke_tool("get_tab_snapshot", {"tab_id": "t1"}, host=host)
    assert snap["title"] == "CH1"
    signals = invoke_tool("check_pending_retrain_signals", {}, host=host)
    assert signals[0]["id"] == 9


def test_function_calling_loop_executes_tool_then_answers(host):
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_tab_snapshot",
                            "arguments": {"tab_id": "t1"},
                        },
                    }
                ],
            }
        # After tool result
        return {
            "role": "assistant",
            "content": "CH1 needs attention due to Warning health and drift.",
            "tool_calls": [],
        }

    result = run_function_calling(
        "Which tab needs attention?",
        host=host,
        chat_fn=fake_chat,
        generate_fn=lambda *a, **k: None,
        reachable_fn=lambda *a, **k: True,
    )
    assert result["ok"] is True
    assert result["llm_used"] is True
    assert result["outcome"] == "function_calling"
    assert len(result["tool_trace"]) == 1
    assert result["tool_trace"][0]["tool"] == "get_tab_snapshot"
    assert result["tool_trace"][0]["ok"] is True
    assert "CH1" in result["reply"]


def test_function_calling_offline_returns_no_reply(host):
    result = run_function_calling(
        "hello",
        host=host,
        reachable_fn=lambda *a, **k: False,
    )
    assert result["ok"] is True
    assert result["reply"] is None
    assert result["llm_used"] is False
    assert result["outcome"] == "ollama_offline"


def test_runner_ask_with_mocked_fc(tmp_path, host, monkeypatch):
    def fake_fc(msg, **kwargs):
        return {
            "ok": True,
            "reply": "Fleet is quiet.",
            "llm_used": True,
            "tool_trace": [{"tool": "get_fleet_status", "params": {}, "ok": True}],
            "outcome": "function_calling",
        }

    # Multi-node ask: R-LLM heuristic + RA-LLM calls function_calling
    monkeypatch.setattr("app.agent.function_calling.run_function_calling", fake_fc)
    monkeypatch.setattr("app.agent.nodes._chat", lambda **kwargs: None)
    audit = AgentAuditLog(tmp_path / "agent_decisions.db")
    runner = AgentRunner(audit=audit, tool_host_getter=lambda: host, allow_llm=True)
    out = runner.ask("Give me a full health report for all tabs")
    assert out["phase"] == 3
    assert out["route"] == "tool"
    assert out["node"] == "RA-LLM"
    assert out["llm_used"] is True
    assert out["tool_trace"][0]["tool"] == "get_fleet_status"
    assert "[R-LLM]" in out["reply"]
    assert audit.list_decisions(limit=1)[0]["outcome"] == "function_calling"


def test_phase3_tools_in_schema():
    names = {t["function"]["name"] for t in to_ollama_tools(include_mutating=True)}
    assert "propose_alert" in names
    assert "search_knowledge" in names
    assert "get_fleet_status" in names
