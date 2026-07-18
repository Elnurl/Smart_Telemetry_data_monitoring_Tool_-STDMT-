"""Anti-spam: create_tab once + best-model deterministic answers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.function_calling import (
    _dedupe_tool_calls,
    _deterministic_create_tab,
    _is_best_model_intent,
    _is_create_tab_intent,
    _is_informational_ask,
    run_function_calling,
    synthesize_fleet_answer,
)
from app.agent.bridge import MainWindowToolHost
from app.agent.tools import propose_create_tab
from app.models.registry import ModelRegistry


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    f = tmp_path / "BatteryTemperature.csv"
    f.write_text(
        "timestamp,Battery_Temp,Bus_Voltage\n"
        + "\n".join(f"2026-01-01T00:{i:02d}:00,{20 + i},{28}" for i in range(40))
        + "\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def host(tmp_path: Path):
    registry = ModelRegistry(db_path=str(tmp_path / "models.sqlite"))

    class Host:
        def list_pending_drafts(self):
            return registry.get_pending_draft_alerts()

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
            return {
                "ok": True,
                "draft_id": draft_id,
                "kind": kind,
                "requires_human_approval": True,
            }

        def list_tabs(self):
            return []

        def list_tab_models(self, tab_id):
            return {"ok": False, "error": "tab_not_found"}

    return Host(), registry


def test_create_tab_intent_detects_path():
    assert _is_create_tab_intent(
        r"Create a tab from 'C:\Users\Elnur\Desktop\STDMS_3.1\data\BatteryTemperature.csv'"
    )
    assert _is_create_tab_intent("Create new tab")
    assert _is_create_tab_intent("new tab")
    assert not _is_create_tab_intent("create sop-13 procedure")
    assert _is_best_model_intent("Which model is best for Battery Temperature Monitoring tab?")


def test_dedupe_tool_calls_keeps_one_create_tab():
    calls = [
        {"name": "propose_create_tab", "arguments": {"title": "A", "data_folder": "data"}},
        {"name": "propose_create_tab", "arguments": {"title": "A", "data_folder": "data"}},
        {"name": "propose_create_tab", "arguments": {"title": "A", "data_folder": "data"}},
        {"name": "inspect_data_folder", "arguments": {"data_folder": "data"}},
    ]
    out = _dedupe_tool_calls(calls, already_ran=set())
    names = [c["name"] for c in out]
    assert names.count("propose_create_tab") == 1
    assert "inspect_data_folder" in names


def test_dedupe_all_tools_once_per_turn():
    calls = [
        {"name": "list_watchlist", "arguments": {}},
        {"name": "list_watchlist", "arguments": {}},
        {"name": "list_watchlist", "arguments": {}},
        {"name": "list_watchlist", "arguments": {}},
        {"name": "list_watchlist", "arguments": {}},
        {"name": "compare_tab_models", "arguments": {"tab_title": "Battery"}},
        {"name": "compare_tab_models", "arguments": {"tab_title": "Battery"}},
    ]
    out = _dedupe_tool_calls(calls, already_ran=set())
    names = [c["name"] for c in out]
    assert names == ["list_watchlist", "compare_tab_models"]

    # Across rounds: already ran list_watchlist → skip it, keep compare
    out2 = _dedupe_tool_calls(calls, already_ran={"list_watchlist"})
    assert [c["name"] for c in out2] == ["compare_tab_models"]


def test_informational_ask_answers_eclipse_tab_not_draft_nag():
    class _SnapTab:
        def __init__(self, title):
            self.config = {"title": title}
            self.last_snapshot = {
                "title": title,
                "health_state": "Nominal",
                "monitoring_active": False,
                "trained_models": 2,
            }

        def get_snapshot(self):
            return dict(self.last_snapshot)

    class _Win:
        def __init__(self):
            self.custom_tabs = {
                "e1": _SnapTab("Eclipse monitoring"),
                "b1": _SnapTab("Battery Health Monitoring"),
            }
            self.model_registry = None

    host = MainWindowToolHost(_Win())
    msg = "which tab created to monitor to the eclipse operation"
    assert _is_informational_ask(msg) is True
    assert _is_create_tab_intent(msg) is False
    reply = synthesize_fleet_answer(msg, host)
    assert "Eclipse monitoring" in reply
    assert "Pending Agent Drafts" not in reply
    assert "Approve" not in reply or "No action needed" in reply


def test_propose_create_tab_pending_dedupe(csv_file: Path, host):
    h, registry = host
    first = propose_create_tab(
        proposed_message="Create once",
        title="Battery Temperature Monitoring",
        data_folder=str(csv_file),
        auto_suggest=True,
        host=h,
    )
    assert first.get("ok")
    second = propose_create_tab(
        proposed_message="Create again",
        title="Battery Temperature Monitoring",
        data_folder=str(csv_file.parent),
        auto_suggest=True,
        host=h,
    )
    assert second.get("ok")
    assert second.get("deduped") is True
    assert second.get("draft_id") == first.get("draft_id")
    drafts = registry.get_pending_draft_alerts(kind="create_tab")
    assert len(drafts) == 1


def test_deterministic_create_tab_shows_data_conclusions(csv_file: Path, host):
    h, registry = host
    msg = f"Create a tab from '{csv_file}'"
    out = _deterministic_create_tab(msg, h)
    assert out and out.get("ok")
    reply = out["reply"]
    assert "Observation" in reply
    assert "Battery_Temp" in reply or "numeric" in reply.lower() or "features=" in reply
    assert "Analysis" in reply
    assert "Recommendation" in reply
    tools = [t["tool"] for t in out["tool_trace"]]
    assert "inspect_data_folder" in tools
    assert "suggest_tab_config" in tools
    assert "propose_create_tab" in tools
    drafts = registry.get_pending_draft_alerts(kind="create_tab")
    assert len(drafts) == 1
    cfg = (drafts[0].get("proposed_payload") or {}).get("config") or {}
    assert cfg.get("selected_features")
    assert cfg.get("models")
    assert "Battery" in str(cfg.get("title") or "")
    assert cfg.get("source_file")


def test_run_fc_create_tab_offline_uses_tools(csv_file: Path, host):
    h, _registry = host
    called = {"chat": 0}

    def chat_fn(*_a, **_k):
        called["chat"] += 1
        return None

    out = run_function_calling(
        f"Create a tab from '{csv_file}'",
        host=h,
        chat_fn=chat_fn,
        reachable_fn=lambda *_: False,
    )
    assert out.get("ok")
    assert called["chat"] == 0
    assert out.get("agent_mode") == "tools"
    assert sum(1 for t in out["tool_trace"] if t["tool"] == "propose_create_tab") == 1


def test_run_fc_create_tab_llm_narrates_after_data_draft(csv_file: Path, host):
    h, registry = host

    def chat_fn(*_a, **_k):
        raise AssertionError("chat should not invent create_tab tools")

    def generate_fn(prompt, **_k):
        assert "Observation" in prompt or "create_tab" in prompt.lower()
        return "[Observation] Data-driven draft ready.\n[Analysis] Features chosen from CSV.\n[Recommendation] Approve once."

    out = run_function_calling(
        f"Create a tab from '{csv_file}'",
        host=h,
        chat_fn=chat_fn,
        generate_fn=generate_fn,
        reachable_fn=lambda *_: True,
    )
    assert out.get("ok")
    assert out.get("llm_used") is True
    assert out.get("agent_mode") == "llm"
    assert "Approve" in (out.get("reply") or "")
    create_calls = [t for t in out["tool_trace"] if t["tool"] == "propose_create_tab"]
    assert len(create_calls) == 1
    cfg = (registry.get_pending_draft_alerts(kind="create_tab")[0].get("proposed_payload") or {}).get(
        "config"
    ) or {}
    assert "Battery" in str(cfg.get("title") or "")
