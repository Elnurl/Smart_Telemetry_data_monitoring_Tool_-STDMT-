"""Intent routing must understand the situation — not command verbs."""

from __future__ import annotations

from collections import deque

from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.function_calling import run_function_calling
from app.agent.nodes import heuristic_route
from app.agent.ops_intent import (
    OpsIntent,
    classify_ops_intent,
    fulfill_intent,
    resolve_tab,
    sanitize_tool_args,
    strip_node_traces,
)


class _FakeTab:
    def __init__(self, tab_id, title, *, mode="nominal", health="Healthy", alerts=0):
        self.tab_id = tab_id
        self.config = {"title": title, "data_folder": r"C:\data\sda_feed"}
        self.last_snapshot = {
            "tab_id": tab_id,
            "title": title,
            "health_state": health,
            "drift": True,
            "fusion_score": 0.8,
            "obs_ok": True,
            "alert_count": alerts,
            "trained_models": 4,
            "mission_mode": mode,
            "monitoring_active": True,
            "data_folder": r"C:\data\sda_feed",
        }
        self.anomaly_events = deque(
            [
                {"timestamp": "2026-09-12 12:00:00", "health_state": "Warning", "message": "spike"},
                {"timestamp": "2026-09-12 12:01:00", "health_state": "Warning", "message": "drift"},
            ],
            maxlen=50,
        )
        self.last_drift_result = {"drift": True, "score": 0.4}
        self.score_history = [0.2, 0.3, 0.8]

    def get_snapshot(self):
        return dict(self.last_snapshot)


class _FakeRegistry:
    def get_pending_retrain_signals(self):
        return [
            {"id": 1, "tab_id": "t-test", "feature": "eclipse", "drift_score": 0.7, "acknowledged": 0},
            {"id": 2, "tab_id": "t-test", "feature": "sun_factor", "drift_score": 0.6, "acknowledged": 0},
        ]


class _FakeWindow:
    def __init__(self):
        self.custom_tabs = {
            "t-test": _FakeTab("t-test", "test", mode="eclipse", health="Healthy", alerts=18),
            "t-ecl": _FakeTab("t-ecl", "Eclipse", mode="eclipse", health="Warning", alerts=3),
        }
        self.model_registry = _FakeRegistry()


def _host():
    h = MainWindowToolHost(_FakeWindow())
    set_tool_host(h)
    return h


def test_intent_without_command_verbs():
    """No göstər / yoxla / neçəsi var — still pick the live situation."""
    assert classify_ops_intent("test tabında vəziyyət necədir") == OpsIntent.TAB_DETAIL
    assert classify_ops_intent("test sağlamdır amma 500 anomaliya") == OpsIntent.EXPLAIN_LIVE
    assert classify_ops_intent("test tabının son hadisələri") == OpsIntent.TAB_HISTORY
    assert classify_ops_intent("test tabında hansı modellər train olunub") == OpsIntent.TAB_MODELS
    assert classify_ops_intent("gözləyən retrain siqnalları") == OpsIntent.PENDING_RETRAIN
    assert classify_ops_intent("test üçün risk və TTF") == OpsIntent.FORECAST
    assert classify_ops_intent(r"C:\Users\x\sda_feed qovluğu") == OpsIntent.INSPECT_FOLDER
    assert classify_ops_intent(
        r"okay use these data folder just see and reconfigure models for this data type --> C:\Users\Elnur\Desktop\Files\Data-genrator\sda_feed"
    ) == OpsIntent.INSPECT_FOLDER
    assert classify_ops_intent(
        "please reconfigure the models for this data type, make sure they are retrained"
    ) == OpsIntent.INSPECT_FOLDER
    assert classify_ops_intent(
        "timestamp,satellite_id,mission_mode,eclipse,sun_factor,battery_voltage,solar_current"
    ) == OpsIntent.INSPECT_FOLDER
    assert classify_ops_intent("eclipse üçün SOP yazmağı təklif et") == OpsIntent.WRITE_SOP
    assert classify_ops_intent("what is the eclipse procedure?") == OpsIntent.KNOWLEDGE
    assert classify_ops_intent("salam") == OpsIntent.CHAT
    assert classify_ops_intent("who are you") == OpsIntent.CHAT
    assert classify_ops_intent("pending retrain signals") == OpsIntent.PENDING_RETRAIN
    assert classify_ops_intent("hansı tablar açıqdır") == OpsIntent.FLEET


def test_heuristic_routes_live_why_to_tools():
    assert heuristic_route("Why is the test tab Healthy with 500 anomalies?") == "tool"
    assert heuristic_route("what is the eclipse procedure?") == "knowledge"
    assert heuristic_route("sop yaz") == "tool"


def test_resolve_tab_uses_title_not_mission_mode():
    host = _host()
    try:
        hit = resolve_tab("test tabında fusion və drift", host)
        assert hit is not None
        assert hit.title == "test"
        assert hit.tab_id == "t-test"
        # Asking about the Eclipse-titled tab still works
        ecl = resolve_tab("Eclipse tab warnings", host)
        assert ecl is not None
        assert ecl.title == "Eclipse"
        # Bare mission-mode word must not steal the named tab
        steal = sanitize_tool_args(
            "list_tab_models",
            {"tab_id": "eclipse"},
            message="test tabında hansı modellər",
            host=host,
        )
        assert steal["tab_id"] == "t-test"
        assert steal["tab_title"] == "test"
    finally:
        set_tool_host(None)


def test_fulfill_history_and_retrain_and_fleet():
    host = _host()
    try:
        hist = fulfill_intent("test tabının son 2 hadisəsi", host)
        assert hist["ok"]
        assert any(t["tool"] == "get_tab_history" for t in hist["tool_trace"])
        assert "spike" in hist["reply"] or "Warning" in hist["reply"]

        retr = fulfill_intent("pending retrain signals", host)
        assert any(t["tool"] == "check_pending_retrain_signals" for t in retr["tool_trace"])
        assert "2" in retr["reply"]

        fleet = fulfill_intent("hansı tablar var", host)
        assert "test" in fleet["reply"]
        assert "Eclipse" in fleet["reply"]

        inspect = fulfill_intent(
            r"okay use this data folder and reconfigure models --> C:\Users\x\sda_feed",
            host,
        )
        assert inspect["intent"] == OpsIntent.INSPECT_FOLDER.value
        assert any(t["tool"] == "inspect_data_folder" for t in inspect["tool_trace"])
        assert "satellite telemetry" in inspect["reply"].lower() or "inspect" in inspect["reply"].lower()
    finally:
        set_tool_host(None)


def test_function_calling_runs_tools_when_llm_only_promises(host_unused=None):
    host = _host()
    try:
        def fake_chat(messages, tools=None, **kwargs):
            return {
                "role": "assistant",
                "content": "I will call get_tab_history for the last events.",
                "tool_calls": [],
            }

        result = run_function_calling(
            "test tabının son hadisələri",
            host=host,
            chat_fn=fake_chat,
            generate_fn=lambda *a, **k: None,
            reachable_fn=lambda *a, **k: True,
        )
        assert result["ok"] is True
        assert any(t.get("tool") == "get_tab_history" for t in result["tool_trace"])
        assert "I will call" not in (result["reply"] or "")
        assert "test" in (result["reply"] or "").lower() or "Warning" in (result["reply"] or "")
    finally:
        set_tool_host(None)


def test_strip_node_traces():
    text = "[R-LLM] routing → tool\n[M-LLM] quiet\n[RA-LLM] tools used: (none)\nFleet is quiet."
    assert strip_node_traces(text) == "Fleet is quiet."
