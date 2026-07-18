"""Propose train / create_tab drafts and Approve execute helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.tools import invoke_tool, propose_create_tab, propose_train, to_ollama_tools
from app.models.registry import ModelRegistry


class _FakeModel:
    def __init__(self, model_type="Random Forest", metrics=None):
        self.model_type = model_type
        self.metrics = metrics or {"precision": 0.91, "recall": 0.88}


class _FakeTrainTab:
    def __init__(self, tab_id="t1"):
        self.tab_id = tab_id
        self.config = {
            "title": "CH1",
            "models": [
                {
                    "model_id": "m1",
                    "model_type": "Random Forest",
                    "model_parameters": {"n_estimators": 100},
                },
                {"model_id": "m2", "model_type": "Isolation Forest", "model_parameters": {}},
            ],
        }
        self.models = {"m1": _FakeModel(), "m2": None}
        self.selected_model_id = "m1"
        self.train_calls = []
        self.remove_calls = []

    def train_model(self, model_id=None, silent=False, preloaded_data=None, _sync=False):
        self.train_calls.append({"model_id": model_id, "silent": silent})
        return True

    def remove_model(self, model_id, silent=False):
        self.remove_calls.append({"model_id": model_id, "silent": silent})
        if model_id in self.models:
            del self.models[model_id]
        self.config["models"] = [
            m for m in self.config.get("models", []) if m.get("model_id") != model_id
        ]
        return True


class _FakeWindow:
    def __init__(self, registry):
        self.custom_tabs = {"t1": _FakeTrainTab("t1")}
        self.model_registry = registry


@pytest.fixture
def registry(tmp_path: Path):
    return ModelRegistry(db_path=str(tmp_path / "approve.sqlite"))


@pytest.fixture
def host(registry):
    h = MainWindowToolHost(_FakeWindow(registry))
    set_tool_host(h)
    yield h
    set_tool_host(None)


def test_propose_train_creates_train_draft(host, registry):
    result = propose_train(
        "t1",
        proposed_message="Retrain after drift",
        agent_reasoning="drift_score high",
        model_id="m1",
        host=host,
    )
    assert result["ok"] is True
    assert result["kind"] == "train"
    drafts = registry.get_pending_draft_alerts(kind="train")
    assert len(drafts) == 1
    assert drafts[0]["proposed_payload"]["model_id"] == "m1"


def test_propose_create_tab_creates_draft(host, registry, tmp_path: Path):
    bus = tmp_path / "bus"
    bus.mkdir()
    (bus / "bus.csv").write_text("timestamp,Bus_Voltage\n2026-01-01,28.1\n", encoding="utf-8")
    result = propose_create_tab(
        proposed_message="Add bus voltage tab",
        title="Bus Voltage",
        data_folder=str(bus),
        agent_reasoning="operator request SOP eclipse check",
        auto_suggest=False,
        host=host,
        config={
            "title": "Bus Voltage",
            "data_folder": str(bus),
            "selected_features": ["value"],
            "models": [{"model_type": "Isolation Forest", "model_parameters": {}}],
        },
    )
    assert result["ok"] is True
    assert result["kind"] == "create_tab"
    assert result["tab_id"] is None
    drafts = registry.get_pending_draft_alerts(kind="create_tab")
    assert len(drafts) == 1
    assert drafts[0]["proposed_payload"]["config"]["title"] == "Bus Voltage"
    assert drafts[0]["proposed_payload"]["config"]["data_folder"] == str(bus)


def test_propose_create_tab_rejects_missing_folder(host, registry):
    result = propose_create_tab(
        proposed_message="Bad path",
        title="Ghost",
        data_folder=r"C:\data\health_metrics",
        agent_reasoning="should fail",
        host=host,
    )
    assert result["ok"] is False
    assert result.get("error") == "data_folder_not_found"
    assert registry.get_pending_draft_alerts(kind="create_tab") == []


def test_invoke_propose_train_and_create_tab(host, registry, tmp_path: Path):
    out = invoke_tool(
        "propose_train",
        {"tab_id": "t1", "proposed_message": "Train now", "model_id": "m2"},
        host=host,
    )
    assert out["ok"] is True
    folder = tmp_path / "temp_tab"
    folder.mkdir()
    (folder / "t.csv").write_text("timestamp,Temperature\n2026-01-01,21.0\n", encoding="utf-8")
    out2 = invoke_tool(
        "propose_create_tab",
        {
            "proposed_message": "New tab",
            "title": "Temp",
            "data_folder": str(folder),
            "config": {
                "schedule_type": "Continuous",
                "selected_features": ["value"],
                "models": [{"model_type": "Z-Score", "model_parameters": {"threshold": 3.0}}],
            },
            "auto_suggest": False,
        },
        host=host,
    )
    assert out2["ok"] is True
    names = {t["function"]["name"] for t in to_ollama_tools()}
    assert "propose_train" in names
    assert "propose_create_tab" in names


def test_get_draft_alert_roundtrip(registry):
    draft_id = registry.create_draft_alert(
        tab_id="t1",
        kind="train",
        proposed_message="Train",
        proposed_payload={"model_id": "abc"},
        severity="INFO",
    )
    row = registry.get_draft_alert(draft_id)
    assert row is not None
    assert row["kind"] == "train"
    assert row["status"] == "pending"
    assert row["proposed_payload"]["model_id"] == "abc"
    assert registry.resolve_draft_alert(draft_id, "approved", actioned_by="op")
    row2 = registry.get_draft_alert(draft_id)
    assert row2["status"] == "approved"


def test_execute_approved_train_hook():
    """Mirror main-window train hook without Qt."""
    tab = _FakeTrainTab("t1")
    draft = {
        "kind": "train",
        "tab_id": "t1",
        "proposed_payload": {"model_id": "m1"},
    }
    widget = {"t1": tab}.get(draft["tab_id"])
    assert widget is not None
    payload = draft.get("proposed_payload") or {}
    ok = widget.train_model(model_id=payload.get("model_id"), silent=False)
    assert ok is True
    assert tab.train_calls == [{"model_id": "m1", "silent": False}]


def test_post_train_hook_runs_only_on_finalize_success():
    """start_monitoring enqueue must wait for finalize, not worker start."""
    calls = []

    class _Tab:
        def __init__(self):
            self.models = {}
            self._agent_post_train_hook = lambda: calls.append("start")

        def _finalize_train_result(self, success, *_a, **_k):
            if success:
                hook = getattr(self, "_agent_post_train_hook", None)
                if callable(hook):
                    self._agent_post_train_hook = None
                    hook()
                return True
            self._agent_post_train_hook = None
            return False

    tab = _Tab()
    # Simulating worker start must NOT fire the hook
    assert calls == []
    tab._finalize_train_result(True)
    assert calls == ["start"]
    # Second finalize without hook set does nothing
    tab._finalize_train_result(True)
    assert calls == ["start"]


def test_propose_start_monitoring_by_title(host, registry):
    from app.agent.tools import propose_start_monitoring

    result = propose_start_monitoring(
        proposed_message="Activate Weekly tab",
        tab_title="CH1",
        agent_reasoning="operator asked to activate",
        host=host,
    )
    assert result["ok"] is True
    assert result["kind"] == "start_monitoring"
    drafts = registry.get_pending_draft_alerts(kind="start_monitoring")
    assert len(drafts) == 1
    assert drafts[0]["tab_id"] == "t1"


def test_propose_start_monitoring_ambiguous(host):
    from app.agent.tools import propose_start_monitoring

    host._window.custom_tabs["t2"] = _FakeTrainTab("t2")
    host._window.custom_tabs["t2"].config = {"title": "CH1 Extra"}
    # "CH1" matches both CH1 and CH1 Extra via partial
    result = propose_start_monitoring(
        proposed_message="Start",
        tab_title="CH1",
        host=host,
    )
    # exact title CH1 should win over partial
    assert result["ok"] is True
    assert result["tab_id"] == "t1"


def test_propose_stop_monitoring(host, registry):
    from app.agent.tools import propose_stop_monitoring

    result = propose_stop_monitoring(
        proposed_message="Stop CH1",
        tab_id="t1",
        host=host,
    )
    assert result["ok"] is True
    assert result["kind"] == "stop_monitoring"
    assert registry.get_pending_draft_alerts(kind="stop_monitoring")


def test_list_tab_models_and_metrics(host):
    from app.agent.tools import get_model_metrics, list_tab_models

    listing = list_tab_models(tab_id="t1", host=host)
    assert listing["ok"] is True
    assert listing["trained_count"] == 1
    assert listing["model_count"] == 2
    m1 = next(m for m in listing["models"] if m["model_id"] == "m1")
    assert m1["trained"] is True
    assert m1["metrics"]["precision"] == 0.91

    detail = get_model_metrics(model_id="m1", tab_id="t1", host=host)
    assert detail["ok"] is True
    assert detail["metrics"]["recall"] == 0.88

    untrained = get_model_metrics(model_id="m2", tab_title="CH1", host=host)
    assert untrained["ok"] is True
    assert untrained["status"] == "not_trained"


def test_propose_remove_model(host, registry):
    from app.agent.tools import propose_remove_model

    result = propose_remove_model(
        model_id="m1",
        proposed_message="Remove weak RF",
        tab_id="t1",
        host=host,
    )
    assert result["ok"] is True
    assert result["kind"] == "remove_model"
    drafts = registry.get_pending_draft_alerts(kind="remove_model")
    assert len(drafts) == 1
    assert drafts[0]["proposed_payload"]["model_id"] == "m1"
