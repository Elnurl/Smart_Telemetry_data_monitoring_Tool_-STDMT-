"""Sprint 3 ToolHost /v1 JWT API — auth, tab CRUD, train trigger."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from app.agent.audit import AgentAuditLog
from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent.jwt_tokens import decode_access_token, issue_access_token, resolve_jwt_secret
from app.agent.runner import AgentRunner


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
            ],
            maxlen=500,
        )

    def get_snapshot(self):
        return dict(self.last_snapshot)

    def train_model(self, model_id=None, silent=False):
        self.trained_model_ids.append(model_id)
        return True

    def start_monitoring(self):
        self.monitoring_active = True

    def stop_monitoring(self):
        self.monitoring_active = False

    def load_latest_data(self, for_monitoring=False, force_reload=False):
        return getattr(self.data_processor, "data", None), None


class _FakeRegistry:
    def get_pending_retrain_signals(self):
        return []

    def get_recent_models(self, limit=5):
        return []


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
        self.user_manager = _FakeUserManager()
        self.current_username = "admin"

    def _create_custom_tab_from_config(self, config):
        tab_id = f"t{len(self.custom_tabs) + 1}"
        self.custom_tabs[tab_id] = _FakeTab(tab_id, config["title"])
        return tab_id


class _FakeUserManager:
    def authenticate(self, username, password):
        if username == "admin" and password == "correct-horse":
            return True, "admin", "Login successful.", False
        return False, None, "Invalid username or password.", False

    def _authenticate_local(self, username, password):
        return self.authenticate(username, password)


@pytest.fixture
def host():
    h = MainWindowToolHost(_FakeWindow())
    set_tool_host(h)
    yield h
    set_tool_host(None)


@pytest.fixture
def client(tmp_path, host):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.agent.server import create_app

    audit = AgentAuditLog(tmp_path / "agent_decisions.db")
    runner = AgentRunner(audit=audit, tool_host_getter=lambda: host)
    token = "test-token-with-at-least-24-characters"
    app = create_app(
        audit=audit,
        runner=runner,
        api_token=token,
        jwt_secret="unit-test-jwt-secret-key-which-is-long",
    )
    return TestClient(app), token


def _login(http):
    response = http.post(
        "/v1/auth/login",
        json={"username": "admin", "password": "correct-horse"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert body["token"]
    return body["token"]


def test_jwt_roundtrip():
    secret = resolve_jwt_secret("fixed-secret-for-unit-tests")
    token = issue_access_token(username="ops", role="analyst", secret=secret, ttl_seconds=120)
    payload = decode_access_token(token, secret)
    assert payload["sub"] == "ops"
    assert payload["role"] == "analyst"


def test_login_returns_token_and_me(client):
    http, _static = client
    token = _login(http)
    auth = {"Authorization": f"Bearer {token}"}
    me = http.get("/v1/auth/me", headers=auth)
    assert me.status_code == 200
    assert me.json()["username"] == "admin"
    assert me.json()["role"] == "admin"


def test_login_rejects_bad_password(client):
    http, _static = client
    denied = http.post(
        "/v1/auth/login",
        json={"username": "admin", "password": "wrong"},
    )
    assert denied.status_code == 401


def test_v1_tabs_requires_token_then_lists_snapshots(client):
    http, _static = client
    assert http.get("/v1/tabs").status_code == 401
    token = _login(http)
    listed = http.get("/v1/tabs", headers={"Authorization": f"Bearer {token}"})
    assert listed.status_code == 200
    tabs = listed.json()["tabs"]
    assert len(tabs) == 2
    assert tabs[0]["snapshot"]["health_state"] == "Nominal"


def test_tab_crud_start_and_train(client, tmp_path, host):
    http, _static = client
    token = _login(http)
    auth = {"Authorization": f"Bearer {token}"}
    data_folder = tmp_path / "telemetry"
    data_folder.mkdir()

    missing = http.post(
        "/v1/tabs",
        json={"name": "No Folder Tab"},
        headers=auth,
    )
    assert missing.status_code == 400

    created = http.post(
        "/v1/tabs",
        json={
            "name": "API Tab",
            "data_folder": str(data_folder),
            "models": [{"model_type": "Isolation Forest", "model_parameters": {}}],
            "schedule": {"type": "Continuous", "interval_ms": 300000},
        },
        headers=auth,
    )
    assert created.status_code == 202, created.text
    op = http.get(f"/operations/{created.json()['operation_id']}", headers=auth)
    assert op.status_code == 200
    assert op.json()["status"] == "completed"
    tab_id = op.json()["result"]["tab_id"]
    assert tab_id in host._window.custom_tabs

    started = http.post(f"/v1/tabs/{tab_id}/start", headers=auth)
    assert started.status_code == 202, started.text
    start_op = http.get(f"/operations/{started.json()['operation_id']}", headers=auth)
    assert start_op.json()["status"] == "completed"
    assert host._window.custom_tabs[tab_id].monitoring_active is True

    train = http.post(
        f"/v1/tabs/{tab_id}/models/train",
        json={"model_id": None},
        headers=auth,
    )
    assert train.status_code == 202, train.text
    body = train.json()
    assert body["status"] == "training_started"
    assert body["job_id"]
    train_op = http.get(f"/operations/{body['job_id']}", headers=auth)
    assert train_op.json()["status"] == "completed"

    models = http.get(f"/v1/tabs/{tab_id}/models", headers=auth)
    assert models.status_code == 200
    assert models.json()["models"]

    preview = http.get(f"/v1/tabs/{tab_id}/data/preview", headers=auth)
    assert preview.status_code == 200
    assert "columns" in preview.json()
    assert "row_count" in preview.json()
    assert len(preview.json()["rows"]) <= 20

    metrics = http.get(f"/v1/tabs/{tab_id}/metrics", headers=auth)
    assert metrics.status_code == 200
    assert metrics.json()["tab_id"] == tab_id

    deleted = http.delete(f"/v1/tabs/{tab_id}", headers=auth)
    assert deleted.status_code == 202


def test_logout_blacklists_token(client):
    http, _static = client
    token = _login(http)
    auth = {"Authorization": f"Bearer {token}"}
    assert http.post("/v1/auth/logout", headers=auth).status_code == 200
    assert http.get("/v1/tabs", headers=auth).status_code == 401


def test_path_traversal_rejected_on_data_load(client, tmp_path):
    http, _static = client
    token = _login(http)
    auth = {"Authorization": f"Bearer {token}"}
    outside = tmp_path / "secret.csv"
    outside.write_text("time,value\n1,2\n", encoding="utf-8")
    denied = http.post(
        "/v1/tabs/t1/data/load",
        json={"file_path": str(outside), "file_type": "csv"},
        headers=auth,
    )
    assert denied.status_code in {400, 409}


def test_legacy_tabs_health_and_agent_run_still_work(client):
    http, static = client
    auth = {"Authorization": f"Bearer {static}"}
    health = http.get("/health")
    assert health.status_code == 200
    assert health.json()["sprint"] == 3
    tabs = http.get("/tabs", headers=auth)
    assert tabs.status_code == 200
    assert len(tabs.json()["tabs"]) == 2
    run = http.post("/agent/run", json={"tab_id": "t1", "task": "summarize"}, headers=auth)
    assert run.status_code == 200
    assert run.json()["outcome"] == "observation_only"
    versioned = http.get("/v1/ops/tabs", headers=auth)
    assert versioned.status_code == 200
