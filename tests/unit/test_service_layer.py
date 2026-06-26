from app.services.service_layer import AppServiceLayer, ServiceLayerContext


class _FakeUserManager:
    def __init__(self):
        self.calls = []

    def check_permission(self, username, permission, resource="*"):
        self.calls.append((username, permission, resource))
        return permission == "view_data"


class _FakeAlertPolicy:
    def process_alert(self, severity, message, payload):
        return {"routed": True, "severity": severity, "payload": payload}

    def acknowledge(self, incident_id, username):
        return incident_id == 1 and username == "alice"


def test_service_layer_authorize_delegates_to_user_manager():
    service = AppServiceLayer(user_manager=_FakeUserManager())
    assert service.authorize("alice", "view_data", "data:*") is True
    assert service.authorize("alice", "manage_users", "admin:*") is False


def test_service_layer_alert_and_acknowledge():
    service = AppServiceLayer(user_manager=_FakeUserManager(), alert_policy_manager=_FakeAlertPolicy())
    result = service.process_alert("warning", "test", {"x": 1})
    assert result["routed"] is True

    context = ServiceLayerContext(username="alice", auth_provider="local", session_id="s1")
    assert service.acknowledge_alert(1, context) is True
    assert service.acknowledge_alert(2, context) is False

