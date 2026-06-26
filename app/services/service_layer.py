from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class ServiceLayerContext:
    username: str
    auth_provider: str = "local"
    session_id: Optional[str] = None


class AppServiceLayer:
    """Thin service boundary between UI and business logic."""

    def __init__(
        self,
        user_manager: Any,
        alert_policy_manager: Any = None,
        audit_writer: Optional[Callable[..., None]] = None,
    ):
        self.user_manager = user_manager
        self.alert_policy_manager = alert_policy_manager
        self.audit_writer = audit_writer

    def authorize(self, username: str, permission: str, resource: str = "*") -> bool:
        return bool(self.user_manager.check_permission(username, permission, resource=resource))

    def process_alert(self, severity: str, message: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.alert_policy_manager is None:
            return {"routed": False, "reason": "alert_policy_not_initialized"}
        return self.alert_policy_manager.process_alert(severity, message, payload or {})

    def acknowledge_alert(self, incident_id: int, context: ServiceLayerContext) -> bool:
        if self.alert_policy_manager is None:
            return False
        return bool(self.alert_policy_manager.acknowledge(incident_id, context.username))

    def audit(
        self,
        context: ServiceLayerContext,
        action: str,
        resource: str = "*",
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.audit_writer is None:
            return
        self.audit_writer(
            actor=context.username,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {},
            auth_provider=context.auth_provider,
        )

