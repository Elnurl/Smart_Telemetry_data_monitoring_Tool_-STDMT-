"""Alerts domain (routing, cooldown, escalation)."""

from app.alerts.policy import AlertPolicyManager, load_alert_routing_config

__all__ = ["AlertPolicyManager", "load_alert_routing_config"]

