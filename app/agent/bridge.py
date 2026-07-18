"""Bridge between SecureAnomalyDetectionTool and the agent HTTP API.

Phase 3: read tools + propose drafts + GUI-queued monitor refresh.
Direct train / email / config apply remain blocked until human approval
(Approve on Pending Agent Drafts runs train / create_tab hooks).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger("STDMS.Agent.Bridge")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765

# Process-wide handle so the FastAPI app can reach the live tool window.
_host_lock = threading.RLock()
_tool_host: Optional["ToolHost"] = None
_server_handle: Any = None


@dataclass
class AgentBridgeContext:
    """Handle returned by attach_agent_bridge for loop / Dashboard wiring."""

    tool_host: Any
    audit: Any
    runner: Any
    server: Any
    allow_llm: bool = False
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT

    @property
    def tool_host_getter(self):
        return get_tool_host


class ToolHost(Protocol):
    """Surface the agent needs from the PyQt tool."""

    def list_tabs(self) -> list[dict[str, Any]]:
        ...

    def get_snapshot(self, tab_id: str) -> Optional[dict[str, Any]]:
        ...

    def get_history(self, tab_id: str, n: int = 50) -> list[dict[str, Any]]:
        ...

    def get_pending_retrain_signals(self) -> list[dict[str, Any]]:
        ...

    def get_tab_analysis(self, tab_id: str) -> dict[str, Any]:
        ...

    def queue_monitor_cycle(self, tab_id: str) -> dict[str, Any]:
        ...

    def propose_retrain(
        self,
        tab_id: str,
        *,
        reasoning: str = "",
        drift_score: float = 0.0,
        drifted_features: Optional[list] = None,
    ) -> dict[str, Any]:
        ...

    def propose_alert(
        self,
        tab_id: Optional[str],
        *,
        proposed_message: str,
        agent_reasoning: str = "",
        severity: str = "WARNING",
        kind: str = "alert",
        proposed_payload: Optional[dict] = None,
    ) -> dict[str, Any]:
        ...

    def list_pending_drafts(self) -> list[dict[str, Any]]:
        ...

    def resolve_draft(self, draft_id: int, status: str, *, actioned_by: Optional[str] = None) -> dict[str, Any]:
        ...

    def list_tab_models(self, tab_id: str) -> dict[str, Any]:
        ...

    def get_model_metrics(self, tab_id: str, model_id: str) -> dict[str, Any]:
        ...


class MainWindowToolHost:
    """Adapter over SecureAnomalyDetectionTool.custom_tabs."""

    def __init__(self, main_window: Any):
        self._window = main_window
        self._lock = threading.RLock()

    def list_tabs(self) -> list[dict[str, Any]]:
        with self._lock:
            tabs = getattr(self._window, "custom_tabs", None) or {}
            result: list[dict[str, Any]] = []
            for tab_id, widget in list(tabs.items()):
                snap = self._safe_snapshot(widget, tab_id)
                result.append(
                    {
                        "tab_id": tab_id,
                        "title": snap.get("title") or self._tab_title(widget, tab_id),
                        "snapshot": snap,
                    }
                )
            return result

    def get_snapshot(self, tab_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            widget = self._get_widget(tab_id)
            if widget is None:
                return None
            return self._safe_snapshot(widget, tab_id)

    def get_history(self, tab_id: str, n: int = 50) -> list[dict[str, Any]]:
        n = max(1, min(int(n), 500))
        with self._lock:
            widget = self._get_widget(tab_id)
            if widget is None:
                return []
            events = getattr(widget, "anomaly_events", None)
            if events is None:
                return []
            try:
                items = list(events)[:n]
            except Exception:
                return []
            history: list[dict[str, Any]] = []
            for item in items:
                if isinstance(item, dict):
                    history.append(dict(item))
                else:
                    history.append({"value": str(item)})
            return history

    def get_pending_retrain_signals(self) -> list[dict[str, Any]]:
        with self._lock:
            registry = getattr(self._window, "model_registry", None)
            if registry is None or not hasattr(registry, "get_pending_retrain_signals"):
                return []
            try:
                signals = registry.get_pending_retrain_signals()
                return [dict(s) for s in signals] if signals else []
            except Exception as exc:
                logger.warning("Failed to read pending retrain signals: %s", exc)
                return []

    def get_tab_analysis(self, tab_id: str) -> dict[str, Any]:
        """Read last cycle anomaly/drift/forecast state (no Qt mutations)."""
        with self._lock:
            widget = self._get_widget(tab_id)
            if widget is None:
                return {"status": "tab_not_found", "tab_id": tab_id}
            snap = self._safe_snapshot(widget, tab_id)
            drift_raw = getattr(widget, "last_drift_result", None) or {}
            drift = dict(drift_raw) if isinstance(drift_raw, dict) else {}
            try:
                events = list(getattr(widget, "anomaly_events", None) or [])[:15]
            except Exception:
                events = []
            forecast = None
            score_history = getattr(widget, "score_history", None)
            if score_history is not None and hasattr(widget, "_run_short_forecast"):
                try:
                    forecast = widget._run_short_forecast(list(score_history))
                except Exception:
                    forecast = None
            has_cycle = bool(
                snap.get("updated_at")
                or snap.get("health_state") not in (None, "", "Unavailable", "Idle")
                or drift
                or events
            )
            return {
                "status": "ok" if has_cycle else "no_recent_cycle",
                "tab_id": tab_id,
                "title": snap.get("title") or self._tab_title(widget, tab_id),
                "snapshot": snap,
                "anomaly": {
                    "health_state": snap.get("health_state"),
                    "fusion_score": snap.get("fusion_score"),
                    "obs_ok": snap.get("obs_ok"),
                    "obs_violations": snap.get("obs_violations"),
                    "alert_count": snap.get("alert_count", 0),
                },
                "drift": drift,
                "forecast": forecast,
                "recent_events": [dict(e) if isinstance(e, dict) else {"value": str(e)} for e in events],
            }

    def queue_monitor_cycle(self, tab_id: str) -> dict[str, Any]:
        """Queue widget.monitor_data on the Qt GUI thread; do not wait."""
        with self._lock:
            widget = self._get_widget(tab_id)
            if widget is None:
                return {"ok": False, "queued": False, "status": "tab_not_found", "tab_id": tab_id}
            if not hasattr(widget, "monitor_data"):
                return {"ok": False, "queued": False, "status": "unsupported", "tab_id": tab_id}
            target = widget
        try:
            from PyQt5.QtCore import QTimer

            QTimer.singleShot(0, target.monitor_data)
            return {"ok": True, "queued": True, "status": "queued", "tab_id": tab_id}
        except Exception as exc:
            logger.warning("queue_monitor_cycle failed for %s: %s", tab_id, exc)
            return {"ok": False, "queued": False, "status": "error", "tab_id": tab_id, "error": str(exc)}

    def propose_retrain(
        self,
        tab_id: str,
        *,
        reasoning: str = "",
        drift_score: float = 0.0,
        drifted_features: Optional[list] = None,
    ) -> dict[str, Any]:
        with self._lock:
            registry = getattr(self._window, "model_registry", None)
            if registry is None or not hasattr(registry, "mark_retrain_needed"):
                return {"ok": False, "error": "registry_unavailable"}
            widget = self._get_widget(tab_id)
            title = self._tab_title(widget, tab_id) if widget is not None else tab_id
            try:
                signal_id = registry.mark_retrain_needed(
                    drift_score=float(drift_score or 0.0),
                    reason=reasoning or "agent_propose_retrain",
                    drifted_features=list(drifted_features or []),
                    tab_id=tab_id,
                    metadata={"tab_title": title, "source": "agent"},
                )
                return {
                    "ok": True,
                    "status": "pending",
                    "signal_id": signal_id,
                    "tab_id": tab_id,
                    "requires_human_approval": True,
                }
            except Exception as exc:
                logger.warning("propose_retrain failed: %s", exc)
                return {"ok": False, "error": str(exc), "tab_id": tab_id}

    def propose_alert(
        self,
        tab_id: Optional[str],
        *,
        proposed_message: str,
        agent_reasoning: str = "",
        severity: str = "WARNING",
        kind: str = "alert",
        proposed_payload: Optional[dict] = None,
    ) -> dict[str, Any]:
        with self._lock:
            registry = getattr(self._window, "model_registry", None)
            if registry is None or not hasattr(registry, "create_draft_alert"):
                return {"ok": False, "error": "registry_unavailable"}
            try:
                draft_id = registry.create_draft_alert(
                    tab_id=tab_id,
                    kind=kind or "alert",
                    agent_reasoning=agent_reasoning or "",
                    proposed_message=proposed_message or "",
                    proposed_payload=proposed_payload or {},
                    severity=severity or "WARNING",
                )
                return {
                    "ok": True,
                    "status": "pending",
                    "draft_id": draft_id,
                    "tab_id": tab_id,
                    "kind": kind or "alert",
                    "requires_human_approval": True,
                }
            except Exception as exc:
                logger.warning("propose_alert failed: %s", exc)
                return {"ok": False, "error": str(exc), "tab_id": tab_id}

    def list_pending_drafts(self) -> list[dict[str, Any]]:
        with self._lock:
            registry = getattr(self._window, "model_registry", None)
            if registry is None or not hasattr(registry, "get_pending_draft_alerts"):
                return []
            try:
                return [dict(d) for d in registry.get_pending_draft_alerts()]
            except Exception as exc:
                logger.warning("list_pending_drafts failed: %s", exc)
                return []

    def resolve_draft(
        self, draft_id: int, status: str, *, actioned_by: Optional[str] = None
    ) -> dict[str, Any]:
        with self._lock:
            registry = getattr(self._window, "model_registry", None)
            if registry is None or not hasattr(registry, "resolve_draft_alert"):
                return {"ok": False, "error": "registry_unavailable"}
            try:
                updated = registry.resolve_draft_alert(
                    int(draft_id), status, actioned_by=actioned_by
                )
                return {"ok": bool(updated), "draft_id": int(draft_id), "status": status}
            except Exception as exc:
                return {"ok": False, "error": str(exc), "draft_id": int(draft_id)}

    @staticmethod
    def _serialize_metric_value(value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(k): MainWindowToolHost._serialize_metric_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MainWindowToolHost._serialize_metric_value(v) for v in value]
        try:
            return float(value)
        except Exception:
            return str(value)

    def list_tab_models(self, tab_id: str) -> dict[str, Any]:
        """List configured models on a tab with trained status and metric keys."""
        with self._lock:
            widget = self._get_widget(tab_id)
            if widget is None:
                return {"ok": False, "status": "tab_not_found", "tab_id": tab_id, "models": []}
            config = getattr(widget, "config", None) or {}
            models_cfg = list(config.get("models") or [])
            live = getattr(widget, "models", None) or {}
            selected = getattr(widget, "selected_model_id", None)
            rows: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for cfg in models_cfg:
                if not isinstance(cfg, dict):
                    continue
                mid = str(cfg.get("model_id") or "").strip()
                if not mid:
                    continue
                seen_ids.add(mid)
                obj = live.get(mid)
                trained = obj is not None
                metrics = {}
                if trained and hasattr(obj, "metrics") and isinstance(getattr(obj, "metrics", None), dict):
                    metrics = {
                        str(k): self._serialize_metric_value(v)
                        for k, v in obj.metrics.items()
                    }
                rows.append(
                    {
                        "model_id": mid,
                        "model_type": cfg.get("model_type") or getattr(obj, "model_type", None) or "Unknown",
                        "trained": trained,
                        "selected": mid == selected,
                        "parameters": cfg.get("model_parameters") or {},
                        "metrics": metrics,
                        "metric_names": list(metrics.keys()),
                    }
                )
            # Include live models missing from config (edge case)
            for mid, obj in live.items():
                mid_s = str(mid)
                if mid_s in seen_ids:
                    continue
                metrics = {}
                if obj is not None and hasattr(obj, "metrics") and isinstance(obj.metrics, dict):
                    metrics = {
                        str(k): self._serialize_metric_value(v) for k, v in obj.metrics.items()
                    }
                rows.append(
                    {
                        "model_id": mid_s,
                        "model_type": getattr(obj, "model_type", None) or "Unknown",
                        "trained": obj is not None,
                        "selected": mid_s == selected,
                        "parameters": {},
                        "metrics": metrics,
                        "metric_names": list(metrics.keys()),
                    }
                )
            trained_n = sum(1 for r in rows if r.get("trained"))
            return {
                "ok": True,
                "status": "ok",
                "tab_id": tab_id,
                "title": self._tab_title(widget, tab_id),
                "selected_model_id": selected,
                "model_count": len(rows),
                "trained_count": trained_n,
                "models": rows,
            }

    def get_model_metrics(self, tab_id: str, model_id: str) -> dict[str, Any]:
        """Return detailed metrics for one model on a tab."""
        with self._lock:
            listing = self.list_tab_models(tab_id)
            if not listing.get("ok"):
                return listing
            mid = str(model_id or "").strip()
            match = next((m for m in listing.get("models") or [] if m.get("model_id") == mid), None)
            if match is None:
                return {
                    "ok": False,
                    "status": "model_not_found",
                    "tab_id": tab_id,
                    "model_id": mid,
                    "available_model_ids": [m.get("model_id") for m in listing.get("models") or []],
                }
            return {
                "ok": True,
                "status": "ok" if match.get("trained") else "not_trained",
                "tab_id": tab_id,
                "title": listing.get("title"),
                "model_id": mid,
                "model_type": match.get("model_type"),
                "trained": bool(match.get("trained")),
                "selected": bool(match.get("selected")),
                "parameters": match.get("parameters") or {},
                "metrics": match.get("metrics") or {},
                "note": None
                if match.get("trained")
                else "Model is configured but not trained yet.",
            }

    def _get_widget(self, tab_id: str) -> Any:
        tabs = getattr(self._window, "custom_tabs", None) or {}
        return tabs.get(tab_id)

    @staticmethod
    def _tab_title(widget: Any, tab_id: str) -> str:
        config = getattr(widget, "config", None) or {}
        return str(config.get("title") or tab_id)

    @staticmethod
    def _safe_snapshot(widget: Any, tab_id: str) -> dict[str, Any]:
        try:
            if hasattr(widget, "get_snapshot"):
                snap = widget.get_snapshot()
                if isinstance(snap, dict):
                    out = dict(snap)
                    out.setdefault("tab_id", tab_id)
                    return out
            last = getattr(widget, "last_snapshot", None)
            if isinstance(last, dict):
                out = dict(last)
                out.setdefault("tab_id", tab_id)
                return out
        except Exception as exc:
            logger.warning("Snapshot read failed for tab %s: %s", tab_id, exc)
        return {"tab_id": tab_id, "health_state": "Unavailable", "error": "snapshot_unavailable"}


def get_tool_host() -> Optional[ToolHost]:
    with _host_lock:
        return _tool_host


def set_tool_host(host: Optional[ToolHost]) -> None:
    global _tool_host
    with _host_lock:
        _tool_host = host


def attach_agent_bridge(
    main_window: Any,
    *,
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
    enabled: bool = True,
    audit_db_path: Optional[str] = None,
    allow_llm: bool = False,
) -> Optional[Any]:
    """Register the live tool window and start the local agent HTTP server.

    Air-gap defaults: loopback bind only, no cloud egress.
    Local Ollama (loopback) is opt-in via allow_llm=True.
    Safe to call when FastAPI/uvicorn are missing — logs a warning and returns None.
    """
    global _server_handle

    if not enabled:
        logger.info("Instrumentation agent bridge disabled")
        return None

    from app.agent.audit import AgentAuditLog
    from app.agent.policy import assert_loopback_bind
    from app.agent.runner import AgentRunner
    from app.agent.server import start_agent_server

    try:
        host = assert_loopback_bind(host)
    except ValueError as exc:
        logger.error("%s", exc)
        return None

    tool_host = MainWindowToolHost(main_window)
    set_tool_host(tool_host)

    if audit_db_path is None:
        data_dir = getattr(main_window, "model_registry", None)
        data_dir = getattr(data_dir, "data_dir", "data") if data_dir is not None else "data"
        from pathlib import Path

        audit_db_path = str(Path(data_dir) / "agent_decisions.db")

    audit = AgentAuditLog(audit_db_path)
    runner = AgentRunner(audit=audit, tool_host_getter=get_tool_host, allow_llm=allow_llm)

    try:
        _server_handle = start_agent_server(
            host=host,
            port=port,
            audit=audit,
            runner=runner,
        )
    except Exception as exc:
        logger.warning("Agent bridge failed to start: %s", exc)
        _server_handle = None
        return None

    context = AgentBridgeContext(
        tool_host=tool_host,
        audit=audit,
        runner=runner,
        server=_server_handle,
        allow_llm=allow_llm,
        host=host,
        port=port,
    )

    main_window._agent_bridge_host = tool_host
    main_window._agent_audit = audit
    main_window._agent_runner = runner
    main_window._agent_server = _server_handle
    main_window.agent_bridge = context
    logger.info(
        "Instrumentation agent (air-gap) listening on http://%s:%s — LLM=%s",
        host,
        port,
        allow_llm,
    )
    return context


def stop_agent_bridge() -> None:
    global _server_handle
    handle = _server_handle
    _server_handle = None
    set_tool_host(None)
    if handle is None:
        return
    stop = getattr(handle, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception as exc:
            logger.warning("Agent bridge stop failed: %s", exc)
