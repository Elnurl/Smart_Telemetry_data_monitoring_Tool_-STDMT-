"""Bridge between SecureAnomalyDetectionTool and the agent HTTP API.

Phase 3: read tools + propose drafts + GUI-queued monitor refresh.
Direct train / email / config apply remain blocked until human approval
(Approve on Pending Agent Drafts runs train / create_tab hooks).
"""

from __future__ import annotations

import logging
import math
import threading
import uuid
import datetime
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol

logger = logging.getLogger("STDMS.Agent.Bridge")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_local_data_path(file_path: str, extra_root: str = "") -> Path:
    """Resolve a local file path and reject path-traversal / non-local targets."""
    raw = str(file_path or "").strip()
    if not raw or "\x00" in raw:
        raise ValueError("file_path is required")
    candidate = Path(raw).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError("file_path could not be resolved") from exc
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(raw)
    roots = [Path.cwd().resolve()]
    data_root = (Path.cwd() / "data").resolve()
    roots.append(data_root)
    extra = str(extra_root or "").strip()
    if extra:
        try:
            roots.append(Path(extra).expanduser().resolve())
        except (OSError, RuntimeError):
            pass
    if not any(_is_under(resolved, root) for root in roots):
        raise ValueError("file_path must stay under the workspace or the tab data folder")
    return resolved

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
    api_token: str = ""
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

    def get_tab_definition(self, tab_id: str) -> Optional[dict[str, Any]]:
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

    def create_tab(self, config: dict[str, Any], *, requested_by: str = "api") -> dict[str, Any]:
        ...

    def update_tab(
        self,
        tab_id: str,
        updates: dict[str, Any],
        *,
        requested_by: str = "api",
    ) -> dict[str, Any]:
        ...

    def delete_tab(self, tab_id: str, *, requested_by: str = "api") -> dict[str, Any]:
        ...

    def train_tab(
        self,
        tab_id: str,
        *,
        model_id: Optional[str] = None,
        requested_by: str = "api",
    ) -> dict[str, Any]:
        ...

    def get_operation(self, operation_id: str) -> Optional[dict[str, Any]]:
        ...

    def get_tab_data_schema(self, tab_id: str) -> dict[str, Any]:
        ...

    def get_tab_data_preview(self, tab_id: str, limit: int = 50, *, head: bool = False) -> dict[str, Any]:
        ...

    def start_tab(self, tab_id: str, *, requested_by: str = "api") -> dict[str, Any]:
        ...

    def stop_tab(self, tab_id: str, *, requested_by: str = "api") -> dict[str, Any]:
        ...

    def get_tab_pipeline_metrics(self, tab_id: str, limit: int = 50) -> dict[str, Any]:
        ...

    def load_tab_data(
        self,
        tab_id: str,
        *,
        file_path: str,
        file_type: str = "csv",
        requested_by: str = "api",
    ) -> dict[str, Any]:
        ...

    def get_registry_models(self, tab_id: str, limit: int = 20) -> dict[str, Any]:
        ...


class _QtOperationDispatcher:
    """Queue Python callables on the Qt GUI thread when Qt is running."""

    def __init__(self) -> None:
        self._qt_object: Any = None
        try:
            from PyQt5.QtCore import QCoreApplication, QObject, pyqtSignal

            if QCoreApplication.instance() is None:
                return

            class _Dispatcher(QObject):
                invoke = pyqtSignal(object)

                def __init__(self):
                    super().__init__()
                    self.invoke.connect(self._run)

                @staticmethod
                def _run(callback):
                    callback()

            self._qt_object = _Dispatcher()
        except Exception as exc:
            logger.debug("Qt operation dispatcher unavailable: %s", exc)

    def submit(self, callback) -> None:
        if self._qt_object is None:
            callback()
            return
        self._qt_object.invoke.emit(callback)


class MainWindowToolHost:
    """Adapter over SecureAnomalyDetectionTool.custom_tabs."""

    def __init__(self, main_window: Any):
        self._window = main_window
        self._lock = threading.RLock()
        self._operations: dict[str, dict[str, Any]] = {}
        self._dispatcher = _QtOperationDispatcher()

    @contextmanager
    def as_operator(self, username: str) -> Iterator[None]:
        """Temporarily apply the API caller's identity to STDMS RBAC checks."""
        window = self._window
        previous = getattr(window, "current_username", None)
        try:
            if username and username != "api":
                window.current_username = username
            yield
        finally:
            try:
                window.current_username = previous
            except Exception:
                pass

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

    def get_tab_definition(self, tab_id: str) -> Optional[dict[str, Any]]:
        """Return a client-safe tab config for future UI/gRPC clients."""
        with self._lock:
            widget = self._get_widget(tab_id)
            if widget is None:
                return None
            config = dict(getattr(widget, "config", None) or {})

        def _safe(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    str(key): (
                        "***"
                        if any(
                            marker in str(key).lower()
                            for marker in ("password", "secret", "token", "api_key")
                        )
                        else _safe(item)
                    )
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [_safe(item) for item in value]
            return self._serialize_metric_value(value)

        return {
            "tab_id": tab_id,
            "title": self._tab_title(widget, tab_id),
            "config": _safe(config),
            "snapshot": self._safe_snapshot(widget, tab_id),
        }

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
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        if isinstance(value, dict):
            return {str(k): MainWindowToolHost._serialize_metric_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MainWindowToolHost._serialize_metric_value(v) for v in value]
        try:
            return float(value)
        except Exception:
            return str(value)

    def _authorized(self, permission: str, resource: str) -> bool:
        """Use the signed-in STDMS operator's existing RBAC when available."""
        service = getattr(self._window, "service_layer", None)
        username = getattr(self._window, "current_username", None)
        if service is None or not hasattr(service, "authorize"):
            return True
        if not username:
            return False
        try:
            return bool(service.authorize(username, permission, resource=resource))
        except Exception as exc:
            logger.warning("ToolHost authorization failed for %s: %s", permission, exc)
            return False

    def _submit_operation(self, kind: str, callback, *, requested_by: str) -> dict[str, Any]:
        operation_id = str(uuid.uuid4())
        operation = {
            "operation_id": operation_id,
            "kind": kind,
            "status": "queued",
            "requested_by": str(requested_by or "api")[:100],
            "result": None,
            "error": None,
        }
        with self._lock:
            self._operations[operation_id] = operation

        def _run() -> None:
            with self._lock:
                operation["status"] = "running"
            try:
                result = callback()
                with self._lock:
                    operation["status"] = "completed"
                    operation["result"] = self._serialize_metric_value(result)
            except Exception as exc:
                logger.exception("ToolHost operation %s failed", operation_id)
                with self._lock:
                    operation["status"] = "failed"
                    operation["error"] = str(exc)[:1000]

        self._dispatcher.submit(_run)
        return {
            "ok": True,
            "accepted": True,
            "operation_id": operation_id,
            "kind": kind,
            "status": operation["status"],
        }

    def get_operation(self, operation_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            operation = self._operations.get(str(operation_id))
            return dict(operation) if operation is not None else None

    def create_tab(self, config: dict[str, Any], *, requested_by: str = "api") -> dict[str, Any]:
        if not self._authorized("create_tab", "*") and not self._authorized(
            "manage_users", "admin/users"
        ):
            return {"ok": False, "status": "forbidden", "error": "permission_denied"}
        if not isinstance(config, dict):
            return {"ok": False, "status": "invalid_config", "error": "config must be an object"}
        clean = dict(config)
        title = str(clean.get("title") or "").strip()
        if not title:
            return {"ok": False, "status": "invalid_config", "error": "title is required"}
        clean["title"] = title[:200]

        def _create():
            create = getattr(self._window, "_create_custom_tab_from_config", None)
            if not callable(create):
                raise RuntimeError("create_tab_unsupported")
            tab_id = create(clean)
            return {"tab_id": tab_id, "title": clean["title"]}

        return self._submit_operation("create_tab", _create, requested_by=requested_by)

    def update_tab(
        self,
        tab_id: str,
        updates: dict[str, Any],
        *,
        requested_by: str = "api",
    ) -> dict[str, Any]:
        widget = self._get_widget(tab_id)
        if widget is None:
            return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
        if not self._authorized("create_tab", f"tab:{tab_id}") and not self._authorized(
            "manage_users", "admin/users"
        ):
            return {"ok": False, "status": "forbidden", "error": "permission_denied"}
        if not isinstance(updates, dict) or not updates:
            return {"ok": False, "status": "invalid_config", "error": "updates are required"}
        forbidden = {
            "tab_id",
            "created_at",
            "models",
            "monitoring_active",
            "last_snapshot",
        }
        clean = {str(k): v for k, v in updates.items() if str(k) not in forbidden}
        if not clean:
            return {"ok": False, "status": "invalid_config", "error": "no editable fields"}
        if "title" in clean:
            clean["title"] = str(clean["title"] or "").strip()[:200]
            if not clean["title"]:
                return {"ok": False, "status": "invalid_config", "error": "title cannot be empty"}

        def _update():
            current = dict(getattr(widget, "config", None) or {})
            current.update(clean)
            widget.config = current
            manager = getattr(self._window, "tab_config_manager", None)
            if manager is None or not hasattr(manager, "add_config"):
                raise RuntimeError("config_manager_unavailable")
            manager.add_config(tab_id, current)
            tabs_widget = getattr(self._window, "tabs", None)
            if tabs_widget is not None and "title" in clean:
                index = tabs_widget.indexOf(widget)
                if index >= 0:
                    tabs_widget.setTabText(index, current["title"])
            refresh = getattr(self._window, "refresh_fleet_dashboard", None)
            if callable(refresh):
                refresh()
            return {"tab_id": tab_id, "updated_fields": sorted(clean)}

        return self._submit_operation("update_tab", _update, requested_by=requested_by)

    def delete_tab(self, tab_id: str, *, requested_by: str = "api") -> dict[str, Any]:
        widget = self._get_widget(tab_id)
        if widget is None:
            return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
        config = dict(getattr(widget, "config", None) or {})
        owner = str(config.get("created_by") or "").strip()
        actor = str(
            getattr(self._window, "current_username", None) or requested_by or ""
        ).strip()
        is_owner = bool(owner and actor and owner == actor)
        if not self._authorized("delete_tab", f"tab:{tab_id}") and not is_owner:
            return {"ok": False, "status": "forbidden", "error": "permission_denied"}

        def _delete():
            if bool(getattr(widget, "monitoring_active", False)) and hasattr(
                widget, "stop_monitoring"
            ):
                widget.stop_monitoring()
            tabs_widget = getattr(self._window, "tabs", None)
            if tabs_widget is not None:
                index = tabs_widget.indexOf(widget)
                if index >= 0:
                    tabs_widget.removeTab(index)
            custom_tabs = getattr(self._window, "custom_tabs", None) or {}
            custom_tabs.pop(tab_id, None)
            manager = getattr(self._window, "tab_config_manager", None)
            if manager is not None and hasattr(manager, "remove_config"):
                manager.remove_config(tab_id)
            refresh = getattr(self._window, "refresh_fleet_dashboard", None)
            if callable(refresh):
                refresh()
            return {"tab_id": tab_id, "deleted": True}

        return self._submit_operation("delete_tab", _delete, requested_by=requested_by)

    def train_tab(
        self,
        tab_id: str,
        *,
        model_id: Optional[str] = None,
        requested_by: str = "api",
    ) -> dict[str, Any]:
        widget = self._get_widget(tab_id)
        if widget is None:
            return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
        if not self._authorized("train_model", f"tab:{tab_id}") and not self._authorized(
            "create_tab", f"tab:{tab_id}"
        ):
            return {"ok": False, "status": "forbidden", "error": "permission_denied"}
        if model_id:
            listing = self.list_tab_models(tab_id)
            available = {str(m.get("model_id")) for m in listing.get("models") or []}
            if str(model_id) not in available:
                return {
                    "ok": False,
                    "status": "model_not_found",
                    "tab_id": tab_id,
                    "model_id": str(model_id),
                }

        def _train():
            train = getattr(widget, "train_model", None)
            if not callable(train):
                raise RuntimeError("train_unsupported")
            started = train(model_id=str(model_id) if model_id else None, silent=True)
            if started is False:
                raise RuntimeError("training_not_started")
            return {"tab_id": tab_id, "model_id": model_id, "started": True}

        result = self._submit_operation("train_tab", _train, requested_by=requested_by)
        result["job_id"] = result.get("operation_id")
        return result

    def _loaded_tab_frame(self, tab_id: str):
        widget = self._get_widget(tab_id)
        if widget is None:
            return None, None
        processor = getattr(widget, "data_processor", None)
        if processor is None:
            return widget, None
        frame = getattr(processor, "preprocessed_data", None)
        if frame is None:
            frame = getattr(processor, "data", None)
        return widget, frame

    def get_tab_data_schema(self, tab_id: str) -> dict[str, Any]:
        with self._lock:
            widget, frame = self._loaded_tab_frame(tab_id)
            if widget is None:
                return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
            config = dict(getattr(widget, "config", None) or {})
            if frame is None:
                return {
                    "ok": True,
                    "status": "data_not_loaded",
                    "tab_id": tab_id,
                    "source": {
                        "data_folder": config.get("data_folder"),
                        "source_file": config.get("source_file"),
                        "file_type": config.get("data_file_type"),
                    },
                    "row_count": 0,
                    "columns": [],
                }
            columns = []
            for col in frame.columns:
                series = frame[col]
                columns.append(
                    {
                        "name": str(col),
                        "dtype": str(series.dtype),
                        "null_count": int(series.isna().sum()),
                    }
                )
            return {
                "ok": True,
                "status": "ok",
                "tab_id": tab_id,
                "source": {
                    "data_folder": config.get("data_folder"),
                    "source_file": config.get("source_file"),
                    "file_type": config.get("data_file_type"),
                },
                "row_count": int(len(frame)),
                "columns": columns,
            }

    def get_tab_data_preview(self, tab_id: str, limit: int = 50, *, head: bool = False) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            widget, frame = self._loaded_tab_frame(tab_id)
            if widget is None:
                return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
            if frame is None:
                return {
                    "ok": True,
                    "status": "data_not_loaded",
                    "tab_id": tab_id,
                    "columns": [],
                    "rows": [],
                }
            preview = frame.head(limit) if head else frame.tail(limit)
            rows = [
                {
                    str(key): self._serialize_metric_value(value)
                    for key, value in record.items()
                }
                for record in preview.to_dict(orient="records")
            ]
            return {
                "ok": True,
                "status": "ok",
                "tab_id": tab_id,
                "columns": [str(col) for col in preview.columns],
                "row_count": int(len(frame)),
                "returned": len(rows),
                "rows": rows,
            }

    def start_tab(self, tab_id: str, *, requested_by: str = "api") -> dict[str, Any]:
        widget = self._get_widget(tab_id)
        if widget is None:
            return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
        if not self._authorized("process_data", f"tab:{tab_id}") and not self._authorized(
            "create_tab", "*"
        ):
            return {"ok": False, "status": "forbidden", "error": "permission_denied"}
        if bool(getattr(widget, "monitoring_active", False)):
            return {
                "ok": True,
                "accepted": True,
                "status": "already_monitoring",
                "tab_id": tab_id,
            }
        count_fn = getattr(widget, "_count_trained_models", None)
        if callable(count_fn):
            try:
                trained_n = int(count_fn())
            except Exception:
                trained_n = -1
            if trained_n == 0:
                return {
                    "ok": False,
                    "status": "not_ready",
                    "error": "no_trained_models",
                    "tab_id": tab_id,
                }

        def _start():
            start = getattr(widget, "start_monitoring", None)
            if not callable(start):
                raise RuntimeError("start_unsupported")
            start()
            return {
                "tab_id": tab_id,
                "monitoring_active": bool(getattr(widget, "monitoring_active", True)),
            }

        return self._submit_operation("start_tab", _start, requested_by=requested_by)

    def stop_tab(self, tab_id: str, *, requested_by: str = "api") -> dict[str, Any]:
        widget = self._get_widget(tab_id)
        if widget is None:
            return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
        if not self._authorized("process_data", f"tab:{tab_id}") and not self._authorized(
            "create_tab", "*"
        ):
            return {"ok": False, "status": "forbidden", "error": "permission_denied"}

        def _stop():
            stop = getattr(widget, "stop_monitoring", None)
            if not callable(stop):
                raise RuntimeError("stop_unsupported")
            stop()
            return {
                "tab_id": tab_id,
                "monitoring_active": bool(getattr(widget, "monitoring_active", False)),
            }

        return self._submit_operation("stop_tab", _stop, requested_by=requested_by)

    def get_tab_pipeline_metrics(self, tab_id: str, limit: int = 50) -> dict[str, Any]:
        widget = self._get_widget(tab_id)
        if widget is None:
            return {"ok": False, "status": "tab_not_found", "tab_id": tab_id, "metrics": []}
        limit = max(1, min(int(limit), 200))
        try:
            from app.monitoring.async_pipeline import get_async_pipeline

            recent = get_async_pipeline().get_recent_metrics(limit=500)
        except Exception as exc:
            return {"ok": True, "tab_id": tab_id, "metrics": [], "error": str(exc)[:200]}
        matched = [m for m in recent if str(m.get("tab_id") or "") == str(tab_id)]
        return {"ok": True, "tab_id": tab_id, "metrics": matched[-limit:]}

    def load_tab_data(
        self,
        tab_id: str,
        *,
        file_path: str,
        file_type: str = "csv",
        requested_by: str = "api",
    ) -> dict[str, Any]:
        widget = self._get_widget(tab_id)
        if widget is None:
            return {"ok": False, "status": "tab_not_found", "tab_id": tab_id}
        if not self._authorized("import_data", f"tab:{tab_id}") and not self._authorized(
            "view_data", "*"
        ):
            return {"ok": False, "status": "forbidden", "error": "permission_denied"}
        kind = str(file_type or "csv").strip().lower()
        if kind not in {"csv", "json"}:
            return {"ok": False, "status": "invalid_config", "error": "file_type must be csv or json"}
        try:
            resolved = _safe_local_data_path(
                file_path,
                extra_root=str((getattr(widget, "config", None) or {}).get("data_folder") or ""),
            )
        except FileNotFoundError:
            return {"ok": False, "status": "not_found", "error": "file_path does not exist"}
        except ValueError as exc:
            return {"ok": False, "status": "invalid_config", "error": str(exc)}

        def _load():
            config = dict(getattr(widget, "config", None) or {})
            config["source_file"] = str(resolved)
            config["data_folder"] = str(resolved.parent)
            config["data_file_type"] = "CSV" if kind == "csv" else "JSON"
            widget.config = config
            manager = getattr(self._window, "tab_config_manager", None)
            if manager is not None and hasattr(manager, "add_config"):
                manager.add_config(tab_id, config)
            loader = getattr(widget, "load_latest_data", None)
            if callable(loader):
                _data, err = loader(for_monitoring=False, force_reload=True)
                if err:
                    raise RuntimeError(err)
            return {"tab_id": tab_id, "source_file": str(resolved), "file_type": kind}

        return self._submit_operation("load_tab_data", _load, requested_by=requested_by)

    def get_registry_models(self, tab_id: str, limit: int = 20) -> dict[str, Any]:
        listing = self.list_tab_models(tab_id)
        if not listing.get("ok"):
            return listing
        registry = getattr(self._window, "model_registry", None)
        recent: list[dict[str, Any]] = []
        getter = getattr(registry, "get_recent_models", None)
        if callable(getter):
            try:
                rows = getter(limit=max(1, min(int(limit), 50)))
                for row in rows or []:
                    if isinstance(row, dict):
                        recent.append(row)
                    elif isinstance(row, (list, tuple)) and len(row) >= 3:
                        recent.append(
                            {
                                "id": row[0],
                                "name": row[1],
                                "model_type": row[2],
                                "last_used": row[3] if len(row) > 3 else None,
                                "use_count": row[4] if len(row) > 4 else None,
                                "filepath": row[5] if len(row) > 5 else None,
                            }
                        )
            except TypeError:
                try:
                    rows = getter(max(1, min(int(limit), 50)))
                    for row in rows or []:
                        if isinstance(row, (list, tuple)) and len(row) >= 3:
                            recent.append({"id": row[0], "name": row[1], "model_type": row[2]})
                except Exception:
                    recent = []
            except Exception:
                recent = []
        return {
            "ok": True,
            "tab_id": tab_id,
            "models": listing.get("models") or [],
            "registry": recent,
        }

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
