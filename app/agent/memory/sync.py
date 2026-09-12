"""Pull fleet / drafts / FSM / audit into a GraphMemoryStore."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("STDMS.Agent.GraphMemory")

_LAST_SYNC = 0.0
_SYNC_LOCK = threading.Lock()
_DEFAULT_MIN_INTERVAL_S = 5.0


def sync_fleet_throttled(
    store: Any,
    host: Any,
    *,
    min_interval_s: float = _DEFAULT_MIN_INTERVAL_S,
) -> dict[str, Any]:
    """Best-effort fleet sync; skips if last sync was recent."""
    global _LAST_SYNC
    now = time.monotonic()
    with _SYNC_LOCK:
        if now - _LAST_SYNC < float(min_interval_s):
            return {"ok": True, "skipped": True, "reason": "throttled"}
        _LAST_SYNC = now
    try:
        return sync_fleet_into(store, host)
    except Exception as exc:
        logger.warning("graph sync_fleet failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def sync_fleet_into(store: Any, host: Any) -> dict[str, Any]:
    """Full sync from ToolHost (+ optional audit) into ``store``."""
    counts = {
        "tabs": 0,
        "modes": 0,
        "drafts": 0,
        "retrain": 0,
        "decisions": 0,
        "models": 0,
        "tools": 0,
    }
    if host is None:
        return {"ok": False, "error": "host_unavailable", **counts}

    tabs = []
    try:
        tabs = host.list_tabs() or []
    except Exception as exc:
        logger.warning("list_tabs for graph sync failed: %s", exc)

    for entry in tabs:
        tid = str(entry.get("tab_id") or "")
        if not tid:
            continue
        snap = entry.get("snapshot") or {}
        if not isinstance(snap, dict):
            snap = {}
        title = str(entry.get("title") or snap.get("title") or tid)
        mode = str(snap.get("mission_mode") or "nominal")
        try:
            scale = float(snap.get("threshold_scale") or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        store.upsert_node(
            "Tab",
            tid,
            {
                "title": title,
                "health": snap.get("health_state") or snap.get("health") or "Idle",
                "mission_mode": mode,
                "threshold_scale": scale,
                "monitoring": bool(snap.get("monitoring_active")),
                "drift": bool(snap.get("drift")),
                "alert_count": int(snap.get("alert_count") or 0),
                "obs_ok": snap.get("obs_ok", True),
            },
        )
        counts["tabs"] += 1
        store.upsert_node("Mode", mode, {"threshold_scale": scale})
        store.upsert_edge("Tab", tid, "IN_MODE", "Mode", mode)
        counts["modes"] += 1

        # Optional trained model count as lightweight Model nodes
        trained = int(snap.get("trained_models") or 0)
        if trained > 0:
            mid = f"{tid}:trained_count"
            store.upsert_node(
                "Model",
                mid,
                {"tab_id": tid, "trained": True, "count": trained, "model_type": "aggregate"},
            )
            store.upsert_edge("Tab", tid, "HAS_MODEL", "Model", mid)
            counts["models"] += 1

    # Pending drafts
    drafts = []
    try:
        if hasattr(host, "list_pending_drafts"):
            drafts = host.list_pending_drafts() or []
    except Exception as exc:
        logger.warning("list_pending_drafts for graph sync failed: %s", exc)
    for d in drafts:
        if not isinstance(d, dict):
            continue
        did = str(d.get("draft_id") or d.get("id") or "")
        if not did:
            continue
        tid = str(d.get("tab_id") or "")
        store.upsert_node(
            "Draft",
            did,
            {
                "kind": d.get("kind") or d.get("alert_type") or "draft",
                "severity": d.get("severity") or "INFO",
                "status": d.get("status") or "pending",
                "tab_id": tid,
                "message": str(d.get("proposed_message") or d.get("message") or "")[:300],
            },
        )
        counts["drafts"] += 1
        if tid:
            store.upsert_edge("Tab", tid, "HAS_DRAFT", "Draft", did)

    # Retrain signals
    signals = []
    try:
        signals = host.get_pending_retrain_signals() or []
    except Exception as exc:
        logger.warning("get_pending_retrain_signals for graph sync failed: %s", exc)
    for s in signals:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or s.get("signal_id") or "")
        tid = str(s.get("tab_id") or "")
        if not sid:
            sid = f"{tid}:{s.get('created_at') or counts['retrain']}"
        store.upsert_node(
            "RetrainSignal",
            sid,
            {
                "tab_id": tid,
                "drift_score": s.get("drift_score"),
                "status": s.get("status") or "pending",
                "features": str(s.get("drifted_features") or "")[:200],
            },
        )
        counts["retrain"] += 1
        if tid:
            store.upsert_edge("Tab", tid, "HAS_RETRAIN", "RetrainSignal", sid)

    # Recent audit decisions
    audit = _resolve_audit(host)
    if audit is not None:
        try:
            decisions = audit.list_decisions(limit=40)
        except Exception as exc:
            logger.warning("list_decisions for graph sync failed: %s", exc)
            decisions = []
        for dec in decisions:
            record_decision_memory(
                store,
                decision_id=dec.get("id"),
                tab_id=dec.get("tab_id"),
                outcome=dec.get("outcome"),
                tool_called=dec.get("tool_called"),
                reasoning=dec.get("reasoning"),
                timestamp=dec.get("timestamp"),
                tool_trace=_tool_names_from_params(dec.get("tool_params")),
            )
            counts["decisions"] += 1

    st = store.stats() if hasattr(store, "stats") else {}
    return {"ok": True, "skipped": False, **counts, "stats": st}


def record_decision_memory(
    store: Any,
    *,
    decision_id: Any,
    tab_id: Optional[str] = None,
    outcome: Optional[str] = None,
    tool_called: Optional[str] = None,
    reasoning: Optional[str] = None,
    timestamp: Optional[str] = None,
    tool_trace: Optional[list[str]] = None,
) -> None:
    """Upsert a Decision node and link to Tab / tools."""
    if store is None or decision_id is None:
        return
    did = str(decision_id)
    try:
        store.upsert_node(
            "Decision",
            did,
            {
                "outcome": outcome or "",
                "tool_called": tool_called or "",
                "reasoning": str(reasoning or "")[:500],
                "timestamp": timestamp or "",
                "tab_id": tab_id or "",
            },
        )
        if tab_id:
            store.upsert_edge("Tab", str(tab_id), "HAD_DECISION", "Decision", did)
        tools = list(tool_trace or [])
        if tool_called and tool_called not in tools:
            tools.append(str(tool_called))
        for name in tools[:12]:
            if not name:
                continue
            store.upsert_node("Tool", str(name), {"name": str(name)})
            store.upsert_edge("Decision", did, "USED_TOOL", "Tool", str(name))
    except Exception as exc:
        logger.warning("record_decision_memory failed: %s", exc)


def _resolve_audit(host: Any) -> Any:
    if host is None:
        return None
    for attr in ("audit", "_audit"):
        a = getattr(host, attr, None)
        if a is not None and hasattr(a, "list_decisions"):
            return a
    window = getattr(host, "_window", None)
    if window is not None:
        bridge = getattr(window, "agent_bridge", None) or getattr(window, "_agent_bridge", None)
        if bridge is not None:
            a = getattr(bridge, "audit", None)
            if a is not None and hasattr(a, "list_decisions"):
                return a
        runner = getattr(window, "agent_runner", None)
        if runner is not None:
            a = getattr(runner, "audit", None)
            if a is not None and hasattr(a, "list_decisions"):
                return a
    return None


def _tool_names_from_params(params: Any) -> list[str]:
    if not isinstance(params, dict):
        return []
    trace = params.get("tool_trace") or []
    names: list[str] = []
    for t in trace:
        if isinstance(t, dict) and t.get("tool"):
            names.append(str(t["tool"]))
    return names
