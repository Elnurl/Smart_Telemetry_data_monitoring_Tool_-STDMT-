"""Proactive fleet cycle drafts with dedupe (Faza C)."""

from __future__ import annotations

import time
from typing import Any, Optional

# key -> last emit monotonic time
_EMIT_CACHE: dict[str, float] = {}
DEFAULT_DEDUPE_SECONDS = 900  # 15 minutes


def _cache_key(tab_id: str, kind: str) -> str:
    return f"{tab_id}|{kind}"


def should_emit(tab_id: str, kind: str, *, dedupe_seconds: int = DEFAULT_DEDUPE_SECONDS) -> bool:
    key = _cache_key(str(tab_id), str(kind))
    now = time.monotonic()
    last = _EMIT_CACHE.get(key)
    if last is not None and (now - last) < max(60, int(dedupe_seconds)):
        return False
    return True


def mark_emitted(tab_id: str, kind: str) -> None:
    _EMIT_CACHE[_cache_key(str(tab_id), str(kind))] = time.monotonic()


def clear_dedupe_cache() -> None:
    _EMIT_CACHE.clear()


def pending_draft_exists(
    host: Any,
    tab_id: str,
    kind: str,
) -> bool:
    """True if a pending draft already exists for this tab_id + kind."""
    if host is None or not hasattr(host, "list_pending_drafts"):
        return False
    kind_l = str(kind or "").strip().lower()
    tid = str(tab_id or "").strip()
    try:
        for d in host.list_pending_drafts() or []:
            if not isinstance(d, dict):
                continue
            if str(d.get("kind") or "").strip().lower() != kind_l:
                continue
            if tid and str(d.get("tab_id") or "").strip() != tid:
                continue
            return True
    except Exception:
        return False
    return False


def proactive_actions_from_tabs(
    tabs: list[dict[str, Any]],
    *,
    dedupe_seconds: int = DEFAULT_DEDUPE_SECONDS,
    host: Any = None,
) -> list[dict[str, Any]]:
    """Decide which propose_* actions to fire for elevated tabs."""
    actions: list[dict[str, Any]] = []
    for entry in tabs or []:
        tab_id = str(entry.get("tab_id") or "")
        snap = entry.get("snapshot") or {}
        title = snap.get("title") or entry.get("title") or tab_id
        if not tab_id:
            continue
        health = str(snap.get("health_state") or "").lower()
        drift = bool(snap.get("drift"))
        obs_ok = snap.get("obs_ok", True)
        fusion = snap.get("fusion_score")
        monitoring = bool(snap.get("monitoring_active"))

        elevated = False
        reasons = []
        severity = "WARNING"
        if health in ("warning", "critical", "alert"):
            elevated = True
            reasons.append(f"health={health}")
            if health == "critical":
                severity = "CRITICAL"
        if drift:
            elevated = True
            reasons.append("drift")
        if obs_ok is False:
            elevated = True
            reasons.append("OBS violation")
            severity = "CRITICAL"
        try:
            if fusion is not None and float(fusion) >= 0.55:
                elevated = True
                reasons.append(f"fusion={fusion}")
        except Exception:
            pass

        if drift and should_emit(tab_id, "retrain", dedupe_seconds=dedupe_seconds):
            actions.append(
                {
                    "tool": "propose_retrain",
                    "tab_id": tab_id,
                    "title": title,
                    "params": {
                        "reasoning": f"Proactive cycle: drift on {title}",
                        "drift_score": float(snap.get("drift_score") or 0.0)
                        if snap.get("drift_score") is not None
                        else 0.5,
                    },
                }
            )

        if elevated and should_emit(tab_id, "alert", dedupe_seconds=dedupe_seconds):
            if host is None or not pending_draft_exists(host, tab_id, "alert"):
                actions.append(
                    {
                        "tool": "propose_alert",
                        "tab_id": tab_id,
                        "title": title,
                        "params": {
                            "proposed_message": (
                                f"Proactive alert for {title}: " + ", ".join(reasons)
                            ),
                            "agent_reasoning": "Agent monitor cycle elevated signals",
                            "severity": severity,
                        },
                    }
                )

        if (not monitoring) and health in ("warning", "critical") and should_emit(
            tab_id, "start_hint", dedupe_seconds=dedupe_seconds
        ):
            if host is None or not pending_draft_exists(host, tab_id, "alert"):
                actions.append(
                    {
                        "tool": "propose_alert",
                        "tab_id": tab_id,
                        "title": title,
                        "params": {
                            "proposed_message": (
                                f"{title} shows elevated health but monitoring is inactive — "
                                "consider Approve start monitoring"
                            ),
                            "agent_reasoning": "Inactive tab with elevated health",
                            "severity": "WARNING",
                        },
                    }
                )

        # Critical + actively monitoring → suggest stop once (pending-draft gated)
        if (
            monitoring
            and health == "critical"
            and should_emit(tab_id, "stop_monitoring", dedupe_seconds=dedupe_seconds)
        ):
            if host is None or not pending_draft_exists(host, tab_id, "stop_monitoring"):
                actions.append(
                    {
                        "tool": "propose_stop_monitoring",
                        "tab_id": tab_id,
                        "title": title,
                        "params": {
                            "proposed_message": f"Stop monitoring on {title}",
                            "agent_reasoning": (
                                f"The tab '{title}' is in a critical health state — "
                                "consider stopping monitoring until models are retrained."
                            ),
                            "severity": "CRITICAL",
                        },
                    }
                )

    return actions


def execute_proactive_actions(
    host: Any,
    actions: list[dict[str, Any]],
    *,
    dedupe_seconds: int = DEFAULT_DEDUPE_SECONDS,
) -> list[dict[str, Any]]:
    """Invoke propose tools on host; mark dedupe on success. Skip if pending exists."""
    from app.agent.tools import propose_alert, propose_retrain, propose_stop_monitoring

    results = []
    for act in actions:
        tool = act.get("tool")
        tab_id = act.get("tab_id")
        params = dict(act.get("params") or {})
        try:
            kind_map = {
                "propose_alert": "alert",
                "propose_stop_monitoring": "stop_monitoring",
                "propose_start_monitoring": "start_monitoring",
            }
            kind = kind_map.get(str(tool) or "")
            if kind and pending_draft_exists(host, str(tab_id), kind):
                results.append(
                    {
                        "tool": tool,
                        "tab_id": tab_id,
                        "ok": True,
                        "deduped": True,
                        "result": {"ok": True, "deduped": True, "kind": kind},
                    }
                )
                mark_emitted(tab_id, kind)
                continue

            if tool == "propose_retrain":
                out = propose_retrain(tab_id, host=host, **params)
                kind = "retrain"
            elif tool == "propose_alert":
                out = propose_alert(tab_id, host=host, **params)
                kind = "alert"
            elif tool == "propose_stop_monitoring":
                out = propose_stop_monitoring(
                    str(params.get("proposed_message") or ""),
                    tab_id=tab_id,
                    agent_reasoning=str(params.get("agent_reasoning") or ""),
                    severity=str(params.get("severity") or "CRITICAL"),
                    host=host,
                )
                kind = "stop_monitoring"
            else:
                out = {"ok": False, "error": f"unknown_tool:{tool}"}
                kind = "unknown"
            ok = bool(out.get("ok"))
            if ok:
                mark_emitted(tab_id, kind)
                if "monitoring is inactive" in str(params.get("proposed_message") or ""):
                    mark_emitted(tab_id, "start_hint")
            results.append(
                {
                    "tool": tool,
                    "tab_id": tab_id,
                    "ok": ok,
                    "deduped": bool(out.get("deduped")),
                    "result": out,
                }
            )
        except Exception as exc:
            results.append({"tool": tool, "tab_id": tab_id, "ok": False, "error": str(exc)})
    return results
