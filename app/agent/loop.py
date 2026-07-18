"""Phase 1 — periodic read-only agent monitor loop + interactive ask.

Reads tab snapshots on an interval, asks local Ollama (Qwen) via REST when
available, otherwise a richer English rule-based analysis. Never mutates
tabs / models / alerts. Updates the Dashboard via a Qt signal.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from app.agent.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    call_ollama,
    ollama_reachable,
)
from app.agent.policy import is_loopback_url

logger = logging.getLogger("STDMS.Agent.Loop")


class AgentMonitorSignals(QObject):
    """Emitted from the monitor thread; slots run on the Qt GUI thread."""

    refreshed = pyqtSignal(dict)  # payload: summary, lines, updated_at, decision_id, llm_used
    chat_reply = pyqtSignal(dict)  # payload: user_message, reply, llm_used, error


class AgentMonitorLoop:
    """
    Every N seconds (default: 60) read all tab snapshots, optionally call
    local Ollama (Qwen), write an audit row, and notify the Dashboard.

    If Ollama is unavailable: collect snapshots and write a rule-based summary.
    """

    def __init__(
        self,
        bridge: Any,
        *,
        parent: Optional[QObject] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        allow_llm: Optional[bool] = None,
        on_update: Optional[Callable[[dict], None]] = None,
    ):
        self.bridge = bridge
        self.ollama_url = (ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self.ollama_model = ollama_model or DEFAULT_OLLAMA_MODEL
        if allow_llm is None:
            allow_llm = bool(getattr(bridge, "allow_llm", False))
        self.allow_llm = bool(allow_llm) and is_loopback_url(self.ollama_url)
        self.on_update = on_update
        self.signals = AgentMonitorSignals(parent)

        self._interval = 60.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self.last_result: Optional[dict[str, Any]] = None

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def llm_status(self) -> dict[str, Any]:
        from app.agent.ollama_client import resolve_chat_model

        up = ollama_reachable(self.ollama_url)
        model = resolve_chat_model(self.ollama_model, base_url=self.ollama_url) if up else self.ollama_model
        self.ollama_model = model
        # Keep runner in sync so Ask and status use the same model
        runner = getattr(self.bridge, "runner", None)
        if runner is not None:
            try:
                runner.model = model
            except Exception:
                pass
        return {
            "ollama_up": up,
            "model": model,
            "url": self.ollama_url,
            "will_use_llm": up and is_loopback_url(self.ollama_url),
        }

    def start(self, interval_seconds: float = 60) -> None:
        with self._lock:
            if self.is_running:
                logger.info("AgentMonitorLoop already running")
                return
            self._interval = max(5.0, float(interval_seconds))
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop_main,
                name="stdms-agent-monitor",
                daemon=True,
            )
            self._thread.start()
            logger.info("AgentMonitorLoop started (interval=%ss)", self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None
        logger.info("AgentMonitorLoop stopped")

    def run_once(self) -> dict[str, Any]:
        """Public helper for tests / Analyze Now."""
        return self._run_cycle()

    def ask(self, user_message: str) -> dict[str, Any]:
        """Interactive Q&A with Phase 2 function calling when Ollama is up."""
        user_message = (user_message or "").strip()
        if not user_message:
            payload = {
                "ok": False,
                "user_message": "",
                "reply": "Please type a question for the agent.",
                "llm_used": False,
                "tool_trace": [],
                "error": "empty",
            }
            self.signals.chat_reply.emit(payload)
            return payload

        host = self._tool_host()
        runner = getattr(self.bridge, "runner", None)
        tool_trace: list = []
        llm_used = False
        reply = None
        decision_id = None
        outcome = "observation_only"
        agent_mode = "offline"

        route = None
        node = None
        log_summary = ""
        node_trace: list = []

        if runner is not None and hasattr(runner, "ask"):
            try:
                result = runner.ask(user_message)
                reply = result.get("reply")
                llm_used = bool(result.get("llm_used"))
                tool_trace = result.get("tool_trace") or []
                decision_id = result.get("decision_id")
                outcome = result.get("outcome") or outcome
                agent_mode = str(result.get("agent_mode") or ("llm" if llm_used else "tools"))
                route = result.get("route")
                node = result.get("node")
                log_summary = result.get("log_summary") or ""
                node_trace = list(result.get("node_trace") or [])
            except Exception as exc:
                logger.warning("Runner.ask failed: %s", exc)
                reply = None

        if not reply:
            snapshots, pending = self._collect_snapshots(host)
            from app.agent.function_calling import _is_write_document_intent
            from app.agent.tools import propose_write_sop_from_user_text

            if _is_write_document_intent(user_message):
                try:
                    auto = propose_write_sop_from_user_text(user_message, host=host)
                    tool_trace.append(
                        {
                            "tool": "propose_write_sop",
                            "params": {"auto_from_user_text": True},
                            "ok": bool(auto.get("ok")),
                        }
                    )
                    if auto.get("ok"):
                        reply = (
                            "Created a pending SOP Word draft for human Approve.\n"
                            f"- draft_id: {auto.get('draft_id')}\n"
                            f"- kind: {auto.get('kind')}\n"
                            "- Next: Dashboard → Pending Agent Drafts → Approve"
                        )
                        outcome = "function_calling"
                        llm_used = True
                    else:
                        reply = f"Could not create SOP draft: {auto.get('error') or auto}"
                        outcome = "observation_only"
                except Exception as exc:
                    logger.warning("SOP auto-propose failed: %s", exc)
                    reply = None
            if not reply:
                summary, _lines = self._rule_based_summary(snapshots, pending_retrain=pending)
                reply = (
                    f"[Ollama offline — heuristic analysis]\n\n{summary}\n\n"
                    f"Regarding your question ({user_message}): "
                    "Start with any Warning/drift/OBS-failing tabs above; "
                    "Idle tabs with no recent file may indicate a stalled feed or monitoring not started."
                )
                llm_used = False
                outcome = "observation_only"
                agent_mode = "offline"

        audit = self._audit()
        if audit is not None and decision_id is None:
            try:
                decision_id = audit.record(
                    tab_id=None,
                    context_summary=f"chat: {user_message[:500]}",
                    tool_called=tool_trace[-1]["tool"] if tool_trace else "agent_chat",
                    tool_params={
                        "llm_used": llm_used,
                        "agent_mode": agent_mode,
                        "phase": 2,
                        "tool_trace": tool_trace,
                    },
                    reasoning=str(reply)[:8000],
                    outcome=str(outcome),
                )
            except Exception as exc:
                logger.warning("Failed to audit chat: %s", exc)

        payload = {
            "ok": True,
            "user_message": user_message,
            "reply": reply,
            "llm_used": llm_used,
            "agent_mode": agent_mode,
            "route": route,
            "node": node,
            "log_summary": log_summary,
            "node_trace": node_trace,
            "tool_trace": tool_trace,
            "decision_id": decision_id,
            "outcome": outcome,
            "phase": 3,
            "error": None,
        }
        self.signals.chat_reply.emit(payload)
        return payload

    def _loop_main(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_cycle()
            except Exception as exc:
                logger.warning("Agent monitor cycle failed: %s", exc)
            deadline = time.monotonic() + self._interval
            while not self._stop.is_set() and time.monotonic() < deadline:
                time.sleep(min(0.5, deadline - time.monotonic()))

    def _tool_host(self) -> Any:
        host = getattr(self.bridge, "tool_host", None)
        if host is not None:
            return host
        getter = getattr(self.bridge, "tool_host_getter", None)
        if callable(getter):
            return getter()
        from app.agent.bridge import get_tool_host

        return get_tool_host()

    def _audit(self) -> Any:
        return getattr(self.bridge, "audit", None)

    def _collect_snapshots(self, host: Any) -> tuple[list[dict[str, Any]], list]:
        if host is None:
            return [], []
        tabs = host.list_tabs() or []
        snapshots = []
        for entry in tabs:
            snap = entry.get("snapshot") or {}
            if not isinstance(snap, dict):
                snap = {}
            merged = dict(snap)
            merged.setdefault("tab_id", entry.get("tab_id"))
            merged.setdefault("title", entry.get("title") or merged.get("tab_id"))
            snapshots.append(merged)
        pending = []
        try:
            pending = host.get_pending_retrain_signals() or []
        except Exception:
            pending = []
        return snapshots, pending

    def _run_cycle(self) -> dict[str, Any]:
        host = self._tool_host()
        updated_at = datetime.now().strftime("%H:%M:%S")
        if host is None:
            payload = {
                "ok": False,
                "summary": "Agent: tool host is not attached.",
                "lines": [],
                "updated_at": updated_at,
                "decision_id": None,
                "llm_used": False,
            }
            self._publish(payload)
            return payload

        runner = getattr(self.bridge, "runner", None)
        if self.allow_llm and runner is not None and hasattr(runner, "run_agent_cycle"):
            try:
                cycle = runner.run_agent_cycle(task="fleet_monitor")
                payload = {
                    "ok": bool(cycle.get("ok", True)),
                    "summary": cycle.get("summary") or cycle.get("reasoning") or "",
                    "lines": cycle.get("lines") or [],
                    "updated_at": updated_at,
                    "decision_id": cycle.get("decision_id"),
                    "llm_used": bool(cycle.get("llm_used")),
                    "tool_trace": cycle.get("tool_trace") or [],
                    "tab_count": cycle.get("tab_count"),
                    "pending_draft_count": cycle.get("pending_draft_count", 0),
                    "pending_retrain_count": cycle.get("pending_retrain_count", 0),
                    "ollama_up": ollama_reachable(self.ollama_url),
                    "phase": 3,
                    "outcome": cycle.get("outcome"),
                }
                mllm = self._run_mllm_log_pass(host)
                payload.update(mllm)
                if mllm.get("mllm_lines"):
                    payload["lines"] = list(payload.get("lines") or []) + list(mllm["mllm_lines"])
                if payload["summary"]:
                    self._publish(payload)
                    return payload
            except Exception as exc:
                logger.warning("run_agent_cycle failed, falling back: %s", exc)

        snapshots, pending = self._collect_snapshots(host)
        context = self._build_context(snapshots, pending_retrain=pending)
        llm_text = None
        llm_used = False
        tool_trace: list = []
        if self.allow_llm and is_loopback_url(self.ollama_url) and ollama_reachable(self.ollama_url):
            from app.agent.function_calling import run_function_calling

            fc = run_function_calling(
                "Periodic fleet monitor. Use tools, then brief the operator "
                "with [Observation]/[Analysis]/[Recommendation].\n\nSeed:\n" + context,
                host=host,
                ollama_url=self.ollama_url,
                model=self.ollama_model,
                include_mutating=True,
            )
            tool_trace = fc.get("tool_trace") or []
            llm_text = fc.get("reply")
            llm_used = bool(fc.get("llm_used"))

        if llm_text:
            summary = llm_text.strip()
            lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
        else:
            summary, lines = self._rule_based_summary(snapshots, pending_retrain=pending)

        decision_id = None
        audit = self._audit()
        if audit is not None:
            try:
                decision_id = audit.record(
                    tab_id=None,
                    context_summary=context[:2000],
                    tool_called=tool_trace[-1]["tool"] if tool_trace else "agent_monitor_cycle",
                    tool_params={
                        "tab_count": len(snapshots),
                        "llm_used": llm_used,
                        "interval": self._interval,
                        "phase": 3,
                        "tool_trace": tool_trace,
                    },
                    reasoning=summary[:8000],
                    outcome="function_calling" if tool_trace else "observation_only",
                )
            except Exception as exc:
                logger.warning("Failed to audit monitor cycle: %s", exc)

        payload = {
            "ok": True,
            "summary": summary,
            "lines": lines,
            "updated_at": updated_at,
            "decision_id": decision_id,
            "llm_used": llm_used,
            "tool_trace": tool_trace,
            "tab_count": len(snapshots),
            "ollama_up": ollama_reachable(self.ollama_url),
            "phase": 3,
        }
        mllm = self._run_mllm_log_pass(host)
        payload.update(mllm)
        if mllm.get("mllm_lines"):
            payload["lines"] = list(payload.get("lines") or []) + list(mllm["mllm_lines"])
        self._publish(payload)
        return payload

    def _run_mllm_log_pass(self, host: Any) -> dict[str, Any]:
        """Analyze ingested logs per tab (M-LLM / rule) and attach dashboard fields."""
        empty = {
            "mllm_summary": "",
            "mllm_lines": [],
            "mllm_analyses": [],
        }
        try:
            from app.agent.log_monitor import get_log_monitor
            from app.models.fsm import MissionModeStore
        except Exception as exc:
            logger.warning("log monitor import failed: %s", exc)
            return empty

        try:
            tabs = host.list_tabs() or []
        except Exception:
            tabs = []
        if not tabs:
            return empty

        monitor = get_log_monitor(
            ollama_url=self.ollama_url,
            ollama_model=self.ollama_model,
            allow_llm=self.allow_llm,
        )
        fsm_store = MissionModeStore()
        titles: dict[str, str] = {}
        configs: dict[str, dict] = {}
        tab_ids: list[str] = []
        widgets: dict[str, Any] = {}
        window = getattr(host, "_window", None)
        custom_tabs = getattr(window, "custom_tabs", None) or {} if window is not None else {}

        for entry in tabs:
            tid = str(entry.get("tab_id") or "")
            if not tid:
                continue
            tab_ids.append(tid)
            titles[tid] = str(entry.get("title") or tid)
            snap = entry.get("snapshot") or {}
            widget = custom_tabs.get(tid)
            if widget is not None:
                widgets[tid] = widget
            if widget is not None and hasattr(widget, "config"):
                configs[tid] = dict(getattr(widget, "config") or {})
            else:
                configs[tid] = {
                    "current_mission_mode": snap.get("mission_mode") or "nominal",
                    "mission_modes": [
                        {"name": "nominal", "threshold_scale": 1.0},
                        {"name": "eclipse", "threshold_scale": float(snap.get("threshold_scale") or 1.5)},
                        {"name": "maneuver", "threshold_scale": 2.0},
                        {"name": "safe_mode", "threshold_scale": 0.5},
                    ],
                }

        analyses: list = []
        for tid in tab_ids:
            if not monitor.get_window(tid):
                continue
            analyses.append(
                monitor.analyze(
                    tid,
                    fsm_store,
                    tab_config=configs.get(tid),
                    apply_transitions=False,  # widget.set_mission_mode persists FSM
                    use_llm=self.allow_llm,
                )
            )
        for analysis in analyses:
            widget = widgets.get(analysis.tab_id)
            if widget is None:
                continue
            try:
                widget.last_mllm_summary = analysis.summary
                widget.last_mllm_analysis = analysis.to_dict()
                if analysis.mode_transitions and hasattr(widget, "set_mission_mode"):
                    last = analysis.mode_transitions[-1]
                    to_mode = last.get("to_mode")
                    if to_mode and str(to_mode) != str(
                        (getattr(widget, "config", {}) or {}).get("current_mission_mode")
                    ):
                        widget.set_mission_mode(str(to_mode), trigger_source="log_event")
                elif hasattr(widget, "_publish_snapshot"):
                    widget._publish_snapshot()
            except Exception:
                pass

        lines = monitor.dashboard_lines(analyses, titles=titles)
        return {
            "mllm_summary": "\n".join(lines),
            "mllm_lines": lines,
            "mllm_analyses": [a.to_dict() for a in analyses],
        }

    def _publish(self, payload: dict[str, Any]) -> None:
        self.last_result = payload
        try:
            self.signals.refreshed.emit(payload)
        except Exception:
            pass
        if callable(self.on_update):
            try:
                self.on_update(payload)
            except Exception as exc:
                logger.warning("on_update callback failed: %s", exc)

    def _build_context(
        self,
        snapshots: list[dict[str, Any]],
        *,
        pending_retrain: Optional[list] = None,
    ) -> str:
        pending_retrain = pending_retrain or []
        lines = [
            "STDMS Instrumentation Agent — fleet snapshot context (read-only).",
            f"Tabs: {len(snapshots)}; pending_retrain: {len(pending_retrain)}.",
        ]
        for snap in snapshots:
            title = snap.get("title") or snap.get("tab_id") or "?"
            lines.append(
                f"- {title}: health={snap.get('health_state', '?')}, "
                f"drift={snap.get('drift')}, fusion={snap.get('fusion_score')}, "
                f"alerts={snap.get('alert_count', 0)}, obs_ok={snap.get('obs_ok')}, "
                f"obs_violations={snap.get('obs_violations', 0)}, "
                f"last_file={snap.get('last_file', '—')}, "
                f"trained={snap.get('trained_models', 0)}, "
                f"monitoring={snap.get('monitoring_active')}, "
                f"watch={snap.get('watch_status', '—')}, "
                f"updated={snap.get('updated_at', '—')}"
            )
        if pending_retrain:
            lines.append("Pending retrain signals:")
            for sig in pending_retrain[:10]:
                lines.append(
                    f"  - tab={sig.get('tab_id')} model={sig.get('model_id')} "
                    f"drift={sig.get('drift_score')} reason={sig.get('reason')}"
                )
        lines.append(
            "Respond in English. Prioritize attention; explain Idle+no recent file as possible "
            "stalled feed / monitoring not started — not as 'healthy'."
        )
        return "\n".join(lines)

    def _call_llm(self, context: str) -> Optional[str]:
        return call_ollama(
            context,
            base_url=self.ollama_url,
            model=self.ollama_model,
        )

    def _rule_based_summary(
        self,
        snapshots: list[dict[str, Any]],
        *,
        pending_retrain: Optional[list] = None,
    ) -> tuple[str, list[str]]:
        """Heuristic English analysis when LLM is offline — not a template of labels only."""
        pending_retrain = pending_retrain or []
        pending_by_tab: dict[str, list] = {}
        for sig in pending_retrain:
            tid = str(sig.get("tab_id") or "")
            pending_by_tab.setdefault(tid, []).append(sig)

        attention: list[tuple[int, str]] = []  # priority, line
        if not snapshots:
            return "No monitoring tabs are loaded.", ["No monitoring tabs are loaded."]

        for snap in snapshots:
            title = snap.get("title") or snap.get("tab_id") or "?"
            health = str(snap.get("health_state") or "Idle")
            health_l = health.lower()
            drift = snap.get("drift")
            fusion = snap.get("fusion_score")
            alerts = int(snap.get("alert_count") or 0)
            obs_ok = snap.get("obs_ok", True)
            last_file = snap.get("last_file") or "—"
            monitoring = bool(snap.get("monitoring_active"))
            trained = int(snap.get("trained_models") or 0)
            tid = str(snap.get("tab_id") or "")
            pri = 0
            reasons: list[str] = []

            if not monitoring and health_l in ("idle", ""):
                pri = max(pri, 40)
                reasons.append(
                    "monitoring is Inactive/Idle — this is not active surveillance; "
                    "start monitoring or check the data folder/scheduler"
                )
            elif health_l not in ("idle", "nominal", "ok", "normal", "healthy", ""):
                pri = max(pri, 70)
                reasons.append(f"health is {health}")

            if drift is True or (
                isinstance(fusion, (int, float)) and float(fusion) >= 0.35
            ):
                pri = max(pri, 80)
                score = fusion if fusion is not None else "flagged"
                reasons.append(f"drift elevated ({score}) — consider retrain review")

            if alerts > 0:
                pri = max(pri, 60)
                reasons.append(f"{alerts} alert(s) in snapshot")

            if obs_ok is False:
                pri = max(pri, 75)
                reasons.append(f"OBS limit violation ({snap.get('obs_violations', '?')})")

            if last_file in ("—", "", None) and monitoring:
                pri = max(pri, 50)
                reasons.append("no last_file recorded while monitoring claims active — feed may be stalled")

            if trained == 0 and monitoring:
                pri = max(pri, 45)
                reasons.append("zero trained models — predictions may be unreliable")

            if pending_by_tab.get(tid):
                pri = max(pri, 85)
                reasons.append("pending retrain signal exists for this tab")

            if not reasons:
                pri = 5
                reasons.append(
                    f"no elevated flags (health={health}, drift={drift}, alerts={alerts}, "
                    f"obs_ok={obs_ok}, last_file={last_file})"
                )

            line = f"Tab {title}: " + "; ".join(reasons) + "."
            attention.append((pri, line))

        attention.sort(key=lambda x: -x[0])
        lines = [ln for _p, ln in attention]
        if attention:
            top_title = attention[0][1].split(":", 1)[0].replace("Tab ", "")
            if attention[0][0] >= 40:
                lines.insert(
                    0,
                    f"Priority: focus on {top_title} first "
                    f"(highest attention score among {len(snapshots)} tabs).",
                )
            else:
                lines.insert(
                    0,
                    f"Fleet looks quiet across {len(snapshots)} tabs; "
                    "confirm feeds are intentional Idle vs stalled.",
                )
        if pending_retrain:
            lines.append(f"Fleet-wide: {len(pending_retrain)} pending retrain signal(s).")

        # Mark as heuristic so operators know this is not LLM reasoning
        lines.append(
            "[Heuristic mode — install/start Ollama + qwen3:8b for real LLM analysis.]"
        )
        return "\n".join(lines), lines
