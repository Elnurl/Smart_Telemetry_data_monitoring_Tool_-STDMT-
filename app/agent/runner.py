"""AgentRunner — Phase 0–3: context gather + function calling + propose drafts."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from app.agent.audit import AgentAuditLog
from app.agent.bridge import ToolHost
from app.agent.function_calling import run_function_calling
from app.agent.policy import DEFAULT_ALLOW_LLM, is_loopback_url
from app.agent.tools import get_all_snapshots, get_pending_signals, get_tab_history, get_tab_snapshot

logger = logging.getLogger("STDMS.Agent.Runner")

CYCLE_PROMPT = (
    "Periodic fleet monitor cycle. Inspect tabs with tools. "
    "If anomaly/drift/OBS issues warrant action, use propose_retrain or propose_alert "
    "(pending human approval only). "
    "Prefer explain_anomaly / detect_patterns / forecast_risk on elevated tabs. "
    "Use get_pending_actions_summary and list_watchlist for operator briefing. "
    "Structure the final briefing as [Observation] / [Analysis] / [Recommendation]."
)


class AgentRunner:
    def __init__(
        self,
        audit: AgentAuditLog,
        tool_host_getter: Callable[[], Optional[ToolHost]],
        *,
        allow_llm: bool = DEFAULT_ALLOW_LLM,
        model: str = "",
        llm_host: str = "http://127.0.0.1:11434",
    ):
        self.audit = audit
        self.tool_host_getter = tool_host_getter
        self.allow_llm = bool(allow_llm) and is_loopback_url(llm_host)
        self.model = model or "qwen3:8b"
        self.llm_host = llm_host
        if allow_llm and not self.allow_llm:
            logger.warning("LLM requested but host is not loopback — LLM disabled")

    def run_agent_cycle(self, *, task: str = "fleet_monitor") -> dict[str, Any]:
        """Phase 3 monitor cycle: FC tool chain → audit → dashboard payload."""
        host = self.tool_host_getter()
        if host is None:
            decision_id = self.audit.record(
                tab_id=None,
                context_summary="tool host not attached",
                outcome="error: tool host not attached",
            )
            return {
                "ok": False,
                "error": "tool_host_not_attached",
                "decision_id": decision_id,
                "summary": "Agent: tool host is not attached.",
                "lines": [],
                "llm_used": False,
                "tool_trace": [],
                "phase": 3,
            }

        tabs = get_all_snapshots(host=host)
        pending = get_pending_signals(host=host)
        context = {"tabs": tabs, "pending_retrain": pending.get("retrain_signals") or []}
        context_summary = self._summarize_context(context, tab_id=None)
        tool_trace: list = []
        llm_used = False
        reasoning = None
        outcome = "observation_only"
        proactive_results: list = []

        # Faza C: rule-based proactive drafts (deduped) even without chat / LLM
        try:
            from app.agent.proactive import execute_proactive_actions, proactive_actions_from_tabs

            actions = proactive_actions_from_tabs(tabs, host=host)
            if actions:
                proactive_results = execute_proactive_actions(host, actions)
                for pr in proactive_results:
                    tool_trace.append(
                        {
                            "tool": pr.get("tool") or "proactive",
                            "params": {"tab_id": pr.get("tab_id"), "source": "cycle"},
                            "ok": bool(pr.get("ok")),
                        }
                    )
        except Exception as exc:
            logger.warning("proactive cycle actions failed: %s", exc)

        user_prompt = (
            f"{CYCLE_PROMPT}\n"
            f"Task: {task}\n"
            f"Seed context summary: {context_summary}\n"
            f"Pending retrain: {pending.get('pending_retrain_count', 0)}; "
            f"pending drafts: {pending.get('pending_draft_count', 0)}.\n"
            f"Proactive drafts this tick: {sum(1 for r in proactive_results if r.get('ok'))}.\n"
            "Use get_fleet_status / run_anomaly_check / check_drift / explain_anomaly as needed."
        )

        if self.allow_llm:
            fc = run_function_calling(
                user_prompt,
                host=host,
                ollama_url=self.llm_host,
                model=self.model,
                include_mutating=True,
            )
            tool_trace = tool_trace + (fc.get("tool_trace") or [])
            llm_used = bool(fc.get("llm_used"))
            reasoning = fc.get("reply")
            outcome = fc.get("outcome") or "observation_only"

        if not reasoning:
            reasoning = self._rule_based_note(
                task=task, context_summary=context_summary, context=context
            )
            if any(r.get("ok") for r in proactive_results):
                reasoning += (
                    "\n[Proactive] Created pending draft(s) for elevated tabs — "
                    "review Dashboard → Pending Agent Drafts (Approve required)."
                )
                outcome = "proactive_drafts"
            else:
                llm_used = False
                outcome = "observation_only"

        try:
            from app.agent.rag.retrieve import knowledge_status as ks

            kstat = str((ks() or {}).get("status") or "").lower()
            if kstat in ("empty", "unavailable", "error"):
                reasoning = (
                    str(reasoning)
                    + "\n[Knowledge] Index not ready — after SOP/doc Approve, "
                    "Rebuild Knowledge Index to refresh RAG."
                )
        except Exception:
            pass

        # Refresh pending counts after propose_* may have run
        try:
            pending_after = get_pending_signals(host=host)
        except Exception:
            pending_after = pending

        decision_id = self.audit.record(
            tab_id=None,
            context_summary=context_summary[:2000],
            tool_called=tool_trace[-1]["tool"] if tool_trace else "agent_monitor_cycle",
            tool_params={
                "task": task,
                "phase": 3,
                "tool_trace": tool_trace,
                "pending_drafts": pending_after.get("pending_draft_count"),
                "proactive_count": len(proactive_results),
            },
            reasoning=str(reasoning)[:8000],
            outcome=str(outcome),
        )

        lines = [ln.strip() for ln in str(reasoning).splitlines() if ln.strip()]
        return {
            "ok": True,
            "decision_id": decision_id,
            "summary": str(reasoning).strip(),
            "lines": lines,
            "context_summary": context_summary,
            "reasoning": reasoning,
            "llm_used": llm_used,
            "tool_trace": tool_trace,
            "pending_draft_count": pending_after.get("pending_draft_count", 0),
            "pending_retrain_count": pending_after.get("pending_retrain_count", 0),
            "proactive_results": proactive_results,
            "air_gap": True,
            "phase": 3,
            "outcome": outcome,
            "tab_count": len(tabs),
        }

    def run(self, *, tab_id: Optional[str] = None, task: str = "summarize") -> dict[str, Any]:
        """Function-calling when Ollama is up; else observation heuristic."""
        if tab_id is None and task in ("summarize", "fleet_monitor", "monitor"):
            # Prefer full Phase 3 cycle for fleet-wide tasks
            cycle = self.run_agent_cycle(task=task if task != "summarize" else "fleet_monitor")
            if cycle.get("ok"):
                return {
                    "ok": True,
                    "decision_id": cycle.get("decision_id"),
                    "tab_id": None,
                    "task": task,
                    "context_summary": cycle.get("context_summary"),
                    "reasoning": cycle.get("reasoning") or cycle.get("summary"),
                    "llm_used": cycle.get("llm_used"),
                    "tool_trace": cycle.get("tool_trace") or [],
                    "air_gap": True,
                    "phase": 3,
                    "outcome": cycle.get("outcome"),
                }

        host = self.tool_host_getter()
        if host is None:
            decision_id = self.audit.record(
                tab_id=tab_id,
                context_summary="tool host not attached",
                tool_called=None,
                reasoning="",
                outcome="error: tool host not attached",
            )
            return {"ok": False, "error": "tool_host_not_attached", "decision_id": decision_id}

        if tab_id:
            snapshot = get_tab_snapshot(tab_id, host=host)
            if snapshot is None:
                decision_id = self.audit.record(
                    tab_id=tab_id,
                    context_summary="tab not found",
                    outcome="error: tab_not_found",
                )
                return {"ok": False, "error": "tab_not_found", "decision_id": decision_id}
            history = get_tab_history(tab_id, n=20, host=host)
            context = {
                "tab_id": tab_id,
                "snapshot": snapshot,
                "history": history,
                "pending_retrain": host.get_pending_retrain_signals(),
            }
        else:
            tabs = get_all_snapshots(host=host)
            context = {
                "tabs": tabs,
                "pending_retrain": host.get_pending_retrain_signals(),
            }

        context_summary = self._summarize_context(context, tab_id=tab_id)
        tool_trace: list = []
        llm_used = False
        reasoning = None
        outcome = "observation_only"

        if self.allow_llm:
            user_prompt = (
                f"Task: {task}\n"
                + (f"Focus tab_id: {tab_id}\n" if tab_id else "Scope: all monitoring tabs\n")
                + f"Seed context summary: {context_summary}\n"
                "Use tools if you need fresher or more detailed live data, then answer."
            )
            fc = run_function_calling(
                user_prompt,
                host=host,
                ollama_url=self.llm_host,
                model=self.model,
                include_mutating=True,
            )
            tool_trace = fc.get("tool_trace") or []
            llm_used = bool(fc.get("llm_used"))
            reasoning = fc.get("reply")
            outcome = fc.get("outcome") or "observation_only"

        if not reasoning:
            reasoning = self._rule_based_note(
                task=task, context_summary=context_summary, context=context
            )
            llm_used = False
            outcome = "observation_only"

        tool_called = tool_trace[-1]["tool"] if tool_trace else (
            "get_tab_snapshot" if tab_id else "get_all_snapshots"
        )
        decision_id = self.audit.record(
            tab_id=tab_id,
            context_summary=context_summary,
            tool_called=tool_called,
            tool_params={
                "task": task,
                "tab_id": tab_id,
                "phase": 3,
                "tool_trace": tool_trace,
            },
            reasoning=str(reasoning)[:8000],
            outcome=str(outcome),
        )

        return {
            "ok": True,
            "decision_id": decision_id,
            "tab_id": tab_id,
            "task": task,
            "context_summary": context_summary,
            "reasoning": reasoning,
            "llm_used": llm_used,
            "tool_trace": tool_trace,
            "air_gap": True,
            "phase": 3,
            "outcome": outcome,
        }

    def ask(self, user_message: str) -> dict[str, Any]:
        """Interactive ask via sequential multi-LLM nodes (R → M → RA|C)."""
        host = self.tool_host_getter()
        tool_trace: list = []
        reply = None
        llm_used = False
        outcome = "observation_only"
        agent_mode = "offline"
        route = None
        node = None
        log_summary = ""
        node_trace: list = []

        # Fast path: write-SOP intent still uses dedicated helper (before routing)
        try:
            from app.agent.function_calling import _is_write_document_intent
            from app.agent.tools import propose_write_sop_from_user_text

            if host is not None and _is_write_document_intent(user_message or ""):
                auto = propose_write_sop_from_user_text(user_message or "", host=host)
                tool_trace.append(
                    {
                        "tool": "propose_write_sop",
                        "params": {"auto_from_user_text": True},
                        "ok": bool(auto.get("ok")),
                    }
                )
                if auto.get("ok"):
                    reply = (
                        "[R-LLM] routing → tool\n"
                        "[RA-LLM] tools used: propose_write_sop✓\n"
                        "Created a pending SOP Word draft for human Approve.\n"
                        f"- draft_id: {auto.get('draft_id')}\n"
                        f"- kind: {auto.get('kind')}\n"
                        "- Next: Home → AI Assistant → Pending Agent Drafts → Approve"
                    )
                    outcome = "function_calling"
                    agent_mode = "RA-LLM"
                    route = "tool"
                    node = "RA-LLM"
        except Exception as exc:
            logger.warning("SOP ask shortcut failed: %s", exc)

        if reply is None:
            try:
                from app.agent.log_monitor import get_log_monitor
                from app.agent.nodes import MultiNodePipeline
                from app.agent.ollama_client import resolve_chat_model

                model = self.model
                if self.allow_llm:
                    model = resolve_chat_model(self.model, base_url=self.llm_host)
                    self.model = model
                monitor = get_log_monitor(
                    ollama_url=self.llm_host,
                    ollama_model=model,
                    allow_llm=self.allow_llm,
                )
                pipeline = MultiNodePipeline(
                    ollama_url=self.llm_host,
                    model=model,
                    allow_llm=self.allow_llm,
                    log_monitor=monitor,
                )
                result = pipeline.run(user_message or "", host=host)
                reply = result.reply
                tool_trace = result.tool_trace
                llm_used = bool(result.llm_used)
                outcome = result.outcome
                agent_mode = result.agent_mode
                route = result.route
                node = result.node
                log_summary = result.log_summary
                node_trace = list(result.node_trace or [])
            except Exception as exc:
                logger.warning("Multi-node ask failed, falling back: %s", exc)
                reply = None

        if reply is None and host is not None:
            # Legacy offline create-tab / best-model helpers
            try:
                from app.agent.function_calling import (
                    _deterministic_best_model,
                    _deterministic_create_tab,
                    _is_best_model_intent,
                    _is_create_tab_intent,
                    synthesize_fleet_answer,
                )

                if _is_create_tab_intent(user_message or ""):
                    det = _deterministic_create_tab(user_message or "", host)
                    if det and det.get("reply"):
                        reply = (
                            "[R-LLM] routing → tool\n[RA-LLM]\n" + str(det.get("reply"))
                        )
                        tool_trace = det.get("tool_trace") or []
                        outcome = det.get("outcome") or "function_calling"
                        agent_mode = "RA-LLM"
                        route = "tool"
                        node = "RA-LLM"
                elif _is_best_model_intent(user_message or ""):
                    det = _deterministic_best_model(user_message or "", host)
                    if det and det.get("reply"):
                        reply = (
                            "[R-LLM] routing → tool\n[RA-LLM]\n" + str(det.get("reply"))
                        )
                        tool_trace = det.get("tool_trace") or []
                        outcome = det.get("outcome") or "function_calling"
                        agent_mode = "RA-LLM"
                        route = "tool"
                        node = "RA-LLM"
                else:
                    tabs = get_all_snapshots(host=host)
                    context = {
                        "tabs": tabs,
                        "pending_retrain": host.get_pending_retrain_signals(),
                    }
                    summary = self._summarize_context(context, tab_id=None)
                    reply = (
                        "[R-LLM] routing → tool\n[RA-LLM]\n"
                        + self._rule_based_note(
                            task="ask", context_summary=summary, context=context
                        )
                    )
                    agent_mode = "RA-LLM"
                    route = "tool"
                    node = "RA-LLM"
                    if not reply:
                        reply = synthesize_fleet_answer(user_message or "", host)
            except Exception as exc:
                logger.warning("ask fallback failed: %s", exc)
                reply = "Agent ask failed; see application log."
                outcome = "error"
                agent_mode = "offline"
        elif reply is None:
            reply = "Tool host is not attached."
            outcome = "error: tool_host_not_attached"

        decision_id = self.audit.record(
            tab_id=None,
            context_summary=f"chat: {(user_message or '')[:500]}",
            tool_called=tool_trace[-1]["tool"] if tool_trace else "agent_chat",
            tool_params={
                "phase": 3,
                "llm_used": llm_used,
                "agent_mode": agent_mode,
                "route": route,
                "node": node,
                "tool_trace": tool_trace,
                "node_trace": node_trace,
            },
            reasoning=str(reply)[:8000],
            outcome=str(outcome),
        )
        return {
            "ok": True,
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
        }

    def _summarize_context(self, context: dict[str, Any], *, tab_id: Optional[str]) -> str:
        if tab_id:
            snap = context.get("snapshot") or {}
            return (
                f"Tab {tab_id} ({snap.get('title', '?')}): "
                f"health={snap.get('health_state', '?')}, "
                f"drift={snap.get('drift')}, "
                f"obs_ok={snap.get('obs_ok')}, "
                f"alerts={snap.get('alert_count', 0)}, "
                f"trained={snap.get('trained_models', 0)}, "
                f"updated={snap.get('updated_at', '?')}"
            )

        tabs = context.get("tabs") or []
        parts = []
        for entry in tabs[:20]:
            snap = entry.get("snapshot") or {}
            parts.append(
                f"{entry.get('tab_id')}:{snap.get('health_state', '?')}"
                f"(drift={snap.get('drift')},obs={snap.get('obs_ok')})"
            )
        pending = context.get("pending_retrain") or []
        return f"{len(tabs)} tab(s): " + (", ".join(parts) if parts else "none") + f"; pending_retrain={len(pending)}"

    def _rule_based_note(
        self, *, task: str, context_summary: str, context: dict[str, Any]
    ) -> str:
        flags: list[str] = []
        entries = []
        if "snapshot" in context:
            entries = [{"snapshot": context["snapshot"]}]
        else:
            entries = context.get("tabs") or []

        for entry in entries:
            snap = entry.get("snapshot") or entry
            title = snap.get("title") or snap.get("tab_id") or "?"
            if snap.get("drift"):
                flags.append(f"{title}: drift detected")
            if snap.get("obs_ok") is False:
                flags.append(f"{title}: OBS violation")
            health = str(snap.get("health_state") or "")
            if health and health.lower() not in ("idle", "nominal", "ok", "normal", "healthy", ""):
                flags.append(f"{title}: health={health}")
            if not snap.get("monitoring_active") and str(health).lower() in ("idle", ""):
                flags.append(f"{title}: Idle / not actively monitoring")

        pending = context.get("pending_retrain") or []
        if pending:
            flags.append(f"{len(pending)} pending retrain signal(s)")

        attention = "; ".join(flags) if flags else "no elevated flags in current snapshots"
        return (
            f"[Observation] {context_summary}\n"
            f"[Analysis] Operator attention: {attention}. "
            f"(heuristic / Ollama offline; task={task})\n"
            "[Recommendation] Start Ollama (qwen3:8b) for tool-chained analysis; "
            "review Warning/drift/OBS tabs first. No data left this host."
        )
