"""Multi-LLM nodes — sequential roles on one local model (8GB VRAM safe).

R-LLM routes → M-LLM summarizes logs → RA-LLM (tools) or C-LLM (knowledge).
Only one Ollama model is loaded; nodes differ by system prompt + call order.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from app.agent.prompts import CLLM_PROMPT, MLLM_PROMPT, RALLM_PROMPT, RLLM_PROMPT

logger = logging.getLogger("STDMS.Agent.Nodes")

Route = Literal["tool", "knowledge"]

_TOOL_HINTS = (
    "create tab",
    "create a tab",
    "new tab",
    "train",
    "propose",
    "start monitor",
    "start monitoring",
    "stop monitor",
    "fleet",
    "which tab",
    "needs attention",
    "alert",
    "drift",
    "inspect",
    "best model",
    "compare model",
    "watchlist",
    "list tabs",
    "status of",
    "run anomaly",
    "approve",
    "remove model",
    "from data/",
    ".csv",
)

_KNOWLEDGE_HINTS = (
    "procedure",
    "sop",
    "what is",
    "what are",
    "how does",
    "how do",
    "hardware",
    "software",
    "manual",
    "documentation",
    "runbook",
    "why is",
    "why are",
    "why does",
    "mode-normal",
    "eclipse procedure",
    "according to",
    "in the sop",
    "knowledge",
)


@dataclass
class AgentDecision:
    reply: str
    route: Route = "tool"
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    llm_used: bool = False
    outcome: str = "observation_only"
    node: str = "RA-LLM"
    log_summary: str = ""
    fsm_mode: str = "nominal"
    threshold_scale: float = 1.0


@dataclass
class RALLMContext:
    query: str
    snapshot: dict[str, Any] | list[Any] | str = field(default_factory=dict)
    log_summary: str = ""
    fsm_mode: str = "nominal"
    scale: float = 1.0
    rag_context: list[dict[str, Any]] = field(default_factory=list)


def heuristic_route(query: str) -> Route:
    """Deterministic router used offline and as LLM fallback."""
    q = (query or "").strip().lower()
    if not q:
        return "tool"
    knowledge_score = sum(1 for h in _KNOWLEDGE_HINTS if h in q)
    tool_score = sum(1 for h in _TOOL_HINTS if h in q)
    # "why is … warnings/anomalies" is knowledge (FSM / SOP explanation)
    if knowledge_score > tool_score:
        return "knowledge"
    if tool_score > 0:
        return "tool"
    if any(w in q for w in ("why", "what", "how", "procedure", "sop", "explain")):
        return "knowledge"
    return "tool"


def _parse_route_token(text: str) -> Optional[Route]:
    raw = (text or "").strip().lower()
    if not raw:
        return None
    # Prefer first token / word
    for token in re.split(r"[\s,.:;]+", raw):
        if token in ("tool", "tools", "action", "monitor"):
            return "tool"
        if token in ("knowledge", "rag", "docs", "sop"):
            return "knowledge"
    if "knowledge" in raw and "tool" not in raw:
        return "knowledge"
    if "tool" in raw:
        return "tool"
    return None


def _format_rag(hits: list[dict[str, Any]], *, limit: int = 6) -> str:
    if not hits:
        return "(no RAG hits)"
    parts: list[str] = []
    for hit in hits[:limit]:
        title = hit.get("title") or hit.get("source") or "doc"
        snippet = (hit.get("snippet") or hit.get("text") or "").strip()
        parts.append(f"- {title}: {snippet[:400]}")
    return "\n".join(parts)


def _chat(
    *,
    system: str,
    user: str,
    ollama_url: str,
    model: str,
    num_predict: int = 200,
    temperature: float = 0.1,
) -> Optional[str]:
    try:
        from app.agent.ollama_client import chat_ollama, ollama_reachable, resolve_chat_model
        from app.agent.policy import is_loopback_url
    except Exception:
        return None
    if not is_loopback_url(ollama_url) or not ollama_reachable(ollama_url):
        return None
    resolved = resolve_chat_model(model, base_url=ollama_url)
    try:
        msg = chat_ollama(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=None,
            base_url=ollama_url,
            model=resolved,
            num_predict=num_predict,
            temperature=temperature,
        )
    except Exception as exc:
        logger.info("node chat failed: %s", exc)
        return None
    if not msg:
        return None
    return (msg.get("content") or "").strip() or None


class RLLMNode:
    """Routing node — returns only tool | knowledge."""

    def __init__(
        self,
        *,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        allow_llm: bool = True,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.allow_llm = bool(allow_llm)

    def route(self, query: str) -> Route:
        heuristic = heuristic_route(query)
        q = (query or "").strip().lower()
        # Strong intents must not be overturned by a chatty router model
        strong_knowledge = any(
            h in q
            for h in (
                "why is",
                "why are",
                "why does",
                "procedure",
                "sop",
                "what is the",
                "what are the",
                "mode-normal",
            )
        )
        strong_tool = any(
            h in q
            for h in (
                "create tab",
                "create a tab",
                "which tab",
                "needs attention",
                "from data/",
                ".csv",
                "train model",
                "start monitoring",
            )
        )
        if strong_knowledge and not strong_tool:
            return "knowledge"
        if strong_tool and not strong_knowledge:
            return "tool"

        if self.allow_llm:
            text = _chat(
                system=RLLM_PROMPT,
                user=f"Query: {query}\nReply with exactly one word: tool or knowledge.",
                ollama_url=self.ollama_url,
                model=self.model,
                num_predict=16,
                temperature=0.0,
            )
            parsed = _parse_route_token(text or "")
            if parsed:
                return parsed
        return heuristic


class MLLMNode:
    """Log monitor node — FSM-aware log window summary."""

    def __init__(
        self,
        *,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        allow_llm: bool = True,
        log_monitor: Any = None,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.allow_llm = bool(allow_llm)
        self.log_monitor = log_monitor

    def summarize(
        self,
        log_analysis: Any = None,
        fsm_state: dict[str, Any] | None = None,
        *,
        tab_id: str | None = None,
    ) -> str:
        fsm_state = fsm_state or {}
        mode = str(fsm_state.get("mode") or fsm_state.get("name") or "nominal")
        scale = float(fsm_state.get("threshold_scale") or fsm_state.get("scale") or 1.0)

        analysis = log_analysis
        if analysis is None and tab_id and self.log_monitor is not None:
            analysis = self.log_monitor.get_last_analysis(tab_id)
            if analysis is None and hasattr(self.log_monitor, "get_window"):
                window = self.log_monitor.get_window(tab_id)
                if window:
                    try:
                        analysis = self.log_monitor.analyze(
                            tab_id,
                            tab_config={
                                "current_mission_mode": mode,
                                "mission_modes": [{"name": mode, "threshold_scale": scale}],
                            },
                            apply_transitions=False,
                            use_llm=False,
                        )
                    except Exception as exc:
                        logger.warning("M-LLM analyze failed: %s", exc)

        if analysis is None:
            return f"No log window for mode={mode} (×{scale:g})."

        base = getattr(analysis, "summary", None) or ""
        if isinstance(analysis, dict):
            base = str(analysis.get("summary") or "")
        if not base:
            filtered = getattr(analysis, "false_positives_filtered", None)
            if filtered is None and isinstance(analysis, dict):
                filtered = analysis.get("false_positives_filtered", 0)
            anomalies = getattr(analysis, "anomalies", None)
            if anomalies is None and isinstance(analysis, dict):
                anomalies = analysis.get("anomalies") or []
            base = (
                f"Mode={mode} (×{scale:g}). "
                f"{int(filtered or 0)} mode-normal filtered; "
                f"{len(anomalies or [])} true anomal(y/ies)."
            )

        if not self.allow_llm:
            return base

        window_size = getattr(analysis, "window_size", None)
        if window_size is None and isinstance(analysis, dict):
            window_size = analysis.get("window_size", 0)
        system = MLLM_PROMPT.format(
            window_size=int(window_size or 0),
            current_mode=mode,
            scale=scale,
        )
        enriched = _chat(
            system=system,
            user=(
                f"FSM state: {mode} (scale={scale:g})\n"
                f"Rule summary: {base}\n"
                "Rewrite as a concise English operator line for the dashboard."
            ),
            ollama_url=self.ollama_url,
            model=self.model,
            num_predict=120,
        )
        return enriched or base


class RALLMNode:
    """Reasoning + Action — tool-calling decision node."""

    def __init__(
        self,
        *,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        allow_llm: bool = True,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.allow_llm = bool(allow_llm)

    def reason(self, context: RALLMContext, *, host: Any = None) -> AgentDecision:
        snap_txt = context.snapshot
        if not isinstance(snap_txt, str):
            snap_txt = str(snap_txt)[:2500]
        rag_txt = _format_rag(context.rag_context)
        system = RALLM_PROMPT.format(
            snapshot=snap_txt,
            log_summary=context.log_summary or "(none)",
            fsm_mode=context.fsm_mode,
            scale=context.scale,
            rag_context=rag_txt,
        )
        # Append tool policy from main system prompt builder (mutators allowed)
        try:
            from app.agent.prompts import build_system_prompt

            system = system + "\n\n" + build_system_prompt(include_mutating=True)
        except Exception:
            pass

        user = (
            f"Operator: {context.query}\n\n"
            f"[FSM] mode={context.fsm_mode} scale={context.scale:g}\n"
            f"[M-LLM] {context.log_summary or '(no logs)'}\n"
            "If logs say mode-normal, do not treat those as anomalies.\n"
            "Use tools when needed, then answer with Observation / Analysis / Recommendation."
        )

        if host is not None and self.allow_llm:
            from app.agent.function_calling import run_function_calling

            fc = run_function_calling(
                user,
                host=host,
                ollama_url=self.ollama_url,
                model=self.model,
                include_mutating=True,
                system_prompt=system,
            )
            return AgentDecision(
                reply=str(fc.get("reply") or "").strip() or self._offline_reason(context),
                route="tool",
                tool_trace=list(fc.get("tool_trace") or []),
                llm_used=bool(fc.get("llm_used")),
                outcome=str(fc.get("outcome") or "function_calling"),
                node="RA-LLM",
                log_summary=context.log_summary,
                fsm_mode=context.fsm_mode,
                threshold_scale=context.scale,
            )

        # Offline / no host: deterministic FSM-aware note
        return AgentDecision(
            reply=self._offline_reason(context),
            route="tool",
            tool_trace=[],
            llm_used=False,
            outcome="observation_only",
            node="RA-LLM",
            log_summary=context.log_summary,
            fsm_mode=context.fsm_mode,
            threshold_scale=context.scale,
        )

    @staticmethod
    def _offline_reason(context: RALLMContext) -> str:
        mode = (context.fsm_mode or "nominal").lower()
        log = (context.log_summary or "").lower()
        mode_normal = (
            "mode-normal" in log
            or "filtered" in log
            or (mode == "eclipse" and "warning" in log and "cdh" not in log)
        )
        if mode == "eclipse" and mode_normal:
            analysis = (
                "Eclipse mode (elevated threshold_scale) explains thermal/TCS warnings "
                "as mode-normal; do not escalate those as anomalies."
            )
            rec = "Keep monitoring; escalate only true anomalies (e.g. CDH ERROR) or Approve pending drafts."
        elif mode == "nominal" and ("warning" in log or "anomal" in log):
            analysis = (
                "Nominal mode has no eclipse tolerance — the same warning pattern "
                "should be treated as an anomaly candidate."
            )
            rec = "Run explain_anomaly / check_drift on the tab; propose_alert if sustained."
        else:
            analysis = f"FSM mode={mode} (×{context.scale:g}); log context applied."
            rec = "Use Analyze Fleet or Approve pending drafts if actions were proposed."
        return (
            f"[Observation] Query: {context.query}. Mode={context.fsm_mode} "
            f"(×{context.scale:g}). M-LLM: {context.log_summary or 'n/a'}\n"
            f"[Analysis] {analysis}\n"
            f"[Recommendation] {rec}"
        )


class CLLMNode:
    """Knowledge node — RAG + FSM/log context, no tool calling."""

    MISSING = "Bu məlumat knowledge bazasında yoxdur"

    def __init__(
        self,
        *,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        allow_llm: bool = True,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.allow_llm = bool(allow_llm)

    def answer(
        self,
        query: str,
        rag_context: list[dict[str, Any]] | None = None,
        fsm_context: dict[str, Any] | None = None,
        *,
        log_summary: str = "",
        host: Any = None,
    ) -> str:
        fsm_context = fsm_context or {}
        # None → retrieve; explicit list (even empty) → trust caller (tests / pipeline)
        if rag_context is None:
            hits = self._retrieve(query, host=host)
        else:
            hits = list(rag_context)

        mode = str(fsm_context.get("mode") or fsm_context.get("name") or "nominal")
        scale = float(fsm_context.get("threshold_scale") or fsm_context.get("scale") or 1.0)
        rag_txt = _format_rag(hits)

        # FSM/log can still answer eclipse false-positive questions without RAG
        log_l = (log_summary or "").lower()
        q_l = (query or "").lower()
        fsm_explains = (
            mode == "eclipse"
            and any(k in q_l for k in ("warning", "anomal", "eclipse", "tcs", "temp"))
            and ("mode-normal" in log_l or "eclipse" in log_l or "filtered" in log_l)
        )

        if not hits and not fsm_explains:
            return self.MISSING

        if not self.allow_llm or not hits and fsm_explains:
            return self._offline_answer(query, hits, mode, scale, log_summary, fsm_explains)

        system = CLLM_PROMPT.format(
            rag_context=rag_txt,
            fsm_mode=mode,
            log_summary=log_summary or "(none)",
        )
        text = _chat(
            system=system,
            user=f"Operator question: {query}",
            ollama_url=self.ollama_url,
            model=self.model,
            num_predict=400,
        )
        if text:
            return text
        return self._offline_answer(query, hits, mode, scale, log_summary, fsm_explains)

    def _retrieve(
        self,
        query: str,
        *,
        host: Any = None,
        segment: str = "auto",
    ) -> list[dict[str, Any]]:
        try:
            from app.agent.tools import search_knowledge

            result = search_knowledge(query, host=host, segment=segment or "auto")
            return list(result.get("hits") or [])
        except Exception as exc:
            logger.warning("C-LLM retrieve failed: %s", exc)
            return []

    def _offline_answer(
        self,
        query: str,
        hits: list[dict[str, Any]],
        mode: str,
        scale: float,
        log_summary: str,
        fsm_explains: bool,
    ) -> str:
        parts: list[str] = []
        if fsm_explains:
            parts.append(
                f"Eclipse entry is reflected in logs; TCS warnings under mode={mode} "
                f"(×{scale:g}) are mode-normal, not true anomalies. "
                f"M-LLM: {log_summary or 'n/a'}"
            )
        for hit in hits[:3]:
            title = hit.get("title") or hit.get("source") or "source"
            snippet = (hit.get("snippet") or "").strip()
            if snippet:
                parts.append(f"From {title}: {snippet[:500]}")
        if not parts:
            return self.MISSING
        return "\n".join(parts)


@dataclass
class MultiNodeResult:
    reply: str
    route: Route
    node: str
    tool_trace: list[dict[str, Any]]
    llm_used: bool
    outcome: str
    log_summary: str
    node_trace: list[str]
    agent_mode: str
    fsm_mode: str = "nominal"
    threshold_scale: float = 1.0


class MultiNodePipeline:
    """Sequential R → M → (RA | C) on a single local model."""

    def __init__(
        self,
        *,
        ollama_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        allow_llm: bool = True,
        log_monitor: Any = None,
    ):
        self.allow_llm = bool(allow_llm)
        self.rllm = RLLMNode(ollama_url=ollama_url, model=model, allow_llm=allow_llm)
        self.mllm = MLLMNode(
            ollama_url=ollama_url,
            model=model,
            allow_llm=allow_llm,
            log_monitor=log_monitor,
        )
        self.rallm = RALLMNode(ollama_url=ollama_url, model=model, allow_llm=allow_llm)
        self.cllm = CLLMNode(ollama_url=ollama_url, model=model, allow_llm=allow_llm)
        self.log_monitor = log_monitor

    def run(self, query: str, *, host: Any = None) -> MultiNodeResult:
        route = self.rllm.route(query)
        node_trace = [f"[R-LLM] routing → {route}"]

        fsm_mode, scale, snap, tab_id = self._fleet_fsm_context(host, query)
        log_summary = self.mllm.summarize(
            fsm_state={"mode": fsm_mode, "threshold_scale": scale},
            tab_id=tab_id,
        )
        node_trace.append(f"[M-LLM] {log_summary}")

        llm_used = False
        if route == "knowledge":
            hits = self.cllm._retrieve(query, host=host, segment="auto")
            answer = self.cllm.answer(
                query,
                rag_context=hits,
                fsm_context={"mode": fsm_mode, "threshold_scale": scale},
                log_summary=log_summary,
                host=host,
            )
            # If LLM path was attempted inside answer, treat as used when online
            if self.allow_llm and answer and answer != CLLMNode.MISSING:
                try:
                    from app.agent.ollama_client import ollama_reachable
                    from app.agent.policy import is_loopback_url

                    llm_used = is_loopback_url(self.rllm.ollama_url) and ollama_reachable(
                        self.rllm.ollama_url
                    )
                except Exception:
                    llm_used = False
            reply = "\n".join(node_trace + [f"[C-LLM]\n{answer}"])
            return MultiNodeResult(
                reply=reply,
                route="knowledge",
                node="C-LLM",
                tool_trace=[],
                llm_used=llm_used,
                outcome="knowledge",
                log_summary=log_summary,
                node_trace=node_trace + ["[C-LLM] answered"],
                agent_mode="C-LLM",
                fsm_mode=fsm_mode,
                threshold_scale=scale,
            )

        # Lightweight RAG seed for RA-LLM (optional context)
        rag_hits: list[dict[str, Any]] = []
        try:
            rag_hits = self.cllm._retrieve(query, host=host)[:4]
        except Exception:
            rag_hits = []

        decision = self.rallm.reason(
            RALLMContext(
                query=query,
                snapshot=snap,
                log_summary=log_summary,
                fsm_mode=fsm_mode,
                scale=scale,
                rag_context=rag_hits,
            ),
            host=host,
        )
        tools_bits = []
        for t in decision.tool_trace[:8]:
            name = t.get("tool") or "?"
            mark = "✓" if t.get("ok", True) else "✗"
            tools_bits.append(f"{name}{mark}")
        tools_line = (
            f"[RA-LLM] tools used: {' '.join(tools_bits)}"
            if tools_bits
            else "[RA-LLM] tools used: (none)"
        )
        node_trace.append(tools_line)
        body = decision.reply or ""
        # Avoid duplicating node headers if FC already included them
        if not body.lstrip().startswith("[R-LLM]"):
            reply = "\n".join(node_trace + [body])
        else:
            reply = body
        return MultiNodeResult(
            reply=reply,
            route="tool",
            node="RA-LLM",
            tool_trace=decision.tool_trace,
            llm_used=decision.llm_used,
            outcome=decision.outcome,
            log_summary=log_summary,
            node_trace=node_trace,
            agent_mode="RA-LLM",
            fsm_mode=fsm_mode,
            threshold_scale=scale,
        )

    def _fleet_fsm_context(
        self, host: Any, query: str
    ) -> tuple[str, float, Any, Optional[str]]:
        """Pick a relevant tab's FSM state + compact snapshot for nodes."""
        tabs: list[dict[str, Any]] = []
        if host is not None:
            try:
                from app.agent.tools import get_all_snapshots

                tabs = get_all_snapshots(host=host) or []
            except Exception:
                tabs = []

        q = (query or "").lower()
        chosen: dict[str, Any] | None = None
        for entry in tabs:
            title = str(entry.get("title") or "").lower()
            snap = entry.get("snapshot") or {}
            mode = str(snap.get("mission_mode") or "").lower()
            if title and title in q:
                chosen = entry
                break
            if mode and mode in q:
                chosen = entry
                break
        if chosen is None and tabs:
            # Prefer elevated health, else first
            for entry in tabs:
                health = str((entry.get("snapshot") or {}).get("health_state") or "").lower()
                if health in ("warning", "critical"):
                    chosen = entry
                    break
            chosen = chosen or tabs[0]

        if not chosen:
            return "nominal", 1.0, {"tabs": 0}, None

        snap = dict(chosen.get("snapshot") or {})
        snap.setdefault("tab_id", chosen.get("tab_id"))
        snap.setdefault("title", chosen.get("title"))
        mode = str(snap.get("mission_mode") or "nominal")
        try:
            scale = float(snap.get("threshold_scale") or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        # Compact fleet list for RA-LLM
        fleet_brief = [
            {
                "tab_id": t.get("tab_id"),
                "title": t.get("title"),
                "health": (t.get("snapshot") or {}).get("health_state"),
                "mission_mode": (t.get("snapshot") or {}).get("mission_mode"),
                "threshold_scale": (t.get("snapshot") or {}).get("threshold_scale"),
            }
            for t in tabs[:15]
        ]
        return mode, scale, {"focus": snap, "fleet": fleet_brief}, str(chosen.get("tab_id") or "") or None
