"""LLM function-calling loop over STDMS agent tools (Phase 2–3).

Uses Ollama ``/api/chat`` with tool definitions including propose_* drafts.
Falls back to plain generation / heuristics when Ollama is offline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from app.agent.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    call_ollama,
    chat_ollama,
    ollama_reachable,
)
from app.agent.policy import is_loopback_url
from app.agent.prompts import SYSTEM_PROMPT, build_system_prompt
from app.agent.tools import (
    invoke_tool,
    list_tools,
    to_ollama_tools,
    tool_result_json,
)

logger = logging.getLogger("STDMS.Agent.FunctionCalling")

MAX_TOOL_ROUNDS = 5

_KNOWLEDGE_HINTS = (
    "who ",
    "who is",
    "what is",
    "sop",
    "procedure",
    "profile",
    "document",
    "according to",
    "pdf",
    "how to",
    "how do i",
    "explain",
    "cv",
    "resume",
    "biography",
    "haqqında",
    "kimdir",
)
_FLEET_HINTS = (
    "which tab",
    "which tabs",
    "what tab",
    "fleet",
    "all tabs",
    "health report",
    "anomaly check",
    "pending retrain",
    "monitoring status",
    "eclipse",
    "monitor to the",
    "created to monitor",
)

_INFO_ASK_HINTS = (
    "which",
    "what",
    "list",
    "show",
    "tell me",
    "how many",
    "hansı",
    "hansi",
    "status",
)

_PROPOSE_TOOLS = frozenset(
    {
        "propose_create_tab",
        "propose_train",
        "propose_start_monitoring",
        "propose_stop_monitoring",
        "propose_remove_model",
        "propose_write_sop",
        "propose_update_sop",
        "propose_write_document",
        "propose_update_document",
        "propose_alert",
        "propose_retrain",
        "propose_model_plan",
    }
)
_WRITE_DOC_HINTS = (
    "create sop",
    "create procedure",
    "write sop",
    "write procedure",
    "new sop",
    "new procedure",
    "update sop",
    "update procedure",
    "crate sop",  # common typo
    "crate procedure",
    "make sop",
    "make procedure",
    "propose_write_sop",
    "sop procedure named",
    "procedure named",
)


def _is_write_document_intent(user_message: str) -> bool:
    m = (user_message or "").lower().strip()
    if not m:
        return False
    if any(h in m for h in _WRITE_DOC_HINTS):
        return True
    # "create ... sop-9" / "create ... procedure"
    if ("create" in m or "write" in m or "update" in m or "crate" in m) and (
        "sop" in m or "procedure" in m or "prosedur" in m
    ):
        return True
    return False


def _is_create_tab_intent(user_message: str) -> bool:
    """True when operator asks to create a monitoring tab (path optional)."""
    m = (user_message or "").lower().strip()
    if not m:
        return False
    if _is_write_document_intent(m):
        return False
    create_words = (
        "create a tab",
        "create tab",
        "create new tab",
        "new tab",
        "make a tab",
        "make new tab",
        "tab yarat",
        "tab yaratmaq",
        "propose_create_tab",
    )
    return any(w in m for w in create_words)


def _is_best_model_intent(user_message: str) -> bool:
    m = (user_message or "").lower().strip()
    if not m:
        return False
    if "model" not in m:
        return False
    return any(
        w in m
        for w in (
            "best",
            "which",
            "recommend",
            "compare",
            "hansı",
            "hansi",
            "ən yaxşı",
            "en yaxsi",
        )
    )


def _is_informational_ask(user_message: str) -> bool:
    """True for read-only questions (which tab / status) — not create/write/train."""
    m = (user_message or "").lower().strip()
    if not m:
        return False
    if _is_write_document_intent(m) or _is_create_tab_intent(m):
        return False
    if _is_best_model_intent(m):
        return True  # still informational (compare), but handled separately
    if any(h in m for h in _FLEET_HINTS):
        return True
    return any(h in m for h in _INFO_ASK_HINTS) and (
        "tab" in m or "fleet" in m or "monitor" in m or "eclipse" in m
    )


def _tab_lines_from_host(host: Any) -> list[dict[str, Any]]:
    try:
        tabs = host.list_tabs() or []
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for t in tabs:
        if not isinstance(t, dict):
            continue
        snap = t.get("snapshot") if isinstance(t.get("snapshot"), dict) else {}
        title = str(t.get("title") or snap.get("title") or t.get("tab_id") or "?")
        out.append(
            {
                "title": title,
                "tab_id": t.get("tab_id"),
                "health": snap.get("health_state") or "—",
                "monitoring": bool(snap.get("monitoring_active")),
                "trained": snap.get("trained_models"),
            }
        )
    return out


def synthesize_fleet_answer(user_message: str, host: Any) -> str:
    """Build a direct factual answer from live tabs (no Approve / draft nag)."""
    rows = _tab_lines_from_host(host)
    msg = (user_message or "").lower()
    if not rows:
        return (
            "[Observation] No monitoring tabs are currently open.\n"
            "[Analysis] There is nothing to match against your question.\n"
            "[Recommendation] Open or create a monitoring tab if you need one."
        )

    # Prefer title keyword matches from the question
    tokens = [
        w
        for w in re.findall(r"[a-z0-9]+", msg)
        if len(w) >= 4 and w not in {"which", "what", "that", "want", "know", "from", "with", "this", "tab", "tabs", "monitor", "created", "operation"}
    ]
    matched = []
    for row in rows:
        title_l = row["title"].lower()
        if any(tok in title_l for tok in tokens) or (
            "eclipse" in msg and "eclipse" in title_l
        ):
            matched.append(row)

    focus = matched or rows
    lines = []
    for r in focus:
        mon = "active" if r["monitoring"] else "idle"
        lines.append(
            f"- {r['title']} (health={r['health']}, monitoring={mon}, trained_models={r['trained']})"
        )

    if matched:
        obs = f"Matching tab(s) for your question ({len(matched)}):"
        analysis = (
            "These titles best match the keywords in your question "
            f"({', '.join(tokens[:6]) or 'eclipse/monitor'})."
        )
    else:
        obs = f"Open monitoring tabs ({len(rows)}):"
        analysis = (
            "No strong title match for the keywords; listing all open tabs so you can pick."
        )

    return (
        f"[Observation] {obs}\n"
        + "\n".join(lines)
        + f"\n[Analysis] {analysis}\n"
        "[Recommendation] No action needed unless you want to start/stop/train a specific tab."
    )


def _reply_after_tools(
    *,
    user_message: str,
    host: Any,
    tool_trace: list[dict[str, Any]],
    write_intent: bool,
) -> Optional[str]:
    """When the LLM returns no final text, answer from facts — not a draft nag."""
    propose_ok = [
        t
        for t in tool_trace
        if t.get("ok") and str(t.get("tool") or "") in _PROPOSE_TOOLS
    ]
    if write_intent and propose_ok:
        tid = propose_ok[-1].get("draft_id")
        return (
            f"[Observation] A pending document draft was created"
            + (f" (id={tid})" if tid else "")
            + ".\n"
            "[Analysis] Document writes stay Propose→Approve.\n"
            "[Recommendation] Review it under Home → AI Assistant → Pending Agent Drafts."
        )
    # Informational / fleet tools ran — answer from live tabs
    fleet_tools = {
        "get_fleet_status",
        "get_all_snapshots",
        "list_watchlist",
        "get_pending_actions_summary",
        "search_knowledge",
        "list_knowledge_documents",
        "read_knowledge_document",
        "explain_anomaly",
        "detect_patterns",
        "forecast_risk",
        "compare_tab_models",
        "list_tab_models",
    }
    if any(t.get("ok") and str(t.get("tool") or "") in fleet_tools for t in tool_trace):
        return synthesize_fleet_answer(user_message, host)
    if _is_informational_ask(user_message):
        return synthesize_fleet_answer(user_message, host)
    if propose_ok:
        names = ", ".join(str(t.get("tool")) for t in propose_ok[:5])
        return (
            f"[Observation] Pending draft tool(s) succeeded: {names}.\n"
            "[Analysis] Critical actions wait for human Approve.\n"
            "[Recommendation] Open Home → AI Assistant → Pending Agent Drafts."
        )
    return None


def _extract_path_from_message(user_message: str) -> str:
    import re
    from pathlib import Path

    text = user_message or ""
    # Quoted path
    for pat in (r"'([^']+)'", r'"([^"]+)"', r"`([^`]+)`"):
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    # Windows / posix path with data
    m = re.search(r"([A-Za-z]:\\[^\s\"']+|/?(?:[\w.-]+/)*data(?:/[^\s\"']+)?)", text)
    if m:
        return m.group(1).strip().rstrip(".,;")
    # Bare filename.csv
    m = re.search(r"([\w.-]+\.csv)", text, re.I)
    if m:
        cand = Path("data") / m.group(1)
        if cand.exists():
            return str(cand.resolve())
    return ""


def _title_from_path(path: str, user_message: str = "") -> str:
    import re
    from pathlib import Path

    p = Path(path)
    stem = p.stem if p.suffix.lower() == ".csv" else (p.name if p.name not in (".", "") else "")
    if not stem or stem.lower() == "data":
        msg = (user_message or "").lower()
        if "battery" in msg or "temp" in msg:
            return "Battery Health Monitoring"
        return "Health Monitoring"
    nice = re.sub(r"(?<!^)(?=[A-Z])", " ", stem).replace("_", " ").replace("-", " ")
    nice = re.sub(r"\s+", " ", nice).strip()
    if "monitor" not in nice.lower():
        nice = f"{nice} Monitoring"
    return nice


def _dedupe_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    already_ran: set[str],
) -> list[dict[str, Any]]:
    """Keep each tool name at most once per operator turn (all tools, not only propose_*).

    Prevents loops like list_watchlist×5 before compare_tab_models.
    """
    seen: set[str] = set(already_ran)
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        name = (call.get("name") or "").strip()
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(call)
    return out


def _format_data_driven_create_reply(
    *,
    title: str,
    path: str,
    insp: dict,
    sug: dict,
    draft_id: Any,
    deduped: bool,
) -> str:
    """Operator-facing summary: data looked at → conclusions → config proposal."""
    cols = insp.get("numeric_cols") or insp.get("numeric_features") or []
    ts = insp.get("timestamp_candidates") or []
    rows = insp.get("row_count") or insp.get("rows_sampled_total") or 0
    ranges = insp.get("sample_ranges") or {}
    features = sug.get("features") or (sug.get("config") or {}).get("selected_features") or []
    models = sug.get("models") or (sug.get("config") or {}).get("models") or []
    window = sug.get("window_size") or (sug.get("config") or {}).get("monitoring_window_rows")
    interval = sug.get("interval_ms") or (sug.get("config") or {}).get("interval_ms")
    schedule = sug.get("schedule_type") or (sug.get("config") or {}).get("schedule_type") or "Continuous"

    range_bits = []
    for f in features[:6]:
        r = ranges.get(f) or {}
        if r:
            range_bits.append(
                f"{f}: min={r.get('min')}, max={r.get('max')}, mean={r.get('mean')}"
            )
    model_bits = []
    for m in models[:3]:
        if isinstance(m, dict):
            why = m.get("why") or ""
            model_bits.append(
                f"{m.get('model_type')}"
                + (f" ({why})" if why else "")
                + f" params={m.get('model_parameters') or {}}"
            )

    obs_lines = [
        f"[Observation] Inspected `{path}` for tab '{title}'.",
        f"- rows_sampled={rows}; numeric_cols={cols}; timestamp={ts or 'none'}",
    ]
    if range_bits:
        obs_lines.append("- sample_ranges: " + "; ".join(range_bits))
    if insp.get("quality_notes"):
        obs_lines.append("- quality: " + "; ".join(str(x) for x in insp["quality_notes"][:3]))

    analysis = sug.get("reasoning") or (
        f"Selected features {features}; models {[m.get('model_type') if isinstance(m, dict) else m for m in models]}; "
        f"{schedule} every {(interval or 0) // 1000}s; window={window}."
    )
    if deduped:
        analysis += " Reused an existing pending draft (same title/folder)."

    rec = (
        f"[Recommendation] Approve draft #{draft_id} once under Dashboard → Pending Agent Drafts. "
        f"Config: features={features}; models={[m.get('model_type') if isinstance(m, dict) else m for m in models]}; "
        f"schedule={schedule}/{interval}ms; window={window}."
    )
    return (
        "\n".join(obs_lines)
        + f"\n[Analysis] {analysis}"
        + (f"\n- model choices: " + "; ".join(model_bits) if model_bits else "")
        + f"\n{rec}"
    )


def _deterministic_create_tab(user_message: str, host: Any) -> Optional[dict[str, Any]]:
    """Inspect data → suggest ideal config → propose create_tab once (data-driven)."""
    from app.agent.smart_config import inspect_data_folder, suggest_tab_config
    from app.agent.tools import propose_create_tab, search_knowledge

    path = _extract_path_from_message(user_message)
    if not path:
        return None
    title = _title_from_path(path, user_message)
    tool_trace: list[dict[str, Any]] = []

    insp = inspect_data_folder(path)
    tool_trace.append(
        {
            "tool": "inspect_data_folder",
            "params": {"data_folder": path},
            "ok": bool(insp.get("ok")),
        }
    )
    if not insp.get("ok"):
        return {
            "ok": False,
            "reply": (
                f"[Observation] Could not inspect `{path}`.\n"
                f"[Analysis] {insp.get('error') or insp}\n"
                "[Recommendation] Pass a folder with CSV files, or a CSV path under data/."
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "error",
            "agent_mode": "tools",
            "deterministic": True,
        }

    sop_hints: list[str] = []
    try:
        rag = search_knowledge(title + " battery health monitoring SOP", host=host)
        for hit in (rag.get("hits") or [])[:3]:
            if isinstance(hit, dict):
                sop_hints.append(str(hit.get("snippet") or hit.get("title") or "")[:160])
        tool_trace.append(
            {
                "tool": "search_knowledge",
                "params": {"query": title},
                "ok": True,
                "hits": len(rag.get("hits") or []),
            }
        )
    except Exception:
        pass

    sug = suggest_tab_config(
        title=title,
        data_folder=str(insp.get("data_folder") or path),
        purpose=title,
        inspection=insp,
        sop_hints=sop_hints,
    )
    tool_trace.append(
        {
            "tool": "suggest_tab_config",
            "params": {"title": title, "data_folder": path},
            "ok": bool(sug.get("ok")),
        }
    )

    cfg = sug.get("config") if isinstance(sug.get("config"), dict) else {}
    features = sug.get("features") or cfg.get("selected_features") or []
    models = sug.get("models") or cfg.get("models") or []
    model_names = [
        m.get("model_type") if isinstance(m, dict) else str(m) for m in models
    ]
    proposed_message = (
        f"Create '{title}' from data analysis of {path}: "
        f"features={features}; models={model_names}; "
        f"schedule={sug.get('schedule_type')}/{sug.get('interval_ms')}ms; "
        f"window={sug.get('window_size')}"
    )
    reasoning = sug.get("reasoning") or ""
    if sop_hints:
        reasoning += " SOP hints used: " + "; ".join(sop_hints[:2])

    out = propose_create_tab(
        proposed_message=proposed_message,
        title=title,
        data_folder=str(cfg.get("data_folder") or insp.get("data_folder") or path),
        config=sug,  # full suggest payload → rich draft
        purpose=title,
        auto_suggest=False,
        agent_reasoning=reasoning,
        host=host,
    )
    ok = bool(out.get("ok"))
    draft_id = out.get("draft_id")
    deduped = bool(out.get("deduped"))
    tool_trace.append(
        {
            "tool": "propose_create_tab",
            "params": {"title": title, "data_folder": path},
            "ok": ok,
            "deduped": deduped,
            "draft_id": draft_id,
        }
    )

    reply = _format_data_driven_create_reply(
        title=title,
        path=str(insp.get("data_folder") or path),
        insp=insp,
        sug=sug,
        draft_id=draft_id,
        deduped=deduped,
    )
    if not ok:
        reply = f"Could not propose tab: {out.get('error') or out}"

    return {
        "ok": ok,
        "reply": reply,
        "llm_used": False,
        "tool_trace": tool_trace,
        "outcome": "function_calling" if ok else "error",
        "agent_mode": "tools",
        "deterministic": True,
        "inspection": {
            "numeric_cols": insp.get("numeric_cols"),
            "row_count": insp.get("row_count"),
            "features": features,
            "models": model_names,
        },
    }


def _deterministic_best_model(user_message: str, host: Any) -> Optional[dict[str, Any]]:
    """Answer 'which model is best' via compare_tab_models without requiring LLM."""
    from app.agent.tools import compare_tab_models_tool, list_tab_models

    # Prefer quoted / explicit tab title fragments
    tab_title = None
    m = (user_message or "")
    for pat in (r"'([^']+)'", r'"([^"]+)"', r"for\s+(.+?)\s+tab", r"tab\s+(.+)$"):
        import re

        match = re.search(pat, m, re.I)
        if match:
            tab_title = match.group(1).strip().rstrip("?.!")
            break
    if not tab_title:
        # fallback: words after "for"
        if " for " in m.lower():
            tab_title = m.lower().split(" for ", 1)[1].strip().rstrip("?.!")
            for noise in ("tab", "the", "monitoring"):
                tab_title = tab_title.replace(noise, "").strip()

    comparison = compare_tab_models_tool(tab_title=tab_title, host=host)
    trace = [{"tool": "compare_tab_models", "params": {"tab_title": tab_title}, "ok": bool(comparison.get("ok"))}]
    if not comparison.get("ok"):
        # try list models on first fuzzy fail
        listing = list_tab_models(tab_title=tab_title, host=host) if tab_title else {"ok": False}
        if listing.get("ok"):
            from app.agent.model_lab import compare_tab_models

            comparison = compare_tab_models(listing)
            trace[0]["ok"] = True
        else:
            return {
                "ok": True,
                "reply": (
                    f"[Observation] Could not resolve tab '{tab_title or '?'}' for model compare.\n"
                    "[Analysis] Open the tab or pass its exact title.\n"
                    "[Recommendation] Ask again with the tab name, e.g. "
                    "'Which model is best for Battery Temperature Monitoring?'"
                ),
                "llm_used": False,
                "tool_trace": trace,
                "outcome": "observation_only",
                "deterministic": True,
            }

    best_id = comparison.get("best_model_id")
    best_type = comparison.get("best_model_type")
    models = comparison.get("models") or []
    lines = [
        f"[Observation] Compared {len(models)} model(s) on tab "
        f"'{comparison.get('title') or tab_title}'.",
        f"[Analysis] {comparison.get('recommendation') or 'See metrics below.'}",
    ]
    for row in models[:5]:
        lines.append(
            f"  - {row.get('model_type')} ({row.get('model_id')}): "
            f"trained={row.get('trained')} score={row.get('score')} metrics={row.get('metrics')}"
        )
    if best_id:
        lines.append(
            f"[Recommendation] Prefer {best_type} (`{best_id}`). "
            "If you want a retrain, Approve a propose_train draft for that model_id."
        )
    else:
        lines.append(
            "[Recommendation] No trained models yet — Approve propose_train after the tab has models."
        )
    return {
        "ok": True,
        "reply": "\n".join(lines),
        "llm_used": False,
        "tool_trace": trace,
        "outcome": "function_calling",
        "deterministic": True,
    }


def _should_prefetch_knowledge(user_message: str) -> bool:
    m = (user_message or "").lower().strip()
    if not m:
        return False
    # Creating/updating SOPs must go to propose_write_sop — don't drown the model in RAG.
    if _is_write_document_intent(m):
        return False
    if any(h in m for h in _FLEET_HINTS) and not any(
        h in m for h in ("sop", "procedure", "how to", "how do i")
    ):
        return False
    return any(h in m for h in _KNOWLEDGE_HINTS)


def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw = message.get("tool_calls") or []
    if not isinstance(raw, list):
        return []
    parsed: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = (fn.get("name") or "").strip()
        if not name:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        parsed.append({"name": name, "arguments": args, "id": item.get("id")})
    return parsed


def run_function_calling(
    user_message: str,
    *,
    host: Any,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    max_rounds: int = MAX_TOOL_ROUNDS,
    include_mutating: bool = True,
    system_prompt: Optional[str] = None,
    chat_fn: Optional[Callable[..., Optional[dict[str, Any]]]] = None,
    generate_fn: Optional[Callable[..., Optional[str]]] = None,
    reachable_fn: Optional[Callable[..., bool]] = None,
) -> dict[str, Any]:
    """Run a tool-augmented LLM turn. Returns reply + tool_trace + llm_used."""
    user_message = (user_message or "").strip()
    chat_fn = chat_fn or chat_ollama
    generate_fn = generate_fn or call_ollama
    reachable_fn = reachable_fn or ollama_reachable
    prompt = system_prompt or build_system_prompt(include_mutating=include_mutating) or SYSTEM_PROMPT

    tool_trace: list[dict[str, Any]] = []
    if not user_message:
        return {
            "ok": False,
            "reply": "Please type a question for the agent.",
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "error: empty",
        }

    if host is None:
        return {
            "ok": False,
            "reply": "Tool host is not attached.",
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "error: tool_host_not_attached",
        }

    create_intent = _is_create_tab_intent(user_message)
    best_model_intent = _is_best_model_intent(user_message)
    info_ask = _is_informational_ask(user_message) and not create_intent and not _is_write_document_intent(
        user_message
    )
    ollama_up = is_loopback_url(ollama_url) and bool(reachable_fn(ollama_url))

    # Read-only fleet / "which tab" questions: answer from live tabs (no draft spam)
    if info_ask and not best_model_intent:
        if not ollama_up:
            return {
                "ok": True,
                "reply": synthesize_fleet_answer(user_message, host),
                "llm_used": False,
                "tool_trace": [{"tool": "get_fleet_status", "ok": True, "deterministic": True}],
                "outcome": "function_calling",
                "agent_mode": "tools",
            }

    # Create-tab: always build the draft from real inspect→suggest (ideal name/features/models).
    # If Ollama is up, still ask LLM to narrate — but the draft is already data-driven.
    if create_intent:
        det = _deterministic_create_tab(user_message, host)
        if det is not None and det.get("ok"):
            if not ollama_up:
                det["agent_mode"] = "tools"
                return det
            # Ollama online: keep draft, ask LLM for a richer operator explanation
            from app.agent.ollama_client import resolve_chat_model

            model = resolve_chat_model(model or DEFAULT_OLLAMA_MODEL, base_url=ollama_url)
            narrate = (
                "You are the STDMS agent. A create_tab draft was already created from real data analysis.\n"
                "Using ONLY the facts below, write [Observation] / [Analysis] / [Recommendation] in English.\n"
                "Do NOT call tools. Do NOT invent columns. Tell the operator to Approve the draft once, "
                "then Approve the follow-up train drafts, then start monitoring.\n\n"
                f"Operator request: {user_message}\n\n"
                f"Agent result:\n{det.get('reply')}\n"
            )
            plain = generate_fn(
                narrate,
                base_url=ollama_url,
                model=model,
            )
            reply = (plain or "").strip() or det.get("reply")
            return {
                "ok": True,
                "reply": reply,
                "llm_used": bool(plain),
                "tool_trace": det.get("tool_trace") or [],
                "outcome": "function_calling",
                "agent_mode": "llm" if plain else "tools",
                "deterministic": True,
            }
        if det is not None:
            # inspect failed etc. — surface that reply
            det["agent_mode"] = det.get("agent_mode") or "tools"
            return det
        # Create intent but no CSV/folder path — do not invent a tab or fall through to fleet brief
        return {
            "ok": True,
            "reply": (
                "[Observation] Create-tab request received, but no data folder or CSV path was given.\n"
                "[Analysis] Ideal features/models come from inspect_data_folder on real telemetry files.\n"
                "[Recommendation] Ask again with a path, e.g.\n"
                "  Create tab from data/BatteryTemperature.csv\n"
                "or a folder under data/. Then Approve the draft under Pending Agent Drafts."
            ),
            "llm_used": False,
            "tool_trace": [],
            "outcome": "need_data_path",
            "agent_mode": "tools",
        }

    if not ollama_up:
        if best_model_intent:
            det = _deterministic_best_model(user_message, host)
            if det is not None:
                det["agent_mode"] = "tools"
                det["llm_used"] = False
                return det
        return {
            "ok": True,
            "reply": None,
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "ollama_offline",
            "agent_mode": "offline",
        }

    from app.agent.ollama_client import resolve_chat_model

    model = resolve_chat_model(model or DEFAULT_OLLAMA_MODEL, base_url=ollama_url)

    tools = to_ollama_tools(include_mutating=include_mutating)
    write_intent = _is_write_document_intent(user_message)

    # Prefetch knowledge for document/person-style questions so the model sees hits
    # even if it forgets to call search_knowledge. Skip for create/update SOP intents.
    knowledge_block = ""
    if _should_prefetch_knowledge(user_message):
        try:
            from app.agent.tools import search_knowledge

            rag = search_knowledge(user_message, host=host)
            rag_hits = rag.get("hits") or []
            if rag_hits:
                knowledge_block = (
                    "\n\n[Local knowledge hits — answer from these when relevant; cite sources]\n"
                    + tool_result_json(rag_hits)
                )
                tool_trace.append(
                    {
                        "tool": "search_knowledge",
                        "params": {"query": user_message[:200]},
                        "ok": True,
                        "result_chars": len(knowledge_block),
                        "prefetched": True,
                    }
                )
        except Exception as exc:
            logger.info("Knowledge prefetch failed: %s", exc)

    write_hint = ""
    if write_intent:
        write_hint = (
            "\n\n[Operator intent: CREATE/UPDATE SOP procedure as Word .docx]\n"
            "You MUST call propose_write_sop (new) or propose_update_sop (existing) with "
            "sop_fields (process_title, sop_id, department, steps, …). "
            "Do NOT give a fleet health report. Do NOT only search_knowledge."
        )
    elif create_intent:
        # Prefetch real data profile so the LLM reasons on facts, not guesses
        data_profile = ""
        try:
            from app.agent.smart_config import inspect_data_folder, suggest_tab_config

            path0 = _extract_path_from_message(user_message)
            if path0:
                insp0 = inspect_data_folder(path0)
                title0 = _title_from_path(path0, user_message)
                sug0 = (
                    suggest_tab_config(
                        title=title0,
                        data_folder=str(insp0.get("data_folder") or path0),
                        purpose=title0,
                        inspection=insp0,
                    )
                    if insp0.get("ok")
                    else {}
                )
                tool_trace.append(
                    {
                        "tool": "inspect_data_folder",
                        "params": {"data_folder": path0},
                        "ok": bool(insp0.get("ok")),
                        "prefetched": True,
                    }
                )
                if sug0.get("ok"):
                    tool_trace.append(
                        {
                            "tool": "suggest_tab_config",
                            "params": {"title": title0},
                            "ok": True,
                            "prefetched": True,
                        }
                    )
                data_profile = (
                    "\n\n[DATA PROFILE — use these facts; do not invent columns]\n"
                    + tool_result_json(
                        {
                            "inspection": {
                                "ok": insp0.get("ok"),
                                "data_folder": insp0.get("data_folder"),
                                "columns": insp0.get("columns"),
                                "numeric_cols": insp0.get("numeric_cols"),
                                "timestamp_candidates": insp0.get("timestamp_candidates"),
                                "row_count": insp0.get("row_count"),
                                "null_rates": insp0.get("null_rates"),
                                "sample_ranges": insp0.get("sample_ranges"),
                                "quality_notes": insp0.get("quality_notes"),
                            },
                            "suggestion": {
                                "tab_name": sug0.get("tab_name"),
                                "features": sug0.get("features"),
                                "models": sug0.get("models"),
                                "schedule_type": sug0.get("schedule_type"),
                                "interval_ms": sug0.get("interval_ms"),
                                "window_size": sug0.get("window_size"),
                                "reasoning": sug0.get("reasoning"),
                            },
                        }
                    )
                )
                knowledge_block = (knowledge_block or "") + data_profile
        except Exception as exc:
            logger.info("Create-tab data prefetch failed: %s", exc)

        write_hint = (
            "\n\n[Operator intent: CREATE MONITORING TAB from a data path]\n"
            "You MUST base your answer on the DATA PROFILE above (columns, ranges, suggested features/models).\n"
            "Explain in [Observation] what you saw in the data, in [Analysis] why those features/models fit, "
            "then call propose_create_tab EXACTLY ONCE with that suggested config "
            "(pass config from suggestion / title / data_folder).\n"
            "Do NOT invent columns. Do NOT call propose_create_tab more than once."
        )
    elif best_model_intent:
        write_hint = (
            "\n\n[Operator intent: WHICH MODEL IS BEST]\n"
            "Call compare_tab_models (and optionally suggest_model_optimization / get_model_metrics). "
            "Recommend the best trained model with metrics. Do not invent rankings."
        )
    elif info_ask:
        write_hint = (
            "\n\n[Operator intent: INFORMATION ONLY]\n"
            "Call get_fleet_status (or get_all_snapshots). Answer which tab matches "
            "(e.g. eclipse → Eclipse monitoring). Do NOT call propose_* tools. "
            "Do NOT tell the operator to Approve drafts."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message + knowledge_block + write_hint},
    ]

    once_ran: set[str] = set()
    for _round in range(max(1, int(max_rounds))):
        message = chat_fn(
            messages,
            tools=tools,
            base_url=ollama_url,
            model=model or DEFAULT_OLLAMA_MODEL,
        )
        if message is None:
            # LLM chat failed mid-flight — still try smart create/best via tools
            if create_intent:
                det = _deterministic_create_tab(user_message, host)
                if det is not None:
                    det["agent_mode"] = "tools"
                    return det
            if best_model_intent:
                det = _deterministic_best_model(user_message, host)
                if det is not None:
                    det["agent_mode"] = "tools"
                    return det
            if info_ask:
                return {
                    "ok": True,
                    "reply": synthesize_fleet_answer(user_message, host),
                    "llm_used": False,
                    "tool_trace": tool_trace
                    or [{"tool": "get_fleet_status", "ok": True, "deterministic": True}],
                    "outcome": "function_calling",
                    "agent_mode": "tools",
                }
            plain = generate_fn(
                prompt + "\n\nOperator question:\n" + user_message,
                base_url=ollama_url,
                model=model or DEFAULT_OLLAMA_MODEL,
            )
            return {
                "ok": True,
                "reply": plain,
                "llm_used": bool(plain),
                "tool_trace": tool_trace,
                "outcome": "function_calling" if tool_trace else ("llm_plain" if plain else "llm_failed"),
                "agent_mode": "llm" if plain else "offline",
            }

        tool_calls = _dedupe_tool_calls(_parse_tool_calls(message), already_ran=once_ran)
        content = (message.get("content") or "").strip()

        if not tool_calls:
            if write_intent or create_intent or best_model_intent:
                # Exit to safety-net / final English summary
                if content and not (write_intent or create_intent):
                    # best_model may already have answered in content after tools in prior rounds
                    pass
                break
            # Informational: prefer model content; else synthesize from fleet
            if info_ask and not content:
                content = synthesize_fleet_answer(user_message, host)
            return {
                "ok": True,
                "reply": content or None,
                "llm_used": True,
                "tool_trace": tool_trace,
                "outcome": "function_calling" if tool_trace else "llm_direct",
                "agent_mode": "llm",
            }

        messages.append(message)
        for call in tool_calls:
            name = call["name"]
            args = call.get("arguments") or {}
            once_ran.add(name)  # every tool at most once this turn (all rounds)
            entry: dict[str, Any] = {"tool": name, "params": args, "ok": False}
            # Info questions must not create Approve drafts
            if info_ask and name in _PROPOSE_TOOLS:
                payload = json.dumps(
                    {
                        "ok": False,
                        "error": "propose_blocked_on_informational_ask",
                        "note": "Operator asked for information only — answer from get_fleet_status / knowledge.",
                    },
                    ensure_ascii=False,
                )
                entry["error"] = "propose_blocked_on_informational_ask"
                tool_trace.append(entry)
                tool_msg = {"role": "tool", "name": name, "content": payload}
                if call.get("id"):
                    tool_msg["tool_call_id"] = call["id"]
                messages.append(tool_msg)
                continue
            try:
                result = invoke_tool(name, args, host=host)
                payload = tool_result_json(result)
                entry["ok"] = True
                entry["result_chars"] = len(payload)
                if isinstance(result, dict) and result.get("deduped"):
                    entry["deduped"] = True
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
                entry["error"] = str(exc)
                logger.info("Tool %s failed: %s", name, exc)
            tool_trace.append(entry)
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": payload,
            }
            tool_msg["name"] = name
            if call.get("id"):
                tool_msg["tool_call_id"] = call["id"]
            messages.append(tool_msg)

    wrote = any(
        t.get("tool") in ("propose_write_sop", "propose_update_sop", "propose_write_document")
        for t in tool_trace
    )
    if write_intent and not wrote:
        try:
            from app.agent.tools import propose_write_sop_from_user_text

            auto = propose_write_sop_from_user_text(user_message, host=host)
            tool_trace.append(
                {
                    "tool": "propose_write_sop",
                    "params": {"auto_from_user_text": True},
                    "ok": bool(auto.get("ok")),
                    "result_chars": len(tool_result_json(auto)),
                }
            )
            if auto.get("ok"):
                reply = (
                    "Created a pending SOP Word draft for human Approve.\n"
                    f"- draft_id: {auto.get('draft_id')}\n"
                    f"- kind: {auto.get('kind')}\n"
                    "- Next: Dashboard → Pending Agent Drafts → Approve\n"
                    "After Approve, the .docx is written under data/knowledge (STDMS SOP Word template)."
                )
            else:
                reply = (
                    f"Could not auto-create SOP draft: {auto.get('error') or auto}. "
                    "Please provide process title and SOP ID."
                )
            return {
                "ok": True,
                "reply": reply,
                "llm_used": True,
                "tool_trace": tool_trace,
                "outcome": "function_calling",
                "agent_mode": "llm",
                "tools_available": [t["name"] for t in list_tools(include_mutating=include_mutating)],
            }
        except Exception as exc:
            logger.info("Auto propose_write_sop failed: %s", exc)

    # Safety net: LLM forgot propose_create_tab — create exactly one draft, then let LLM explain
    proposed_tab = any(t.get("tool") == "propose_create_tab" and t.get("ok") for t in tool_trace)
    if create_intent and not proposed_tab:
        det = _deterministic_create_tab(user_message, host)
        if det is not None:
            tool_trace.extend(det.get("tool_trace") or [])
            messages.append(
                {
                    "role": "tool",
                    "name": "propose_create_tab",
                    "content": tool_result_json(
                        {
                            "ok": det.get("ok"),
                            "note": "Safety-net single draft created by agent runtime",
                            "reply_preview": (det.get("reply") or "")[:500],
                        }
                    ),
                }
            )

    compared = any(t.get("tool") == "compare_tab_models" and t.get("ok") for t in tool_trace)
    if best_model_intent and not compared:
        det = _deterministic_best_model(user_message, host)
        if det is not None:
            tool_trace.extend(det.get("tool_trace") or [])
            messages.append(
                {
                    "role": "tool",
                    "name": "compare_tab_models",
                    "content": tool_result_json({"ok": True, "summary": (det.get("reply") or "")[:1500]}),
                }
            )

    if write_intent:
        final_instruction = (
            "Stop calling tools. Tell the operator clearly whether propose_write_sop / "
            "propose_update_sop created a pending draft, the filename, and that they must "
            "Approve it under Pending Agent Drafts. Do not invent a fleet health report."
        )
    elif create_intent:
        final_instruction = (
            "Stop calling tools. Summarize the single create_tab draft for the operator: "
            "tab title, key features, chosen 2–3 models, schedule, draft_id if known, "
            "and that they must Approve once under Pending Agent Drafts. "
            "Use [Observation] / [Analysis] / [Recommendation]."
        )
    elif best_model_intent:
        final_instruction = (
            "Stop calling tools. State which model is best with metrics from tool results. "
            "Use [Observation] / [Analysis] / [Recommendation]."
        )
    elif info_ask:
        final_instruction = (
            "Stop calling tools. Answer the operator's information question directly "
            "from the tool results (which tab / status / facts). "
            "Do NOT tell them to Approve drafts unless they asked to create/train/write. "
            "Name matching tab titles clearly. Use [Observation] / [Analysis] / [Recommendation]."
        )
    else:
        final_instruction = (
            "Stop calling tools. Using the tool results above, give your final "
            "English answer for the operator now. Use "
            "[Observation] / [Analysis] / [Recommendation] when useful. "
            "Only mention Pending Agent Drafts if a propose_* draft was actually created."
        )
    messages.append({"role": "user", "content": final_instruction})
    final = chat_fn(
        messages,
        tools=None,
        base_url=ollama_url,
        model=model or DEFAULT_OLLAMA_MODEL,
    )
    reply = None
    if final:
        reply = (final.get("content") or "").strip() or None
    if not reply:
        reply = generate_fn(
            (
                "Tell the operator the SOP draft status in English.\nQuestion: "
                if write_intent
                else "Answer the operator question in English using the tool facts.\nQuestion: "
            )
            + user_message,
            base_url=ollama_url,
            model=model or DEFAULT_OLLAMA_MODEL,
        )
    if not (reply or "").strip():
        reply = _reply_after_tools(
            user_message=user_message,
            host=host,
            tool_trace=tool_trace,
            write_intent=write_intent,
        ) or synthesize_fleet_answer(user_message, host)

    return {
        "ok": True,
        "reply": reply,
        "llm_used": bool(reply),
        "tool_trace": tool_trace,
        "outcome": "function_calling",
        "agent_mode": "llm",
        "tools_available": [t["name"] for t in list_tools(include_mutating=include_mutating)],
    }
