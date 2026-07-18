"""Agent tool registry (Phase 0–3).

Read tools inspect live tab state. Propose tools write pending drafts only
(human approval required). search_knowledge is a RAG stub until Phase 4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from app.agent.bridge import ToolHost, get_tool_host


ToolHandler = Callable[..., Any]

KNOWLEDGE_STATUS = "unavailable"


AGENT_TOOLS: list[dict[str, Any]] = [
    # --- Phase 0 aliases (kept for API/tests) ---
    {
        "name": "get_all_snapshots",
        "description": "Return snapshots for all active monitoring tabs.",
        "parameters": {},
        "properties": {},
        "required": [],
        "phase": 0,
        "mutates": False,
    },
    {
        "name": "get_tab_snapshot",
        "description": "Return the latest snapshot for one monitoring tab by tab_id.",
        "parameters": {"tab_id": "str"},
        "properties": {"tab_id": {"type": "string", "description": "Monitoring tab id"}},
        "required": ["tab_id"],
        "phase": 0,
        "mutates": False,
    },
    {
        "name": "get_tab_history",
        "description": "Return the last N monitoring/anomaly events for a tab.",
        "parameters": {"tab_id": "str", "n": "int"},
        "properties": {
            "tab_id": {"type": "string"},
            "n": {"type": "integer", "description": "Max events (1-100)", "default": 20},
        },
        "required": ["tab_id"],
        "phase": 0,
        "mutates": False,
    },
    {
        "name": "check_pending_retrain_signals",
        "description": "List unacknowledged drift-driven retrain signals.",
        "parameters": {},
        "properties": {},
        "required": [],
        "phase": 0,
        "mutates": False,
    },
    # --- Phase 3 read / soft-action ---
    {
        "name": "get_fleet_status",
        "description": "Fleet overview: all tabs current health/drift/alerts (alias of get_all_snapshots).",
        "parameters": {},
        "properties": {},
        "required": [],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "get_tab_detail",
        "description": "Full tab detail: snapshot, history, last anomaly/drift/forecast analysis.",
        "parameters": {"tab_id": "str", "n": "int"},
        "properties": {
            "tab_id": {"type": "string"},
            "n": {"type": "integer", "default": 20},
        },
        "required": ["tab_id"],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "list_tab_models",
        "description": (
            "List models configured on a monitoring tab: model_id, type, trained status, "
            "and available metric values. Use this to inspect trained models (not just tab health)."
        ),
        "parameters": {"tab_id": "str", "tab_title": "str"},
        "properties": {
            "tab_id": {"type": "string", "description": "Tab UUID if known"},
            "tab_title": {"type": "string", "description": "Tab title to resolve if tab_id unknown"},
        },
        "required": [],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "get_model_metrics",
        "description": (
            "Get performance metrics for one trained model on a tab "
            "(same info as Model Metrics in the UI)."
        ),
        "parameters": {"tab_id": "str", "tab_title": "str", "model_id": "str"},
        "properties": {
            "tab_id": {"type": "string"},
            "tab_title": {"type": "string"},
            "model_id": {"type": "string", "description": "Model UUID from list_tab_models"},
        },
        "required": ["model_id"],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "inspect_data_folder",
        "description": (
            "Inspect a CSV data folder before creating a tab. Returns columns, numeric_cols, "
            "timestamp_candidates, row_count, null_rates, sample_ranges (min/max/mean)."
        ),
        "parameters": {"data_folder": "str"},
        "properties": {
            "data_folder": {
                "type": "string",
                "description": "Absolute or relative folder with CSV files (e.g. data or data/battery)",
            },
        },
        "required": ["data_folder"],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "suggest_tab_config",
        "description": (
            "From inspect + purpose/SOP, suggest tab_name, features (2–3 models max), "
            "schedule_type, window_size, and reasoning. Pass returned config to propose_create_tab."
        ),
        "parameters": {"title": "str", "data_folder": "str", "purpose": "str"},
        "properties": {
            "title": {"type": "string"},
            "data_folder": {"type": "string"},
            "purpose": {
                "type": "string",
                "description": "e.g. battery health, eclipse monitoring",
            },
        },
        "required": ["title", "data_folder"],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "compare_tab_models",
        "description": "Compare trained models on a tab by metrics and recommend the best.",
        "parameters": {"tab_id": "str", "tab_title": "str"},
        "properties": {"tab_id": {"type": "string"}, "tab_title": {"type": "string"}},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "propose_model_plan",
        "description": (
            "Suggest a multi-model training plan with hyperparameters. "
            "Does not train; use propose_train after models exist on the tab."
        ),
        "parameters": {"purpose": "str"},
        "properties": {"purpose": {"type": "string", "default": "health"}},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "suggest_model_optimization",
        "description": "From compare_tab_models, suggest Approve-safe optimize/train next steps.",
        "parameters": {"tab_id": "str", "tab_title": "str"},
        "properties": {"tab_id": {"type": "string"}, "tab_title": {"type": "string"}},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "explain_anomaly",
        "description": (
            "Deep root-cause explanation for a tab: fusion, drift, OBS, XAI events, model notes."
        ),
        "parameters": {"tab_id": "str", "tab_title": "str"},
        "properties": {"tab_id": {"type": "string"}, "tab_title": {"type": "string"}},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "detect_patterns",
        "description": "Detect fusion score trends/spikes and recurring warning patterns on a tab.",
        "parameters": {"tab_id": "str", "tab_title": "str"},
        "properties": {"tab_id": {"type": "string"}, "tab_title": {"type": "string"}},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "forecast_risk",
        "description": (
            "Combine forecast + patterns + anomaly/drift into risk_score and what-to-do actions."
        ),
        "parameters": {"tab_id": "str", "tab_title": "str"},
        "properties": {"tab_id": {"type": "string"}, "tab_title": {"type": "string"}},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "get_pending_actions_summary",
        "description": "Summarize pending drafts, retrain signals, and agent watchlist for the operator.",
        "parameters": {},
        "properties": {},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "list_watchlist",
        "description": "List tabs on the agent attention watchlist.",
        "parameters": {},
        "properties": {},
        "required": [],
        "phase": 4,
        "mutates": False,
    },
    {
        "name": "add_watchlist_tab",
        "description": "Add a tab to the agent watchlist (persistent).",
        "parameters": {"tab_id": "str", "tab_title": "str", "note": "str"},
        "properties": {
            "tab_id": {"type": "string"},
            "tab_title": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": [],
        "phase": 4,
        "mutates": True,
    },
    {
        "name": "remove_watchlist_tab",
        "description": "Remove a tab from the agent watchlist.",
        "parameters": {"tab_id": "str", "tab_title": "str"},
        "properties": {"tab_id": {"type": "string"}, "tab_title": {"type": "string"}},
        "required": [],
        "phase": 4,
        "mutates": True,
    },
    {
        "name": "run_anomaly_check",
        "description": (
            "Read last anomaly/fusion/OBS results for a tab (does not start a new heavy cycle). "
            "Returns status=no_recent_cycle when no monitoring data exists yet."
        ),
        "parameters": {"tab_id": "str"},
        "properties": {"tab_id": {"type": "string"}},
        "required": ["tab_id"],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "run_forecast",
        "description": "Return last short-horizon forecast score for a tab if available.",
        "parameters": {"tab_id": "str"},
        "properties": {"tab_id": {"type": "string"}},
        "required": ["tab_id"],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "check_drift",
        "description": "Return last drift score, is_drift flag, and drifted features for a tab.",
        "parameters": {"tab_id": "str"},
        "properties": {"tab_id": {"type": "string"}},
        "required": ["tab_id"],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "get_pending_signals",
        "description": "List pending retrain signals and pending draft alerts/config proposals.",
        "parameters": {},
        "properties": {},
        "required": [],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "request_monitor_cycle",
        "description": (
            "Queue a GUI monitoring refresh for a tab (non-blocking). "
            "Next agent cycle can read updated snapshot."
        ),
        "parameters": {"tab_id": "str"},
        "properties": {"tab_id": {"type": "string"}},
        "required": ["tab_id"],
        "phase": 3,
        "mutates": False,  # soft side-effect; auto-allowed
        "side_effect": "queue_gui",
    },
    {
        "name": "search_knowledge",
        "description": (
            "Search department SOP/runbook knowledge base (dual local RAG: space + ground) "
            "and built-in product how-to. Use segment=space for satellite/FSM/telemetry, "
            "ground for alert/maintenance/operator SOPs, auto to classify, both for fleet anomalies."
        ),
        "parameters": {"query": "str", "segment": "str"},
        "properties": {
            "query": {"type": "string", "description": "Natural language procedure query"},
            "segment": {
                "type": "string",
                "enum": ["auto", "space", "ground", "both"],
                "description": "RAG store segment (default auto)",
                "default": "auto",
            },
        },
        "required": ["query"],
        "phase": 3,
        "mutates": False,
    },
    # --- Propose (pending human approval) ---
    {
        "name": "propose_retrain",
        "description": (
            "Create a pending retrain signal for a tab (human must acknowledge). "
            "Does not start training."
        ),
        "parameters": {"tab_id": "str", "reasoning": "str", "drift_score": "float"},
        "properties": {
            "tab_id": {"type": "string"},
            "reasoning": {"type": "string"},
            "drift_score": {"type": "number", "default": 0.0},
        },
        "required": ["tab_id"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_alert",
        "description": (
            "Create a pending alert draft for human approval. Does not send email or open incidents."
        ),
        "parameters": {
            "tab_id": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
            "severity": "str",
        },
        "properties": {
            "tab_id": {"type": "string"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
            "severity": {
                "type": "string",
                "description": "INFO|WARNING|CRITICAL",
                "default": "WARNING",
            },
        },
        "required": ["tab_id", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_config_change",
        "description": (
            "Propose a tab configuration change as a pending draft (kind=config). "
            "Does not apply the change."
        ),
        "parameters": {
            "tab_id": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
            "config_patch": "object",
        },
        "properties": {
            "tab_id": {"type": "string"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
            "config_patch": {"type": "object", "description": "Suggested config fields"},
        },
        "required": ["tab_id", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_train",
        "description": (
            "Propose training a model on an existing tab (kind=train). "
            "Does not start training until a human Approves the draft."
        ),
        "parameters": {
            "tab_id": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
            "model_id": "str",
        },
        "properties": {
            "tab_id": {"type": "string"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
            "model_id": {
                "type": "string",
                "description": "Optional model_id; omit to train the tab's selected/default model",
            },
        },
        "required": ["tab_id", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_create_tab",
        "description": (
            "Propose creating a new monitoring tab (kind=create_tab). "
            "Prefer inspect_data_folder → suggest_tab_config first for health/battery tabs. "
            "With auto_suggest=true (default), empty selected_features are filled from the data folder. "
            "Does not create the tab until a human Approves the draft."
        ),
        "parameters": {
            "proposed_message": "str",
            "agent_reasoning": "str",
            "title": "str",
            "data_folder": "str",
            "config": "object",
            "purpose": "str",
            "auto_suggest": "bool",
        },
        "properties": {
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
            "title": {"type": "string", "description": "Tab title"},
            "data_folder": {"type": "string", "description": "CSV/data folder path"},
            "config": {
                "type": "object",
                "description": (
                    "Partial tab config (selected_features, models[], schedule_type, "
                    "interval_ms, data_folder) merged with defaults on Approve"
                ),
            },
            "purpose": {
                "type": "string",
                "description": "e.g. battery health — used when auto-suggesting features/models",
            },
            "auto_suggest": {
                "type": "boolean",
                "description": "If true and features empty, run suggest_tab_config (default true)",
                "default": True,
            },
        },
        "required": ["proposed_message", "title"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_start_monitoring",
        "description": (
            "Propose starting monitoring on an existing tab (kind=start_monitoring). "
            "Use when the operator says activate/start a tab. "
            "Does not start until a human Approves the draft. "
            "Pass tab_id from get_fleet_status, or tab_title to match by name."
        ),
        "parameters": {
            "tab_id": "str",
            "tab_title": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
        },
        "properties": {
            "tab_id": {"type": "string", "description": "Tab UUID if known"},
            "tab_title": {
                "type": "string",
                "description": "Tab display title to resolve (e.g. Weekly h/c)",
            },
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
        },
        "required": ["proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_stop_monitoring",
        "description": (
            "Propose stopping monitoring on an existing tab (kind=stop_monitoring). "
            "Does not stop until a human Approves the draft. "
            "Use WARNING/CRITICAL severity when health is elevated. "
            "If a pending stop_monitoring draft already exists for the tab, it is reused."
        ),
        "parameters": {
            "tab_id": "str",
            "tab_title": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
            "severity": "str",
        },
        "properties": {
            "tab_id": {"type": "string"},
            "tab_title": {"type": "string"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
            "severity": {
                "type": "string",
                "description": "INFO|WARNING|CRITICAL — use CRITICAL when tab health is critical",
                "default": "WARNING",
            },
        },
        "required": ["proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_remove_model",
        "description": (
            "Propose removing a model from a tab (kind=remove_model). "
            "Does not delete until a human Approves the draft."
        ),
        "parameters": {
            "tab_id": "str",
            "tab_title": "str",
            "model_id": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
        },
        "properties": {
            "tab_id": {"type": "string"},
            "tab_title": {"type": "string"},
            "model_id": {"type": "string"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
        },
        "required": ["model_id", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "list_knowledge_documents",
        "description": (
            "List Markdown/TXT documents in data/knowledge that the agent can read or propose updates for."
        ),
        "parameters": {},
        "properties": {},
        "required": [],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "read_knowledge_document",
        "description": "Read a Markdown/TXT knowledge document by filename (e.g. SOP-7_anomaly_response.md).",
        "parameters": {"filename": "str"},
        "properties": {
            "filename": {"type": "string", "description": "Bare filename under data/knowledge"},
        },
        "required": ["filename"],
        "phase": 3,
        "mutates": False,
    },
    {
        "name": "propose_write_document",
        "description": (
            "Propose creating a NEW Markdown/TXT file under data/knowledge (kind=write_document). "
            "Does not write until a human Approves. Filename must end with .md or .txt."
        ),
        "parameters": {
            "filename": "str",
            "content": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
        },
        "properties": {
            "filename": {"type": "string", "description": "e.g. SOP-8_new_procedure.md"},
            "content": {"type": "string", "description": "Full document body (Markdown/TXT)"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
        },
        "required": ["filename", "content", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_update_document",
        "description": (
            "Propose updating an EXISTING Markdown/TXT file under data/knowledge (kind=update_document). "
            "Provide the full new content. Does not write until a human Approves."
        ),
        "parameters": {
            "filename": "str",
            "content": "str",
            "proposed_message": "str",
            "agent_reasoning": "str",
        },
        "properties": {
            "filename": {"type": "string"},
            "content": {"type": "string", "description": "Full replacement body"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
        },
        "required": ["filename", "content", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_write_sop",
        "description": (
            "Propose creating a NEW SOP procedure as a Word .docx using the STDMS SOP template "
            "(General Information, Process Overview, Process Steps). "
            "Does not write until human Approves (kind=write_sop)."
        ),
        "parameters": {
            "filename": "str",
            "sop_fields": "object",
            "proposed_message": "str",
            "agent_reasoning": "str",
        },
        "properties": {
            "filename": {
                "type": "string",
                "description": "Optional .docx name; default from sop_id/title",
            },
            "sop_fields": {
                "type": "object",
                "description": (
                    "process_title, department, contact_info, sop_id, effective_date, "
                    "revision_number, process_description, purpose_scope, "
                    "definitions_related, steps[{wbs,task,owner}]"
                ),
            },
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
        },
        "required": ["sop_fields", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
    {
        "name": "propose_update_sop",
        "description": (
            "Propose updating an EXISTING SOP Word .docx with the STDMS SOP template fields. "
            "Does not write until human Approves (kind=update_sop)."
        ),
        "parameters": {
            "filename": "str",
            "sop_fields": "object",
            "proposed_message": "str",
            "agent_reasoning": "str",
        },
        "properties": {
            "filename": {"type": "string", "description": "Existing .docx under data/knowledge"},
            "sop_fields": {"type": "object"},
            "proposed_message": {"type": "string"},
            "agent_reasoning": {"type": "string"},
        },
        "required": ["filename", "sop_fields", "proposed_message"],
        "phase": 3,
        "mutates": True,
    },
]

# Backward-compatible name used by older imports/tests
READ_ONLY_TOOLS = AGENT_TOOLS


def list_tools(*, include_mutating: bool = False) -> list[dict[str, Any]]:
    if include_mutating:
        return list(AGENT_TOOLS)
    return [t for t in AGENT_TOOLS if not t.get("mutates")]


def to_ollama_tools(*, include_mutating: bool = True) -> list[dict[str, Any]]:
    """Convert registry entries to Ollama tool definitions (propose tools included by default)."""
    out: list[dict[str, Any]] = []
    for tool in list_tools(include_mutating=include_mutating):
        props = tool.get("properties") or {}
        required = tool.get("required") or []
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or tool["name"],
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            }
        )
    return out


def _require_host(host: Optional[ToolHost] = None) -> ToolHost:
    resolved = host if host is not None else get_tool_host()
    if resolved is None:
        raise RuntimeError("Tool host is not attached (PyQt tool not running)")
    return resolved


def get_all_snapshots(host: Optional[ToolHost] = None) -> list[dict[str, Any]]:
    return _require_host(host).list_tabs()


def get_tab_snapshot(tab_id: str, host: Optional[ToolHost] = None) -> Optional[dict[str, Any]]:
    return _require_host(host).get_snapshot(tab_id)


def get_tab_history(tab_id: str, n: int = 50, host: Optional[ToolHost] = None) -> list[dict[str, Any]]:
    return _require_host(host).get_history(tab_id, n=n)


def check_pending_retrain_signals(host: Optional[ToolHost] = None) -> list[dict[str, Any]]:
    return _require_host(host).get_pending_retrain_signals()


def get_fleet_status(host: Optional[ToolHost] = None) -> list[dict[str, Any]]:
    return get_all_snapshots(host=host)


def get_tab_detail(tab_id: str, n: int = 20, host: Optional[ToolHost] = None) -> dict[str, Any]:
    h = _require_host(host)
    analysis = h.get_tab_analysis(tab_id)
    history = h.get_history(tab_id, n=max(1, min(int(n), 100)))
    analysis = dict(analysis)
    analysis["history"] = history
    try:
        analysis["models"] = h.list_tab_models(tab_id)
    except Exception as exc:
        analysis["models"] = {"ok": False, "error": str(exc)}
    return analysis


def list_tab_models(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    out = h.list_tab_models(str(resolved["tab_id"]))
    out["resolved_title"] = resolved.get("title")
    return out


def get_model_metrics(
    model_id: str,
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    out = h.get_model_metrics(str(resolved["tab_id"]), str(model_id))
    out["resolved_title"] = resolved.get("title")
    return out


def inspect_data_folder(data_folder: str, host: Optional[ToolHost] = None) -> dict[str, Any]:
    from app.agent.smart_config import inspect_data_folder as _inspect

    _ = host
    return _inspect(data_folder)


def suggest_tab_config(
    title: str,
    data_folder: str,
    purpose: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.smart_config import suggest_tab_config as _suggest

    _ = host
    sop_hints: list[str] = []
    try:
        rag = search_knowledge(purpose or title, host=host)
        for hit in (rag.get("hits") or [])[:3]:
            if isinstance(hit, dict):
                sop_hints.append(str(hit.get("snippet") or hit.get("title") or hit)[:200])
    except Exception:
        pass
    return _suggest(
        title=title,
        data_folder=data_folder,
        purpose=purpose or title,
        sop_hints=sop_hints,
    )


def compare_tab_models_tool(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.model_lab import compare_tab_models

    listing = list_tab_models(tab_id=tab_id, tab_title=tab_title, host=host)
    return compare_tab_models(listing)


def propose_model_plan_tool(purpose: str = "health", host: Optional[ToolHost] = None) -> dict[str, Any]:
    from app.agent.model_lab import propose_model_plan

    _ = host
    return propose_model_plan(purpose=purpose or "health")


def suggest_model_optimization(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.model_lab import suggest_optimize_from_comparison

    comparison = compare_tab_models_tool(tab_id=tab_id, tab_title=tab_title, host=host)
    return suggest_optimize_from_comparison(comparison)


def explain_anomaly_tool(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.insights import explain_anomaly

    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    tid = str(resolved["tab_id"])
    analysis = h.get_tab_analysis(tid)
    models = h.list_tab_models(tid)
    return explain_anomaly(analysis, models_listing=models)


def detect_patterns_tool(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.insights import detect_patterns

    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    tid = str(resolved["tab_id"])
    widget = None
    try:
        # access via host private helper if available
        widget = h._get_widget(tid)  # type: ignore[attr-defined]
    except Exception:
        widget = None
    score_history = list(getattr(widget, "score_history", None) or []) if widget else []
    value_history: list = []
    if widget is not None:
        for attr in ("value_history", "feature_history", "latest_values"):
            raw = getattr(widget, attr, None)
            if isinstance(raw, list) and raw:
                try:
                    value_history = [float(x) for x in raw if x is not None]
                except Exception:
                    value_history = []
                if value_history:
                    break
    events = []
    try:
        events = list(getattr(widget, "anomaly_events", None) or [])[:50] if widget else []
    except Exception:
        events = h.get_history(tid, n=50)
    out = detect_patterns(
        score_history=score_history,
        events=events,
        value_history=value_history or None,
    )
    out["tab_id"] = tid
    out["title"] = resolved.get("title")
    return out


def forecast_risk_tool(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.insights import forecast_risk

    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    tid = str(resolved["tab_id"])
    analysis = h.get_tab_analysis(tid)
    patterns = detect_patterns_tool(tab_id=tid, host=host)
    out = forecast_risk(
        forecast_value=analysis.get("forecast"),
        patterns=patterns,
        anomaly=analysis.get("anomaly") or {},
        drift=analysis.get("drift") or {},
    )
    out["tab_id"] = tid
    out["title"] = resolved.get("title")
    return out


def get_pending_actions_summary(host: Optional[ToolHost] = None) -> dict[str, Any]:
    from app.agent.watchlist import pending_actions_summary

    return pending_actions_summary(_require_host(host))


def list_watchlist(host: Optional[ToolHost] = None) -> dict[str, Any]:
    from app.agent.watchlist import load_watchlist

    _ = host
    return load_watchlist()


def add_watchlist_tab(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    note: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.watchlist import add_to_watchlist

    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    return add_to_watchlist(
        str(resolved["tab_id"]),
        title=str(resolved.get("title") or ""),
        note=note or "",
    )


def remove_watchlist_tab(
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.watchlist import remove_from_watchlist

    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    return remove_from_watchlist(str(resolved["tab_id"]))


def run_anomaly_check(tab_id: str, host: Optional[ToolHost] = None) -> dict[str, Any]:
    analysis = _require_host(host).get_tab_analysis(tab_id)
    return {
        "status": analysis.get("status"),
        "tab_id": tab_id,
        "title": analysis.get("title"),
        "anomaly": analysis.get("anomaly") or {},
        "recent_events": analysis.get("recent_events") or [],
        "snapshot": analysis.get("snapshot"),
    }


def run_forecast(tab_id: str, host: Optional[ToolHost] = None) -> dict[str, Any]:
    analysis = _require_host(host).get_tab_analysis(tab_id)
    forecast = analysis.get("forecast")
    if analysis.get("status") == "tab_not_found":
        return analysis
    if forecast is None:
        return {
            "status": "no_forecast",
            "tab_id": tab_id,
            "forecast": None,
            "note": "Not enough score history or no recent cycle.",
        }
    return {"status": "ok", "tab_id": tab_id, "forecast": forecast}


def check_drift(tab_id: str, host: Optional[ToolHost] = None) -> dict[str, Any]:
    analysis = _require_host(host).get_tab_analysis(tab_id)
    if analysis.get("status") == "tab_not_found":
        return analysis
    drift = analysis.get("drift") or {}
    if not drift:
        return {
            "status": "no_recent_cycle",
            "tab_id": tab_id,
            "drift": {},
            "note": "No drift result available yet.",
        }
    return {
        "status": "ok",
        "tab_id": tab_id,
        "is_drift": bool(drift.get("is_drift")),
        "drift_score": drift.get("drift_score"),
        "drifted_features": drift.get("drifted_features") or [],
        "drift": drift,
    }


def get_pending_signals(host: Optional[ToolHost] = None) -> dict[str, Any]:
    h = _require_host(host)
    retrain = h.get_pending_retrain_signals()
    drafts = h.list_pending_drafts() if hasattr(h, "list_pending_drafts") else []
    return {
        "retrain_signals": retrain,
        "draft_alerts": drafts,
        "pending_retrain_count": len(retrain),
        "pending_draft_count": len(drafts),
    }


def request_monitor_cycle(tab_id: str, host: Optional[ToolHost] = None) -> dict[str, Any]:
    return _require_host(host).queue_monitor_cycle(tab_id)


def search_knowledge(
    query: str,
    host: Optional[ToolHost] = None,
    *,
    segment: str = "auto",
) -> dict[str, Any]:
    """Search department knowledge (dual space/ground RAG) + built-in product how-to FAQ."""
    _ = host
    q_raw = (query or "").strip()
    q = q_raw.lower()
    hits: list[dict[str, Any]] = []
    resolved_segment = segment or "auto"

    # Phase 4: dual local vector indexes (space / ground)
    try:
        from app.agent.rag.dual import DualStoreRetriever

        rag = DualStoreRetriever().search(q_raw, segment=resolved_segment, top_k=6)
        for hit in rag.get("hits") or []:
            hits.append(hit)
        rag_status = rag.get("knowledge_status")
        rag_note = rag.get("note") or ""
        rag_index = rag.get("index") or {}
        resolved_segment = str(rag.get("segment") or resolved_segment)
    except Exception as exc:
        rag_status = "error"
        rag_note = f"RAG search failed: {exc}"
        rag_index = {}

    # Built-in product FAQ (always available, air-gap)
    start_keys = (
        "start",
        "monitor",
        "run tab",
        "run the tab",
        "how to run",
        "begin",
        "activate",
        "polling",
    )
    if q and any(k in q for k in start_keys):
        hits.append(
            {
                "id": "builtin-start-monitoring",
                "title": "Start monitoring a custom tab",
                "source": "builtin_product_guide",
                "snippet": (
                    "Open the custom tab → train/load at least one model → "
                    "Monitoring Controls → Start Monitoring. "
                    "There is no Start-all-tabs button; repeat per tab. "
                    "Continuous/Scheduled/On-Demand come from Edit Config. "
                    "Dashboard Ask/Analyze Fleet does not start tab monitoring."
                ),
                "score": 1.0,
            }
        )

    fleet_keys = ("fleet", "dashboard", "all tabs", "status")
    if q and any(k in q for k in fleet_keys):
        if not any(h.get("id") == "builtin-fleet-dashboard" for h in hits):
            hits.append(
                {
                    "id": "builtin-fleet-dashboard",
                    "title": "Fleet / Dashboard overview",
                    "source": "builtin_product_guide",
                    "snippet": (
                        "Use the Dashboard tab for fleet health, Pending Retrain Signals, "
                        "and Pending Agent Drafts (Approve/Reject)."
                    ),
                    "score": 0.95,
                }
            )

    if any(str(h.get("id", "")).startswith("rag-") for h in hits):
        status = "available"
    elif hits:
        status = "partial_builtin"
    else:
        status = rag_status or KNOWLEDGE_STATUS

    note_parts = []
    if rag_note:
        note_parts.append(str(rag_note))
    if any(h.get("source") == "builtin_product_guide" for h in hits):
        note_parts.append("Includes built-in product how-to.")
    if not hits:
        note_parts.append(
            "No hits. Add PDF/MD under data/knowledge and run Rebuild Knowledge Index."
        )

    return {
        "hits": hits,
        "knowledge_status": status,
        "query": q_raw,
        "segment": resolved_segment,
        "note": " ".join(note_parts).strip(),
        "index": rag_index,
    }


def _find_pending_draft(
    host: ToolHost,
    *,
    tab_id: Optional[str],
    kind: str,
) -> Optional[dict[str, Any]]:
    """Return existing pending draft for the same tab_id + kind, if any."""
    if not hasattr(host, "list_pending_drafts"):
        return None
    kind_l = str(kind or "").strip().lower()
    tid = str(tab_id or "").strip()
    try:
        drafts = host.list_pending_drafts() or []
    except Exception:
        return None
    for d in drafts:
        if not isinstance(d, dict):
            continue
        if str(d.get("kind") or "").strip().lower() != kind_l:
            continue
        if tid and str(d.get("tab_id") or "").strip() != tid:
            continue
        if not tid and d.get("tab_id"):
            continue
        return d
    return None


def _reuse_pending_draft(existing: dict[str, Any], *, kind: str) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "pending",
        "draft_id": existing.get("id"),
        "tab_id": existing.get("tab_id"),
        "kind": kind,
        "deduped": True,
        "requires_human_approval": True,
        "severity": existing.get("severity"),
        "note": f"Reused existing pending {kind} draft (same tab+kind) — no duplicate created.",
    }


def _infer_draft_severity(
    agent_reasoning: str = "",
    proposed_message: str = "",
    *,
    default: str = "WARNING",
) -> str:
    text = f"{agent_reasoning or ''} {proposed_message or ''}".lower()
    if any(k in text for k in ("critical", "obs violation", "obs_ok=false")):
        return "CRITICAL"
    if any(k in text for k in ("warning", "warn", "elevated", "drift", "anomaly")):
        return "WARNING"
    sev = str(default or "WARNING").strip().upper()
    return sev if sev in ("INFO", "WARNING", "CRITICAL") else "WARNING"


def propose_retrain(
    tab_id: str,
    reasoning: str = "",
    drift_score: float = 0.0,
    drifted_features: Optional[list] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    return _require_host(host).propose_retrain(
        tab_id,
        reasoning=reasoning or "",
        drift_score=float(drift_score or 0.0),
        drifted_features=list(drifted_features or []),
    )


def propose_alert(
    tab_id: str,
    proposed_message: str,
    agent_reasoning: str = "",
    severity: str = "WARNING",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    h = _require_host(host)
    existing = _find_pending_draft(h, tab_id=tab_id, kind="alert")
    if existing:
        return _reuse_pending_draft(existing, kind="alert")
    sev = _infer_draft_severity(
        agent_reasoning, proposed_message, default=severity or "WARNING"
    )
    return h.propose_alert(
        tab_id,
        proposed_message=proposed_message,
        agent_reasoning=agent_reasoning or "",
        severity=sev,
        kind="alert",
    )


def propose_config_change(
    tab_id: str,
    proposed_message: str,
    agent_reasoning: str = "",
    config_patch: Optional[dict] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    return _require_host(host).propose_alert(
        tab_id,
        proposed_message=proposed_message,
        agent_reasoning=agent_reasoning or "",
        severity="INFO",
        kind="config",
        proposed_payload={"config_patch": config_patch or {}},
    )


def propose_train(
    tab_id: str,
    proposed_message: str,
    agent_reasoning: str = "",
    model_id: Optional[str] = None,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if model_id:
        payload["model_id"] = str(model_id)
    return _require_host(host).propose_alert(
        tab_id,
        proposed_message=proposed_message,
        agent_reasoning=agent_reasoning or "",
        severity="INFO",
        kind="train",
        proposed_payload=payload,
    )


def propose_create_tab(
    proposed_message: str,
    title: str,
    agent_reasoning: str = "",
    data_folder: str = "",
    config: Optional[dict] = None,
    auto_suggest: bool = True,
    purpose: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    """Create a pending create_tab draft. Accepts GUI config or suggest_tab_config payload (A.3).

    Dedupes against an existing *pending* create_tab draft with the same title + data_folder
    so the LLM cannot spam 5 identical proposals in one turn.
    """
    from pathlib import Path

    from app.agent.smart_config import (
        is_weak_title,
        resolve_data_paths,
        suggestion_to_tab_config,
        suggest_tab_config,
        _ideal_title_from_hint,
    )

    raw = dict(config) if isinstance(config, dict) else {}
    # Agent may pass the whole suggest_tab_config result as `config`
    if raw.get("features") or raw.get("tab_name") or raw.get("window_size"):
        cfg = suggestion_to_tab_config(raw)
        if not agent_reasoning and raw.get("reasoning"):
            agent_reasoning = str(raw.get("reasoning") or "")
    elif isinstance(raw.get("config"), dict):
        cfg = suggestion_to_tab_config(raw)
        if not agent_reasoning and raw.get("reasoning"):
            agent_reasoning = str(raw.get("reasoning") or "")
    else:
        cfg = dict(raw)

    path_in = str(cfg.get("source_file") or data_folder or cfg.get("data_folder") or "").strip()
    resolved = (
        resolve_data_paths(path_in)
        if path_in
        else {"ok": False, "data_folder": "", "source_file": "", "title_hint": "", "error": "empty_path"}
    )
    if path_in and not resolved.get("ok"):
        return {
            "ok": False,
            "error": "data_folder_not_found",
            "data_folder": resolved.get("data_folder") or path_in,
            "source_file": resolved.get("source_file") or "",
            "note": (
                f"Data folder does not exist: {resolved.get('data_folder') or path_in}. "
                "Use a real folder or a CSV under data/ (e.g. data/BatteryTemperature.csv)."
            ),
            "requires_human_approval": False,
        }
    if resolved.get("data_folder"):
        cfg["data_folder"] = resolved["data_folder"]
        data_folder = resolved["data_folder"]
    if resolved.get("source_file"):
        cfg["source_file"] = resolved["source_file"]

    # Ideal title: never keep weak names like "Monitoring"
    ideal = _ideal_title_from_hint(
        resolved.get("title_hint")
        or (Path(resolved["source_file"]).stem if resolved.get("source_file") else ""),
        fallback="Health Monitoring",
    )
    if title and not is_weak_title(title):
        cfg["title"] = str(title).strip()
    elif cfg.get("title") and not is_weak_title(str(cfg.get("title"))):
        pass
    else:
        cfg["title"] = ideal
        title = ideal

    # Enrich empty/partial configs from data inspection
    needs_suggest = auto_suggest and bool(data_folder or cfg.get("data_folder")) and (
        not cfg.get("selected_features") or not cfg.get("models") or is_weak_title(str(cfg.get("title") or ""))
    )
    if needs_suggest:
        try:
            sug = suggest_tab_config(
                title=str(cfg.get("title") or title or ideal),
                data_folder=str(cfg.get("source_file") or cfg.get("data_folder") or data_folder),
                purpose=purpose or str(cfg.get("title") or title or ""),
            )
            if sug.get("ok") and isinstance(sug.get("config"), dict):
                merged = dict(sug["config"])
                for k, v in cfg.items():
                    if v not in (None, "", [], {}) and k != "title":
                        merged[k] = v
                # Prefer suggested ideal title over weak caller title
                if is_weak_title(str(cfg.get("title") or "")):
                    merged["title"] = sug.get("tab_name") or merged.get("title") or ideal
                else:
                    merged["title"] = cfg["title"]
                cfg = merged
                if not agent_reasoning:
                    agent_reasoning = sug.get("reasoning") or ""
        except Exception:
            pass

    # Final title guard
    if is_weak_title(str(cfg.get("title") or "")):
        cfg["title"] = ideal

    # Final path guard after suggest merge (must exist)
    final_folder = str(cfg.get("data_folder") or "").strip()
    if not final_folder or not Path(final_folder).is_dir():
        return {
            "ok": False,
            "error": "data_folder_not_found",
            "data_folder": final_folder or path_in,
            "note": (
                f"Data folder does not exist: {final_folder or path_in or '(empty)'}. "
                "Pass an existing CSV folder or file path."
            ),
            "requires_human_approval": False,
        }

    # Attach SOP / mission-procedure hits into draft reasoning when missing
    reasoning = str(agent_reasoning or "").strip()
    if "SOP" not in reasoning and "sop" not in reasoning.lower():
        try:
            intent = str(cfg.get("title") or title or purpose or "monitoring")
            rag = search_knowledge(
                f"{intent} mission operation procedure SOP anomaly response",
                host=host,
            )
            hits = rag.get("hits") or []
            snippets = []
            for hit in hits[:3]:
                if not isinstance(hit, dict):
                    continue
                bit = str(hit.get("snippet") or hit.get("title") or "").strip()
                if bit:
                    snippets.append(bit[:160])
            if snippets:
                reasoning = (reasoning + " " if reasoning else "") + (
                    "SOP/context: " + "; ".join(snippets)
                )
        except Exception:
            pass
    agent_reasoning = reasoning

    h = _require_host(host)
    title_key = str(cfg.get("title") or title or "").strip().lower()
    folder_key = str(cfg.get("data_folder") or "").strip().lower()
    source_key = str(cfg.get("source_file") or "").strip().lower()

    # Dedupe: reuse pending draft with same title + folder/source
    try:
        pending = h.list_pending_drafts() if hasattr(h, "list_pending_drafts") else []
        for d in pending or []:
            if str(d.get("kind") or "").lower() != "create_tab":
                continue
            payload = d.get("proposed_payload") or {}
            if not isinstance(payload, dict):
                payload = {}
            existing_cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
            et = str(existing_cfg.get("title") or "").strip().lower()
            ef = str(existing_cfg.get("data_folder") or "").strip().lower()
            es = str(existing_cfg.get("source_file") or "").strip().lower()
            same_src = (source_key and es == source_key) or (folder_key and ef == folder_key)
            if title_key and et == title_key and (same_src or not folder_key):
                return {
                    "ok": True,
                    "status": "pending",
                    "draft_id": d.get("id"),
                    "kind": "create_tab",
                    "deduped": True,
                    "requires_human_approval": True,
                    "note": "Reused existing pending create_tab draft (same title/folder).",
                    "proposed_payload": payload,
                }
    except Exception:
        pass

    return h.propose_alert(
        None,
        proposed_message=proposed_message,
        agent_reasoning=agent_reasoning or "",
        severity="INFO",
        kind="create_tab",
        proposed_payload={"config": cfg},
    )


def _resolve_tab_id(
    host: ToolHost,
    *,
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve tab_id from id or fuzzy title match against live tabs."""
    tid = str(tab_id or "").strip()
    title_q = str(tab_title or "").strip()
    tabs = host.list_tabs() or []
    by_id = {str(t.get("tab_id")): t for t in tabs if t.get("tab_id")}

    if tid and tid in by_id:
        t = by_id[tid]
        return {
            "ok": True,
            "tab_id": tid,
            "title": t.get("title") or tid,
            "matches": 1,
        }

    if not title_q and tid:
        return {
            "ok": False,
            "error": "tab_not_found",
            "tab_id": tid,
            "available": [
                {"tab_id": t.get("tab_id"), "title": t.get("title")} for t in tabs
            ],
        }

    if not title_q:
        return {
            "ok": False,
            "error": "tab_id_or_tab_title_required",
            "available": [
                {"tab_id": t.get("tab_id"), "title": t.get("title")} for t in tabs
            ],
        }

    needle = title_q.lower()
    exact = []
    partial = []
    for t in tabs:
        name = str(t.get("title") or "").strip()
        low = name.lower()
        if low == needle:
            exact.append(t)
        elif needle in low or low in needle:
            partial.append(t)
    matches = exact or partial
    if len(matches) == 1:
        t = matches[0]
        return {
            "ok": True,
            "tab_id": t.get("tab_id"),
            "title": t.get("title"),
            "matches": 1,
        }
    if not matches:
        return {
            "ok": False,
            "error": "tab_title_not_found",
            "tab_title": title_q,
            "available": [
                {"tab_id": t.get("tab_id"), "title": t.get("title")} for t in tabs
            ],
        }
    return {
        "ok": False,
        "error": "ambiguous_tab_title",
        "tab_title": title_q,
        "candidates": [
            {"tab_id": t.get("tab_id"), "title": t.get("title")} for t in matches
        ],
    }


def propose_start_monitoring(
    proposed_message: str,
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    agent_reasoning: str = "",
    severity: str = "INFO",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    tid = resolved["tab_id"]
    existing = _find_pending_draft(h, tab_id=tid, kind="start_monitoring")
    if existing:
        out = _reuse_pending_draft(existing, kind="start_monitoring")
        out["resolved_title"] = resolved.get("title")
        return out
    out = h.propose_alert(
        tid,
        proposed_message=proposed_message
        or f"Start monitoring on {resolved.get('title') or tid}",
        agent_reasoning=agent_reasoning or "",
        severity=severity or "INFO",
        kind="start_monitoring",
        proposed_payload={"action": "start", "title": resolved.get("title")},
    )
    out["resolved_title"] = resolved.get("title")
    return out


def propose_stop_monitoring(
    proposed_message: str,
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    agent_reasoning: str = "",
    severity: str = "WARNING",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    tid = resolved["tab_id"]
    existing = _find_pending_draft(h, tab_id=tid, kind="stop_monitoring")
    if existing:
        out = _reuse_pending_draft(existing, kind="stop_monitoring")
        out["resolved_title"] = resolved.get("title")
        return out
    msg = proposed_message or f"Stop monitoring on {resolved.get('title') or tid}"
    sev = _infer_draft_severity(agent_reasoning, msg, default=severity or "WARNING")
    out = h.propose_alert(
        tid,
        proposed_message=msg,
        agent_reasoning=agent_reasoning or "",
        severity=sev,
        kind="stop_monitoring",
        proposed_payload={"action": "stop", "title": resolved.get("title")},
    )
    out["resolved_title"] = resolved.get("title")
    return out


def propose_remove_model(
    model_id: str,
    proposed_message: str,
    tab_id: Optional[str] = None,
    tab_title: Optional[str] = None,
    agent_reasoning: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    h = _require_host(host)
    resolved = _resolve_tab_id(h, tab_id=tab_id, tab_title=tab_title)
    if not resolved.get("ok"):
        return resolved
    tid = str(resolved["tab_id"])
    mid = str(model_id or "").strip()
    if not mid:
        return {"ok": False, "error": "model_id_required"}
    detail = h.get_model_metrics(tid, mid)
    if not detail.get("ok") and detail.get("status") == "model_not_found":
        return detail
    out = h.propose_alert(
        tid,
        proposed_message=proposed_message or f"Remove model {mid}",
        agent_reasoning=agent_reasoning or "",
        severity="WARNING",
        kind="remove_model",
        proposed_payload={
            "model_id": mid,
            "model_type": detail.get("model_type"),
            "title": resolved.get("title"),
        },
    )
    out["resolved_title"] = resolved.get("title")
    out["model_id"] = mid
    return out


def list_knowledge_documents(host: Optional[ToolHost] = None) -> dict[str, Any]:
    from app.agent.docs_io import list_knowledge_documents as _list

    _ = host  # host unused; docs are filesystem-backed
    return _list()


def read_knowledge_document(filename: str, host: Optional[ToolHost] = None) -> dict[str, Any]:
    from app.agent.docs_io import read_knowledge_document as _read

    _ = host
    return _read(filename)


def propose_write_document(
    filename: str,
    content: str,
    proposed_message: str,
    agent_reasoning: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.docs_io import list_knowledge_documents as _list
    from app.agent.docs_io import sanitize_filename

    try:
        name = sanitize_filename(filename)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "filename": filename}
    existing = {d["filename"] for d in (_list().get("documents") or [])}
    if name in existing:
        return {
            "ok": False,
            "error": "file_already_exists",
            "filename": name,
            "hint": "Use propose_update_document instead.",
        }
    body = content if isinstance(content, str) else str(content or "")
    if not body.strip():
        return {"ok": False, "error": "content_required"}
    return _require_host(host).propose_alert(
        None,
        proposed_message=proposed_message or f"Create knowledge doc {name}",
        agent_reasoning=agent_reasoning or "",
        severity="INFO",
        kind="write_document",
        proposed_payload={"filename": name, "content": body, "action": "write"},
    )


def propose_update_document(
    filename: str,
    content: str,
    proposed_message: str,
    agent_reasoning: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.docs_io import list_knowledge_documents as _list
    from app.agent.docs_io import sanitize_filename

    try:
        name = sanitize_filename(filename)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "filename": filename}
    existing = {d["filename"] for d in (_list().get("documents") or [])}
    if name not in existing:
        return {
            "ok": False,
            "error": "file_not_found",
            "filename": name,
            "available": sorted(existing),
            "hint": "Use propose_write_document to create a new file.",
        }
    body = content if isinstance(content, str) else str(content or "")
    if not body.strip():
        return {"ok": False, "error": "content_required"}
    return _require_host(host).propose_alert(
        None,
        proposed_message=proposed_message or f"Update knowledge doc {name}",
        agent_reasoning=agent_reasoning or "",
        severity="INFO",
        kind="update_document",
        proposed_payload={"filename": name, "content": body, "action": "update"},
    )


def propose_write_sop_from_user_text(
    user_message: str,
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    """Best-effort parse of 'Create SOP … SOP-9' style asks into propose_write_sop."""
    import re

    text = (user_message or "").strip()
    low = text.lower()
    sop_id = ""
    m_id = re.search(r"\bsop[-\s]?(\d+[a-zA-Z0-9_-]*)\b", low, flags=re.I)
    if m_id:
        sop_id = f"SOP-{m_id.group(1).upper()}" if not m_id.group(0).upper().startswith("SOP") else m_id.group(0).upper().replace(" ", "-")
        # normalize SOP-9
        sop_id = re.sub(r"(?i)^sop[-\s]?", "SOP-", sop_id)

    title = ""
    m_named = re.search(
        r"(?:named|called|title)\s+([^,\n]+?)(?:\s*,\s*|\s+sop\b|$)",
        text,
        flags=re.I,
    )
    if m_named:
        title = m_named.group(1).strip(" .")
    if not title:
        # "Create SOP procedure named Eclipse monitoring, SOP-9"
        m2 = re.search(r"procedure\s+(.+?)(?:,\s*sop|\s+sop-|\s*$)", text, flags=re.I)
        if m2:
            title = m2.group(1).strip(" .")
            title = re.sub(r"(?i)^named\s+", "", title).strip()
    if not title and sop_id:
        title = sop_id
    if not title:
        title = "New Procedure"

    fields = {
        "process_title": title,
        "sop_id": sop_id or "SOP-NEW",
        "department": "",
        "contact_info": "",
        "effective_date": "",
        "revision_number": "1",
        "process_description": f"Procedure for {title}",
        "purpose_scope": "Operators and analysts using STDMS monitoring tabs.",
        "definitions_related": "See STDMS SOP Word template (stdms_sop_word_v1).",
        "steps": [
            {"wbs": "1", "task": "Open related monitoring tab", "owner": "Operator"},
            {"wbs": "2", "task": "Verify models are trained", "owner": "Analyst"},
            {"wbs": "3", "task": "Start monitoring and review health/drift", "owner": "Operator"},
        ],
    }
    filename = f"{fields['sop_id']}.docx"
    return propose_write_sop(
        sop_fields=fields,
        proposed_message=f"Create SOP Word: {title} ({fields['sop_id']})",
        filename=filename,
        agent_reasoning=f"Auto-parsed from operator ask: {text[:300]}",
        host=host,
    )


def propose_write_sop(
    sop_fields: Optional[dict] = None,
    proposed_message: str = "",
    filename: str = "",
    agent_reasoning: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.docs_io import list_knowledge_documents as _list
    from app.agent.docs_io import sanitize_filename
    from app.agent.sop_template import (
        SOP_TEMPLATE_ID,
        normalize_sop_fields,
        suggest_sop_filename,
        sop_fields_to_preview_markdown,
    )

    fields = normalize_sop_fields(sop_fields if isinstance(sop_fields, dict) else {})
    if not (fields.get("process_title") or fields.get("sop_id")):
        return {
            "ok": False,
            "error": "process_title_or_sop_id_required",
            "template_id": SOP_TEMPLATE_ID,
        }
    name = (filename or "").strip() or suggest_sop_filename(fields)
    if not name.lower().endswith(".docx"):
        name = Path(name).stem + ".docx"
    try:
        name = sanitize_filename(name)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "filename": name}
    existing = {d["filename"] for d in (_list().get("documents") or [])}
    if name in existing:
        return {
            "ok": False,
            "error": "file_already_exists",
            "filename": name,
            "hint": "Use propose_update_sop instead.",
        }
    preview = sop_fields_to_preview_markdown(fields)
    return _require_host(host).propose_alert(
        None,
        proposed_message=proposed_message or f"Create SOP Word doc {name}",
        agent_reasoning=agent_reasoning or "",
        severity="INFO",
        kind="write_sop",
        proposed_payload={
            "filename": name,
            "sop_fields": fields,
            "action": "write",
            "template_id": SOP_TEMPLATE_ID,
            "preview_markdown": preview,
        },
    )


def propose_update_sop(
    filename: str,
    sop_fields: Optional[dict] = None,
    proposed_message: str = "",
    agent_reasoning: str = "",
    host: Optional[ToolHost] = None,
) -> dict[str, Any]:
    from app.agent.docs_io import list_knowledge_documents as _list
    from app.agent.docs_io import sanitize_filename
    from app.agent.sop_template import (
        SOP_TEMPLATE_ID,
        normalize_sop_fields,
        sop_fields_to_preview_markdown,
    )

    fields = normalize_sop_fields(sop_fields if isinstance(sop_fields, dict) else {})
    name = (filename or "").strip()
    if not name.lower().endswith(".docx"):
        name = Path(name).stem + ".docx"
    try:
        name = sanitize_filename(name)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "filename": filename}
    existing = {d["filename"] for d in (_list().get("documents") or [])}
    if name not in existing:
        return {
            "ok": False,
            "error": "file_not_found",
            "filename": name,
            "available": sorted(f for f in existing if f.lower().endswith(".docx")),
            "hint": "Use propose_write_sop to create a new SOP.",
        }
    preview = sop_fields_to_preview_markdown(fields)
    return _require_host(host).propose_alert(
        None,
        proposed_message=proposed_message or f"Update SOP Word doc {name}",
        agent_reasoning=agent_reasoning or "",
        severity="INFO",
        kind="update_sop",
        proposed_payload={
            "filename": name,
            "sop_fields": fields,
            "action": "update",
            "template_id": SOP_TEMPLATE_ID,
            "preview_markdown": preview,
        },
    )


_HANDLERS: dict[str, ToolHandler] = {
    "get_all_snapshots": lambda host=None, **_: get_all_snapshots(host=host),
    "get_tab_snapshot": lambda tab_id, host=None, **_: get_tab_snapshot(tab_id, host=host),
    "get_tab_history": lambda tab_id, n=50, host=None, **_: get_tab_history(tab_id, n=int(n), host=host),
    "check_pending_retrain_signals": lambda host=None, **_: check_pending_retrain_signals(host=host),
    "get_fleet_status": lambda host=None, **_: get_fleet_status(host=host),
    "get_tab_detail": lambda tab_id, n=20, host=None, **_: get_tab_detail(tab_id, n=int(n), host=host),
    "list_tab_models": lambda tab_id=None, tab_title=None, host=None, **_: list_tab_models(
        tab_id=tab_id, tab_title=tab_title, host=host
    ),
    "get_model_metrics": lambda model_id="", tab_id=None, tab_title=None, host=None, **_: get_model_metrics(
        model_id=str(model_id or ""),
        tab_id=tab_id,
        tab_title=tab_title,
        host=host,
    ),
    "inspect_data_folder": lambda data_folder="", host=None, **_: inspect_data_folder(
        str(data_folder or ""), host=host
    ),
    "suggest_tab_config": lambda title="", data_folder="", purpose="", host=None, **_: suggest_tab_config(
        title=str(title or ""),
        data_folder=str(data_folder or ""),
        purpose=str(purpose or ""),
        host=host,
    ),
    "compare_tab_models": lambda tab_id=None, tab_title=None, host=None, **_: compare_tab_models_tool(
        tab_id=tab_id, tab_title=tab_title, host=host
    ),
    "propose_model_plan": lambda purpose="health", host=None, **_: propose_model_plan_tool(
        purpose=str(purpose or "health"), host=host
    ),
    "suggest_model_optimization": lambda tab_id=None, tab_title=None, host=None, **_: suggest_model_optimization(
        tab_id=tab_id, tab_title=tab_title, host=host
    ),
    "explain_anomaly": lambda tab_id=None, tab_title=None, host=None, **_: explain_anomaly_tool(
        tab_id=tab_id, tab_title=tab_title, host=host
    ),
    "detect_patterns": lambda tab_id=None, tab_title=None, host=None, **_: detect_patterns_tool(
        tab_id=tab_id, tab_title=tab_title, host=host
    ),
    "forecast_risk": lambda tab_id=None, tab_title=None, host=None, **_: forecast_risk_tool(
        tab_id=tab_id, tab_title=tab_title, host=host
    ),
    "get_pending_actions_summary": lambda host=None, **_: get_pending_actions_summary(host=host),
    "list_watchlist": lambda host=None, **_: list_watchlist(host=host),
    "add_watchlist_tab": lambda tab_id=None, tab_title=None, note="", host=None, **_: add_watchlist_tab(
        tab_id=tab_id, tab_title=tab_title, note=str(note or ""), host=host
    ),
    "remove_watchlist_tab": lambda tab_id=None, tab_title=None, host=None, **_: remove_watchlist_tab(
        tab_id=tab_id, tab_title=tab_title, host=host
    ),
    "run_anomaly_check": lambda tab_id, host=None, **_: run_anomaly_check(tab_id, host=host),
    "run_forecast": lambda tab_id, host=None, **_: run_forecast(tab_id, host=host),
    "check_drift": lambda tab_id, host=None, **_: check_drift(tab_id, host=host),
    "get_pending_signals": lambda host=None, **_: get_pending_signals(host=host),
    "request_monitor_cycle": lambda tab_id, host=None, **_: request_monitor_cycle(tab_id, host=host),
    "search_knowledge": lambda query="", segment="auto", host=None, **_: search_knowledge(
        str(query or ""), host=host, segment=str(segment or "auto")
    ),
    "propose_retrain": lambda tab_id, reasoning="", drift_score=0.0, drifted_features=None, host=None, **_: propose_retrain(
        tab_id,
        reasoning=str(reasoning or ""),
        drift_score=float(drift_score or 0.0),
        drifted_features=drifted_features,
        host=host,
    ),
    "propose_alert": lambda tab_id, proposed_message="", agent_reasoning="", severity="WARNING", host=None, **_: propose_alert(
        tab_id,
        proposed_message=str(proposed_message or ""),
        agent_reasoning=str(agent_reasoning or ""),
        severity=str(severity or "WARNING"),
        host=host,
    ),
    "propose_config_change": lambda tab_id, proposed_message="", agent_reasoning="", config_patch=None, host=None, **_: propose_config_change(
        tab_id,
        proposed_message=str(proposed_message or ""),
        agent_reasoning=str(agent_reasoning or ""),
        config_patch=config_patch if isinstance(config_patch, dict) else {},
        host=host,
    ),
    "propose_train": lambda tab_id, proposed_message="", agent_reasoning="", model_id=None, host=None, **_: propose_train(
        tab_id,
        proposed_message=str(proposed_message or ""),
        agent_reasoning=str(agent_reasoning or ""),
        model_id=model_id,
        host=host,
    ),
    "propose_create_tab": lambda proposed_message="", title="", agent_reasoning="", data_folder="", config=None, purpose="", auto_suggest=True, host=None, **_: propose_create_tab(
        proposed_message=str(proposed_message or ""),
        title=str(title or ""),
        agent_reasoning=str(agent_reasoning or ""),
        data_folder=str(data_folder or ""),
        config=config if isinstance(config, dict) else {},
        purpose=str(purpose or ""),
        auto_suggest=bool(auto_suggest),
        host=host,
    ),
    "propose_start_monitoring": lambda proposed_message="", tab_id=None, tab_title=None, agent_reasoning="", severity="INFO", host=None, **_: propose_start_monitoring(
        proposed_message=str(proposed_message or ""),
        tab_id=tab_id,
        tab_title=tab_title,
        agent_reasoning=str(agent_reasoning or ""),
        severity=str(severity or "INFO"),
        host=host,
    ),
    "propose_stop_monitoring": lambda proposed_message="", tab_id=None, tab_title=None, agent_reasoning="", severity="WARNING", host=None, **_: propose_stop_monitoring(
        proposed_message=str(proposed_message or ""),
        tab_id=tab_id,
        tab_title=tab_title,
        agent_reasoning=str(agent_reasoning or ""),
        severity=str(severity or "WARNING"),
        host=host,
    ),
    "propose_remove_model": lambda model_id="", proposed_message="", tab_id=None, tab_title=None, agent_reasoning="", host=None, **_: propose_remove_model(
        model_id=str(model_id or ""),
        proposed_message=str(proposed_message or ""),
        tab_id=tab_id,
        tab_title=tab_title,
        agent_reasoning=str(agent_reasoning or ""),
        host=host,
    ),
    "list_knowledge_documents": lambda host=None, **_: list_knowledge_documents(host=host),
    "read_knowledge_document": lambda filename="", host=None, **_: read_knowledge_document(
        str(filename or ""), host=host
    ),
    "propose_write_document": lambda filename="", content="", proposed_message="", agent_reasoning="", host=None, **_: propose_write_document(
        filename=str(filename or ""),
        content=str(content or ""),
        proposed_message=str(proposed_message or ""),
        agent_reasoning=str(agent_reasoning or ""),
        host=host,
    ),
    "propose_update_document": lambda filename="", content="", proposed_message="", agent_reasoning="", host=None, **_: propose_update_document(
        filename=str(filename or ""),
        content=str(content or ""),
        proposed_message=str(proposed_message or ""),
        agent_reasoning=str(agent_reasoning or ""),
        host=host,
    ),
    "propose_write_sop": lambda sop_fields=None, proposed_message="", filename="", agent_reasoning="", host=None, **_: propose_write_sop(
        sop_fields=sop_fields if isinstance(sop_fields, dict) else {},
        proposed_message=str(proposed_message or ""),
        filename=str(filename or ""),
        agent_reasoning=str(agent_reasoning or ""),
        host=host,
    ),
    "propose_update_sop": lambda filename="", sop_fields=None, proposed_message="", agent_reasoning="", host=None, **_: propose_update_sop(
        filename=str(filename or ""),
        sop_fields=sop_fields if isinstance(sop_fields, dict) else {},
        proposed_message=str(proposed_message or ""),
        agent_reasoning=str(agent_reasoning or ""),
        host=host,
    ),
}

ALLOWED_TOOL_NAMES = frozenset(_HANDLERS.keys())


def invoke_tool(
    name: str,
    params: Optional[dict[str, Any]] = None,
    *,
    host: Optional[ToolHost] = None,
) -> Any:
    """Dispatch a registered tool. Propose tools write pending drafts only."""
    if name not in ALLOWED_TOOL_NAMES:
        raise KeyError(f"Unknown or disallowed tool: {name}")
    # Hard-block true destructive names if ever registered without approval path
    if name in ("train_model", "send_alert", "apply_config"):
        raise PermissionError(f"Destructive tool blocked: {name}")
    handler = _HANDLERS[name]
    args = dict(params or {})
    args["host"] = host
    return handler(**args)


def tool_result_json(value: Any, *, limit: int = 12000) -> str:
    """Serialize tool output for the LLM (bounded size)."""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[: limit - 20] + "…[truncated]"
    return text
