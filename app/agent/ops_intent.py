"""Operator intent for SatOps — understand the situation, then pick tools.

Classification is about *what the operator needs* (live tab state, documents,
a SOP draft), not whether they used command verbs like show/check/how-many.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("STDMS.Agent.OpsIntent")


class OpsIntent(str, Enum):
    CHAT = "chat"
    KNOWLEDGE = "knowledge"
    FLEET = "fleet"
    TAB_DETAIL = "tab_detail"
    TAB_HISTORY = "tab_history"
    TAB_MODELS = "tab_models"
    COMPARE_MODELS = "compare_models"
    PENDING_RETRAIN = "pending_retrain"
    PENDING_ACTIONS = "pending_actions"
    INSPECT_FOLDER = "inspect_folder"
    FORECAST = "forecast"
    EXPLAIN_LIVE = "explain_live"
    WRITE_SOP = "write_sop"
    CREATE_TAB = "create_tab"
    MUTATE = "mutate"


LIVE_INTENTS = frozenset(
    {
        OpsIntent.FLEET,
        OpsIntent.TAB_DETAIL,
        OpsIntent.TAB_HISTORY,
        OpsIntent.TAB_MODELS,
        OpsIntent.COMPARE_MODELS,
        OpsIntent.PENDING_RETRAIN,
        OpsIntent.PENDING_ACTIONS,
        OpsIntent.INSPECT_FOLDER,
        OpsIntent.FORECAST,
        OpsIntent.EXPLAIN_LIVE,
        OpsIntent.WRITE_SOP,
        OpsIntent.CREATE_TAB,
        OpsIntent.MUTATE,
    }
)

# FSM / generator labels — never treat these as tab ids unless a tab is titled that.
MISSION_MODES = frozenset(
    {
        "eclipse",
        "nominal",
        "imaging",
        "sunlight",
        "safe",
        "safehold",
        "safe_hold",
        "maneuver",
        "commissioning",
        "transfer",
    }
)

_STOP_TOKENS = frozenset(
    {
        "which",
        "what",
        "that",
        "this",
        "with",
        "from",
        "want",
        "know",
        "tab",
        "tabs",
        "the",
        "and",
        "for",
        "please",
        "hansı",
        "hansi",
        "nedir",
        "nədir",
        "olan",
        "var",
    }
)

_PROMISE_RE = re.compile(
    r"(?i)("
    r"i will (?:call|fetch|get|check|look|use|run)|"
    r"i(?:'| a)m going to|"
    r"let me (?:call|fetch|check|look|get)|"
    r"i(?:'| wi)ll now|"
    r"çağıraca[gğ]|yoxlayaca[gğ]|alaca[gğ]|baxaca[gğ]|istifadə edəcə|"
    r"indi (?:çağır|yoxla|alaca)"
    r")"
)


class _Topic:
    """Meaning clusters — topics and entities, not operator command verbs."""

    LIVE_STATE = (
        "health",
        "fusion",
        "drift",
        "alert",
        "monitoring",
        "anomaly",
        "anomal",
        "warning",
        "critical",
        "healthy",
        "snapshot",
        "attention",
        "diqqet",
        "diqqət",
        "fleet",
        "status",
        "veziyyet",
        "vəziyyət",
        "sağlam",
        "saglam",
        "xəbərdar",
        "xeberdar",
        "monitorinq",
    )
    HISTORY = (
        "event",
        "history",
        "hadis",
        "tarixç",
        "tarixc",
        "son hadis",
        "recent alert",
        "last event",
        "son 10",
        "last 10",
    )
    MODELS = ("model", "trained", "ml ", "xgboost", "lstm", "isolation")
    COMPARE = ("best model", "compare model", "ən yaxşı model", "en yaxsi model", "hansı model yaxşı")
    RETRAIN = (
        "retrain",
        "yenidən öyrət",
        "yeniden oyret",
        "pending retrain",
        "gözləyən retrain",
        "gozleyen retrain",
        "retrain signal",
    )
    PENDING_DRAFT = (
        "pending draft",
        "pending action",
        "gözləyən draft",
        "approve draft",
        "pending agent",
    )
    FORECAST = (
        "forecast",
        "ttf",
        "time to fail",
        "time-to-fail",
        "risk",
        "proqnoz",
        "uğursuz",
        "ugursuz",
    )
    FOLDER = (
        ".csv",
        "data/",
        "sda_feed",
        "qovluq",
        "folder",
        "data folder",
        "data type",
        "reconfig",
        "yenidən qur",
        "yeniden qur",
    )
    WRITE_SOP = (
        "write sop",
        "create sop",
        "new sop",
        "update sop",
        "write procedure",
        "create procedure",
        "propose_write_sop",
        "yazmağı təklif",
        "yazmagi teklif",
        "sop yaz",
        "prosedur yaz",
        "sop yarat",
        "prosedur yarat",
    )
    CREATE_TAB = (
        "create tab",
        "create a tab",
        "new tab",
        "make a tab",
        "tab yarat",
        "tab yaratmaq",
        "propose_create_tab",
    )
    MUTATE = (
        "start monitoring",
        "stop monitoring",
        "start monitor",
        "stop monitor",
        "train model",
        "remove model",
        "işə sal",
        "ise sal",
        "dayandır",
        "dayandir",
    )
    DOC_DEF = (
        "procedure",
        "sop",
        "prosedur",
        "runbook",
        "documentation",
        "according to",
        "in the sop",
        "who is",
        "who are",
        "kimdir",
        "haqqında",
        "haqqinda",
        "what is the",
        "what are the",
        "necə işləyir",
        "nece isleyir",
        "how does",
        "how do i",
        "hardware",
        "software",
        "manual",
    )
    GREET = (
        "salam",
        "hello",
        "hi ",
        "hola",
        "hey",
        "thanks",
        "təşəkkür",
        "tesekkur",
        "who are you",
        "who r you",
        "sən kimsən",
        "sen kimsen",
        "bu node",
        "who you are",
    )


@dataclass
class ResolvedTab:
    tab_id: str
    title: str
    snapshot: dict[str, Any] = field(default_factory=dict)


def operator_query(text: str) -> str:
    """Strip pipeline wrappers so FSM mode cannot pollute intent / tab matching."""
    raw = (text or "").strip()
    if not raw:
        return ""
    m = re.match(r"(?is)^operator:\s*(.+?)(?:\n\s*\n|\n\[|$)", raw)
    if m:
        return m.group(1).strip()
    # Drop trailing injected blocks if present
    raw = re.split(r"\n\[(?:FSM|M-LLM|Resolved tab)\]", raw, maxsplit=1)[0]
    return raw.strip()


def _norm(text: str) -> str:
    q = (text or "").lower().replace("ı", "i").replace("ə", "e")
    q = q.replace("ö", "o").replace("ü", "u").replace("ş", "s").replace("ğ", "g").replace("ç", "c")
    return re.sub(r"\s+", " ", q).strip()


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(text))


def _contains_any(hay: str, needles: tuple[str, ...]) -> bool:
    h = _norm(hay)
    c = _compact(hay)
    for n in needles:
        nn = _norm(n)
        if nn and nn in h:
            return True
        cn = _compact(n)
        if len(cn) >= 5 and cn in c:
            return True
    return False


def looks_azerbaijani(text: str) -> bool:
    q = text or ""
    if re.search(r"[əğıöüşçƏĞİÖÜŞÇ]", q):
        return True
    return _contains_any(
        q,
        ("tabi", "hansi", "saglam", "hadis", "prosedur", "teklif", "qovluq", "gozleyen"),
    )


def is_write_sop_intent(message: str) -> bool:
    q = operator_query(message)
    n = _norm(q)
    if _contains_any(q, _Topic.WRITE_SOP):
        return True
    if ("sop" in n or "procedure" in n or "prosedur" in n) and any(
        w in n for w in ("yaz", "create", "write", "update", "crate", "təklif", "teklif", "propose", "yarat")
    ):
        return True
    return False


def is_create_tab_intent(message: str) -> bool:
    q = operator_query(message)
    if is_write_sop_intent(q):
        return False
    return _contains_any(q, _Topic.CREATE_TAB)


def _is_tiny_greeting(q: str) -> bool:
    n = re.sub(r"^(ok(ay)?|got it|sure|alright|beli)[,.\s]+", "", _norm(q)).strip(" !.?,")
    if not n or len(n) <= 2:
        return True
    return n in ("salam", "hello", "hi", "hola", "hey", "thanks", "ok", "okay", "tesekkur")


_GREET_WORD_RE = re.compile(
    r"(?i)\b("
    r"salam|hello|hola|hey|thanks|tesekkur|"
    r"who are you|who r you|who you are|"
    r"sen kimsen|bu node"
    r")\b"
)


def _is_obvious_chat(q: str) -> bool:
    n = _norm(q)
    if _is_tiny_greeting(q):
        return True
    # "okay …" must not steal an ops question; "hi" must not match inside "which"
    if len(n) > 28:
        return False
    if _GREET_WORD_RE.search(n) and not _has_filesystem_target(q) and not _looks_like_telemetry_dump(q):
        return True
    return False


def _looks_like_telemetry_dump(q: str) -> bool:
    n = _norm(q)
    hits = sum(
        1
        for k in (
            "satellite_id",
            "battery_voltage",
            "sun_factor",
            "solar_current",
            "mission_mode",
            "packet_loss",
        )
        if k in n
    )
    return hits >= 2


def _wants_model_reconfig(q: str) -> bool:
    n = _norm(q)
    return any(
        w in n
        for w in (
            "reconfig",
            "reconfigure",
            "yeniden qur",
            "yenidən qur",
            "fine-tune",
            "finetune",
            "data type",
            "bu data",
            "this data",
        )
    ) or (
        _contains_any(q, _Topic.MODELS)
        and (_has_filesystem_target(q) or _looks_like_telemetry_dump(q) or "folder" in n or "qovluq" in n)
    )


def _is_doc_definition(q: str) -> bool:
    """Procedure/person/document meaning — not a live tab status question."""
    if is_write_sop_intent(q):
        return False
    if _contains_any(q, _Topic.GREET):
        return False
    n = _norm(q)
    if not _contains_any(q, _Topic.DOC_DEF):
        return False
    # "who are you" is identity, not a knowledge lookup
    if re.search(r"(?i)\bwho are you\b|\bwho r you\b|\bs[əe]n kims[əe]n\b", q):
        return False
    # "what is test tab health" is live, not a SOP definition
    if _contains_any(q, _Topic.LIVE_STATE) and ("tab" in n or "tabi" in n):
        return False
    if _contains_any(q, _Topic.RETRAIN + _Topic.HISTORY + _Topic.FORECAST + _Topic.MODELS):
        return False
    return True


def _has_tab_noun(q: str) -> bool:
    n = _norm(q)
    c = _compact(q)
    return "tab" in n or "tabi" in n or "tabinda" in c or "tabinda" in n


def _has_filesystem_target(q: str) -> bool:
    if re.search(r"[A-Za-z]:\\", q) or re.search(r"(?i)\.csv\b", q):
        return True
    if re.search(r"(?i)(?:^|[\s`\"'>])(?:data[/\\]|sda_feed)", q):
        return True
    n = _norm(q)
    if "sda_feed" in n or "data-genrator" in n or "datagenerator" in n:
        return True
    if _contains_any(q, _Topic.FOLDER):
        return True
    return False


def _n_from_message(q: str, default: int = 10) -> int:
    m = re.search(r"\b(\d{1,3})\b", q or "")
    if not m:
        return default
    try:
        return max(1, min(int(m.group(1)), 50))
    except ValueError:
        return default


def list_open_tabs(host: Any) -> list[dict[str, Any]]:
    if host is None:
        return []
    try:
        tabs = host.list_tabs() or []
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for t in tabs:
        if not isinstance(t, dict):
            continue
        snap = t.get("snapshot") if isinstance(t.get("snapshot"), dict) else {}
        title = str(t.get("title") or snap.get("title") or t.get("tab_id") or "").strip()
        tid = str(t.get("tab_id") or snap.get("tab_id") or "").strip()
        if not tid and not title:
            continue
        row = dict(t)
        row["title"] = title or tid
        row["tab_id"] = tid or title
        row["snapshot"] = snap or dict(t.get("snapshot") or {})
        out.append(row)
    return out


def resolve_tab(message: str, host: Any) -> Optional[ResolvedTab]:
    """Match an open tab by title / quoted name / 'X tabı'. Never by mission mode."""
    q_raw = operator_query(message)
    q = _norm(q_raw)
    tabs = list_open_tabs(host)
    if not tabs:
        return None

    candidates: list[tuple[int, dict[str, Any]]] = []

    quoted: list[str] = []
    for pat in (r"'([^']+)'", r'"([^"]+)"', r"`([^`]+)`"):
        quoted.extend(m.group(1).strip() for m in re.finditer(pat, q_raw))
    for pat in (
        r"(?i)\btab(?:ı|i|inda|ında)?\s+([A-Za-z0-9._-]{2,40})",
        r"(?i)\b([A-Za-z0-9._-]{2,40})\s+tab(?:ı|i|inda|ında)?\b",
        r"(?i)for\s+(.+?)\s+tab",
    ):
        m = re.search(pat, q_raw)
        if m:
            quoted.append(m.group(1).strip().rstrip("?.!,"))

    for tab in tabs:
        title = str(tab.get("title") or "")
        tid = str(tab.get("tab_id") or "")
        tl = _norm(title)
        tc = _compact(title)
        score = 0
        if not tl:
            continue
        if any(_norm(x) == tl or _compact(x) == tc for x in quoted if x):
            score = 100 + len(tl)
        elif tl and tl in q:
            score = 80 + len(tl)
        elif tc and len(tc) >= 3 and tc in _compact(q_raw):
            score = 70 + len(tc)
        elif tid and tid.lower() in q.split():
            score = 60
        if score:
            candidates.append((score, tab))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    return ResolvedTab(
        tab_id=str(best.get("tab_id") or ""),
        title=str(best.get("title") or ""),
        snapshot=dict(best.get("snapshot") or {}),
    )


def classify_ops_intent(
    message: str,
    *,
    host: Any = None,
    tab_titles: Optional[list[str]] = None,
) -> OpsIntent:
    q = operator_query(message)
    if not q.strip():
        return OpsIntent.CHAT
    n = _norm(q)
    titles = list(tab_titles or [])
    if host is not None and not titles:
        titles = [str(t.get("title") or "") for t in list_open_tabs(host)]
    mentions_known = False
    qc = _compact(q)
    for title in titles:
        tl = _norm(title)
        tc = _compact(title)
        if tl and (tl in n or (len(tc) >= 3 and tc in qc)):
            mentions_known = True
            break
    resolved = resolve_tab(q, host) if host is not None else None
    if resolved:
        mentions_known = True
    tab_noun = _has_tab_noun(q) or mentions_known

    if _is_tiny_greeting(q) and not _has_filesystem_target(q) and not _looks_like_telemetry_dump(q):
        return OpsIntent.CHAT
    if is_write_sop_intent(q):
        return OpsIntent.WRITE_SOP
    if is_create_tab_intent(q):
        return OpsIntent.CREATE_TAB
    if _contains_any(q, _Topic.MUTATE) and not _wants_model_reconfig(q):
        return OpsIntent.MUTATE
    if (
        _has_filesystem_target(q)
        or _looks_like_telemetry_dump(q)
        or _wants_model_reconfig(q)
    ) and not is_create_tab_intent(q):
        return OpsIntent.INSPECT_FOLDER
    if _contains_any(q, _Topic.RETRAIN):
        # "retrain these models for this data" ≠ the pending-retrain queue
        if _wants_model_reconfig(q) or _has_filesystem_target(q) or _looks_like_telemetry_dump(q):
            return OpsIntent.INSPECT_FOLDER
        if not any(w in n for w in ("pending", "gozleyen", "gözləyən", "signal", "siqnal", "queue")):
            if _contains_any(q, _Topic.MODELS) or "reconfig" in n:
                return OpsIntent.INSPECT_FOLDER
        return OpsIntent.PENDING_RETRAIN
    if _contains_any(q, _Topic.PENDING_DRAFT):
        return OpsIntent.PENDING_ACTIONS
    if _contains_any(q, _Topic.FORECAST):
        return OpsIntent.FORECAST
    if _contains_any(q, _Topic.HISTORY):
        return OpsIntent.TAB_HISTORY
    if _contains_any(q, _Topic.COMPARE) or (
        "model" in n and any(w in n for w in ("best", "compare", "yaxsi", "yaxşı", "recommend"))
    ):
        return OpsIntent.COMPARE_MODELS
    if _contains_any(q, _Topic.MODELS) and tab_noun:
        return OpsIntent.TAB_MODELS

    live = _contains_any(q, _Topic.LIVE_STATE)
    why = any(w in n for w in ("why", "niye", "niyə", "sebeb", "səbəb", "contradict", "amma", "but "))
    if live and why:
        return OpsIntent.EXPLAIN_LIVE
    if tab_noun and live:
        if mentions_known:
            return OpsIntent.TAB_DETAIL
        if any(w in n for w in ("all", "which", "hansi", "hansı", "fleet", "report", "every")):
            return OpsIntent.FLEET
        return OpsIntent.TAB_DETAIL
    if live and not _is_doc_definition(q):
        if mentions_known:
            return OpsIntent.TAB_DETAIL
        if any(w in n for w in ("fleet", "all tab", "which tab", "hansi tab", "hansı tab", "health report")):
            return OpsIntent.FLEET
        return OpsIntent.FLEET
    if _is_doc_definition(q):
        return OpsIntent.KNOWLEDGE
    if _is_obvious_chat(q):
        return OpsIntent.CHAT
    if mentions_known or (tab_noun and any(w in n for w in ("status", "veziyyet", "vəziyyət", "how is"))):
        return OpsIntent.TAB_DETAIL
    if any(w in n for w in ("which tab", "hansi tab", "hansı tab", "all tab", "fleet")):
        return OpsIntent.FLEET
    # Unknown short chatter vs unexplained ops ask
    if len(n) < 18 and not tab_noun and not live:
        return OpsIntent.CHAT
    return OpsIntent.KNOWLEDGE if _contains_any(q, ("why", "what", "how", "explain")) else OpsIntent.CHAT


def route_for_intent(intent: OpsIntent) -> str:
    if intent == OpsIntent.CHAT:
        return "chat"
    if intent == OpsIntent.KNOWLEDGE:
        return "knowledge"
    return "tool"


def required_tools(intent: OpsIntent) -> frozenset[str]:
    return {
        OpsIntent.FLEET: frozenset({"get_fleet_status", "get_all_snapshots", "list_tabs"}),
        OpsIntent.TAB_DETAIL: frozenset(
            {"get_tab_snapshot", "get_tab_detail", "get_fleet_status", "get_all_snapshots"}
        ),
        OpsIntent.TAB_HISTORY: frozenset({"get_tab_history", "get_tab_detail"}),
        OpsIntent.TAB_MODELS: frozenset({"list_tab_models", "compare_tab_models", "get_tab_detail"}),
        OpsIntent.COMPARE_MODELS: frozenset({"compare_tab_models", "list_tab_models"}),
        OpsIntent.PENDING_RETRAIN: frozenset({"check_pending_retrain_signals"}),
        OpsIntent.PENDING_ACTIONS: frozenset({"get_pending_actions_summary"}),
        OpsIntent.INSPECT_FOLDER: frozenset({"inspect_data_folder"}),
        OpsIntent.FORECAST: frozenset({"forecast_risk", "detect_patterns"}),
        OpsIntent.EXPLAIN_LIVE: frozenset(
            {"explain_anomaly", "get_tab_detail", "get_tab_snapshot", "list_tab_models"}
        ),
        OpsIntent.WRITE_SOP: frozenset({"propose_write_sop", "propose_update_sop"}),
        OpsIntent.CREATE_TAB: frozenset({"propose_create_tab"}),
        OpsIntent.MUTATE: frozenset(),
        OpsIntent.CHAT: frozenset(),
        OpsIntent.KNOWLEDGE: frozenset(),
    }.get(intent, frozenset())


def looks_like_promise(reply: Optional[str]) -> bool:
    text = (reply or "").strip()
    if not text:
        return False
    if _PROMISE_RE.search(text):
        return True
    # Model announced a tool name but produced no facts
    if re.search(r"(?i)\b(get_tab_history|inspect_data_folder|check_pending_retrain|forecast_risk)\b", text):
        if not re.search(r"(?i)\[observation\]|health=|draft_id|rows_sampled", text):
            return True
    return False


def tools_ran(tool_trace: list[dict[str, Any]], names: frozenset[str]) -> bool:
    ran = {str(t.get("tool") or "") for t in (tool_trace or []) if t.get("ok")}
    return bool(ran & names)


def needs_fulfillment(
    intent: OpsIntent,
    tool_trace: list[dict[str, Any]],
    reply: Optional[str],
) -> bool:
    """Run tools ourselves only when the model skipped them or only promised to."""
    if intent not in LIVE_INTENTS:
        return False
    req = required_tools(intent)
    if not req:
        return False
    if looks_like_promise(reply):
        return True
    if not (reply or "").strip():
        return True
    if any(t.get("ok") and t.get("tool") for t in (tool_trace or [])):
        return False
    return True


def extract_folder_path(message: str, host: Any = None) -> str:
    from pathlib import Path

    text = operator_query(message)
    for pat in (r"'([^']+)'", r'"([^"]+)"', r"`([^`]+)`"):
        m = re.search(pat, text)
        if m and (":" in m.group(1) or "/" in m.group(1) or "\\" in m.group(1) or m.group(1).lower().endswith(".csv")):
            return m.group(1).strip()
    m = re.search(r"([A-Za-z]:\\[^\s\"']+)", text)
    if m:
        return m.group(1).strip().rstrip(".,;")
    m = re.search(r"(/?(?:[\w.-]+/)*data(?:/[^\s\"']+)?)", text)
    if m:
        return m.group(1).strip().rstrip(".,;")
    m = re.search(r"([\w.-]+\.csv)", text, re.I)
    if m:
        cand = Path("data") / m.group(1)
        if cand.exists():
            return str(cand.resolve())
        return m.group(1)
    # Folder name mentioned in a live tab's data_folder
    needle = ""
    for token in ("sda_feed",):
        if token in _norm(text):
            needle = token
            break
    if host is not None:
        for tab in list_open_tabs(host):
            snap = tab.get("snapshot") or {}
            folder = str(snap.get("data_folder") or "")
            if needle and needle in folder.lower():
                return folder
            if not needle and folder:
                # Operator said "this data" with an open tab already pointed at a feed
                if _looks_like_telemetry_dump(text) or _wants_model_reconfig(text):
                    return folder
            cfg = tab.get("config") if isinstance(tab.get("config"), dict) else {}
            folder = str(cfg.get("data_folder") or "")
            if needle and needle in folder.lower():
                return folder
            if not needle and folder and (_looks_like_telemetry_dump(text) or _wants_model_reconfig(text)):
                return folder
    return ""


def sanitize_tool_args(
    name: str,
    args: dict[str, Any],
    *,
    message: str,
    host: Any,
) -> dict[str, Any]:
    """Replace hallucinated tab_id (mission mode / wrong title) with the resolved tab."""
    out = dict(args or {})
    resolved = resolve_tab(message, host)
    live_ids = {str(t.get("tab_id")) for t in list_open_tabs(host)}
    tid = str(out.get("tab_id") or "").strip()
    title = str(out.get("tab_title") or "").strip()

    def _bind(tab: ResolvedTab) -> None:
        out["tab_id"] = tab.tab_id
        out["tab_title"] = tab.title

    if name in {
        "get_tab_snapshot",
        "get_tab_history",
        "get_tab_detail",
        "list_tab_models",
        "get_model_metrics",
        "compare_tab_models",
        "explain_anomaly",
        "detect_patterns",
        "forecast_risk",
        "run_anomaly_check",
        "add_watchlist_tab",
        "remove_watchlist_tab",
        "propose_start_monitoring",
        "propose_stop_monitoring",
        "propose_train",
        "propose_remove_model",
        "propose_retrain",
        "propose_alert",
    }:
        bad_id = tid and tid not in live_ids
        mode_as_id = tid.lower() in MISSION_MODES and tid not in live_ids
        if (bad_id or mode_as_id or (not tid and not title)) and resolved:
            _bind(resolved)
        elif title and resolved and _norm(title) != _norm(resolved.title):
            # Title looks like a mission mode but operator named a real tab
            if _norm(title) in MISSION_MODES:
                _bind(resolved)
    if name == "inspect_data_folder" and not str(out.get("data_folder") or "").strip():
        path = extract_folder_path(message, host)
        if path:
            out["data_folder"] = path
    return out


def _fmt_tab_line(tab: dict[str, Any]) -> str:
    snap = tab.get("snapshot") if isinstance(tab.get("snapshot"), dict) else {}
    title = tab.get("title") or snap.get("title") or tab.get("tab_id")
    health = snap.get("health_state") or tab.get("health") or "—"
    fusion = snap.get("fusion_score")
    drift = snap.get("drift")
    alerts = snap.get("alert_count")
    mon = snap.get("monitoring_active")
    trained = snap.get("trained_models")
    mode = snap.get("mission_mode")
    bits = [f"health={health}"]
    if fusion is not None:
        bits.append(f"fusion={fusion}")
    if drift is not None:
        bits.append(f"drift={drift}")
    if alerts is not None:
        bits.append(f"alerts={alerts}")
    bits.append("monitoring=" + ("active" if mon else "idle"))
    if trained is not None:
        bits.append(f"trained_models={trained}")
    if mode:
        bits.append(f"mission_mode={mode}")
    return f"- {title} ({', '.join(str(b) for b in bits)})"


def _oar(obs: str, analysis: str, rec: str) -> str:
    return f"[Observation] {obs}\n[Analysis] {analysis}\n[Recommendation] {rec}"


def _trace(name: str, params: dict[str, Any], ok: bool, extra: Optional[dict] = None) -> dict[str, Any]:
    row = {"tool": name, "params": params, "ok": ok, "deterministic": True}
    if extra:
        row.update(extra)
    return row


def fulfill_intent(message: str, host: Any, *, intent: Optional[OpsIntent] = None) -> dict[str, Any]:
    """Execute the tools this situation needs and answer from the results."""
    q = operator_query(message)
    intent = intent or classify_ops_intent(q, host=host)
    tool_trace: list[dict[str, Any]] = []
    az = looks_azerbaijani(q)

    def fail(obs: str, analysis: str, rec: str) -> dict[str, Any]:
        return {
            "ok": True,
            "reply": _oar(obs, analysis, rec),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    if intent in (OpsIntent.CHAT, OpsIntent.KNOWLEDGE, OpsIntent.MUTATE, OpsIntent.CREATE_TAB):
        return {
            "ok": True,
            "reply": "",
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "observation_only",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    if host is None:
        return fail(
            "Tool host is not attached.",
            "Live questions need the running desktop.",
            "Open SDA and ask again.",
        )

    if intent == OpsIntent.WRITE_SOP:
        from app.agent.tools import propose_write_sop_from_user_text

        auto = propose_write_sop_from_user_text(q, host=host)
        tool_trace.append(_trace("propose_write_sop", {"auto_from_user_text": True}, bool(auto.get("ok"))))
        if auto.get("ok"):
            return {
                "ok": True,
                "reply": (
                    "Created a pending SOP Word draft. It is not written to disk until you Approve.\n"
                    f"- draft_id: {auto.get('draft_id')}\n"
                    f"- kind: {auto.get('kind')}\n"
                    "- Next: Home → AI Assistant → Pending Agent Drafts → Approve\n"
                    "After Approve, the .docx is stored under data/knowledge."
                ),
                "llm_used": False,
                "tool_trace": tool_trace,
                "outcome": "function_calling",
                "agent_mode": "tools",
                "intent": intent.value,
            }
        return fail(
            f"Could not create a SOP draft: {auto.get('error') or auto}.",
            "SOP writes stay Propose → Approve.",
            "Give a process title and SOP id, then Approve the draft.",
        )

    if intent == OpsIntent.FLEET:
        tabs = list_open_tabs(host)
        tool_trace.append(_trace("get_fleet_status", {}, True))
        if not tabs:
            return fail(
                "No monitoring tabs are currently open." if not az else "Açıq monitoring tabı yoxdur.",
                "There is nothing live to list.",
                "Open or create a tab if you need one.",
            )
        lines = "\n".join(_fmt_tab_line(t) for t in tabs)
        return {
            "ok": True,
            "reply": _oar(
                f"Open monitoring tabs ({len(tabs)}):" if not az else f"Açıq tablar ({len(tabs)}):",
                lines,
                "Ask about a tab by its title for health, events, or models.",
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    if intent == OpsIntent.PENDING_RETRAIN:
        from app.agent.tools import check_pending_retrain_signals

        try:
            signals = check_pending_retrain_signals(host=host) or []
            tool_trace.append(_trace("check_pending_retrain_signals", {}, True))
        except Exception as exc:
            tool_trace.append(_trace("check_pending_retrain_signals", {}, False, {"error": str(exc)}))
            return fail(
                "Pending retrain lookup failed.",
                str(exc),
                "Retry after the desktop host is ready.",
            )
        n = len(signals) if isinstance(signals, list) else 0
        preview = []
        for s in (signals if isinstance(signals, list) else [])[:8]:
            if isinstance(s, dict):
                preview.append(
                    f"- tab={s.get('tab_id') or s.get('title') or '?'} "
                    f"feature={s.get('feature') or s.get('drifted_feature') or '—'} "
                    f"score={s.get('drift_score', s.get('score', '—'))}"
                )
        body = "\n".join(preview) if preview else "(no signal details)"
        return {
            "ok": True,
            "reply": _oar(
                f"Pending retrain signals: {n}.",
                body,
                "Acknowledge or Approve retrains under Pending Retrain Signals — I do not auto-retrain.",
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    if intent == OpsIntent.PENDING_ACTIONS:
        from app.agent.tools import get_pending_actions_summary

        try:
            summary = get_pending_actions_summary(host=host)
            tool_trace.append(_trace("get_pending_actions_summary", {}, True))
        except Exception as exc:
            tool_trace.append(_trace("get_pending_actions_summary", {}, False, {"error": str(exc)}))
            return fail("Pending actions lookup failed.", str(exc), "Retry from Home → Pending Agent Drafts.")
        return {
            "ok": True,
            "reply": _oar(
                "Pending agent actions (live).",
                str(summary)[:1500],
                "Approve or Reject under Home → AI Assistant → Pending Agent Drafts.",
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    if intent == OpsIntent.INSPECT_FOLDER:
        from app.agent.tools import inspect_data_folder

        path = extract_folder_path(q, host)
        if not path:
            return fail(
                "No data folder or CSV path found in the question.",
                "Folder inspection needs a real path (or a tab that already points at that folder).",
                "Ask again with the folder path, e.g. the sda_feed directory.",
            )
        try:
            insp = inspect_data_folder(path, host=host)
            tool_trace.append(_trace("inspect_data_folder", {"data_folder": path}, bool(insp.get("ok"))))
        except Exception as exc:
            tool_trace.append(_trace("inspect_data_folder", {"data_folder": path}, False, {"error": str(exc)}))
            return fail(f"inspect_data_folder failed for `{path}`.", str(exc), "Check the path exists and retry.")
        cols = insp.get("columns") or insp.get("numeric_cols") or []
        numeric = insp.get("numeric_cols") or []
        suggest_txt = ""
        try:
            from app.agent.tools import suggest_tab_config

            sug = suggest_tab_config(
                title="test",
                data_folder=str(insp.get("data_folder") or path),
                purpose="LEO satellite bus telemetry (power/thermal/comms/attitude)",
                host=host,
            )
            tool_trace.append(_trace("suggest_tab_config", {"data_folder": path}, bool(sug.get("ok", True))))
            models = sug.get("models") or []
            feats = sug.get("features") or numeric
            model_bits = []
            for m in models[:4]:
                if isinstance(m, dict):
                    model_bits.append(str(m.get("model_type") or m.get("name") or m))
                else:
                    model_bits.append(str(m))
            suggest_txt = (
                f" Suggested features: {feats}. "
                f"Suggested models: {model_bits or 'Isolation Forest, Autoencoder, LSTM'}. "
                f"{sug.get('reasoning') or ''}"
            )
        except Exception as exc:
            tool_trace.append(_trace("suggest_tab_config", {"data_folder": path}, False, {"error": str(exc)}))
            suggest_txt = f" (suggest_tab_config failed: {exc})"
        return {
            "ok": True,
            "reply": _oar(
                f"Inspected `{insp.get('data_folder') or path}` — this is satellite telemetry, not a chat log.",
                f"ok={insp.get('ok')}; rows={insp.get('row_count')}; columns={cols}; "
                f"numeric={numeric}; timestamp={insp.get('timestamp_candidates')}; "
                f"notes={insp.get('quality_notes')}.{suggest_txt} "
                "Do not train on eclipse/sun_factor/mission_mode — those are orbit flags, not faults.",
                "If you want this applied to the open tab, say Approve after I create propose_train drafts. "
                "I will not silently retrain.",
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    # Tab-scoped live intents
    resolved = resolve_tab(q, host)
    tabs = list_open_tabs(host)
    if resolved is None:
        if len(tabs) == 1 and intent in {
            OpsIntent.TAB_DETAIL,
            OpsIntent.TAB_HISTORY,
            OpsIntent.TAB_MODELS,
            OpsIntent.COMPARE_MODELS,
            OpsIntent.FORECAST,
            OpsIntent.EXPLAIN_LIVE,
        }:
            only = tabs[0]
            resolved = ResolvedTab(
                tab_id=str(only.get("tab_id") or ""),
                title=str(only.get("title") or ""),
                snapshot=dict(only.get("snapshot") or {}),
            )
        else:
            names = ", ".join(str(t.get("title") or t.get("tab_id")) for t in tabs) or "(none)"
            tool_trace.append(_trace("get_fleet_status", {}, True))
            return fail(
                "Could not tell which tab you mean.",
                f"Open tabs: {names}. Mission mode names (eclipse, imaging, …) are not tab ids.",
                "Repeat the question with the tab title, e.g. the `test` tab.",
            )

    tid, title = resolved.tab_id, resolved.title

    if intent == OpsIntent.TAB_HISTORY:
        from app.agent.tools import get_tab_history

        n = _n_from_message(q, 10)
        try:
            events = get_tab_history(tid, n=n, host=host) or []
            tool_trace.append(_trace("get_tab_history", {"tab_id": tid, "n": n}, True))
        except Exception as exc:
            tool_trace.append(_trace("get_tab_history", {"tab_id": tid, "n": n}, False, {"error": str(exc)}))
            return fail(f"History for '{title}' failed.", str(exc), "Retry after a monitor cycle.")
        if not events:
            body = "No stored events yet."
        else:
            lines = []
            for ev in events[:n]:
                if isinstance(ev, dict):
                    lines.append(
                        f"- {ev.get('timestamp') or ev.get('ts') or '—'} "
                        f"{ev.get('health_state') or ev.get('status') or ''} "
                        f"{ev.get('message') or ev.get('detail') or ev.get('value') or ''}".strip()
                    )
                else:
                    lines.append(f"- {ev}")
            body = "\n".join(lines)
        return {
            "ok": True,
            "reply": _oar(
                f"Last {min(n, len(events) if events else n)} event(s) on '{title}'.",
                body,
                "These are stored cycle events, not a new monitor run.",
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    if intent in (OpsIntent.TAB_MODELS, OpsIntent.COMPARE_MODELS):
        from app.agent.tools import compare_tab_models_tool, list_tab_models

        listing = list_tab_models(tab_id=tid, tab_title=title, host=host)
        tool_trace.append(_trace("list_tab_models", {"tab_id": tid, "tab_title": title}, bool(listing.get("ok"))))
        models = listing.get("models") or listing.get("items") or []
        if intent == OpsIntent.COMPARE_MODELS:
            cmp = compare_tab_models_tool(tab_id=tid, tab_title=title, host=host)
            tool_trace.append(_trace("compare_tab_models", {"tab_id": tid}, bool(cmp.get("ok"))))
            return {
                "ok": True,
                "reply": _oar(
                    f"Compared models on '{cmp.get('title') or title}'.",
                    f"{cmp.get('recommendation') or cmp} best={cmp.get('best_model_type')} "
                    f"({cmp.get('best_model_id')}).",
                    "Retrain only after you Approve a propose_train draft.",
                ),
                "llm_used": False,
                "tool_trace": tool_trace,
                "outcome": "function_calling",
                "agent_mode": "tools",
                "intent": intent.value,
            }
        if not listing.get("ok"):
            return fail(
                f"Could not list models on '{title}'.",
                str(listing.get("error") or listing),
                "Open the tab and confirm models exist in Model Management.",
            )
        lines = []
        for m in models[:12]:
            if isinstance(m, dict):
                lines.append(
                    f"- {m.get('model_type') or m.get('name')} id={m.get('model_id') or m.get('id')} "
                    f"trained={m.get('trained')} score={m.get('score')}"
                )
            else:
                lines.append(f"- {m}")
        return {
            "ok": True,
            "reply": _oar(
                f"Trained / registered models on '{title}': {len(models)}.",
                "\n".join(lines) or "(none)",
                "Ask which model is best if you want a metric comparison.",
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    if intent == OpsIntent.FORECAST:
        from app.agent.tools import forecast_risk_tool

        try:
            out = forecast_risk_tool(tab_id=tid, tab_title=title, host=host)
            tool_trace.append(_trace("forecast_risk", {"tab_id": tid}, bool(out.get("ok", True))))
        except Exception as exc:
            tool_trace.append(_trace("forecast_risk", {"tab_id": tid}, False, {"error": str(exc)}))
            return fail(f"Forecast for '{title}' failed.", str(exc), "Need a recent monitor cycle with score history.")
        return {
            "ok": True,
            "reply": _oar(
                f"Risk / TTF for '{out.get('title') or title}'.",
                f"risk={out.get('risk') or out.get('level')} ttf={out.get('ttf') or out.get('time_to_fail')} "
                f"detail={out}",
                "This is a short-horizon estimate from fusion history, not a guaranteed failure time.",
            ),
            "llm_used": False,
            "tool_trace": tool_trace,
            "outcome": "function_calling",
            "agent_mode": "tools",
            "intent": intent.value,
        }

    # TAB_DETAIL + EXPLAIN_LIVE
    from app.agent.tools import explain_anomaly_tool, get_tab_detail

    try:
        detail = get_tab_detail(tid, n=12, host=host)
        tool_trace.append(_trace("get_tab_detail", {"tab_id": tid}, True))
    except Exception as exc:
        tool_trace.append(_trace("get_tab_detail", {"tab_id": tid}, False, {"error": str(exc)}))
        return fail(f"Could not read '{title}'.", str(exc), "Confirm the tab is open.")

    snap = detail.get("snapshot") or resolved.snapshot or {}
    anomaly = detail.get("anomaly") or {}
    models = detail.get("models") or {}
    model_list = models.get("models") if isinstance(models, dict) else models
    n_models = len(model_list) if isinstance(model_list, list) else models.get("count") if isinstance(models, dict) else "—"
    explain_txt = ""
    if intent == OpsIntent.EXPLAIN_LIVE:
        try:
            expl = explain_anomaly_tool(tab_id=tid, tab_title=title, host=host)
            tool_trace.append(_trace("explain_anomaly", {"tab_id": tid}, bool(expl.get("ok", True))))
            explain_txt = str(expl.get("explanation") or expl.get("summary") or expl)[:900]
        except Exception as exc:
            tool_trace.append(_trace("explain_anomaly", {"tab_id": tid}, False, {"error": str(exc)}))
            explain_txt = f"(explain_anomaly failed: {exc})"

    health = anomaly.get("health_state") or snap.get("health_state")
    fusion = anomaly.get("fusion_score") if anomaly.get("fusion_score") is not None else snap.get("fusion_score")
    alerts = anomaly.get("alert_count") if anomaly.get("alert_count") is not None else snap.get("alert_count")
    drift = (detail.get("drift") or {}).get("drift")
    if drift is None:
        drift = snap.get("drift")
    analysis = (
        f"health={health}, fusion={fusion}, drift={drift}, alerts={alerts}, "
        f"models={n_models}, mission_mode={snap.get('mission_mode') or '—'}. "
        "Fusion is how worried the models are about the window — not the same as the Healthy/Warning label. "
        "A high anomaly-flag rate can still show Healthy if the adaptive threshold sits above fusion."
    )
    if explain_txt:
        analysis = f"{analysis}\n{explain_txt}"
    return {
        "ok": True,
        "reply": _oar(
            f"Live state for '{detail.get('title') or title}'.",
            analysis,
            "Use events / models questions for the raw lists. Retrain stays Approve-only.",
        ),
        "llm_used": False,
        "tool_trace": tool_trace,
        "outcome": "function_calling",
        "agent_mode": "tools",
        "intent": intent.value,
    }


def strip_node_traces(text: str) -> str:
    """Hide R/M/C/RA-LLM breadcrumbs from the operator-facing reply."""
    raw = text or ""
    raw = re.sub(
        r"(?im)^\s*\[(?:R-LLM|M-LLM|C-LLM|RA-LLM|chat)\][^\n]*\n?",
        "",
        raw,
    )
    return raw.strip()
