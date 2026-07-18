"""M-LLM log monitor — sliding-window log analysis in FSM context."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.models.fsm import MissionMode, MissionModeStore, default_fsm_db_path, resolve_current_mode

logger = logging.getLogger("STDMS.Agent.LogMonitor")

_LEVELS = ("DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL")

# GMSEC-ish / common formats:
# 2026-07-15T16:05:31 INFO TCS "Eclipse entry detected"
# [2026-07-15 16:05:31] [WARNING] [TCS] Temperature exceeding nominal range
_LINE_RE = re.compile(
    r"^\s*(?:\[?(?P<ts1>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]?\s+)?"
    r"(?:\[?(?P<level>" + "|".join(_LEVELS) + r")\]?\s+)?"
    r"(?:\[?(?P<source>[A-Za-z][A-Za-z0-9_\-]{0,31})\]?\s+)?"
    r"(?P<msg>.+?)\s*$",
    re.IGNORECASE,
)

_MODE_TRIGGERS: list[tuple[str, re.Pattern[str]]] = [
    ("eclipse", re.compile(r"\beclipse\s+(entry|enter|entered|start|begun|begin)\b", re.I)),
    ("nominal", re.compile(r"\beclipse\s+(exit|exited|end|complete|leave)\b", re.I)),
    ("maneuver", re.compile(r"\bmaneuver\s+(start|begun|begin|in\s+progress)\b", re.I)),
    ("nominal", re.compile(r"\bmaneuver\s+(complete|completed|end|finished)\b", re.I)),
    ("safe_mode", re.compile(r"\bsafe[\s_\-]?mode\s+(enter|entry|entered|activated|engage)", re.I)),
    ("nominal", re.compile(r"\bsafe[\s_\-]?mode\s+(exit|exited|clear|deactivated)", re.I)),
    # Do NOT match phrases like "nominal range" (telemetry text, not a mode change)
    (
        "nominal",
        re.compile(
            r"\b(sunlight\s+mode|return\s+to\s+nominal|nominal\s+mode|nominal\s+ops|"
            r"nominal\s+operations)\b",
            re.I,
        ),
    ),
]

# Subsystems whose warnings are often expected in specific modes
_MODE_NORMAL_SOURCES = {
    "eclipse": {"TCS", "THERMAL", "POWER", "EPS"},
    "maneuver": {"ADCS", "ACS", "AOCS", "GNC"},
    "safe_mode": set(),  # safe_mode: do not filter — more sensitive
}


@dataclass
class LogEvent:
    tab_id: str
    timestamp: datetime
    level: str
    source: str
    message: str
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tab_id": self.tab_id,
            "timestamp": self.timestamp.isoformat(sep=" "),
            "level": self.level,
            "source": self.source,
            "message": self.message,
        }

    def format_line(self) -> str:
        return (
            f"{self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')} "
            f"{self.level} {self.source} {self.message}"
        )


@dataclass
class LogAnalysis:
    tab_id: str
    window_size: int
    mode_transitions: list[dict[str, Any]] = field(default_factory=list)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    false_positives_filtered: int = 0
    summary: str = ""
    current_mode: str = "nominal"
    threshold_scale: float = 1.0
    llm_used: bool = False
    analyzed_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _parse_timestamp(raw: str | None) -> datetime:
    if not raw:
        return datetime.now()
    text = raw.strip().replace("T", " ")
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text[:26], fmt)
        except ValueError:
            continue
    return datetime.now()


def parse_log_line(line: str, *, tab_id: str = "", default_ts: datetime | None = None) -> Optional[LogEvent]:
    """Parse one log line into a LogEvent; return None for blank/comment lines."""
    text = (line or "").strip()
    if not text or text.startswith("#"):
        return None
    # Strip surrounding quotes on message portion later
    match = _LINE_RE.match(text)
    if not match:
        return LogEvent(
            tab_id=tab_id,
            timestamp=default_ts or datetime.now(),
            level="INFO",
            source="UNKNOWN",
            message=text,
        )
    level = (match.group("level") or "INFO").upper()
    if level == "WARN":
        level = "WARNING"
    if level == "FATAL":
        level = "CRITICAL"
    source = (match.group("source") or "UNKNOWN").upper()
    # Avoid treating a second timestamp token as source
    if re.match(r"^\d", source):
        source = "UNKNOWN"
    msg = (match.group("msg") or "").strip().strip('"').strip("'")
    # If level missing but first token looked like source-only, keep message
    if not match.group("level") and match.group("source") and not match.group("ts1"):
        # line like: TCS Temperature rising — already handled
        pass
    return LogEvent(
        tab_id=tab_id,
        timestamp=_parse_timestamp(match.group("ts1")) if match.group("ts1") else (default_ts or datetime.now()),
        level=level,
        source=source,
        message=msg or text,
    )


def parse_log_file(path: str | Path, *, tab_id: str = "") -> list[LogEvent]:
    """Parse a .log / .txt file into LogEvent list (skips unreadable lines)."""
    p = Path(path)
    events: list[LogEvent] = []
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Failed to read log file %s: %s", path, exc)
        return events
    for line in raw.splitlines():
        ev = parse_log_line(line, tab_id=tab_id)
        if ev is not None:
            events.append(ev)
    return events


def detect_mode_transition(message: str) -> Optional[str]:
    """Return target mode name if message implies an FSM transition."""
    for mode_name, pattern in _MODE_TRIGGERS:
        if pattern.search(message or ""):
            return mode_name
    return None


def is_mode_normal_event(event: LogEvent, mode: MissionMode | str) -> bool:
    """Heuristic: WARNING from thermal/power during eclipse is mode-normal."""
    mode_name = mode.name if isinstance(mode, MissionMode) else str(mode or "nominal").lower()
    level = (event.level or "").upper()
    if level in ("ERROR", "CRITICAL"):
        return False
    if level not in ("WARNING", "WARN"):
        return False
    source = (event.source or "").upper()
    allowed = _MODE_NORMAL_SOURCES.get(mode_name) or set()
    if source not in allowed:
        return False
    msg = (event.message or "").lower()
    # Temperature / thermal / power swing language
    thermal_hints = ("temp", "thermal", "heat", "cold", "nominal range", "exceed", "battery")
    if mode_name == "eclipse" and any(h in msg for h in thermal_hints):
        return True
    if mode_name == "maneuver" and any(h in msg for h in ("vibrat", "rate", "torque", "slew", "jitter")):
        return True
    # Source alone in allowed set + WARNING during matching mode
    return source in allowed and mode_name in ("eclipse", "maneuver")


def format_mllm_prompt(window_size: int, current_mode: str, scale: float) -> str:
    from app.agent.prompts import MLLM_PROMPT

    return MLLM_PROMPT.format(
        window_size=window_size,
        current_mode=current_mode,
        scale=scale,
    )


class LogMonitor:
    """Sliding-window log ingest + M-LLM (or rule-based) FSM-aware analysis."""

    def __init__(
        self,
        *,
        window_size: int = 50,
        db_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        allow_llm: bool = True,
    ):
        self.window_size = max(5, int(window_size))
        self.db_path = db_path or default_fsm_db_path(data_dir)
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.allow_llm = bool(allow_llm)
        self._windows: dict[str, deque[LogEvent]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._last_analysis: dict[str, LogAnalysis] = {}
        self._ensure_tables()

    def _connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tab_log_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tab_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT,
                source TEXT,
                message TEXT,
                ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tab_log_events_tab_ts
            ON tab_log_events(tab_id, timestamp DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS log_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tab_id TEXT,
                analyzed_at DATETIME,
                window_size INTEGER,
                mode_transitions_json TEXT,
                anomalies_json TEXT,
                false_positives_filtered INTEGER,
                summary TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def ingest(
        self,
        tab_id: str,
        timestamp: datetime | str | None,
        level: str,
        source: str,
        message: str,
    ) -> LogEvent:
        if isinstance(timestamp, str):
            ts = _parse_timestamp(timestamp)
        elif isinstance(timestamp, datetime):
            ts = timestamp
        else:
            ts = datetime.now()
        event = LogEvent(
            tab_id=str(tab_id),
            timestamp=ts,
            level=(level or "INFO").upper(),
            source=(source or "UNKNOWN").upper(),
            message=str(message or ""),
        )
        if event.level == "WARN":
            event.level = "WARNING"
        event_id = self._persist_event(event)
        event.id = event_id
        self._windows[str(tab_id)].append(event)
        return event

    def ingest_events(self, tab_id: str, events: list[LogEvent]) -> int:
        count = 0
        for ev in events:
            ev.tab_id = str(tab_id)
            self.ingest(tab_id, ev.timestamp, ev.level, ev.source, ev.message)
            count += 1
        return count

    def ingest_file(self, tab_id: str, path: str | Path) -> int:
        events = parse_log_file(path, tab_id=tab_id)
        return self.ingest_events(tab_id, events)

    def _persist_event(self, event: LogEvent) -> int:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tab_log_events (tab_id, timestamp, level, source, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                event.tab_id,
                event.timestamp.isoformat(sep=" "),
                event.level,
                event.source,
                event.message,
            ),
        )
        event_id = int(cur.lastrowid)
        conn.commit()
        conn.close()
        return event_id

    def get_window(self, tab_id: str) -> list[LogEvent]:
        """Return last N events (memory first, else DB)."""
        key = str(tab_id)
        mem = list(self._windows.get(key) or [])
        if mem:
            return mem[-self.window_size :]
        return self._load_window_from_db(key)

    def _load_window_from_db(self, tab_id: str) -> list[LogEvent]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, level, source, message
            FROM tab_log_events
            WHERE tab_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (tab_id, self.window_size),
        )
        rows = cur.fetchall()
        conn.close()
        events: list[LogEvent] = []
        for row in reversed(rows):
            events.append(
                LogEvent(
                    id=int(row[0]),
                    tab_id=tab_id,
                    timestamp=_parse_timestamp(row[1]),
                    level=str(row[2] or "INFO"),
                    source=str(row[3] or "UNKNOWN"),
                    message=str(row[4] or ""),
                )
            )
        self._windows[tab_id] = deque(events, maxlen=self.window_size)
        return events

    def analyze(
        self,
        tab_id: str,
        fsm_store: MissionModeStore | None = None,
        *,
        tab_config: dict[str, Any] | None = None,
        apply_transitions: bool = True,
        use_llm: Optional[bool] = None,
    ) -> LogAnalysis:
        """
        M-LLM / rule analysis of the sliding window in FSM context.

        Always runs deterministic FSM filtering so tests and offline mode work;
        optionally enriches ``summary`` via Ollama when available.
        """
        tab_id = str(tab_id)
        window = self.get_window(tab_id)
        mode = self._resolve_mode(tab_id, fsm_store, tab_config)

        transitions: list[dict[str, Any]] = []
        anomalies: list[dict[str, Any]] = []
        filtered = 0
        effective_mode = mode

        for ev in window:
            target = detect_mode_transition(ev.message)
            if target and target != effective_mode.name:
                transitions.append(
                    {
                        "from_mode": effective_mode.name,
                        "to_mode": target,
                        "timestamp": ev.timestamp.isoformat(sep=" "),
                        "source": ev.source,
                        "message": ev.message,
                        "trigger": "log_event",
                    }
                )
                # Update effective mode for subsequent lines in this window
                scale = effective_mode.threshold_scale
                if fsm_store is not None:
                    for m in fsm_store.get_tab_modes(tab_id):
                        if m.name == target:
                            scale = m.threshold_scale
                            break
                elif tab_config:
                    for m in tab_config.get("mission_modes") or []:
                        if str(m.get("name", "")).lower() == target:
                            try:
                                scale = float(m.get("threshold_scale", 1.0))
                            except (TypeError, ValueError):
                                scale = 1.0
                            break
                effective_mode = MissionMode(name=target, threshold_scale=scale)
                if apply_transitions and fsm_store is not None:
                    try:
                        fsm_store.set_mode(
                            tab_id,
                            target,
                            threshold_scale=scale,
                            trigger_source="log_event",
                        )
                    except Exception as exc:
                        logger.warning("FSM set_mode from log failed: %s", exc)

            level = (ev.level or "").upper()
            if level not in ("WARNING", "ERROR", "CRITICAL"):
                continue
            if is_mode_normal_event(ev, effective_mode):
                filtered += 1
                continue
            anomalies.append(
                {
                    "timestamp": ev.timestamp.isoformat(sep=" "),
                    "level": ev.level,
                    "source": ev.source,
                    "message": ev.message,
                    "reason": self._anomaly_reason(ev, effective_mode),
                }
            )

        summary = self._rule_summary(
            tab_id=tab_id,
            mode=effective_mode,
            transitions=transitions,
            anomalies=anomalies,
            filtered=filtered,
            window=window,
        )
        llm_used = False
        want_llm = self.allow_llm if use_llm is None else bool(use_llm)
        if want_llm and window:
            llm_text = self._call_mllm(window, effective_mode)
            if llm_text:
                summary = llm_text.strip()
                llm_used = True

        analysis = LogAnalysis(
            tab_id=tab_id,
            window_size=len(window),
            mode_transitions=transitions,
            anomalies=anomalies,
            false_positives_filtered=filtered,
            summary=summary,
            current_mode=effective_mode.name,
            threshold_scale=effective_mode.threshold_scale,
            llm_used=llm_used,
            analyzed_at=datetime.now().isoformat(sep=" "),
        )
        self._persist_analysis(analysis)
        self._last_analysis[tab_id] = analysis
        return analysis

    def get_last_analysis(self, tab_id: str) -> Optional[LogAnalysis]:
        return self._last_analysis.get(str(tab_id))

    def analyze_all_tabs(
        self,
        tab_ids: list[str],
        fsm_store: MissionModeStore | None = None,
        *,
        tab_configs: dict[str, dict[str, Any]] | None = None,
        use_llm: Optional[bool] = None,
    ) -> list[LogAnalysis]:
        results: list[LogAnalysis] = []
        configs = tab_configs or {}
        for tid in tab_ids:
            window = self.get_window(tid)
            if not window:
                continue
            results.append(
                self.analyze(
                    tid,
                    fsm_store,
                    tab_config=configs.get(tid),
                    use_llm=use_llm,
                )
            )
        return results

    def dashboard_lines(self, analyses: list[LogAnalysis], *, titles: dict[str, str] | None = None) -> list[str]:
        titles = titles or {}
        lines: list[str] = []
        for a in analyses:
            title = titles.get(a.tab_id) or a.tab_id
            lines.append(f"[M-LLM] Tab {title}: {a.summary}")
        return lines

    def _resolve_mode(
        self,
        tab_id: str,
        fsm_store: MissionModeStore | None,
        tab_config: dict[str, Any] | None,
    ) -> MissionMode:
        if tab_config:
            return resolve_current_mode(tab_config)
        if fsm_store is not None:
            current = fsm_store.get_current_mode(tab_id)
            if current is not None:
                return current
        return MissionMode(name="nominal", threshold_scale=1.0)

    @staticmethod
    def _anomaly_reason(event: LogEvent, mode: MissionMode) -> str:
        level = (event.level or "").upper()
        if level in ("ERROR", "CRITICAL"):
            return f"{event.source} {level} is treated as a true anomaly in any mission mode"
        return (
            f"{event.source} {level} is not explained by mode '{mode.name}' "
            f"(threshold_scale={mode.threshold_scale:g})"
        )

    @staticmethod
    def _rule_summary(
        *,
        tab_id: str,
        mode: MissionMode,
        transitions: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        filtered: int,
        window: list[LogEvent],
    ) -> str:
        parts: list[str] = []
        if transitions:
            last = transitions[-1]
            parts.append(
                f"{last['to_mode'].title()} transition detected "
                f"({last.get('timestamp', '')})."
            )
        else:
            parts.append(f"Mode={mode.name} (×{mode.threshold_scale:g}).")
        if filtered:
            parts.append(f"{filtered} WARNING filtered (mode-normal).")
        if anomalies:
            a0 = anomalies[0]
            extra = f" (+{len(anomalies) - 1} more)" if len(anomalies) > 1 else ""
            parts.append(
                f"{len(anomalies)} true anomal{'y' if len(anomalies) == 1 else 'ies'}: "
                f"{a0.get('source')} {a0.get('level')}{extra}."
            )
        elif window:
            parts.append("No true anomalies in window.")
        else:
            parts.append("No log events in window.")
        return " ".join(parts)

    def _call_mllm(self, window: list[LogEvent], mode: MissionMode) -> Optional[str]:
        try:
            from app.agent.ollama_client import (
                DEFAULT_OLLAMA_URL,
                chat_ollama,
                ollama_reachable,
                resolve_chat_model,
            )
            from app.agent.policy import is_loopback_url
        except Exception:
            return None

        base = (self.ollama_url or DEFAULT_OLLAMA_URL).rstrip("/")
        if not is_loopback_url(base) or not ollama_reachable(base):
            return None
        model = resolve_chat_model(self.ollama_model, base_url=base)
        system = format_mllm_prompt(len(window), mode.name, mode.threshold_scale)
        body = "\n".join(ev.format_line() for ev in window)
        user = (
            f"Current mission mode: {mode.name} (threshold_scale={mode.threshold_scale:g})\n"
            f"Log window ({len(window)} lines):\n{body}\n\n"
            "Reply with a short operator summary in English. "
            "Mention mode transitions, true anomalies, and mode-normal filters."
        )
        try:
            msg = chat_ollama(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=None,
                base_url=base,
                model=model,
                num_predict=280,
                temperature=0.1,
            )
        except Exception as exc:
            logger.info("M-LLM chat failed: %s", exc)
            return None
        if not msg:
            return None
        content = (msg.get("content") or "").strip()
        return content or None

    def _persist_analysis(self, analysis: LogAnalysis) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO log_analyses (
                tab_id, analyzed_at, window_size, mode_transitions_json,
                anomalies_json, false_positives_filtered, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.tab_id,
                analysis.analyzed_at,
                analysis.window_size,
                json.dumps(analysis.mode_transitions),
                json.dumps(analysis.anomalies),
                int(analysis.false_positives_filtered),
                analysis.summary,
            ),
        )
        conn.commit()
        conn.close()


# Shared lazy singleton for agent cycle + UI ingest
_DEFAULT_MONITOR: Optional[LogMonitor] = None


def get_log_monitor(**kwargs: Any) -> LogMonitor:
    """Return process-wide LogMonitor; pass ``_fresh=True`` to replace it (tests)."""
    global _DEFAULT_MONITOR
    fresh = bool(kwargs.pop("_fresh", False))
    if fresh or _DEFAULT_MONITOR is None:
        _DEFAULT_MONITOR = LogMonitor(**kwargs)
    elif kwargs:
        # Refresh runtime LLM settings without dropping ingested windows
        if "ollama_url" in kwargs and kwargs["ollama_url"]:
            _DEFAULT_MONITOR.ollama_url = kwargs["ollama_url"]
        if "ollama_model" in kwargs and kwargs["ollama_model"]:
            _DEFAULT_MONITOR.ollama_model = kwargs["ollama_model"]
        if "allow_llm" in kwargs:
            _DEFAULT_MONITOR.allow_llm = bool(kwargs["allow_llm"])
    return _DEFAULT_MONITOR


def reset_log_monitor() -> None:
    global _DEFAULT_MONITOR
    _DEFAULT_MONITOR = None
