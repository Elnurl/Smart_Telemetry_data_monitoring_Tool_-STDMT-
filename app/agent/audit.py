"""SQLite audit trail for agent decisions (Phase 0+)."""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional


class AgentAuditLog:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tab_id TEXT,
                        timestamp TEXT NOT NULL,
                        context_summary TEXT,
                        tool_called TEXT,
                        tool_params TEXT,
                        reasoning TEXT,
                        outcome TEXT
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def record(
        self,
        *,
        tab_id: Optional[str] = None,
        context_summary: str = "",
        tool_called: Optional[str] = None,
        tool_params: Optional[dict[str, Any]] = None,
        reasoning: str = "",
        outcome: str = "",
    ) -> int:
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        params_json = json.dumps(tool_params or {}, ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO agent_decisions
                        (tab_id, timestamp, context_summary, tool_called, tool_params, reasoning, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tab_id,
                        ts,
                        context_summary,
                        tool_called,
                        params_json,
                        reasoning,
                        outcome,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)
            finally:
                conn.close()

    def list_decisions(self, limit: int = 50, tab_id: Optional[str] = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._lock:
            conn = self._connect()
            try:
                if tab_id:
                    rows = conn.execute(
                        """
                        SELECT id, tab_id, timestamp, context_summary, tool_called,
                               tool_params, reasoning, outcome
                        FROM agent_decisions
                        WHERE tab_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (tab_id, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT id, tab_id, timestamp, context_summary, tool_called,
                               tool_params, reasoning, outcome
                        FROM agent_decisions
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                result: list[dict[str, Any]] = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["tool_params"] = json.loads(item.get("tool_params") or "{}")
                    except Exception:
                        pass
                    result.append(item)
                return result
            finally:
                conn.close()
