"""Mission-mode FSM — per-tab operating regimes and threshold scaling."""

from __future__ import annotations

import datetime
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger("STDMS.FSM")

DEFAULT_DATA_DIR = "data"


@dataclass
class MissionMode:
    """Satellite / subsystem operating regime for a monitoring tab."""

    name: str
    threshold_scale: float = 1.0
    expected_deviation_multiplier: float = 1.0
    expected_patterns: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MissionMode":
        data = data or {}
        name = str(data.get("name") or "nominal").strip() or "nominal"
        try:
            scale = float(data.get("threshold_scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        if scale <= 0:
            scale = 1.0
        try:
            expected = float(data.get("expected_deviation_multiplier", scale))
        except (TypeError, ValueError):
            expected = scale
        patterns = data.get("expected_patterns") or {}
        if not isinstance(patterns, dict):
            patterns = {}
        return cls(
            name=name.lower(),
            threshold_scale=scale,
            expected_deviation_multiplier=expected,
            expected_patterns=patterns,
            description=str(data.get("description") or ""),
        )


@dataclass
class FSMTransition:
    """Record of a mode change for a tab."""

    from_mode: str
    to_mode: str
    trigger: str
    timestamp: datetime.datetime
    tab_id: str = ""
    threshold_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_mode": self.from_mode,
            "to_mode": self.to_mode,
            "trigger": self.trigger,
            "timestamp": self.timestamp.isoformat(),
            "tab_id": self.tab_id,
            "threshold_scale": self.threshold_scale,
        }


DEFAULT_MISSION_MODES: list[dict[str, Any]] = [
    {
        "name": "nominal",
        "threshold_scale": 1.0,
        "expected_deviation_multiplier": 1.0,
        "description": "Normal operations",
    },
    {
        "name": "eclipse",
        "threshold_scale": 1.5,
        "expected_deviation_multiplier": 1.5,
        "description": "Eclipse — thermal swings expected",
    },
    {
        "name": "maneuver",
        "threshold_scale": 2.0,
        "expected_deviation_multiplier": 2.0,
        "description": "Maneuver — vibration / transient spikes expected",
    },
    {
        "name": "safe_mode",
        "threshold_scale": 0.5,
        "expected_deviation_multiplier": 0.5,
        "description": "Safe mode — tighter sensitivity",
    },
]


def default_fsm_db_path(data_dir: Optional[str] = None) -> str:
    base = data_dir or DEFAULT_DATA_DIR
    return os.path.join(base, "mission_fsm.sqlite")


def default_mission_modes() -> list[MissionMode]:
    return [MissionMode.from_dict(item) for item in DEFAULT_MISSION_MODES]


def normalize_mission_modes(raw: Any) -> list[MissionMode]:
    """Parse tab-config mission_modes; fall back to defaults when empty/invalid."""
    if not isinstance(raw, list) or not raw:
        return default_mission_modes()
    modes: list[MissionMode] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        mode = MissionMode.from_dict(item)
        if mode.name in seen:
            continue
        seen.add(mode.name)
        modes.append(mode)
    return modes or default_mission_modes()


def mission_modes_to_config(modes: list[MissionMode]) -> list[dict[str, Any]]:
    return [m.to_dict() for m in modes]


def resolve_current_mode(config: dict[str, Any] | None) -> MissionMode:
    """Return the active MissionMode for a tab config dict."""
    config = config or {}
    modes = normalize_mission_modes(config.get("mission_modes"))
    by_name = {m.name: m for m in modes}
    current = str(config.get("current_mission_mode") or "nominal").strip().lower()
    if current in by_name:
        return by_name[current]
    return by_name.get("nominal") or modes[0]


def ensure_fsm_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Mutate *config* so mission mode fields are present and normalized."""
    modes = normalize_mission_modes(config.get("mission_modes"))
    config["mission_modes"] = mission_modes_to_config(modes)
    current = str(config.get("current_mission_mode") or "nominal").strip().lower()
    names = {m.name for m in modes}
    if current not in names:
        current = "nominal" if "nominal" in names else modes[0].name
    config["current_mission_mode"] = current
    return config


def scale_bound(base: float, operator: str, scale: float) -> float:
    """Widen (scale>1) or tighten (scale<1) an OBS numeric bound."""
    if scale <= 0:
        scale = 1.0
    if operator in (">", ">="):
        return base * scale
    if operator in ("<", "<="):
        return base / scale
    return base


def apply_threshold_scale(value: float, scale: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    """Scale a fusion / adaptive decision threshold and clamp to [lo, hi]."""
    if scale <= 0:
        scale = 1.0
    return float(max(lo, min(hi, value * scale)))


class MissionModeStore:
    """SQLite persistence for per-tab mode definitions and transition history."""

    def __init__(self, db_path: Optional[str] = None, data_dir: Optional[str] = None):
        if db_path is None:
            db_path = default_fsm_db_path(data_dir)
        self.db_path = db_path
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_db(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tab_mission_modes (
                tab_id TEXT NOT NULL,
                mode_name TEXT NOT NULL,
                threshold_scale REAL NOT NULL DEFAULT 1.0,
                expected_deviation_multiplier REAL NOT NULL DEFAULT 1.0,
                description TEXT,
                expected_patterns TEXT,
                PRIMARY KEY (tab_id, mode_name)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mission_mode_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tab_id TEXT NOT NULL,
                mode_name TEXT NOT NULL,
                threshold_scale REAL NOT NULL DEFAULT 1.0,
                entered_at DATETIME NOT NULL,
                exited_at DATETIME,
                trigger_source TEXT NOT NULL DEFAULT 'manual'
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mission_mode_history_tab
            ON mission_mode_history(tab_id, entered_at DESC)
            """
        )
        conn.commit()
        conn.close()

    def sync_tab_modes(self, tab_id: str, modes: list[MissionMode] | list[dict[str, Any]]) -> None:
        """Replace stored mode definitions for a tab."""
        parsed = [
            m if isinstance(m, MissionMode) else MissionMode.from_dict(m) for m in modes
        ]
        import json

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tab_mission_modes WHERE tab_id = ?", (tab_id,))
        for mode in parsed:
            cursor.execute(
                """
                INSERT INTO tab_mission_modes (
                    tab_id, mode_name, threshold_scale,
                    expected_deviation_multiplier, description, expected_patterns
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tab_id,
                    mode.name,
                    mode.threshold_scale,
                    mode.expected_deviation_multiplier,
                    mode.description,
                    json.dumps(mode.expected_patterns),
                ),
            )
        conn.commit()
        conn.close()

    def get_tab_modes(self, tab_id: str) -> list[MissionMode]:
        import json

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT mode_name, threshold_scale, expected_deviation_multiplier,
                   description, expected_patterns
            FROM tab_mission_modes WHERE tab_id = ?
            ORDER BY mode_name
            """,
            (tab_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            return default_mission_modes()
        modes: list[MissionMode] = []
        for name, scale, expected, desc, patterns_json in rows:
            try:
                patterns = json.loads(patterns_json) if patterns_json else {}
            except Exception:
                patterns = {}
            modes.append(
                MissionMode(
                    name=str(name),
                    threshold_scale=float(scale),
                    expected_deviation_multiplier=float(expected),
                    expected_patterns=patterns if isinstance(patterns, dict) else {},
                    description=str(desc or ""),
                )
            )
        return modes

    def get_current_mode(self, tab_id: str) -> Optional[MissionMode]:
        """Open history entry (exited_at IS NULL) → mode definition."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT mode_name, threshold_scale FROM mission_mode_history
            WHERE tab_id = ? AND exited_at IS NULL
            ORDER BY entered_at DESC LIMIT 1
            """,
            (tab_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        mode_name, scale = row[0], float(row[1])
        for mode in self.get_tab_modes(tab_id):
            if mode.name == mode_name:
                return mode
        return MissionMode(name=str(mode_name), threshold_scale=scale)

    def set_mode(
        self,
        tab_id: str,
        mode_name: str,
        *,
        threshold_scale: Optional[float] = None,
        trigger_source: str = "manual",
        modes: list[MissionMode] | None = None,
    ) -> FSMTransition:
        """Enter a new mode; close previous open history row."""
        mode_name = str(mode_name).strip().lower()
        catalog = modes or self.get_tab_modes(tab_id)
        by_name = {m.name: m for m in catalog}
        target = by_name.get(mode_name) or MissionMode(name=mode_name, threshold_scale=1.0)
        if threshold_scale is not None and threshold_scale > 0:
            target = MissionMode(
                name=target.name,
                threshold_scale=float(threshold_scale),
                expected_deviation_multiplier=target.expected_deviation_multiplier,
                expected_patterns=target.expected_patterns,
                description=target.description,
            )

        now = datetime.datetime.now()
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, mode_name FROM mission_mode_history
            WHERE tab_id = ? AND exited_at IS NULL
            ORDER BY entered_at DESC LIMIT 1
            """,
            (tab_id,),
        )
        open_row = cursor.fetchone()
        from_mode = "nominal"
        if open_row:
            from_mode = str(open_row[1])
            cursor.execute(
                "UPDATE mission_mode_history SET exited_at = ? WHERE id = ?",
                (now.isoformat(sep=" "), open_row[0]),
            )
        cursor.execute(
            """
            INSERT INTO mission_mode_history (
                tab_id, mode_name, threshold_scale, entered_at, exited_at, trigger_source
            ) VALUES (?, ?, ?, ?, NULL, ?)
            """,
            (
                tab_id,
                target.name,
                target.threshold_scale,
                now.isoformat(sep=" "),
                trigger_source,
            ),
        )
        conn.commit()
        conn.close()
        return FSMTransition(
            from_mode=from_mode,
            to_mode=target.name,
            trigger=trigger_source,
            timestamp=now,
            tab_id=tab_id,
            threshold_scale=target.threshold_scale,
        )

    def history(self, tab_id: str, limit: int = 50) -> list[dict[str, Any]]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT mode_name, threshold_scale, entered_at, exited_at, trigger_source
            FROM mission_mode_history
            WHERE tab_id = ?
            ORDER BY entered_at DESC
            LIMIT ?
            """,
            (tab_id, int(limit)),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "mode_name": r[0],
                "threshold_scale": r[1],
                "entered_at": r[2],
                "exited_at": r[3],
                "trigger_source": r[4],
            }
            for r in rows
        ]
