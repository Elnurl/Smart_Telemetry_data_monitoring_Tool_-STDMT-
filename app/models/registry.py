"""Canonical model registry — trained models, metrics, and retrain signals."""

from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "data"

RETRAIN_SIGNALS_REQUIRED_COLUMNS = frozenset({
    "id",
    "model_id",
    "tab_id",
    "drift_score",
    "drifted_features",
    "reason",
    "created_at",
    "acknowledged",
    "acknowledged_at",
    "acknowledged_by",
})


def format_registry_error(exc: BaseException) -> str:
    """Return a short operator-facing message for registry failures."""
    if isinstance(exc, sqlite3.OperationalError):
        message = str(exc).lower()
        if "disk" in message or "full" in message:
            return "Model registry could not be updated — disk may be full."
        if "locked" in message or "busy" in message:
            return "Model registry is busy; try again in a moment."
        if "readonly" in message or "read-only" in message:
            return "Model registry is read-only; check file permissions."
        return "Model registry database error; contact your administrator."
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, PermissionError):
        return "You do not have permission to update the model registry."
    return "Retrain signal could not be saved; see the application log for details."


def default_registry_db_path(data_dir: Optional[str] = None) -> str:
    base = data_dir or DEFAULT_DATA_DIR
    return os.path.join(base, "model_registry.sqlite")


class ModelRegistry:
    """Registry for trained models and drift-driven retrain signals."""

    def __init__(self, db_path: Optional[str] = None, data_dir: Optional[str] = None):
        if db_path is None:
            db_path = default_registry_db_path(data_dir)
        self.db_path = db_path
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _create_retrain_signals_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                tab_id TEXT,
                drift_score REAL,
                drifted_features TEXT,
                reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_at DATETIME,
                acknowledged_by TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
            """
        )

    def _migrate_retrain_signals_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='retrain_signals'"
        )
        if not cursor.fetchone():
            self._create_retrain_signals_table(cursor)
            return

        cursor.execute("PRAGMA table_info(retrain_signals)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if RETRAIN_SIGNALS_REQUIRED_COLUMNS.issubset(existing_columns):
            return

        logger.warning(
            "Legacy retrain_signals schema detected (%s); backing up and recreating table",
            sorted(existing_columns),
        )
        backup_name = "retrain_signals_legacy"
        suffix = 1
        while True:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (backup_name,),
            )
            if not cursor.fetchone():
                break
            suffix += 1
            backup_name = f"retrain_signals_legacy_{suffix}"

        cursor.execute(f"ALTER TABLE retrain_signals RENAME TO {backup_name}")
        self._create_retrain_signals_table(cursor)

    def _initialize_db(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                model_type TEXT,
                created_at DATETIME,
                last_used DATETIME,
                use_count INTEGER DEFAULT 0,
                filepath TEXT,
                metadata TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                metric_name TEXT,
                metric_value REAL,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
            """
        )

        self._migrate_retrain_signals_table(cursor)

        conn.commit()
        conn.close()

    def register_model(
        self,
        name: str,
        model_type: str,
        created_at: Optional[datetime.datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if created_at is None:
            created_at = datetime.datetime.now()

        metadata = metadata or {}
        metadata_json = json.dumps(metadata)

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO models (name, model_type, created_at, last_used, use_count, filepath, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                model_type,
                created_at.isoformat(),
                created_at.isoformat(),
                1,
                metadata.get("filepath", "") if metadata else "",
                metadata_json,
            ),
        )
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return int(model_id)

    def resolve_model_id(
        self,
        name: str,
        model_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM models WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return int(row[0])
        return self.register_model(name, model_type, metadata=metadata)

    def record_model_usage(self, model_id: int) -> bool:
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE models
                SET last_used = ?, use_count = use_count + 1
                WHERE id = ?
                """,
                (datetime.datetime.now().isoformat(), model_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as exc:
            logger.error("Error recording model usage: %s", exc)
            return False

    def get_recent_models(self, limit: int = 5) -> List[tuple]:
        try:
            if not os.path.exists(self.db_path):
                logger.warning("Model registry database not found: %s", self.db_path)
                return []

            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, name, model_type, last_used, use_count, filepath
                FROM models
                ORDER BY last_used DESC
                LIMIT ?
                """,
                (limit,),
            )
            models = cursor.fetchall()
            conn.close()
            return models
        except Exception as exc:
            logger.error("Error getting recent models: %s", exc)
            return []

    def add_model_metrics(self, model_id: int, metrics: Dict[str, Any]) -> bool:
        try:
            conn = self._connect()
            cursor = conn.cursor()

            for metric_name, metric_value in metrics.items():
                try:
                    metric_float = float(metric_value)
                    cursor.execute(
                        """
                        INSERT INTO model_metrics (model_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                        """,
                        (model_id, metric_name, metric_float),
                    )
                except (ValueError, TypeError):
                    logger.warning("Skipping non-numeric metric '%s': %s", metric_name, metric_value)

            conn.commit()
            conn.close()
            return True
        except Exception as exc:
            logger.error("Error adding model metrics: %s", exc)
            return False

    def mark_retrain_needed(
        self,
        drift_score: float,
        reason: str,
        drifted_features: Optional[List[str]] = None,
        tab_id: Optional[str] = None,
        model_id: Optional[int] = None,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a drift-driven retrain signal; returns the new signal id."""
        metadata = metadata or {}

        if model_id is None:
            if model_name:
                model_id = self.resolve_model_id(
                    model_name,
                    model_type=str(metadata.get("model_type", "fusion")),
                    metadata=metadata,
                )
            elif tab_id:
                title = str(metadata.get("tab_title") or tab_id)
                fusion_name = f"{title}::fusion"
                meta = dict(metadata)
                meta.setdefault("tab_id", tab_id)
                model_id = self.resolve_model_id(fusion_name, model_type="fusion", metadata=meta)
            else:
                raise ValueError("mark_retrain_needed requires model_id, model_name, or tab_id")

        features_json = json.dumps(list(drifted_features or []))
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO retrain_signals
            (model_id, tab_id, drift_score, drifted_features, reason, created_at, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (
                int(model_id),
                tab_id,
                float(drift_score),
                features_json,
                reason,
                datetime.datetime.now().isoformat(),
            ),
        )
        signal_id = int(cursor.lastrowid)
        conn.commit()
        conn.close()
        return signal_id

    def get_pending_retrain_signals(self, model_id: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._connect()
        cursor = conn.cursor()
        columns = (
            "rs.id",
            "rs.model_id",
            "rs.tab_id",
            "rs.drift_score",
            "rs.drifted_features",
            "rs.reason",
            "rs.created_at",
            "rs.acknowledged",
            "rs.acknowledged_at",
            "rs.acknowledged_by",
            "m.name",
        )
        select_cols = ", ".join(columns)
        result_keys = (
            "id",
            "model_id",
            "tab_id",
            "drift_score",
            "drifted_features",
            "reason",
            "created_at",
            "acknowledged",
            "acknowledged_at",
            "acknowledged_by",
            "model_name",
        )

        if model_id is not None:
            cursor.execute(
                f"""
                SELECT {select_cols}
                FROM retrain_signals rs
                LEFT JOIN models m ON m.id = rs.model_id
                WHERE rs.acknowledged = 0 AND rs.model_id = ?
                ORDER BY rs.created_at DESC
                """,
                (model_id,),
            )
        else:
            cursor.execute(
                f"""
                SELECT {select_cols}
                FROM retrain_signals rs
                LEFT JOIN models m ON m.id = rs.model_id
                WHERE rs.acknowledged = 0
                ORDER BY rs.created_at DESC
                """
            )

        rows = cursor.fetchall()
        conn.close()

        results: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(zip(result_keys, row))
            try:
                item["drifted_features"] = json.loads(item.get("drifted_features") or "[]")
            except json.JSONDecodeError:
                item["drifted_features"] = []
            results.append(item)
        return results

    def acknowledge_retrain_signal(self, signal_id: int, acknowledged_by: Optional[str] = None) -> bool:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE retrain_signals
            SET acknowledged = 1,
                acknowledged_at = ?,
                acknowledged_by = ?
            WHERE id = ?
            """,
            (datetime.datetime.now().isoformat(), acknowledged_by, signal_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated
