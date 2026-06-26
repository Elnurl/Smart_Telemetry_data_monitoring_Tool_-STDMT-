from pathlib import Path

import sqlite3

from app.models.registry import ModelRegistry, format_registry_error


def test_drift_triggers_registry_signal(tmp_path: Path):
    registry = ModelRegistry(db_path=str(tmp_path / "test.sqlite"))
    model_id = registry.register_model("test_model", "isolation_forest")

    signal_id = registry.mark_retrain_needed(
        model_id=model_id,
        drift_score=0.42,
        reason="concept_drift",
        drifted_features=["temp"],
        tab_id="tab-1",
    )
    assert signal_id is not None

    pending = registry.get_pending_retrain_signals(model_id)
    assert len(pending) == 1
    assert pending[0]["acknowledged"] == 0
    assert pending[0]["drift_score"] == 0.42
    assert pending[0]["drifted_features"] == ["temp"]

    registry.acknowledge_retrain_signal(signal_id, acknowledged_by="elnur")
    pending_after = registry.get_pending_retrain_signals(model_id)
    assert len(pending_after) == 0


def test_mark_retrain_needed_resolves_fusion_model_by_name(tmp_path: Path):
    registry = ModelRegistry(db_path=str(tmp_path / "fusion.sqlite"))

    signal_id = registry.mark_retrain_needed(
        drift_score=0.55,
        reason="concept_drift",
        drifted_features=["voltage"],
        tab_id="clock-tab",
        model_name="Clock data::fusion",
        metadata={"tab_title": "Clock data"},
    )
    assert signal_id > 0

    pending = registry.get_pending_retrain_signals()
    assert len(pending) == 1
    assert pending[0]["tab_id"] == "clock-tab"


def test_format_registry_error_disk_full():
    exc = sqlite3.OperationalError("database or disk is full")
    message = format_registry_error(exc)
    assert "disk" in message.lower()
    assert "traceback" not in message.lower()


def test_format_registry_error_generic():
    message = format_registry_error(RuntimeError("unexpected internal failure"))
    assert "log" in message.lower()


def test_retrain_signals_schema_migration(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE models (
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
        CREATE TABLE retrain_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            drift_score REAL,
            reason TEXT,
            drifted_features TEXT,
            status TEXT DEFAULT 'pending',
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

    registry = ModelRegistry(db_path=str(db_path))
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(retrain_signals)")
    columns = {row[1] for row in cursor.fetchall()}
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'retrain_signals_legacy%'"
    )
    backup_tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    assert "model_id" in columns
    assert "acknowledged" in columns
    assert backup_tables
    assert registry.get_pending_retrain_signals() == []
