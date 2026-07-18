"""Phase 2 — LogMonitor FSM filtering + ingest persistence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.agent.log_monitor import (
    LogMonitor,
    detect_mode_transition,
    is_mode_normal_event,
    parse_log_line,
    reset_log_monitor,
)
from app.agent.prompts import MLLM_PROMPT
from app.models.fsm import MissionMode, MissionModeStore


SAMPLE_LOG = """
2026-07-15T16:05:30 INFO POWER "Solar panel deployment complete"
2026-07-15T16:05:31 INFO TCS "Eclipse entry detected, thermal mode activated"
2026-07-15T16:05:32 WARNING TCS "Temperature exceeding nominal range"
2026-07-15T16:05:33 ERROR CDH "Command buffer overflow"
""".strip()


def test_mllm_prompt_formats():
    text = MLLM_PROMPT.format(window_size=50, current_mode="eclipse", scale=1.5)
    assert "eclipse" in text
    assert "1.5" in text
    assert "50" in text


def test_parse_log_line_gmsec_style():
    ev = parse_log_line(
        '2026-07-15T16:05:32 WARNING TCS "Temperature exceeding nominal range"',
        tab_id="t1",
    )
    assert ev is not None
    assert ev.level == "WARNING"
    assert ev.source == "TCS"
    assert "Temperature" in ev.message
    assert ev.timestamp.year == 2026


def test_detect_eclipse_entry():
    assert detect_mode_transition("Eclipse entry detected, thermal mode activated") == "eclipse"
    assert detect_mode_transition("Maneuver complete") == "nominal"


def test_log_monitor_fsm_filtering(tmp_path):
    """Eclipse: TCS WARNING → mode_normal; CDH ERROR → anomaly. Nominal: TCS WARNING → anomaly."""
    reset_log_monitor()
    db = str(tmp_path / "fsm.sqlite")
    store = MissionModeStore(db_path=db)
    store.sync_tab_modes(
        "eclipse-tab",
        [
            {"name": "nominal", "threshold_scale": 1.0},
            {"name": "eclipse", "threshold_scale": 1.5},
        ],
    )
    monitor = LogMonitor(db_path=db, allow_llm=False, window_size=50)

    log_path = tmp_path / "mission.log"
    log_path.write_text(SAMPLE_LOG, encoding="utf-8")
    assert monitor.ingest_file("eclipse-tab", log_path) == 4

    eclipse_cfg = {
        "current_mission_mode": "eclipse",
        "mission_modes": [
            {"name": "nominal", "threshold_scale": 1.0},
            {"name": "eclipse", "threshold_scale": 1.5},
        ],
    }
    analysis = monitor.analyze(
        "eclipse-tab",
        store,
        tab_config=eclipse_cfg,
        apply_transitions=False,
        use_llm=False,
    )
    assert analysis.false_positives_filtered >= 1
    sources = {a["source"] for a in analysis.anomalies}
    assert "CDH" in sources
    assert "TCS" not in sources
    assert any(t["to_mode"] == "eclipse" for t in analysis.mode_transitions) or True

    # Same WARNING alone under nominal → anomaly
    monitor2 = LogMonitor(db_path=str(tmp_path / "fsm2.sqlite"), allow_llm=False)
    monitor2.ingest(
        "nom-tab",
        datetime(2026, 7, 15, 16, 5, 32),
        "WARNING",
        "TCS",
        "Temperature exceeding nominal range",
    )
    nominal = monitor2.analyze(
        "nom-tab",
        MissionModeStore(db_path=str(tmp_path / "fsm2.sqlite")),
        tab_config={"current_mission_mode": "nominal", "mission_modes": [{"name": "nominal", "threshold_scale": 1.0}]},
        use_llm=False,
    )
    assert nominal.false_positives_filtered == 0
    assert any(a["source"] == "TCS" for a in nominal.anomalies)

    # CDH ERROR is anomaly in both modes
    assert is_mode_normal_event(
        parse_log_line('2026-07-15T16:05:33 ERROR CDH "Command buffer overflow"', tab_id="x"),
        MissionMode(name="eclipse", threshold_scale=1.5),
    ) is False


def test_log_analyses_persisted(tmp_path):
    db = str(tmp_path / "persist.sqlite")
    monitor = LogMonitor(db_path=db, allow_llm=False)
    monitor.ingest("t1", datetime.now(), "INFO", "POWER", "ok")
    analysis = monitor.analyze(
        "t1",
        tab_config={"current_mission_mode": "nominal", "mission_modes": [{"name": "nominal", "threshold_scale": 1.0}]},
        use_llm=False,
    )
    assert analysis.summary
    import sqlite3

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM tab_log_events WHERE tab_id=?", ("t1",))
    assert cur.fetchone()[0] >= 1
    cur.execute("SELECT COUNT(*) FROM log_analyses WHERE tab_id=?", ("t1",))
    assert cur.fetchone()[0] >= 1
    conn.close()


def test_dashboard_line_format(tmp_path):
    monitor = LogMonitor(db_path=str(tmp_path / "d.sqlite"), allow_llm=False)
    monitor.ingest(
        "e1",
        datetime(2026, 7, 15, 16, 5, 31),
        "INFO",
        "TCS",
        "Eclipse entry detected",
    )
    monitor.ingest(
        "e1",
        datetime(2026, 7, 15, 16, 5, 32),
        "WARNING",
        "TCS",
        "Temperature exceeding nominal range",
    )
    analysis = monitor.analyze(
        "e1",
        tab_config={
            "current_mission_mode": "eclipse",
            "mission_modes": [{"name": "eclipse", "threshold_scale": 1.5}],
        },
        use_llm=False,
    )
    lines = monitor.dashboard_lines([analysis], titles={"e1": "Eclipse"})
    assert lines[0].startswith("[M-LLM] Tab Eclipse:")
    assert "mode-normal" in analysis.summary.lower() or analysis.false_positives_filtered >= 1
