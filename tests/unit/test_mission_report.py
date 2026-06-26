"""Unit tests for offline mission report generation."""

from pathlib import Path

from app.reports.mission_report import export_fleet_mission_report, export_mission_report


def test_export_mission_report_writes_html(tmp_path: Path):
    events = [
        {
            "timestamp": "2026-01-01 12:00:00",
            "health_state": "Warning",
            "fused_score": 0.75,
            "xai": "spike in ch_1",
        }
    ]
    path = export_mission_report(
        tmp_path,
        "Clock data",
        {"health_state": "OK", "fusion_score": 0.1},
        events,
    )
    report = Path(path)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Mission Report: Clock data" in text
    assert "health_state" in text
    assert "Warning" in text
    assert "0.7500" in text
    assert "spike in ch_1" in text


def test_export_mission_report_escapes_html(tmp_path: Path):
    path = export_mission_report(
        tmp_path,
        "<tab>",
        {"note": "<script>alert(1)</script>"},
        [{"timestamp": "", "health_state": "<bad>", "fused_score": 0, "xai": ""}],
    )
    text = Path(path).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text
    assert "&lt;bad&gt;" in text


def test_export_fleet_mission_report_writes_sections(tmp_path: Path):
    path = export_fleet_mission_report(
        tmp_path,
        {
            "tab_a": {"title": "Alpha", "health_state": "OK"},
            "tab_b": {"title": "Beta", "health_state": "Warning"},
        },
    )
    text = Path(path).read_text(encoding="utf-8")
    assert "Fleet Mission Report" in text
    assert "Alpha" in text
    assert "Beta" in text
    assert "Warning" in text
