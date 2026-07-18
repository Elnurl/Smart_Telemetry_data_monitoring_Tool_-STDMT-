"""Faza A — inspect_data_folder / suggest_tab_config / propose_create_tab enrichment."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.smart_config import (
    inspect_data_folder,
    suggest_tab_config,
    suggestion_to_tab_config,
)
from app.agent.tools import invoke_tool, propose_create_tab
from app.models.registry import ModelRegistry


@pytest.fixture
def csv_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "battery"
    folder.mkdir()
    (folder / "BatteryTemperature.csv").write_text(
        "timestamp,Battery_Temp,Bus_Voltage,Total_Current\n"
        + "\n".join(
            f"2026-01-01T00:{i:02d}:00,{22.0 + i * 0.1},{28.4 - i * 0.01},{1.2 - i * 0.01}"
            for i in range(60)
        )
        + "\n",
        encoding="utf-8",
    )
    return folder


def test_inspect_a1_schema(csv_folder: Path):
    out = inspect_data_folder(str(csv_folder))
    assert out["ok"] is True
    # GUI DataReader contract: time / value / value2 / …
    assert "value" in out["columns"]
    assert "value" in out["numeric_cols"]
    assert "time" in out["timestamp_candidates"]
    assert out["row_count"] >= 60
    assert "value" in out["null_rates"]
    assert "value" in out["sample_ranges"]
    rng = out["sample_ranges"]["value"]
    assert "min" in rng and "max" in rng and "mean" in rng
    assert rng["max"] >= rng["min"]
    assert out["data_quality_ok"] is True


def test_inspect_headerless_csv_uses_value(tmp_path: Path):
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "BatteryTemperature.csv").write_text(
        "\n".join(f"2026-01-01T00:{i:02d}:00,{20.0 + i * 0.1}" for i in range(40)) + "\n",
        encoding="utf-8",
    )
    out = inspect_data_folder(str(folder))
    assert out["ok"] is True
    assert "value" in out["numeric_cols"]
    assert "time" in out["timestamp_candidates"]
    # Must not keep first data cell as a feature name
    assert not any(str(c).replace(".", "", 1).isdigit() for c in out["numeric_cols"])


def test_resolve_relative_data_and_reject_missing(tmp_path: Path, monkeypatch):
    from app.agent.smart_config import resolve_data_paths

    missing = resolve_data_paths(r"C:\data\health_metrics")
    assert missing["ok"] is False
    assert missing.get("error") == "path_not_found"

    data = tmp_path / "data"
    data.mkdir()
    csv = data / "BusVoltage.csv"
    csv.write_text("t,v\n1,2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    ok = resolve_data_paths("data/BusVoltage.csv")
    assert ok["ok"] is True
    assert ok["source_file"]
    assert Path(ok["data_folder"]).is_dir()


def test_suggest_a2_schema(csv_folder: Path):
    out = suggest_tab_config(
        title="Battery Health",
        data_folder=str(csv_folder),
        purpose="battery health monitoring",
        sop_hints=["Check battery temp during eclipse"],
    )
    assert out["ok"] is True
    assert out["tab_name"] == "Battery Health"
    assert out["features"]
    assert "time" not in out["features"]
    assert "timestamp" not in out["features"]
    assert "value" in out["features"]
    assert 2 <= len(out["models"]) <= 3
    assert out["schedule_type"] == "Continuous"
    assert out["window_size"] >= 50
    assert out["reasoning"]
    assert "SOP" in out["reasoning"] or "eclipse" in out["reasoning"].lower()
    cfg = out["config"]
    assert cfg["selected_features"] == out["features"]
    assert cfg["monitoring_window_rows"] == out["window_size"]
    assert any(m["model_type"] == "Isolation Forest" for m in out["models"])


def test_propose_create_tab_from_suggest_payload(csv_folder: Path, tmp_path: Path):
    registry = ModelRegistry(db_path=str(tmp_path / "models.sqlite"))

    class _Host:
        def propose_alert(
            self,
            tab_id,
            *,
            proposed_message="",
            agent_reasoning="",
            severity="WARNING",
            kind="alert",
            proposed_payload=None,
        ):
            draft_id = registry.create_draft_alert(
                tab_id=tab_id,
                kind=kind or "alert",
                agent_reasoning=agent_reasoning or "",
                proposed_message=proposed_message or "",
                proposed_payload=proposed_payload or {},
                severity=severity or "INFO",
            )
            return {
                "ok": True,
                "status": "pending",
                "draft_id": draft_id,
                "kind": kind,
                "requires_human_approval": True,
            }

    sug = suggest_tab_config(
        title="Battery Health",
        data_folder=str(csv_folder),
        purpose="battery health",
    )
    out = propose_create_tab(
        proposed_message="Create battery health tab from data/battery",
        title=sug["tab_name"],
        data_folder=str(csv_folder),
        config=sug,  # pass full suggest payload (A.3)
        auto_suggest=False,
        host=_Host(),
    )
    assert out.get("ok") is True
    drafts = registry.get_pending_draft_alerts(kind="create_tab")
    assert len(drafts) == 1
    cfg = (drafts[0].get("proposed_payload") or {}).get("config") or {}
    assert cfg.get("selected_features") == sug["features"]
    assert cfg.get("models")
    assert cfg.get("monitoring_window_rows") == sug["window_size"]
    assert cfg.get("data_folder")
    assert drafts[0].get("agent_reasoning")


def test_propose_create_tab_auto_suggest(csv_folder: Path, tmp_path: Path):
    registry = ModelRegistry(db_path=str(tmp_path / "models.sqlite"))

    class _Host:
        def propose_alert(
            self,
            tab_id,
            *,
            proposed_message="",
            agent_reasoning="",
            severity="WARNING",
            kind="alert",
            proposed_payload=None,
        ):
            draft_id = registry.create_draft_alert(
                tab_id=tab_id,
                kind=kind or "alert",
                agent_reasoning=agent_reasoning or "",
                proposed_message=proposed_message or "",
                proposed_payload=proposed_payload or {},
                severity=severity or "INFO",
            )
            return {"ok": True, "draft_id": draft_id, "kind": kind}

    out = propose_create_tab(
        proposed_message="Create tab from folder",
        title="Battery Health",
        data_folder=str(csv_folder),
        purpose="battery health",
        auto_suggest=True,
        host=_Host(),
    )
    assert out.get("ok") is True
    cfg = (registry.get_pending_draft_alerts(kind="create_tab")[0].get("proposed_payload") or {}).get(
        "config"
    ) or {}
    assert cfg.get("selected_features")
    assert 2 <= len(cfg.get("models") or []) <= 3


def test_suggestion_to_tab_config_partial():
    cfg = suggestion_to_tab_config(
        {
            "tab_name": "X",
            "features": ["a", "b"],
            "models": [{"model_type": "Z-Score", "model_parameters": {"threshold": 2.5}}],
            "schedule_type": "Continuous",
            "window_size": 120,
            "interval_ms": 60_000,
        }
    )
    assert cfg["title"] == "X"
    assert cfg["selected_features"] == ["a", "b"]
    assert cfg["monitoring_window_rows"] == 120
    assert cfg["interval_ms"] == 60_000


def test_inspect_tool_via_invoke(csv_folder: Path):
    out = invoke_tool("inspect_data_folder", {"data_folder": str(csv_folder)}, host=None)
    assert out.get("ok") is True
    assert "sample_ranges" in out
