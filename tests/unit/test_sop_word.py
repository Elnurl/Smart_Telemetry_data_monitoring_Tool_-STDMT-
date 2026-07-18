"""SOP Word template propose + Approve."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent import docs_io
from app.agent.docs_io import write_sop_document
from app.agent.sop_template import build_sop_docx, normalize_sop_fields, suggest_sop_filename
from app.agent.tools import propose_update_sop, propose_write_sop
from app.models.registry import ModelRegistry


class _FakeWindow:
    def __init__(self, registry):
        self.custom_tabs = {}
        self.model_registry = registry


@pytest.fixture
def registry(tmp_path: Path):
    return ModelRegistry(db_path=str(tmp_path / "sop.sqlite"))


@pytest.fixture
def host(registry):
    h = MainWindowToolHost(_FakeWindow(registry))
    set_tool_host(h)
    yield h
    set_tool_host(None)


@pytest.fixture
def kdir(tmp_path: Path, monkeypatch):
    d = tmp_path / "knowledge"
    d.mkdir()
    monkeypatch.setattr(docs_io, "DEFAULT_KNOWLEDGE_DIR", d)
    return d


def test_build_sop_docx(kdir):
    fields = normalize_sop_fields(
        {
            "process_title": "Anomaly Response",
            "department": "Ops",
            "contact_info": "on-call",
            "sop_id": "SOP-8",
            "effective_date": "2026-07-13",
            "revision_number": "1",
            "process_description": "Respond to anomalies",
            "purpose_scope": "All monitoring tabs",
            "definitions_related": "Fusion score",
            "steps": [
                {"wbs": "1", "task": "Confirm tab", "owner": "Operator"},
                {"wbs": "2", "task": "Review drift", "owner": "Analyst"},
            ],
        }
    )
    path = kdir / suggest_sop_filename(fields)
    build_sop_docx(fields, path)
    assert path.is_file()
    assert path.stat().st_size > 1000


def test_propose_write_sop(host, registry, kdir):
    out = propose_write_sop(
        sop_fields={
            "process_title": "Weekly Check",
            "sop_id": "SOP-11",
            "department": "NV",
            "steps": [{"wbs": "1", "task": "Start monitoring", "owner": "Ops"}],
        },
        proposed_message="Create SOP-11 Word",
        host=host,
    )
    assert out["ok"] is True
    assert out["kind"] == "write_sop"
    drafts = registry.get_pending_draft_alerts(kind="write_sop")
    assert len(drafts) == 1
    assert drafts[0]["proposed_payload"]["filename"] == "SOP-11.docx"
    assert drafts[0]["proposed_payload"]["template_id"] == "stdms_sop_word_v1"


def test_approve_write_sop(kdir, registry):
    fields = {
        "process_title": "Test Proc",
        "sop_id": "SOP-12",
        "steps": [{"task": "Do thing", "owner": "A"}],
    }
    result = write_sop_document(fields, filename="SOP-12.docx", root=kdir, mode="write")
    assert result["ok"] is True
    assert (kdir / "SOP-12.docx").is_file()

    draft_id = registry.create_draft_alert(
        kind="update_sop",
        proposed_message="bump",
        proposed_payload={"filename": "SOP-12.docx", "sop_fields": fields},
    )
    assert draft_id > 0
    updated = write_sop_document(
        {**fields, "revision_number": "2"},
        filename="SOP-12.docx",
        root=kdir,
        mode="update",
    )
    assert updated["ok"] is True
    assert updated["status"] == "updated"


def test_propose_write_sop_from_user_text(host, registry, kdir):
    from app.agent.tools import propose_write_sop_from_user_text

    out = propose_write_sop_from_user_text(
        "Create SOP procedure named Eclipse monitoring, SOP-9",
        host=host,
    )
    assert out["ok"] is True
    assert out["kind"] == "write_sop"
    drafts = registry.get_pending_draft_alerts(kind="write_sop")
    assert drafts
    assert drafts[0]["proposed_payload"]["filename"] == "SOP-9.docx"
    assert "Eclipse" in drafts[0]["proposed_payload"]["sop_fields"]["process_title"]
    out = propose_update_sop(
        filename="missing.docx",
        sop_fields={"process_title": "X", "sop_id": "X"},
        proposed_message="upd",
        host=host,
    )
    assert out["ok"] is False
    assert out["error"] == "file_not_found"
