"""Knowledge document propose + Approve helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.bridge import MainWindowToolHost, set_tool_host
from app.agent import docs_io
from app.agent.docs_io import (
    list_knowledge_documents,
    read_knowledge_document,
    sanitize_filename,
    write_knowledge_document,
)
from app.agent.tools import (
    invoke_tool,
    propose_update_document,
    propose_write_document,
)
from app.models.registry import ModelRegistry


class _FakeWindow:
    def __init__(self, registry):
        self.custom_tabs = {}
        self.model_registry = registry


@pytest.fixture
def registry(tmp_path: Path):
    return ModelRegistry(db_path=str(tmp_path / "docs.sqlite"))


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
    (d / "SOP-7.md").write_text("# SOP-7\nExisting.\n", encoding="utf-8")
    monkeypatch.setattr(docs_io, "DEFAULT_KNOWLEDGE_DIR", d)
    return d


def test_sanitize_filename_rejects_escape():
    with pytest.raises(ValueError):
        sanitize_filename("../secret.md")
    with pytest.raises(ValueError):
        sanitize_filename("notes.pdf")
    assert sanitize_filename("SOP-8_test.md") == "SOP-8_test.md"


def test_list_read_write_docs(kdir):
    listing = list_knowledge_documents(root=kdir)
    assert listing["ok"] is True
    assert any(d["filename"] == "SOP-7.md" for d in listing["documents"])

    read = read_knowledge_document("SOP-7.md", root=kdir)
    assert read["ok"] is True
    assert "SOP-7" in read["content"]

    created = write_knowledge_document(
        "SOP-8.md", "# SOP-8\nNew.\n", root=kdir, mode="write"
    )
    assert created["ok"] is True
    assert created["status"] == "created"

    dup = write_knowledge_document("SOP-8.md", "x", root=kdir, mode="write")
    assert dup["ok"] is False

    updated = write_knowledge_document(
        "SOP-8.md", "# SOP-8\nUpdated.\n", root=kdir, mode="update"
    )
    assert updated["ok"] is True
    assert "Updated" in (kdir / "SOP-8.md").read_text(encoding="utf-8")


def test_propose_write_document(host, registry, kdir):
    out = propose_write_document(
        filename="SOP-9_new.md",
        content="# SOP-9\nBody\n",
        proposed_message="Add SOP-9",
        host=host,
    )
    assert out["ok"] is True
    assert out["kind"] == "write_document"
    drafts = registry.get_pending_draft_alerts(kind="write_document")
    assert len(drafts) == 1
    assert drafts[0]["proposed_payload"]["filename"] == "SOP-9_new.md"
    assert "SOP-9" in drafts[0]["proposed_payload"]["content"]


def test_propose_write_rejects_existing(host, kdir):
    out = propose_write_document(
        filename="SOP-7.md",
        content="# overwrite\n",
        proposed_message="bad",
        host=host,
    )
    assert out["ok"] is False
    assert out["error"] == "file_already_exists"


def test_propose_update_document(host, registry, kdir):
    out = propose_update_document(
        filename="SOP-7.md",
        content="# SOP-7\nRevised.\n",
        proposed_message="Revise SOP-7",
        host=host,
    )
    assert out["ok"] is True
    assert out["kind"] == "update_document"
    drafts = registry.get_pending_draft_alerts(kind="update_document")
    assert len(drafts) == 1
    assert "Revised" in drafts[0]["proposed_payload"]["content"]


def test_invoke_list_and_propose(host, kdir):
    listing = invoke_tool("list_knowledge_documents", {}, host=host)
    assert listing["ok"] is True
    assert listing["count"] >= 1
    names = {t["function"]["name"] for t in __import__("app.agent.tools", fromlist=["to_ollama_tools"]).to_ollama_tools()}
    assert "propose_write_document" in names
    assert "propose_update_document" in names
    assert "read_knowledge_document" in names


def test_approve_execute_write(kdir, registry):
    """Mirror main-window document Approve without Qt."""
    draft = {
        "kind": "write_document",
        "proposed_payload": {
            "filename": "SOP-10.md",
            "content": "# Ten\n",
            "action": "write",
        },
    }
    payload = draft["proposed_payload"]
    result = write_knowledge_document(
        payload["filename"], payload["content"], root=kdir, mode="write"
    )
    assert result["ok"] is True
    assert (kdir / "SOP-10.md").is_file()
    assert registry.create_draft_alert(
        kind="write_document",
        proposed_message="x",
        proposed_payload=payload,
    )
