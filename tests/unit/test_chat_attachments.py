"""Unit tests for agent chat file attachments."""

from __future__ import annotations

from pathlib import Path

from app.agent.chat_attachments import build_message_with_attachments, read_attachment


def test_read_txt_attachment(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("hello fleet\nline2", encoding="utf-8")
    result = read_attachment(p)
    assert result["ok"] is True
    assert "hello fleet" in result["content"]
    assert result["filename"] == "note.txt"


def test_build_message_includes_attachment(tmp_path: Path):
    p = tmp_path / "sop.md"
    p.write_text("# Eclipse procedure\nDo X then Y.", encoding="utf-8")
    out = build_message_with_attachments("What does this say?", [str(p)])
    assert out["ok"] is True
    assert "Eclipse procedure" in out["message"]
    assert "Attached file: sop.md" in out["message"]
    assert "sop.md" in out["display"]


def test_attachment_only_gets_default_prompt(tmp_path: Path):
    p = tmp_path / "log.txt"
    p.write_text("ERROR battery low", encoding="utf-8")
    out = build_message_with_attachments("", [str(p)])
    assert out["ok"] is True
    assert "review the attached" in out["message"].lower()
    assert "battery low" in out["message"]


def test_unsupported_type(tmp_path: Path):
    p = tmp_path / "photo.bin"
    p.write_bytes(b"\x00\x01\x02")
    result = read_attachment(p)
    assert result["ok"] is False
    assert "unsupported" in str(result.get("error", "")).lower()
