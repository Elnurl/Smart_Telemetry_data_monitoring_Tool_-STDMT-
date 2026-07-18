"""Safe read/write helpers for department knowledge documents (Markdown/TXT).

Agent may only propose writes under data/knowledge; Approve executes them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

DEFAULT_KNOWLEDGE_DIR = Path("data") / "knowledge"
ALLOWED_SUFFIXES = {".md", ".txt", ".docx"}
MAX_CONTENT_CHARS = 200_000


def knowledge_dir(root: Optional[Path | str] = None) -> Path:
    return Path(root) if root is not None else DEFAULT_KNOWLEDGE_DIR


def sanitize_filename(filename: str) -> str:
    """Return a bare safe filename (no directories)."""
    raw = str(filename or "").strip()
    if not raw:
        raise ValueError("invalid_filename")
    # Reject any path separators or parent references before normalizing
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError("invalid_filename")
    name = Path(raw).name.strip()
    if not name or name in (".", ".."):
        raise ValueError("invalid_filename")
    if not re.match(r"^[\w.\- ()]+\.(md|txt|docx)$", name, flags=re.IGNORECASE):
        raise ValueError("filename_must_be_md_txt_or_docx")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("filename_must_be_md_txt_or_docx")
    return name


def resolve_knowledge_file(
    filename: str, *, root: Optional[Path | str] = None, must_exist: bool = False
) -> Path:
    base = knowledge_dir(root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    name = sanitize_filename(filename)
    path = (base / name).resolve()
    if base not in path.parents and path != base:
        # path must be directly under base
        if path.parent != base:
            raise ValueError("path_escape_blocked")
    if path.parent != base:
        raise ValueError("path_escape_blocked")
    if must_exist and not path.is_file():
        raise FileNotFoundError(name)
    return path


def list_knowledge_documents(*, root: Optional[Path | str] = None) -> dict[str, Any]:
    base = knowledge_dir(root)
    if not base.exists():
        return {"ok": True, "status": "empty", "directory": str(base), "documents": []}
    docs: list[dict[str, Any]] = []
    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        docs.append(
            {
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": size,
            }
        )
    return {
        "ok": True,
        "status": "ok",
        "directory": str(base),
        "count": len(docs),
        "documents": docs,
    }


def read_knowledge_document(
    filename: str, *, root: Optional[Path | str] = None, max_chars: int = 50_000
) -> dict[str, Any]:
    try:
        path = resolve_knowledge_file(filename, root=root, must_exist=True)
    except FileNotFoundError:
        listing = list_knowledge_documents(root=root)
        return {
            "ok": False,
            "status": "not_found",
            "filename": filename,
            "available": [d["filename"] for d in listing.get("documents") or []],
        }
    except ValueError as exc:
        return {"ok": False, "status": "invalid", "error": str(exc), "filename": filename}
    if path.suffix.lower() == ".docx":
        try:
            from docx import Document

            doc = Document(str(path))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
            text = "\n".join(parts)
        except Exception as exc:
            return {
                "ok": False,
                "status": "read_error",
                "filename": path.name,
                "error": str(exc),
            }
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return {
        "ok": True,
        "status": "ok",
        "filename": path.name,
        "path": str(path),
        "content": text,
        "truncated": truncated,
        "char_count": len(text),
    }


def write_knowledge_document(
    filename: str,
    content: str,
    *,
    root: Optional[Path | str] = None,
    mode: str = "write",
) -> dict[str, Any]:
    """Write or update a knowledge MD/TXT file. mode=write|update."""
    mode_norm = (mode or "write").strip().lower()
    if mode_norm not in ("write", "update"):
        return {"ok": False, "error": "mode_must_be_write_or_update"}
    body = content if content is not None else ""
    if not isinstance(body, str):
        body = str(body)
    if len(body) > MAX_CONTENT_CHARS:
        return {
            "ok": False,
            "error": "content_too_large",
            "max_chars": MAX_CONTENT_CHARS,
            "char_count": len(body),
        }
    try:
        path = resolve_knowledge_file(filename, root=root, must_exist=(mode_norm == "update"))
    except FileNotFoundError:
        return {
            "ok": False,
            "status": "not_found",
            "error": "file_not_found_for_update",
            "filename": filename,
            "hint": "Use propose_write_document to create a new file.",
        }
    except ValueError as exc:
        return {"ok": False, "status": "invalid", "error": str(exc), "filename": filename}

    existed = path.is_file()
    if mode_norm == "write" and existed:
        return {
            "ok": False,
            "status": "already_exists",
            "error": "file_already_exists",
            "filename": path.name,
            "hint": "Use propose_update_document to overwrite an existing file.",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return {
        "ok": True,
        "status": "updated" if existed else "created",
        "filename": path.name,
        "path": str(path),
        "char_count": len(body),
        "requires_knowledge_rebuild": True,
    }


def write_sop_document(
    fields: dict[str, Any],
    *,
    filename: Optional[str] = None,
    root: Optional[Path | str] = None,
    mode: str = "write",
) -> dict[str, Any]:
    """Create/update an SOP Word file from template fields."""
    from app.agent.sop_template import (
        SOP_TEMPLATE_ID,
        build_sop_docx,
        normalize_sop_fields,
        suggest_sop_filename,
        sop_fields_to_preview_markdown,
    )

    mode_norm = (mode or "write").strip().lower()
    if mode_norm not in ("write", "update"):
        return {"ok": False, "error": "mode_must_be_write_or_update"}
    data = normalize_sop_fields(fields)
    name = filename or suggest_sop_filename(data)
    try:
        if not str(name).lower().endswith(".docx"):
            name = str(Path(name).stem) + ".docx"
        path = resolve_knowledge_file(name, root=root, must_exist=(mode_norm == "update"))
    except FileNotFoundError:
        return {
            "ok": False,
            "status": "not_found",
            "error": "file_not_found_for_update",
            "filename": name,
            "hint": "Use propose_write_sop to create a new SOP.",
        }
    except ValueError as exc:
        return {"ok": False, "status": "invalid", "error": str(exc), "filename": name}

    existed = path.is_file()
    if mode_norm == "write" and existed:
        return {
            "ok": False,
            "status": "already_exists",
            "error": "file_already_exists",
            "filename": path.name,
            "hint": "Use propose_update_sop to overwrite an existing SOP.",
        }
    try:
        build_sop_docx(data, path)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "filename": path.name}

    return {
        "ok": True,
        "status": "updated" if existed else "created",
        "filename": path.name,
        "path": str(path),
        "template_id": SOP_TEMPLATE_ID,
        "preview_markdown": sop_fields_to_preview_markdown(data),
        "requires_knowledge_rebuild": True,
    }
