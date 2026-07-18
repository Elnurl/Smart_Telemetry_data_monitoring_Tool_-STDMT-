"""Phase 4 — local dual RAG (air-gap): space + ground SQLite stores."""

from __future__ import annotations

from pathlib import Path

from app.agent.rag.dual import (
    GROUND_INDEX_PATH,
    SPACE_INDEX_PATH,
    DualStoreRetriever,
    classify_segment,
    dual_knowledge_status,
)
from app.agent.rag.retrieve import DEFAULT_INDEX_PATH, knowledge_status, search_knowledge_index

DEFAULT_DOCS_DIR = Path("data") / "knowledge"


def ingest_knowledge_dir(*args, **kwargs):
    from app.agent.rag.ingest import ingest_knowledge_dir as _ingest

    return _ingest(*args, **kwargs)


def ingest_dual_knowledge(*args, **kwargs):
    from app.agent.rag.ingest import ingest_dual_knowledge as _ingest

    return _ingest(*args, **kwargs)


def export_fsm_to_knowledge(*args, **kwargs):
    from app.agent.rag.ingest import export_fsm_to_knowledge as _export

    return _export(*args, **kwargs)


__all__ = [
    "DEFAULT_DOCS_DIR",
    "DEFAULT_INDEX_PATH",
    "DualStoreRetriever",
    "GROUND_INDEX_PATH",
    "SPACE_INDEX_PATH",
    "classify_segment",
    "dual_knowledge_status",
    "export_fsm_to_knowledge",
    "ingest_dual_knowledge",
    "ingest_knowledge_dir",
    "knowledge_status",
    "search_knowledge_index",
]
