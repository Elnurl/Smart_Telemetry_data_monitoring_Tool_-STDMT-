"""Graph memory singleton factory (backend selected by env / argument)."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

from app.agent.memory.protocol import GraphMemoryStore

logger = logging.getLogger("STDMS.Agent.GraphMemory")

_STORE: Optional[GraphMemoryStore] = None
_LOCK = threading.Lock()


def get_graph_memory(
    backend: Optional[str] = None,
    *,
    db_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    _fresh: bool = False,
) -> GraphMemoryStore:
    """Return process-wide graph memory store.

    Backend: ``networkx`` (default) or ``neo4j`` (stub — raises until implemented).
    Override with ``STDMS_GRAPH_BACKEND``.
    """
    global _STORE
    name = (backend or os.environ.get("STDMS_GRAPH_BACKEND") or "networkx").strip().lower()
    with _LOCK:
        if _fresh or _STORE is None:
            _STORE = _build(name, db_path=db_path, data_dir=data_dir)
        return _STORE


def reset_graph_memory_for_tests() -> None:
    global _STORE
    with _LOCK:
        if _STORE is not None:
            try:
                _STORE.clear()
            except Exception:
                pass
        _STORE = None


def _build(
    name: str,
    *,
    db_path: Optional[str],
    data_dir: Optional[str],
) -> GraphMemoryStore:
    if name in ("neo4j", "neo"):
        from app.agent.memory.neo4j_store import Neo4jGraphStore

        logger.warning("Neo4j backend requested — Sprint 2 seam only")
        return Neo4jGraphStore()  # raises
    if name not in ("networkx", "nx", "local", ""):
        logger.warning("Unknown graph backend %r — falling back to networkx", name)
    from app.agent.memory.networkx_store import NetworkXGraphStore

    store = NetworkXGraphStore(db_path=db_path, data_dir=data_dir)
    logger.info("Graph memory backend=networkx db=%s", store.db_path)
    return store
